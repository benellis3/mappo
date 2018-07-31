import datetime
from functools import partial
from math import ceil
import numpy as np
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.dict2namedtuple import convert
from utils.logging import get_logger, append_scalar, log_stats, HDFLogger
from utils.timehelper import time_left, time_str

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import MultiAgentController
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log, pymongo_client):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    # convert _config dict to GenericDict objects (which cannot be overwritten later)
    args = convert(_config)
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    import os
    _log.info("OS ENVIRON KEYS: {}\n\n".format(os.environ))

    if _config.get("debug_mode", None) is not None:
        _log.warning("ATTENTION DEBUG MODE: {}".format(_config["debug_mode"]))

    # configure logging
    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if args.use_tensorboard:
        import tensorboard
        if tensorboard:
            from tensorboard_logger import configure, log_value
        import os
        from os.path import dirname, abspath
        file_tb_path = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        configure(os.path.join(file_tb_path, "{}").format(unique_token))

    # set up logging objects to be passed on from now on
    logging_struct = SN(py_logger=_log,
                        sacred_log_scalar_fn=partial(append_scalar, run=_run))
    if args.use_tensorboard:
        logging_struct.tensorboard_log_scalar_fn=log_value

    if args.use_hdf_logger:
        logging_struct.hdf_logger = HDFLogger(path=args.local_results_path, name=args.name)

    # Run and train
    run_sequential(args=args, _run=_run, _logging_struct=logging_struct, unique_token=unique_token)

    # Clean up after finishing
    print("Exiting Main")

    if pymongo_client is not None:
        print("Attempting to close mongodb client")
        pymongo_client.close()
        print("Mongodb client closed")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def run_sequential(args, _logging_struct, _run, unique_token):

    # Init runner so we can get env info
    runner_obj = r_REGISTRY[args.runner](args=args,
                                         logging_struct=_logging_struct)

    # Set up schemes and groups here
    env_info = runner_obj.get_env_info()
    n_agents = env_info["n_agents"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.int},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8}
    }
    groups = {
        "agents": n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=n_agents)])
    }

    # Setup multiagent controller here
    mac = MultiAgentController(env_info["n_agents"], scheme, groups, preprocess, args)  # Dummy for testing

    # Give runner the scheme
    runner_obj.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner_obj = None # Temp

    # replay buffer
    # TODO: Add device
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"], preprocess=preprocess)

    # start training
    episode = 0
    last_test_T = 0
    model_save_time = 0
    start_time = time.time()

    _logging_struct.py_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner_obj.T_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner_obj.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            learner_obj.train(episode_sample, T_env=runner_obj.T_env)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner_obj.batch_size)
        if ( runner_obj.T_env - last_test_T) / args.test_interval >= 1.0:

            _logging_struct.py_logger.info("T_env: {} / {}".format(runner_obj.T_env, args.t_max))
            _logging_struct.py_logger.info("Estimated time left: {}. Time passed: {}".format(time_left(start_time, runner_obj.T_env, args.t_max), time_str(time.time() - start_time)))
            runner_obj.log() # log runner statistics derived from training runs

            last_test_T = runner_obj.T_env
            for _ in range(n_test_runs):
                runner_obj.run(test_mode=True)

            runner_obj.log()  # log runner statistics derived from test runs
            learner_obj.log()

        # save model once in a while
        if args.save_model and (runner_obj.T_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner_obj.T_env
            _logging_struct.py_logger.info("Saving models")

            save_path = os.path.join(args.local_results_path, "models") #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)

            # learner obj will save all agent and further models used
            learner_obj.save_models(path=save_path, token=unique_token, T=runner_obj.T_env)

        episode += 1
        # Actually
        log()

    _logging_struct.py_logger.info("Finished Training")

def log():
    # TODO: Log stuff
    # if args.save_episode_samples:
    #     assert args.use_hdf_logger, "use_hdf_logger needs to be enabled if episode samples are to be stored!"
    #     _logging_struct.hdf_logger.log("", episode_sample, runner_obj.T_env)

# TODO: Clean this up
def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
        "need to use replay buffer if running in parallel mode!"

    assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    if config["learner"] == "coma":
       assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
       (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
           "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config
