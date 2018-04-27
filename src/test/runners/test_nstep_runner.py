from runners.nstep_runner import NStepRunner
from utils.dict2namedtuple import convert

def test1():
    _args = {"n_agents": 2,
             "env": "integration_test",
             "batch_size_run": 1,
             "obs_epsilon": False,
             "learner":"coma",
             "multiagent_controller": "independent_mac",
             "t_max":20,
             "tensorboard": True,
             "name":"DEBUG1",
             "agent":"basic_ac",
             "agent_output_type": "policies",
             "agent_model":"DQN",
             "observe":True,
             "observe_db":True,
             "action_selector":"multinomial",
             "use_blitzz":True,
             "obs_last_action":True,
             "batch_size":1,
             "test_interval":5000,
             "epsilon_start":1.0,
             "epsilon_finish":0.05,
             "epsilon_time_length":1000000,
             "epsilon_decay":"exp",
             "obs_agent_id": True,
             "obs_last_action": False,
             "share_agent_params": True,
             "use_cuda": True,
             "buffer_size": 32,
             "lr": 5e-4, # 5e-4
             "gamma":0.99,
             "td_lambda": 0.5,
             "tensorboard": True,
             "n_loops_per_thread_or_sub_or_main_process": 0,
             "n_threads_per_subprocess_or_main_process": 0,
             "n_subprocesses": 0
             }
    _args["target_update_interval"] = _args["batch_size_run"] * 500
    _args["env_args"] = {"episode_limit": 0}
    args = convert(_args)

    runner = NStepRunner(args=args, test_mode=False)

    runner.run()
    pass

def test2():
    """
    test subprocess-mode: one subprocess per runner batch element
    """

    _args = {"n_agents": 2,
             "env": "integration_test",
             "batch_size_run": 3,
             "obs_epsilon": False,
             "learner":"coma",
             "multiagent_controller": "independent_mac",
             "t_max":20,
             "tensorboard": True,
             "name":"DEBUG1",
             "agent":"basic_ac",
             "agent_output_type": "policies",
             "agent_model":"DQN",
             "observe":True,
             "observe_db":True,
             "action_selector":"multinomial",
             "use_blitzz":True,
             "obs_last_action":True,
             "batch_size":3,
             "test_interval":5000,
             "epsilon_start":1.0,
             "epsilon_finish":0.05,
             "epsilon_time_length":1000000,
             "epsilon_decay":"exp",
             "obs_agent_id": True,
             "obs_last_action": False,
             "share_agent_params": True,
             "use_cuda": True,
             "buffer_size": 32,
             "lr": 5e-4, # 5e-4
             "gamma":0.99,
             "td_lambda": 0.5,
             "tensorboard": True,
             "n_loops_per_thread_or_sub_or_main_process": 2,
             "n_threads_per_subprocess_or_main_process": 0,
             }
    _args["n_subprocesses"] = 3
    _args["n_loops_per_thread_or_sub_or_main_process"] = 1
    _args["target_update_interval"] = _args["batch_size_run"] * 500
    _args["env_args"] = {"episode_limit": 0}
    args = convert(_args)

    runner = NStepRunner(args=args, test_mode=False)

    for _ in range(10):
        results = runner.run()
        a = results.to_pd()
        b = 5
    pass

def main():
    # test1()
    test2()
    pass

if __name__ == "__main__":
    main()