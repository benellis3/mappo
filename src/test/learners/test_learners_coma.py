from utils.blitzz.runners import NStepRunner
from utils.blitzz.replay_buffer import ContiguousReplayBuffer
from utils.blitzz.scheme import Scheme
from utils.blitzz.sequence_buffer import BatchEpisodeBuffer
from utils.blitzz.learners.coma import COMALearner, COMAPolicyLoss, COMACriticLoss
from utils.dict2namedtuple import convert

from utils.blitzz.debug import to_np
from utils.blitzz.transforms import _build_model_inputs

import datetime
import numpy as np
import torch as th
from torch.autograd import Variable

def test1():
    """
    Does the critic loss go to zero if if I keep the replay buffer sample fixed?
    """
    _args = dict(env="pred_prey",
                 n_agents=4,
                 t_max=1000000,
                 learner="coma",
                 env_args=dict(prey_movement="escape",
                               predator_prey_shape=(3,3),
                               nagent_capture_enabled=False),
                 tensorboard=True,
                 name="33pp_comatest_dqn_new",
                 target_critic_update_interval=200000*2000, # DEBUG # 200000*20
                 agent="basic",
                 agent_model="DQN",
                 observe=True,
                 observe_db=True,
                 action_selector="multinomial",
                 use_blitzz=True,
                 obs_last_action=True,
                 test_interval=5000,
                 epsilon_start=1.0,
                 epsilon_finish=0.05,
                 epsilon_time_length=100000,
                 epsilon_decay="exp",
                 test_nepisode=50,
                 batch_size=32,
                 batch_size_run=32,
                 n_learner_reps=200
                 )

    # required args
    _required_args = dict(obs_epsilon=False,
                          obs_agent_id=True,
                          share_params=True,
                          use_cuda=True,
                          lr_critic=5e-4,
                          lr_agent=5e-4,
                          multiagent_controller="independent",
                          td_lambda=1.0,
                          gamma=0.99)

    _args.update(_required_args)

    _args["target_update_interval"] = _args["batch_size_run"] * 500
    _args["env_args"] = {"predator_prey_shape":(3,3),
                         "prey_movement": "escape",
                         "nagent_capture_enabled":False}
    args = convert(_args)

    unique_name = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if args.tensorboard:
        from tensorboard_logger import configure, log_value
        configure("results/tb_logs/{}".format(unique_name))

    runner = NStepRunner(args=args)
    transitions = runner.run()
    dbg_trans = transitions.to_pd()

    # set up the learner
    learner_obj = COMALearner(multiagent_controller=runner.multiagent_controller,
                              args=args)
    learner_obj.create_models(runner.transition_scheme)

    T = 0
    episode = 0
    last_test_T = 0
    import time
    start_time = time.time()
    has_learnt = False

    # log_config()
    print("Beginning training for {} timesteps".format(args.t_max))

    while T <= args.t_max:
        learner_obj.train(transitions) #fixed_sample
        stats = {}
        learner_stats = learner_obj.get_stats()
        stats["critic_loss"] = np.mean(learner_stats["critic_loss"][max(-100, -len(learner_stats["critic_loss"])):])
        stats["policy_loss"] = np.mean(learner_stats["policy_loss"][max(-100, -len(learner_stats["policy_loss"])):])
        stats["critic_grad_norm"] = np.mean(
            learner_stats["critic_grad_norm"][max(-100, -len(learner_stats["critic_grad_norm"])):])
        stats["policy_grad_norm"] = np.mean(
            learner_stats["policy_grad_norm"][max(-100, -len(learner_stats["policy_grad_norm"])):])
        stats["critic_mean"] = np.mean(learner_stats["critic_mean"][max(-100, -len(learner_stats["critic_mean"])):])
        stats["target_critic_mean"] = np.mean(
            learner_stats["target_critic_mean"][max(-100, -len(learner_stats["target_critic_mean"])):])
        print("T: {} critic loss: {} policy loss: {}".format(T, stats["critic_loss"], stats["policy_loss"]))
        T += 1
    pass

def test2():
    """
    Test critic and policy losses
    """
    n_agents = 2
    n_actions = 4 # don't change
    n_bs = 1
    n_t = 1

    # test critic loss forward + backward
    critic_loss = COMACriticLoss()
    inputs = Variable(th.FloatTensor([0.1, 0.2, 0.3, 0.4])).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(n_agents, n_bs, n_t, 1)
    targets =  Variable(th.FloatTensor([0.1, 0.3, 0.2, 1.0])).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(n_agents, n_bs, n_t, 1)
    tformat = "a*bs*t*v"
    critic_loss_output, _ = critic_loss(input=inputs,
                                     target=targets,
                                     tformat=tformat)
    dbg = critic_loss_output.data.numpy()

    # test policy loss forward + backward
    policy_loss = COMAPolicyLoss()
    log_policies = Variable(th.FloatTensor([0.1, 0.2, 0.3, 0.4])).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(n_agents, n_bs, n_t, 1)
    advantages = Variable(th.FloatTensor([4.0, 3.0, 2.0, 1.0])).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(n_agents, n_bs, n_t, 1)
    actions = Variable(th.FloatTensor([0.0, 2.0, 1.0, 3.0])).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(n_agents, n_bs, n_t, 1)
    tformat = "a*bs*t*v"
    policy_loss_output, _ = policy_loss(log_policies=log_policies,
                              advantages=advantages,
                              actions=actions,
                              tformat=tformat)

def test3():
    """
    integration test: does env data flow through the system appropriately?
    """

    _args = dict(env="integration_test",
                 n_agents=2,
                 t_max=1000000,
                 learner="coma",
                 env_args=dict(prey_movement="escape",
                               predator_prey_shape=(3,3),
                               nagent_capture_enabled=False),
                 tensorboard=True,
                 name="33pp_comatest_dqn_new",
                 target_critic_update_interval=200000*2000, # DEBUG # 200000*20
                 agent="basic",
                 agent_model="DQN",
                 observe=True,
                 observe_db=True,
                 action_selector="multinomial",
                 use_blitzz=True,
                 obs_last_action=True,
                 test_interval=5000,
                 epsilon_start=0.0, # NO EPSILON: DEBUG
                 epsilon_finish=0.0,
                 epsilon_time_length=100000,
                 epsilon_decay="exp",
                 test_nepisode=50,
                 batch_size=3,
                 batch_size_run=3,
                 n_learner_reps=20
                 )

    # required args
    _required_args = dict(obs_epsilon=False,
                          obs_agent_id=True,
                          share_params=True,
                          use_cuda=True,
                          lr_critic=5e-4,
                          lr_agent=5e-4,
                          multiagent_controller="independent",
                          td_lambda=1.0,
                          gamma=0.99)

    _args.update(_required_args)

    _args["target_update_interval"] = _args["batch_size_run"] * 500
    _args["env_args"] = {"predator_prey_shape":(3,3),
                         "prey_movement": "escape",
                         "nagent_capture_enabled":False}
    args = convert(_args)

    unique_name = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if args.tensorboard:
        from tensorboard_logger import configure, log_value
        configure("results/tb_logs/{}".format(unique_name))

    # set up the runner
    runner = NStepRunner(args=args)

    # set up the learner
    learner_obj = COMALearner(multiagent_controller=runner.multiagent_controller,
                              args=args)
    learner_obj.create_models(runner.transition_scheme)

    # Test whether env resets work fine
    transitions_0 = runner.run()
    transitions_0_pd = transitions_0.to_pd()
    [_env.reset() for _env in runner.envs]
    transitions = runner.run()
    transitions_pd = transitions.to_pd()
    transitions_0.data[transitions_0.data!=transitions_0.data] = 99.0
    transitions.data[transitions.data != transitions.data] = 99.0
    a = th.sum(transitions_0.data-transitions.data)
    assert th.sum(transitions_0.data-transitions.data) == 0.0, "env reset seems broken."

    # test several runner loops to see whether runner triggers env resets properly
    for _i in range(3):
        transitions = runner.run()
        transitions_pd = transitions.to_pd()
        transitions.data[transitions.data!=transitions.data] = 99.0
        assert th.sum(transitions_0.data - transitions.data) == 0.0, "runner call to env reset seems broken."

    # Test whether view() produces the right outputs
    #transitions.data= transitions.data.cpu()
    transitions = runner.run()
    transitions_pd = transitions.to_pd()
    data_inputs, data_inputs_tformat = transitions.view(dict_of_schemes=learner_obj.joint_scheme_dict,
                                                        bs_ids=None,
                                                        fill_zero=False)
    data_inputs_pd = {_k:_v.to_pd() for _k, _v in data_inputs.items()}

    # Test whether model inputs function produces correct outputs
    coma_model_inputs, coma_model_inputs_tformat = _build_model_inputs(column_dict=learner_obj.input_columns,
                                                                       inputs=data_inputs,
                                                                       inputs_tformat=data_inputs_tformat,
                                                                       to_variable=True)

    critic_agent_action = coma_model_inputs["critic"]["agent_action"][0,0,0,:].cpu().data.numpy()
    critic_agent_policy = coma_model_inputs["critic"]["agent_policy"][0, 0, 0, :].cpu().data.numpy()
    critic_q_function = coma_model_inputs["critic"]["qfunction"][0, 0, 0, :].cpu().data.numpy()
    action_agent = transitions.data[0,0,transitions.columns["actions__agent0"][0]:transitions.columns["actions__agent0"][1]].cpu().numpy()
    policy_agent = transitions.data[0, 0, transitions.columns["policies__agent0"][0]:transitions.columns["policies__agent0"][1]].cpu().numpy()
    np.testing.assert_array_equal(critic_agent_action, action_agent, err_msg="_build_model_inputs seems broken!")
    np.testing.assert_array_equal(critic_agent_policy, policy_agent, err_msg="_build_model_inputs seems broken!")

    critic_agent_action = coma_model_inputs["critic"]["agent_action"][1,2,1,:].cpu().data.numpy()
    critic_agent_policy = coma_model_inputs["critic"]["agent_policy"][1, 2, 1, :].cpu().data.numpy()
    critic_q_function = coma_model_inputs["critic"]["qfunction"][1, 2, 1, :].cpu().data.numpy()
    action_agent = transitions.data[2,1,transitions.columns["actions__agent1"][0]:transitions.columns["actions__agent1"][1]].cpu().numpy()
    policy_agent = transitions.data[2, 1, transitions.columns["policies__agent1"][0]:transitions.columns["policies__agent1"][1]].cpu().numpy()
    np.testing.assert_array_equal(critic_agent_action, action_agent, err_msg="_build_model_inputs seems broken!")
    np.testing.assert_array_equal(critic_agent_policy, policy_agent, err_msg="_build_model_inputs seems broken!")

    critic_agent_action = coma_model_inputs["critic"]["agent_action"][1,2,2,:].cpu().data.numpy()
    critic_agent_policy = coma_model_inputs["critic"]["agent_policy"][1, 2, 2, :].cpu().data.numpy()
    critic_q_function = coma_model_inputs["critic"]["qfunction"][1, 2, 2, :].cpu().data.numpy()
    action_agent = transitions.data[2,2,transitions.columns["actions__agent1"][0]:transitions.columns["actions__agent1"][1]].cpu().numpy()
    policy_agent = transitions.data[2, 2, transitions.columns["policies__agent1"][0]:transitions.columns["policies__agent1"][1]].cpu().numpy()
    np.testing.assert_array_equal(critic_agent_action, action_agent, err_msg="_build_model_inputs seems broken!")
    np.testing.assert_array_equal(critic_agent_policy, policy_agent, err_msg="_build_model_inputs seems broken!")

    x = transitions.get_col("agent_id__agent0")
    a = 5
    # while T <= args.t_max:
    #     learner_obj.train(transitions) #fixed_sample
    #     stats = {}
    #     learner_stats = learner_obj.get_stats()
    #     stats["critic_loss"] = np.mean(learner_stats["critic_loss"][max(-100, -len(learner_stats["critic_loss"])):])
    #     stats["policy_loss"] = np.mean(learner_stats["policy_loss"][max(-100, -len(learner_stats["policy_loss"])):])
    #     stats["critic_grad_norm"] = np.mean(
    #         learner_stats["critic_grad_norm"][max(-100, -len(learner_stats["critic_grad_norm"])):])
    #     stats["policy_grad_norm"] = np.mean(
    #         learner_stats["policy_grad_norm"][max(-100, -len(learner_stats["policy_grad_norm"])):])
    #     stats["critic_mean"] = np.mean(learner_stats["critic_mean"][max(-100, -len(learner_stats["critic_mean"])):])
    #     stats["target_critic_mean"] = np.mean(
    #         learner_stats["target_critic_mean"][max(-100, -len(learner_stats["target_critic_mean"])):])
    #     print("T: {} critic loss: {} policy loss: {}".format(T, stats["critic_loss"], stats["policy_loss"]))
    #     T += 1
    pass


def main():
    #test1()
    #test2()
    #test1b()
    #test2()
    test3()
    pass

if __name__ == "__main__":
    main()