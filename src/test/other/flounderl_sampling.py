# from utils.blitzz.runners import NStepRunner
# from utils.blitzz.replay_buffer import ContiguousReplayBuffer
# from utils.blitzz.scheme import Scheme
# from utils.blitzz.sequence_buffer import BatchEpisodeBuffer
# from utils.blitzz.learners.coma import COMALearner, COMAPolicyLoss, COMACriticLoss
import numpy.testing as npt
from torch import nn
from utils.dict2namedtuple import convert
from utils.mackrel import _ordered_agent_pairings

from models import REGISTRY as mo_REGISTRY
from runners import REGISTRY as r_REGISTRY

import datetime
import numpy as np
import torch as th
from torch.autograd import Variable

def test1():
    """
    Simple 3-agent test without any NaN-entries anywhere
    """
    _args = dict(env="pred_prey",
                 n_agents=4,
                 t_max=1000000,
                 learner="coma",
                 env_args=dict(prey_movement="escape",
                               predator_prey_shape=(3,3),
                               nagent_capture_enabled=False,
                               predator_prey_toroidal=True,
                               n_agents=3,
                               n_prey=1,
                               agent_obs=(1,1),
                               episode_limit=3, # 40
                               intersection_global_view=True),
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
                 batch_size=2, #32,
                 batch_size_run=2, #32,
                 n_critic_learner_reps=200,
                 runner="flounderl",
                 n_loops_per_thread_or_sub_or_main_process=0,
                 n_threads_per_subprocess_or_main_process=0,
                 n_subprocesses=0,
                 multiagent_controller="flounderl_mac",
                 flounderl_agent_model="flounderl_agent",
                 flounderl_agent_model_level1="flounderl_recurrent_agent_level1",
                 flounderl_agent_model_level2="flounderl_recurrent_agent_level2",
                 flounderl_agent_model_level3="flounderl_recurrent_agent_level3",
                 flounderl_agent_use_past_actions=True,
                 flounderl_critic='flounderl_critic',
                 flounderl_critic_sample_size=1000,
                 flounderl_critic_use_past_actions=True,
                 flounderl_critic_use_sampling=False,
                 flounderl_delegation_probability_bias=0.0,
                 flounderl_entropy_loss_regularization_factor=5e-06,
                 flounderl_epsilon_decay_mode_level1='exp',
                 flounderl_epsilon_decay_mode_level2='exp',
                 flounderl_epsilon_decay_mode_level3='exp',
                 flounderl_epsilon_finish_level1=0.01,
                 flounderl_epsilon_finish_level2=0.01,
                 flounderl_epsilon_finish_level3=0.01,
                 flounderl_epsilon_start_level1=0.5,
                 flounderl_epsilon_start_level2=0.5,
                 flounderl_epsilon_start_level3=0.5,
                 flounderl_epsilon_time_length_level1=50000,
                 flounderl_epsilon_time_length_level2=50000,
                 flounderl_epsilon_time_length_level3=50000,
                 flounderl_exploration_mode_level1="softmax",
                 flounderl_exploration_mode_level2="softmax",
                 flounderl_exploration_mode_level3="softmax",
                 flounderl_use_entropy_regularizer=False,
                 flounderl_use_obs_intersections=True,
                 use_cuda=False, #True,
                 agent_level1_share_params=True,
                 agent_level2_share_params=True,
                 agent_level3_share_params=True,
                 agents_hidden_state_size=64,
                 debug_mode=True,
                 n_pair_samples=1,
                 debug_verbose=False,
                 share_agent_params=True
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
    _logging_struct = None

    args = convert(_args)
    # set up train runner
    runner_obj = r_REGISTRY[args.runner](args=args,
                                         logging_struct=_logging_struct)
    batch_history = runner_obj.run(test_mode=False)

    action_selection_inputs, \
    action_selection_inputs_tformat = runner_obj.episode_buffer.view(
        dict_of_schemes=runner_obj.multiagent_controller.joint_scheme_dict,
        to_cuda=runner_obj.args.use_cuda,
        to_variable=True,
        bs_ids=None,
        fill_zero=True,  # TODO: DEBUG!!!
        )

    avail_actions, avail_actions_format = runner_obj.episode_buffer.get_col(col="avail_actions",
                                                                            agent_ids=list(range(runner_obj.n_agents)))

    data_inputs, data_inputs_tformat = batch_history.view(  dict_of_schemes=runner_obj.multiagent_controller.joint_scheme_dict,
                                                            to_cuda=args.use_cuda,
                                                            to_variable=True,
                                                            bs_ids=None,
                                                            fill_zero=True)

    bh = batch_history.to_pd()

    stats = {}
    probs = {}
    n_samples = 10
    for _i in range(n_samples):

        hidden_states, hidden_states_tformat = runner_obj.multiagent_controller.generate_initial_hidden_states(
            len(batch_history))

        hidden_states, selected_actions, action_selector_outputs, selected_actions_format = \
            runner_obj.multiagent_controller.select_actions(inputs=action_selection_inputs,
                                                            avail_actions=avail_actions,
                                                            tformat=action_selection_inputs_tformat,
                                                            info=dict(T_env=runner_obj.T_env),
                                                            hidden_states=hidden_states,
                                                            test_mode=True)

        env_actions = [ a["data"] for a in selected_actions if (a["name"] == "actions")][0]
        b = env_actions.max()

        hidden_states, hidden_states_tformat = runner_obj.multiagent_controller.generate_initial_hidden_states(
            len(batch_history))

        agent_controller_output, \
        agent_controller_output_tformat = runner_obj.multiagent_controller.get_outputs(data_inputs,
                                                                                       hidden_states=hidden_states,
                                                                                       loss_fn=lambda policies, tformat: None,
                                                                                       tformat=data_inputs_tformat,
                                                                                       avail_actions=None,
                                                                                       test_mode=False,
                                                                                       actions=env_actions,
                                                                                       batch_history=batch_history)
        print("sample ", _i)
        actions_view = env_actions.view(runner_obj.n_agents, -1)
        policies_view = agent_controller_output["policies"].view(1, -1)
        for act in range(actions_view.shape[1]):
            if not act in stats:
                stats[act] = {}
            act_tuple = tuple(actions_view[:, act].tolist())
            if not act_tuple in stats[act]:
                stats[act][act_tuple] = 1.0
            else:
                stats[act][act_tuple] += 1.0
            if not act in probs:
                probs[act] = {}
            if not act_tuple in probs[act]:
                probs[act][act_tuple] = policies_view[:, act].item()

    import pprint
    pprint.pprint({ k:{ _k: "{} ({})".format(_v / float(n_samples), probs[k][_k]) for _k, _v in v.items()} for k, v in stats.items()})

    a= 5

    # n_actions = 2
    # n_agents = 3
    # n_agent_pairings = len(_ordered_agent_pairings(n_agents))

    # level1_out = th.zeros(1,1,1,n_agent_pairings)
    # level2_out = th.zeros(n_agent_pairings,1,1,n_actions*n_actions+2)
    # level3_out = th.zeros(n_agents,1,1,n_actions+1)
    #
    # level1_out[0,0,0,:] = th.FloatTensor([0.2, 0.3, 0.5])
    # level2_out[:,0,0,:] = th.FloatTensor([[0.1, 0.2, 0.4, 0.1, 0.1, 0.1],[0.5, 0.1, 0.1, 0.05, 0.2, 0.05],[0.0, 0.2, 0.3, 0.05, 0.45, 0.0]])
    # level3_out[:,0,0,:] = th.FloatTensor([[0.3, 0.7, 0.0], [0.2, 0.7, 0.1], [0.7, 0.25, 0.05]])
    #
    # avail_actions_pair = th.ones(n_agent_pairings,1,1,n_actions*n_actions+2)
    # actions = th.zeros(n_agents,1,1,1)
    # actions[:,0,0,0] = th.LongTensor([1, 0, 1])
    # avail_actions = th.ones(n_agents, 1, 1, n_actions + 1)
    #
    # _args.update(_required_args)
    #
    # _args["target_update_interval"] = _args["batch_size_run"] * 500
    # _args["env_args"] = {"predator_prey_shape":(3,3),
    #                      "prey_movement": "escape",
    #                      "nagent_capture_enabled":False}
    # args = convert(_args)
    #
    # model_class = mo_REGISTRY["flounderl_agent"]
    #
    # class Model(nn.Module):
    #     def __init__(self):
    #         super(Model, self).__init__()
    #         pass
    #
    # class tmp(model_class):
    #     def __init__(self, **kwargs):
    #         nn.Module.__init__(self)
    #         self.model_level1 = Model()
    #         self.models = {"level2_{}".format(0): Model(), "level3_{}".format(0): Model()}
    #         self.n_agents = n_agents
    #         self.n_actions = n_actions
    #         pass
    #
    # model = tmp()
    # def level1_forward(*args, **kwargs):
    #     return level1_out, None, None, "a*bs*t*v"
    # model.model_level1.forward = level1_forward
    #
    # def level2_forward(*args, **kwargs):
    #     return level2_out, None, None, "a*bs*t*v"
    # model.models["level2_{}".format(0)].forward = level2_forward
    #
    # def level3_forward(*args, **kwargs):
    #     return level3_out, None, None, "a*bs*t*v"
    # model.models["level3_{}".format(0)].forward = level3_forward
    #
    # tformat = {"level1":"a*bs*t*v", "level2":"a*bs*t*v", "level3": "a*bs*t*v"}
    # inputs = {"level1":{"agent_input_level1": None},
    #           "level2": {"agent_input_level2": {"avail_actions_pair":avail_actions_pair}},
    #           "level3": {"agent_input_level3": {"avail_actions":avail_actions}}}
    # hidden_states = {"level1":None, "level2":None, "level3":None}
    # ret = model(inputs=inputs,
    #             actions=actions,
    #             hidden_states=hidden_states,
    #             tformat=tformat,
    #             loss_fn=lambda policies, tformat: None)
    #
    # npt.assert_almost_equal(ret[0].sum().item(), 0.12795)

    pass
#
# def test2():
#     """
#     Simple 3-agent test without any NaN-entries anywhere
#     """
#     _args = dict(env="pred_prey",
#                  n_agents=4,
#                  t_max=1000000,
#                  learner="coma",
#                  env_args=dict(prey_movement="escape",
#                                predator_prey_shape=(3,3),
#                                nagent_capture_enabled=False),
#                  tensorboard=True,
#                  name="33pp_comatest_dqn_new",
#                  target_critic_update_interval=200000*2000, # DEBUG # 200000*20
#                  agent="basic",
#                  agent_model="DQN",
#                  observe=True,
#                  observe_db=True,
#                  action_selector="multinomial",
#                  use_blitzz=True,
#                  obs_last_action=True,
#                  test_interval=5000,
#                  epsilon_start=1.0,
#                  epsilon_finish=0.05,
#                  epsilon_time_length=100000,
#                  epsilon_decay="exp",
#                  test_nepisode=50,
#                  batch_size=32,
#                  batch_size_run=32,
#                  n_critic_learner_reps=200
#                  )
#
#     # required args
#     _required_args = dict(obs_epsilon=False,
#                           obs_agent_id=True,
#                           share_params=True,
#                           use_cuda=True,
#                           lr_critic=5e-4,
#                           lr_agent=5e-4,
#                           multiagent_controller="independent",
#                           td_lambda=1.0,
#                           gamma=0.99)
#
#     n_actions = 2
#     n_agents = 3
#     n_agent_pairings = len(_ordered_agent_pairings(n_agents))
#
#     level1_out = th.zeros(1,1,1,n_agent_pairings)
#     level2_out = th.zeros(n_agent_pairings,1,1,n_actions*n_actions+2)
#     level3_out = th.zeros(n_agents,1,1,n_actions+1)
#
#     level1_out[0,0,0,:] = th.FloatTensor([0.2, 0.3, 0.5])
#     level2_out[:,0,0,:] = th.FloatTensor([[0.1, 0.2, 0.4, 0.1, 0.1, 0.1],[0.5, 0.1, 0.1, 0.05, 0.2, 0.05],[0.0, 0.2, 0.3, 0.05, 0.45, 0.0]])
#     level3_out[:,0,0,:] = th.FloatTensor([[0.3, 0.7, 0.0], [0.2, 0.7, 0.1], [0.7, 0.25, 0.05]])
#
#     level1_out = th.cat([level1_out, level1_out * float("nan")], dim=2)
#     level2_out = th.cat([level2_out, level2_out * float("nan")], dim=2)
#     level3_out = th.cat([level3_out, level3_out * float("nan")], dim=2)
#
#     avail_actions_pair = th.ones(n_agent_pairings,1,1,n_actions*n_actions+2)
#     actions = th.zeros(n_agents,1,1,1)
#     actions[:,0,0,0] = th.LongTensor([1,0,1])
#     actions = th.cat([actions, actions*float("nan")], dim=2)
#     avail_actions = th.ones(n_agents, 1, 1, n_actions + 1)
#
#     _args.update(_required_args)
#
#     _args["target_update_interval"] = _args["batch_size_run"] * 500
#     _args["env_args"] = {"predator_prey_shape":(3,3),
#                          "prey_movement": "escape",
#                          "nagent_capture_enabled":False}
#     args = convert(_args)
#
#     model_class = mo_REGISTRY["flounderl_agent"]
#
#     class Model(nn.Module):
#         def __init__(self):
#             super(Model, self).__init__()
#             pass
#
#     class tmp(model_class):
#         def __init__(self, **kwargs):
#             nn.Module.__init__(self)
#             self.model_level1 = Model()
#             self.models = {"level2_{}".format(0): Model(), "level3_{}".format(0): Model()}
#             self.n_agents = n_agents
#             self.n_actions = n_actions
#             pass
#
#     model = tmp()
#     def level1_forward(*args, **kwargs):
#         return level1_out, None, None, "a*bs*t*v"
#     model.model_level1.forward = level1_forward
#
#     def level2_forward(*args, **kwargs):
#         return level2_out, None, None, "a*bs*t*v"
#     model.models["level2_{}".format(0)].forward = level2_forward
#
#     def level3_forward(*args, **kwargs):
#         return level3_out, None, None, "a*bs*t*v"
#     model.models["level3_{}".format(0)].forward = level3_forward
#
#     tformat = {"level1":"a*bs*t*v", "level2":"a*bs*t*v", "level3": "a*bs*t*v"}
#     inputs = {"level1":{"agent_input_level1": None},
#               "level2": {"agent_input_level2": {"avail_actions_pair":avail_actions_pair}},
#               "level3": {"agent_input_level3": {"avail_actions":avail_actions}}}
#     hidden_states = {"level1":None, "level2":None, "level3":None}
#     ret = model(inputs=inputs,
#                 actions=actions,
#                 hidden_states=hidden_states,
#                 tformat=tformat,
#                 loss_fn=lambda policies, tformat: None)
#
#     npt.assert_almost_equal(ret[0][0,0,0,0].item(), 0.12795)
#
#     pass
#

def main():
    test1()
    # test2()
    pass

if __name__ == "__main__":
    main()