from copy import deepcopy
from functools import partial
from itertools import combinations
from models import REGISTRY as mo_REGISTRY
import numpy as np
from numpy.random import randint
import torch as th
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop

from debug.debug import IS_PYCHARM_DEBUG
from components.scheme import Scheme
from components.transforms import _adim, _bsdim, _tdim, _vdim, \
    _generate_input_shapes, _generate_scheme_shapes, _build_model_inputs, \
    _join_dicts, _seq_mean, _copy_remove_keys, _make_logging_str, _underscore_to_cap, _check_nan
from components.losses import EntropyRegularisationLoss
from components.transforms import _to_batch, \
    _from_batch, _naninfmean
from utils.mackrel import _n_agent_pairings, _agent_ids_2_pairing_id, _pairing_id_2_agent_ids, _n_agent_pair_samples, _agent_ids_2_pairing_id

from .basic import BasicLearner

class FLOUNDERLPolicyLoss(nn.Module):

    def __init__(self):
        super(FLOUNDERLPolicyLoss, self).__init__()

    def forward(self, policies, advantages, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        policy_mask = (policies == 0.0)
        log_policies = th.log(policies.masked_fill(policy_mask, 1.0))
        log_policies = log_policies.masked_fill(policy_mask, 0.0)
        log_policies[log_policies!=log_policies] = 0.0 # just take out of final loss product

        _adv = advantages.unsqueeze(0).clone().detach()
        _adv=_adv.repeat(log_policies.shape[_adim(tformat)],1,1,1)
        _adv[_adv != _adv] = 0.0 # n-step return leads to NaNs

        # _act = actions.clone()
        # _act[_act!=_act] = 0.0 # mask NaNs in _act
        #
        # _active_logits = th.gather(log_policies, _vdim(tformat), _act.long())
        # _active_logits[actions != actions] = 0.0 # mask logits for actions that are actually NaNs
        # _adv[actions != actions] = 0.0

        loss_mean = -(log_policies.squeeze(_vdim(tformat)) * _adv.squeeze(_vdim(tformat))).mean(dim=_bsdim(tformat)) #DEBUG: MINUS?
        output_tformat = "a*t"

        return loss_mean, output_tformat

class FLOUNDERLCriticLoss(nn.Module):

    def __init__(self):
        super(FLOUNDERLCriticLoss, self).__init__()
    def forward(self, input, target, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        # targets may legitimately have NaNs - want to zero them out, and also zero out inputs at those positions
        # target = target.detach()
        target_nans = (target != target)
        target[target_nans] = 0.0
        input[target_nans] = 0.0

        # calculate mean-square loss
        ret = (input - target.detach())**2

        # sum over whole sequences
        ret = ret.mean(dim=_tdim(tformat), keepdim=True)

        #sum over agents
        ret = ret.mean(dim=_adim(tformat), keepdim=True)

        # average over batches
        ret = ret.mean(dim=_bsdim(tformat), keepdim=True)

        output_tformat = "s" # scalar
        return ret, output_tformat

class FLOUNDERLLearner(BasicLearner):

    def __init__(self, multiagent_controller, logging_struct=None, args=None):
        self.args = args
        self.multiagent_controller = multiagent_controller
        # self.agents = multiagent_controller.agents # for now, do not use any other multiagent controller functionality!!
        self.n_agents = multiagent_controller.n_agents
        self.n_actions = self.multiagent_controller.n_actions
        for _i in range(1, 4):
            setattr(self, "T_policy_level{}".format(_i), 0)
            setattr(self, "T_critic_level{}".format(_i), 0)

        #self.T_target_critic_update_interval=args.target_critic_update_interval
        self.stats = {}
        # self.n_critic_learner_reps = args.n_critic_learner_reps
        self.logging_struct = logging_struct

        self.critic_class = mo_REGISTRY[self.args.flounderl_critic]

        self.critic_scheme = Scheme([#dict(name="actions",
                                     #     rename="past_actions",
                                     #     select_agent_ids=list(range(self.n_agents)),
                                     #     transforms=[("shift", dict(steps=1)),
                                     #                ("one_hot", dict(range=(0, self.n_actions-1)))],
                                     #   switch=self.args.flounderl_critic_use_past_actions),
                                     #dict(name="actions",
                                     #     select_agent_ids=list(range(self.n_agents)),
                                     #     transforms=[("one_hot", dict(range=(0, self.n_actions - 1)))]),
                                     dict(name="state")
                                   ])
        self.target_critic_scheme = self.critic_scheme

        # Set up schemes
        self.scheme = {}
        # level 1
        # for _i in range(self.n_agents):
        #     self.scheme["critic__agent{}".format(_i)] = self.critic_scheme
        #     self.scheme["target_critic__agent{}".format(_i)] = self.critic_scheme
        self.scheme["critic"] = self.critic_scheme
        self.scheme["target_critic"] = self.critic_scheme

        # create joint scheme from the critic scheme
        self.joint_scheme_dict = _join_dicts(self.scheme,
                                             self.multiagent_controller.joint_scheme_dict)

        # construct model-specific input regions
        self.input_columns = {}
        self.input_columns["critic"] = {"vfunction":Scheme([{"name":"state"},
                                                           ])}
        self.input_columns["target_critic"] = self.input_columns["critic"]

        # for _i in range(self.n_agents):
        #     self.input_columns["critic__agent{}".format(_i)] = {"vfunction":Scheme([{"name":"state"},
        #                                                                             #{"name":"past_actions",
        #                                                                             # "select_agent_ids":list(range(self.n_agents))},
        #                                                                             #{"name": "actions",
        #                                                                             # "select_agent_ids": list(
        #                                                                             #     range(self.n_agents))}
        #                                                                             ])}
        #
        # for _i in range(self.n_agents):
        #     self.input_columns["target_critic__agent{}".format(_i)] = self.input_columns["critic__agent{}".format(_i)]


        self.last_target_update_T_critic = 0
        self.T_critic = 0
        self.T_policy = 0
        pass


    def create_models(self, transition_scheme):

        self.scheme_shapes = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                     dict_of_schemes=self.scheme)

        self.input_shapes = _generate_input_shapes(input_columns=self.input_columns,
                                                   scheme_shapes=self.scheme_shapes)


        # set up critic model
        self.critic_model = self.critic_class(input_shapes=self.input_shapes["critic"],
                                              n_agents=self.n_agents,
                                              n_actions=self.n_actions,
                                              args=self.args)
        if self.args.use_cuda:
            self.critic_model = self.critic_model.cuda()
        self.target_critic_model = deepcopy(self.critic_model)


        # set up optimizers
        if self.args.agent_share_params:
            self.agent_parameters = self.multiagent_controller.get_parameters()
        else:
            assert False, "TODO"
        self.agent_optimiser = RMSprop(self.agent_parameters, lr=self.args.lr_agent)

        self.critic_parameters = []
        if self.args.critic_share_params:
            self.critic_parameters.extend(self.critic_model.parameters())
        else:
            assert False, "TODO"
        self.critic_optimiser = RMSprop(self.critic_parameters, lr=self.args.lr_critic)

        # this is used for joint retrieval of data from all schemes
        self.joint_scheme_dict = _join_dicts(self.scheme, self.multiagent_controller.joint_scheme_dict)

        self.args_sanity_check() # conduct FLOUNDERL sanity check on arg parameters
        pass

    def args_sanity_check(self):
        """
        :return:
        """
        pass

    def train(self,
              batch_history,
              T_env=None):

        # -------------------------------------------------------------------------------
        # |  We follow the algorithmic description of FLOUNDERL as supplied in Algorithm 1   |
        # |  (Counterfactual Multi-Agent Policy Gradients, Foerster et al 2018)         |
        # |  Note: Instead of for-looping backwards through the sample, we just run     |
        # |  repetitions of the optimization procedure sampling from the same batch     |
        # -------------------------------------------------------------------------------

        # Retrieve and view all data that can be retrieved from batch_history in a single step (caching efficient)

        data_inputs, data_inputs_tformat = batch_history.view(dict_of_schemes=self.joint_scheme_dict,
                                                              to_cuda=self.args.use_cuda,
                                                              to_variable=True,
                                                              bs_ids=None,
                                                              fill_zero=True)

        self._train(batch_history, data_inputs, data_inputs_tformat, T_env)
        pass

    def _train(self, batch_history, data_inputs, data_inputs_tformat, T_env):
        # Update target if necessary
        if (self.T_critic - self.last_target_update_T_critic) / self.args.T_target_critic_update_interval > 1.0:
            self.update_target_nets(level=1)
            self.last_target_update_T_critic_level1 = self.T_critic
            print("updating target net!")

        # assert self.n_agents <= 4, "not implemented for >= 4 agents!"
        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions",
                                                         agent_ids=list(range(0, self.n_agents)),
                                                         stack=True)
        rewards, rewards_tformat = batch_history["reward"]
        # actions = actions.unsqueeze(0)
        #th.stack(actions_level1)


        # do single forward pass in critic
        flounderl_model_inputs, flounderl_model_inputs_tformat = _build_model_inputs(column_dict=self.input_columns,
                                                                                     inputs=data_inputs,
                                                                                     inputs_tformat=data_inputs_tformat,
                                                                                     to_variable=True,
                                                                                     stack=True)
        # pimp up to agent dimension 1
        #flounderl_model_inputs = {_k1:{_k2:_v2.unsqueeze(0) for _k2, _v2 in _v1.items()} for _k1, _v1 in flounderl_model_inputs.items()}
        flounderl_model_inputs_tformat = "bs*t*v"

        #data_inputs = {_k1:{_k2:_v2.unsqueeze(0) for _k2, _v2 in _v1.items()} for _k1, _v1 in data_inputs.items()}
        #data_inputs_tformat = "a*bs*t*v"

        self._optimize(batch_history=batch_history,
                       flounderl_model_inputs=flounderl_model_inputs,
                       flounderl_model_inputs_tformat=flounderl_model_inputs_tformat,
                       data_inputs=data_inputs,
                       data_inputs_tformat=data_inputs_tformat,
                       agent_optimiser=self.agent_optimiser,
                       agent_parameters=self.agent_parameters,
                       critic_optimiser=self.critic_optimiser,
                       critic_parameters=self.critic_parameters,
                       critic=self.critic_model,
                       target_critic=self.target_critic_model,
                       flounderl_critic_use_sampling=self.args.flounderl_critic_use_sampling,
                       flounderl_critic_sample_size=self.args.flounderl_critic_sample_size,
                       T_critic_str="T_critic",
                       T_policy_str="T_policy",
                       T_env=T_env,
                       level=1,
                       actions=actions,
                       rewards=rewards,
                       gamma=self.args.gamma)
        pass

    def _optimize(self,
                  batch_history,
                  flounderl_model_inputs,
                  flounderl_model_inputs_tformat,
                  data_inputs,
                  data_inputs_tformat,
                  agent_optimiser,
                  agent_parameters,
                  critic_optimiser,
                  critic_parameters,
                  critic,
                  target_critic,
                  flounderl_critic_use_sampling,
                  flounderl_critic_sample_size,
                  T_critic_str,
                  T_policy_str,
                  T_env,
                  level,
                  actions,
                  rewards,
                  gamma
                  ):

        critic_loss_arr = []
        critic_mean_arr = []
        target_critic_mean_arr = []
        critic_grad_norm_arr = []

        def _optimize_critic(**kwargs):
            level = kwargs["level"]
            inputs_critic= kwargs["flounderl_model_inputs"]["critic"]
            inputs_target_critic=kwargs["flounderl_model_inputs"]["target_critic"]
            inputs_critic_tformat=kwargs["tformat"]
            inputs_target_critic_tformat = kwargs["tformat"]

            # construct target-critic targets and carry out necessary forward passes
            # same input scheme for both target critic and critic!
            output_target_critic, output_target_critic_tformat = target_critic.forward(inputs_target_critic,
                                                                                       actions=actions,
                                                                                       tformat=flounderl_model_inputs_tformat)


            # values = th.max(output_target_critic["qvalue"], dim=)
            target_critic_td_targets, \
            target_critic_td_targets_tformat = batch_history.get_stat("td_lambda_targets",
                                                                       bs_ids=None,
                                                                       td_lambda=self.args.td_lambda,
                                                                       gamma=self.args.gamma,
                                                                       value_function_values=output_target_critic["vvalue"].unsqueeze(0).detach(),
                                                                       to_variable=True,
                                                                       n_agents=1,
                                                                       to_cuda=self.args.use_cuda)

            # sample!!
            if flounderl_critic_use_sampling:
                critic_shape = inputs_critic[list(inputs_critic.keys())[0]].shape
                sample_ids = randint(critic_shape[_bsdim(inputs_target_critic_tformat)] \
                                        * critic_shape[_tdim(inputs_target_critic_tformat)],
                                     size = flounderl_critic_sample_size)
                sampled_ids_tensor = th.from_numpy(sample_ids).long().cuda() if inputs_critic[list(inputs_critic.keys())[0]].is_cuda else th.from_numpy(sample_ids).long()
                _inputs_critic = {}
                for _k, _v in inputs_critic.items():
                    batch_sample = _v.view(
                        _v.shape[_adim(inputs_critic_tformat)],
                        -1,
                        _v.shape[_vdim(inputs_critic_tformat)])[:, sampled_ids_tensor, :]
                    _inputs_critic[_k] = batch_sample.view(_v.shape[_adim(inputs_critic_tformat)],
                                                      -1,
                                                      1,
                                                      _v.shape[_vdim(inputs_critic_tformat)])

                batch_sample_qtargets = target_critic_td_targets.view(target_critic_td_targets.shape[_adim(inputs_critic_tformat)],
                                                                      -1,
                                                                      target_critic_td_targets.shape[_vdim(inputs_critic_tformat)])[:, sampled_ids_tensor, :]
                vtargets = batch_sample_qtargets.view(target_critic_td_targets.shape[_adim(inputs_critic_tformat)],
                                                      -1,
                                                      1,
                                                      target_critic_td_targets.shape[_vdim(inputs_critic_tformat)])
            else:
                _inputs_critic = inputs_critic
                vtargets = target_critic_td_targets

            output_critic, output_critic_tformat = critic.forward(_inputs_critic,
                                                                  actions = actions,
                                                                  tformat=flounderl_model_inputs_tformat)


            critic_loss, \
            critic_loss_tformat = FLOUNDERLCriticLoss()(input=output_critic["vvalue"].unsqueeze(0),
                                                        target=Variable(vtargets, requires_grad=False),
                                                        tformat=target_critic_td_targets_tformat)

            # optimize critic loss
            critic_optimiser.zero_grad()
            critic_loss.backward()

            critic_grad_norm = th.nn.utils.clip_grad_norm(critic_parameters, 50)
            critic_optimiser.step()

            # Calculate critic statistics and update
            target_critic_mean = _naninfmean(output_target_critic["vvalue"])

            critic_mean = _naninfmean(output_critic["vvalue"])

            critic_loss_arr.append(np.asscalar(critic_loss.data.cpu().numpy()))
            critic_mean_arr.append(critic_mean)
            target_critic_mean_arr.append(target_critic_mean)
            critic_grad_norm_arr.append(critic_grad_norm)

            setattr(self, T_critic_str, getattr(self, T_critic_str) + len(batch_history) * batch_history._n_t)

            return output_critic


        output_critic = None

        # optimize the critic as often as necessary to get the critic loss down reliably
        for _i in range(getattr(self.args, "n_critic_learner_reps".format(level))): #self.n_critic_learner_reps):
            _ = _optimize_critic(flounderl_model_inputs=flounderl_model_inputs,
                                 tformat=flounderl_model_inputs_tformat,
                                 actions=actions,
                                 level=level)

        # get advantages
        output_critic, output_critic_tformat = critic.forward(flounderl_model_inputs["critic"],
                                                              actions=actions,
                                                              tformat=flounderl_model_inputs_tformat)
        TD = rewards.clone().zero_()
        TD[:, :-1, :] = rewards[:, 1:, :] + gamma * output_critic["vvalue"][:, 1:, :] - output_critic["vvalue"][:, :-1, :]
        n_step_return = TD # TODO: 1-step return so far only
        # n_step_return, n_step_return_tformat = batch_history.get_stat("n_step_return",
        #                                                                bs_ids=None,
        #                                                                n=self.args.n_step_return_n,
        #                                                                gamma=self.args.gamma,
        #                                                                value_function_values=output_critic["vvalue"].unsqueeze(0).detach(),
        #                                                                to_variable=True,
        #                                                                n_agents=1,
        #                                                                to_cuda=self.args.use_cuda)

        advantages = n_step_return - output_critic["vvalue"] # Q-V

        # only train the policy once in order to stay on-policy!
        policy_loss_function = partial(FLOUNDERLPolicyLoss(),
                                       advantages=advantages.detach())

        hidden_states, hidden_states_tformat = self.multiagent_controller.generate_initial_hidden_states(
            len(batch_history))

        agent_controller_output, \
        agent_controller_output_tformat = self.multiagent_controller.get_outputs(data_inputs,
                                                                                 hidden_states=hidden_states,
                                                                                 loss_fn=policy_loss_function,
                                                                                 loss_level=level,
                                                                                 tformat=data_inputs_tformat,
                                                                                 avail_actions=None,
                                                                                 test_mode=False,
                                                                                 actions=actions,
                                                                                 batch_history=batch_history)
        FLOUNDERL_loss, _ = agent_controller_output["losses"]
        FLOUNDERL_loss = FLOUNDERL_loss.mean()

        if self.args.flounderl_use_entropy_regularizer:
            FLOUNDERL_loss += self.args.flounderl_entropy_loss_regularization_factor * \
                         EntropyRegularisationLoss()(policies=agent_controller_output["policies"],
                                                     tformat="a*bs*t*v").sum()

        # carry out optimization for agents

        agent_optimiser.zero_grad()
        FLOUNDERL_loss.backward()

        #if self.args.debug_mode:
        #    _check_nan(agent_parameters)
        policy_grad_norm = th.nn.utils.clip_grad_norm(agent_parameters, 50)
        try:
            _check_nan(agent_parameters)
            # agent_optimiser.step() # DEBUG
        except:
            print("NaN in gradient or model!")
            for p in agent_parameters:
                print(p.grad)
            a = 5

        #for p in self.agent_level1_parameters:
        #    print('===========\ngradient:\n----------\n{}'.format(p.grad))

        # increase episode counter (the fastest one is always)
        setattr(self, T_policy_str, getattr(self, T_policy_str) + len(batch_history) * batch_history._n_t)

        # Calculate policy statistics
        advantage_mean = _naninfmean(advantages)
        self._add_stat("advantage_mean_level{}".format(level), advantage_mean, T_env=T_env)
        self._add_stat("policy_grad_norm_level{}".format(level), policy_grad_norm, T_env=T_env)
        self._add_stat("policy_loss_level{}".format(level), FLOUNDERL_loss.data.cpu().numpy(), T_env=T_env)
        self._add_stat("critic_loss_level{}".format(level), np.mean(critic_loss_arr), T_env=T_env)
        self._add_stat("critic_mean_level{}".format(level), np.mean(critic_mean_arr), T_env=T_env)
        self._add_stat("target_critic_mean_level{}".format(level), np.mean(target_critic_mean_arr), T_env=T_env)
        self._add_stat("critic_grad_norm_level{}".format(level), np.mean(critic_grad_norm_arr), T_env=T_env)
        self._add_stat(T_policy_str, getattr(self, T_policy_str), T_env=T_env)
        self._add_stat(T_critic_str, getattr(self, T_critic_str), T_env=T_env)

        pass

    def update_target_nets(self, level):
        if self.args.critic_level1_share_params and level==1:
            # self.target_critic.load_state_dict(self.critic.state_dict())
            self.critic_models["level1"].load_state_dict(self.critic_models["level1"].state_dict())
        if self.args.critic_level2_share_params and level==2:
            self.critic_models["level2_0:1"].load_state_dict(self.critic_models["level2_0:1"].state_dict())
        if self.args.critic_level3_share_params and level==3:
            self.critic_models["level3_{}".format(0)].load_state_dict(self.critic_models["level3_{}".format(0)].state_dict())
        pass

    def get_stats(self):
        if hasattr(self, "_stats"):
            return self._stats
        else:
            return []

    def log(self, log_directly = True):
        """
        Each learner has it's own logging routine, which logs directly to the python-wide logger if log_directly==True,
        and returns a logging string otherwise

        Logging is triggered in run.py
        """
        stats = self.get_stats()
        logging_dict = {}
        logging_str = ""
        for _i in range(1,4):
            logging_dict.update({"advantage_mean_level{}".format(_i): _seq_mean(stats["advantage_mean_level{}".format(_i)]),
                                 "critic_grad_norm_level{}".format(_i): _seq_mean(stats["critic_grad_norm_level{}".format(_i)]),
                                 "critic_loss_level{}".format(_i):_seq_mean(stats["critic_loss_level{}".format(_i)]),
                                 "policy_grad_norm_level{}".format(_i): _seq_mean(stats["policy_grad_norm_level{}".format(_i)]),
                                 "policy_loss_level{}".format(_i): _seq_mean(stats["policy_loss_level{}".format(_i)]),
                                 "target_critic_mean_level{}".format(_i): _seq_mean(stats["target_critic_mean_level{}".format(_i)]),
                                 "T_critic_level{}".format(_i): getattr(self, "T_critic_level{}".format(_i)),
                                 "T_policy_level{}".format(_i): getattr(self, "T_policy_level{}".format(_i))}
                                )
            logging_str = "T_policy_level{}={:g}, T_critic_level{}={:g}, ".format(_i, logging_dict["T_policy_level{}".format(_i)],
                                                                                  _i, logging_dict["T_critic_level{}".format(_i)])

        logging_str += _make_logging_str(_copy_remove_keys(logging_dict, ["T_policy_level1",
                                                                          "T_critic_level1",
                                                                          "T_policy_level2",
                                                                          "T_critic_level2",
                                                                          "T_policy_level3",
                                                                          "T_critic_level3"]))

        if log_directly:
            self.logging_struct.py_logger.info("{} LEARNER INFO: {}".format(self.args.learner.upper(), logging_str))

        return logging_str, logging_dict

    def save_models(self, path, token, T):

        self.multiagent_controller.save_models(path=path, token=token, T=T)
        th.save(self.critic.state_dict(),"results/models/{}/{}_critic__{}_T.weights".format(self.args.learner,
                                                                                            token,
                                                                                            T))
        th.save(self.target_critic.state_dict(), "results/models/{}/{}_target_critic__{}_T.weights".format(self.args.learner,
                                                                                                           token,
                                                                                                           T))
        pass