from copy import deepcopy
from functools import partial
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
    _join_dicts, _seq_mean, _copy_remove_keys, _make_logging_str, _underscore_to_cap, \
    _n_step_return
from components.losses import EntropyRegularisationLoss
from components.transforms import _to_batch, _from_batch, _naninfmean
from models.centralV import CentralVCritic

from .basic import BasicLearner

class CentralVPolicyLoss(nn.Module):

    def __init__(self):
        super(CentralVPolicyLoss, self).__init__()

    def forward(self, policies, advantages, actions, tformat):

        actions = actions.clone()
        actions_mask = (actions != actions)
        actions[actions_mask] = 0.0
        policies = th.gather(policies, _vdim(tformat), actions.long())
        policies[actions_mask] = float("nan")
        mask = (policies < 10E-40) | (policies != policies)
        policies[mask] = 1.0
        log_policies = th.log(policies)

        _adv = advantages.clone().detach()
        _adv=_adv.repeat(log_policies.shape[_adim(tformat)],1,1,1)
        _adv[_adv != _adv] = 0.0 # n-step return leads to NaNs
        _adv[mask] = 0.0 # prevent gradients from flowing through illegitimate policy entries

        loss_mean = - (log_policies * _adv).mean()
        output_tformat = "a*t"

        return loss_mean, output_tformat

class CentralVCriticLoss(nn.Module):

    def __init__(self):
        super(CentralVCriticLoss, self).__init__()
    def forward(self, input, target, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        # targets may legitimately have NaNs - want to zero them out, and also zero out inputs at those positions
        # target = target.detach()
        nans = (target != target)
        target[nans] = 0.0
        input[nans] = 0.0

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

class CentralVLearner(BasicLearner):

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

        self.critic_class = CentralVCritic

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
        if self.args.share_agent_params:
            self.agent_parameters = self.multiagent_controller.get_parameters()
        else:
            assert False, "TODO"
        self.agent_optimiser = RMSprop(self.agent_parameters, lr=self.args.lr_agent)

        self.critic_parameters = []
        if not (hasattr(self.args, "critic_share_params") and not self.args.critic_share_params):
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
        # |  We follow the algorithmic description of CentralV as supplied in Algorithm 1   |
        # |  (Counterfactual Multi-Agent Policy Gradients, Foerster et al 2018)         |
        # |  Note: Instead of for-looping backwards through the sample, we just run     |
        # |  repetitions of the optimization procedure sampling from the same batch     |
        # -------------------------------------------------------------------------------


        # Update target if necessary
        if (self.T_critic - self.last_target_update_T_critic) / self.args.target_critic_update_interval > 1.0:
            self.update_target_nets()
            self.last_target_update_T_critic = self.T_critic
            print("updating target net!")

        # Retrieve and view all data that can be retrieved from batch_history in a single step (caching efficient)

        # create one single batch_history view suitable for all
        data_inputs, data_inputs_tformat = batch_history.view(dict_of_schemes=self.joint_scheme_dict,
                                                              to_cuda=self.args.use_cuda,
                                                              to_variable=True,
                                                              bs_ids=None,
                                                              fill_zero=True) # DEBUG: Should be True

        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions",
                                                         agent_ids=list(range(0, self.n_agents)),
                                                         stack=True)

        # do single forward pass in critic
        coma_model_inputs, coma_model_inputs_tformat = _build_model_inputs(column_dict=self.input_columns,
                                                                           inputs=data_inputs,
                                                                           inputs_tformat=data_inputs_tformat,
                                                                           to_variable=True)

        critic_loss_arr = []
        critic_mean_arr = []
        target_critic_mean_arr = []
        critic_grad_norm_arr = []

        def _optimize_critic(**kwargs):
            inputs_critic= kwargs["coma_model_inputs"]["critic"]
            inputs_target_critic=kwargs["coma_model_inputs"]["target_critic"]
            inputs_critic_tformat=kwargs["tformat"]
            inputs_target_critic_tformat = kwargs["tformat"]


            # construct target-critic targets and carry out necessary forward passes
            # same input scheme for both target critic and critic!
            output_target_critic, output_target_critic_tformat = self.target_critic_model.forward(inputs_target_critic,
                                                                                                  tformat="bs*t*v")


            target_critic_td_targets, \
            target_critic_td_targets_tformat = batch_history.get_stat("td_lambda_targets",
                                                                       bs_ids=None,
                                                                       td_lambda=self.args.td_lambda,
                                                                       gamma=self.args.gamma,
                                                                       value_function_values=output_target_critic["vvalue"].unsqueeze(0).detach(),
                                                                       to_variable=True,
                                                                       n_agents=1,
                                                                       to_cuda=self.args.use_cuda)

            _inputs_critic = inputs_critic
            vtargets = target_critic_td_targets

            output_critic, output_critic_tformat = self.critic_model.forward(_inputs_critic,
                                                                             tformat="bs*t*v")


            critic_loss, \
            critic_loss_tformat = CentralVCriticLoss()(input=output_critic["vvalue"],
                                                       target=Variable(vtargets.squeeze(0), requires_grad=False),
                                                       tformat=target_critic_td_targets_tformat)

            # optimize critic loss
            self.critic_optimiser.zero_grad()
            critic_loss.backward()

            critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_parameters,
                                                          50)
            self.critic_optimiser.step()

            # Calculate critic statistics and update
            target_critic_mean = _naninfmean(output_target_critic["vvalue"])

            critic_mean = _naninfmean(output_critic["vvalue"])

            critic_loss_arr.append(np.asscalar(critic_loss.data.cpu().numpy()))
            critic_mean_arr.append(critic_mean)
            target_critic_mean_arr.append(target_critic_mean)
            critic_grad_norm_arr.append(critic_grad_norm)

            self.T_critic += len(batch_history) * batch_history._n_t

            return output_critic


        output_critic = None
        # optimize the critic as often as necessary to get the critic loss down reliably
        for _i in range(self.args.n_critic_learner_reps):
            _ = _optimize_critic(coma_model_inputs=coma_model_inputs,
                                 tformat=coma_model_inputs_tformat,
                                 actions=actions)

        # get advantages
        output_critic, output_critic_tformat = self.critic_model.forward(coma_model_inputs["critic"],
                                                                         tformat="bs*t*v")
        # advantages = output_critic["advantage"]
        advantages = _n_step_return(values=output_critic["vvalue"].unsqueeze(0),
                                    rewards=batch_history["reward"][0],
                                    terminated=batch_history["terminated"][0],
                                    truncated=batch_history["truncated"][0],
                                    seq_lens=batch_history.seq_lens,
                                    horizon=batch_history._n_t-1,
                                    n=1 if not hasattr(self.args, "n_step_return_n") else self.args.n_step_return_n,
                                    gamma=self.args.gamma) - output_critic["vvalue"]

        # only train the policy once in order to stay on-policy!
        policy_loss_function = partial(CentralVPolicyLoss(),
                                       actions = actions,
                                       advantages=advantages.detach())

        hidden_states, hidden_states_tformat = self.multiagent_controller.generate_initial_hidden_states(
            len(batch_history))

        agent_controller_output, \
        agent_controller_output_tformat = self.multiagent_controller.get_outputs(data_inputs,
                                                                                 hidden_states=hidden_states,
                                                                                 loss_fn=policy_loss_function,
                                                                                 tformat=data_inputs_tformat,
                                                                                 test_mode=False)
        CentralV_loss = agent_controller_output["losses"]
        CentralV_loss = CentralV_loss.mean()

        if self.args.coma_use_entropy_regularizer:
            CentralV_loss += self.args.coma_entropy_loss_regularization_factor * \
                         EntropyRegularisationLoss()(policies=agent_controller_output["policies"],
                                                     tformat="a*bs*t*v").sum()

        # carry out optimization for agents
        self.agent_optimiser.zero_grad()
        CentralV_loss.backward()

        policy_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_parameters, 50)
        self.agent_optimiser.step()

        # increase episode counter (the fastest one is always)
        self.T_policy += len(batch_history) * batch_history._n_t

        # Calculate policy statistics
        advantage_mean = _naninfmean(advantages)
        self._add_stat("advantage_mean", advantage_mean, T_env=T_env)
        self._add_stat("policy_grad_norm", policy_grad_norm, T_env=T_env)
        self._add_stat("policy_loss", CentralV_loss.data.cpu().numpy(), T_env=T_env)
        self._add_stat("critic_loss", np.mean(critic_loss_arr), T_env=T_env)
        self._add_stat("critic_mean", np.mean(critic_mean_arr), T_env=T_env)
        self._add_stat("target_critic_mean", np.mean(target_critic_mean_arr), T_env=T_env)
        self._add_stat("critic_grad_norm", np.mean(critic_grad_norm_arr), T_env=T_env)
        self._add_stat("T_policy", self.T_policy, T_env=T_env)
        self._add_stat("T_critic", self.T_critic, T_env=T_env)

        pass

    def update_target_nets(self):
        self.target_critic_model.load_state_dict(self.critic_model.state_dict())

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
        logging_dict =  dict(advantage_mean = _seq_mean(stats["advantage_mean"]),
                             critic_grad_norm = _seq_mean(stats["critic_grad_norm"]),
                             critic_loss =_seq_mean(stats["critic_loss"]),
                             policy_grad_norm = _seq_mean(stats["policy_grad_norm"]),
                             policy_loss = _seq_mean(stats["policy_loss"]),
                             target_critic_mean = _seq_mean(stats["target_critic_mean"]),
                             T_critic=self.T_critic,
                             T_policy=self.T_policy
                            )
        logging_str = "T_policy={:g}, T_critic={:g}, ".format(logging_dict["T_policy"], logging_dict["T_critic"])
        logging_str += _make_logging_str(_copy_remove_keys(logging_dict, ["T_policy", "T_critic"]))

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