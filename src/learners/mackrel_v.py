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
from components.transforms_old import _adim, _bsdim, _tdim, _vdim, \
    _generate_input_shapes, _generate_scheme_shapes, _build_model_inputs, \
    _join_dicts, _seq_mean, _copy_remove_keys, _make_logging_str, _underscore_to_cap, _check_nan
from components.losses import EntropyRegularisationLoss
from components.transforms_old import _to_batch, \
    _from_batch, _naninfmean
from utils.mackrel import _n_agent_pairings, _agent_ids_2_pairing_id, _pairing_id_2_agent_ids, _n_agent_pair_samples, _agent_ids_2_pairing_id

from .basic import BasicLearner

class MACKRELPolicyLoss(nn.Module):

    def __init__(self):
        super(MACKRELPolicyLoss, self).__init__()

    def forward(self, policies, advantages, actions, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        policy_mask = (policies == 0.0)
        log_policies = th.log(policies)
        log_policies = log_policies.masked_fill(policy_mask, 0.0)

        _adv = advantages.clone().detach()

        _act = actions.clone()
        _act[_act!=_act] = 0.0 # mask NaNs in _act

        _active_logits = th.gather(log_policies, _vdim(tformat), _act.long())
        _active_logits[actions != actions] = 0.0 # mask logits for actions that are actually NaNs
        _adv[actions != actions] = 0.0

        loss_mean = -(_active_logits.squeeze(_vdim(tformat)) * _adv.squeeze(_vdim(tformat))).mean(dim=_bsdim(tformat)) #DEBUG: MINUS?
        output_tformat = "a*t"

        return loss_mean, output_tformat

class MACKRELVLoss(nn.Module):

    def __init__(self):
        super(MACKRELVLoss, self).__init__()
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

        output_tformat = "s"
        return ret, output_tformat

class MACKRELVLearner(BasicLearner):

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

        self.V = mo_REGISTRY[self.args.mackrel_V]

        # set up input schemes for all of our models
        self.scheme_V = Scheme([dict(name="state")])
        self.scheme_target_V = self.scheme_V

        self.schemes = {}
        self.schemes["V__all_levels"] = self.scheme_V

        # create joint scheme from the critics schemes
        self.joint_scheme_dict = _join_dicts(self.schemes,
                                             self.multiagent_controller.joint_scheme_dict)

        # construct model-specific input regions

        self.input_columns_V = {
            "V__all_levels": {"main":Scheme([dict(name="state")])}
        }

        self.last_target_update_T_V = 0
        self.T_V = 0
        pass


    def create_models(self, transition_scheme):

        self.scheme_shapes_V = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                       dict_of_schemes=self.schemes)

        self.input_shapes_V = _generate_input_shapes(input_columns=self.input_columns_V,
                                                     scheme_shapes=self.scheme_shapes_V)


        # Set up V model

        self.V_model = self.V(input_shapes=self.input_shapes_V["V__all_levels"],
                              args=self.args)
        if self.args.use_cuda:
            self.V_model = self.V_model.cuda()
        self.target_V_model = deepcopy(self.V_model)

        # set up optimizers
        if self.args.agent_level1_share_params:
            self.agent_level1_parameters = self.multiagent_controller.get_parameters(level=1)
        else:
            assert False, "TODO"
        self.agent_level1_optimiser = RMSprop(self.agent_level1_parameters, lr=self.args.lr_agent_level1)

        if self.args.agent_level2_share_params:
            self.agent_level2_parameters = self.multiagent_controller.get_parameters(level=2)
        else:
            assert False, "TODO"
        self.agent_level2_optimiser = RMSprop(self.agent_level2_parameters, lr=self.args.lr_agent_level2)

        if self.args.agent_level3_share_params:
            self.agent_level3_parameters = self.multiagent_controller.get_parameters(level=3)
        else:
            assert False, "TODO"
        self.agent_level3_optimiser = RMSprop(self.agent_level3_parameters, lr=self.args.lr_agent_level3)

        self.V_parameters = list(self.V_model.parameters())
        self.V_optimiser = RMSprop(self.V_parameters, lr=self.args.lr_V)

        # this is used for joint retrieval of data from all schemes
        self.joint_scheme_dict_level1 = _join_dicts(self.multiagent_controller.joint_scheme_dict_level1)
        self.joint_scheme_dict_level2 = _join_dicts(self.multiagent_controller.joint_scheme_dict_level2)
        self.joint_scheme_dict_level3 = _join_dicts(self.multiagent_controller.joint_scheme_dict_level3)

        self.args_sanity_check() # conduct MACKREL sanity check on arg parameters
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
        # |  We follow the algorithmic description of MACKREL as supplied in Algorithm 1   |
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

        self.train_V(batch_history, data_inputs, data_inputs_tformat, T_env)
        self.train_level1(batch_history, data_inputs, data_inputs_tformat, T_env)
        self.train_level2(batch_history, data_inputs, data_inputs_tformat, T_env) # DEBUG
        self.train_level3(batch_history, data_inputs, data_inputs_tformat, T_env)
        pass

    def train_V(self, batch_history, data_inputs, data_inputs_tformat, T_env):
        if (self.T_V - self.last_target_update_T_V) / self.args.T_target_V_update_interval > 1.0:
            self.update_target_nets(level=1)
            self.last_target_update_T_V = self.T_V
            print("updating target net!")


        inputs_V, inputs_V_tformat = _build_model_inputs(column_dict=self.input_columns_V,
                                                         inputs=data_inputs,
                                                         inputs_tformat=data_inputs_tformat,
                                                         to_variable=True,
                                                         stack=False)

        V_loss_arr = []
        V_mean_arr = []
        target_V_mean_arr = []
        V_grad_norm_arr = []

        def _optimize_V():
            # construct target-critic targets and carry out necessary forward passes
            # same input scheme for both target critic and critic!
            inputs_target_V_tformat = inputs_V_tformat
            output_target_V, output_target_V_tformat = self.target_V_model.forward(inputs_V["V__all_levels"],
                                                                                   tformat=inputs_target_V_tformat)

            target_V_td_targets, \
            target_V_td_targets_tformat = batch_history.get_stat("td_lambda_targets",
                                                                  bs_ids=None,
                                                                  td_lambda=self.args.td_lambda,
                                                                  gamma=self.args.gamma,
                                                                  value_function_values=output_target_V["vvalue"].detach().unsqueeze(0),
                                                                  to_variable=True,
                                                                  n_agents=1,
                                                                  to_cuda=self.args.use_cuda)

            # sample!!
            if self.args.mackrel_V_use_sampling:
                V_shape = inputs_V["V__all_levels"][list(inputs_V["V__all_levels"].keys())[0]].shape
                sample_ids = randint(V_shape[_bsdim(inputs_target_V_tformat)] \
                                        * V_shape[_tdim(inputs_target_V_tformat)],
                                     size = self.args.mackrel_V_sample_size)
                sampled_ids_tensor = th.from_numpy(sample_ids).long().cuda() if inputs_V["V__all_levels"][list(inputs_V["V__all_levels"].keys())[0]].is_cuda else th.from_numpy(sample_ids).long()
                _inputs_V = {}
                for _k, _v in inputs_V["V__all_levels"].items():
                    batch_sample = _v.view(-1,
                                           _v.shape[_vdim(inputs_V_tformat)])[sampled_ids_tensor, :]
                    _inputs_V[_k] = batch_sample.view(-1,
                                                      1,
                                                      _v.shape[_vdim(inputs_V_tformat)])

                batch_sample_vtargets = target_V_td_targets.view(-1,
                                                                 target_V_td_targets.shape[_vdim(inputs_V_tformat)])[sampled_ids_tensor, :]
                vtargets = batch_sample_vtargets.view(-1,
                                                      1,
                                                      target_V_td_targets.shape[_vdim(inputs_V_tformat)])
            else:
                _inputs_V = inputs_V["V__all_levels"]
                vtargets = target_V_td_targets

            output_V, output_V_tformat = self.V_model.forward(_inputs_V,
                                                              tformat=inputs_V_tformat)

            output_V["vvalue"].unsqueeze_(0)

            V_loss, \
            V_loss_tformat = MACKRELVLoss()(input=output_V["vvalue"],
                                            target=Variable(vtargets, requires_grad=False),
                                            tformat=target_V_td_targets_tformat)

            # optimize critic loss
            self.V_optimiser.zero_grad()
            V_loss.backward()

            V_grad_norm = th.nn.utils.clip_grad_norm(self.V_parameters, 50)
            # print(V_grad_norm)
            try:
                _check_nan(self.V_parameters)
                self.V_optimiser.step()
            except:
                print("NaN in gradient or model!")

            # Calculate critic statistics and update
            target_V_mean = _naninfmean(output_target_V["vvalue"])

            V_mean = _naninfmean(output_V["vvalue"])

            V_loss_arr.append(np.asscalar(V_loss.data.cpu().numpy()))
            V_mean_arr.append(V_mean)
            target_V_mean_arr.append(target_V_mean)
            V_grad_norm_arr.append(V_grad_norm)

            self.T_V += len(batch_history) * batch_history._n_t

        # optimize the critic as often as necessary to get the critic loss down reliably
        for _i in range(getattr(self.args, "n_V_learner_reps")):
            _optimize_V()

        self._add_stat("V_loss", np.mean(V_loss_arr), T_env=T_env)
        self._add_stat("V_mean", np.mean(V_mean_arr), T_env=T_env)
        self._add_stat("target_V_mean", np.mean(target_V_mean_arr), T_env=T_env)
        self._add_stat("V_grad_norm", np.mean(V_grad_norm_arr), T_env=T_env)
        self._add_stat("T_V", self.T_V, T_env=T_env)

        pass

    def train_level1(self, batch_history, data_inputs, data_inputs_tformat, T_env):

        # assert self.n_agents <= 4, "not implemented for >= 4 agents!"
        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions_level1__sample{}".format(0),
                                                         # agent_ids=list(range(0, self.n_agents)),
                                                         stack=True)
        actions_level1 = actions.unsqueeze(0)
        #th.stack(actions_level1)


        self._optimize(batch_history=batch_history,
                       data_inputs=data_inputs,
                       data_inputs_tformat=data_inputs_tformat,
                       agent_optimiser=self.agent_level1_optimiser,
                       agent_parameters=self.agent_level1_parameters,
                       T_policy_str="T_policy_level1",
                       T_env=T_env,
                       level=1,
                       actions=actions_level1)
        pass

    def train_level2(self, batch_history, data_inputs, data_inputs_tformat, T_env):

        assert self.n_agents <= 3 or self.args.n_pair_samples == 1 , "only implemented for 3 or fewer agents, or if n_pair_samples == 1"
        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions_level2__sample{}".format(0))
        actions = actions.unsqueeze(0)


        self._optimize(batch_history,
                       data_inputs,
                       data_inputs_tformat,
                       agent_optimiser=self.agent_level2_optimiser,
                       agent_parameters=self.agent_level2_parameters,
                       T_policy_str="T_policy_level2",
                       T_env=T_env,
                       level=2,
                       actions=actions)
        pass

    def train_level3(self, batch_history, data_inputs, data_inputs_tformat, T_env):

        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions_level3",
                                                         agent_ids=list(range(0, self.n_agents)),
                                                         stack=True)

        self._optimize(batch_history,
                       data_inputs,
                       data_inputs_tformat,
                       agent_optimiser=self.agent_level3_optimiser,
                       agent_parameters=self.agent_level3_parameters,
                       T_policy_str="T_policy_level3",
                       T_env=T_env,
                       level=3,
                       actions=actions)
        pass

    def _optimize(self,
                  batch_history,
                  data_inputs,
                  data_inputs_tformat,
                  agent_optimiser,
                  agent_parameters,
                  T_policy_str,
                  T_env,
                  level,
                  actions
                  ):

        critic_loss_arr = []
        critic_mean_arr = []
        target_critic_mean_arr = []
        critic_grad_norm_arr = []



        # output_critic = None
        #
        # # optimize the critic as often as necessary to get the critic loss down reliably
        # for _i in range(getattr(self.args, "n_critic_level{}_learner_reps".format(level))): #self.n_critic_learner_reps):
        #     _ = _optimize_critic(mackrel_model_inputs=mackrel_model_inputs,
        #                          tformat=mackrel_model_inputs_tformat,
        #                          actions=actions,
        #                          level=level)

        inputs_V, inputs_V_tformat = _build_model_inputs(column_dict=self.input_columns_V,
                                                         inputs=data_inputs,
                                                         inputs_tformat=data_inputs_tformat,
                                                         to_variable=True,
                                                         stack=False)

        # get advantages
        output_V, output_V_tformat = self.V_model.forward(inputs_V["V__all_levels"],
                                                          tformat=inputs_V_tformat)

        target_V_td_targets, \
        target_V_td_targets_tformat = batch_history.get_stat("td_lambda_targets",
                                                             bs_ids=None,
                                                             td_lambda=0, # monte carlo!
                                                             gamma=self.args.gamma,
                                                             value_function_values=output_V["vvalue"].detach().unsqueeze(0),
                                                             to_variable=True,
                                                             n_agents=1,
                                                             to_cuda=self.args.use_cuda)
        advantages = Variable((target_V_td_targets - output_V["vvalue"].unsqueeze(0).data).repeat(self.n_agents, 1,1,1), requires_grad=False)
        advantages[advantages != advantages] = 0.0 # i.e.: the policies evaluated for last time step should not contribute to agent losses!

        # try:
        #     _check_nan(target_V_td_targets)
        # except Exception as e:
        #     a = 5
        #     pass
        #
        # try:
        #     _check_nan(output_V["vvalue"])
        # except Exception as e:
        #     a = 5
        #     pass
        #
        # try:
        #     _check_nan(advantages)
        # except Exception as e:
        #     a = 5
        #     pass
        #
        # try:
        #     _check_nan(actions)
        # except Exception as e:
        #     a = 5
        #     pass

        # only train the policy once in order to stay on-policy!
        policy_loss_function = partial(MACKRELPolicyLoss(),
                                       actions=Variable(actions),
                                       advantages=advantages)

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
                                                                                 batch_history=batch_history)
        MACKREL_loss = agent_controller_output["losses"]
        MACKREL_loss = MACKREL_loss.mean()


        if self.args.mackrel_use_entropy_regularizer:
            MACKREL_loss += self.args.mackrel_entropy_loss_regularization_factor * \
                         EntropyRegularisationLoss()(policies=agent_controller_output["policies"],
                                                     tformat="a*bs*t*v").sum()

        # carry out optimization for agents

        agent_optimiser.zero_grad()
        MACKREL_loss.backward()

        #if self.args.debug_mode:
        #    _check_nan(agent_parameters)
        policy_grad_norm = th.nn.utils.clip_grad_norm(agent_parameters, 50)
        try:
            _check_nan(agent_parameters)
            agent_optimiser.step()
        except:
            print("NaN in gradient or model!")

        #for p in self.agent_level1_parameters:
        #    print('===========\ngradient:\n----------\n{}'.format(p.grad))

        # increase episode counter (the fastest one is always)
        setattr(self, T_policy_str, getattr(self, T_policy_str) + len(batch_history) * batch_history._n_t)

        # Calculate policy statistics
        advantage_mean = _naninfmean(advantages)
        self._add_stat("advantage_mean_level{}".format(level), advantage_mean, T_env=T_env)
        self._add_stat("policy_grad_norm_level{}".format(level), policy_grad_norm, T_env=T_env)
        self._add_stat("policy_loss_level{}".format(level), MACKREL_loss.data.cpu().numpy(), T_env=T_env)
        self._add_stat(T_policy_str, getattr(self, T_policy_str), T_env=T_env)

        pass

    def update_target_nets(self, level):
        self.target_V_model.load_state_dict(self.V_model.state_dict())
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

        logging_dict.update({"V_grad_norm": _seq_mean(stats["V_grad_norm"]),
                             "V_loss":_seq_mean(stats["V_loss"]),
                             "T_V": getattr(self, "T_V"),
                             "target_critic_mean": _seq_mean(stats["target_V_mean"])})
        
        for _i in range(1,4):
            logging_dict.update({"advantage_mean_level{}".format(_i): _seq_mean(stats["advantage_mean_level{}".format(_i)]),
                                 "policy_grad_norm_level{}".format(_i): _seq_mean(stats["policy_grad_norm_level{}".format(_i)]),
                                 "policy_loss_level{}".format(_i): _seq_mean(stats["policy_loss_level{}".format(_i)]),
                                 "T_policy_level{}".format(_i): getattr(self, "T_policy_level{}".format(_i))}
                                )
            logging_str += "T_policy_level{}={:g}".format(_i, logging_dict["T_policy_level{}".format(_i)])

        logging_str += "T_V={:g}".format(logging_dict["T_V"])

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