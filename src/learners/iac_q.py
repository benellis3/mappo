from copy import deepcopy
from functools import partial
import torch as th
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop

from debug.debug import IS_PYCHARM_DEBUG
from models.iac import IACCritic
from numpy.random import randint
from components.scheme import Scheme
from components.transforms import _adim, _bsdim, _tdim, _vdim, \
    _generate_input_shapes, _generate_scheme_shapes, _build_model_inputs, \
    _join_dicts, _seq_mean, _copy_remove_keys, _make_logging_str, _underscore_to_cap

from .basic import BasicLearner

class IACqPolicyLoss(nn.Module):

    def __init__(self):
        super(IACqPolicyLoss, self).__init__()

    def forward(self, policies, advantages, actions, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        log_policies = th.log(policies)
        _adv = advantages.clone().detach()
        _act = actions.clone()

        assert not (_act!=_act).any(), "_act has nan!"
        assert not (_act > log_policies.shape[_vdim(tformat)]).any(), "_act too large!: {} vs. {}".format(_act.max(),
                                                                                                          log_policies.shape)
        assert not (_act < 0).any(), "_act too small!"

        _active_logits = th.gather(log_policies, _vdim(tformat), _act.long())

        loss_mean = -(_active_logits.squeeze(_vdim(tformat)) * _adv.squeeze(_vdim(tformat))).mean(dim=_bsdim(tformat)) #DEBUG: MINUS?
        output_tformat = "a*t"
        return loss_mean, output_tformat

class IACqCriticLoss(nn.Module):

    def __init__(self):
        super(IACqCriticLoss, self).__init__()
    def forward(self, input, target, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        # targets may legitimately have NaNs - want to zero them out, and also zero out inputs at those positions
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

class IACqLearner(BasicLearner):

    def __init__(self, multiagent_controller, logging_struct, args):
        self.args = args
        self.multiagent_controller = multiagent_controller
        self.agents = multiagent_controller.agents # for now, do not use any other multiagent controller functionality!!
        self.n_agents = len(self.agents)
        self.n_actions = self.multiagent_controller.n_actions
        self.T_policy = 0
        self.T_critic = 0
        self.target_critic_update_interval=args.target_critic_update_interval
        self.stats = {}
        self.n_critic_learner_reps = args.n_critic_learner_reps
        self.logging_struct = logging_struct

        # set up input schemes for all of our models
        self.critic_scheme_fn = lambda _agent_id: Scheme([dict(name="agent_id",
                                                                select_agent_ids=[_agent_id],
                                                                transforms=[("one_hot", dict(range=(0, self.n_agents-1)))],
                                                               ),
                                                           dict(name="observations",
                                                                rename="agent_observation",
                                                                select_agent_ids=[_agent_id],
                                                               ),
                                                           dict(name="actions",
                                                                rename="past_actions",
                                                                transforms=[("shift", dict(steps=1, fill=0))],
                                                                select_agent_ids=[_agent_id],
                                                              ),
                                                           dict(name="actions",
                                                                rename="agent_action",
                                                                select_agent_ids=[_agent_id], # do NOT one-hot!
                                                                ),
                                                           dict(name="policies",
                                                               rename="agent_policy",
                                                               select_agent_ids=[_agent_id], )
                                                           ])
        self.target_critic_scheme_fn = self.critic_scheme_fn

        self.schemes = {}
        for _agent_id in range(self.n_agents):
            self.schemes["critic__agent{}".format(_agent_id)] = self.critic_scheme_fn(_agent_id).agent_flatten()

        for _agent_id in range(self.n_agents):
            self.schemes["target_critic__agent{}".format(_agent_id)] = self.target_critic_scheme_fn(_agent_id).agent_flatten()

        self.input_columns = {}
        for _agent_id in range(self.n_agents):
            self.input_columns["critic__agent{}".format(_agent_id)] = {}
            self.input_columns["critic__agent{}".format(_agent_id)]["qfunction"] = Scheme([dict(name="agent_action", select_agent_ids=[_agent_id]),
                                                                                           dict(name="agent_observation", select_agent_ids=[_agent_id]),
                                                                                           dict(name="agent_id", select_agent_ids=[_agent_id]),
                                                                                           dict(name="past_actions", select_agent_ids=[_agent_id])]).agent_flatten()
            self.input_columns["critic__agent{}".format(_agent_id)]["agent_action"] = Scheme([dict(name="agent_action", select_agent_ids=[_agent_id])]).agent_flatten()
            self.input_columns["critic__agent{}".format(_agent_id)]["agent_policy"] = Scheme([dict(name="agent_policy", select_agent_ids=[_agent_id])]).agent_flatten()
            self.input_columns["target_critic__agent{}".format(_agent_id)] = self.input_columns["critic__agent{}".format(_agent_id)]

        self.last_target_update_T_critic = 0
        pass


    def create_models(self, transition_scheme):

        self.scheme_shapes = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                     dict_of_schemes=self.schemes)

        self.input_shapes = _generate_input_shapes(input_columns=self.input_columns,
                                                   scheme_shapes=self.scheme_shapes)

        # only use one critic model as all agents have same input shapes
        # if we cannot make this assumption one day, then just create one input shape per agent
        self.critic = IACCritic(input_shapes=self.input_shapes["critic__agent{}".format(0)],
                                n_actions=self.n_actions,
                                args=self.args,
                                version="advantage")
        self.target_critic = deepcopy(self.critic)

        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False

        if self.args.use_cuda:
            self.critic = self.critic.cuda()
            self.target_critic = self.target_critic.cuda()

        self.agent_parameters = []
        for agent in self.agents:
            self.agent_parameters.extend(agent.get_parameters())
            if self.args.share_agent_params:
                break
        self.agent_optimiser = RMSprop(self.agent_parameters, lr=self.args.lr_agent)

        self.critic_parameters = []
        self.critic_parameters.extend(self.critic.parameters())
        self.critic_optimiser = RMSprop(self.critic_parameters, lr=self.args.lr_critic)


        self.joint_scheme_dict = _join_dicts(self.schemes, self.multiagent_controller.joint_scheme_dict) #this is used for joint retrieval of data from all schemes

        self.args_sanity_check() # conduct COMA sanity check on arg parameters
        pass

    def args_sanity_check(self):
        """
        :return:
        """
        pass

    def train(self, batch_history, T_env):
        # -------------------------------------------------------------------------------
        # |  We follow the algorithmic description of COMA as supplied in Algorithm 1   |
        # |  (Counterfactual Multi-Agent Policy Gradients, Foerster et al 2018)         |
        # |  Note: Instead of for-looping backwards through the sample, we just run     |
        # |  n_critic_learner_reps repetitions of the optimization procedure on the same batch |
        # -------------------------------------------------------------------------------

        #if IS_PYCHARM_DEBUG:
        #    a = batch_history.to_pd() # DEBUG

        # Update target if necessary
        if (self.T_critic - self.last_target_update_T_critic) / self.target_critic_update_interval > 1.0:
            self.update_target_nets()
            self.last_target_update_T_critic = self.T_critic
            print("updating target net!")

        # Retrieve and view all data that can be retrieved from batch_history in a single step (caching efficient)

        # create one single batch_history view suitable for all
        data_inputs, data_inputs_tformat = batch_history.view(dict_of_schemes=self.joint_scheme_dict,
                                                              to_cuda=self.args.use_cuda,
                                                              to_variable=True,
                                                              bs_ids=None,
                                                              fill_zero=True)


        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions",
                                                         agent_ids=list(range(0, self.n_agents)),
                                                         stack=True)
        actions[actions != actions] = 0.0  # mask NaNs

        # do single forward pass in critic
        iac_model_inputs, iac_model_inputs_tformat = _build_model_inputs(column_dict=self.input_columns,
                                                                         inputs=data_inputs,
                                                                         inputs_tformat=data_inputs_tformat,
                                                                         to_variable=True)

        def _optimize_critic(**kwargs):
            inputs_critic= kwargs["iac_model_inputs"]["critic"]
            inputs_target_critic=kwargs["iac_model_inputs"]["target_critic"]
            inputs_critic_tformat=kwargs["tformat"]
            inputs_target_critic_tformat = kwargs["tformat"]

            # construct target-critic targets and carry out necessary forward passes
            # same input scheme for both target critic and critic!
            output_target_critic, output_target_critic_tformat = self.target_critic.forward(inputs_target_critic,
                                                                                            tformat=iac_model_inputs_tformat)


            target_critic_td_targets, \
            target_critic_td_targets_tformat = batch_history.get_stat("td_lambda_targets",
                                                                       bs_ids=None,
                                                                       td_lambda=self.args.td_lambda,
                                                                       gamma=self.args.gamma,
                                                                       value_function_values=output_target_critic["qvalue"].detach(),
                                                                       to_variable=True,
                                                                       to_cuda=self.args.use_cuda)

            # sample!!
            if self.args.iac_critic_use_sampling:
                critic_shape = inputs_critic[list(inputs_critic.keys())[0]].shape
                sample_ids = randint(critic_shape[_bsdim(inputs_target_critic_tformat)] \
                                     * critic_shape[_tdim(inputs_target_critic_tformat)],
                                     size=self.args.iac_critic_sample_size)
                sampled_ids_tensor = th.LongTensor(sample_ids).cuda() if inputs_critic[
                    list(inputs_critic.keys())[0]].is_cuda else th.LongTensor()
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

                batch_sample_qtargets = target_critic_td_targets.view(
                    target_critic_td_targets.shape[_adim(inputs_critic_tformat)],
                    -1,
                    target_critic_td_targets.shape[_vdim(inputs_critic_tformat)])[:, sampled_ids_tensor, :]
                qtargets = batch_sample_qtargets.view(target_critic_td_targets.shape[_adim(inputs_critic_tformat)],
                                                      -1,
                                                      1,
                                                      target_critic_td_targets.shape[_vdim(inputs_critic_tformat)])
            else:
                _inputs_critic = inputs_critic
                qtargets = target_critic_td_targets

            output_critic, output_critic_tformat = self.critic.forward(_inputs_critic,
                                                                       tformat=iac_model_inputs_tformat)

            critic_loss, \
            critic_loss_tformat = IACqCriticLoss()(input=output_critic["qvalue"],
                                                   target=Variable(qtargets, requires_grad=False),
                                                   tformat=target_critic_td_targets_tformat)

            # optimize critic loss
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm(self.critic_parameters, 50)
            self.critic_optimiser.step()

            # Calculate critic statistics
            target_critic_mean = output_target_critic["qvalue"].mean().data.cpu().numpy()
            critic_mean = output_critic["qvalue"].mean().data.cpu().numpy()
            self._add_stat("critic_loss", critic_loss.data.cpu().numpy(), T_env=T_env)
            self._add_stat("critic_mean", critic_mean, T_env=T_env)
            self._add_stat("target_critic_mean", target_critic_mean, T_env=T_env)
            self._add_stat("critic_grad_norm", critic_grad_norm, T_env=T_env)

            self.T_critic += len(batch_history) * batch_history._n_t

            return output_critic

        # optimize the critic as often as necessary to get the critic loss down reliably
        for _i in range(self.n_critic_learner_reps):
            _ = _optimize_critic(iac_model_inputs=iac_model_inputs,
                                             actions=actions,
                                             tformat=iac_model_inputs_tformat)

        # get advantages
        output_critic, output_critic_tformat = self.critic.forward(iac_model_inputs["critic"],
                                                                   tformat=iac_model_inputs_tformat)
        advantages = output_critic["advantage"]

        # only train the policy once in order to stay on-policy!
        policy_loss_function = partial(IACqPolicyLoss(),
                                       actions=Variable(actions),
                                       advantages=advantages)

        hidden_states, hidden_states_tformat = self.multiagent_controller.generate_initial_hidden_states(
            len(batch_history))

        agent_controller_output, \
        agent_controller_output_tformat = self.multiagent_controller.get_outputs(data_inputs,
                                                                                 hidden_states=hidden_states,
                                                                                 loss_fn=policy_loss_function,
                                                                                 tformat=data_inputs_tformat,
                                                                                 test_mode=False)
        COMA_loss = agent_controller_output["losses"]
        COMA_loss = COMA_loss.mean()

        # carry out optimization for agents
        self.agent_optimiser.zero_grad()
        COMA_loss.backward()
        policy_grad_norm = th.nn.utils.clip_grad_norm(self.agent_parameters, 50)
        self.agent_optimiser.step()  # DEBUG

        # Calculate policy statistics
        advantage_mean = output_critic["advantage"].mean().data.cpu().numpy()
        self._add_stat("advantage_mean", advantage_mean, T_env=T_env)
        self._add_stat("policy_grad_norm", policy_grad_norm, T_env=T_env)
        self._add_stat("policy_loss", COMA_loss.data.cpu().numpy(), T_env=T_env)

        # increase episode counter (the fastest one is always)
        self.T_policy += len(batch_history) * batch_history._n_t

        pass


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

    def update_target_nets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def get_stats(self):
        if hasattr(self, "_stats"):
            return self._stats
        else:
            return []

    def _log(self):
        pass
