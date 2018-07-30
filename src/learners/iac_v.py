from copy import deepcopy
from functools import partial
from numpy.random import randint
import torch as th
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop

from debug.debug import IS_PYCHARM_DEBUG
from models.iac import IACCritic
from components.scheme import Scheme
from components.transforms_old import _bsdim, _vdim, _tdim, _adim, _generate_scheme_shapes, \
    _generate_input_shapes, _join_dicts, _build_model_inputs

from .basic import BasicLearner

class IACvPolicyLoss(nn.Module):

    def __init__(self):
        super(IACvPolicyLoss, self).__init__()

    def forward(self, log_policies, td_errors, actions, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        _td = td_errors.clone().detach()
        _act = actions.clone()

        assert not (_act!=_act).any(), "_act has nan!"
        assert not (_act > log_policies.shape[_vdim(tformat)]).any(), "_act too large!: {} vs. {}".format(_act.max(),
                                                                                                          log_policies.shape)
        assert not (_act < 0).any(), "_act too small!"

        _active_logits = th.gather(log_policies, _vdim(tformat), _act.long())

        loss_mean = (-1)*(_active_logits.squeeze(_vdim(tformat)) * _td.squeeze(_vdim(tformat))).mean(dim=_bsdim(tformat))
        output_tformat = "a*t"
        return loss_mean, output_tformat

class IACvCriticLoss(nn.Module):

    def __init__(self):
        super(IACvCriticLoss, self).__init__()
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

class IACvLearner(BasicLearner):

    def __init__(self, multiagent_controller, args):
        self.args = args
        self.multiagent_controller = multiagent_controller
        self.agents = multiagent_controller.agents # for now, do not use any other multiagent controller functionality!!
        self.n_agents = len(self.agents)
        self.n_actions = self.multiagent_controller.n_actions
        self.T = 0
        self.T_critic = 0
        self.target_critic_update_interval=args.target_critic_update_interval
        self.stats = {}
        self.n_critic_learner_reps = args.n_critic_learner_reps

        # set up input schemes for all of our models
        self.critic_scheme_fn = lambda _agent_id: Scheme([dict(name="agent_id",
                                                                select_agent_ids=[_agent_id],
                                                                transforms=[("one_hot", dict(range=(0, self.args.n_agents-1)))],
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
                                                           dict(name="reward"),
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
            self.input_columns["critic__agent{}".format(_agent_id)]["rewards"] = Scheme([dict(name="reward")])
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
                                version="td")
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

    def train(self, batch_history):
        # -------------------------------------------------------------------------------
        # |  We follow the algorithmic description of IAC_v as supplied in Section 3/4  |
        # |  (Counterfactual Multi-Agent Policy Gradients, Foerster et al 2018)         |
        # |  Note: Instead of for-looping backwards through the sample, we just run     |
        # |  repetitions of the optimization procedure sampling from the same batch     |
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
                                                                       value_function_values=output_target_critic["vvalues"].detach(),
                                                                       to_variable=True,
                                                                       to_cuda=self.args.use_cuda)
            # sample!!
            if self.args.coma_critic_use_sampling:
                critic_shape = inputs_critic[list(inputs_critic.keys())[0]].shape
                sample_ids = randint(critic_shape[_bsdim(inputs_target_critic_tformat)] \
                                     * critic_shape[_tdim(inputs_target_critic_tformat)],
                                     size=self.args.coma_critic_sample_size)
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
            critic_loss_tformat = IACvCriticLoss()(input=output_critic["vvalues"],
                                                   target=Variable(qtargets, requires_grad=False),
                                                   tformat=target_critic_td_targets_tformat)

            # optimize critic loss
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm(self.critic_parameters, 50)
            self.critic_optimiser.step()

            # Calculate critic statistics
            target_critic_mean = output_target_critic["vvalues"].mean().data.cpu().numpy()
            critic_mean = output_critic["vvalues"].mean().data.cpu().numpy()
            self._add_stat("critic_loss", critic_loss.data.cpu().numpy())
            self._add_stat("critic_mean", critic_mean)
            self._add_stat("target_critic_mean", target_critic_mean)
            self._add_stat("critic_grad_norm", critic_grad_norm)

            self.T_critic += len(batch_history) * batch_history._n_t

            return output_critic

        # optimize the critic as often as necessary to get the critic loss down reliably
        output_critic = None
        for _i in range(self.n_critic_learner_reps):
            output_critic = _optimize_critic(iac_model_inputs=iac_model_inputs,
                                             actions=actions,
                                             tformat=iac_model_inputs_tformat)

        # only train the policy once in order to stay on-policy!
        policy_loss_function = partial(IACvPolicyLoss(),
                                       actions=Variable(actions),
                                       td_errors=output_critic["td_errors"])

        hidden_states, hidden_states_tformat = self.multiagent_controller.generate_initial_hidden_states(
            len(batch_history))

        agent_controller_output, \
        agent_controller_output_tformat = self.multiagent_controller.get_outputs(data_inputs,
                                                                                 hidden_states=hidden_states,
                                                                                 loss_fn=policy_loss_function,
                                                                                 log_softmax=True,
                                                                                 tformat=data_inputs_tformat)
        COMA_loss = agent_controller_output["losses"]
        COMA_loss = COMA_loss.mean()

        # carry out optimization for agents
        self.agent_optimiser.zero_grad()
        COMA_loss.backward()
        policy_grad_norm = th.nn.utils.clip_grad_norm(self.agent_parameters, 50)
        self.agent_optimiser.step()  # DEBUG

        # Calculate policy statistics
        td_mean = output_critic["td_errors"].mean().data.cpu().numpy()
        self._add_stat("advantage_mean", td_mean)
        self._add_stat("policy_grad_norm", policy_grad_norm)
        self._add_stat("policy_loss", COMA_loss.data.cpu().numpy())

        # increase episode counter (the fastest one is always)
        self.T += len(batch_history) * batch_history._n_t

        pass

    def update_target_nets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _add_stat(self, name, value):
        if not hasattr(self, "_stats"):
            self._stats = {}
        if name not in self._stats:
            self._stats[name] = []
            self._stats[name+"_T"] = []
        self._stats[name].append(value)
        self._stats[name+"_T"].append(self.T)

        if hasattr(self, "max_stats_len") and len(self._stats) > self.max_stats_len:
            self._stats[name].pop(0)
            self._stats[name+"_T"].pop(0)

        return

    def get_stats(self):
        if hasattr(self, "_stats"):
            return self._stats
        else:
            return []

    def _log(self):
        pass
