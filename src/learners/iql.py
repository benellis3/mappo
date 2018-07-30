from copy import deepcopy
from functools import partial
import torch as th
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop

from components.scheme import Scheme
from components.transforms_old import _adim, _bsdim, _tdim, _vdim, \
    _generate_input_shapes, _generate_scheme_shapes, _build_model_inputs, \
    _join_dicts, _seq_mean, _copy_remove_keys, _make_logging_str, _underscore_to_cap
from debug.debug import IS_PYCHARM_DEBUG

from .basic import BasicLearner

class IQLLoss(nn.Module):

    def __init__(self):
        super(IQLLoss, self).__init__()
        self.mixer = None
    def forward(self, qvalues, actions, target, tformat, states):
        """
        calculate sum_i ||r_{t+1} + max_a Q^i(s_{t+1}, a) - Q^i(s_{t}, a_{t})||_2^2
        where i is agent_id

        inputs: qvalues, actions and targets
        """

        assert tformat in ["a*bs*t*v"], "invalid input format!"

        # Need to shift the target and chosen Q-Values to ensure they are properly aligned
        qvalues = qvalues[:,:,:-1,:]
        actions = actions[:,:,:-1,:]
        target = target[:,:,1:,:]

        # NaN actions are turned into 0 for convenience, they should be masked out through the target masking anyway
        action_mask = (actions!=actions)
        actions[action_mask] = 0.0

        # targets may legitimately have NaNs - want to zero them out, and also zero out inputs at those positions
        chosen_qvalues = th.gather(qvalues, _vdim(tformat), actions.long())

        # DEBUG STUFF
        # target_mask = (target != target)
        # non_nan_elements_before = (1 - target_mask).sum().type_as(th.FloatTensor())
        # tar_before = target.clone()
        # DEBUG

        # targets with a NaN are padded elements, mask them out
        # 0-out stuff before mixing
        target_mask = (target != target)
        chosen_qvalues[target_mask] = 0
        target_mask[target_mask] = 0

        # The mixer is for VDN and QMIX
        if self.mixer is not None:
            state_mask = (states != states)
            states[state_mask] = 0
            chosen_qvalues = self.mixer(chosen_qvalues, tformat=tformat, states=states[:,:-1,:])
            target = self.mixer(target, tformat=tformat, states=states[:,1:,:])

        target_mask = (target != target)
        # target[target_mask] = 0.0
        # chosen_qvalues[target_mask] = 0.0
        non_nan_elements = (1 - target_mask).sum().type_as(th.FloatTensor()).item()

        info = {}
        # td_error
        td_error = (chosen_qvalues - target.detach())

        td_error[target_mask] = 0

        mean_td_error = td_error.sum() / non_nan_elements
        info["td_error"] = mean_td_error

        # calculate mean-square loss
        total_loss = td_error**2

        # average over non-nan elements
        mean_loss = total_loss.sum() / non_nan_elements

        output_tformat = "s" # scalar
        return mean_loss, output_tformat, info

class IQLLearner(BasicLearner):

    def __init__(self, multiagent_controller, logging_struct=None, args=None):
        self.args = args
        self.multiagent_controller = multiagent_controller
        self.agents = multiagent_controller.agents # for now, do not use any other multiagent controller functionality!!
        self.n_agents = len(self.agents)
        self.n_actions = self.multiagent_controller.n_actions
        self.T_q = 0
        self.target_update_interval=args.target_update_interval
        self.stats = {}
        self.logging_struct = logging_struct
        self.last_target_update_T = 0

        self.loss_func = IQLLoss()

        self.episodes = 0

        self.args_sanity_check()

    def args_sanity_check(self):
        """
        :return:
        """
        if self.args.td_lambda != 0:
            self.logging_struct.py_logger.warning("For original IQL, td_lambda should be 0!")
        pass


    def create_models(self, transition_scheme):

        self.agent_parameters = []
        for agent in self.agents:
            self.agent_parameters.extend(agent.get_parameters())
            if self.args.share_agent_params:
                break
        self.agent_optimiser =  RMSprop(self.agent_parameters, lr=self.args.lr_q)

        # calculate a grand joint scheme
        self.joint_scheme_dict = self.multiagent_controller.joint_scheme_dict
        pass

    def train(self, batch_history, T_env=None):

        # Update target if necessary
        if (self.episodes - self.last_target_update_T) / self.target_update_interval >= 1.0:
            self.update_target_nets()
            self.last_target_update_T = self.episodes
            self.logging_struct.py_logger.info("Updating target network at T: {}".format(T_env))
        self.episodes += 1

        # create one single batch_history view suitable for all
        data_inputs, data_inputs_tformat = batch_history.view(dict_of_schemes=self.joint_scheme_dict,
                                                              to_cuda=self.args.use_cuda,
                                                              to_variable=True,
                                                              bs_ids=None,
                                                              fill_zero=True)
        # get target outputs
        hidden_states, hidden_states_tformat = self.multiagent_controller.generate_initial_hidden_states(
            len(batch_history))

        target_mac_output, \
        target_mac_output_tformat = self.multiagent_controller.get_outputs(data_inputs,
                                                                           hidden_states=hidden_states,
                                                                           loss_fn=None,
                                                                           tformat=data_inputs_tformat,
                                                                           test_mode=False,
                                                                           target_mode=True)

        avail_actions, avail_actions_tformat = batch_history.get_col(col="avail_actions", agent_ids=list(range(self.n_agents)))
        avail_actions_byte = (1 - avail_actions).type_as(th.ByteTensor())
        target_mac_output["qvalues"][avail_actions_byte] = -50000000 # TODO: Safer/better way to do this?

        target_qvalues, _ = target_mac_output["qvalues"].detach().max(dim=_vdim(target_mac_output_tformat), keepdim=True)

        # calculate targets
        rewards, rewards_tformat = batch_history.get_col(col="reward")
        terminated, term_tformat = batch_history.get_col(col="terminated")

        bootstrapping_mask = (1 - terminated).round() # To ensure it is 0 or 1, probably a better/safer way
        expanded_bs_mask = bootstrapping_mask.expand([self.n_agents, -1, -1, -1])

        td_targets = rewards + self.args.gamma * target_qvalues * expanded_bs_mask

        # td_targets, \
        # td_targets_tformat = batch_history.get_stat("td_lambda_targets",
        #                                             bs_ids=None,
        #                                             td_lambda=self.args.td_lambda,
        #                                             gamma=self.args.gamma,
        #                                             value_function_values=qvalues,
        #                                             to_variable=True,
        #                                             to_cuda=self.args.use_cuda)

        actions, actions_tformat = batch_history.get_col(col="actions",
                                                         agent_ids=list(range(self.n_agents)))

        # To make the mixing easier for VDN and QMIX
        states, states_tformat = batch_history.get_col(col="state")
        iql_loss_fn = partial(self.loss_func,
                              target=Variable(td_targets.detach(), requires_grad=False),
                              actions=Variable(actions, requires_grad=False),
                              states=Variable(states, requires_grad=False))

        hidden_states, hidden_states_tformat = self.multiagent_controller.generate_initial_hidden_states(len(batch_history))

        mac_output, \
        mac_output_tformat = self.multiagent_controller.get_outputs(data_inputs,
                                                                    hidden_states=hidden_states,
                                                                    loss_fn=None,
                                                                    tformat=data_inputs_tformat,
                                                                    test_mode=False,
                                                                    target_mode=False)

        q_values = mac_output["qvalues"]

        IQL_loss, loss_tformat, loss_info = iql_loss_fn(qvalues=q_values, tformat="a*bs*t*v")
        td_error = loss_info["td_error"]

        # IQL_loss = IQL_loss.mean()

        # carry out optimization for agents
        self.agent_optimiser.zero_grad()
        IQL_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_parameters, self.args.grad_norm_clip)
        self.agent_optimiser.step() #DEBUG

        # increase episode counter
        self.T_q += len(batch_history) * batch_history._n_t

        # Calculate statistics
        self._add_stat("q_loss", IQL_loss.data.cpu().numpy(), T_env=T_env)
        self._add_stat("td_error", td_error.data.cpu().numpy(), T_env=T_env)
        self._add_stat("grad_norm", grad_norm, T_env=T_env)
        self._add_stat("target_q_mean", target_qvalues.data.cpu().numpy().mean(), T_env=T_env)
        self._add_stat("q_mean", q_values.data.cpu().numpy().mean(), T_env=T_env)
        self._add_stat("T_learner", T_env, T_env=T_env) # For convenience

        pass

    def update_target_nets(self):
        self.multiagent_controller.update_target()
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
        if stats == []:
            # Stats is empty, don't have anything to log
            return
        logging_dict =  dict(q_loss = _seq_mean(stats["q_loss"]),
                             target_q_mean = _seq_mean(stats["target_q_mean"]))

        logging_str = _make_logging_str(logging_dict)

        if log_directly:
            self.logging_struct.py_logger.info("{} LEARNER INFO: {}".format(self.args.learner.upper(), logging_str))

        return logging_str, logging_dict

    def _log(self):
        pass
