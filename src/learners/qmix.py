from copy import deepcopy
from functools import partial
from torch.autograd import Variable
from models import central_critic
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

# from .scheme_logger import SchemeLogger
from utils.blitzz.scheme import Scheme, SCHEME_CACHE
from utils.blitzz.transforms import _build_input, _build_inputs, _stack_by_key, _split_batch, _bsdim, _vdim, _build_model_inputs, _tdim, _adim
from utils.blitzz.debug import IS_PYCHARM_DEBUG

class IQLLoss(nn.Module):

    def __init__(self):
        super(IQLLoss, self).__init__()
    def forward(self, input, target, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        # targets may legitimately have NaNs - want to zero them out, and also zero out inputs at those positions
        nans = (target != target)
        target[nans] = 0.0
        input[nans] = 0.0

        # calculate mean-square loss
        ret = (input - target)**2

        # sum over whole sequences
        ret = ret.sum(dim=_tdim(tformat), keepdim=True)

        #sum over agents
        ret = ret.sum(dim=_adim(tformat), keepdim=True)

        # average over batches
        ret = ret.mean(dim=_bsdim(tformat), keepdim=True)

        output_tformat = "s"
        return ret, output_tformat

class IQLLearner():

    def __init__(self, multiagent_controller, args):
        self.args = args
        self.multiagent_controller = multiagent_controller
        self.agents = multiagent_controller.agents # for now, do not use any other multiagent controller functionality!!
        self.n_agents = len(self.agents)
        self.n_actions = self.multiagent_controller.n_actions
        self.T = 0
        self.target_critic_update_interval=args.target_critic_update_interval
        self.stats = {}

        self.last_target_update_T = 0
        pass


    def create_models(self, transition_scheme):

        self.agent_parameters = []
        for agent in self.agents:
            self.agent_parameters.extend(agent.get_parameters())
            if self.args.share_agent_params:
                break
        self.agent_optimiser = Adam(self.agent_parameters, lr=self.args.lr_agent)

        # calculate a grand joint scheme
        self.joint_scheme = Scheme([])
        self.joint_scheme.name = "IQL" # set name if want to add to cache
        self.joint_scheme.join([_a.scheme(_i).agent_flatten() for _i, _a in enumerate(self.agents)])

        pass

    def train(self, batch_history):
        # ------------------------------------------------------------------------------
        # |  We follow the algorithmic description of COMA as supplied in Algorithm 1  |
        # |  (Counterfactual Multi-Agent Policy Gradients, Foerster et al 2018)        |
        # ------------------------------------------------------------------------------

        if IS_PYCHARM_DEBUG:
            a = batch_history.to_pd() # DEBUG

        if (self.T - self.last_target_update_T) / self.target_critic_update_interval > 1.0:
            self.update_target_nets()
            self.last_target_update_T = self.T
            print("updating target net!")

        # create one single batch_history view suitable for all
        inputs, inputs_tformat = batch_history.view(scheme=self.joint_scheme,
                                                    to_cuda=self.args.use_cuda,
                                                    to_variable=True,
                                                    bs_ids=None,
                                                    fill_zero=True)


        observations, observations_tformat = batch_history.get_col(bs=None,
                                                         col="observations",
                                                         agent_ids=list(range(0, self.n_agents)),
                                                         stack=True)
        # TODO: SHIFT observation to t+1: _tdim(observations_tformat)

        # TODO: Handle NaNs
        # observations[observations!=observations] = 0.0 # mask NaNs

        # TODO: Get rewards
        rewards, rewards_tformat = batch_history.get_col(bs=None,
                                                         col="rewards"),

        # TODO: Calculate targets!
        targets = None

        iql_loss_function = partial(IQLLoss(),
                                       targets=Variable(targets))

        hidden_states, hidden_states_tformat = self.multiagent_controller.generate_initial_hidden_states(len(batch_history))

        agent_controller_output, \
        agent_controller_output_tformat = self.multiagent_controller.get_outputs(inputs,
                                                                                 hidden_states=hidden_states,
                                                                                 loss_fn=iql_loss_function,
                                                                                 log_softmax=False,
                                                                                 softmax=False,
                                                                                 tformat=inputs_tformat)
        IQL_loss, IQL_loss_tformat = agent_controller_output["losses"]
        IQL_loss = IQL_loss.mean()

        # carry out optimization for agents
        self.agent_optimiser.zero_grad()
        IQL_loss.backward()
        policy_grad_norm = th.nn.utils.clip_grad_norm(self.agent_parameters, 50)
        self.agent_optimiser.step() #DEBUG

        # increase episode counter
        self.T += len(batch_history) * batch_history._n_t

        # Calculate statistics
        #target_critic_mean = output_target_critic["qvalue"].mean().data.cpu().numpy()
        #critic_mean = output_critic["qvalue"].mean().data.cpu().numpy()
        #advantage_mean = output_critic["advantage"].mean().data.cpu().numpy()



        self._add_stat("critic_loss", critic_loss.data.cpu().numpy())
        self._add_stat("critic_mean", critic_mean)
        self._add_stat("advantage_mean", advantage_mean)
        self._add_stat("target_critic_mean", target_critic_mean)
        self._add_stat("critic_grad_norm", critic_grad_norm)
        self._add_stat("policy_grad_norm", policy_grad_norm)
        self._add_stat("policy_loss", COMA_loss.data.cpu().numpy())

        #a = batch_history.to_pd()
        #b = target_critic_td_targets

        # DEBUGGING SECTION
        # print(min(batch_history.seq_lens))
        # for i, p in enumerate(batch_history.seq_lens):
        #     if p < batch_history.data.shape[1]:
        #         a = batch_history.to_pd()
        #         b = target_critic_td_targets[:, i, :, :]
        #         c = 5
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
