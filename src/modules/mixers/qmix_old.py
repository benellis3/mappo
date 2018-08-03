from functools import partial
from math import sqrt
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from components.transforms_old import _check_inputs_validity, _to_batch, _from_batch, _adim, _bsdim, _tdim, _vdim, _pick_keys
from models import REGISTRY as m_REGISTRY
from models.basic import RNN as RecurrentAgent, DQN as NonRecurrentAgent

class QMIXMixerSimple(nn.Module):

    def __init__(self, state_size, n_agents, mixing_dim):
        super(QMIXMixerSimple, self).__init__()
        self.state_size = state_size
        self.n_agents = n_agents
        self.mixing_dim = mixing_dim

        self.w1_hypernet = nn.Linear(state_size, n_agents * mixing_dim)
        self.b1_hypernet = nn.Linear(state_size, mixing_dim)
        self.w2_hypernet = nn.Linear(state_size, mixing_dim * 1)
        self.b2_hypernet = nn.Sequential(*[nn.Linear(state_size, mixing_dim), nn.ReLU(), nn.Linear(mixing_dim, 1)])

    def forward(self, qvalues, tformat=None, states=None):

        states = states.reshape(-1, self.state_size)
        # Produce the weights and biases
        w1 = th.abs(self.w1_hypernet(states))
        b1 = self.b1_hypernet(states)
        w2 = th.abs(self.w2_hypernet(states))
        b2 = self.b2_hypernet(states)

        # Reshape the tensors involved
        w1 = w1.view(-1, self.n_agents, self.mixing_dim)
        w2 = w2.view(-1, self.mixing_dim, 1)
        b1 = b1.view(-1, 1, self.mixing_dim)
        b2 = b2.view(-1, 1, 1)

        # qvalues_transpose = qvalues.reshape(-1, 1, self.n_agents)
        qvalues_transpose = qvalues.transpose(_vdim(tformat),_adim(tformat)).reshape(-1, 1, self.n_agents)

        x = F.elu(th.bmm(qvalues_transpose, w1) + b1)

        x = th.bmm(x, w2) + b2

        return x


class HyperLinear():
    """
    Linear network layers that allows for two additional complications:
        - parameters admit to be connected via a hyper-network like structure
        - network weights are transformed according to some rule before application
    """

    def __init__(self, in_size, out_size, use_hypernetwork=True):
        super(HyperLinear, self).__init__()

        self.use_hypernetwork = use_hypernetwork

        if not self.use_hypernetwork:
            self.w = nn.Linear(in_size, out_size)
        self.b = nn.Parameter(out_size)

        # initialize layers
        stdv = 1. / sqrt(in_size)
        if not self.use_hypernetwork:
            self.w.weight.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

        pass

    def forward(self, inputs, weights, weight_mod="abs", hypernet=None, **kwargs):
        """
        we assume inputs are of shape [a*bs*t]*v
        """
        assert inputs.dim() == 2, "we require inputs to be of shape [a*bs*t]*v"

        if self.use_hypernetwork:
            assert weights is not None, "if using hyper-network, need to supply the weights!"
            w = weights
        else:
            w = self.w.weights

        weight_mod_fn = None
        if weight_mod in ["abs"]:
            weight_mod_fn = th.abs
        elif weight_mod in ["pow"]:
            exponent = kwargs.get("exponent", 2)
            weight_mod_fn = partial(th.pow, exponent=exponent)
        elif callable(weight_mod):
            weight_mod_fn = weight_mod

        if weight_mod_fn is not None:
            w = weight_mod_fn(w)

        x = th.bmm(inputs, w) + self.b
        return x


class QMIXMixer(nn.Module):

    def __init__(self, n_agents, input_shapes, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers
        """
        super(QMIXMixer, self).__init__()
        self.args = args
        self.n_agents = n_agents # not needed in this precise context

        # Set up input regions automatically if required (if sensible)
        expected_input_shapes =  {"chosen_qvalues"}
        if self.args.qmix_use_state:
            expected_input_shapes.update("states")
        assert set(input_shapes.keys()) == expected_input_shapes,\
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes = {}
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["hyper_fc1"] = {"in":self.input_shapes["chosen_qvalues"], "out":self.args.qmix_mixer_hidden_layer_size}
        self.layer_args["hyper_fc2"] = {"in":self.layer_args["hyper_fc1"]["out"], "out":1}
        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["output_layer"] = self.n_actions # will return a*bs*t*n_actions
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)


        if self.args.qmix_use_state:
            self.hyper_network_1 = nn.Linear(self.input_shapes["states"],
                                             self.layer_args["hyper_fc1"]["in"]*self.layer_args["hyper_fc1"]["out"])
            self.hyper_network_2 = nn.Linear(self.input_shapes["states"],
                                             self.layer_args["hyper_fc2"]["in"]*self.layer_args["hyper_fc2"]["out"])

            self.hyper_fc1 = HyperLinear(self.layer_args["hyper_fc1"]["in"],
                                         self.layer_args["hyper_fc1"]["out"],
                                         use_hypernetwork=True)
            self.hyper_fc2 = HyperLinear(self.layer_args["hyper_fc2"]["in"],
                                         self.layer_args["hyper_fc1"]["out"],
                                         use_hypernetwork=True)
        else:
            self.hyper_fc1 = HyperLinear(self.layer_args["hyper_fc1"]["in"],
                                         self.layer_args["hyper_fc1"]["out"],
                                         use_hypernetwork=False)
            self.hyper_fc2 = HyperLinear(self.layer_args["hyper_fc2"]["in"],
                                         self.layer_args["hyper_fc1"]["out"],
                                         use_hypernetwork=False)
        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, chosen_qvalues, states, tformat, baseline = True): # DEBUG!!
        #_check_inputs_validity(inputs, self.input_shapes, tformat)

        if self.qmix_use_state is not None:
            assert states is not None, "states cannot be None if qmix is to use state"

            # WTF?
            w1 = self.hyper_network_1(states)
            w2 = self.hyper_network_2(states)
            x = F.elu(self.hyper_fc1(chosen_qvalues, weights=w1))
            x = self.hyper_fc2(x, weights=w2)

        return x

    pass


class QMIXMixingNetwork(nn.Module):
    def __init__(self,
                 input_shapes,
                 output_shapes=None,
                 layer_args=None,
                 n_agents=None,
                 n_actions=None,
                 args=None):
        """
        "glues" together agent network(s) and QMIXMixingNetwork
        """
        assert args.share_agent_params, "global arg 'share_agent_params' has to be True for this setup!"

        super(QMIXMixingNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        # Set up input regions automatically if required (if sensible)

        expected_agent_input_shapes = {*["agent_input__agent{}".format(_agent_id) for _agent_id in range(self.n_agents)]}
        self.input_shapes = {}
        assert set(input_shapes.keys()) == expected_agent_input_shapes, \
            "set of input_shapes does not coincide with model structure!"
        if self.input_shapes is not None:
            self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["output_layer"] = self.n_actions # will return a*bs*t*n_actions
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["encoder"] = {"in":self.input_shapes["agent_input__agent0"]["main"], "out":self.args.agents_encoder_size}
        self.layer_args["gru"] = {"in":self.layer_args["encoder"]["out"], "hidden":self.args.agents_hidden_state_size}
        self.layer_args["output_layer"] = {"in":self.layer_args["gru"]["hidden"], "out":self.output_shapes["output_layer"]}
        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up network layers
        self.encoder = m_REGISTRY[self.args.agent_encoder](input_shapes=dict(main=self.layer_args["encoder"]["in"]),
                                                           output_shapes=dict(main=self.layer_args["encoder"]["out"]))
        self.gru = nn.GRUCell(self.layer_args["gru"]["in"], self.layer_args["gru"]["hidden"])
        self.output_layer = nn.Linear(self.layer_args["output_layer"]["in"], self.layer_args["output_layer"]["out"])

        # Set up mixer model
        self.mixer_model = m_REGISTRY[self.args.qmix_mixer_model](input_shapes={},
                                                                  n_agents=self.n_agents,
                                                                  args=self.args)

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass

    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        test_mode = kwargs.get("test_mode", False)
        actions = kwargs.get("actions", None)
        states = kwargs.get("states", None)

        _inputs = inputs["agent_input"]["main"]
        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _inputs.shape[t_dim]

        x_list = []
        h_list = [hidden_states]

        for t in range(t_len):

            x = _inputs[:, :, slice(t, t + 1), :].contiguous()
            x, tformat = self.encoder({"main":x}, tformat)

            x, params_x, tformat_x = _to_batch(x, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            h = self.gru(x, h)
            x = self.output_layer(h)

            h = _from_batch(h, params_h, tformat_h)
            x = _from_batch(x, params_x, tformat_x)

            h_list.append(h)
            x_list.append(x)

        qvalues = th.cat(x_list, dim=_tdim(tformat))

        if actions is not None:



            if isinstance(actions, str) and actions in ["greedy"]:
                chosen_qvalues, _ = th.max(qvalues, dim=_vdim(tformat), keepdim=True)
            else:
                # actions which are nans are basically just ignored
                action_mask = (actions != actions)
                actions[action_mask] = 0.0
                chosen_qvalues = th.gather(qvalues, _vdim(tformat), actions.long())
                chosen_qvalues[action_mask] = float("nan")

            q_tot = self.mixer_model(chosen_qvalues=chosen_qvalues,
                                     states=states,
                                     tformat=tformat)

            if loss_fn is not None:
                loss = loss_fn(q_tot, tformat=tformat)[0]

            ret = dict(qvalues=qvalues,
                       chosen_qvalues=chosen_qvalues,
                       q_tot=q_tot)

            return ret, \
                   th.cat(h_list[1:], t_dim), \
                   loss, \
                   tformat

        else:
            assert loss_fn is None, "For QMIX, loss_fn has to be None if actions are not supplied!"

            return qvalues, \
                   th.cat(h_list[1:], t_dim), \
                   loss, \
                   tformat

# class VDNNonRecurrentAgent(NonRecurrentAgent):
#
#     def forward(self, inputs, tformat, loss_fn=None, hidden_states=None, **kwargs):
#         x, params, tformat = _to_batch(inputs["main"], tformat)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = F.softmax(x, dim=1)
#
#         # add softmax exploration (if switched on)
#         if self.args.coma_exploration_mode in ["softmax"]:
#             epsilons = inputs["epsilons"].repeat(x.shape[0], 1)
#             x = epsilons/self.n_actions + x * (1-epsilons)
#         x = _from_batch(x, params, tformat)
#
#         if loss_fn is not None:
#             losses, _ = loss_fn(x, tformat=tformat)
#
#         return x, hidden_states, losses, tformat
#
# class VDNRecurrentAgent(RecurrentAgent):
#
#     def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
#         #_check_inputs_validity(inputs, self.input_shapes, tformat)
#         test_mode = kwargs["test_mode"]
#
#         _inputs = inputs["main"]
#         loss = None
#         t_dim = _tdim(tformat)
#         assert t_dim == 2, "t_dim along unsupported axis"
#         t_len = _inputs.shape[t_dim]
#
#         x_list = []
#         h_list = [hidden_states]
#
#         for t in range(t_len):
#
#             x = _inputs[:, :, slice(t, t + 1), :].contiguous()
#             x, tformat = self.encoder({"main":x}, tformat)
#
#             x, params_x, tformat_x = _to_batch(x, tformat)
#             h, params_h, tformat_h = _to_batch(h_list[-1], tformat)
#
#             h = self.gru(x, h)
#             x = self.output(h)
#             x = F.softmax(x, dim=1)
#
#             if self.args.coma_exploration_mode in ["softmax"] and not test_mode:
#                 epsilons = inputs["epsilons"].unsqueeze(_tdim(tformat))
#                 epsilons, _, _ = _to_batch(epsilons, tformat)
#                 x = epsilons / self.n_actions + x * (1 - epsilons)
#
#             h = _from_batch(h, params_h, tformat_h)
#             x = _from_batch(x, params_x, tformat_x)
#
#             h_list.append(h)
#             x_list.append(x)
#
#         if loss_fn is not None:
#             _x = th.cat(x_list, dim=_tdim(tformat))
#             loss = loss_fn(_x, tformat=tformat)[0]
#
#         return th.cat(x_list, t_dim), \
#                th.cat(h_list[1:], t_dim), \
#                loss, \
#                tformat
