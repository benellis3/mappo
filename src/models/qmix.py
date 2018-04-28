from math import sqrt
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from components.transforms import _check_inputs_validity, _to_batch, _from_batch, _adim, _bsdim, _tdim, _vdim, _pick_keys
from models import REGISTRY as m_REGISTRY
from models.basic import RNN as RecursiveAgent, DQN as NonRecursiveAgent

class QMIXMixer(nn.Module):

    def __init__(self, n_agents, input_shapes, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers
        """
        super(QMIXMixer, self).__init__()
        self.args = args
        self.n_agents = n_agents # not needed in this precise context

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == set(),\
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, chosen_qvalues, states, tformat, baseline = True): # DEBUG!!
        #_check_inputs_validity(inputs, self.input_shapes, tformat)

        if states is not None:
            assert False, "state mixing in QMIX is not yet implemented"

        return chosen_qvalues.sum(dim=_vdim(tformat), keepdim=True)

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
                chosen_qvalues = th.gather(qvalues, _vdim(tformat), actions.long())

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

# class VDNNonRecursiveAgent(NonRecursiveAgent):
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
# class VDNRecursiveAgent(RecursiveAgent):
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
