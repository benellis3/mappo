import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.transforms_old import _tdim, _vdim, _to_batch, _from_batch, _check_inputs_validity, _shift
from models.basic import RNN as RecurrentAgent, DQN as NonRecurrentAgent

class IACVQFunction(nn.Module):

    def __init__(self, input_shapes, output_shapes={}, layer_args={}, n_actions=None, args=None):

        super(IACVQFunction, self).__init__()

        self.args = args
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["qvalues"] = self.n_actions # qvals
        self.output_shapes["vvalues"] = 1
        self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in":self.input_shapes["main"], "out":64}
        self.layer_args["fc2"] = {"in":self.layer_args["fc1"]["out"], "out":self.output_shapes["qvalues"]}
        self.layer_args.update(layer_args)

        # Set up network layers
        self.fc1 = nn.Linear(self.layer_args["fc1"]["in"], self.layer_args["fc1"]["out"])
        self.fc2 = nn.Linear(self.layer_args["fc2"]["in"], self.layer_args["fc2"]["out"])

        # DEBUG
        # self.fc2.weight.data.zero_()
        # self.fc2.bias.data.zero_()

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, tformat):
        _check_inputs_validity(inputs, self.input_shapes, tformat, allow_nonseq=True)

        main, params, tformat = _to_batch(inputs.get("main"), tformat)
        x = F.relu(self.fc1(main))
        qvalues = self.fc2(x)
        vvalues, _ = qvalues.max(dim=1, keepdim=True)

        return _from_batch(qvalues, params, tformat), _from_batch(vvalues, params, tformat), tformat

class IACTDError(nn.Module):

    def __init__(self, n_actions, input_shapes, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers
        """
        super(IACTDError, self).__init__()
        self.args = args
        self.n_actions = n_actions
        self.gamma = args.gamma

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"vvalues", "rewards"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, tformat):
        _check_inputs_validity(inputs, self.input_shapes, tformat)
        assert tformat in ["a*bs*t*v"], "invalid input format!"
        TD = inputs["rewards"].clone().zero_()
        TD[:, :, :-1, :] = inputs["rewards"][:, :, 1:, :] + self.gamma * inputs["vvalues"][:, :, 1:, :]  - inputs["vvalues"][:, :, :-1, :]
        return TD, tformat


class IACAdvantage(nn.Module):

    def __init__(self, n_actions, input_shapes, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers
        """
        super(IACAdvantage, self).__init__()
        self.args = args
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"qvalues", "agent_action", "agent_policy"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, tformat):
        _check_inputs_validity(inputs, self.input_shapes, tformat)

        qvalues, params_qv, tformat_qv = _to_batch(inputs.get("qvalues"), tformat)
        agent_action, params_aa, tformat_aa = _to_batch(inputs.get("agent_action"), tformat)
        agent_policy, params_ap, tformat_ap = _to_batch(inputs.get("agent_policy"), tformat)

        baseline = th.bmm(agent_policy.unsqueeze(1),
                          qvalues.unsqueeze(2)).squeeze(2)

        Q = th.gather(qvalues, 1, agent_action.long())
        A = Q - baseline
        return _from_batch(A, params_qv, tformat_qv), _from_batch(Q, params_qv, tformat_qv), tformat


class IACCritic(nn.Module):
    """
    Concats COMAQFunction and COMAAdvantage together to an advantage and qvalue function
    """

    def __init__(self, input_shapes, n_actions, output_shapes={}, layer_args={}, version="TD", args=None):
        """
        This model contains no network layers but only sub-models
        """

        super(IACCritic, self).__init__()
        self.args = args
        self.n_actions = n_actions
        self.version = version

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        if self.version in ["td"]:
            self.input_shapes["rewards"] = 1
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        if self.version in ["advantage"]:
            self.output_shapes["advantage"] = 1
            self.output_shapes["qvalue"] = 1
        if self.version in ["td"]:
            self.output_shapes["td"] = 1
            self.output_shapes["vvalues"] = 1
        self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["qfunction"] = {}
        self.layer_args.update(layer_args)

        self.IACVQFunction = IACVQFunction(input_shapes={"main":self.input_shapes["qfunction"]},
                                           output_shapes={},
                                           layer_args={"main":self.layer_args["qfunction"]},
                                           n_actions = self.n_actions,
                                           args=self.args)

        if self.version in ["advantage"]:
            self.IACAdvantage = IACAdvantage(input_shapes={"qvalues":self.IACVQFunction.output_shapes["qvalues"],
                                                           "agent_action":self.input_shapes["agent_action"],
                                                           "agent_policy":self.input_shapes["agent_policy"]},
                                              output_shapes={},
                                              n_actions=self.n_actions,
                                              args=self.args)
        if self.version in ["td"]:
            self.IACTDError = IACTDError(input_shapes={"vvalues":self.IACVQFunction.output_shapes["vvalues"],
                                                       "rewards":self.input_shapes["rewards"]},
                                         output_shapes={},
                                         n_actions=self.n_actions,
                                         args=self.args)
        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, tformat):
        _check_inputs_validity(inputs, self.input_shapes, tformat)
        ret_dict = {}

        qvalues, vvalues, qvvalues_tformat = self.IACVQFunction(inputs={"main":inputs["qfunction"]},
                                                      tformat=tformat)

        if self.version in ["advantage"]:
            advantage, qvalue, _ = self.IACAdvantage(inputs={"qvalues":qvalues,
                                                             "agent_action":inputs["agent_action"],
                                                             "agent_policy":inputs["agent_policy"]},
                                                      tformat=tformat)
            ret_dict["advantage"] = advantage
            ret_dict["qvalue"] =  qvalue

        if self.version in ["td"]:
            td_errors, _  = self.IACTDError(inputs={"vvalues":vvalues,
                                                    "rewards":inputs["rewards"]},
                                            tformat=tformat)
            ret_dict["vvalues"] = vvalues
            ret_dict["td_errors"] = td_errors

        return ret_dict, tformat

class IACNonRecurrentAgent(NonRecurrentAgent):

    def forward(self, inputs, tformat, loss_fn=None, hidden_states=None, **kwargs):
        x, params, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        # add softmax exploration (if switched on)
        if self.args.coma_exploration_mode in ["softmax"]:
            epsilons = inputs["epsilons"].repeat(x.shape[0], 1)
            x = epsilons/self.n_actions + x * (1-epsilons)
        x = _from_batch(x, params, tformat)

        if loss_fn is not None:
            losses, _ = loss_fn(x, tformat=tformat)

        return x, hidden_states, losses, tformat

class IACRecurrentAgent(RecurrentAgent):

    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        #_check_inputs_validity(inputs, self.input_shapes, tformat)
        test_mode = kwargs["test_mode"]

        _inputs = inputs["main"]
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
            x = self.output(h)
            x = F.softmax(x, dim=1)

            if self.args.iac_exploration_mode in ["softmax"] and not test_mode:
                epsilons = inputs["epsilons"].unsqueeze(_tdim(tformat))
                epsilons, _, _ = _to_batch(epsilons, tformat)
                x = epsilons / self.n_actions + x * (1 - epsilons)

            h = _from_batch(h, params_h, tformat_h)
            x = _from_batch(x, params_x, tformat_x)

            h_list.append(h)
            x_list.append(x)

        if loss_fn is not None:
            _x = th.cat(x_list, dim=_tdim(tformat))
            loss = loss_fn(_x, tformat=tformat)[0]

        return th.cat(x_list, t_dim), \
               th.cat(h_list[1:], t_dim), \
               loss, \
               tformat