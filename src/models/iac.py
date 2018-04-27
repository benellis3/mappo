import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.transforms import _tdim, _vdim, _to_batch, _from_batch, _check_inputs_validity, _shift

class IACVQFunction(nn.Module):
    # modelled after https://github.com/oxwhirl/hardercomns/blob/master/code/model/StarCraftMicro.lua 5e00920

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
    # modelled after https://github.com/oxwhirl/hardercomns/blob/master/code/model/StarCraftMicro.lua 5e00920

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
        TD[:, :, :-1, :] = inputs["rewards"][:, :, :-1, :] + self.gamma * inputs["vvalues"][:, :, 1:, :]  - inputs["vvalues"][:, :, :-1, :]
        return TD, tformat


class IACAdvantage(nn.Module):
    # modelled after https://github.com/oxwhirl/hardercomns/blob/master/code/model/StarCraftMicro.lua 5e00920

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
