from torch import nn

class VDNMixingNetwork(nn.Module):

    def __init__(self, input_shapes, output_shapes=None, layer_args=None, n_actions=None, args=None):
        super(VDNMixingNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"chosen_qvalues", "states"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["q_tot"] = 1  # qvals
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in": self.input_shapes["chosen_qvalues"], "out": self.args.vdn_mixer_model_hidden_layer_size}
        self.layer_args["fc2"] = {"in": self.layer_args["fc1"]["out"], "out": self.output_shapes["q_tot"]}
        if layer_args is not None:
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
        return _from_batch(qvalues, params, tformat)

class PowFC():

    def __init__(self, in_size, out_size, pow=2):

