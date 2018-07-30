from math import sqrt
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from components.transforms_old import _check_inputs_validity, _to_batch, _from_batch, _adim, _bsdim, _tdim, _vdim, _pick_keys
from models import REGISTRY as m_REGISTRY

class poMACENoiseNetwork(nn.Module):
    """
    takes as input state, pomace epsilon / pomace_epsilon_seed and (agent inputs[agent observations and (agent ids)])
    """

    def __init__(self, input_shapes, output_shapes=None, layer_args=None, n_agents=None, n_actions=None, args=None):

        super(poMACENoiseNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        # Set up input regions automatically if required (if sensible)
        expected_epsilon_input_shapes = {"pomace_epsilon_seeds"} if self.args.pomace_use_epsilon_seed else {"pomace_epsilons"}
        expected_epsilon_variances_input_shapes = {"pomace_epsilon_variances"}
        expected_state_input_shapes =   {"state"}
        self.input_shapes = {}
        assert set(input_shapes.keys()) == expected_epsilon_input_shapes \
                                         | expected_state_input_shapes \
                                         | expected_epsilon_variances_input_shapes, \
            "set of input_shapes does not coincide with model structure!"
        if input_shapes is not None:
            self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        for _agent_id in range(self.n_agents):
            self.output_shapes["lambda__agent{}".format(_agent_id)] = self.args.pomace_lambda_size
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        # self.layer_args["fc1"] = {"in":self.input_shapes["main"], "out":64}
        # self.layer_args["fc2"] = {"in":self.layer_args["fc1"]["out"], "out":self.output_shapes["qvalues"]}
        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up network layers
        # Set up state encoder (if needed) - IGNORE FOR NOW
        # if self.args.pomace.state_encoder is not None:
        #     if self.args.pomace_state_encoder in ["conv"]:
        #         self.state_encoder = nn.Conv2d(self.input_shapes["state"], self.layer_args["state_encoder"]["out"]) #TODO: PARAMS
        #     else:
        #         self.state_encoder = nn.Linear(self.input_shapes["state"], self.layer_args["state_encoder"]["out"])

        # set up lambda_fuser (fuses <encoded> state and epsilon)
        # if self.args.pomace.state_encoder is not None:
        #     assert False, "Not implemented yet!"
        # else:
        #     self.sigma_fuser = nn.Linear(self.input_shapes["state"] + self.args.pomace_epsilon_size,
        #                                   self.args.pomace_lambda_fuser_size)

        self.sigma_encoder = nn.Linear(self.input_shapes["state"], self.args.pomace_epsilon_size)

        #self.lambda_agent_fusers = {}
        #for _agent_id in range(self.args.n_agents):
        #    self.lambda_agent_fusers["agent__agent{}".format(_agent_id)] = nn.Linear(self.input_shapes["agent_input__agent{}".format(_agent_id)]\
        #                                                                             + self.args.pomace_lambda_size,
        #                                                                             self.args.pomace_lambda_agent_fuser_size)
    pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, state_tformat, tformat, test_mode=False):
        # _check_inputs_validity(inputs, self.input_shapes, tformat, allow_nonseq=True)
        epsilon_variances = inputs["pomace_epsilon_variances"]

        if not test_mode:
            # generate epsilon of necessary
            if self.args.pomace_use_epsilon_seed:
                # generate epsilon from seed
                if inputs["pomace_epsilon_seeds"].is_cuda:
                    _initial_rng_state_all = th.cuda.get_rng_state_all()
                    epsilons = th.cuda.FloatTensor(epsilon_variances.shape[_bsdim(tformat)],
                                                   self.args.pomace_epsilon_size)  # not sure if can do this directly on GPU using pytorch.dist...
                    for _bs in range(epsilon_variances.shape[0]):
                        th.cuda.manual_seed_all(int(inputs["pomace_epsilon_seeds"].data[_bs, 0]))
                        epsilons[_bs].normal_(mean=0.0, std=epsilon_variances.data[_bs,0])
                    th.cuda.set_rng_state_all(_initial_rng_state_all)
                else:
                    _initial_rng_state = th.get_rng_state()
                    epsilons = th.FloatTensor(epsilon_variances.shape[_bsdim(tformat)],
                                              self.args.pomace_epsilon_size)
                    for _bs in range(epsilon_variances.shape[0]): # could use pytorch dist
                        th.manual_seed(int(inputs["pomace_epsilon_seeds"].data[_bs, 0]))
                        epsilons[_bs].normal_(mean=0.0, std=epsilon_variances.data[_bs,0])
                    th.set_rng_state(_initial_rng_state)
                epsilons = Variable(epsilons, requires_grad=False)
            else:
                epsilons = inputs["epsilon"]*epsilon_variances
        else:
            if inputs["pomace_epsilon_seeds"].is_cuda:
                epsilons = Variable(th.cuda.FloatTensor(epsilon_variances.shape[_bsdim(tformat)],
                                              self.args.pomace_epsilon_size).zero_(), requires_grad=False)
            else:
                epsilons = Variable(th.FloatTensor(epsilon_variances.shape[_bsdim(tformat)],
                                              self.args.pomace_epsilon_size).zero_(), requires_grad=False)

        # scale from bs*1 to 1*bs*t*1
        epsilons = epsilons.unsqueeze(1).repeat(1, inputs["state"].shape[_tdim(state_tformat)], 1)
        states = inputs["state"]
        tformat = "bs*t*v"

        # convert to bs*v format
        epsilons_inputs, epsilons_params, epsilons_tformat = _to_batch(epsilons, "bs*t*v")
        state_inputs, state_params, state_tformat = _to_batch(states, tformat)

        # pass through model layers
        sigma = F.relu(self.sigma_encoder(state_inputs))
        noise = sigma * epsilons_inputs # TODO: Check that epsilons are correctly aligned with states!


        return _from_batch(noise, epsilons_params, epsilons_tformat), epsilons_tformat

class poMACEMultiagentNetwork(nn.Module):
    def __init__(self,
                 input_shapes,
                 output_shapes=None,
                 layer_args=None,
                 n_agents=None,
                 n_actions=None,
                 args=None):
        """
        "glues" together agent network(s) and poMACENoiseNetwork
        """
        assert args.share_agent_params, "global arg 'share_agent_params' has to be True for this setup!"

        super(poMACEMultiagentNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        # Set up input regions automatically if required (if sensible)

        expected_agent_input_shapes = {*["agent_input__agent{}".format(_agent_id) for _agent_id in range(self.n_agents)]}
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {*["agent_input__agent{}".format(_agent_id) for _agent_id in range(self.n_agents)]} \
                                           | {"lambda_network"},\
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
        self.encoder = pomaceAgentMLPEncoder(input_shapes=dict(main=self.layer_args["encoder"]["in"]),
                                    output_shapes=dict(main=self.layer_args["encoder"]["out"]))
        self.gru = nn.GRUCell(self.layer_args["gru"]["in"], self.layer_args["gru"]["hidden"])
        self.output_layer = nn.Linear(self.layer_args["output_layer"]["in"]+self.args.pomace_lambda_size, self.layer_args["output_layer"]["out"])

        # Set up LambdaNetwork
        self.noise_network = m_REGISTRY[self.args.pomace_noise_network](input_shapes=self.input_shapes["lambda_network"],
                                                                        n_actions=self.n_actions,
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

        #_check_inputs_validity(inputs, self.input_shapes, tformat)
        _inputs = inputs["agent_input"]["main"]
        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _inputs.shape[t_dim]

        loss_x = []
        output_x = []
        h_list = [hidden_states]

        # construct noise network input
        if self.args.pomace_use_epsilon_seed:
            noise_inputs = _pick_keys(inputs["lambda_network"], ["state", "pomace_epsilon_seeds", "pomace_epsilon_variances"])
        else:
            noise_inputs = _pick_keys(inputs["lambda_network"], ["state", "pomace_epsilons", "pomace_epsilon_variances"])

        # get noise from noise network output: cast from bs*t*v to a*bs*t*v
        noise_inputs, noise_inputs_tformat = self.noise_network(noise_inputs, state_tformat="bs*t*v", tformat="bs*v", test_mode=test_mode)

        for t in range(t_len):
            x = _inputs[:, :, slice(t, t + 1), :].contiguous()
            noise_x = noise_inputs[:, slice(t, t + 1), :].unsqueeze(0).repeat(x.shape[_adim(tformat)], 1, 1, 1).contiguous()
            noise_x, params_noise_x, tformat_noise_x = _to_batch(noise_x, tformat)

            # agent encoder layer
            x, tformat_x = self.encoder({"main":x}, tformat)

            x, params_x, tformat_x = _to_batch(x, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            # TODO: make sure noise aligns properly!

            h = self.gru(x, h)
            x = self.output_layer(th.cat([h, noise_x], dim=1)) # join noise and x along vdim

            log_softmax = kwargs.get("log_softmax", False)
            if log_softmax:
                x = F.log_softmax(x, dim=1)
            else:
                x = F.softmax(x, dim=1)

            h = _from_batch(h, params_h, tformat_h)
            x = _from_batch(x, params_x, tformat_x)
            h_list.append(h)
            loss_x.append(x)

            # we will not branch the variables if loss_fn is set - instead return only tensor values for x in that case
            output_x.append(x) if loss_fn is None else output_x.append(x.clone())

        if loss_fn is not None:
            _x = th.cat(loss_x, dim=_tdim(tformat))
            loss = loss_fn(_x, tformat=tformat)[0]

        return th.cat(output_x, t_dim), \
               th.cat(h_list[1:], t_dim), \
               loss, \
               tformat

class pomaceAgentMLPEncoder(nn.Module):
    def __init__(self, input_shapes, output_shapes=None, layer_args=None, args=None):
        super(pomaceAgentMLPEncoder, self).__init__()
        self.args = args

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        self.output_shapes["fc1"] = 64 # qvals
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in":input_shapes["main"], "out":output_shapes["main"]}
        if layer_args is not None:
            self.layer_args.update(layer_args)

        #Set up network layers
        self.fc1 = nn.Linear(self.input_shapes["main"], self.output_shapes["main"])
        pass

    def forward(self, inputs, tformat):

        x, n_seq, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        return _from_batch(x, n_seq, tformat), tformat
