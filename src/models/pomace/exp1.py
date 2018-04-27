from math import sqrt
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from components.transforms import _check_inputs_validity, _to_batch, _from_batch, _adim, _bsdim, _tdim, _vdim, _pick_keys
from models import REGISTRY as m_REGISTRY


class poMACEExp1NoiseNetwork(nn.Module):  # Mainly copied from poMACENoiseNetwork
    def __init__(self, input_shapes, output_shapes=None, layer_args=None, n_agents=None, n_actions=None, args=None):

        super(poMACEExp1NoiseNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        # Set up input regions automatically if required (if sensible)
        expected_epsilon_input_shapes = {"pomace_epsilon_seeds"} if self.args.pomace_use_epsilon_seed else {"pomace_epsilons"}
        expected_epsilon_variances_input_shapes = {"pomace_epsilon_variances"}
        expected_state_input_shapes = {"state"}

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

        self.sigma_encoder = nn.Linear(
            self.n_agents + self.args.agents_hidden_state_size + self.input_shapes["state"],
            self.args.pomace_epsilon_size)
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
                    for _bs in range(epsilon_variances.shape[0]):  # could use pytorch dist
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
        state_t_size = inputs["state"].shape[_tdim(state_tformat)]
        epsilons = epsilons.unsqueeze(1).repeat(1, state_t_size, 1)
        states = inputs["state"]
        xi = inputs["xi"]
        agent_id = inputs["agent_id"]
        tformat = "bs*t*v"

        # convert to bs*v format
        epsilons_inputs, epsilons_params, epsilons_tformat = _to_batch(epsilons, "bs*t*v")
        state_inputs, state_params, state_tformat = _to_batch(states, tformat)
        agent_id_inputs, agent_id_params, agent_id_tformat = _to_batch(agent_id, "a*bs*t*v")

        temp = []
        # pass through model layers
        for a in range(self.n_agents):
            xi_inputs, xi_params, xi_tformat = _to_batch(xi[a, :, :, :].repeat(1, state_t_size, 1), "bs*t*v")
            agent_id_inputs, agent_id_params, agent_id_tformat = _to_batch(
                agent_id[a, :, :, :].repeat(1, state_t_size, 1), "bs*t*v")
            sigma = F.relu(self.sigma_encoder(th.cat((agent_id_inputs, xi_inputs, state_inputs), 1)))
            sigeps_params = (state_params[0], state_params[1], self.args.pomace_epsilon_size)
            sigeps = _from_batch(sigma * epsilons_inputs, sigeps_params, "bs*t*v")
            temp.append(sigeps.unsqueeze(0))
        noise = th.cat(temp, 0)  # TODO: Check that epsilons are correctly aligned with states!

        return noise, epsilons_tformat


class poMACEExp1Network(nn.Module):
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

        super(poMACEExp1Network, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        # Set up input regions automatically if required (if sensible)

        expected_agent_input_shapes = {
            *["agent_input__agent{}".format(_agent_id) for _agent_id in range(self.n_agents)]}
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {
            *["agent_input__agent{}".format(_agent_id) for _agent_id in range(self.n_agents)]} \
               | {"lambda_network"}, \
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
        self.layer_args["fc1"] = {"in": self.input_shapes["agent_input__agent0"]["main"],
                                  "in2": self.input_shapes["agent_input__agent0"]["secondary"],
                                  "out": self.args.agents_encoder_size}
        self.layer_args["gru"] = {"in": self.layer_args["fc1"]["out"], "hidden": self.args.agents_hidden_state_size}
        self.layer_args["fc2"] = {"in": self.layer_args["gru"]["hidden"],
                                  "in2": self.args.pomace_lambda_size,
                                  "out": self.args.pomace_lambda_size}
        self.layer_args["output_layer"] = {"in": self.layer_args["fc2"]["out"],
                                           "out": self.output_shapes["output_layer"]}

        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up network layers
        self.fc1 = pomaceExp1AgentMLPEncoder(input_shapes=dict(main=self.layer_args["fc1"]["in"],
                                                               secondary=self.layer_args["fc1"]["in2"]),
                                             output_shapes=dict(main=self.layer_args["fc1"]["out"]))
        self.gru = nn.GRUCell(self.layer_args["gru"]["in"], self.layer_args["gru"]["hidden"])

        assert self.args.pomace_exp_variant == 1 or self.args.pomace_exp_variant == 2, "No other variants implemented"
        if self.args.pomace_exp_variant == 1:
            self.fc2 = pomaceExp1AgentMLPEncoder(input_shapes=dict(main=self.layer_args["fc2"]["in"]),
                                                 output_shapes=dict(main=self.layer_args["fc2"]["out"]))
        else:
            self.fc2 = pomaceExp1AgentMLPEncoder(
                input_shapes=dict(main=self.layer_args["fc2"]["in"], secondary=self.layer_args["fc2"]["in2"]),
                output_shapes=dict(main=self.layer_args["fc2"]["out"]))

        self.output_layer = nn.Linear(self.layer_args["output_layer"]["in"], self.layer_args["output_layer"]["out"])

        # Set up LambdaNetwork
        self.noise_network = m_REGISTRY[self.args.pomace_noise_network](
            input_shapes=self.input_shapes["lambda_network"],
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
        _tau_inputs = inputs["agent_input"]["main"]
        _agent_id_inputs = inputs["agent_input"]["secondary"]
        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _tau_inputs.shape[t_dim]

        loss_x = []
        output_x = []
        h_list = [hidden_states]

        # construct noise network input
        if self.args.pomace_use_epsilon_seed:
            noise_inputs = _pick_keys(inputs["lambda_network"],
                                      ["state", "pomace_epsilon_seeds", "pomace_epsilon_variances"])
        else:
            noise_inputs = _pick_keys(inputs["lambda_network"],
                                      ["state", "pomace_epsilons", "pomace_epsilon_variances"])

        for t in range(t_len):
            tau = _tau_inputs[:, :, slice(t, t + 1), :].contiguous()
            aid = _agent_id_inputs[:, :, slice(t, t + 1), :].contiguous()


            # agent encoder layer
            x, tformat_x = self.fc1({"main": tau, "secondary": aid}, tformat)

            x, params_x, tformat_x = _to_batch(x, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            # TODO: make sure noise aligns properly!

            h = self.gru(x, h)
            h = _from_batch(h, params_h, tformat_h)

            noise_inputs["agent_id"] = aid
            noise_inputs["xi"] = h
            # get noise from noise network output: cast from bs*t*v to a*bs*t*v
            noise_x, noise_x_tformat = self.noise_network(noise_inputs, state_tformat="bs*t*v",
                                                                    tformat="bs*v", test_mode=test_mode)
            noise_x = th.sum(noise_x, dim=2).unsqueeze(2)
            four_d_noise_x = noise_x
            noise_x, params_noise_x, tformat_noise_x = _to_batch(noise_x, "a*bs*t*v")

            if self.args.pomace_exp_variant == 1:
                mu, tformat_mu = self.fc2({"main": h}, tformat_h)
                mu, params_mu, tformat_mu = _to_batch(mu, "a*bs*t*v")
                noise_mu = mu + noise_x
            else:
                noise_mu, tformat_noise_mu = self.fc2({"main": h, "secondary": four_d_noise_x}, tformat_h)
                noise_mu, params_noise_mu, tformat_noise_mu = _to_batch(noise_mu, "a*bs*t*v")

            x = self.output_layer(noise_mu)

            log_softmax = kwargs.get("log_softmax", False)
            if log_softmax:
                x = F.log_softmax(x, dim=1)
            else:
                x = F.softmax(x, dim=1)

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


class pomaceExp1AgentMLPEncoder(nn.Module):
    def __init__(self, input_shapes, output_shapes=None, layer_args=None, args=None):
        super(pomaceExp1AgentMLPEncoder, self).__init__()
        self.args = args
        self.has_in2 = "secondary" in input_shapes.keys()

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"} or set(input_shapes.keys()) == {"main", "secondary"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc"] = {"in":input_shapes["main"],
                                 "in2": input_shapes["secondary"] if self.has_in2 else 0,
                                 "out":output_shapes["main"]}
        if layer_args is not None:
            self.layer_args.update(layer_args)

        #Set up network layers
        self.fc = nn.Linear(self.layer_args["fc"]["in"] + self.layer_args["fc"]["in2"], self.layer_args["fc"]["out"])
        pass

    def forward(self, inputs, tformat):
        if "secondary" not in inputs:
            self.has_in2 = False

        main_inputs, main_params, main_tformat = _to_batch(inputs["main"], tformat)
        if self.has_in2:
            secondary_inputs, secondary_params, secondary_tformat = _to_batch(inputs["secondary"], tformat)
            x = th.cat((main_inputs, secondary_inputs), 1)
            params = (main_params[0], main_params[1], main_params[2], main_params[3] + secondary_params[3])
        else:
            x = main_inputs
            params = main_params

        x = F.relu(self.fc(x))

        return _from_batch(x, params, main_tformat), tformat
