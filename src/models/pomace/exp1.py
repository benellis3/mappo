from math import sqrt
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from components.transforms import _check_inputs_validity, \
    _to_batch, _from_batch, _adim, _bsdim, _tdim, _vdim, \
    _pick_keys, _unpack_random_seed
from models import REGISTRY as m_REGISTRY


class poMACEExp1NoiseNetwork(nn.Module):  # Mainly copied from poMACENoiseNetwork
    def __init__(self, input_shapes, output_shapes=None, layer_args=None, n_agents=None, n_actions=None, args=None):

        super(poMACEExp1NoiseNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        if self.args.pomace_exp_variant == 1:
            self.pomace_epsilon_size = self.n_actions
        else:
            self.pomace_epsilon_size = self.args.pomace_epsilon_size

        # Set up input regions automatically if required (if sensible)
        expected_epsilon_input_shapes = {"pomace_epsilon_seeds"} if self.args.pomace_use_epsilon_seed else {"pomace_epsilons"}
        expected_epsilon_variances_input_shapes = {"pomace_epsilon_variances"}
        expected_state_input_shapes = {"state"}
        expected_agent_ids_input_shapes = {"agent_ids__agent{}".format(_i) for _i in range(self.n_agents)}

        self.input_shapes = {}
        assert set(input_shapes.keys()) == expected_epsilon_input_shapes \
                                         | expected_state_input_shapes \
                                         | expected_epsilon_variances_input_shapes \
                                         | expected_agent_ids_input_shapes, \
            "set of input_shapes does not coincide with model structure!"
        if input_shapes is not None:
            self.input_shapes.update(input_shapes)

        self.sigma_encoder = nn.Sequential(nn.Linear(self.input_shapes["agent_ids__agent0"]
                                                     + self.args.agents_hidden_state_size
                                                     + self.input_shapes["state"],
                                                     self.args.pomace_noise_hidden_layer_size),
                                           nn.ReLU(),
                                           nn.Linear(self.args.pomace_noise_hidden_layer_size,
                                                     self.pomace_epsilon_size)
                                           )
    pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass

    def forward(self, inputs, tformats, test_mode=False):
        # _check_inputs_validity(inputs, self.input_shapes, tformat, allow_nonseq=True)
        epsilon_variances = inputs.get("pomace_epsilon_variances", None)
        epsilon_seeds = inputs.get("pomace_epsilon_seeds", None)

        if not test_mode:
            # generate epsilon of necessary
            if self.args.pomace_use_epsilon_seed:
                gen_fn = lambda out, bs: out.normal_(mean=0.0, std=sqrt(epsilon_variances[bs]))
                epsilons = _unpack_random_seed(seeds=inputs["pomace_epsilon_seeds"],
                                               output_shape=(epsilon_variances.shape[_bsdim(tformats["epsilon_variances"])],
                                                             self.pomace_epsilon_size),
                                               gen_fn=gen_fn)
            else:
                if self.args.pomace_exp_variant == 1:
                    assert self.args.pomace_epsilon_size == self.n_actions, \
                        "For pomace exp1 variant1, pomace_epsilon_size != n_actions ({})".format(self.n_actions)
                epsilons = inputs["epsilon"] * epsilon_variances
        else:
            if epsilon_seeds.is_cuda:
                epsilons = Variable(th.cuda.FloatTensor(epsilon_variances.shape[_bsdim(tformats["epsilon_variances"])],
                                              self.pomace_epsilon_size).zero_(), requires_grad=False)
            else:
                epsilons = Variable(th.FloatTensor(epsilon_variances.shape[_bsdim(tformats["epsilon_variances"])],
                                              self.pomace_epsilon_size).zero_(), requires_grad=False)

        # scale from bs*1 to 1*bs*t*1
        h = inputs["h"][:, :, 1:, :].contiguous()
        a_dim = h.shape[_adim(tformats["h"])]
        t_dim = h.shape[_tdim(tformats["h"])]

        # scale inputs to a*bs*t*v through repetition
        epsilons = epsilons.unsqueeze(0).unsqueeze(2).repeat(a_dim, 1, t_dim, 1)
        states = inputs["state"].unsqueeze(0).repeat(a_dim, 1, 1, 1)
        agent_ids = inputs["agent_ids"]

        # convert to [a*bs*t]*v format
        epsilons_inputs, epsilons_params, epsilons_tformat = _to_batch(epsilons, "a*bs*t*v")
        state_inputs, state_params, state_tformat = _to_batch(states, "a*bs*t*v")
        agent_id_inputs, agent_id_params, agent_id_tformat = _to_batch(agent_ids, "a*bs*t*v")
        h_inputs, h_params, h_tformat = _to_batch(h, "a*bs*t*v")

        # pass through model layers
        sigma = self.sigma_encoder(th.cat([agent_id_inputs, h_inputs, state_inputs], 1))
        noise = sigma * epsilons_inputs

        return _from_batch(noise, h_params, h_tformat), h_tformat


class poMACEExp1MultiagentNetwork(nn.Module):
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

        super(poMACEExp1MultiagentNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        if self.args.pomace_exp_variant == 1:
            self.lambda_size = self.n_actions
        elif self.args.pomace_exp_variant == 2:
            self.lambda_size = self.args.pomace_lambda_size

        # if self.args.pomace_exp_variant == 1:
        #     assert self.args.pomace_lambda_size == self.n_actions, \
        #         "for pomace variant1, pomace_lambda_size != n_actions ({})".format(self.n_actions)

        # Set up input regions automatically if required (if sensible)

        expected_agent_input_shapes = {
            *["agent_input__agent{}".format(_agent_id) for _agent_id in range(self.n_agents)]} | {"lambda_network"}
        self.input_shapes = {}
        assert set(input_shapes.keys()) == expected_agent_input_shapes,\
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
                                  "out": self.args.agents_encoder_size}
        self.layer_args["gru"] = {"in": self.layer_args["fc1"]["out"],
                                  "hidden": self.args.agents_hidden_state_size}
        self.layer_args["fc2"] = {"in": self.layer_args["gru"]["hidden"],
                                  "out": self.n_actions}
        if self.args.pomace_exp_variant == 2:
            self.layer_args["fc2"]["in"] += self.lambda_size
        self.layer_args["output_layer"] = {"in": self.layer_args["fc2"]["out"],
                                           "out": self.output_shapes["output_layer"]}

        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up network layers
        self.encoder = pomaceExp1AgentMLPEncoder(input_shapes=dict(main=self.layer_args["fc1"]["in"]),
                                             output_shapes=dict(main=self.layer_args["fc1"]["out"]))
        self.gru = nn.GRUCell(self.layer_args["gru"]["in"], self.layer_args["gru"]["hidden"])

        assert self.args.pomace_exp_variant == 1 or self.args.pomace_exp_variant == 2, "No other variants implemented"
        self.fc2 = nn.Linear(self.layer_args["fc2"]["in"],
                             self.n_actions)

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

        # _check_inputs_validity(inputs, self.input_shapes, tformat)
        agent_inputs = inputs["agent_input"]["main"]
        agent_ids = th.stack([inputs["lambda_network"]["agent_ids__agent{}".format(_agent_id)] for _agent_id in range(agent_inputs.shape[_adim(tformat)])])
        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = agent_inputs.shape[t_dim]

        # construct noise network input
        if self.args.pomace_use_epsilon_seed:
            noise_inputs = _pick_keys(inputs["lambda_network"],
                                      ["state", "pomace_epsilon_seeds", "pomace_epsilon_variances"])
        else:
            noise_inputs = _pick_keys(inputs["lambda_network"],
                                      ["state", "pomace_epsilons", "pomace_epsilon_variances"])

        # forward through agent encoder
        enc_agent_inputs, _ = self.encoder({"main": agent_inputs}, tformat)

        # propagate through GRU
        h_list = [hidden_states]
        for t in range(t_len):
            # propagate agent input through agent network up until after recurrent unit
            x = enc_agent_inputs[:, :, slice(t, t + 1), :].contiguous()
            x, params_x, tformat_x = _to_batch(x, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)
            h = self.gru(x, h)
            h = _from_batch(h, params_h, tformat_h)
            h_list.append(h)

        h = th.cat(h_list, dim=_tdim(tformat))

        # get noise from noise network output: cast from bs*t*v to a*bs*t*v
        noise_x, noise_x_tformat = self.noise_network(dict(**noise_inputs,
                                                           h=h,
                                                           agent_ids=agent_ids),
                                                      tformats=dict(state="bs*t*v",
                                                                    epsilon_variances="bs*v",
                                                                    h="a*bs*t*v",),
                                                      test_mode=test_mode)

        if self.args.pomace_exp_variant == 1:
            h, params_h, tformat_h = _to_batch(h[:, :, 1:, :].contiguous(), "a*bs*t*v")
            mu = F.relu(self.fc2(h))
            mu = _from_batch(mu, params_h, tformat_h)
            noise_mu = mu + noise_x
        elif self.args.pomace_exp_variant == 2:
            h, params_h, tformat_h = _to_batch(h[:, :, 1:, :].contiguous(), "a*bs*t*v")
            noise_x, params_noise_x, tformat_noise_x = _to_batch(noise_x, "a*bs*t*v")
            noise_mu = self.fc2(th.cat([h, noise_x], dim=1))
            noise_mu = _from_batch(noise_mu,
                                   params_noise_x,
                                   tformat_noise_x)
        x = F.softmax(noise_mu, dim=_vdim(tformat))

        if loss_fn is not None:
            loss = loss_fn(x, tformat=tformat)[0]

        return x, \
               h, \
               loss, \
               tformat


class pomaceExp1AgentMLPEncoder(nn.Module):
    def __init__(self, input_shapes, output_shapes=None, layer_args=None, args=None):
        super(pomaceExp1AgentMLPEncoder, self).__init__()
        self.args = args

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc"] = {"in": input_shapes["main"],
                                 "out": output_shapes["main"]}
        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up network layers
        self.fc = nn.Linear(self.layer_args["fc"]["in"], self.layer_args["fc"]["out"])
        pass

    def forward(self, inputs, tformat):

        main_inputs, main_params, main_tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc(main_inputs))
        return _from_batch(x, main_params, main_tformat), tformat
