from math import sqrt
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from components.transforms import _check_inputs_validity, \
    _to_batch, _from_batch, _adim, _bsdim, _tdim, _vdim, \
    _pick_keys, _unpack_random_seed
from models import REGISTRY as m_REGISTRY
from copy import copy
from copy import deepcopy


class MCCECoordinationNetwork(nn.Module):
    def __init__(self, input_shapes, output_shapes=None, layer_args=None, n_agents=None, n_actions=None, args=None):

        super(MCCECoordinationNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        if self.args.mcce_exp_variant == 1:
            self.mcce_epsilon_size = self.n_actions
        else:
            self.mcce_epsilon_size = self.args.mcce_epsilon_size

        # Set up input regions automatically if required (if sensible)
        expected_epsilon_input_shapes = {"mcce_epsilon_seeds"} if self.args.mcce_use_epsilon_seed else {"mcce_epsilons"}
        expected_epsilon_variances_input_shapes = {"mcce_epsilon_variances"}
        expected_state_input_shapes = {"state"}
        expected_agent_ids_input_shapes = {"agent_ids__agent{}".format(_i) for _i in range(self.n_agents)}
        expected_avail_actions_input_shapes = {"avail_actions__agent{}".format(_i) for _i in range(self.n_agents)}

        self.input_shapes = {}
        # assert set(input_shapes.keys()) == expected_epsilon_input_shapes | \
        #                                    expected_epsilon_variances_input_shapes | \
        #                                    expected_state_input_shapes | \
        #                                    expected_agent_ids_input_shapes | \
        #                                    expected_avail_actions_input_shapes, \
        #         "set of input_shapes does not coincide with model structure!"
        if input_shapes is not None:
            self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["output_layer"] = self.n_actions  # will return a*bs*t*n_actions
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        self.psi_encoder = [nn.Sequential(nn.Linear(self.input_shapes["mcce_network"]["state"]
                                                    + self.input_shapes["agent_input__agent0"]["main"]
                                                    + 2,
                                                    self.args.mcce_coordination_network_hidden_layer_size),
                                          nn.ReLU(),
                                          nn.Linear(self.args.mcce_coordination_network_hidden_layer_size,
                                                    self.output_shapes["output_layer"]))
                            for _ in range(self.n_agents)]
    pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass

    def forward(self, inputs, tformats, test_mode=False):
        # _check_inputs_validity(inputs, self.input_shapes, tformat, allow_nonseq=True)
        # epsilon_variances = inputs.get("mcce_epsilon_variances", None)
        # epsilon_seeds = inputs.get("mcce_epsilon_seeds", None)

        # if not test_mode:
        #     # generate epsilon of necessary
        #     if self.args.mcce_use_epsilon_seed:
        #         gen_fn = lambda out, bs: out.normal_(mean=0.0, std=sqrt(epsilon_variances[bs]))
        #         epsilons = _unpack_random_seed(seeds=inputs["mcce_epsilon_seeds"],
        #                                        output_shape=(epsilon_variances.shape[_bsdim(tformats["epsilon_variances"])],
        #                                                      self.mcce_epsilon_size),
        #                                        gen_fn=gen_fn)
        #     else:
        #         if self.args.mcce_exp_variant == 1:
        #             assert self.args.mcce_epsilon_size == self.n_actions, \
        #                 "For mcce exp1 variant1, mcce_epsilon_size != n_actions ({})".format(self.n_actions)
        #         epsilons = inputs["epsilon"] * epsilon_variances
        # else:
        #     if epsilon_seeds.is_cuda:
        #         epsilons = Variable(th.cuda.FloatTensor(epsilon_variances.shape[_bsdim(tformats["epsilon_variances"])],
        #                                                 self.mcce_epsilon_size).zero_(), requires_grad=False)
        #     else:
        #         epsilons = Variable(th.FloatTensor(epsilon_variances.shape[_bsdim(tformats["epsilon_variances"])],
        #                                            self.mcce_epsilon_size).zero_(), requires_grad=False)

        # scale from bs*1 to 1*bs*t*1
        # a_dim = pi.shape[_adim(tformats["pi"])]
        # t_dim = 1

        # scale inputs to a*bs*t*v through repetition
        # epsilons = epsilons.unsqueeze(0).unsqueeze(2).repeat(a_dim, 1, t_dim, 1)
        # states = inputs["state"].unsqueeze(0).repeat(a_dim, 1, 1, 1)
        # actions = inputs["actions"].unsqueeze(0).repeat(a_dim, 1, 1, 1)
        # agent_ids = inputs["agent_ids"]

        # convert to [a*bs*t]*v format
        # epsilons_inputs, epsilons_params, epsilons_tformat = _to_batch(epsilons, "a*bs*t*v")
        # state_inputs, state_params, state_tformat = _to_batch(inputs["state"], "a*bs*t*v")
        # actions_inputs, actions_params, actions_tformat = _to_batch(actions, "a*bs*t*v")
        # agent_id_inputs, agent_id_params, agent_id_tformat = _to_batch(agent_ids, "a*bs*t*v")
        # h_inputs, h_params, h_tformat = _to_batch(h, "a*bs*t*v")

        # agent_inputs, agent_params, agent_tformat = _to_batch(inputs["agent_input"]["main"][:inputs["agent"], :, :, :],
        #                                                       tformats["trajectories"])

        agents = inputs["agent"]+1
        pi = inputs["pi"]

        states = inputs["state"].unsqueeze(0).repeat(inputs["agent"], 1, 1, 1)
        state_inputs, state_params, state_tformat = _to_batch(states, "a*bs*t*v")

        agent_inputs, agent_params, agent_tformat = _to_batch(inputs["agent_input"]["main"][:inputs["agent"], :, :, :],
                                                              tformats["trajectories"])
        actions = {}
        action_inputs, action_params, action_tformat = {}, {}, {}

        for j in range(agents):
            actions["agent_{}".format(j)] = th.tensor(inputs["actions"][j].unsqueeze(0).repeat(inputs["agent"], 1, 1, 1), dtype=th.float32)
            action_inputs["agent_{}".format(j)], action_params["agent_{}".format(j)], action_tformat["agent_{}".format(j)] = \
                _to_batch(actions["agent_{}".format(j)], "a*bs*t*v")

        psi_hat = {}
        psi_hat_sum = 0
        product = 1

        for j in range(agents-1):
            # pass through model layers to compute psi_hat(u^a, u^j, s)
            psi_hat["psi_hat_agent_{}".format(j)] = self.psi_encoder[j](th.cat((state_inputs,
                                                                                agent_inputs,
                                                                                action_inputs["agent_{}".format(inputs["agent"])],
                                                                                action_inputs["agent_{}".format(j)]), 1))

            psi_hat["psi_hat_output_agent_{}".format(j)] = _from_batch(psi_hat["psi_hat_agent_{}".format(j)], state_params, state_tformat)

            # compute psi_hat(u^a, u^{-a}, s) = sum_{j=1}^{a-1} psi^j(u^a, u^j, s)
            psi_hat_sum += psi_hat["psi_hat_output_agent_{}".format(j)]

            product = product * psi_hat_sum * pi[j, :, :, :]

        # psi_hat_sum = psi_hat_sum.view(self.args.batch_size, 1, self.n_actions)
        # pi = pi.view(4, 32, 5)
        # pi = pi.view(-1, 4, 5)
        # pi = pi.view(32, 5, 4)

        psi = psi_hat_sum - th.sum(product, 3).unsqueeze(3)

        return psi, "a*bs*t*v"


class MCCEDecentralizedPolicyNetwork(nn.Module):
    def __init__(self,
                 input_shapes,
                 output_shapes=None,
                 layer_args=None,
                 n_agents=None,
                 n_actions=None,
                 args=None):
        """
        computes decentralized agent policies
        """
        assert args.share_agent_params, "global arg 'share_agent_params' has to be True for this setup!"

        super(MCCEDecentralizedPolicyNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        # if self.args.mcce_exp_variant == 1:
        #     self.lambda_size = self.n_actions
        # elif self.args.mcce_exp_variant == 2:
        #     self.lambda_size = self.args.mcce_lambda_size

        # if self.args.mcce_exp_variant == 1:
        #     assert self.args.mcce_lambda_size == self.n_actions, \
        #         "for mcce variant1, mcce_lambda_size != n_actions ({})".format(self.n_actions)

        # Set up input regions automatically if required (if sensible)

        # expected_agent_input_shapes = {
        #     *["agent_input__agent{}".format(_agent_id) for _agent_id in range(self.n_agents)]} | {"mcce_network"}
        self.input_shapes = {}
        # assert set(input_shapes.keys()) == expected_agent_input_shapes, \
        #     "set of input_shapes does not coincide with model structure!"
        if self.input_shapes is not None:
            self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["output_layer"] = self.n_actions  # will return a*bs*t*n_actions
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

        self.layer_args["output_layer"] = {"in": self.layer_args["fc2"]["out"],
                                           "out": self.output_shapes["output_layer"]}

        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up network layers
        self.encoder = MCCEAgentMLPEncoder(input_shapes=dict(main=self.layer_args["fc1"]["in"]),
                                           output_shapes=dict(main=self.layer_args["fc1"]["out"]))

        self.gru = nn.GRUCell(self.layer_args["gru"]["in"], self.layer_args["gru"]["hidden"])

        self.decoder = nn.Linear(self.layer_args["fc2"]["in"], self.n_actions)

        self.output_layer = nn.Linear(self.layer_args["output_layer"]["in"], self.layer_args["output_layer"]["out"])

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass

    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        # _check_inputs_validity(inputs, self.input_shapes, tformat)
        agent_inputs = inputs["main"]
        # agent_ids = th.stack([inputs["mcce_coordination_network"]["agent_ids__agent{}".format(_agent_id)] for _agent_id in range(agent_inputs.shape[_adim(tformat)])])
        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = agent_inputs.shape[t_dim]

        # forward through agent encoder
        enc_agent_inputs, _ = self.encoder(inputs, tformat)

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

        h, params_h, tformat_h = _to_batch(h[:, :, 1:, :].contiguous(), "a*bs*t*v")
        f = F.relu(self.decoder(h))
        f = _from_batch(f, params_h, tformat_h)
        h = _from_batch(h, params_h, tformat_h)
        pi_decentralized = F.softmax(f, dim=_vdim(tformat))

        # if loss_fn is not None:
        #     loss = loss_fn(pi_decentralized, tformat=tformat)[0]

        return pi_decentralized, \
               h, \
               loss, \
               tformat


class MCCEMultiAgentNetwork(nn.Module):
    def __init__(self,
                 input_shapes,
                 output_shapes=None,
                 layer_args=None,
                 n_agents=None,
                 n_actions=None,
                 args=None):
        """
        "glues" together agent policy networks and coordination networks
        """
        assert args.share_agent_params, "global arg 'share_agent_params' has to be True for this setup!"

        super(MCCEMultiAgentNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        # Set up input regions automatically if required (if sensible)

        expected_agent_input_shapes = {
            *["agent_input__agent{}".format(_agent_id) for _agent_id in range(self.n_agents)]} | {"mcce_network"}
        self.input_shapes = {}
        assert set(input_shapes.keys()) == expected_agent_input_shapes, \
            "set of input_shapes does not coincide with model structure!"
        if self.input_shapes is not None:
            self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["output_layer"] = self.n_actions  # will return a*bs*t*n_actions
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up coordination network
        self.mcce_coordination_network = m_REGISTRY[self.args.mcce_coordination_network](
                                        input_shapes=self.input_shapes,
                                        n_actions=self.n_actions,
                                        n_agents=self.n_agents,
                                        args=self.args)

        # Set up decentralized policy network
        self.mcce_decentralized_policy_network = m_REGISTRY[self.args.mcce_decentralized_policy_network](
                                        input_shapes=self.input_shapes,
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
        mcce_decentralized_policy_network_inputs = inputs["agent_input"]

        # construct coordination network input
        if self.args.mcce_use_epsilon_seed:
            mcce_network_inputs = _pick_keys(inputs["mcce_network"],
                                             ["state", "mcce_epsilon_seeds", "mcce_epsilon_variances"])
        else:
            mcce_network_inputs = _pick_keys(inputs["mcce_network"],
                                             ["state", "mcce_epsilons", "mcce_epsilon_variances"])

        agent_ids = th.stack([inputs["mcce_network"]["agent_ids__agent{}".format(_agent_id)]
                              for _agent_id in range(mcce_decentralized_policy_network_inputs["main"].shape[_adim(tformat)])])

        action_selector = inputs["action_selector"]
        # TODO: make this prettier
        avail_actions = inputs["mcce_network"]["avail_actions__agent0"].unsqueeze(0)
        for a in range(self.n_agents-1):
            avail_actions = th.cat((avail_actions, inputs["mcce_network"]["avail_actions__agent{}".format(a+1)].unsqueeze(0)))

        loss = None

        # compute decentralized policies for all agents
        pi_decentralized, h, loss_decentralized, pi_decentralized_tformat = \
            self.mcce_decentralized_policy_network(mcce_decentralized_policy_network_inputs,
                                                   hidden_states=hidden_states, tformat=tformat)

        multiagent_controller_outputs_decentralized = {"policies": pi_decentralized, "format": pi_decentralized_tformat}
        multiagent_controller_outputs_centralized = copy(multiagent_controller_outputs_decentralized)

        # initialize centralized policies
        pi_central = copy(pi_decentralized)

        for a in range(self.n_agents):
            if a == 0:
                pass
            else:
                # if th.min(multiagent_controller_outputs_centralized["policies"]) < 0:
                print("------------------------------BREAK------------------------------")
                print(multiagent_controller_outputs_centralized["policies"])
                selected_actions, modified_inputs, selected_actions_format = \
                    action_selector.select_action(multiagent_controller_outputs_centralized,
                                                  avail_actions=avail_actions,
                                                  tformat=tformat,  # TODO: find correct tformat
                                                  test_mode=test_mode)

                # get coordination function output: cast from bs*t*v to a*bs*t*v
                psi, psi_tformat = self.mcce_coordination_network(dict(**mcce_network_inputs,
                                                                       agent_input=inputs["agent_input"],
                                                                       pi=pi_decentralized,
                                                                       agent_ids=agent_ids,
                                                                       agent=a,
                                                                       actions=selected_actions),
                                                                  tformats=dict(state="bs*t*v",
                                                                                epsilon_variances="bs*v",
                                                                                pi=pi_decentralized_tformat,
                                                                                actions=selected_actions_format,
                                                                                trajectories=tformat),
                                                                  test_mode=test_mode)

                # update agent's centralized policy using the centralized coordination function psi

                multiagent_controller_outputs_centralized["policies"][1:a+1] = copy(F.softmax(pi_central[1:a+1] + psi[:a],
                                                                                              dim=_vdim(tformat)))

                # pi_central_temp = copy(pi_central[1:(a+1)]) + copy(psi[:a])
                # pi_central_shifted = (pi_central_temp + th.max(pi_central_temp))
                # pi_central_normalized = pi_central_shifted * (1 / th.sum(pi_central_shifted, 3).unsqueeze(3))
                # multiagent_controller_outputs_centralized["policies"][1:(a+1)] = copy(pi_central_normalized)

        pi_central = copy(F.softmax(th.cat((pi_central[:1], pi_central[1:self.n_agents] + psi), 0), dim=_vdim(tformat)))

        if th.min(pi_central) < 0:
            print("-----------------------BREAK-----------------------")
            print(pi_central)

        if loss_fn is not None:
            loss = loss_fn(pi_central, tformat=tformat)[0]

        return pi_central, \
               h, \
               loss, \
               tformat


class MCCEAgentMLPEncoder(nn.Module):
    def __init__(self, input_shapes, output_shapes=None, layer_args=None, args=None):
        super(MCCEAgentMLPEncoder, self).__init__()
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
