from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch as th
import numpy as np

from components.transforms_old import _to_batch, _from_batch, _check_inputs_validity, _tdim, _vdim


class COMAJointQFunction(nn.Module):
    # modelled after https://github.com/oxwhirl/hardercomns/blob/master/code/model/StarCraftMicro.lua 5e00920

    def __init__(self, input_shapes, output_shapes={}, layer_args={}, n_actions=None, args=None):

        super(COMAJointQFunction, self).__init__()

        self.args = args
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["qvalues"] = self.n_actions*self.n_actions # qvals
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
        # _check_inputs_validity(inputs, self.input_shapes, tformat, allow_nonseq=True)

        main, params, tformat = _to_batch(inputs.get("main"), tformat)
        x = F.relu(self.fc1(main))
        qvalues = self.fc2(x)
        return _from_batch(qvalues, params, tformat)


class COMAJointAdvantage(nn.Module):
    # modelled after https://github.com/oxwhirl/hardercomns/blob/master/code/model/StarCraftMicro.lua 5e00920

    def __init__(self, n_actions, input_shapes, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers
        """
        super(COMAJointAdvantage, self).__init__()
        self.args = args
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"qvalues", "agent_action", "agent_policy", "avail_actions"},\
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, tformat, baseline = True): # DEBUG!!
        # _check_inputs_validity(inputs, self.input_shapes, tformat)

        qvalues, params_qv, tformat_qv = _to_batch(inputs.get("qvalues").clone(), tformat)
        agent_action, params_aa, tformat_aa = _to_batch(inputs.get("agent_action"), tformat)
        agent_policy, params_ap, tformat_ap = _to_batch(inputs.get("agent_policy"), tformat)

        if baseline:
        # Fuse to MACKREL advantage
            baseline = th.bmm(
                agent_policy.unsqueeze(1),
                qvalues.unsqueeze(2)).squeeze(2)
        else:
            baseline = 0

        _aa = agent_action.clone()
        _aa = _aa.masked_fill(_aa!=_aa, 0.0)
        Q = th.gather(qvalues, 1, _aa.long())
        Q = Q.masked_fill(agent_action!=agent_action, float("nan"))

        A = Q - baseline

        return _from_batch(A, params_qv, tformat_qv), _from_batch(Q, params_qv, tformat_qv), tformat

class COMAJointCritic(nn.Module):

    """
    Concats MACKRELQFunction and MACKRELAdvantage together to an advantage and qvalue function
    """

    def __init__(self, input_shapes, n_actions, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers but only sub-models
        """

        super(COMAJointCritic, self).__init__()
        self.args = args
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["advantage"] = 1
        self.output_shapes["qvalue"] = 1
        self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["qfunction"] = {}
        self.layer_args.update(layer_args)

        self.COMAJointQFunction = COMAJointQFunction(input_shapes={"main":self.input_shapes["qfunction"]},
                                                       output_shapes={},
                                                       layer_args={"main":self.layer_args["qfunction"]},
                                                       n_actions = self.n_actions,
                                                       args=self.args)

        self.COMAJointAdvantage = COMAJointAdvantage(input_shapes={"avail_actions":(self.input_shapes["avail_actions_id1"]*self.input_shapes["avail_actions_id2"]),
                                                         "qvalues":self.COMAJointQFunction.output_shapes["qvalues"],
                                                         "agent_action":self.input_shapes["agent_action"],
                                                         "agent_policy":self.input_shapes["joint_policy"]},
                                         output_shapes={},
                                         n_actions=self.n_actions,
                                         args=self.args)

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, tformat, baseline=True):
        #_check_inputs_validity(inputs, self.input_shapes, tformat)

        qvalues = self.COMAJointQFunction(inputs={"main":inputs["qfunction"]},
                                     tformat=tformat)

        advantage, qvalue, _ = self.COMAJointAdvantage(inputs={"avail_actions":inputs["avail_actions"],
                                                          "qvalues":qvalues,
                                                          "agent_action":inputs["agent_action"],
                                                          "agent_policy":inputs["joint_policy"]},
                                                  tformat=tformat,
                                                  baseline=baseline)
        return {"advantage": advantage, "qvalue": qvalue}, tformat


class COMAJointNonRecurrentMultiAgentNetwork(nn.Module):
    def __init__(self,
                 input_shapes,
                 output_shapes=None,
                 layer_args=None,
                 n_agents=None,
                 n_actions=None,
                 args=None):

        super(COMAJointNonRecurrentMultiAgentNetwork, self).__init__()

        self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["fc2"] = self.n_actions  # output
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in": self.input_shapes["central_agent"]["observations"]
                                        + self.input_shapes["central_agent"]["state"],
                                  "out": 64}
        self.layer_args["fc2"] = {"in": self.layer_args["fc1"]["out"], "out": self.output_shapes["fc2"]**2}
        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up network layers
        self.fc1 = nn.Linear(self.layer_args["fc1"]["in"], self.layer_args["fc1"]["out"])
        self.fc2 = nn.Linear(self.layer_args["fc2"]["in"], self.layer_args["fc2"]["out"])

    def init_hidden(self, batch_size, *args, **kwargs):
        """
        model has no hidden state, but we will pretend otherwise for consistency
        """
        vbl = Variable(th.zeros(batch_size, 1, 1))
        tformat = "bs*t*v"
        return vbl.cuda() if self.args.use_cuda else vbl, tformat

    def forward(self, inputs, tformat, loss_fn=None, hidden_states=None, **kwargs):
        test_mode = kwargs["test_mode"]

        loss = None

        agent_inputs = inputs["central_agent"]["observations"]
        state_input = inputs["central_agent"]["state"]

        avail_actions1, params_aa1, _ = _to_batch(inputs["central_agent"]["avail_actions__agent1"].unsqueeze(0), tformat)
        avail_actions2, params_aa2, _ = _to_batch(inputs["central_agent"]["avail_actions__agent2"].unsqueeze(0), tformat)
        tmp = (avail_actions1 * avail_actions2)
        pairwise_avail_actions = th.bmm(tmp.unsqueeze(2), tmp.unsqueeze(1))
        avail_actions = pairwise_avail_actions.view(pairwise_avail_actions.shape[0], -1)

        x, params, tformat = _to_batch(th.cat((agent_inputs.unsqueeze(0), state_input.unsqueeze(0)), 3), tformat)  # TODO: right cat dim?
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # mask policy elements corresponding to unavailable actions
        n_available_actions = avail_actions.detach().sum(dim=1, keepdim=True)
        x = th.exp(x)
        x = x.masked_fill(avail_actions == 0, float(np.finfo(np.float32).tiny))
        x = th.div(x, x.sum(dim=1, keepdim=True))

        # add softmax exploration (if switched on)
        if self.args.coma_exploration_mode in ["softmax"] and not test_mode:
            epsilons = inputs["central_agent"]["epsilons"].unsqueeze(0).unsqueeze(_tdim(tformat))
            epsilons, _, _ = _to_batch(epsilons, tformat)
            epsilons = epsilons.repeat(agent_inputs.shape[1], avail_actions.shape[1])
            x = avail_actions * epsilons / n_available_actions + x * (1 - epsilons)

        x = _from_batch(x, params, tformat)

        if loss_fn is not None:
            loss, _ = loss_fn(x, tformat=tformat)

        return x, \
               hidden_states, \
               loss, \
               tformat


# class COMAJointRecurrentAgent(RecurrentAgent):
#     # Simply copied from standard coma
#     def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
#         _check_inputs_validity(inputs, self.input_shapes, tformat)
#
#         test_mode = kwargs["test_mode"]
#
#         _inputs = inputs["main"]
#         _inputs_aa = inputs["avail_actions"]
#
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
#             avail_actions = _inputs_aa[:, :, slice(t, t + 1), :].contiguous()
#             x, tformat = self.encoder({"main":x}, tformat)
#
#             x, params_x, tformat_x = _to_batch(x, tformat)
#             avail_actions, params_aa, tformat_aa = _to_batch(avail_actions, tformat)
#             h, params_h, tformat_h = _to_batch(h_list[-1], tformat)
#
#             h = self.gru(x, h)
#             x = self.output(h)
#
#             # mask policy elements corresponding to unavailable actions
#             n_available_actions = avail_actions.detach().sum(dim=1, keepdim=True)
#             x = th.exp(x)
#             x = x.masked_fill(avail_actions == 0, float(np.finfo(np.float32).tiny))
#             x = th.div(x, x.sum(dim=1, keepdim=True))
#
#             # Alternative variant
#             #x = th.nn.functional.softmax(x).clone()
#             #x.masked_fill_(avail_actions.long() == 0, float(np.finfo(np.float32).tiny))
#             #x = th.div(x, x.sum(dim=1, keepdim=True))
#
#             # add softmax exploration (if switched on)
#             if self.args.coma_exploration_mode in ["softmax"] and not test_mode:
#                epsilons = inputs["epsilons"].unsqueeze(_tdim(tformat))
#                epsilons, _, _ = _to_batch(epsilons, tformat)
#                x = avail_actions * epsilons / n_available_actions + x * (1 - epsilons)
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

