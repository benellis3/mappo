from itertools import combinations
import numpy as np
from torch.autograd import Variable
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.transforms import _check_inputs_validity, _to_batch, _from_batch, _adim, _bsdim, _tdim, _vdim, _check_nan
from models.basic import RNN as RecurrentAgent, DQN as NonRecurrentAgent
from utils.mackrel import _n_agent_pairings, _agent_ids_2_pairing_id, _ordered_agent_pairings, _action_pair_2_joint_actions

class FLOUNDERLVFunction(nn.Module):
    # modelled after https://github.com/oxwhirl/hardercomns/blob/master/code/model/StarCraftMicro.lua 5e00920

    def __init__(self, input_shapes, n_agents, n_actions, output_shapes={}, layer_args={}, args=None):

        super(FLOUNDERLVFunction, self).__init__()

        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["vvalue"] = 1 # qvals
        self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in":self.input_shapes["main"], "out":64}
        self.layer_args["fc2"] = {"in":self.layer_args["fc1"]["out"], "out":self.output_shapes["vvalue"]}
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

        main, params, m_tformat = _to_batch(inputs.get("main"), tformat)
        x = F.relu(self.fc1(main))
        vvalue = self.fc2(x)
        return _from_batch(vvalue, params, m_tformat), m_tformat

# class FLOUNDERLQFunctionLevel1(nn.Module):
#     # modelled after https://github.com/oxwhirl/hardercomns/blob/master/code/model/StarCraftMicro.lua 5e00920
#
#     def __init__(self, input_shapes, n_agents, n_actions, output_shapes={}, layer_args={}, args=None):
#
#         super(FLOUNDERLQFunctionLevel1, self).__init__()
#
#         self.args = args
#         self.n_agents = n_agents
#         self.n_actions = n_actions
#
#         # Set up input regions automatically if required (if sensible)
#         self.input_shapes = {}
#         assert set(input_shapes.keys()) == {"main"}, \
#             "set of input_shapes does not coincide with model structure!"
#         self.input_shapes.update(input_shapes)
#
#         # Set up output_shapes automatically if required
#         self.output_shapes = {}
#         self.output_shapes["qvalues"] = self.n_actions # qvals
#         self.output_shapes.update(output_shapes)
#
#         # Set up layer_args automatically if required
#         self.layer_args = {}
#         self.layer_args["fc1"] = {"in":self.input_shapes["main"], "out":64}
#         self.layer_args["fc2"] = {"in":self.layer_args["fc1"]["out"], "out":self.output_shapes["qvalues"]}
#         self.layer_args.update(layer_args)
#
#         # Set up network layers
#         self.fc1 = nn.Linear(self.layer_args["fc1"]["in"], self.layer_args["fc1"]["out"])
#         self.fc2 = nn.Linear(self.layer_args["fc2"]["in"], self.layer_args["fc2"]["out"])
#
#         # DEBUG
#         # self.fc2.weight.data.zero_()
#         # self.fc2.bias.data.zero_()
#
#     def init_hidden(self):
#         """
#         There's no hidden state required for this model.
#         """
#         pass
#
#
#     def forward(self, inputs, tformat):
#         # _check_inputs_validity(inputs, self.input_shapes, tformat, allow_nonseq=True)
#
#         main, params, m_tformat = _to_batch(inputs.get("main"), tformat)
#         actions, params, a_tformat = _to_batch(inputs.get("actions"), tformat)
#         mask = (actions != actions)
#         actions[mask] = 0
#         x = F.relu(self.fc1(main))
#         qvalues = self.fc2(x)
#         qvalue = qvalues.gather(1, actions.long())
#         qvalue[mask] = np.nan
#         vvalue, _ = th.max(qvalues, dim=1)
#         return _from_batch(qvalues, params, m_tformat),\
#                _from_batch(qvalue, params, m_tformat),\
#                _from_batch(vvalue, params, m_tformat)

#
# class FLOUNDERLAdvantage(nn.Module):
#     # modelled after https://github.com/oxwhirl/hardercomns/blob/master/code/model/StarCraftMicro.lua 5e00920
#
#     def __init__(self, n_actions, input_shapes, output_shapes={}, layer_args={}, args=None):
#         """
#         This model contains no network layers
#         """
#         super(FLOUNDERLAdvantage, self).__init__()
#         self.args = args
#         self.n_actions = n_actions
#
#         # Set up input regions automatically if required (if sensible)
#         self.input_shapes = {}
#         assert set(input_shapes.keys()) == {"qvalues", "agent_action", "agent_policy", "avail_actions"},\
#             "set of input_shapes does not coincide with model structure!"
#         self.input_shapes.update(input_shapes)
#
#         pass
#
#     def init_hidden(self):
#         """
#         There's no hidden state required for this model.
#         """
#         pass
#
#
#     def forward(self, inputs, tformat, baseline = True): # DEBUG!!
#         # _check_inputs_validity(inputs, self.input_shapes, tformat)
#
#         qvalues, params_qv, tformat_qv = _to_batch(inputs.get("qvalues").clone(), tformat)
#         agent_action, params_aa, tformat_aa = _to_batch(inputs.get("agent_action"), tformat)
#         agent_policy, params_ap, tformat_ap = _to_batch(inputs.get("agent_policy"), tformat)
#
#         if baseline:
#         # Fuse to FLOUNDERL advantage
#             a = agent_policy.unsqueeze(1)
#             b = qvalues.unsqueeze(2)
#             baseline = th.bmm(
#                 agent_policy.unsqueeze(1),
#                 qvalues.unsqueeze(2)).squeeze(2)
#         else:
#             baseline = 0
#
#         _aa = agent_action.clone()
#         _aa = _aa.masked_fill(_aa!=_aa, 0.0)
#         Q = th.gather(qvalues, 1, _aa.long())
#         Q = Q.masked_fill(agent_action!=agent_action, float("nan"))
#
#         A = Q - baseline
#
#         return _from_batch(A, params_qv, tformat_qv), _from_batch(Q, params_qv, tformat_qv), tformat


class FLOUNDERLCritic(nn.Module):

    """
    Concats FLOUNDERLQFunction and FLOUNDERLAdvantage together to an advantage and qvalue function
    """

    def __init__(self, input_shapes, n_actions, n_agents, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers but only sub-models
        """

        super(FLOUNDERLCritic, self).__init__()
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        self.input_shapes["avail_actions"] = self.n_actions
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["advantage"] = 1
        self.output_shapes["vvalue"] = 1
        self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["vfunction"] = {}
        self.layer_args.update(layer_args)

        self.FLOUNDERLVFunction = FLOUNDERLVFunction(input_shapes={"main":self.input_shapes["vfunction"]},
                                                     output_shapes={},
                                                     layer_args={"main":self.layer_args["vfunction"]},
                                                     n_agents = self.n_agents,
                                                     n_actions = self.n_actions,
                                                     args=self.args)

        # self.FLOUNDERLAdvantage = FLOUNDERLAdvantage(input_shapes={"avail_actions":self.input_shapes["avail_actions"],
        #                                                "qvalues":self.FLOUNDERLQFunction.output_shapes["qvalues"],
        #                                                "agent_action":self.input_shapes["agent_action"],
        #                                                "agent_policy":self.input_shapes["agent_policy"]},
        #                                  output_shapes={},
        #                                  n_actions=self.n_actions,
        #                                  args=self.args)

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, actions, tformat, baseline=True):
        #_check_inputs_validity(inputs, self.input_shapes, tformat)

        vvalue, vvalue_tformat = self.FLOUNDERLVFunction(inputs={"main":inputs["vfunction"],
                                                                 "actions":actions,},
                                                         tformat=tformat)

        # advantage, qvalue, _ = self.FLOUNDERLAdvantage(inputs={"avail_actions":qvalues.clone().fill_(1.0),
        #                                                  # critic level1 has all actions available forever
        #                                                  "qvalues":qvalues,
        #                                                  "agent_action":inputs["agent_action"],
        #                                                  "agent_policy":inputs["agent_policy"]},
        #                                          tformat=tformat,
        #                                          baseline=baseline)
        return {"vvalue":vvalue}, vvalue_tformat

class MLPEncoder(nn.Module):
    def __init__(self, input_shapes, output_shapes={}, layer_args={}, args=None):
        super(MLPEncoder, self).__init__()
        self.args = args

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        self.output_shapes["fc1"] = 64 # output
        self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in":input_shapes["main"], "out":output_shapes["main"]}
        self.layer_args.update(layer_args)

        #Set up network layers
        self.fc1 = nn.Linear(self.input_shapes["main"], self.output_shapes["main"])
        pass

    def forward(self, inputs, tformat):

        x, n_seq, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        return _from_batch(x, n_seq, tformat), tformat


class FLOUNDERLNonRecurrentAgentLevel1(NonRecurrentAgent):

    def forward(self, inputs, n_agents, tformat, loss_fn=None, hidden_states=None, **kwargs):
        test_mode = kwargs["test_mode"]

        avail_actions, params_aa, tformat_aa = _to_batch(inputs["avail_actions"], tformat)
        x, params, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # mask policy elements corresponding to unavailable actions
        x = th.exp(x)
        x_sum = x.sum(dim=1, keepdim=True)
        second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny)) * x.shape[1])
        x_sum = x_sum.masked_fill(second_mask, 1.0)
        x = th.div(x, x_sum)

        # throw debug warning if second masking was necessary
        if th.sum(second_mask) > 0:
            if self.args.debug_verbose:
                print('Warning in FLOUNDERLNonRecurrentAgentLevel1.forward(): some sum during the softmax has been 0!')

        # add softmax exploration (if switched on)
        # if self.args.flounderl_exploration_mode_level1 in ["softmax"] and not test_mode:
        #     epsilons = inputs["epsilons_central_level1"].unsqueeze(_tdim(tformat)).unsqueeze(0)
        #     epsilons, _, _ = _to_batch(epsilons, tformat)
        #     x = epsilons / _n_agent_pairings(n_agents) + x * (1 - epsilons)

        x = _from_batch(x, params, tformat)

        if loss_fn is not None:
            losses, _ = loss_fn(x, tformat=tformat)

        return x, hidden_states, losses, tformat

class FLOUNDERLRecurrentAgentLevel1(nn.Module):

    def __init__(self, input_shapes, n_agents, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super(FLOUNDERLRecurrentAgentLevel1, self).__init__()

        self.args = args
        self.n_agents = n_agents
        assert output_type is not None, "you have to set an output_type!"
        # self.output_type=output_type

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        # assert set(input_shapes.keys()) == {"main"}, \
        #     "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        self.output_shapes["output"] = _n_agent_pairings(self.n_agents) # output
        if self.output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["encoder"] = {"in":self.input_shapes["main"], "out":64}
        self.layer_args["gru"] = {"in":self.layer_args["encoder"]["out"], "hidden":64}
        self.layer_args["output"] = {"in":self.layer_args["gru"]["hidden"], "out":self.output_shapes["output"]}
        self.layer_args.update(layer_args)

        # Set up network layers
        self.encoder = MLPEncoder(input_shapes=dict(main=self.layer_args["encoder"]["in"]),
                                    output_shapes=dict(main=self.layer_args["encoder"]["out"]))
        self.gru = nn.GRUCell(self.layer_args["gru"]["in"], self.layer_args["gru"]["hidden"])
        self.output = nn.Linear(self.layer_args["output"]["in"], self.layer_args["output"]["out"])

    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        _check_inputs_validity(inputs, self.input_shapes, tformat)

        test_mode = kwargs["test_mode"]
        n_agents = kwargs["n_agents"]

        _inputs = inputs["main"].unsqueeze(0) # as agent dimension is lacking
        #_inputs_aa = inputs["avail_actions"]

        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _inputs.shape[t_dim]

        x_list = []
        h_list = [hidden_states]

        for t in range(t_len):

            x = _inputs[:, :, slice(t, t + 1), :].contiguous()
            #avail_actions = _inputs_aa[:, :, slice(t, t + 1), :].contiguous()
            x, tformat = self.encoder({"main":x}, tformat)

            x, params_x, tformat_x = _to_batch(x, tformat)
            #avail_actions, params_aa, tformat_aa = _to_batch(avail_actions, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            h = self.gru(x, h)
            x = self.output(h)

            # mask policy elements corresponding to unavailable actions
            #n_available_actions = avail_actions.detach().sum(dim=1, keepdim=True)

            # DEBUG
            x = th.exp(x)
            x_sum = x.sum(dim=1, keepdim=True)
            second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny))*x.shape[1])
            x_sum = x_sum.masked_fill(second_mask, 1.0)
            x = th.div(x, x_sum)

            # throw debug warning if second masking was necessary
            if th.sum(second_mask.data) > 0:
                if self.args.debug_verbose:
                    print('Warning in FLOUNDERLRecurrentAgentLevel1.forward(): some sum during the softmax has been 0!')

            # Alternative variant
            #x = th.nn.functional.softmax(x).clone()
            #x.masked_fill_(avail_actions.long() == 0, float(np.finfo(np.float32).tiny))
            #x = th.div(x, x.sum(dim=1, keepdim=True))

            if self.args.flounderl_exploration_mode_level1 in ["softmax"] and not test_mode:
               epsilons = inputs["epsilons_central_level1"].unsqueeze(_tdim("bs*t*v")).detach()
               epsilons, _, _ = _to_batch(epsilons, "bs*t*v")
               x = epsilons / _n_agent_pairings(self.n_agents) + x * (1 - epsilons)

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

class FLOUNDERLNonRecurrentAgentLevel2(NonRecurrentAgent):

    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super().__init__(input_shapes,
        n_actions,
        output_type = output_type,
        output_shapes = dict(output=n_actions*n_actions + 2),
        layer_args = layer_args,
        args = args, ** kwargs)  # need to expand using no-op action

    def forward(self, inputs, tformat, loss_fn=None, hidden_states=None, **kwargs):
        test_mode = kwargs["test_mode"]

        avail_actions, params_aa, tformat_aa = _to_batch(inputs["avail_actions"], tformat)
        x, params, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # mask policy elements corresponding to unavailable actions
        # n_available_actions = avail_actions.detach().sum(dim=1, keepdim=True)
        # x = th.exp(x)
        # x = x.masked_fill(avail_actions == 0, float(np.finfo(np.float32).tiny))
        # x = th.div(x, x.sum(dim=1, keepdim=True))
        n_available_actions = avail_actions.sum(dim=1, keepdim=True)
        x = th.exp(x)
        x = x.masked_fill(avail_actions == 0, np.sqrt(float(np.finfo(np.float32).tiny)))
        x_sum = x.sum(dim=1, keepdim=True)
        second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny)) * avail_actions.shape[1])
        x_sum = x_sum.masked_fill(second_mask, 1.0)
        x = th.div(x, x_sum)

        # throw debug warning if second masking was necessary
        if th.sum(second_mask) > 0:
            if self.args.debug_verbose:
                print('Warning in FLOUNDERLNonRecurrentAgentLevel2.forward(): some sum during the softmax has been 0!')

        # add softmax exploration (if switched on)
        if self.args.coma_exploration_mode in ["softmax"] and not test_mode:
            epsilons = inputs["epsilons"].unsqueeze(_tdim(tformat))
            epsilons, _, _ = _to_batch(epsilons, tformat)
            x = avail_actions * epsilons / n_available_actions + x * (1 - epsilons)

        x = _from_batch(x, params, tformat)

        if loss_fn is not None:
            losses, _ = loss_fn(x, tformat=tformat)

        return x, hidden_states, losses, tformat

class FLOUNDERLRecurrentAgentLevel2(nn.Module):

    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super(FLOUNDERLRecurrentAgentLevel2, self).__init__()

        self.args = args
        self.n_actions = n_actions
        assert output_type is not None, "you have to set an output_type!"
        # self.output_type=output_type

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        # assert set(input_shapes.keys()) == {"main"}, \
        #     "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        self.output_shapes["output"] = 2 + self.n_actions*self.n_actions # includes delegation action and no-op
        if self.output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["encoder"] = {"in":self.input_shapes["main"], "out":64}
        self.layer_args["gru"] = {"in":self.layer_args["encoder"]["out"], "hidden":64}
        self.layer_args["output"] = {"in":self.layer_args["gru"]["hidden"], "out":self.output_shapes["output"]}
        self.layer_args.update(layer_args)

        # Set up network layers
        self.encoder = MLPEncoder(input_shapes=dict(main=self.layer_args["encoder"]["in"]),
                                    output_shapes=dict(main=self.layer_args["encoder"]["out"]))
        self.gru = nn.GRUCell(self.layer_args["gru"]["in"], self.layer_args["gru"]["hidden"])
        self.output = nn.Linear(self.layer_args["output"]["in"], self.layer_args["output"]["out"])

    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        _check_inputs_validity(inputs, self.input_shapes, tformat)

        test_mode = kwargs["test_mode"]
        pairwise_avail_actions = kwargs["pairwise_avail_actions"].detach()
        pairwise_avail_actions.requires_grad = False

        # ttype = th.cuda.FloatTensor if pairwise_avail_actions.is_cuda else th.FloatTensor
        # delegation_avails = Variable(ttype(pairwise_avail_actions.shape[0],
        #                                    pairwise_avail_actions.shape[1],
        #                                    pairwise_avail_actions.shape[2], 1).fill_(1.0), requires_grad=False)
        # pairwise_avail_actions = th.cat([delegation_avails, pairwise_avail_actions], dim=_vdim(tformat))

        # select pairs that were actually sampled
        #_inputs = inputs["main"].gather(0, Variable(sampled_pair_ids.long(),
        #                                            requires_grad=False).repeat(1,1,1,inputs["main"].shape[_vdim(tformat)]))
        #hidden_states = hidden_states.gather(0, Variable(sampled_pair_ids.long(),
        #                                            requires_grad=False).repeat(1,1,1,hidden_states.shape[_vdim(tformat)]))
        _inputs = inputs["main"]

        # # compute avail_actions
        # avail_actions1, params_aa1, tformat_aa1 = _to_batch(inputs["avail_actions_id1"], tformat)
        # avail_actions2, params_aa2, _ = _to_batch(inputs["avail_actions_id2"], tformat)
        # tmp = (avail_actions1 * avail_actions2)
        # pairwise_avail_actions = th.bmm(tmp.unsqueeze(2), tmp.unsqueeze(1))
        # ttype = th.cuda.FloatTensor if pairwise_avail_actions.is_cuda else th.FloatTensor
        # delegation_avails = Variable(ttype(pairwise_avail_actions.shape[0], 1).fill_(1.0), requires_grad=False)
        # other_avails = pairwise_avail_actions.view(pairwise_avail_actions.shape[0], -1)
        # pairwise_avail_actions = th.cat([delegation_avails, other_avails], dim=1)
        # pairwise_avail_actions = _from_batch(pairwise_avail_actions, params_aa2, tformat_aa1)

        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _inputs.shape[t_dim]

        x_list = []
        h_list = [hidden_states]

        for t in range(t_len):

            x = _inputs[:, :, slice(t, t + 1), :].contiguous()
            avail_actions = pairwise_avail_actions[:, :, slice(t, t + 1), :].contiguous().detach()
            x, tformat = self.encoder({"main":x}, tformat)

            x, params_x, tformat_x = _to_batch(x, tformat)
            avail_actions, params_aa, tformat_aa = _to_batch(avail_actions, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            h = self.gru(x, h)
            x = self.output(h)

            # mask policy elements corresponding to unavailable actions
            n_available_actions = avail_actions.sum(dim=1, keepdim=True)
            # if self.args.mackrel_logit_bias != 0:
            #     x = th.cat([x[:, 0:1] + self.args.mackrel_logit_bias, x[:, 1:]], dim=1)
            #     x = th.exp(x)
            #     x = x.masked_fill(avail_actions == 0, np.sqrt(float(np.finfo(np.float32).tiny)))
            #     x[:, 0] = th.div(x[:, 0].clone(), 1.0 + x[:, 0])
            #     x_sum = x[:,1:].sum(dim=1, keepdim=True)
            #     second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny))*avail_actions.shape[1]*10)
            #     x_sum = x_sum.masked_fill(second_mask, 1.0)
            #     z =  (1 - x[:, 0:1])
            #     z = z.masked_fill(z <= np.sqrt(float(np.finfo(np.float32).tiny))*avail_actions.shape[1]*10, np.sqrt(float(np.finfo(np.float32).tiny)))
            #     y = x[:, 0:1]
            #     y = y.masked_fill(y <= np.sqrt(float(np.finfo(np.float32).tiny))*avail_actions.shape[1]*10, np.sqrt(float(np.finfo(np.float32).tiny)))
            #     m = th.div(x[:,1:].clone(), x_sum)
            #     m = m.masked_fill(m <= np.sqrt(float(np.finfo(np.float32).tiny))*avail_actions.shape[1]*10, np.sqrt(float(np.finfo(np.float32).tiny)))
            #     x = th.cat([ y, z * m], dim=1).clone()
            # else:
            #DEBUG: TODO: Re-enable logit bias!!!
            x = th.exp(x)
            x = x.masked_fill(avail_actions == 0, np.sqrt(float(np.finfo(np.float32).tiny)))
            x_sum = x.sum(dim=1, keepdim=True)
            second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny))*avail_actions.shape[1])
            x_sum = x_sum.masked_fill(second_mask, 1.0)
            x = th.div(x, x_sum)


            # add softmax exploration (if switched on)
            if self.args.flounderl_exploration_mode_level2 in ["softmax"] and not test_mode:
               epsilons = inputs["epsilons_central_level2"].unsqueeze(_tdim(tformat)).detach()
               epsilons, _, _ = _to_batch(epsilons, tformat)
               n_available_actions[n_available_actions==0.0] = np.sqrt(float(np.finfo(np.float32).tiny))
               x = avail_actions.detach() * epsilons / n_available_actions + x * (1 - epsilons)
               #epsilons = inputs["epsilons_central_level2"].unsqueeze(_tdim(tformat)).detach()
               #epsilons, _, _ = _to_batch(epsilons, tformat)
               #n_available_actions[n_available_actions.data == 1] = 2.0 # mask zeros!!
               #x = th.cat([epsilons * self.args.flounderl_delegation_probability_bias,
               #            avail_actions[:, 1:] * (epsilons / (n_available_actions - 1)) * (
               #                        1 - self.args.flounderl_delegation_probability_bias)], dim=1) \
               #    + x * (1 - epsilons)
            #    if self.args.debug_mode:
            #        _check_nan(x)


            h = _from_batch(h, params_h, tformat_h)
            x = _from_batch(x, params_x, tformat_x)

            # select appropriate pairs
            #sampled_pair_ids_slice = sampled_pair_ids[:, :, slice(t, t + 1), :].contiguous()

            #x = x.gather(0, Variable(sampled_pair_ids_slice.long(), requires_grad=False).repeat(1,1,1,x.shape[_vdim(tformat)]))

            h_list.append(h)
            x_list.append(x)

        if loss_fn is not None:
            _x = th.cat(x_list, dim=_tdim(tformat))
            loss = loss_fn(_x, tformat=tformat)[0]

        return th.cat(x_list, t_dim), \
               th.cat(h_list[1:], t_dim), \
               loss, \
               tformat

class FLOUNDERLNonRecurrentAgentLevel3(NonRecurrentAgent):

    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super().__init__(input_shapes,
        n_actions,
        output_type = output_type,
        output_shapes = dict(output=n_actions + 1),
        layer_args = layer_args,
        args = args, ** kwargs)  # need to expand using no-op action


    def forward(self, inputs, tformat, loss_fn=None, hidden_states=None, **kwargs):
        test_mode = kwargs["test_mode"]

        avail_actions, params_aa, tformat_aa = _to_batch(inputs["avail_actions"], tformat)
        x, params, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # mask policy elements corresponding to unavailable actions
        n_available_actions = avail_actions.sum(dim=1, keepdim=True)
        x = th.exp(x)
        x = x.masked_fill(avail_actions == 0, np.sqrt(float(np.finfo(np.float32).tiny)))
        x_sum = x.sum(dim=1, keepdim=True)
        second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny)) * avail_actions.shape[1])
        x_sum = x_sum.masked_fill(second_mask, 1.0)
        x = th.div(x, x_sum)

        # throw debug warning if second masking was necessary
        if th.sum(second_mask.data) > 0:
            if self.args.debug_verbose:
                print('Warning in FLOUNDERLNonRecurrentAgentLevel3.forward(): some sum during the softmax has been 0!')

        # add softmax exploration (if switched on)
        if self.args.coma_exploration_mode in ["softmax"] and not test_mode:
            epsilons = inputs["epsilons"].unsqueeze(_tdim(tformat))
            epsilons, _, _ = _to_batch(epsilons, tformat)
            x = avail_actions * epsilons / n_available_actions + x * (1 - epsilons)

        x = _from_batch(x, params, tformat)

        if loss_fn is not None:
            losses, _ = loss_fn(x, tformat=tformat)

        return x, hidden_states, losses, tformat

class FLOUNDERLRecurrentAgentLevel3(RecurrentAgent):

    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super().__init__(input_shapes,
        n_actions,
        output_type = output_type,
        output_shapes = dict(output=1+n_actions), # do include no-op!
        layer_args = layer_args,
        args = args, ** kwargs)

    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        _check_inputs_validity(inputs, self.input_shapes, tformat)

        test_mode = kwargs["test_mode"]

        _inputs = inputs["main"]
        _inputs_aa = inputs["avail_actions"]

        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _inputs.shape[t_dim]

        x_list = []
        h_list = [hidden_states]

        for t in range(t_len):

            x = _inputs[:, :, slice(t, t + 1), :].contiguous()
            avail_actions = _inputs_aa[:, :, slice(t, t + 1), :].contiguous()
            x, tformat = self.encoder({"main":x}, tformat)

            x, params_x, tformat_x = _to_batch(x, tformat)
            avail_actions, params_aa, tformat_aa = _to_batch(avail_actions, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            h = self.gru(x, h)
            x = self.output(h)

            # mask policy elements corresponding to unavailable actions
            n_available_actions = avail_actions.sum(dim=1, keepdim=True)
            x = th.exp(x)
            x = x.masked_fill(avail_actions == 0, np.sqrt(float(np.finfo(np.float32).tiny)))
            x_sum = x.sum(dim=1, keepdim=True)
            second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny))*avail_actions.shape[1])
            x_sum = x_sum.masked_fill(second_mask, 1.0)
            x = th.div(x, x_sum)

            # throw debug warning if second masking was necessary
            if th.sum(second_mask.data) > 0:
                if self.args.debug_verbose:
                    print('Warning in FLOUNDERLRecurrentAgentLevel3.forward(): some sum during the softmax has been 0!')

            # add softmax exploration (if switched on)
            if self.args.flounderl_exploration_mode_level3 in ["softmax"] and not test_mode:
               epsilons = inputs["epsilons_central_level3"].unsqueeze(_tdim(tformat)).detach()
               epsilons, _, _ = _to_batch(epsilons, tformat)
               n_available_actions[n_available_actions==0.0] = np.sqrt(float(np.finfo(np.float32).tiny))
               x = avail_actions.detach() * epsilons / n_available_actions + x * (1 - epsilons) # avail_actions * epsilons / n_available_actions + x * (1 - epsilons)

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

class FLOUNDERLAgent(nn.Module):

    def __init__(self,
                 input_shapes,
                 n_agents,
                 n_actions,
                 args,
                 model_classes,
                 agent_output_type="policies",
                 **kwargs):
        super().__init__()

        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.models = {}
        # set up models level 1 - these output p(u_a | tau_a)
        self.model_level1 = model_classes["level1"](input_shapes=input_shapes["level1"],
                                                    n_agents=n_agents,
                                                    output_type=agent_output_type,
                                                    args=args)
        if self.args.use_cuda:
            self.model_level1 = self.model_level1.cuda()

        # set up models level 2 - these output p(u_a, u_b|CK_01)
        if self.args.agent_level2_share_params:
            model_level2 = model_classes["level2"](input_shapes=input_shapes["level2"],
                                                   n_actions=n_actions,
                                                   output_type=agent_output_type,
                                                   args=self.args)
            if self.args.use_cuda:
                model_level2 = model_level2.cuda()

            for _agent_id1, _agent_id2 in sorted(combinations(list(range(n_agents)), 2)):
                self.models["level2_{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = model_level2
        else:
            assert False, "TODO"

        # set up models level 3 - these output p(ps=a,b)
        self.models_level3 = {}
        if self.args.agent_level3_share_params:
            model_level3 = model_classes["level3"](input_shapes=input_shapes["level3"],
                                                   n_actions=n_actions,
                                                   output_type=agent_output_type,
                                                   args=args)
            if self.args.use_cuda:
                model_level3 = model_level3.cuda()

            for _agent_id in range(n_agents):
                self.models["level3_{}".format(_agent_id)] = model_level3
        else:
            assert False, "TODO"

        # set up models for the pair probability modules - these output p(u_a, u_b|tau_0, tau_1, CK_01)

        # add models to parameter hook
        for _k, _v in self.models.items():
            setattr(self, _k, _v)
        pass


    def forward(self, inputs, actions, hidden_states, tformat, loss_fn=None, **kwargs):
        # TODO: How do we handle the loss propagation for recurrent layers??

        # generate level 1-3 outputs
        out_level1, hidden_states_level1, losses_level1, tformat_level1 = self.model_level1(inputs=inputs["level1"]["agent_input_level1"],
                                                                                            hidden_states=hidden_states["level1"],
                                                                                            loss_fn=None, #loss_fn,
                                                                                            tformat=tformat["level1"],
                                                                                            #n_agents=self.n_agents,
                                                                                            **kwargs)

        pairwise_avail_actions = inputs["level2"]["agent_input_level2"]["avail_actions_pair"]
        ttype = th.cuda.FloatTensor if pairwise_avail_actions.is_cuda else th.FloatTensor
        delegation_avails = Variable(ttype(pairwise_avail_actions.shape[0],
                                           pairwise_avail_actions.shape[1],
                                           pairwise_avail_actions.shape[2], 1).fill_(1.0), requires_grad=False)
        pairwise_avail_actions = th.cat([delegation_avails, pairwise_avail_actions], dim=_vdim(tformat["level2"]))
        out_level2, hidden_states_level2, losses_level2, tformat_level2 = self.models["level2_{}".format(0)](inputs=inputs["level2"]["agent_input_level2"],
                                                                                                            hidden_states=hidden_states["level2"],
                                                                                                            loss_fn=None, #loss_fn,
                                                                                                            tformat=tformat["level2"],
                                                                                                            pairwise_avail_actions=pairwise_avail_actions,
                                                                                                            **kwargs)

        out_level3, hidden_states_level3, losses_level3, tformat_level3 = self.models["level3_{}".format(0)](inputs["level3"]["agent_input_level3"],
                                                                                                            hidden_states=hidden_states["level3"],
                                                                                                            loss_fn=None,
                                                                                                            tformat=tformat["level3"],
                                                                                                            **kwargs)


        # for each agent pair (a,b), calculate p_a_b = p(u_a, u_b|tau_a tau_b CK_ab) = p(u_d|CK_ab)*pi(u_a|tau_a)*pi(u_b|tau_b) + p(u_ab|CK_ab)
        # output dim of p_a_b is (n_agents choose 2) x bs x t x n_actions**2

        # Bulk NaN masking
        # out_level1[out_level1 != out_level1] = 0.0
        # out_level2[out_level2 != out_level2] = 0.0
        # out_level3[out_level3 != out_level3] = 0.0
        # actions = actions.detach()
        # actions[actions!=actions] = 0.0

        p_d = out_level2[:, :, :, 0:1]
        p_ab = out_level2[:, :, :, 1:]

        _actions = actions.clone()
        _actions[actions != actions] = 0.0
        pi_actions_selected = out_level3.gather(_vdim(tformat_level3), _actions.long()).clone()
        # pi_actions_selected[pi_actions_selected  != pi_actions_selected ] = 0.0 #float("nan")
        pi_actions_selected[actions != actions] = float("nan")
        avail_actions_level3 = inputs["level3"]["agent_input_level3"]["avail_actions"].clone().data
        avail_actions_selected = avail_actions_level3.gather(_vdim(tformat_level3), _actions.long()).clone()

        pi_a_cross_pi_b_list = []
        pi_ab_list = []
        pi_c_prod_list = []
        for _i, (_a, _b) in enumerate(_ordered_agent_pairings(self.n_agents)):
            # calculate pi_a_cross_pi_b # TODO: Set disallowed joint actions to NaN!
            x, params_x, tformat_x = _to_batch(out_level3[_a:_a+1], tformat["level3"])
            y, params_y, tformat_y  = _to_batch(out_level3[_b:_b+1], tformat["level3"])
            actions_masked = actions.clone()
            actions_masked[actions!=actions] = 0.0
            _actions_x, _actions_params, _actions_tformat = _to_batch(actions_masked[_a:_a+1], tformat["level3"])
            _actions_y, _actions_params, _actions_tformat = _to_batch(actions_masked[_b:_b+1], tformat["level3"])
            _x = x.gather(1, _actions_x.long())
            _y = y.gather(1, _actions_y.long())
            z = _x * _y
            u = _from_batch(z, params_x, tformat_x)
            pi_a_cross_pi_b_list.append(u)
            # calculate p_ab_selected
            _p_ab = p_ab[_i:_i+1]
            joint_actions = _action_pair_2_joint_actions((actions_masked[_a:_a+1], actions_masked[_b:_b+1]), self.n_actions)
            try:
                _z = _p_ab.gather(_vdim(tformat_level2), joint_actions.long())
            except Exception as e:
                a = actions_masked[_a:_a+1].max()
                b = actions_masked[_b:_b+1].max()
                c = joint_actions.max()
                d = self.n_actions
                pass
            # Set probabilities corresponding to jointly-disallowed actions to 0.0
            avail_flags = pairwise_avail_actions[_i:_i+1].gather(_vdim(tformat_level2), joint_actions.long())
            _z[avail_flags==0.0] = 0.0
            pi_ab_list.append(_z)
            # calculate pi_c_prod
            _pi_actions_selected = pi_actions_selected.clone()
            _pi_actions_selected[_pi_actions_selected!=_pi_actions_selected] = 0.0
            #_pi_actions_selected[_pi_actions_selected!=_pi_actions_selected] = 1.0 # mask shit
            _pi_actions_selected[_a:_a + 1] = 1.0
            _pi_actions_selected[_b:_b + 1] = 1.0
            _pi_actions_selected[avail_actions_selected==0.0] = 0.0 # should never happen!
            # Set probabilities corresponding to individually disallowed actions to 0.0
            _k = th.prod(_pi_actions_selected, dim=_adim(tformat_level3), keepdim=True)
            pi_c_prod_list.append(_k)

        pi_a_cross_pi_b = th.cat(pi_a_cross_pi_b_list, dim=0)
        pi_ab_selected = th.cat(pi_ab_list, dim=0)
        pi_c_prod = th.cat(pi_c_prod_list, dim=0)
        p_a_b = p_d * pi_a_cross_pi_b + pi_ab_selected

        # next, calculate p_a_b * prod(p, -a-b)
        p_prod = p_a_b * pi_c_prod

        # now, calculate p_a_b_c
        _tmp =  out_level1.transpose(_adim(tformat_level1), _vdim(tformat_level1))
        _tmp[_tmp!=_tmp] = 0.0
        p_a_b_c = (p_prod * _tmp).sum(dim=_adim(tformat_level1), keepdim=True)

        if self.args.debug_mode in ["check_probs"]:
            if not hasattr(self, "action_table"):
                self.action_table = {}
            if not hasattr(self, "actions_sampled"):
                self.actions_sampled = 0
            actions_flat = actions.view(self.n_agents, -1)
            for id in range(actions_flat.shape[1]):
                act = tuple(actions_flat[:, id].tolist())
                if act in self.action_table:
                    self.action_table[act] += 1
                else:
                    self.action_table[act] = 0
            self.actions_sampled += actions_flat.shape[0]

        if self.args.debug_mode in ["check_probs"]:
            actions_flat = actions.view(self.n_agents, -1)
            for id in range(actions_flat.shape[1]):
                print("sampled: ",
                      self.action_table[tuple(actions_flat[:, id].tolist())] / self.actions_sampled,
                      " pred: ",
                      p_a_b_c.view(-1)[id])

        # DEBUG MODE HERE!
        # _check_nan(pi_c_prod)
        # _check_nan(p_d)
        # _check_nan(pi_ab_selected)
        # _check_nan(pi_a_cross_pi_b)
        # _check_nan(p_a_b)
        # _check_nan(p_a_b_c)

        # agent_parameters = list(self.parameters())
        # pi_c_prod.sum().backward(retain_graph=True) # p_prod throws NaN
        # _check_nan(agent_parameters)
        # p_a_b.sum().backward(retain_graph=True)
        # _check_nan(agent_parameters)
        # p_prod.sum().backward(retain_graph=True)
        # _check_nan(agent_parameters)

        hidden_states = {"level1": hidden_states_level1,
                         "level2": hidden_states_level2,
                         "level3": hidden_states_level3
                        }

        loss = loss_fn(policies=p_a_b_c, tformat=tformat_level3)
        # loss = p_a_b_c.sum(), "a*bs*t*v"
        #loss[0].sum().backward(retain_graph=True)
        # loss[0].backward(retain_graph=True)

        # try:
        #     _check_nan(agent_parameters)
        # except Exception as e:
        #     for a, b in self.named_parameters():
        #         print("{}:{}".format(a, b.grad))
        #     a = 5
        #     pass

        return p_a_b_c, hidden_states, loss, tformat_level3 # note: policies can have NaNs in it!!