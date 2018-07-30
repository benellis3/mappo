from itertools import combinations
import numpy as np
from torch.autograd import Variable
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.transforms_old import _check_inputs_validity, _to_batch, _from_batch, _adim, _bsdim, _tdim, _vdim, _check_nan
from models.basic import RNN as RecurrentAgent, DQN as NonRecurrentAgent
from utils.mackrel import _n_agent_pairings, _agent_ids_2_pairing_id, _ordered_agent_pairings, _action_pair_2_joint_actions

class CentralVFunction(nn.Module):
    # modelled after https://github.com/oxwhirl/hardercomns/blob/master/code/model/StarCraftMicro.lua 5e00920

    def __init__(self, input_shapes, n_agents, n_actions, output_shapes={}, layer_args={}, args=None):

        super(CentralVFunction, self).__init__()

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


class CentralVCritic(nn.Module):

    """
    Concats FLOUNDERLQFunction and FLOUNDERLAdvantage together to an advantage and qvalue function
    """

    def __init__(self, input_shapes, n_actions, n_agents, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers but only sub-models
        """

        super(CentralVCritic, self).__init__()
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

        self.CentralVFunction= CentralVFunction(input_shapes={"main":self.input_shapes["vfunction"]},
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


    def forward(self, inputs, tformat, baseline=True):
        #_check_inputs_validity(inputs, self.input_shapes, tformat)

        vvalue, vvalue_tformat = self.CentralVFunction(inputs={"main":inputs["vfunction"]},
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
