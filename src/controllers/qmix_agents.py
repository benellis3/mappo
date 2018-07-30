from copy import deepcopy
import numpy as np
from torch.autograd import Variable
import torch as th

from components.action_selectors import REGISTRY as as_REGISTRY
from components import REGISTRY as co_REGISTRY
from components.scheme import Scheme
from components.episode_buffer_old import BatchEpisodeBuffer
from components.transforms_old import _build_model_inputs, _join_dicts, _generate_scheme_shapes, _generate_input_shapes
from models import REGISTRY as m_REGISTRY


class QMIXMultiagentController():
    """
    container object for a set of independent agents
    TODO: may need to propagate test_mode in here as well!
    """

    def __init__(self, runner, n_agents, n_actions, action_selector=None, args=None, **kwargs):
        self.args = args
        self.runner = runner
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_str = args.agent
        assert self.args.agent_output_type in ["qvalues"], "agent_output_type has to be set to 'qvalues' for QMIX - makes no sense with other methods!"
        self.agent_output_type = "qvalues"
        self.has_target_network = kwargs.get("has_target_network", True)

        # Set up action selector
        if action_selector is None:
            self.action_selector = as_REGISTRY[args.action_selector](args=self.args)
        else:
            self.action_selector = action_selector

        self.mixer_scheme = Scheme([dict(name="state",
                                         switch=self.args.qmix_use_state)])

        self.agent_scheme_fn = lambda _agent_id: Scheme([dict(name="agent_id",
                                                              transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                              select_agent_ids=[_agent_id],),
                                                         dict(name="observations",
                                                              rename="agent_observation",
                                                              select_agent_ids=[_agent_id]),
                                                         dict(name="actions",
                                                              rename="past_action",
                                                              select_agent_ids=[_agent_id],
                                                              transforms=[("shift", dict(steps=1)),
                                                                          ("one_hot", dict(range=(0, self.n_actions-1)))],
                                                              switch=self.args.obs_last_action),
                                                         dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id])
                                                        ])


        # Set up schemes
        self.schemes = {}
        for _agent_id in range(self.n_agents):
            self.schemes["agent_input__agent{}".format(_agent_id)] = self.agent_scheme_fn(_agent_id).agent_flatten()
        self.schemes["mixer"] = self.mixer_scheme

        # create joint scheme from the agents schemes and mixer_scheme
        self.joint_scheme_dict = _join_dicts(self.schemes)

        # construct model-specific input regions
        self.input_columns = {}
        for _agent_id in range(self.n_agents):
            self.input_columns["agent_input__agent{}".format(_agent_id)] = {}
            self.input_columns["agent_input__agent{}".format(_agent_id)]["main"] = \
                Scheme([dict(name="agent_id", select_agent_ids=[_agent_id]),
                        dict(name="agent_observation", select_agent_ids=[_agent_id]),
                        dict(name="past_action",
                             select_agent_ids=[_agent_id],
                             switch=self.args.obs_last_action)]).agent_flatten()
            self.input_columns["agent_input__agent{}".format(_agent_id)]["secondary"] = \
                Scheme([dict(name="agent_id", select_agent_ids=[_agent_id])]).agent_flatten()

        pass

    def get_parameters(self):
        parameters = self.mixing_network.parameters()
        return parameters

    def select_actions(self, inputs, avail_actions, tformat, info, test_mode=False):
        selected_actions, modified_inputs, selected_actions_format = \
            self.action_selector.select_action(inputs,
                                               avail_actions=avail_actions,
                                               tformat=tformat,
                                               test_mode=test_mode)
        return selected_actions, modified_inputs, selected_actions_format

    def create_model(self, transition_scheme):

        self.scheme_shapes = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                     dict_of_schemes=self.schemes)

        self.input_shapes = _generate_input_shapes(input_columns=self.input_columns,
                                                   scheme_shapes=self.scheme_shapes)

        # set up lambda network
        self.mixing_network = m_REGISTRY[self.args.qmix_mixing_network](input_shapes=self.input_shapes,
                                                                        n_actions=self.n_actions,
                                                                        n_agents=self.n_agents,
                                                                        args=self.args)

        if self.args.use_cuda:
            self.mixing_network = self.mixing_network.cuda()

        # create target networks if required
        self.target_mixing_network = deepcopy(self.mixing_network)
        return

    def generate_initial_hidden_states(self, batch_size):
        """
        generates initial hidden states for each agent
        TODO: would be nice to expand for use with recurrency in lambda (could just be last first slice of hidden states though, \
        or dict element
        """
        agent_hidden_states = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(self.n_agents)])
        agent_hidden_states = agent_hidden_states.cuda() if self.args.use_cuda else agent_hidden_states.cpu()
        return agent_hidden_states, "a*bs*t*v"

    def share_memory(self):
        self.mixing_network.share_memory()
        pass

    def update_target(self):
        assert self.has_target_network, "update_target requires has_target_network to be set!"
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        pass

    def get_outputs(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):

        assert isinstance(inputs, dict) and \
               isinstance(inputs["agent_input__agent0"], BatchEpisodeBuffer), "wrong format (inputs)"
        if self.args.share_agent_params:

            actions = kwargs.get("actions", None)

            inputs, inputs_tformat = _build_model_inputs(self.input_columns,
                                                         inputs,
                                                         to_variable=True,
                                                         inputs_tformat=tformat)

            out, hidden_states, losses, tformat = self.mixing_network(inputs,
                                                                      hidden_states=hidden_states,
                                                                      loss_fn=loss_fn,
                                                                      tformat=inputs_tformat,
                                                                      **kwargs)
            ret = {"hidden_states": hidden_states,
                   "losses": losses,
                   "format":tformat}

            if actions is None:
                out_key = self.agent_output_type
                ret[out_key] = out
            else:
                ret.update(out)
            return ret, tformat
        else:
            assert False, "Not yet implemented."

    def save_models(self, path, token, T):
        # if self.args.share_agent_params:
        #     th.save(self.mixing_network.state_dict(),
        #             "results/models/{}/{}_agentsp__{}_T.weights".format(token, self.args.learner, T))
        # else:
        #     for _agent_id in range(self.args.n_agents):
        #         th.save(self.mixing_network["agent__agent{}".format(_agent_id)].state_dict(),
        #                 "results/models/{}/{}_agent{}__{}_T.weights".format(token, self.args.learner, _agent_id, T))

        th.save(self.mixing_network.state_dict(),
                        "results/models/{}/{}_lambda_network{}__T.weights".format(token, self.args.learner, T))

        pass

    pass




