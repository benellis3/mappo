import torch as th

import controllers
x = controllers.REGISTRY
from components.action_selectors import REGISTRY as as_REGISTRY

from controllers import REGISTRY as c_REGISTRY
from components.scheme import Scheme
from components.episode_buffer import BatchEpisodeBuffer
from components.transforms import _build_model_inputs, _join_dicts


class IndependentMultiagentController():
    """
    container object for a set of independent agents
    """

    def __init__(self, runner, n_agents, n_actions, action_selector=None, args=None, **kwargs):
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_str = args.agent
        self.agent_output_type = self.args.agent_output_type
        self.runner = runner
        self.has_target_network = kwargs.get("has_target_network", True)

        # Set up action selector
        if action_selector is None:
            self.action_selector = as_REGISTRY[args.action_selector](args=self.args)
        else:
            self.action_selector = action_selector

        # now set up agents
        self.agents = []
        for _i in range(self.n_agents):
            agent = c_REGISTRY[self.agent_str](args=self.args,
                                               agent_id=_i,
                                               n_agents=n_agents,
                                               n_actions=self.n_actions,
                                               output_type=self.agent_output_type)
            self.agents.append(agent)

        # Set up schemes
        self.schemes = {}
        # create joint scheme from the agents schemes
        for _i, agent in enumerate(self.agents):
            self.schemes["agent__agent{}".format(_i)] = agent.scheme["main"]  # agent.scheme_fn(_i).agent_flatten()
        self.joint_scheme_dict = _join_dicts(self.schemes)

        # construct model-specific input regions
        self.input_columns = {}
        self.input_columns = _join_dicts(*[{"agent__agent{}".format(_agent_id): self.agents[_agent_id].input_columns["main"]}
                                           for _agent_id in range(self.n_agents)])
        pass

    def get_parameters(self):
        #parameters = {}
        #for _i, agent in enumerate(self.agents):
        #    parameters["agent{}".format(_i)] = agent.get_parameters()
        parameters = []
        for _i, agent in enumerate(self.agents):
            parameters.extend(agent.get_parameters())
        return parameters

    def select_actions(self, inputs, avail_actions, tformat, info, test_mode=False):
        selected_actions, modified_inputs, selected_actions_format = \
            self.action_selector.select_action(inputs,
                                               avail_actions=avail_actions,
                                               tformat=tformat,
                                               test_mode=test_mode)
        return selected_actions, modified_inputs, selected_actions_format

    def create_model(self, transition_scheme):

        # load agent models
        if self.args.share_agent_params:
            model = None
            for _i, _agent in enumerate(self.agents):
                if _i == 0:
                    _agent.create_model(transition_scheme)
                    model = _agent.model
                else:
                    _agent.model = model
        else:
            for _agent in self.agents:
                _agent.create_model(transition_scheme)

        # create target networks if required
        if self.has_target_network:
            if self.args.share_agent_params:
                from copy import deepcopy
                self.target_model = deepcopy(self.agents[0].model)
            else:
                self.target_models = [_a.model.clone() for _a in self.agents]

        return

    def generate_initial_hidden_states(self, batch_size):
        """
        generates initial hidden states for each agent
        """
        hidden_states = th.stack([_a.generate_initial_hidden_states(batch_size)[0] for _a in self.agents])
        format = "a*bs*t*v"
        return hidden_states, format

    def share_memory(self):
        [_a.model.share_memory() for _a in self.agents]
        pass

    def update_target(self):
        assert self.has_target_network, "update_target requires has_target_network to be set!"
        if self.args.share_agent_params:
            self.target_model.load_state_dict(self.agents[0].model.state_dict())
        else:
            [_t.load_state_dict(_a.model.state_dict()) for _a, _t in zip(self.agents, self.target_models)]
        pass

    def get_outputs(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        target_mode = kwargs.get("target_mode", False)

        assert isinstance(inputs, dict) and \
               isinstance(inputs["agent__agent0"], BatchEpisodeBuffer), "wrong format (inputs)"
        if self.args.share_agent_params:
            inputs, inputs_tformat = _build_model_inputs(self.input_columns,
                                                         inputs,
                                                         to_variable=True,
                                                         inputs_tformat=tformat)

            if not target_mode:
                out, hidden_states, losses, tformat = self.agents[0].model(inputs["agent"],
                                                                       hidden_states=hidden_states,
                                                                       loss_fn=loss_fn,
                                                                       tformat=inputs_tformat,
                                                                       **kwargs)
            else:
                out, hidden_states, losses, tformat = self.target_model(inputs["agent"],
                                                                        hidden_states=hidden_states,
                                                                        loss_fn=loss_fn,
                                                                        tformat=inputs_tformat,
                                                                        **kwargs)

            ret = {"hidden_states": hidden_states,
                   "losses": losses,
                   "format": tformat}

            out_key = self.agent_output_type
            ret[out_key] = out
            return ret, tformat
        else:
            assert False, "Not yet implemented."

    def save_models(self, path, token, T):
        if self.args.share_agent_params:
            th.save(self.agents[0].model.state_dict(),
                    "results/models/{}/{}_agentsp__{}_T.weights".format(token, self.args.learner, T))
        else:
            for _agentid, agent in enumerate(self.agents):
                th.save(self.agents[_agentid].model.state_dict(),
                        "results/models/{}/{}_agent{}__{}_T.weights".format(token, self.args.learner, _agentid, T))
        pass

    pass




