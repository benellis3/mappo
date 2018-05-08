import numpy as np
from torch.autograd import Variable
import torch as th

from components.action_selectors import REGISTRY as as_REGISTRY
from components import REGISTRY as co_REGISTRY
from components.scheme import Scheme
from components.episode_buffer import BatchEpisodeBuffer
from components.transforms import _build_model_inputs, _join_dicts, _generate_scheme_shapes, _generate_input_shapes
from models import REGISTRY as m_REGISTRY


class poMACEMultiagentController():
    """
    container object for a set of independent agents
    TODO: may need to propagate test_mode in here as well!
    """

    def __init__(self, runner, n_agents, n_actions, action_selector=None, args=None):
        self.args = args
        self.runner = runner
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_str = args.agent
        assert self.args.agent_output_type in ["policies"], "agent_output_type has to be set to 'policies' for poMACE - makes no sense with other methods!"
        self.agent_output_type = "policies"

        # Set up action selector
        if action_selector is None:
            self.action_selector = as_REGISTRY[args.action_selector](args=self.args)
        else:
            self.action_selector = action_selector

        self.lambda_network_scheme = Scheme([dict(name="pomace_epsilons",
                                                  scope="episode",
                                                  requires_grad=False,
                                                  switch=self.args.multiagent_controller in ["pomace_mac"] and \
                                                         not self.args.pomace_use_epsilon_seed),
                                                 dict(name="pomace_epsilon_seeds",
                                                      scope="episode",
                                                      requires_grad=False,
                                                      switch=self.args.multiagent_controller in ["pomace_mac"] and \
                                                             self.args.pomace_use_epsilon_seed),
                                                 dict(name="pomace_epsilon_variances",
                                                      scope="episode",
                                                      requires_grad=False,
                                                      switch=self.args.multiagent_controller in ["pomace_mac"]),
                                                 dict(name="state"),
                                                 dict(name="agent_id",
                                                      select_agent_ids=list(range(self.n_agents)))
                                                 ])

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
                                                                       ("one_hot", dict(range=(0, self.n_actions-1)))], # DEBUG!
                                                           switch=self.args.obs_last_action),
                                                       dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id])
                                                      ])


        # Set up schemes
        self.schemes = {}
        for _agent_id in range(self.n_agents):
            self.schemes["agent_input__agent{}".format(_agent_id)] = self.agent_scheme_fn(_agent_id).agent_flatten()
        self.schemes["lambda_network"] = self.lambda_network_scheme
        # create joint scheme from the agents schemes and lambda_network_scheme
        self.joint_scheme_dict = _join_dicts(self.schemes)

        # construct model-specific input regions
        self.input_columns = {}
        for _agent_id in range(self.n_agents):
            self.input_columns["agent_input__agent{}".format(_agent_id)] = {}
            self.input_columns["agent_input__agent{}".format(_agent_id)]["main"] = \
                Scheme([dict(name="agent_observation", select_agent_ids=[_agent_id]),
                        dict(name="past_action",
                             select_agent_ids=[_agent_id],
                             switch=self.args.obs_last_action),
                        dict(name="agent_id", select_agent_ids=[_agent_id])])
            # self.input_columns["agent_input__agent{}".format(_agent_id)]["agent_ids"] = \
            #     Scheme([dict(name="agent_id", select_agent_ids=[_agent_id])])

        self.input_columns["lambda_network"] = {}
        self.input_columns["lambda_network"]["pomace_epsilon_variances"] = Scheme([dict(name="pomace_epsilon_variances",
                                                                                        scope="episode")])
        if self.args.pomace_use_epsilon_seed:
            self.input_columns["lambda_network"]["pomace_epsilon_seeds"] = Scheme([dict(name="pomace_epsilon_seeds",
                                                                                        scope="episode")])
        else:
            self.input_columns["lambda_network"]["pomace_epsilons"] =      Scheme([dict(name="pomace_epsilons",
                                                                                        scope="episode")])
        self.input_columns["lambda_network"]["state"] = Scheme([dict(name="state")])
        for _agent_id in range(self.n_agents):
            self.input_columns["lambda_network"]["agent_ids__agent{}".format(_agent_id)] = \
                Scheme([dict(name="agent_id",
                             select_agent_ids=[_agent_id])])

        pass

    def get_parameters(self):
        parameters = self.lambda_network_model.parameters()
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
        self.lambda_network_model = m_REGISTRY[self.args.pomace_multiagent_network](input_shapes=self.input_shapes,
                                                                                    n_actions=self.n_actions,
                                                                                    n_agents=self.n_agents,
                                                                                    args=self.args)

        if self.args.use_cuda:
            self.lambda_network_model = self.lambda_network_model.cuda()
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
        self.lambda_network_model.share_memory()
        pass

    def get_outputs(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        assert isinstance(inputs, dict) and \
               isinstance(inputs["agent_input__agent0"], BatchEpisodeBuffer), "wrong format (inputs)"
        if self.args.share_agent_params:
            inputs, inputs_tformat = _build_model_inputs(self.input_columns,
                                                         inputs,
                                                         to_variable=True,
                                                         inputs_tformat=tformat)

            out, hidden_states, losses, tformat = self.lambda_network_model(inputs,
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
            th.save(self.agent_model.state_dict(),
                    "results/models/{}/{}_agentsp__{}_T.weights".format(token, self.args.learner, T))
        else:
            for _agent_id in range(self.args.n_agents):
                th.save(self.agent_models["agent__agent{}".format(_agent_id)].state_dict(),
                        "results/models/{}/{}_agent{}__{}_T.weights".format(token, self.args.learner, _agent_id, T))

        th.save(self.lambda_network_model.state_dict(),
                        "results/models/{}/{}_lambda_network{}__T.weights".format(token, self.args.learner, T))

        pass

    pass




