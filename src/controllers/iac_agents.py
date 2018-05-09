from components.scheme import Scheme
from controllers.basic_agent import BasicAgentController
from controllers.independent_agents import IndependentMultiagentController

class IACAgentController(BasicAgentController):

    def __init__(self, n_agents, n_actions, args, agent_id=None, model=None, output_type="policies", scheme=None):

        scheme_fn = lambda _agent_id: Scheme([dict(name="agent_id",
                                                   transforms=[("one_hot",dict(range=(0, n_agents-1)))],
                                                   select_agent_ids=[_agent_id],),
                                                   # better to have ON all the time as shared_params
                                                   #switch=self.args.obs_agent_id),
                                              dict(name="observations",
                                                   rename="agent_observation",
                                                   select_agent_ids=[_agent_id]),
                                              dict(name="actions",
                                                   rename="past_action",
                                                   select_agent_ids=[_agent_id],
                                                   transforms=[("shift", dict(steps=1)),
                                                               ("one_hot", dict(range=(0, n_actions-1)))], # DEBUG!
                                                   switch=args.obs_last_action),
                                              dict(name="iac_epsilons",
                                                   rename="epsilons",
                                                   scope="episode")
                                             ]).agent_flatten()

        input_columns = {}
        input_columns["main"] = {}
        input_columns["main"]["main"] = \
            Scheme([dict(name="agent_id", select_agent_ids=[agent_id]),
                    dict(name="agent_observation", select_agent_ids=[agent_id]),
                    dict(name="past_action",
                         select_agent_ids=[agent_id],
                         switch=args.obs_last_action),
                    ]).agent_flatten()
        input_columns["main"]["epsilons"] = Scheme([dict(name="epsilons", scope="episode")]).agent_flatten()

        if model is not None:
            assert model in ["iac_recursive", "iac_non_recursive"], "wrong IAC model set!"

        super().__init__(n_agents=n_agents,
                         n_actions=n_actions,
                         args=args,
                         agent_id=agent_id,
                         model=model,
                         scheme_fn=scheme_fn,
                         input_columns=input_columns)
        pass

class IACMultiAgentController(IndependentMultiagentController):

    def __init__(self, runner, n_agents, n_actions, action_selector=None, args=None, **kwargs):
        assert args.action_selector in ["multinomial"], "wrong IAC action selector set!"
        assert args.agent in ["iac_recursive_ac", "iac_non_recursive_ac"], "wrong IAC model set!"

        super().__init__(runner=runner,
                         n_agents=n_agents,
                         n_actions=n_actions,
                         action_selector=action_selector,
                         args=args,
                         **kwargs)
        pass
