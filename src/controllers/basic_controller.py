class BasicMAC:

    def __init__(self, agents, args):
        self.agents = agents
        self.args = args

    def select_actions(self, inputs, test_mode=False):
        return {"actions": [0 for _ in range(self.n_agents)]}  # Dummy for quick testing

    def forward(self, inputs, ):
        pass