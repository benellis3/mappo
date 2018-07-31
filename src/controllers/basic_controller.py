from modules.agents import REGISTRY as agent_REGISTRY

import torch as th

class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        input_shape = self._get_input_shape(scheme)
        self.args = args
        self._build_agents(input_shape)

        self.hidden_states = None

    def select_actions(self, ep_batch, t, test_mode=False):
        agent_outputs = self.forward(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        # TODO: Use an action selector
        return {"actions": [0 for _ in range(self.n_agents)]}  # Dummy for quick testing

    def forward(self, ep_batch, t):
        # TODO: Is this too hacky?
        if t == 0:
            # Make initial hidden states
            self.hidden_states = self.agent.init_hidden().expand(self.n_agents, -1).unsqueeze(0) # 1av
            self.hidden_states.to(ep_batch.device)
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # TODO: Return a dictionary?
        return agent_outs

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, 1, -1, -1))

        inputs = th.cat([x.squeeze(1) for x in inputs], dim=2)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape