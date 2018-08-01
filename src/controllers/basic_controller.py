from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t, test_mode=False):
        agent_outputs = self.forward(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        chosen_actions = self.action_selector.select_action(agent_outputs, avail_actions, t, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t):
        # TODO: Is this too hacky?
        if t == 0:
            # Make initial hidden states
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(ep_batch.batch_size, self.n_agents, -1) # bav
            self.hidden_states.to(ep_batch.device)
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # TODO: Return a dictionary?
        return agent_outs

    def get_params(self):
        return self.agent.parameters()

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
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, 1, -1, -1))

        inputs = th.cat([x.squeeze(1) for x in inputs], dim=2)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape