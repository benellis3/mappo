import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VanillaCritic(nn.Module):
    def __init__(self, scheme, args):
        super(VanillaCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.central_v = getattr(self.args, 'is_central_value', False)

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q.view(bs, self.n_agents, -1)

    def forward_obs(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # observations
        if self.central_v:
            state = batch["state"][:, ts].expand((-1, self.n_agents, -1))
            inputs.append(state)
        else:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        if self.central_v:
            input_shape = scheme["state"]["vshape"]
        else:
            input_shape = scheme["obs"]["vshape"]

        # last action
        if getattr(self.args, 'obs_last_action', None):
            input_shape += scheme["actions_onehot"]["vshape"][0]

        # agent id
        if getattr(self.args, 'obs_agent_id', None):
            input_shape += self.n_agents
        return input_shape
