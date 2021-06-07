import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from components.running_mean_std import RunningMeanStd


class AgentSpecificCentralCritic(nn.Module):
    def __init__(self, scheme, args):
        super(AgentSpecificCentralCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        if getattr(args, "is_observation_normalized", None):
            self.is_obs_normalized = True
            self.obs_rms = RunningMeanStd(shape=np.prod(input_shape))
        else:
            self.is_obs_normalized = False

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)

        if self.is_obs_normalized: 
            inputs = (inputs - self.obs_rms.mean) / th.sqrt(self.obs_rms.var)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q.view(bs, max_t, self.n_agents, 1)

    def update_rms(self, batch):
        inputs, _, _ = self._build_inputs(batch)
        self.obs_rms.update(inputs)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # state
        central_state = batch["state"][:, ts]
        central_state = central_state.unsqueeze(dim=2)
        # repeat for n agents
        central_state = central_state.repeat(1, 1, self.n_agents, 1)

        inputs.append(central_state)

        # agent-specific observation
        inputs.append(batch["obs"][:, ts])

        inputs = th.cat([x.reshape(bs * max_t * self.n_agents, -1) for x in inputs], dim=1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # agent-specific observation
        input_shape += scheme["obs"]["vshape"] * 1
        return input_shape
