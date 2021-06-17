import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.running_mean_std import RunningMeanStd

class CentralCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # need to normalize state
        self.is_state_normalized = getattr(self.args, "is_observation_normalized", False)
        if self.is_state_normalized:
            self.state_rms = RunningMeanStd()

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q.view(bs, max_t, 1)

    def update_rms(self, batch, mask):
        state = batch["state"][:, :-1].cuda()
        flat_state = state.reshape(-1, state.shape[-1])
        flat_mask = mask.flatten()
        # ensure the length matches
        assert flat_state.shape[0] == flat_mask.shape[0]
        state_index = th.nonzero(flat_mask).squeeze()
        valid_state = flat_state[state_index]
        # update state_rms
        self.state_rms.update(valid_state)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        state = batch['state'][:, ts]
        # update value_rms
        if self.is_state_normalized:
            state_mean = self.state_rms.mean.unsqueeze(0).unsqueeze(0)
            state_var = self.state_rms.var.unsqueeze(0).unsqueeze(0)
            state_mean = state_mean.expand(bs, max_t, -1)
            state_var = state_var.expand(bs, max_t, -1)
            state = (state - state_mean) / (state_var + 1e-6)
        inputs.append(state)

        # observations
        # inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))

        # last actions
        # if t == 0:
        #     inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
        # elif isinstance(t, int):
        #     inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
        # else:
        #     last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
        #     last_actions = last_actions.view(bs, max_t, 1, -1)
        #     inputs.append(last_actions)

        inputs = th.cat([x.reshape(bs * max_t, -1) for x in inputs], dim=1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        # input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        # input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        return input_shape
