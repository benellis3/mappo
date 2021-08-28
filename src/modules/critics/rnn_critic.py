import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from components.running_mean_std import RunningMeanStd


class RNNCritic(nn.Module):
    def __init__(self, scheme, args):
        super(RNNCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        input_shape = self._get_input_shape(scheme)
        if getattr(args, "is_observation_normalized", None):
            self.is_obs_normalized = True
            self.obs_rms = RunningMeanStd(shape=np.prod(input_shape))
        else:
            self.is_obs_normalized = False

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

        self.detach_every = getattr(args, "detach_every", None)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, batch):
        bs = batch.batch_size
        hidden_states = self.init_hidden().unsqueeze(0).expand(batch.batch_size, self.n_agents, -1)  # bav

        outputs = []
        h_in = hidden_states.reshape(-1, self.args.rnn_hidden_dim)

        for t in range(batch.max_seq_length):
            inputs = self._build_inputs(batch, t)

            if self.is_obs_normalized: 
                inputs = (inputs - self.obs_rms.mean) / th.sqrt(self.obs_rms.var)

            if self.detach_every and  ((t % self.detach_every) == 0):
                h_in = h_in.detach()

            x = F.relu(self.fc1(inputs))
            h_in = self.rnn(x, h_in)
            q = self.fc2(h_in)
            outputs.append(q.reshape(bs, 1, self.n_agents)) # bs * ts * n_agents

        output_tensor = th.stack(outputs, dim=1)
        return output_tensor

    def update_rms(self, batch_obs):
        self.obs_rms.update(batch_obs)

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

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        return input_shape
