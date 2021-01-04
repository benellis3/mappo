import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from components.running_mean_std import RunningMeanStd


class CNNCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CNNCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.num_frames = getattr(args, "num_frames", 4)

        input_shape = self._get_input_shape(scheme)
        if getattr(args, "is_observation_normalized", None):
            self.is_obs_normalized = True
            self.obs_rms = RunningMeanStd(shape=np.prod(input_shape))
        else:
            self.is_obs_normalized = False

        self.cnn1 = nn.Conv1d(in_channels=self.num_frames, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)

        input_dim = input_shape // 2 - 4 # based on cnn output size formula: https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer
        self.fc1 = nn.Linear(256 * input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, batch):
        bs = batch.batch_size

        outputs = []

        for t in range(batch.max_seq_length):
            inputs = self._build_inputs(batch, t)
            if self.is_obs_normalized:
                inputs = (inputs - self.obs_rms.mean.repeat(self.num_frames)) / th.sqrt(self.obs_rms.var.repeat(self.num_frames))

            input_shape = inputs.shape
            assert input_shape[1] % self.num_frames == 0
            inputs = inputs.view(input_shape[0], self.num_frames, input_shape[1]//self.num_frames)

            x = F.relu(self.cnn1(inputs))
            x = F.relu(self.cnn2(x))
            x = F.relu(self.cnn3(x))
            x = x.view(inputs.shape[0], -1)
            x = F.relu(self.fc1(x))
            q = self.fc2(x)

            outputs.append(q.reshape(bs, 1, self.n_agents)) # bs * ts * n_agents

        output_tensor = th.stack(outputs, dim=1)
        return output_tensor
    
    def update_rms(self, batch_obs):
        self.obs_rms.update(batch_obs)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        for i in reversed(range(self.num_frames)): # stacking 4 frames
            if t - i < 0:
                inputs.append(th.zeros_like(batch["obs"][:, t]))
            else:
                inputs.append(batch["obs"][:, t-i])

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        return input_shape
