import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class Conv1dCritic(nn.Module):
    def __init__(self, scheme, args):
        self.args = args
        input_shape = self._get_input_shape(scheme)
        assert isinstance(input_shape, tuple), "Conv1d agent only accepts input_shape in tuple format"
        super(Conv1dCritic, self).__init__()
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.output_type = "v"

        self.dim_channels = input_shape[0]
        self.conv1 = nn.Conv1d(self.dim_channels, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=0)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=0)
        self.fc1 = nn.Linear(self._count_input(input_shape), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, ep_batch, t=None):
        inputs = self._build_inputs(ep_batch, t)
        assert inputs.shape[-1] % self.dim_channels == 0 # frames are stacked exactly
        inputs = inputs.view(inputs.shape[0], 
                             self.dim_channels, 
                             inputs.shape[-1] // self.dim_channels)

        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _only_conv1d(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # flatten
        return x

    def _count_input(self, image_dim):
        return self._only_conv1d(torch.rand(1, *(image_dim))).data.size(1)

    def init_hidden(self):
        pass

    def _get_input_shape(self, scheme):
        # state
        framestack_num = self.args.env_args.get("framestack_num", 1)
        obs_shape = scheme["obs"]["vshape"]

        assert obs_shape % framestack_num == 0
        input_shape = (framestack_num, obs_shape // framestack_num) # channels come first

        return input_shape

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs