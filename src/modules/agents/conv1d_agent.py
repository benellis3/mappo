import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class Conv1dAgent(nn.Module):
    def __init__(self, input_shape, args, n_outs=None):
        assert isinstance(input_shape, tuple), "Conv1d agent only accepts input_shape in tuple format"
        super(Conv1dAgent, self).__init__()
        self.args = args

        self.dim_channels = input_shape[0]
        self.conv1 = nn.Conv1d(self.dim_channels, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=0)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=0)
        self.fc1 = nn.Linear(self._count_input(input_shape), 256)
        self.fc2 = nn.Linear(256, 128)
        if n_outs is None:
            n_outs = args.n_actions
        self.fc3 = nn.Linear(128, n_outs)
        self.fc_val = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, inputs, t=None):
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
        val = self.fc_val(x)
        other_outs = {}
        other_outs.update({"hidden_states": None})
        other_outs.update({"values": val})
        return q, other_outs

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

class Conv1dRnnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Conv1dRnnAgent, self).__init__()
        self.type = "rnn_cell"
        self.args = args

        in_channels = input_shape[0]
        self.conv1 = nn.Conv1d(in_channels, 64, 3,  stride=2)
        self.conv2 = nn.Conv1d(64, 128, 3)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.flatten = nn.Flatten(1)

        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc1 = nn.Linear(self._count_input(input_shape), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.n_actions)

    def forward(self, inputs, hidden_states, t=None):
        #inputs = self._build_inputs(inputs)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        h_in = hidden_states.reshape(-1, self.args.rnn_hidden_dim)
        x = self.rnn(x, h_in)
        h = x.clone()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, h

    def _only_conv1d(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return x

    def _count_input(self, image_dim):
        return self._only_conv1d(torch.rand(1, *(image_dim))).data.size(1)

    def init_hidden(self):
        pass

class Conv1dRnnNoncellAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Conv1dRnnNoncellAgent, self).__init__()
        self.type = "rnn_noncell"
        self.args = args

        in_channels = input_shape[0]
        self.conv1 = nn.Conv1d(in_channels, 64, 3,  stride=2)
        self.conv2 = nn.Conv1d(64, 128, 3)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.flatten = nn.Flatten(2) # only flatten from index after time

        self.rnn = nn.GRU(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc1 = nn.Linear(self._count_input(input_shape), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.n_actions)

    def forward(self, inputs, hidden_states_init, seq_lens, t=None):
        #inputs = self._build_inputs(inputs)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = nn.utils.rnn.pack_padded_sequence(x, seq_lens)
        h_in = hidden_states_init.reshape(-1, self.args.rnn_hidden_dim)
        x = self.rnn(x, h_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, None

    def _only_conv1d(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return x
    def _count_input(self, image_dim):
        return self._only_conv1d(torch.rand(1, *(image_dim))).data.size(1)

    def init_hidden(self):
        pass