import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNAgent, self).__init__()
        self.args = args
        self.num_channels = input_shape[0] 

        self.cnn1 = nn.Conv1d(in_channels=self.num_channels, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        input_shape = inputs.shape
        assert input_shape[1] // self.num_channels == 0
        inputs = inputs.view(inputs.shape[0], input_shape//self.num_channels, self.num_channels)
        x = F.relu(self.cnn1(inputs))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x.view(inputs.shape[0], -1)
        q = self.fc1(x)
        return q, None
