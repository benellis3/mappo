import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNAgent, self).__init__()
        self.args = args
        self.num_frames = getattr(args, "num_frames", 4)

        self.cnn1 = nn.Conv1d(in_channels=self.num_frames, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)

        input_dim = input_shape
        input_dim = input_dim // 2 - 4  # based on cnn output size formula: https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer

        self.fc1 = nn.Linear(256 * input_dim, 128)
        self.fc2 = nn.Linear(128, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        input_shape = inputs.shape
        assert input_shape[1] % self.num_frames == 0
        inputs = inputs.view(inputs.shape[0], self.num_frames, input_shape[1]//self.num_frames)
        x = F.relu(self.cnn1(inputs))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = x.view(inputs.shape[0], -1)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q, None
