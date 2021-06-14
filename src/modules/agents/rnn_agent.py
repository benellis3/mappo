import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self, bs):
        # make hidden states on same device as model
        self.h_in = self.fc1.weight.new(bs, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        self.h_in = self.rnn(x, self.h_in)
        logits = self.fc2(self.h_in)
        return logits
