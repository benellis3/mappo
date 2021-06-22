import torch as th
import torch.nn as nn
import torch.nn.functional as F

class FFAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FFAgent, self).__init__()
        self.args = args

        # Easiest to reuse rnn_hidden_dim variable
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        return self.fc1.weight.new(batch_size, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        logits = self.fc3(h)
        return logits
