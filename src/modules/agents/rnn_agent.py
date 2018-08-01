import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return th.zeros((1,self.args.rnn_hidden_dim))

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        # TODO: Is this right? It feels hacky reshaping here
        x = x.reshape(-1, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        h = h.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)
        q = self.fc2(h)
        return q, h
