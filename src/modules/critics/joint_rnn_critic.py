import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.popart import PopArt

class JointRNNCritic(nn.Module):
    def __init__(self, scheme, args):
        super(JointRNNCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.input_shape = self._get_input_shape(scheme)

        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        if getattr(self.args, "is_popart", False):
            self.v_out = PopArt(args.rnn_hidden_dim, 1)
        else:
            self.v_out = nn.Linear(args.rnn_hidden_dim, 1)

        self.detach_every = getattr(args, "detach_every", None)

    def forward(self, batch, t=None):
        bs, max_t = batch.batch_size, batch.max_seq_length

        h_in = self.fc1.weight.new(batch.batch_size, self.args.rnn_hidden_dim).zero_() 

        outputs = []
        for t in range(batch.max_seq_length):
            inputs, _, _ = self._build_inputs(batch, t=t)

            if self.detach_every and  ((t % self.detach_every) == 0):
                h_in = h_in.detach()

            x = F.relu(self.fc1(inputs))
            h_in = self.rnn(x, h_in)
            q = self.v_out(h_in)
            outputs.append(q) # bs * ts

        output_tensor = th.stack(outputs, dim=1)
        return output_tensor

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # state
        central_state = batch["state"][:, ts]
        inputs.append(central_state)

        inputs = th.cat([x.reshape(bs * max_t, -1) for x in inputs], dim=1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]

        return input_shape
