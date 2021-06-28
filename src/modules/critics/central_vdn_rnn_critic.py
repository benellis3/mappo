import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CentralVDNRNNCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVDNRNNCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, batch, t=None):
        bs, max_t = batch.batch_size, batch.max_seq_length

        h_in = self.fc1.weight.new(batch.batch_size * self.n_agents, self.args.rnn_hidden_dim).zero_() 

        # avail_actions = batch['avail_actions'].cuda()
        # alive_mask = ( (avail_actions[:, :, :, 0] != 1.0) * (th.sum(avail_actions, dim=-1) != 0.0) ).float()
        # dead_mask = alive_mask.unsqueeze(-1).repeat(1, 1, 1, self.input_shape)
        # dead_mask[:, :, :, -self.n_agents:] = 1.0

        outputs = []
        for t in range(batch.max_seq_length):
            inputs, _, _ = self._build_inputs(batch, t=t)
            # inputs = inputs * dead_mask[:, t].reshape(batch.batch_size * self.n_agents, self.input_shape)

            x = F.relu(self.fc1(inputs))
            h_in = self.rnn(x, h_in)
            q = self.fc2(h_in)
            outputs.append(q.view(bs, self.n_agents)) # bs * ts * n_agents

        output_tensor = th.stack(outputs, dim=1)
        # vdn mix
        output_tensor = th.sum(output_tensor, dim=2)

        return output_tensor

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # state
        central_state = batch["state"][:, ts]
        central_state = central_state.unsqueeze(dim=2)
        # repeat for n agents
        central_state = central_state.repeat(1, 1, self.n_agents, 1)

        inputs.append(central_state)

        # agent-specific observation
        inputs.append(batch["obs"][:, ts])

        # agent id
        agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, -1, -1, -1)
        inputs.append(agent_ids)

        # last actions
        # if t == 0:
        #     inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
        # elif isinstance(t, int):
        #     inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
        # else:
        #     last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
        #     last_actions = last_actions.view(bs, max_t, 1, -1)
        #     inputs.append(last_actions)

        inputs = th.cat([x.reshape(bs * max_t * self.n_agents, -1) for x in inputs], dim=1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # agent-specific observation
        input_shape += scheme["obs"]["vshape"] * 1
        # agent id
        input_shape += self.n_agents
        # last actions
        # input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        return input_shape
