import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QTranBase(nn.Module):
    def __init__(self, args):
        super(QTranBase, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        # Q(s,u)
        # Q takes [state, u] as input
        q_input_size = self.state_dim + (self.n_agents * self.n_actions)

        if self.args.network_size == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
        elif self.args.network_size == "big":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
        else:
            assert False

    def forward(self, batch, actions=None):
        bs = batch.batch_size
        ts = batch.max_seq_length

        states = batch["state"].reshape(bs * ts, self.state_dim)

        if actions is None:
            # Use the actions taken by the agents
            actions = batch["actions_onehot"].reshape(bs * ts, self.n_agents * self.n_actions)
        else:
            # It will arrive as (bs, ts, agents, actions), we need to reshape it
            actions = actions.reshape(bs * ts, self.n_agents * self.n_actions)

        inputs = th.cat([states, actions], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["state"].reshape(bs * ts, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs


class QTranAlt(nn.Module):
    def __init__(self, args):
        super(QTranAlt, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        # Q(s,-,u-i)
        # Q takes [state, u-i, i] as input
        q_input_size = self.state_dim + (self.n_agents * self.n_actions) + self.n_agents

        if self.args.network_size == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.n_actions))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
        elif self.args.network_size == "big":
             # Adding another layer
             self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.n_actions))
            # V(s)
             self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
        else:
            assert False

    def forward(self, batch, masked_actions=None):
        bs = batch.batch_size
        ts = batch.max_seq_length
        # Repeat each state n_agents times
        repeated_states = batch["state"].repeat(1, 1, self.n_agents).view(-1, self.state_dim)

        if masked_actions is None:
            actions = batch["actions_onehot"].repeat(1, 1, self.n_agents, 1)
            agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
            agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions)#.view(self.n_agents, -1)
            masked_actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)
            masked_actions = masked_actions.view(-1, self.n_agents * self.n_actions)

        agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).repeat(bs, ts, 1, 1).view(-1, self.n_agents)

        inputs = th.cat([repeated_states, masked_actions, agent_ids], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["state"].repeat(1,1,self.n_agents).view(-1, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs


