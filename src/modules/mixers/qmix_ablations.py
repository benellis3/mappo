import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixerLin(nn.Module):
    def __init__(self, args):
        super(QMixerLin, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        self.hyper_w = nn.Linear(self.state_dim, self.n_agents)        
        if getattr(self.args, "hypernet_layers", 1) > 1:
            assert self.args.hypernet_layers == 2, "Only 1 or 2 hypernet_layers is supported atm!"
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w_final = th.abs(self.hyper_w(states))
        w_final = w_final.view(-1, self.n_agents, 1)
        # State-dependent bias

        v = self.V(states).view(-1, 1, 1)

        y = th.bmm(agent_qs, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class QMixerNS(nn.Module):
    def __init__(self, args):
        super(QMixerNS, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        # First layer
        w_1 = th.empty(self.n_agents, self.embed_dim)
        th.nn.init.kaiming_uniform_(w_1)
        self.w_1 = nn.Parameter(w_1)
        b_1 = th.empty(self.embed_dim)
        fan_in, _ = th.nn.init._calculate_fan_in_and_fan_out(self.w_1)
        bound = 1 / np.sqrt(fan_in)
        th.nn.init.uniform_(b_1, -bound, bound)
        self.b_1 = nn.Parameter(b_1)

        # Last layer
        w_final = th.empty(self.embed_dim, 1)
        th.nn.init.kaiming_uniform_(w_final)
        self.w_final = nn.Parameter(w_final)
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = th.abs(self.w_1)
        w1 = w1.view(self.n_agents, self.embed_dim)
        hidden = F.elu(th.matmul(agent_qs, w1) + self.b_1)

        # Second layer
        w_final = th.abs(self.w_final)
        w_final = w_final.view(self.embed_dim, 1)

        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)

        y = th.matmul(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class VDNState(nn.Module):
    def __init__(self, args):
        super(VDNState, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        v = self.V(states).view(-1, 1, 1)

        y = th.sum(agent_qs, dim=2, keepdim=True) + v
        q_tot = y.view(bs, -1, 1)
        return q_tot


class QMixer2LayerLin(nn.Module):
    def __init__(self, args):
        super(QMixer2LayerLin, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        if getattr(self.args, "hypernet_layers", 1) > 1:
            assert self.args.hypernet_layers == 2, "Only 1 or 2 hypernet_layers is supported atm!"
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))

        # Initialise the hyper networks with a fixed variance, if specified
        if self.args.hyper_initialization_nonzeros > 0:
            std = self.args.hyper_initialization_nonzeros ** -0.5
            self.hyper_w_1.weight.data.normal_(std=std)
            self.hyper_w_1.bias.data.normal_(std=std)
            self.hyper_w_final.weight.data.normal_(std=std)
            self.hyper_w_final.bias.data.normal_(std=std)

        # Initialise the hyper-network of the skip-connections, such that the result is close to VDN
        # if self.args.skip_connections:
        #     self.skip_connections = nn.Linear(self.state_dim, self.args.n_agents, bias=True)
        #     self.skip_connections.bias.data.fill_(1.0)  # bias produces initial VDN weights

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        if self.args.gated:
            self.gate = nn.Parameter(th.ones(size=(1,)) * 0.5)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        # NO NON-LINEARITY HERE!!!
        hidden = th.bmm(agent_qs, w1) + b1
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Skip connections
        s = 0
        if self.args.skip_connections:
            s = agent_qs.sum(dim=2, keepdim=True)

        if self.args.gated:
            y = th.bmm(hidden, w_final) * self.gate + v + s
        else:
            # Compute final output
            y = th.bmm(hidden, w_final) + v + s
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot