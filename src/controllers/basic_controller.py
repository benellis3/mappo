from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from components.running_mean_std import RunningMeanStd


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)

        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.need_agent_logits = getattr(args, "need_agent_logits", False)
        self.detach_every = getattr(args, "detach_every", None)
        self.replace_every = getattr(args, "replace_every", None)

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.other_outs = None

        if getattr(self.args, "is_observation_normalized", False):
            # need to normalize both obs & state
            self.obs_rms = RunningMeanStd()
            self.is_obs_normalized = True
        else:
            self.is_obs_normalized = False

    def update_rms(self, batch, alive_mask):
        obs = batch["obs"][:, :].cuda() # ignore the last obs
        flat_obs = obs.clone().reshape(-1, obs.shape[-1])
        flat_alive_mask = alive_mask.flatten()
        # ensure the length matches
        assert flat_obs.shape[0] == flat_alive_mask.shape[0]
        obs_index = th.nonzero(flat_alive_mask).squeeze()
        valid_obs = flat_obs[obs_index]
        # update obs_rms
        self.obs_rms.update(valid_obs)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        if self.args.agent == "rnn":
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, enable_norm=True)
        else:
            agent_outputs = self.forward_ff(ep_batch)[:, t_ep]
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, enable_norm=False):
        # For training: obs normalization has already been done in learner
        agent_inputs = self._build_inputs(ep_batch, t, enable_norm=enable_norm)

        if self.detach_every and  ((t % self.detach_every) == 0):
            self.agent.detach_hidden()
            assert self.agent.h_in.is_leaf, self.agent.h_in
        elif t != 0:
            assert not self.agent.h_in.is_leaf, self.agent.h_in

        if self.replace_every and  ((t % self.replace_every) == 0):
            self.agent.replace_hidden(ep_batch["hidden"][:,t].reshape((ep_batch.batch_size*self.n_agents, -1)))

        agent_outs = self.agent(agent_inputs)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def forward_ff(self, ep_batch):
        bs, max_t = ep_batch.batch_size, ep_batch.max_seq_length
        agent_inputs = self._build_inputs_ff(ep_batch)
        agent_outs = self.agent(agent_inputs)
        return agent_outs.view(bs, max_t, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.agent.init_hidden(batch_size * self.n_agents)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        self.obs_rms.save_model(path, "obs")

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.obs_rms.load_model(path, "obs")

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs_ff(self, batch):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs, max_t = batch.batch_size, batch.max_seq_length
        inputs = []

        inputs.append(batch['obs']) # ignore the last entry

        if self.args.obs_last_action:
            actions_onehot = th.zeros_like(batch['actions_onehot'])
            actions_onehot[:, 1:] = batch['actions_onehot']
            inputs.append(actions_onehot) # ignore the last entry

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs*max_t*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_inputs(self, batch, t, enable_norm=False):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []

        obs =batch["obs"][:, t]
        # normalize obs
        if enable_norm and self.is_obs_normalized:
            obs_mean = self.obs_rms.mean.unsqueeze(0).unsqueeze(0)
            obs_var = self.obs_rms.var.unsqueeze(0).unsqueeze(0)

            obs_mean = obs_mean.expand(bs, self.n_agents, -1)
            obs_var = obs_var.expand(bs, self.n_agents, -1)

            # update obs directly in batch
            obs = (obs - obs_mean) / th.sqrt(obs_var + 1e-6 )

        inputs.append(obs)

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        if getattr(self.args, 'obs_last_action', None):
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if getattr(self.args, 'obs_agent_id', None):
            input_shape += self.n_agents

        return input_shape
