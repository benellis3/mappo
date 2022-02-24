import numpy as np
import torch as th
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from collections import defaultdict

from components.episode_buffer import EpisodeBatch
from modules.critics import critic_REGISTRY
from components.running_mean_std import RunningMeanStd

def compute_logp_entropy(logits, actions, masks):
    masked_logits = th.where(masks, logits, th.tensor(th.finfo(logits.dtype).min).to(logits.device))
    # normalize logits
    masked_logits = masked_logits - masked_logits.logsumexp(dim=-1, keepdim=True)

    probs = th.nn.functional.softmax(masked_logits, dim=-1)
    p_log_p = masked_logits * probs
    p_log_p = th.where(masks, p_log_p, th.tensor(0.0).to(p_log_p.device))
    entropy = -p_log_p.sum(-1)

    logpac = th.gather(masked_logits, dim=-1, index=actions)

    result = {
        'logp' : th.squeeze(masked_logits),
        'logpac' : th.squeeze(logpac),
        'entropy' : th.squeeze(entropy),
    }
    return result

class TrustRegionLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())

        self.critic = critic_REGISTRY[self.args.critic](scheme, args)
        self.critic_params = list(self.critic.parameters())

        self.optimiser_actor = Adam(params=self.agent_params, lr=args.lr_actor, eps=args.optim_eps)
        self.optimiser_critic = Adam(params=self.critic_params, lr=args.lr_critic, eps=args.optim_eps)
        self.t = 0
        self.scheduler_actor = LambdaLR(optimizer=self.optimiser_actor, lr_lambda=lambda epoch: 1.0 - (self.t / args.t_max))
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.mini_epochs_actor = getattr(self.args, "mini_epochs_actor", 4)
        self.mini_epochs_critic = getattr(self.args, "mini_epochs_critic", 4)
        self.advantage_calc_method = getattr(self.args, "advantage_calc_method", "GAE")
        self.agent_type = getattr(self.args, "agent", None)
        self.env_type = getattr(self.args, "env", None)

        self.is_obs_normalized = getattr(self.args, "is_observation_normalized", False)
        self.is_value_normalized = getattr(self.args, "is_value_normalized", False)
        self.is_popart = getattr(self.args, "is_popart", False)
        self.clip_range = getattr(self.args, "clip_range", 0.1)

        self.bootstrap_timeouts = getattr(self.args, "bootstrap_timeouts", False)

        if (self.is_value_normalized and self.is_popart):
            raise ValueError("Either `is_value_normalized` or `is_popart` is specified, but not both.")

        if self.is_value_normalized:
            # need to normalize value
            self.value_rms = RunningMeanStd()

        if self.is_obs_normalized:
            self.state_rms = RunningMeanStd()

    def normalize_value(self, returns, mask):
        bs, ts = returns.shape[0], returns.shape[1]

        flat_returns = returns.flatten()
        flat_mask = mask.flatten()
        # ensure the length matches
        assert flat_returns.shape[0] == flat_mask.shape[0]
        returns_index = th.nonzero(flat_mask).squeeze()
        valid_returns = flat_returns[returns_index]

        if self.is_value_normalized:
            # update value_rms
            self.value_rms.update(valid_returns)
            value_mean = self.value_rms.mean.unsqueeze(0)
            value_var = self.value_rms.var.unsqueeze(0)
            value_mean = value_mean.expand(bs, ts)
            value_var = value_var.expand(bs, ts)
            normalized_returns = (returns - value_mean) / th.sqrt(value_var + 1e-6) * mask

        elif self.is_popart:
            # update popart
            self.critic.v_out.update(valid_returns)
            normalized_returns = self.critic.v_out.normalize(returns)

        return normalized_returns

    def denormalize_value(self, values):
        if self.is_value_normalized:
            bs, ts = values.shape[0], values.shape[1]
            value_mean = self.value_rms.mean.unsqueeze(0)
            value_var = self.value_rms.var.unsqueeze(0)
            value_mean = value_mean.expand(bs, ts)
            value_var = value_var.expand(bs, ts)
            denormalized_values = values * th.sqrt(value_var + 1e-6) + value_mean

        elif self.is_popart:
            denormalized_values = self.critic.v_out.denormalize(values)

        return denormalized_values

    def normalize_state(self, batch, mask):
        bs, ts = batch.batch_size, batch.max_seq_length

        state = batch["state"][:, :].cuda()
        flat_state = state.reshape(-1, state.shape[-1])
        flat_mask = mask.flatten()
        # ensure the length matches
        assert flat_state.shape[0] == flat_mask.shape[0]
        state_index = th.nonzero(flat_mask).squeeze()
        valid_state = flat_state[state_index]
        # update state_rms
        self.state_rms.update(valid_state)
        state_mean = self.state_rms.mean.unsqueeze(0).unsqueeze(0)
        state_var = self.state_rms.var.unsqueeze(0).unsqueeze(0)
        state_mean = state_mean.expand(bs, ts, -1)
        state_var = state_var.expand(bs, ts, -1)
        expanded_mask = mask.expand(-1, -1, state_mean.shape[-1])
        batch.data.transition_data['state'][:, :] = (batch['state'][:, :] - state_mean) / th.sqrt(state_var + 1e-6) * expanded_mask

    def normalize_obs(self, batch, alive_mask):
        bs, ts = batch.batch_size, batch.max_seq_length
        obs_mean = self.mac.obs_rms.mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        obs_var = self.mac.obs_rms.var.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        obs_mean = obs_mean.expand(bs, ts, self.n_agents, -1)
        obs_var = obs_var.expand(bs, ts, self.n_agents, -1)

        expanded_alive_mask = alive_mask.unsqueeze(-1).expand(-1, -1, -1, obs_mean.shape[-1])

        # update obs directly in batch
        batch.data.transition_data['obs'][:, :] = (batch['obs'][:, :] - obs_mean) / th.sqrt(obs_var + 1e-6 ) * expanded_alive_mask

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.t = t_env
        critic_train_stats = defaultdict(lambda: [])
        max_t = batch.max_seq_length

        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        timed_out = batch["timed_out"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.bootstrap_timeouts:
            # Do not train on states that timeout, or else their value target will be calculated as terminal without bootstrap
            # Note: The environment could still mess this up if it terminates itself at timeout without setting episode_limit=true
            mask = mask * (1 - timed_out)
            # make sure it is not the case you have a timeout out flag without terminated
            # in logical form: assert all(not (batch["timed_out"] and not batch["terminated"])):
            assert th.all(1 - ((batch["timed_out"] * (1-batch["terminated"]))) )
        ## get dead agents
        # no-op (valid only when dead)
        # https://github.com/oxwhirl/smac/blob/013cf27001024b4ce47f9506f2541eca0b247c95/smac/env/starcraft2/starcraft2.py#L499
        avail_actions = batch['avail_actions'][:, :].cuda()

        if self.env_type == 'matrix_game':
            alive_mask = ( th.sum(avail_actions, dim=-1)>0.0 ).float()
        elif self.env_type == 'sc2':
            alive_mask = ( (avail_actions[:, :, :, 0] < 1.0) * (th.sum(avail_actions, dim=-1) > 0.0) ).float()
        else:
            raise NotImplementedError

        num_alive_agents = th.sum(alive_mask, dim=-1).float()
        avail_actions = avail_actions.byte()

        if getattr(self.args, "is_observation_normalized", False):
            # NOTE: obs normalizer needs to be in basic_controller
            self.mac.update_rms(batch, alive_mask)
            # NOTE: obs in batch is being updated
            self.normalize_obs(batch, alive_mask)
            # NOTE: state in batch is being updated
            self.normalize_state(batch, mask)

        if self.agent_type == "rnn":
            old_action_logits = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(max_t):
                actor_outs = self.mac.forward(batch, t = t, test_mode=False)
                old_action_logits.append(actor_outs)
            old_action_logits = th.stack(old_action_logits, dim=1)

        elif self.agent_type == "ff":
            old_action_logits = self.mac.forward_ff(batch)

        else:
            raise NotImplementedError

        old_values_before = self.critic(batch).squeeze(dim=-1).detach()
        # append 0's for terminal state value. (Simplifies operations below.)
        old_values_before = th.cat((old_values_before, th.zeros_like(old_values_before[:, 0:1, ...]),), dim=1)
        assert old_values_before.shape[1] == max_t+1, (old_values_before.shape, max_t)
        if getattr(self.args, "is_popart", False):
            old_values_before = self.denormalize_value(old_values_before)

        # expand reward/mask to n_agent copies
        rewards = rewards.repeat(1, 1, self.n_agents)
        terminated = terminated.repeat(1, 1, self.n_agents)
        timed_out = timed_out.repeat(1, 1, self.n_agents)
        expanded_mask = mask.repeat(1, 1, self.n_agents)

        if self.advantage_calc_method == "GAE":
            returns, _ = self._compute_returns_advs(old_values_before, rewards, terminated, timed_out,
                                                    self.args.gamma, self.args.tau)

        else:
            raise NotImplementedError

        ## update the critics
        critic_loss_lst = []
        if getattr(self.args, "is_popart", False):
            returns = self.normalize_value(returns, mask)

        for _ in range(0, self.mini_epochs_critic):
            new_values = self.critic(batch).squeeze()
            critic_loss = 0.5 * ((new_values - returns)**2 * expanded_mask).sum() / expanded_mask.sum()
            self.optimiser_critic.zero_grad()
            critic_loss.backward()
            critic_loss_lst.append(critic_loss)
            self.optimiser_critic.step()

        old_values_after = self.critic(batch).squeeze(dim=-1).detach()
        old_values_after = th.cat((old_values_after, th.zeros_like(old_values_after[:, 0:1, ...]),), dim=1)
        if getattr(self.args, "is_popart", False):
            old_values_after = self.denormalize_value(old_values_after)

        if self.advantage_calc_method == "GAE":
            returns, advantages = self._compute_returns_advs(old_values_after, rewards, terminated, timed_out,
                                                    self.args.gamma, self.args.tau)

        else:
            raise NotImplementedError

        original_advantages = advantages[...]
        if getattr(self.args, "is_advantage_normalized", False):
            # only consider valid advantages
            flat_mask = alive_mask.flatten()
            flat_advantage = advantages.flatten()
            assert flat_mask.shape[0] == flat_advantage.shape[0]
            adv_index = th.nonzero(flat_mask).squeeze()
            valid_adv = flat_advantage[adv_index]
            batch_mean = th.mean(valid_adv, dim=0)
            batch_var = th.var(valid_adv, dim=0)
            flat_advantage = (flat_advantage - batch_mean) / th.sqrt(batch_var + 1e-6) * flat_mask
            bs, ts, _ = advantages.shape
            advantages = flat_advantage.reshape(bs, ts, self.n_agents)

        # no-op (valid only when dead)
        # https://github.com/oxwhirl/smac/blob/013cf27001024b4ce47f9506f2541eca0b247c95/smac/env/starcraft2/starcraft2.py#L499
        old_action_logits =old_action_logits.detach()
        old_meta_data = compute_logp_entropy(old_action_logits, actions, avail_actions)
        old_log_p, old_log_pac = old_meta_data['logp'], old_meta_data['logpac']

        entropy_lst = []
        actor_loss_lst = []

        mask = mask.squeeze(dim=-1)

        ## update the actor
        for _ in range(0, self.mini_epochs_actor):
            if self.agent_type == "rnn":
                action_logits = []
                self.mac.init_hidden(batch.batch_size)
                for t in range(max_t):
                    actor_outs = self.mac.forward(batch, t = t, test_mode=False)
                    action_logits.append(actor_outs)
                action_logits = th.stack(action_logits, dim=1)

            elif self.agent_type == "ff":
                action_logits = self.mac.forward_ff(batch)

            else:
                raise NotImplementedError

            meta_data = compute_logp_entropy(action_logits, actions, avail_actions)
            log_p, log_pac = meta_data['logp'], meta_data['logpac']

            ## TV divergence for all agents
            prob_diff = th.exp(log_p) - th.exp(old_log_p)
            indepent_approxtv = th.max( 0.5 * th.abs(prob_diff).sum(dim=-1) ).detach()
            joint_approxtv = th.max( ( 0.5 * th.abs(prob_diff).sum(dim=-1) ).sum(dim=-1) ).detach()

            # for shared policy, maximize the policy entropy averaged over all agents & episodes
            entropy = (meta_data['entropy'] * alive_mask).sum() / alive_mask.sum()
            entropy_lst.append(entropy)

            prob_ratio = th.clamp(th.exp(log_pac - old_log_pac), 0.0, 16.0)
            pg_loss_unclipped = - advantages * prob_ratio

            pg_loss_clipped = - advantages * th.clamp(prob_ratio, 1 - self.clip_range, 1 + self.clip_range)

            pg_loss = th.sum(th.max(pg_loss_unclipped, pg_loss_clipped) * alive_mask) / alive_mask.sum()
            surr_objective = th.sum(original_advantages * prob_ratio * alive_mask) / alive_mask.sum()

            # Construct overall loss
            actor_loss = pg_loss - self.args.entropy_loss_coeff * entropy
            actor_loss_lst.append(actor_loss)

            self.optimiser_actor.zero_grad()
            actor_loss.backward()
            self.optimiser_actor.step()

        ratios = prob_ratio.detach().cpu().numpy()
        epsilon = np.abs(ratios - 1.0)
        epsilon_sum = np.sum( np.amax(epsilon, axis=(0, 1)) )
        epsilon_max = np.max(epsilon)

        # log stuff
        critic_train_stats["rewards"].append(th.mean(rewards).item())
        critic_train_stats["returns"].append((th.mean(returns)).item())
        critic_train_stats["independent_approx_TV"].append(indepent_approxtv.item())
        critic_train_stats["joint_approx_TV"].append(joint_approxtv.item())
        critic_train_stats["epsilon_sum"].append(epsilon_sum)
        critic_train_stats["epsilon_max"].append(epsilon_max)
        critic_train_stats["ratios_max"].append(np.max(ratios))
        critic_train_stats["ratios_min"].append(np.min(ratios))
        critic_train_stats["ratios_mean"].append(np.mean(ratios))
        critic_train_stats["entropy"].append(th.mean(th.tensor(entropy_lst)).item())
        critic_train_stats["critic_loss"].append(th.mean(th.tensor(critic_loss_lst)).item())
        critic_train_stats["actor_loss"].append(th.mean(th.tensor(actor_loss_lst)).item())

        critic_train_stats["surr_objective"].append(surr_objective.item())
        critic_train_stats["lr"].append(self.optimiser_actor.param_groups[0]["lr"])
        self.scheduler_actor.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k,v in critic_train_stats.items():
                self.logger.log_stat(k, np.mean(np.array(v)), t_env)
            self.log_stats_t = t_env

    def _compute_returns_advs(self, _values, _rewards, _terminated, _timed_out, gamma, tau):
        returns = th.zeros_like(_rewards)
        advs = th.zeros_like(_rewards)
        lastgaelam = th.zeros_like(_rewards[:, 0]).flatten()
        ts = _rewards.size(1)

        bad_mask = (1.0 - _timed_out)

        for t in reversed(range(ts)):
            nextnonterminal = (1 - _terminated[:, t]).flatten()
            nextvalues = _values[:, t+1].flatten()

            reward_t = _rewards[:, t].flatten()
            delta = reward_t + gamma * nextvalues * nextnonterminal  - _values[:, t].flatten()
            lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam

            if self.bootstrap_timeouts:
                # lastgaelam must be set to 0 for states that timeout so that prior state computes advs[:, t] = delta + 0
                # Note that this will still calculate an invalid (0) return/adv in advs[:, t] for the states that timeout, but
                #   this will be masked off anyway.
                lastgaelam = bad_mask[:, t] * lastgaelam

            advs[:, t] = lastgaelam.view(_rewards[:, t].shape)

        returns = advs + _values[:, :-1]

        return returns, advs

    def cuda(self):
        self.mac.cuda()
        if hasattr(self, "critic"):
            self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mac.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.optimiser_actor.state_dict(), "{}/opt_actor.th".format(path))
        th.save(self.optimiser_critic.state_dict(), "{}/opt_critic.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.mac.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        if hasattr(self, "target_critic"):
            self.target_critic.load_state_dict(self.critic.state_dict())
        self.optimiser_actor.load_state_dict("{}/opt_actor.th".format(path))
        self.optimiser_critic.load_state_dict("{}/opt_critic.th".format(path))
