import numpy as np
import torch as th
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR
from collections import defaultdict

from components.episode_buffer import EpisodeBatch
from modules.critics import critic_REGISTRY
from components.running_mean_std import RunningMeanStd


def compute_logp_entropy(logits, actions, masks):
    masked_logits = th.where(
        masks, logits, th.tensor(th.finfo(logits.dtype).min).to(logits.device)
    )
    # normalize logits
    masked_logits = masked_logits - masked_logits.logsumexp(dim=-1, keepdim=True)

    probs = th.nn.functional.softmax(masked_logits, dim=-1)
    p_log_p = masked_logits * probs
    p_log_p = th.where(masks, p_log_p, th.tensor(0.0).to(p_log_p.device))
    entropy = -p_log_p.sum(-1)

    logp = th.gather(masked_logits, dim=-1, index=actions)

    result = {
        "logp": th.squeeze(logp),
        "entropy": th.squeeze(entropy),
    }
    return result


class CentralPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.ppo_policy_clip_params = args.ppo_policy_clip_param

        self.critic = critic_REGISTRY[self.args.critic](scheme, args)
        self.critic_params = list(self.critic.parameters())
        self.optimiser_actor = Adam(
            params=self.agent_params, lr=args.lr_actor, eps=args.optim_eps
        )
        self.t = 0
        self.scheduler_actor = MultiplicativeLR(
            optimizer=self.optimiser_actor,
            lr_lambda=lambda epoch: 1.0 - (self.t / args.t_max),
        )
        self.optimiser_critic = Adam(
            params=self.critic_params, lr=args.lr_critic, eps=args.optim_eps
        )

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.mini_epochs_actor = getattr(self.args, "mini_epochs_actor", 4)
        self.mini_epochs_critic = getattr(self.args, "mini_epochs_critic", 4)
        self.advantage_calc_method = getattr(self.args, "advantage_calc_method", "GAE")
        self.agent_type = getattr(self.args, "agent", None)

        self.is_obs_normalized = getattr(self.args, "is_observation_normalized", False)
        self.is_value_normalized = getattr(self.args, "is_value_normalized", False)
        self.is_popart = getattr(self.args, "is_popart", False)

        if self.is_value_normalized and self.is_popart:
            raise ValueError(
                "Either `is_value_normalized` or `is_popart` is specified, but not both."
            )

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
            normalized_returns = (
                (returns - value_mean) / th.sqrt(value_var + 1e-6) * mask
            )

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
        bs, ts = batch.batch_size, batch.max_seq_length - 1

        state = batch["state"][:, :-1].cuda()
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
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, state_mean.shape[-1])
        batch.data.transition_data["state"][:, :-1] = (
            (batch["state"][:, :-1] - state_mean)
            / th.sqrt(state_var + 1e-6)
            * expanded_mask
        )

    def normalize_obs(self, batch, alive_mask):
        bs, ts = batch.batch_size, batch.max_seq_length - 1
        obs_mean = self.mac.obs_rms.mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        obs_var = self.mac.obs_rms.var.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        obs_mean = obs_mean.expand(bs, ts, self.n_agents, -1)
        obs_var = obs_var.expand(bs, ts, self.n_agents, -1)

        expanded_alive_mask = alive_mask.unsqueeze(-1).expand(
            -1, -1, -1, obs_mean.shape[-1]
        )

        # update obs directly in batch
        batch.data.transition_data["obs"][:, :-1] = (
            (batch["obs"][:, :-1] - obs_mean)
            / th.sqrt(obs_var + 1e-6)
            * expanded_alive_mask
        )

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.t = t_env
        max_t = batch.max_seq_length - 1
        critic_train_stats = defaultdict(lambda: [])

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        mask = mask.squeeze(dim=-1)
        terminated = terminated.squeeze(dim=-1)

        ## get dead agents
        # no-op (valid only when dead)
        # https://github.com/oxwhirl/smac/blob/013cf27001024b4ce47f9506f2541eca0b247c95/smac/env/starcraft2/starcraft2.py#L499
        alive_mask = (
            (avail_actions[:, :, :, 0] != 1.0) * (th.sum(avail_actions, dim=-1) != 0.0)
        ).float()
        num_alive_agents = th.sum(alive_mask, dim=-1).float()
        avail_actions = avail_actions.byte()

        if self.is_obs_normalized:
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
                actor_outs = self.mac.forward(batch, t=t, test_mode=False)
                old_action_logits.append(actor_outs)
            old_action_logits = th.stack(old_action_logits, dim=1)

        elif self.agent_type == "ff":
            old_action_logits = self.mac.forward_ff(batch)

        else:
            raise NotImplementedError

        old_values_before = self.critic(batch).squeeze(dim=-1).detach()
        if self.is_value_normalized or self.is_popart:
            old_values_before = self.denormalize_value(old_values_before)

        rewards = rewards.squeeze(dim=-1)

        if self.advantage_calc_method == "GAE":
            returns, advantages = self._compute_returns_advs(
                old_values_before, rewards, terminated, self.args.gamma, self.args.tau
            )
        elif self.advantage_calc_method == "TD":
            returns = (
                rewards + self.args.gamma * (1 - terminated) * old_values_before[:, 1:]
            )
            advantages = returns - old_values_before[:, :-1]
        else:
            raise NotImplementedError

        if getattr(self.args, "is_advantage_normalized", False):
            # only consider valid advantages
            flat_mask = mask.flatten()
            flat_advantage = advantages.flatten()
            assert flat_mask.shape[0] == flat_advantage.shape[0]
            adv_index = th.nonzero(flat_mask).squeeze()
            valid_adv = flat_advantage[adv_index]
            batch_mean = th.mean(valid_adv, dim=0)
            batch_var = th.var(valid_adv, dim=0)

            advantages = (advantages - batch_mean) / th.sqrt(batch_var + 1e-6) * mask

        # action prob
        old_action_logits = old_action_logits.detach()  # detached
        old_meta_data = compute_logp_entropy(old_action_logits, actions, avail_actions)
        old_log_pac = old_meta_data["logp"]

        # joint probability
        central_old_log_pac = th.sum(old_log_pac * alive_mask, dim=-1)

        approxkl_lst = []
        entropy_lst = []
        actor_loss_lst = []

        ## update the actor
        # target_kl = 0.2
        for _ in range(0, self.mini_epochs_actor):
            if self.agent_type == "rnn":
                action_logits = []
                self.mac.init_hidden(batch.batch_size)
                for t in range(max_t):
                    actor_outs = self.mac.forward(batch, t=t, test_mode=False)
                    action_logits.append(actor_outs)
                action_logits = th.stack(action_logits, dim=1)

            elif self.agent_type == "ff":
                action_logits = self.mac.forward_ff(batch)

            else:
                raise NotImplementedError

            meta_data = compute_logp_entropy(action_logits, actions, avail_actions)
            log_pac = meta_data["logp"]

            # joint probability
            central_log_pac = th.sum(log_pac * alive_mask, dim=-1)

            with th.no_grad():
                approxkl = (
                    0.5
                    * th.sum((central_log_pac - central_old_log_pac) ** 2)
                    / alive_mask.sum()
                )
                approxkl_lst.append(approxkl)
                # if approxkl > 1.5 * target_kl:
                #     break

            entropy = (
                th.sum(meta_data["entropy"] * alive_mask) / alive_mask.sum()
            )  # mask out dead agents
            entropy_lst.append(entropy)

            prob_ratio = th.clamp(
                th.exp(central_log_pac - central_old_log_pac), 0.0, 16.0
            )

            pg_loss_unclipped = -advantages * prob_ratio
            pg_loss_clipped = -advantages * th.clamp(
                prob_ratio,
                1 - self.args.ppo_policy_clip_param,
                1 + self.args.ppo_policy_clip_param,
            )

            pg_loss = (
                th.max(pg_loss_unclipped, pg_loss_clipped).sum() / alive_mask.sum()
            )

            # Construct overall loss
            actor_loss = pg_loss - self.args.entropy_loss_coeff * entropy
            actor_loss_lst.append(actor_loss)

            self.optimiser_actor.zero_grad()
            actor_loss.backward()
            self.optimiser_actor.step()

        ## update the critics
        critic_loss_lst = []
        if self.is_value_normalized or self.is_popart:
            returns = self.normalize_value(returns, mask)

        for _ in range(0, self.mini_epochs_critic):
            new_values = self.critic(batch).squeeze()
            new_values = new_values[:, :-1]

            # vf_losses1 = (new_values - returns) ** 2
            # clipped_values = old_values_before[:, :-1] + th.clamp(new_values - old_values_before[:, :-1], \
            #                     -self.args.ppo_policy_clip_param, self.args.ppo_policy_clip_param)
            # vf_losses2 = (clipped_values - returns) ** 2
            # vf_loss = th.sum( th.max(vf_losses1, vf_losses2) * mask ) / mask.sum()

            vf_loss = th.sum((new_values - returns) ** 2 * mask) / mask.sum()

            self.optimiser_critic.zero_grad()
            vf_loss.backward()
            self.optimiser_critic.step()
            critic_loss_lst.append(vf_loss)

        # log stuff
        critic_train_stats["rewards"].append(th.mean(rewards).item())
        critic_train_stats["returns"].append((th.mean(returns)).item())
        critic_train_stats["approx_KL"].append(th.mean(th.tensor(approxkl_lst)).item())
        critic_train_stats["entropy"].append(th.mean(th.tensor(entropy_lst)).item())
        critic_train_stats["critic_loss"].append(
            th.mean(th.tensor(critic_loss_lst)).item()
        )
        critic_train_stats["actor_loss"].append(
            th.mean(th.tensor(actor_loss_lst)).item()
        )
        self.scheduler_actor.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k, v in critic_train_stats.items():
                self.logger.log_stat(k, np.mean(np.array(v)), t_env)
            self.log_stats_t = t_env

    def _compute_returns_advs(self, _values, _rewards, _terminated, gamma, tau):
        returns = th.zeros_like(_rewards)
        advs = th.zeros_like(_rewards)
        lastgaelam = th.zeros_like(_rewards[:, 0])
        ts = _rewards.size(1)

        for t in reversed(range(ts)):
            nextnonterminal = 1.0 - _terminated[:, t]
            nextvalues = _values[:, t + 1]

            reward_t = _rewards[:, t]
            delta = reward_t + gamma * nextvalues * nextnonterminal - _values[:, t]
            advs[:, t] = lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam

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
        self.mac.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        if hasattr(self, "target_critic"):
            self.target_critic.load_state_dict(self.critic.state_dict())
        self.optimiser_actor.load_state_dict("{}/opt_actor.th".format(path))
        self.optimiser_critic.load_state_dict("{}/opt_critic.th".format(path))
