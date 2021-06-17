import numpy as np
import torch as th
from torch.optim import Adam
from collections import defaultdict

from components.episode_buffer import EpisodeBatch
from modules.critics import critic_REGISTRY
from components.running_mean_std import RunningMeanStd

class DecentralPPOLearner:
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
        self.optimiser_actor = Adam(params=self.agent_params, lr=args.lr_actor, eps=args.optim_eps)
        self.optimiser_critic = Adam(params=self.critic_params, lr=args.lr_critic, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.mini_epochs_actor = getattr(self.args, "mini_epochs_actor", 4)
        self.mini_epochs_critic = getattr(self.args, "mini_epochs_critic", 4)
        self.advantage_calc_method = getattr(self.args, "advantage_calc_method", "GAE")
        self.agent_type = getattr(self.args, "agent", None)

        self.kl_clipping_mode = getattr(self.args, "kl_clipping_mode", "default")
        assert self.kl_clipping_mode in ['default', 'epoch_adaptive']

        if getattr(self.args, "is_observation_normalized", False):
            # need to normalize state
            self.state_rms = RunningMeanStd()

    def normalize_state(self, batch, mask):
        bs, ts = batch.batch_size, batch.max_seq_length-1

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
        batch.data.transition_data['state'][:, :-1] = (batch['state'][:, :-1] - state_mean) / (state_var + 1e-6) * expanded_mask

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
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
        avail_actions = batch['avail_actions'][:, :-1].cuda()
        alive_mask = ( (avail_actions[:, :, :, 0] != 1.0) * (th.sum(avail_actions, dim=-1) != 0.0) ).float()
        num_alive_agents = th.sum(alive_mask, dim=-1).float() * mask

        if getattr(self.args, "is_observation_normalized", False):
            # NOTE: obs has already been normalized in basic_controller, only need to update rms
            self.mac.update_rms(batch, alive_mask)
            # NOTE: state are being updated
            self.normalize_state(batch, mask)

        if self.agent_type == "rnn":
            action_logits = th.zeros((batch.batch_size, rewards.shape[1], self.n_agents, self.n_actions)).cuda()
            self.mac.init_hidden(batch.batch_size)
            for t in range(rewards.shape[1]):
                action_logits[:, t] = self.mac.forward(batch, t = t, test_mode=False)

        else:
            raise NotImplementedError

        old_values = self.critic(batch).squeeze(dim=-1).detach()

        # expand reward to n_agent copies
        rewards = rewards.repeat(1, 1, self.n_agents)
        terminated = terminated.unsqueeze(dim=-1).repeat(1, 1, self.n_agents)
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, self.n_agents)

        if self.advantage_calc_method == "GAE":
            returns, _ = self._compute_returns_advs(old_values, rewards, terminated, 
                                                    self.args.gamma, self.args.tau)

        else:
            raise NotImplementedError

        ## update the critics
        critic_loss_lst = []
        for _ in range(0, self.mini_epochs_critic):
            new_values = self.critic(batch).squeeze()
            new_values = new_values[:, :-1]
            critic_loss = 0.5 * th.sum((new_values - returns)**2 * mask_expanded) / th.sum(mask_expanded)
            self.optimiser_critic.zero_grad()
            critic_loss.backward()
            critic_loss_lst.append(critic_loss)
            self.optimiser_critic.step()

        ## compute advantage
        old_values = self.critic(batch).squeeze(dim=-1).detach()
        if self.advantage_calc_method == "GAE":
            _, advantages = self._compute_returns_advs(old_values, rewards, terminated, 
                                                       self.args.gamma, self.args.tau)

        else:
            raise NotImplementedError

        # no-op (valid only when dead)
        # https://github.com/oxwhirl/smac/blob/013cf27001024b4ce47f9506f2541eca0b247c95/smac/env/starcraft2/starcraft2.py#L499
        action_probs = th.nn.functional.log_softmax(action_logits, dim=-1)
        old_log_pac = th.gather(action_probs, dim=3, index=actions).squeeze(3).detach()

        # joint probability
        central_old_log_pac = th.sum(old_log_pac, dim=-1)

        approxkl_lst = [] 
        entropy_lst = [] 
        actor_loss_lst = []

        target_kl = 0.2
        if getattr(self.args, "is_advantage_normalized", False):
            # only consider valid advantages
            flat_mask = mask_expanded.flatten()
            flat_advantage = advantages.flatten()
            assert flat_mask.shape[0] == flat_advantage.shape[0]
            adv_index = th.nonzero(flat_mask).squeeze()
            valid_adv = flat_advantage[adv_index]
            batch_mean = th.mean(valid_adv, dim=0)
            batch_var = th.var(valid_adv, dim=0)
            flat_advantage = (flat_advantage - batch_mean) / (batch_var + 1e-6) * flat_mask
            bs, ts, _ = advantages.shape
            advantages = flat_advantage.reshape(bs, ts, self.n_agents)

        ## update the actor
        for _ in range(0, self.mini_epochs_actor):
            if self.agent_type == "rnn":
                logits = []
                self.mac.init_hidden(batch.batch_size)
                for t in range(rewards.shape[1]):
                    logits.append( self.mac.forward(batch, t = t, test_mode=False) )
                logits = th.transpose(th.stack(logits), 0, 1)

            else:
                raise NotImplementedError

            log_prob_dist = th.nn.functional.log_softmax(logits, dim=-1)
            log_pac = th.gather(log_prob_dist, dim=3, index=actions).squeeze(-1)

            # joint probability
            central_log_pac = th.sum(log_pac, dim=-1)

            with th.no_grad():
                # KL divergence for all agents
                approxkl = 0.5 * th.sum((central_log_pac - central_old_log_pac)**2 * mask) / th.sum(mask)
                approxkl_lst.append(approxkl)
                if approxkl > 1.5 * target_kl:
                    break

            # for shared policy, maximize the policy entropy averaged over all agents & episodes
            # consider entropy for only alive agents
            # log_prob_dist: n_batch * n_timesteps * n_agents * n_actions
            entropy_all_agents = th.sum(-1.0 * log_prob_dist * th.exp(log_prob_dist), dim=-1)
            entropy = th.sum( entropy_all_agents * alive_mask ) / th.sum(alive_mask)
            entropy_lst.append(entropy)

            prob_ratio = th.clamp(th.exp(log_pac - old_log_pac), 0.0, 16.0)

            pg_loss_unclipped = - advantages * prob_ratio
            pg_loss_clipped = - advantages * th.clamp(prob_ratio,
                                                1 - self.args.ppo_policy_clip_param,
                                                1 + self.args.ppo_policy_clip_param)

            pg_loss = th.sum(th.max(pg_loss_unclipped, pg_loss_clipped) * mask_expanded) / th.sum(mask_expanded)

            # Construct overall loss
            actor_loss = pg_loss - self.args.entropy_loss_coeff * entropy
            actor_loss_lst.append(actor_loss)

            self.optimiser_actor.zero_grad()
            actor_loss.backward()
            self.optimiser_actor.step()

        # log stuff
        critic_train_stats["rewards"].append(th.mean(rewards).item())
        critic_train_stats["returns"].append((th.mean(returns)).item())
        critic_train_stats["approx_KL"].append(th.mean(th.tensor(approxkl_lst)).item())
        critic_train_stats["entropy"].append(th.mean(th.tensor(entropy_lst)).item())
        critic_train_stats["critic_loss"].append(th.mean(th.tensor(critic_loss_lst)).item())
        critic_train_stats["actor_loss"].append(th.mean(th.tensor(actor_loss_lst)).item())

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k,v in critic_train_stats.items():
                self.logger.log_stat(k, np.mean(np.array(v)), t_env)
            self.log_stats_t = t_env

    def _compute_returns_advs(self, _values, _rewards, _terminated, gamma, tau):
        returns = th.zeros_like(_rewards)
        advs = th.zeros_like(_rewards)
        lastgaelam = th.zeros_like(_rewards[:, 0])
        ts = _rewards.size(1)

        for t in reversed(range(ts)):
            nextnonterminal = 1.0 - _terminated[:, t]
            nextvalues = _values[:, t+1]

            reward_t = _rewards[:, t]
            delta = reward_t + gamma * nextvalues * nextnonterminal  - _values[:, t]
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
        self.mac.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        if hasattr(self, "target_critic"):
            self.target_critic.load_state_dict(self.critic.state_dict())
        self.optimiser_actor.load_state_dict("{}/opt_actor.th".format(path))
        self.optimiser_critic.load_state_dict("{}/opt_critic.th".format(path))
