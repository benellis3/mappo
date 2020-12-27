import copy
from components.episode_buffer import EpisodeBatch
from modules.critics import critic_REGISTRY
import numpy as np
import torch as th
import torch.nn.functional as F
from collections import defaultdict
from torch.optim import RMSprop, Adam
from modules.critics import critic_REGISTRY
from components.episode_buffer import EpisodeBatch

class PPOLearner:
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
        self.optimiser_actor = RMSprop(params=self.agent_params, lr=args.lr_actor,
                                       alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_critic = RMSprop(params=self.critic_params, lr=args.lr_critic,
                                       alpha=args.optim_alpha, eps=args.optim_eps)

        self.central_v = getattr(self.args, "is_central_value", False)
        self.actor_critic_mode = getattr(self.args, "actor_critic_mode", False)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.mini_epochs_actor = getattr(self.args, "mini_epochs_actor", 4)
        self.mini_epochs_critic = getattr(self.args, "mini_epochs_critic", 4)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        critic_train_stats = defaultdict(lambda: [])

        actions = batch["actions"][:, :-1]
        rewards = batch["reward"][:, :-1]
        rewards = rewards.repeat(1, 1, self.n_agents)

        # right shift terminated flag, to be aligned with openai/baseline setups
        terminated = batch["terminated"].float()
        terminated[:, 1:] = terminated[:, :-1].clone()
        mask = mask * (1 - terminated)
        mask = mask.repeat(1, 1, self.n_agents)

        obs_mask = mask[:, :-1].clone()
        obs_mask = obs_mask.flatten()
        obs_index = th.nonzero(obs_mask).squeeze()
        obs = obs.reshape((-1, obs.shape[-1]))[obs_index]

        if getattr(self.args, "is_observation_normalized", False):
            self.mac.update_rms(obs)
            if self.is_separate_actor_critic:
                self.critic.update_rms(obs)

        old_values = th.zeros((batch.batch_size, rewards.shape[1]+1, self.n_agents))
        action_probs = th.zeros((batch.batch_size, rewards.shape[1] + 1, self.n_agents, self.n_actions))

        action_probs = action_probs[:, :-1]
        old_log_pac = th.log(th.gather(action_probs, dim=3, index=actions).squeeze(3))

        for t in range(rewards.shape[1] + 1):
            action_probs[:, t] = self.mac.forward(batch, t = t, test_mode=False)
            old_values[:, t] = self.critic(batch, t).squeeze()
        old_values, action_probs = old_values.detach(), action_probs.detach()

        returns, advantages = self._compute_returns_advs(old_values, rewards, terminated, 
                                                            self.args.gamma, self.args.tau)
        if getattr(self.args, "is_advantage_normalized", False):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        ## feeding and training
        approxkl_lst = [] 
        entropy_lst = [] 
        critic_loss_lst = []
        actor_loss_lst = []
        loss_lst = []

        target_kl = 0.2

        for _ in range(0, self.mini_epochs_actor):
            for t in range(rewards.shape[1] + 1):
                pac[:, t] = self.mac.forward(batch, t = t, test_mode=False)
            log_pac = th.log(th.gather(pac, dim=1, index=actions.unsqueeze(-1)).squeeze(-1))

            with th.no_grad():
                approxkl = 0.5 * ((log_pac - old_log_pac)**2).mean()
                if approxkl > 1.5 * target_kl:
                    break

            entropy = -1.0 * (log_pac * th.exp(log_pac)).sum(dim=-1).mean()
            entropy_lst.append(entropy)

            prob_ratio = th.clamp(th.exp(log_pac - old_log_pac), 0.0, 16.0)

            pg_loss_unclipped = - advantages * prob_ratio
            pg_loss_clipped = - advantages * th.clamp(prob_ratio,
                                                1 - self.args.ppo_policy_clip_param,
                                                1 + self.args.ppo_policy_clip_param)

            pg_loss = th.mean(th.max(pg_loss_unclipped, pg_loss_clipped))

            # Construct overall loss
            actor_loss = pg_loss - self.args.entropy_loss_coeff * entropy
            actor_loss_lst.append(actor_loss)

            self.optimiser_actor.zero_grad()
            actor_loss.backward()
            self.optimiser_actor.step()

        for _ in range(0, self.mini_epochs_critic):
            for t in range(rewards.shape[1] + 1):
                new_values[:, t] = self.critic(batch, t).squeeze()
            critic_loss = 0.5 * th.mean((new_values - returns)**2)
            self.optimiser_critic.zero_grad()
            critic_loss.backward()
            self.optimiser_critic.step()

        # log stuff
        critic_train_stats["values"].append((th.mean(values)).item())
        critic_train_stats["rewards"].append(th.mean(rewards).item())
        critic_train_stats["returns"].append((th.mean(returns)).item())
        critic_train_stats["approx_KL"].append(th.mean(th.tensor(approxkl_lst)).item())
        critic_train_stats["entropy"].append(th.mean(th.tensor(entropy_lst)).item())
        critic_train_stats["critic_loss"].append(th.mean(th.tensor(critic_loss_lst)).item())
        critic_train_stats["actor_loss"].append(th.mean(th.tensor(actor_loss_lst)).item())
        critic_train_stats["loss"].append(th.mean(th.tensor(loss_lst)).item())

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k,v in critic_train_stats.items():
                self.logger.log_stat(k, np.mean(np.array(v)), t_env)
            self.log_stats_t = t_env

    def _compute_returns_advs(self, _values, _rewards, _terminated, gamma, tau):
        # _values
        returns = th.zeros_like(_values)
        advs = th.zeros_like(_values)
        lastgaelam = th.zeros_like(_values[:, 0])

        for t in reversed(range(_rewards.size(1))):
            nextnonterminal = 1.0 - _terminated[:, t+1]
            nextvalues = _values[:, t+1]

            nextnonterminal = nextnonterminal.expand_as(nextvalues).float()
            reward_t = _rewards[:, t].expand_as(nextvalues)

            delta = reward_t + gamma * nextvalues * nextnonterminal  - _values[:, t]
            advs[:, t] = lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam

        returns = advs + _values

        return returns[:, :-1], advs[:, :-1]

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
