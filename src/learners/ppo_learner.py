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
        self.advantage_calc_method = getattr(self.args, "advantage_calc_method", "GAE")

        self.clip_loss = getattr(self.args, "clip_loss", True)
        self.agent_type = getattr(self.args, "agent", None)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        critic_train_stats = defaultdict(lambda: [])

        actions = batch["actions"][:, :-1].cuda()
        rewards = batch["reward"][:, :-1].cuda()
        filled_mask = batch['filled'].float().cuda()
        rewards = rewards.repeat(1, 1, self.n_agents)

        # right shift terminated flag, to be aligned with openai/baseline setups
        terminated = batch["terminated"].float()
        terminated[:, 1:] = terminated[:, :-1].clone()
        # if the episode terminates on the first step,
        # we have to explicitly fill the first entry with a 0
        terminated[:, 0] = 0.0
        filled_mask = filled_mask * (1 - terminated)
        filled_mask = filled_mask[:, :-1]
        mask = filled_mask.repeat(1, 1, self.n_agents)

        if getattr(self.args, "is_observation_normalized", False):
            obs_mask = mask[...].clone()
            obs_mask = obs_mask.flatten()

            obs = batch["obs"][:, :-1].cuda()
            obs_index = th.nonzero(obs_mask).squeeze()
            obs = obs.reshape((-1, obs.shape[-1]))[obs_index]
            self.mac.update_rms(obs)
            self.critic.update_rms(obs)

        if self.agent_type == "cnn":
            action_logits = self.mac.forward_cnn(batch)
            action_logits = action_logits[:, :-1]
        elif self.agent_type == "rnn":
            action_logits = th.zeros((batch.batch_size, rewards.shape[1], self.n_agents, self.n_actions)).cuda()
            self.mac.init_hidden(batch.batch_size)
            for t in range(rewards.shape[1]):
                action_logits[:, t] = self.mac.forward(batch, t = t, test_mode=False)
        else:
            raise NotImplementedError

        action_probs = th.nn.functional.log_softmax(action_logits, dim=-1)
        old_log_pac = th.gather(action_probs, dim=3, index=actions).squeeze(3).detach()

        old_values = self.critic(batch).squeeze().detach()

        if self.advantage_calc_method == "GAE":
            returns, advantages = self._compute_returns_advs(old_values, rewards, terminated, 
                                                                self.args.gamma, self.args.tau)
        elif self.advantage_calc_method == "TD_Error":
            returns = rewards + self.args.gamma * old_values[:, 1:] * (1 - terminated[:, 1:]) # terminated has been shifted
            advantages = returns - old_values[:, :-1]
        else:
            raise NotImplementedError

        if getattr(self.args, "is_advantage_normalized", False):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        ## feeding and training
        approxkl_lst = [] 
        entropy_lst = [] 
        critic_loss_lst = []
        actor_loss_lst = []

        target_kl = 0.2

        for _ in range(0, self.mini_epochs_actor):

            if self.agent_type == "cnn":
                logits = self.mac.forward_cnn(batch)
                logits = logits[:, :-1]
            elif self.agent_type == "rnn":
                logits = []
                self.mac.init_hidden(batch.batch_size)
                for t in range(rewards.shape[1]):
                    logits.append( self.mac.forward(batch, t = t, test_mode=False) )
                logits = th.transpose(th.stack(logits), 0, 1)
            else:
                raise NotImplementedError

            pacs = th.nn.functional.log_softmax(logits, dim=-1)
            log_pac = th.gather(pacs, dim=3, index=actions).squeeze(-1)

            with th.no_grad():
                approxkl = 0.5 * th.sum((log_pac - old_log_pac)**2 * mask) / th.sum(mask)
                approxkl_lst.append(approxkl)
                if approxkl > 1.5 * target_kl:
                    break

            entropy = th.sum( th.sum(-1.0 * pacs * th.exp(pacs), dim=-1) * mask.squeeze() ) / th.sum(mask.squeeze())
            entropy_lst.append(entropy)

            prob_ratio = th.clamp(th.exp(log_pac - old_log_pac), 0.0, 16.0)
            pg_loss_unclipped = - advantages * prob_ratio
            if self.clip_loss:
                pg_loss_clipped = - advantages * th.clamp(prob_ratio,
                                                1 - self.args.ppo_policy_clip_param,
                                                1 + self.args.ppo_policy_clip_param)

                pg_loss = th.sum(th.max(pg_loss_unclipped, pg_loss_clipped) * mask) / th.sum(mask)
            else:
                pg_loss = th.sum(-log_pac * advantages * mask) / th.sum(mask)
            # Construct overall loss
            actor_loss = pg_loss - self.args.entropy_loss_coeff * entropy
            actor_loss_lst.append(actor_loss)

            self.optimiser_actor.zero_grad()
            actor_loss.backward()
            self.optimiser_actor.step()

        for _ in range(0, self.mini_epochs_critic):
            new_values = self.critic(batch).squeeze()
            new_values = new_values[:, :-1]
            critic_loss = 0.5 * th.sum((new_values - returns)**2 * mask) / th.sum(mask)
            self.optimiser_critic.zero_grad()
            critic_loss.backward()
            critic_loss_lst.append(critic_loss)
            self.optimiser_critic.step()

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
