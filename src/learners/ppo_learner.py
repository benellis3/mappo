import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.conv1d_critic import Conv1dCritic
# from modules.critics.independent import IndependentCritic
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from utils.rl_utils import build_td_lambda_targets

# from torch.distributions import Categorical

class PPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.ppo_policy_clip_params = args.ppo_policy_clip_param

        if getattr(self.args, "is_separate_actor_critic", False):
            self.is_separate_actor_critic = True
            self.conv1d_critic = Conv1dCritic(scheme, args)
            self.critic_params = list(self.conv1d_critic.parameters())
            self.optimiser_actor = Adam(params=self.agent_params, lr=args.lr_actor)
            self.optimiser_critic = Adam(params=self.critic_params, lr=args.lr_critic)
        else:
            self.is_separate_actor_critic = False
            self.params = self.agent_params
            self.optimiser = Adam(params=self.params, lr=args.lr)

        # self.optimiser = RMSprop(params=self.params,
        #                          lr=args.lr, alpha=args.optim_alpha, 
        #                          eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.mini_epochs_num = getattr(self.args, "mini_epochs_num", 1)
        self.mini_batches_num = getattr(self.args, "mini_batches_num", 4)

    def make_transitions(self, batch: EpisodeBatch):
        actions = batch["actions"][:, :-1].cuda()
        avail_actions = batch["avail_actions"][:, :-1].cuda()
        rewards = batch["reward"][:, :-1].cuda()
        mask = batch["filled"].float().cuda()
        obs = batch["obs"][:, :-1].cuda()
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
                self.conv1d_critic.update_rms(obs)

        old_values = th.zeros((batch.batch_size, rewards.shape[1]+1, self.n_agents)).cuda()
        action_probs = th.zeros((batch.batch_size, rewards.shape[1] + 1, self.n_agents, self.n_actions)).cuda()

        for t in range(rewards.shape[1] + 1):
            action_probs[:, t] = self.mac.forward(batch, t = t, test_mode=True)
            if self.is_separate_actor_critic:
                old_values[:, t] = self.conv1d_critic(batch, t)
            else:
                old_values[:, t] = self.mac.other_outs["values"].view(batch.batch_size, self.n_agents)

        old_values[mask == 0.0] = 0.0

        action_probs = action_probs[:, :-1]
        pi_taken = th.gather(action_probs, dim=3, index=actions).squeeze(3)

        returns, advs = self._compute_returns_advs(old_values, rewards, terminated, 
                                                   self.args.gamma, self.args.tau)
        old_values = old_values[:, :-1]
        
        mask = mask[:, :-1]
        mask = mask.flatten()
        index = th.nonzero(mask).squeeze()
        actions = actions.flatten()[index]
        avail_actions = avail_actions.reshape((-1, avail_actions.shape[-1]))[index]
        pi_taken = pi_taken.flatten()[index]
        actions_neglogp = -th.log(pi_taken + 1e-6)
        returns = returns.flatten()[index]
        values = old_values.flatten()[index]
        advantages = advs.flatten()[index]
        rewards = rewards.flatten()[index]

        transitions = {}
        transitions.update({"num": int(th.sum(mask))})
        transitions.update({"actions": actions.detach()})
        transitions.update({"avail_actions": avail_actions.detach()})
        transitions.update({"actions_neglogp": actions_neglogp.detach()})
        transitions.update({"returns": returns.detach()})
        transitions.update({"values": values.detach()})
        transitions.update({"advantages": advantages.detach()})
        transitions.update({"obs": obs.detach()})
        transitions.update({"rewards": rewards.detach()})

        return transitions

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        critic_train_stats = defaultdict(lambda: [])

        transitions = self.make_transitions(batch)
        num             = transitions["num"]
        actions         = transitions["actions"]
        avail_actions   = transitions["avail_actions"]
        actions_neglogp = transitions["actions_neglogp"]
        returns         = transitions["returns"]
        values          = transitions["values"]
        advantages      = transitions["advantages"]
        obs             = transitions["obs"]
        rewards         = transitions["rewards"]

        if getattr(self.args, "is_advantage_normalized", False):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        ## feeding and training
        approxkl_lst = [] 
        entropy_lst = [] 
        critic_loss_lst = []
        actor_loss_lst = []
        loss_lst = []

        for _ in range(0, self.args.mini_epochs_num):
            rnd_idx = np.random.permutation(num)
            rnd_step = len(rnd_idx) // self.mini_batches_num
            for j in range(0, self.mini_batches_num):
                curr_idx = rnd_idx[j*rnd_step : j*rnd_step+rnd_step]

                mb_actions      = actions[curr_idx]
                mb_avail_actions= avail_actions[curr_idx]
                mb_adv          = advantages[curr_idx]
                mb_ret          = returns[curr_idx]
                mb_old_values   = values[curr_idx]
                mb_old_neglogp  = actions_neglogp[curr_idx]
                mb_obs          = obs[curr_idx]

                mb_logp = self.mac.forward_obs(mb_obs, mb_avail_actions)
                if self.is_separate_actor_critic:
                    mb_values = self.conv1d_critic.forward_obs(mb_obs)
                else:
                    mb_values = self.mac.other_outs["values"]

                mb_neglogp = -th.gather(mb_logp, dim=1, index=mb_actions.unsqueeze(-1)).squeeze(-1)

                with th.no_grad():
                    approxkl = 0.5 * ((mb_old_neglogp - mb_neglogp)**2).mean()
                    approxkl_lst.append(approxkl)

                entropy = -1.0 * (mb_logp * th.exp(mb_logp)).sum(dim=-1).mean()
                entropy_lst.append(entropy)

                prob_ratio = th.clamp(th.exp(mb_old_neglogp - mb_neglogp), 0.0, 16.0)

                pg_loss_unclipped = - mb_adv * prob_ratio
                pg_loss_clipped = - mb_adv * th.clamp(prob_ratio,
                                                    1 - self.args.ppo_policy_clip_param,
                                                    1 + self.args.ppo_policy_clip_param)

                pg_loss = th.mean(th.max(pg_loss_unclipped, pg_loss_clipped))

                # Construct overall loss
                actor_loss = pg_loss - self.args.entropy_loss_coeff * entropy
                actor_loss_lst.append(actor_loss)

                critic_loss = 0.5 * th.mean((mb_values - mb_ret)**2)
                critic_loss_lst.append(critic_loss)

                loss = actor_loss + self.args.critic_loss_coeff * critic_loss
                loss_lst.append(loss)

                if getattr(self.args, "is_separate_actor_critic", False):
                    self.optimiser_actor.zero_grad()
                    actor_loss.backward()
                    self.optimiser_actor.step()

                    self.optimiser_critic.zero_grad()
                    critic_loss.backward()
                    self.optimiser_critic.step()

                else:
                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()

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
        if hasattr(self, "conv1d_critic"):
            self.conv1d_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mac.critic.state_dict(), "{}/critic.th".format(path))
        if getattr(self.args, "is_separate_actor_critic", False):
            th.save(self.optimiser_actor.state_dict(), "{}/opt_actor.th".format(path))
            th.save(self.optimiser_critic.state_dict(), "{}/opt_critic.th".format(path))
        else:
            th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.mac.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        if hasattr(self, "target_critic"):
            self.target_critic.load_state_dict(self.critic.state_dict())
        if getattr(self.args, "is_separate_actor_critic", False):
            self.optimiser_actor.load_state_dict("{}/opt_actor.th".format(path))
            self.optimiser_critic.load_state_dict("{}/opt_critic.th".format(path))
        else:
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
