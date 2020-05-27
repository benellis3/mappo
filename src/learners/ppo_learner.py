import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.conv1d_critic import Conv1dCritic
# from modules.critics.independent import IndependentCritic
import torch as th
from torch.optim import RMSprop
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
        self.ppo_value_clip_params = args.ppo_value_clip_param

        # self.conv1d_critic = Conv1dCritic(scheme, args)
        # self.critic_params = list(self.conv1d_critic.parameters())
        # self.params = self.agent_params + self.critic_params

        self.params = self.agent_params
        self.optimiser = RMSprop(params=self.params,
                                 lr=args.agent_lr, alpha=args.optim_alpha, 
                                 eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.mini_epochs_num = getattr(self.args, "mini_epochs_num", 1)
        self.mini_batches_num = getattr(self.args, "mini_batches_num", 1)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        critic_train_stats = defaultdict(lambda: [])

        ## preparing data
        actions = batch["actions"][:, :-1]
        rewards= batch["reward"][:, :-1]
        mask = batch["filled"].float()

        # right shift terminated flag, to be aligned with openai/baseline setups
        terminated = batch["terminated"].float()
        terminated[:, 1:] = terminated[:, :-1].clone()
        mask = mask * (1 - terminated)

        mask = mask.repeat(1, 1, self.n_agents)
        mask_probs = mask.unsqueeze(-1).repeat(1, 1, 1, self.n_actions)
        actions = actions.detach()

        old_values = th.zeros((batch.batch_size, rewards.shape[1]+1, self.n_agents))
        action_probs = th.zeros((batch.batch_size, rewards.shape[1] + 1, self.n_agents, self.n_actions))

        for t in range(rewards.shape[1] + 1):
            # old_values[:, t] = self.conv1d_critic(batch, t)
            action_probs[:, t] = self.mac.forward(batch, t = t)
            old_values[:, t] = self.mac.other_outs["values"].view(batch.batch_size, self.n_agents)

        old_values[mask == 0.0] = 0.0
        action_probs[mask_probs == 0.0] = 1.0

        mask = mask[:, :-1]
        mask_probs = mask_probs[:, :-1]

        action_probs = action_probs[:, :-1]
        pi_taken = th.gather(action_probs, dim=3, index=actions).squeeze(3)
        old_neglogp = -th.log(pi_taken).detach()

        returns, advs = self._compute_returns_advs(old_values, rewards, terminated, 
                                                     self.args.gamma, self.args.tau)
        old_values = old_values[:, :-1]
        
        old_values = old_values.squeeze(dim=-1).detach()
        returns = returns.squeeze(dim=-1).detach()
        advs = advs.squeeze(dim=-1).detach()

        if getattr(self.args, "is_advantage_normalized", False):
            advs = (advs - advs.mean()) / (advs.std() + 1e-5)
            advs[mask == 0.0] = 0.0

        ## feeding and training
        approxkl_lst = []
        entropy_lst = []
        critic_loss_lst = []
        actor_loss_lst = []
        loss_lst = []
        grad_norm_lst = []

        mb_idx = np.random.permutation(batch.batch_size)
        ts_idx = np.random.permutation(batch.max_seq_length - 1)

        mb_step = len(mb_idx) // self.mini_epochs_num
        ts_step = len(ts_idx) // self.mini_batches_num

        for i in range(0, self.args.mini_epochs_num):
            curr_mb_idx = mb_idx[i*mb_step : i*mb_step + mb_step]
            for j in range(0, self.args.mini_batches_num):
                curr_ts_idx = ts_idx[j*ts_step : j*ts_step+ts_step]

                mb_adv = advs[curr_mb_idx][:, curr_ts_idx]
                mb_ret = returns[curr_mb_idx][:, curr_ts_idx]
                mb_old_values = old_values[curr_mb_idx][:, curr_ts_idx]
                mb_old_neglogp = old_neglogp[curr_mb_idx][:, curr_ts_idx]
                mb_mask = mask[curr_mb_idx][:, curr_ts_idx]

                values = th.zeros((batch.batch_size, rewards.shape[1], self.n_agents))
                mac_out = th.zeros((batch.batch_size, rewards.shape[1], self.n_agents, self.n_actions))

                for t in range(rewards.shape[1]):
                    # values[:, t] = self.conv1d_critic(batch, t)
                    mac_out[:, t] = self.mac.forward(batch, t, updating_rms=True)
                    values[:, t] = self.mac.other_outs["values"].view(batch.batch_size, self.n_agents)

                values[mask == 0.0] = 0.0
                mac_out[mask_probs == 0.0] = 1.0

                new_pi_taken = th.gather(mac_out, dim=3, index=actions).squeeze(3)
                neglogp = -th.log(new_pi_taken)
                mb_neglogp = neglogp[curr_mb_idx][:, curr_ts_idx]

                mb_values = values[curr_mb_idx][:, curr_ts_idx]
                actor_mask = mb_mask.expand_as(mb_neglogp)

                with th.no_grad():
                    approxkl = 0.5 * ((old_neglogp - neglogp)**2 * mask).mean()
                    approxkl_lst.append(approxkl)

                entropy_matrix = -1.0 * (mac_out * th.log(mac_out)).sum(dim=-1)
                entropy = th.sum(entropy_matrix * mask)/( mask.sum() + 1e-5)
                entropy_lst.append(entropy)

                prob_ratio = th.clamp(th.exp(mb_old_neglogp - mb_neglogp), 0.0, 16.0)

                pg_loss_unclipped = - mb_adv * prob_ratio
                pg_loss_clipped = - mb_adv * th.clamp(prob_ratio,
                                                    1 - self.args.ppo_policy_clip_param,
                                                    1 + self.args.ppo_policy_clip_param)

                pg_loss = th.sum(th.max(pg_loss_unclipped, pg_loss_clipped) * actor_mask) \
                                     / ( actor_mask.sum() + 1e-5)

                # Construct overall loss
                actor_loss = pg_loss - self.args.entropy_loss_coeff * entropy
                actor_loss_lst.append(actor_loss)

                critic_mask = mb_mask.expand_as(mb_adv)

                # Construct critic loss
                clipped_values = mb_old_values + th.clamp(mb_values - mb_old_values,
                                                        -self.ppo_value_clip_params,
                                                        +self.ppo_value_clip_params)

                # have to extend rewards to each agent if we have independent learning setting
                critic_loss_clipped = (clipped_values - mb_ret)**2
                critic_loss_raw = (mb_values - mb_ret)**2
                critic_loss = 0.5 * th.sum(th.max(critic_loss_raw, critic_loss_clipped) * critic_mask) \
                                                 / ( critic_mask.sum() + 1e-5)

                critic_loss_lst.append(critic_loss)

                loss = actor_loss + critic_loss
                loss_lst.append(loss)

                self.optimiser.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.params, 
                                                        self.args.grad_norm_clip)
                grad_norm_lst.append(grad_norm if isinstance(grad_norm, float) else grad_norm.item()) #pytorch1.5 fix
                self.optimiser.step()

        # log stuff
        critic_train_stats["values"].append((th.sum(old_values*mask)/mask.sum()).item())
        critic_train_stats["rewards"].append((th.mean(rewards)).item())
        critic_train_stats["returns"].append((th.sum(returns*mask)/mask.sum()).item())
        critic_train_stats["approx_KL"].append(th.mean(th.tensor(approxkl_lst)).item())
        critic_train_stats["entropy"].append(th.mean(th.tensor(entropy_lst)).item())
        critic_train_stats["critic_loss"].append(th.mean(th.tensor(critic_loss_lst)).item())
        critic_train_stats["actor_loss"].append(th.mean(th.tensor(actor_loss_lst)).item())
        critic_train_stats["loss"].append(th.mean(th.tensor(loss_lst)).item())
        critic_train_stats["advantage"].append((th.sum(advs*mask)/mask.sum()).mean().item())
        critic_train_stats["grad_norm"].append(np.mean(grad_norm_lst))

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
            self.mac.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mac.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.mac.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        if hasattr(self, "target_critic"):
            self.target_critic.load_state_dict(self.critic.state_dict())
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))