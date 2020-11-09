import copy
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

        self.mini_epochs_num = getattr(self.args, "mini_epochs_num", 1)
        self.mini_batches_num = getattr(self.args, "mini_batches_num", 4)

    def make_transitions(self, batch, mask, action_logits, old_values, terminated):
        bs = batch.batch_size
        ts = batch["obs"].shape[1]
        actions = batch["actions"][:, :-1]
        rewards = batch["reward"][:, :-1]
        rewards = rewards.repeat(1, 1, self.n_agents)

        old_values[mask == 0.0] = 0.0
        pi_neglogp = -th.log_softmax(action_logits[:, :-1], dim=-1)
        actions_neglogp = th.gather(pi_neglogp, dim=3, index=actions).squeeze(3)

        # Generalized advantage estimation
        returns, advs = self._compute_returns_advs(old_values, rewards, terminated, 
                                                   self.args.gamma, self.args.tau)

        # TD error
        # returns = rewards + self.args.gamma * old_values[:, 1:]
        # advs = returns - old_values[:, :-1]

        old_values = old_values[:, :-1]
        
        mask_flat = mask[:, :-1]
        mask_flat = mask_flat.flatten()
        index = th.nonzero(mask_flat).squeeze()
        actions = actions.flatten()[index]
        actions_neglogp = actions_neglogp.flatten()[index]
        returns = returns.flatten()[index]
        values = old_values.flatten()[index]
        advantages = advs.flatten()[index]
        rewards = rewards.flatten()[index]

        transitions = {}
        transitions.update({"num": int(th.sum(mask_flat))})
        transitions.update({"actions": actions})
        transitions.update({"actions_neglogp": actions_neglogp})
        transitions.update({"returns": returns})
        transitions.update({"values": values})
        transitions.update({"advantages": advantages})
        transitions.update({"rewards": rewards})

        return transitions

    def flatten_obs_states(self, batch, mask):
        shape = batch["obs"].shape
        bs, ts = shape[0], shape[1]
        obs = batch["obs"][:, :-1]
        states = batch["state"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = th.zeros_like(actions_onehot)
        # right shift actions
        last_actions_onehot[:, 1:] = actions_onehot[:, :-1]

        inputs_cv = []
        inputs_cv.append(states)
        inputs_cv.append(obs.view(bs, ts-1, -1))
        inputs_cv.append(last_actions_onehot.view(bs, ts-1, -1))
        inputs_cv = th.cat(inputs_cv, dim=2) # dim: 0 for bs, 1 for ts, 2 for feature
        inputs_cv = inputs_cv.reshape((-1, inputs_cv.shape[-1]))

        cv_mask = copy.deepcopy(mask[:, :-1, 0])
        cv_mask_index = th.nonzero(cv_mask.flatten()).squeeze()
        inputs_cv = inputs_cv[cv_mask_index]

        states_exp = states.unsqueeze(dim=2).expand(-1, -1, self.n_agents, -1)
        obs_mask = copy.deepcopy(mask[:, :-1])
        obs_index = th.nonzero(obs_mask.flatten()).squeeze()
        obs = obs.reshape((-1, obs.shape[-1]))[obs_index]
        states_exp = states_exp.reshape((-1, states_exp.shape[-1]))[obs_index]

        return obs, states_exp, inputs_cv

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        critic_train_stats = defaultdict(lambda: [])

        bs = batch.batch_size
        ts = batch["obs"].shape[1]

        mask = batch["filled"].float()
        # right shift terminated flag, to be aligned with openai/baseline setups
        terminated = batch["terminated"].float()
        terminated[:, 1:] = terminated[:, :-1].clone()
        mask = mask * (1 - terminated)
        mask = mask.repeat(1, 1, self.n_agents)

        action_logits = th.zeros((bs, ts, self.n_agents, self.n_actions)).cuda()
        if getattr(self.args, "is_observation_normalized", False):
            obs, states, inputs_cv = self.flatten_obs_states(batch, mask)
            self.mac.update_rms(obs)
            if self.central_v:
                self.critic.update_rms(inputs_cv)
            else:
                self.critic.update_rms(obs)

        self.mac.init_hidden(bs)
        for t in range(ts):
            action_logits[:, t] = self.mac.forward(batch, t = t, test_mode=False)
        old_values = self.critic(batch).squeeze()

        transitions     = self.make_transitions(batch, mask, action_logits, old_values, terminated)
        num             = transitions["num"]
        actions_neglogp = transitions["actions_neglogp"].detach()
        returns         = transitions["returns"].detach()
        rewards         = transitions["rewards"].detach()
        values          = transitions["values"].detach()
        advantages      = transitions["advantages"].detach()

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

                mb_adv          = advantages[curr_idx]
                mb_ret          = returns[curr_idx]
                mb_old_neglogp  = actions_neglogp[curr_idx]

                tmp_action_logits = th.zeros((bs, ts, self.n_agents, self.n_actions)).cuda()

                self.mac.init_hidden(bs)
                for t in range(ts):
                    tmp_action_logits[:, t] = self.mac.forward(batch, t = t, test_mode=False)
                tmp_old_values = self.critic(batch).squeeze()

                transitions = self.make_transitions(batch, mask, tmp_action_logits, tmp_old_values, terminated)
                mb_neglogp = transitions["actions_neglogp"][curr_idx]
                mb_values = transitions["values"][curr_idx]
                mb_old_values = values[curr_idx]

                with th.no_grad():
                    approxkl = 0.5 * ((mb_old_neglogp - mb_neglogp)**2).mean()
                    approxkl_lst.append(approxkl)

                entropy = -1.0 * (th.softmax(tmp_action_logits, dim=-1) * th.log_softmax(tmp_action_logits, dim=-1)).sum(dim=-1).mean()
                entropy_lst.append(entropy)

                if self.actor_critic_mode:
                    # actor critic loss
                    pg_loss = th.mean(mb_adv * mb_neglogp)
                else: 
                    # ppo loss
                    prob_ratio = th.clamp(th.exp(mb_old_neglogp - mb_neglogp), 0.0, 16.0)
                    pg_loss_unclipped = - mb_adv * prob_ratio
                    pg_loss_clipped = - mb_adv * th.clamp(prob_ratio,
                                                        1 - self.args.ppo_policy_clip_param,
                                                        1 + self.args.ppo_policy_clip_param)
                    pg_loss = th.mean(th.max(pg_loss_unclipped, pg_loss_clipped))

                # Construct overall loss
                actor_loss = pg_loss - self.args.entropy_loss_coeff * entropy
                actor_loss_lst.append(actor_loss)

                mb_values_clipped = mb_old_values + th.clamp(mb_values - mb_old_values, -0.2, 0.2)
                critic_loss_clipped = (mb_values_clipped - mb_ret)**2
                critic_loss = (mb_values - mb_ret)**2 
                critic_loss = th.mean( th.max(critic_loss_clipped, critic_loss) ) 
                critic_loss_lst.append(critic_loss)

                loss = actor_loss + self.args.critic_loss_coeff * critic_loss
                loss_lst.append(loss)

                self.optimiser_actor.zero_grad()
                actor_loss.backward()
                self.optimiser_actor.step()

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
