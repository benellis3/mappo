import numpy as np
import torch as th
from torch.optim import Adam
from collections import defaultdict

from components.episode_buffer import EpisodeBatch
from modules.critics import critic_REGISTRY

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
        self.optimiser_actor = Adam(params=self.agent_params, lr=args.lr_actor, eps=args.optim_eps)
        self.optimiser_critic = Adam(params=self.critic_params, lr=args.lr_critic, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.mini_epochs_actor = getattr(self.args, "mini_epochs_actor", 4)
        self.mini_epochs_critic = getattr(self.args, "mini_epochs_critic", 4)
        self.advantage_calc_method = getattr(self.args, "advantage_calc_method", "GAE")
        self.agent_type = getattr(self.args, "agent", None)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        critic_train_stats = defaultdict(lambda: [])

        # TODO: what if one agent dies
        actions = batch["actions"].cuda()
        rewards = batch["reward"].cuda()
        filled_mask = batch['filled'].float().cuda()
        terminated = batch["terminated"].float().cuda()

        mask = filled_mask.squeeze(dim=-1)
        terminated = terminated.squeeze(dim=-1)

        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]) # mask out the last entry

        ## get dead agents
        # no-op (valid only when dead)
        # https://github.com/oxwhirl/smac/blob/013cf27001024b4ce47f9506f2541eca0b247c95/smac/env/starcraft2/starcraft2.py#L499
        avail_actions = batch['avail_actions'].cuda()
        alive_mask = ( (avail_actions[:, :, :, 0] != 1.0) * (th.sum(avail_actions, dim=-1) != 0.0) ).float()
        num_alive_agents = th.sum(alive_mask, dim=-1).float() * mask

        if getattr(self.args, "is_observation_normalized", False):
            obs = batch["obs"].cuda()
            bs, ts = batch.batch_size, batch.max_seq_length

            inputs = []
            inputs.append(obs)

            if self.args.obs_last_action:
                actions_input = th.zeros_like(batch["actions_onehot"])
                actions_input[:, 1:] = batch["actions_onehot"][:, :-1]
                inputs.append(actions_input)

            if self.args.obs_agent_id:
                agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, ts, -1, -1)
                inputs.append(agent_ids)

            inputs = th.cat([x.reshape(bs*ts*self.n_agents, -1) for x in inputs], dim=1)

            obs_mask = mask[...].clone()
            obs_mask = obs_mask.flatten()
            obs_index = th.nonzero(obs_mask).squeeze()
            inputs = inputs[obs_index]

            self.mac.update_rms(inputs)
            self.critic.update_rms(batch)

        if self.agent_type == "rnn":
            action_logits = th.zeros((batch.batch_size, rewards.shape[1], self.n_agents, self.n_actions)).cuda()
            self.mac.init_hidden(batch.batch_size)
            for t in range(rewards.shape[1]):
                action_logits[:, t] = self.mac.forward(batch, t = t, test_mode=False)

        else:
            raise NotImplementedError

        old_values = self.critic(batch).squeeze(dim=-1).detach()
        rewards = rewards.squeeze(dim=-1)

        if self.advantage_calc_method == "GAE":
            returns, _ = self._compute_returns_advs(old_values, rewards, terminated, 
                                                    self.args.gamma, self.args.tau)

        else:
            raise NotImplementedError

        ## update the critics
        critic_loss_lst = []
        for _ in range(0, self.mini_epochs_critic):
            new_values = self.critic(batch).squeeze()
            critic_loss = 0.5 * th.sum((new_values - returns)**2 * mask) / th.sum(mask)
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

        log_prob_dist = th.nn.functional.log_softmax(action_logits, dim=-1)
        old_log_pac = th.gather(log_prob_dist, dim=3, index=actions).squeeze(3).detach()

        # mask out the dead agents
        central_old_log_pac = th.sum(old_log_pac, dim=-1)

        approxkl_lst = [] 
        entropy_lst = [] 
        actor_loss_lst = []

        target_kl = 0.2
        if getattr(self.args, "is_advantage_normalized", False):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

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
                approxkl = 0.5 * th.sum((central_log_pac - central_old_log_pac)**2 * mask) / th.sum(mask)
                approxkl_lst.append(approxkl)
                if approxkl > 1.5 * target_kl:
                    break

            # for shared policy, maximize the policy entropy averaged over all agents & episodes
            # consider entropy for only alive agents
            # log_prob_dist: n_batch * n_timesteps * n_agents * n_actions
            entropy_all_agents = th.sum(-1.0 * log_prob_dist * th.exp(log_prob_dist), dim=-1) 
            entropy = th.sum( entropy_all_agents * alive_mask) / th.sum(alive_mask)
            entropy_lst.append(entropy)

            prob_ratio = th.clamp(th.exp(central_log_pac - central_old_log_pac), 0.0, 16.0)

            pg_loss_unclipped = - advantages * prob_ratio
            pg_loss_clipped = - advantages * th.clamp(prob_ratio,
                                                1 - self.args.ppo_policy_clip_param,
                                                1 + self.args.ppo_policy_clip_param)

            pg_loss = th.sum(th.max(pg_loss_unclipped, pg_loss_clipped) * mask) / th.sum(mask)

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
        returns = th.zeros_like(_values)
        advs = th.zeros_like(_values)
        lastgaelam = th.zeros_like(_values[:, 0])
        ts = _rewards.size(1) - 1

        for t in reversed(range(ts)):
            nextnonterminal = 1.0 - _terminated[:, t]
            nextvalues = _values[:, t+1]

            reward_t = _rewards[:, t]
            delta = reward_t + gamma * nextvalues * nextnonterminal  - _values[:, t]
            advs[:, t] = lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam

        returns = advs + _values

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
