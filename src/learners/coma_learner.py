import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from utils.rl_utils import build_targets
import torch as th
from torch.optim import RMSprop


class COMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.critic = COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals = self._train_critic(batch, rewards, terminated, actions, avail_actions, critic_mask, bs, max_t, t_env)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum(), t_env)

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        # log_pi = mac_out.log_softmax(dim=-1).view(-1, self.n_actions)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        # q_taken_reshape = q_taken.reshape(bs, max_t-1, self.n_agents)
        # next_qs = q_taken_reshape[:, 1:]
        # one_step_qsa = rewards[:, :-1] + self.args.gamma * next_qs
        # hacky_thing = q_taken_reshape.clone()
        # hacky_thing[:, 1:] = hacky_thing[:, 1:] * 1.0
        # hacky_thing[:, :-1] = hacky_thing[:, :-1] + 0.0 * one_step_qsa
        # advantages = (hacky_thing.view(-1) - baseline).detach()

        advantages = (q_taken - baseline).detach()

        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum(), t_env)

        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
        self.logger.log_stat("agent_grad_norm", grad_norm, t_env)

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t, t_env):
        # Optimise critic
        target_q_vals = self.target_critic(batch)
        target_q_vals = target_q_vals.view(bs, max_t, self.n_agents, self.n_actions)[:, 1:]
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        targets = build_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)

        q_vals = th.zeros_like(target_q_vals)

        running_log = {
            "critic_loss" : [],
            "critic_grad_norm" : [],
            "abs_td_error" : [],
            "mean_q_taken" : [],
            "mean_target" : []
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents).reshape(-1)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic(batch, t)
            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            q_taken = th.gather(q_t, dim=1, index=actions[:, t].reshape(-1, 1)).squeeze(1)
            targets_t = targets[:, t].reshape(-1)

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()  # Not dividing by number of agents, only # valid timesteps
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, 5)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            running_log["abs_td_error"].append((masked_td_error.abs().sum().item() / mask_t.sum()))
            running_log["mean_q_taken"].append((q_taken * mask_t).sum().item() / (mask_t.sum()))
            running_log["mean_target"].append((targets_t * mask_t).sum().item() / mask_t.sum())

        ts_logged = len(running_log["critic_loss"])
        for key in ["critic_loss", "critic_grad_norm", "abs_td_error", "mean_q_taken", "mean_target"]:
            self.logger.log_stat(key, sum(running_log[key])/ts_logged, t_env)

        return q_vals.view(-1, self.n_actions)


    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
