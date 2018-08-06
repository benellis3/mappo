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

        self.last_target_update_episode = 0

        self.critic = COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        # TODO: separate lrs
        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.lr)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"]
        actions = batch["actions"]
        terminated = batch["terminated"]
        mask = batch["filled"].float()
        # can't train critic using last step unless terminal
        critic_mask = mask.clone()

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions
        avail_actions = batch["avail_actions"]
        mac_out[avail_actions == 0] = -9999999  # From OG deepmarl

        q_vals = self.critic(batch)

        # Calculated advantage
        pi = mac_out.softmax(dim=-1).view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.view(-1, 1)).squeeze(1)
        log_pi = mac_out.log_softmax(dim=-1).view(-1, self.n_actions)
        log_pi_taken = th.gather(log_pi, dim=1, index=actions.view(-1, 1)).squeeze(1)

        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        coma_loss = - ((q_taken.detach() * log_pi_taken - baseline) * mask).sum() / mask.sum()

        # TODO: is this faster if losses are summed and pytorch can parallelise(??) agent and critic backwards?
        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        # Optimise critic
        target_q_vals = self.target_critic(batch)
        target_q_vals[avail_actions.view(-1, self.n_actions) == 0] = -9999999
        self._train_critic(rewards, terminated, q_taken, target_q_vals, critic_mask, bs, max_t, t_env)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
        self.logger.log_stat("agent_grad_norm", grad_norm, t_env)

    def _train_critic(self, rewards, terminated, q_taken, target_q_vals, mask, bs, max_t, t_env):
        q_taken = q_taken.view(bs, max_t, self.n_agents)
        target_q_vals = target_q_vals.view(bs, max_t, self.n_agents, self.n_actions)
        # Max over target Q-Values
        target_max_qvals = target_q_vals.max(dim=3)[0]

        # Add dummy target to allow training when last state is terminal
        target_max_qvals = th.cat([target_max_qvals[:, 1:], th.zeros_like(target_max_qvals[:, -1:])], dim=1)

        # Calculate 1-step Q-Learning targets TODO: td-lambda
        targets = rewards + self.args.gamma * (1 - terminated).float() * target_max_qvals
        # targets = build_targets(rewards, terminated, mask, target_max_qvals, self.n_agents, self.args.gamma, 0.8)

        # Only train last step if terminal
        mask[:, -1] = terminated[:, -1]
        # Td-error
        td_error = (q_taken - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()  # Not dividing by number of agents, only # valid timesteps
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        self.logger.log_stat("critic_loss", loss.item(), t_env)
        self.logger.log_stat("critic_grad_norm", grad_norm, t_env)
        self.logger.log_stat("td_error", (masked_td_error.sum().item() / mask.sum()), t_env)
        self.logger.log_stat("mean_q_value",
                             (q_taken * mask).sum().item() / (mask.sum() * self.args.n_agents), t_env)
        self.logger.log_stat("mean_target", (targets * mask).sum().item() / (mask.sum() * self.args.n_agents), t_env)
        pass


    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
