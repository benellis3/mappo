import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.optimiser = RMSprop(params=self.params, lr=args.lr)

        self.last_target_update_episode = 0

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Get the relevant quantities
        # can't bootstrap from last step UNLESS TERMINAL: TODO: handle this case!
        rewards = batch["reward"]
        actions = batch["actions"]
        terminated = batch["terminated"]
        mask = batch["filled"]
        # can't train using last step unless terminal
        mask[:, -1] = terminated[:, -1]

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        avail_actions = batch["avail_actions"][:, 1:]
        target_mac_out[avail_actions == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        target_max_qvals = target_mac_out.max(dim=3)[0]

        # Add dummy target to allow training when last state is terminal
        target_max_qvals = th.cat([target_max_qvals, th.zeros_like(target_max_qvals[:, -1:])], dim=1)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated).float() * target_max_qvals

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time again

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out, dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() # Not dividing by number of agents, only # valid timesteps

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        self.logger.log_stat("loss", loss.item(), t_env)
        self.logger.log_stat("grad_norm", grad_norm, t_env)
        self.logger.log_stat("td_error", (masked_td_error.sum().item()/mask.sum()), t_env)
        self.logger.log_stat("mean_q_value", (chosen_action_qvals * mask).sum().item()/(mask.sum() * self.args.n_agents), t_env)
        self.logger.log_stat("mean_target", (targets * mask).sum().item()/(mask.sum() * self.args.n_agents), t_env)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")
