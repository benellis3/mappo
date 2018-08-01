from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, logging_struct, args):
        self.args = args
        self.mac = mac
        self.logging = logging_struct

        self.params = mac.get_params()
        self.optimiser = RMSprop(params=self.params, lr=args.lr)

        # TODO: Have a seperate mac for targets!
        self.target_mac = mac

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # TODO: Target network updating based on episodes passed

        # Get the relevant quantities and time-shift as necessary
        rewards = batch["reward"][:,:-1]
        actions = batch["actions"][:,:-1]
        avail_actions = batch["avail_actions"][:,1:]
        terminated = batch["terminated"][:,:-1]
        mask = batch["filled"][:,:-1] # Maybe?

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = target_mac_out[:,1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions == 0] = -9999999 # From OG deepmarl

        # Max over target Q-Values
        target_max_qvals = target_mac_out.max(dim=3)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated).float() * target_max_qvals

        # Calculate estimated Q-Values
        mac_out = []
        for t in range(batch.max_seq_length - 1):  # Don't need to calculate it for the last possible timestep
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
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # TODO: Log stuff!

    # TODO: Get rid of this, log directly using the logging_struct
    def log(self):
        pass # Who needs to log anyway