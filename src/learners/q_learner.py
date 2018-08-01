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

        rewards = batch["reward"]
        actions = batch["actions"]
        avail_actions = batch["avail_actions"]
        terminated = batch["terminated"]
        mask = batch["filled"][:,1:] # Maybe?

        target_mac_out = []
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1) #Concat across time
        # Get rid of first estimate in time for targets
        target_mac_out = target_mac_out[:,1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:,1:] == 0] = -9999999 # From OG deepmarl

        # Max over target Q-Values
        target_max_qvals = target_mac_out.max(dim=3)[0]

        # repeated_rewards = rewards[:,:-1].expand(-1, -1, self.args.n_agents)

        targets = rewards[:,:-1] + self.args.gamma * (1 - terminated[:,1:]).float() * target_max_qvals

        mac_out = []
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out, dim=3, index=actions[:,:-1]).squeeze(3)

        td_error = (chosen_action_qvals - targets.detach())

        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

    def log(self):
        pass # Who needs to log anyway