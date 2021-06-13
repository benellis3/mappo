import numpy as np
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical, MultivariateNormal
from torch.nn.functional import softmax
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}

class GaussianActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, t_env, test_mode=False):
        self.epsilon = self.schedule.eval(t_env)

        mu, sigma = agent_inputs[0], agent_inputs[1]
        if test_mode:
            return mu
        else:
            distr = th.distributions.Normal(mu, sigma)
            return distr.sample().squeeze()

REGISTRY["gaussian"] = GaussianActionSelector


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = -1e6

        if test_mode:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(logits=masked_policies).sample().long()

        # add randomness
        random_numbers = th.rand_like(agent_inputs[:,:,0])
        pick_random = (random_numbers < 0.1).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * picked_actions

        return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        # Was there so I used it
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:,:,0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
