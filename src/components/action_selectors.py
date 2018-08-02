import numpy as np
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.functional import softmax
# from .transforms_old import _to_batch, _from_batch, _adim, _vdim, _bsdim, _check_nan
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}

class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args
        self.output_type = "policies"
        pass

    def select_action(self, inputs, avail_actions, tformat, test_mode=False):
        assert tformat in ["a*bs*t*v"], "invalid format!"

        if isinstance(inputs["policies"], Variable):
            agent_policies = inputs["policies"].data.clone()
        else:
            agent_policies = inputs["policies"].clone()  # might not be necessary

        if avail_actions is not None:
            """
            NOTE: MULTINOMIAL ACTION SELECTION  is usually performed by on-policy algorithms.
            ON-POLICY mean that avail_actions have to be handled strictly within the model, and need to form part
            of the backward pass.
            However, sometimes, numerical instabilities require to use non-zero masking (i.e. using tiny values) of the
            unavailable actions in the model - else the backward might erratically return NaNs.
            In this case, non-available actions may be hard-set to 0 in the action selector. The off-policy shift that
            this creates can usually be assumed to be extremely tiny.
            """
            _sum = th.sum(agent_policies * avail_actions, dim=_vdim(tformat), keepdim=True)
            _sum_mask = (_sum == 0.0)
            _sum.masked_fill_(_sum_mask, 1.0)
            masked_policies = agent_policies * avail_actions / _sum

            # if no action is available, choose an action uniformly anyway...
            masked_policies.masked_fill_(_sum_mask.repeat(1, 1, 1, avail_actions.shape[_vdim(tformat)]),
                                         1.0 / avail_actions.shape[_vdim(tformat)])
            # throw debug message
            if th.sum(_sum_mask) > 0:
                if self.args.debug_verbose:
                    print('Warning in MultinomialActionSelector.available_action(): some input policies sum up to 0!')
        else:
            masked_policies = agent_policies
        masked_policies_batch, params, tformat = _to_batch(masked_policies, tformat)

        _check_nan(masked_policies_batch)
        mask = (masked_policies_batch != masked_policies_batch)
        masked_policies_batch.masked_fill_(mask, 0.0)
        assert th.sum(masked_policies_batch < 0) == 0, "negative value in masked_policies_batch"

        a = masked_policies_batch.cpu().numpy()
        try:
            _samples = Categorical(masked_policies_batch).sample().unsqueeze(1).float()
        except RuntimeError as e:
            print('Warning in MultinomialActionSelector.available_action(): Categorical throws error {}!'.format(e))
            masked_policies_batch.random_(0, 2)
            _samples = Categorical(masked_policies_batch).sample().unsqueeze(1).float()
            pass

        _samples = _samples.masked_fill_(mask.long().sum(dim=1, keepdim=True) > 0, float("nan"))

        samples = _from_batch(_samples, params, tformat)
        _check_nan(samples)

        return samples, masked_policies, tformat

REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        # Was there so I used it
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")

    def select_action(self, agent_inputs, avail_actions, t, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t)

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
