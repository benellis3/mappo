import numpy as np
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.functional import softmax
from .transforms import _to_batch, _from_batch, _adim, _vdim, _bsdim, _check_nan

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

        # NOTE: Usually, on-policy algorithms should perform action masking in the model itself!
        masked_policies = agent_policies * avail_actions / th.sum(agent_policies * avail_actions, dim=_vdim(tformat), keepdim=True)
        masked_policies_batch, params, tformat = _to_batch(masked_policies, tformat)

        mask = (masked_policies_batch != masked_policies_batch)
        masked_policies_batch.masked_fill_(mask, 0.0)
        _samples = Categorical(masked_policies_batch).sample().unsqueeze(1).float()
        _samples = _samples.masked_fill_(mask.long().sum(dim=1, keepdim=True) > 0, float("nan"))

        samples = _from_batch(_samples, params, tformat)
        _check_nan(samples)

        return samples, masked_policies, tformat

REGISTRY["multinomial"] = MultinomialActionSelector

class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args
        self.output_type = "qvalues"

    def _get_epsilons(self):
        assert False, "function _get_epsilon must be overwritten by user in runner!"
        pass

    def select_action(self, inputs, avail_actions, tformat, test_mode=False):
        assert tformat in ["a*bs*t*v"], "invalid format!"

        if isinstance(inputs["qvalues"], Variable):
            agent_qvalues = inputs["qvalues"].data.clone()
        else:
            agent_qvalues = inputs["qvalues"].clone() # might not be necessary

        # greedy action selection
        assert avail_actions.sum(dim=_vdim(tformat)).prod() > 0.0, \
            "at least one batch entry has no available action!"

        # mask actions that are excluded from selection
        agent_qvalues[avail_actions == 0.0] = -float("inf") # should never be selected!

        masked_qvalues_batch, params, tformat = _to_batch(agent_qvalues, tformat)
        _, _argmaxes = masked_qvalues_batch.max(dim=1, keepdim=True)
        #_argmaxes.unsqueeze_(1)

        if not test_mode: # normal epsilon-greedy action selection
            epsilons, epsilons_tformat = self._get_epsilons()
            random_numbers = epsilons.clone().uniform_()
            _avail_actions, params, tformat = _to_batch(avail_actions, tformat)
            random_actions = Categorical(_avail_actions).sample().unsqueeze(1)
            epsilon_pos = (random_numbers < epsilons).repeat(agent_qvalues.shape[_adim(tformat)], 1) # sampling uniformly from actions available
            epsilon_pos = epsilon_pos[:random_actions.shape[0], :]
            _argmaxes[epsilon_pos] = random_actions[epsilon_pos]
            eps_argmaxes = _from_batch(_argmaxes, params, tformat)
            return eps_argmaxes, agent_qvalues, tformat
        else: # don't use epsilon!
            # sanity check: there always has to be at least one action available.
            argmaxes = _from_batch(_argmaxes, params, tformat)
            return argmaxes, agent_qvalues, tformat

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
