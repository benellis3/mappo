from .q_learner import QLearner
from .coma_learner import COMALearner
from .policy_grad_learner import PolicyGradLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["policy_grad_learner"] = PolicyGradLearner