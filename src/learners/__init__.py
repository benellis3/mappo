from .q_learner import QLearner
from .coma_learner import COMALearner
from .actor_critic_learner import ActorCriticLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner