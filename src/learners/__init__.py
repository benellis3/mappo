from .q_learner import QLearner
from .coma_learner import COMALearner
from .actor_critic_learner import ActorCriticLearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["ppo_learner"] = PPOLearner