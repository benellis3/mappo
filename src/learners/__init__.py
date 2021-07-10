from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner
from .centralized_ppo_learner import CentralPPOLearner
from .decentralized_ppo_learner import DecentralPPOLearner
from .independent_ppo_learner import IndependentPPOLearner
from .adaptive_kl_learner import AdaptiveKLLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["central_ppo_learner"] = CentralPPOLearner
REGISTRY["decentral_ppo_learner"] = DecentralPPOLearner
REGISTRY["independent_ppo_learner"] = IndependentPPOLearner
REGISTRY["adaptive_kl_learner"] = AdaptiveKLLearner