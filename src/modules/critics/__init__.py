critic_REGISTRY = {}

from .centralV import CentralVCritic
from .conv1d_critic import Conv1dCritic

critic_REGISTRY["central_v"] = CentralVCritic
critic_REGISTRY["conv1d_critic"] = Conv1dCritic