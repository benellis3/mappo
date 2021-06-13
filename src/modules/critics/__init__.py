critic_REGISTRY = {}

from .centralV import CentralVCritic
from .vanilla_critic import VanillaCritic
from .rnn_critic import RNNCritic
from .rnn_critic_sum import RNNCriticSum
from .cnn_critic import CNNCritic
from .central_critic import CentralCritic
from .central_rnn_critic import CentralRNNCritic
from .agent_specific_central_critic import AgentSpecificCentralCritic

critic_REGISTRY["vanilla"] = VanillaCritic
critic_REGISTRY["rnn"] = RNNCritic
critic_REGISTRY["rnn_sum"] = RNNCriticSum
critic_REGISTRY["central_v"] = CentralVCritic
critic_REGISTRY["cnn"] = CNNCritic
critic_REGISTRY["central_critic"] = CentralCritic
critic_REGISTRY["central_rnn_critic"] = CentralRNNCritic
critic_REGISTRY["agent_specific_central_critic"] = AgentSpecificCentralCritic
