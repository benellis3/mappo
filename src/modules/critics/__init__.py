critic_REGISTRY = {}

from .centralV import CentralVCritic
from .vanilla_critic import VanillaCritic
from .rnn_critic import RNNCritic
from .rnn_critic_sum import RNNCriticSum
from .cnn_critic import CNNCritic
from .centralV_rnn_critic import CentralVRNNCritic
from .central_vdn_rnn_critic import CentralVDNRNNCritic
from .decentral_critic import DecentralCritic
from .independent_rnn_critic import IndependentRNNCritic
from .joint_rnn_critic import JointRNNCritic

critic_REGISTRY["vanilla"] = VanillaCritic
critic_REGISTRY["rnn"] = RNNCritic
critic_REGISTRY["rnn_sum"] = RNNCriticSum
critic_REGISTRY["central_v"] = CentralVCritic
critic_REGISTRY["cnn"] = CNNCritic
critic_REGISTRY["central_vdn_rnn_critic"] = CentralVDNRNNCritic
critic_REGISTRY["decentral_critic"] = DecentralCritic
critic_REGISTRY["independent_rnn_critic"] = IndependentRNNCritic
critic_REGISTRY["centralV_rnn_critic"] = CentralVRNNCritic
critic_REGISTRY["joint_rnn_critic"] = JointRNNCritic