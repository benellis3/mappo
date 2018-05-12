REGISTRY = {}

from .basic import DQN, RNN, FCEncoder
REGISTRY["DQN"] = DQN
REGISTRY["RNN"] = RNN
REGISTRY["fc_encoder"] = FCEncoder

from .coma import COMANonRecurrentAgent, COMARecurrentAgent
REGISTRY["coma_recursive"] = COMARecurrentAgent
REGISTRY["coma_non_recursive"] = COMANonRecurrentAgent

from .iac import IACNonRecurrentAgent, IACRecurrentAgent
REGISTRY["iac_recursive"] = IACRecurrentAgent
REGISTRY["iac_non_recursive"] = IACNonRecurrentAgent

from .pomace.basic import poMACENoiseNetwork, poMACEMultiagentNetwork

REGISTRY["pomace_noise_nn1"] = poMACENoiseNetwork
REGISTRY["pomace_noise_multiagent_nn1"] = poMACEMultiagentNetwork

from .pomace.exp1 import poMACEExp1NoiseNetwork, poMACEExp1MultiagentNetwork
REGISTRY["pomace_noise_exp1_nn1"] = poMACEExp1NoiseNetwork
REGISTRY["pomace_nn_exp1"] = poMACEExp1MultiagentNetwork

from .pomace.exp2 import poMACEExp2NoiseNetwork, poMACEExp2MultiagentNetwork
REGISTRY["pomace_noise_exp2_nn1"] = poMACEExp2NoiseNetwork
REGISTRY["pomace_nn_exp2"] = poMACEExp2MultiagentNetwork

from .mcce.mcce_policy import MCCECoordinationNetwork, MCCEDecentralizedPolicyNetwork, MCCEMultiAgentNetwork
REGISTRY["mcce_policy_coordination_nn"] = MCCECoordinationNetwork
REGISTRY["mcce_policy_decentralized_policy_nn"] = MCCEDecentralizedPolicyNetwork
REGISTRY["mcce_policy_multiagent_nn"] = MCCEMultiAgentNetwork

from .vdn import VDNMixingNetwork, VDNMixer
REGISTRY["vdn_mixing_network"] = VDNMixingNetwork
REGISTRY["vdn_mixer"] = VDNMixer

from .qmix import QMIXMixingNetwork, QMIXMixer
REGISTRY["qmix_mixing_network"] = QMIXMixingNetwork
REGISTRY["qmix_mixer"] = VDNMixer

from .xxx import XXXCriticLevel1, XXXCriticLevel2, XXXCriticLevel3
REGISTRY["xxx_critic_level1"] = XXXCriticLevel1
REGISTRY["xxx_critic_level2"] = XXXCriticLevel2
REGISTRY["xxx_critic_level3"] = XXXCriticLevel3
from .xxx import XXXRecurrentAgentLevel1, XXXRecurrentAgentLevel2, XXXRecurrentAgentLevel3
REGISTRY["xxx_recurrent_agent_level1"] = XXXRecurrentAgentLevel1
REGISTRY["xxx_recurrent_agent_level2"] = XXXRecurrentAgentLevel2
REGISTRY["xxx_recurrent_agent_level3"] = XXXRecurrentAgentLevel3
from .xxx import XXXNonRecurrentAgentLevel1, XXXNonRecurrentAgentLevel2, XXXNonRecurrentAgentLevel3
REGISTRY["xxx_nonrecurrent_agent_level1"] = XXXNonRecurrentAgentLevel1
REGISTRY["xxx_nonrecurrent_agent_level2"] = XXXNonRecurrentAgentLevel2
REGISTRY["xxx_nonrecurrent_agent_level3"] = XXXNonRecurrentAgentLevel3




