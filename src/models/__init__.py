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

from .coma_joint import COMAJointNonRecurrentMultiAgentNetwork
# REGISTRY["coma_joint_recurrent_nn"] = COMAJointRecurrentMultiAgentNetwork
REGISTRY["coma_joint_non-recurrent_multiagent_nn"] = COMAJointNonRecurrentMultiAgentNetwork

from .vdn import VDNMixingNetwork, VDNMixer
REGISTRY["vdn_mixing_network"] = VDNMixingNetwork
REGISTRY["vdn_mixer"] = VDNMixer

from .qmix import QMIXMixingNetwork, QMIXMixer
REGISTRY["qmix_mixing_network"] = QMIXMixingNetwork
REGISTRY["qmix_mixer"] = VDNMixer

from .mackrel import MACKRELCriticLevel1, MACKRELCriticLevel2, MACKRELCriticLevel3
REGISTRY["mackrel_critic_level1"] = MACKRELCriticLevel1
REGISTRY["mackrel_critic_level2"] = MACKRELCriticLevel2
REGISTRY["mackrel_critic_level3"] = MACKRELCriticLevel3
from .mackrel import MACKRELRecurrentAgentLevel1, MACKRELRecurrentAgentLevel2, MACKRELRecurrentAgentLevel3
REGISTRY["mackrel_recurrent_agent_level1"] = MACKRELRecurrentAgentLevel1
REGISTRY["mackrel_recurrent_agent_level2"] = MACKRELRecurrentAgentLevel2
REGISTRY["mackrel_recurrent_agent_level3"] = MACKRELRecurrentAgentLevel3
from .mackrel import MACKRELNonRecurrentAgentLevel1, MACKRELNonRecurrentAgentLevel2, MACKRELNonRecurrentAgentLevel3
REGISTRY["mackrel_nonrecurrent_agent_level1"] = MACKRELNonRecurrentAgentLevel1
REGISTRY["mackrel_nonrecurrent_agent_level2"] = MACKRELNonRecurrentAgentLevel2
REGISTRY["mackrel_nonrecurrent_agent_level3"] = MACKRELNonRecurrentAgentLevel3

from .mackrel_v import MACKRELV
REGISTRY["mackrel_v"] = MACKRELV




