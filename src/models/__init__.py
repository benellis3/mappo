REGISTRY = {}

from .basic import DQN, RNN, FCEncoder
REGISTRY["DQN"] = DQN
REGISTRY["RNN"] = RNN
REGISTRY["fc_encoder"] = FCEncoder

from .coma import COMANonRecursiveAgent, COMARecursiveAgent
REGISTRY["coma_recursive"] = COMARecursiveAgent
REGISTRY["coma_non_recursive"] = COMANonRecursiveAgent

from .iac import IACNonRecursiveAgent, IACRecursiveAgent
REGISTRY["iac_recursive"] = IACRecursiveAgent
REGISTRY["iac_non_recursive"] = IACNonRecursiveAgent

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






