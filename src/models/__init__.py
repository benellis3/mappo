REGISTRY = {}

from .basic import DQN, RNN, FCEncoder
REGISTRY["DQN"] = DQN
REGISTRY["RNN"] = RNN
REGISTRY["fc_encoder"] = FCEncoder

from .coma import COMANonRecursiveAgent, COMARecursiveAgent
REGISTRY["coma_recursive"] = COMARecursiveAgent
REGISTRY["coma_non_recursive"] = COMANonRecursiveAgent

from .pomace.basic import poMACENoiseNetwork, poMACEMultiagentNetwork

REGISTRY["pomace_noise_nn1"] = poMACENoiseNetwork
REGISTRY["pomace_noise_multiagent_nn1"] = poMACEMultiagentNetwork

from .pomace.exp1 import poMACEExp1NoiseNetwork, poMACEExp1Network
REGISTRY["pomace_noise_exp1_nn1"] = poMACEExp1NoiseNetwork
REGISTRY["pomace_nn_exp1"] = poMACEExp1Network

from .pomace.exp2 import poMACEExp2Network
REGISTRY["pomace_nn_exp2"] = poMACEExp2Network

from .vdn import VDNMixingNetwork, VDNMixer
REGISTRY["vdn_mixing_network"] = VDNMixingNetwork
REGISTRY["vdn_mixer"] = VDNMixer

from .qmix import QMIXMixingNetwork, QMIXMixer
REGISTRY["qmix_mixing_network"] = QMIXMixingNetwork
REGISTRY["qmix_mixer"] = VDNMixer






