REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .cnn_agent import CNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["cnn"] = CNNAgent
REGISTRY["ff"] = FFAgent