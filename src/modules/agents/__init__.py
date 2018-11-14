REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
