REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .conv1d_agent import Conv1dAgent, Conv1dRnnAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["conv1d"] = Conv1dAgent
REGISTRY["conv1d_rnn"] = Conv1dRnnAgent