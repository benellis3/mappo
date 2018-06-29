REGISTRY = {}

from .basic import BasicLearner
REGISTRY["basic"] = BasicLearner

from .coma import COMALearner
from .iac_v import IACvLearner
from .iac_q import IACqLearner
from .iql import IQLLearner

REGISTRY["coma"] = COMALearner
REGISTRY["iac_v"] = IACvLearner
REGISTRY["iac_q"] = IACqLearner
REGISTRY["iql"] = IQLLearner

from .vdn import VDNLearner
REGISTRY["vdn"] = VDNLearner

from .qmix import QMIXLearner
REGISTRY["qmix"] = QMIXLearner

from .coma_joint import COMAJointLearner
REGISTRY["coma_joint"] = COMAJointLearner

from .mackrel import MACKRELLearner
REGISTRY["mackrel"] = MACKRELLearner

from .mackrel_v import MACKRELVLearner
REGISTRY["mackrel_v"] = MACKRELVLearner

from .flounderl import FLOUNDERLLearner
REGISTRY["flounderl"] = FLOUNDERLLearner

from .centralV import CentralVLearner
REGISTRY["centralV"] = CentralVLearner