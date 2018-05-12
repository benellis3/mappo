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

from .xxx import XXXLearner
REGISTRY["xxx"] = XXXLearner