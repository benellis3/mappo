from .coma import COMALearner
from .iac_v import IACvLearner
from .iac_q import IACqLearner
from .iql import IQLLearner

REGISTRY = {}
REGISTRY["coma"] = COMALearner
REGISTRY["iac_v"] = IACvLearner
REGISTRY["iac_q"] = IACqLearner
REGISTRY["iql"] = IQLLearner
