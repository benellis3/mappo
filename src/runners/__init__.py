REGISTRY = {}

from .nstep_runner import NStepRunner
REGISTRY["nstep"] = NStepRunner

from .coma_runner import COMARunner
REGISTRY["coma"] = COMARunner

from .iac_runner import IACRunner
REGISTRY["iac"] = IACRunner

from .pomace_runner import poMACERunner
REGISTRY["pomace"] = poMACERunner

from .mcce_runner import MCCERunner
REGISTRY["mcce"] = MCCERunner

from .coma_joint_runner import COMAJointRunner
REGISTRY["coma_joint"] = COMAJointRunner

from .iql_runner import IQLRunner
REGISTRY["iql"] = IQLRunner

from .vdn_runner import VDNRunner
REGISTRY["vdn"] = VDNRunner

from .qmix_runner import QMIXRunner
REGISTRY["qmix"] = QMIXRunner

from .mackrel_runner import MACKRELRunner
REGISTRY["mackrel"] = MACKRELRunner