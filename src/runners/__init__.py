REGISTRY = {}

from .nstep_runner import NStepRunner
REGISTRY["nstep"] = NStepRunner

from .coma_runner import COMARunner
REGISTRY["coma"] = COMARunner

from .pomace_runner import poMACERunner
REGISTRY["pomace"] = poMACERunner

from .iql_runner import IQLRunner
REGISTRY["iql"] = IQLRunner

from .vdn_runner import VDNRunner
REGISTRY["vdn"] = VDNRunner

from .qmix_runner import QMIXRunner
REGISTRY["qmix"] = QMIXRunner