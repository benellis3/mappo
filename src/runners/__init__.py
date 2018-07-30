REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .nstep_runner import NStepRunner
REGISTRY["nstep"] = NStepRunner

from .iac_runner import IACRunner
REGISTRY["iac"] = IACRunner

from .iql_runner import IQLRunner
REGISTRY["iql"] = IQLRunner