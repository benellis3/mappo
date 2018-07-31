REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .nstep_runner import NStepRunner
REGISTRY["nstep"] = NStepRunner