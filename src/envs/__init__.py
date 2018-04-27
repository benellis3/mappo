from functools import partial

from components.transforms import _join_dicts
from .predator_prey import PredatorPreyEnv
from .starcraft2 import StarCraft2Env
from .test import IntegrationTestEnv

def env_fn(env, **kwargs): # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)

REGISTRY = {}
REGISTRY["pred_prey"] = partial(env_fn,
                                    env = PredatorPreyEnv)
REGISTRY["integration_test"] = partial(env_fn,
                                       env=IntegrationTestEnv)
REGISTRY["sc2"] = partial(env_fn,
                          env=StarCraft2Env)
