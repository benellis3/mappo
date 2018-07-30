from functools import partial

from components.transforms_old import _join_dicts
from .predator_prey import PredatorPreyCapture
from .box_pushing import CoopBoxPushing
#from .predator_prey import PredatorPreyCaptureCython
from .predator_prey__old import PredatorPreyEnv as PredatorPreyOldEnv
from .matrix_game import NormalFormMatrixGame
from .test import IntegrationTestEnv

def env_fn(env, **kwargs): # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)

REGISTRY = {}
REGISTRY["pred_prey__old"] = partial(env_fn, env=PredatorPreyOldEnv)
REGISTRY["pred_prey"] = partial(env_fn, env=PredatorPreyCapture)
#REGISTRY["pred_prey_cython"] = partial(env_fn,
#                                     env = PredatorPreyCaptureCython)
REGISTRY["matrix_game"] = partial(env_fn, env=NormalFormMatrixGame)
REGISTRY["integration_test"] = partial(env_fn, env=IntegrationTestEnv)
REGISTRY["box_push"] = partial(env_fn, env=CoopBoxPushing)

try:
    from .starcraft1 import StarCraft1Env
    REGISTRY["sc1"] = partial(env_fn,
                              env=StarCraft1Env)
except Exception as e:
    print(e)

from .starcraft2 import StarCraft2Env
REGISTRY["sc2"] = partial(env_fn,
                          env=StarCraft2Env)

