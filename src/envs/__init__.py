from functools import partial

from .predator_prey import PredatorPreyCapture
from .box_pushing import CoopBoxPushing
from .predator_prey__old import PredatorPreyEnv as PredatorPreyOldEnv
from .matrix_game import NormalFormMatrixGame
from .test import IntegrationTestEnv
from .multiagentenv import MultiAgentEnv
from .stag_hunt import StagHunt
#from .starcraft2 import StarCraft2Env
from smac.env import MultiAgentEnv, StarCraft2Env
try:
    from smac.env import StarCraft2CustomEnv
except Exception as e:
    print(e)
from .gaussiansqueeze.squeeze import GaussianSqueeze
from .matrix_game.matrix_game_simple import Matrixgame
from .matrix_game.matrix_game_random import RandomMatrixgame
from .test.parallel_test import RandomEnv
from .env_wrappers import FrameStackStartCraft2Env


# TODO: Do we need this?
def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["pred_prey__old"] = partial(env_fn, env=PredatorPreyOldEnv)
REGISTRY["pred_prey"] = partial(env_fn, env=PredatorPreyCapture)
REGISTRY["rnd"] = partial(env_fn, env=RandomEnv)
#REGISTRY["pred_prey_cython"] = partial(env_fn,
#                                     env = PredatorPreyCaptureCython)
REGISTRY["matrix_game"] = partial(env_fn, env=Matrixgame)
REGISTRY["random_matrix_game"] = partial(env_fn, env=RandomMatrixgame)
REGISTRY["integration_test"] = partial(env_fn, env=IntegrationTestEnv)
REGISTRY["box_push"] = partial(env_fn, env=CoopBoxPushing)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2framestack"] = partial(env_fn, env=FrameStackStartCraft2Env)
try:
    REGISTRY["sc2custom"] = partial(env_fn, env=StarCraft2CustomEnv)
except Exception as e:
    print(e)
REGISTRY["squeeze"] = partial(env_fn, env=GaussianSqueeze)

try:
    from .starcraft1 import StarCraft1Env
    REGISTRY["sc1"] = partial(env_fn,
                              env=StarCraft1Env)
except Exception as e:
    print(e)

# MaGYM
try:
    from .ma_gym.wrapper import MaGymWrapper
    REGISTRY["Switch2-v0"] = partial(env_fn, env=MaGymWrapper)
    REGISTRY["Switch4-v0"] = partial(env_fn, env=MaGymWrapper)
    REGISTRY["Combat-v0"] = partial(env_fn, env=MaGymWrapper)
    REGISTRY["Checkers-v0"] = partial(env_fn, env=MaGymWrapper)
    REGISTRY["PredatorPrey5x5-v0"] = partial(env_fn, env=MaGymWrapper)
    REGISTRY["PredatorPrey7x7-v0"] = partial(env_fn, env=MaGymWrapper)
    for p in [-2, -3, -4]:
        REGISTRY["PredatorPrey7x7P{}-v0".format(p)] = partial(env_fn, env=MaGymWrapper)
except Exception as e:
    print(e)
