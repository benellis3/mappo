from .starcraft2 import SC2 as StarCraft2Env, StatsAggregator
from .starcraft2 import map_param_registry
from pysc2 import maps

for name in map_param_registry.keys():
    globals()[name] = type(name, (maps.melee.Melee,), dict(filename=name))

