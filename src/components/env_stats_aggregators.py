REGISTRY = {}

from envs.starcraft1 import StatsAggregator as SC1StatsAggregator
REGISTRY["sc1"] = SC1StatsAggregator

from envs.starcraft2 import StatsAggregator as SC2StatsAggregator
REGISTRY["sc2"] = SC2StatsAggregator