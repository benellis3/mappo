from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import torch as th


class EpisodeRunner:

    def __init__(self, args, logging_struct):
        self.args = args
        self.logging = logging_struct
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args)
        self.t = 0

        self.T_env = 0

    def get_env_info(self):
        #TODO: move this to env
        env_info = {"state_shape": self.env.get_state_size(),
                    "obs_shape": self.env.get_obs_size(),
                    "n_actions": self.env.get_total_actions(),
                    "n_agents": self.env.n_agents,
                    "episode_limit": self.env.episode_limit}
        self.episode_limit = self.env.episode_limit
        return env_info

    def setup(self, scheme, groups, mac):
        self.new_batch = partial(EpisodeBatch(scheme, groups, self.episode_limit, self.batch_size))
        self.mac = mac
        self.scheme = scheme
        self.groups = groups

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self):
        self.reset()

        env_data = {
            "state": self.env.get_state(),
            "avail_actions": self.env.get_avail_actions(),
            "obs": self.env.get_obs()
        }

        # TODO: Bs should take into account if an env has terminated
        self.batch.update_transition_data(env_data, bs=slice(None), ts=self.envs_t)

        # TODO: Pass batch to mac to get actions out
        # TODO: Update the rest of the data
        # TODO: Loop over time

        pass  # return EpisodeBatch of experiences

    def log(self):
        pass
