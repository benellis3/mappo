from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import torch as th


class EpisodeRunner:

    def __init__(self, args, logging_struct):
        self.args = args
        self.logging = logging_struct
        self.batch_size = self.args.batch_size_run

        env_fn = partial(env_REGISTRY[self.args.env], env_args=self.args.env_args)
        # Simple list of envs for now to nail down the api
        self.envs = [env_fn() for _ in range(self.batch_size)]
        self.envs_t = [0 for _ in range(self.batch_size)]

        self.T_env = 0

    def get_env_info(self):
        env = self.envs[0]
        env_info = {"state_shape": env.get_state_size(),
                    "obs_shape": env.get_obs_size(),
                    "n_actions": env.get_total_actions(),
                    "n_agents": env.n_agents,
                    "episode_limit": env.episode_limit}
        self.episode_limit = env.episode_limit
        return env_info

    def setup(self, scheme, groups, mac):
        self.new_batch = partial(EpisodeBatch(scheme, groups, self.episode_limit, self.batch_size))
        self.mac = mac
        self.scheme = scheme
        self.groups = groups

    def reset(self):
        self.batch = self.new_batch()
        for env in self.envs:
            env.reset()
        self.envs_t = [0 for _ in range(self.batch_size)]
        pass

    def run(self):
        self.reset()

        env_data = {
            "state": th.FloatTensor([env.get_state() for env in self.envs]),
            "avail_actions": th.LongTensor([env.get_avail_actions() for env in self.envs]),
            "obs": th.FloatTensor([env.get_obs() for env in self.envs])
        }

        # TODO: Bs should take into account if an env has terminated
        self.batch.update_transition_data(env_data, bs=slice(None), ts=self.envs_t)

        # TODO: Pass batch to mac to get actions out
        # TODO: Update the rest of the data
        # TODO: Loop over time

        pass  # return EpisodeBatch of experiences

    def log(self):
        pass
