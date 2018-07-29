from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch


class EpisodeRunner:

    def __init__(self, args, logging_struct):
        self.args = args
        self.logging = logging_struct

        env_fn = partial(env_REGISTRY[self.args.env], env_args=self.args.env_args)
        # Simple list of envs for now to nail down the api
        self.envs = [env_fn() for _ in range(self.args.batch_size_run)]

    def get_env_info(self):
        env = self.envs[0]
        env_info = {"state_shape": env.get_state_size(),
                    "obs_shape": env.get_obs_size(),
                    "n_actions": env.get_total_actions(),
                    "n_agents": env.n_agents,
                    "episode_limit": env.episode_limit}
        return env_info

    def setup(self, scheme, groups):
        pass

    def reset(self):
        pass

    def run(self):
        pass # return EpisodeBatch of experiences
