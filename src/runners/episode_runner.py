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
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.T_env = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit, preprocess=preprocess)
        self.mac = mac
        # TODO: Remove these if the runner doesn't need them
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_reward = 0

        while not terminated:

            env_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(env_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            agent_outputs = self.mac.select_actions(self.batch, t=self.t, test_mode=test_mode)
            # TODO: Should we just return a list of actions directly?
            actions = agent_outputs["actions"]

            # TODO: Return episode limit from the environment separately from env_info
            reward, terminated, env_info = self.env.step(actions)
            episode_reward += reward

            # TODO: Use a better name
            env_data_2 = {
                "actions": [actions],
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(env_data_2, ts=self.t)

            self.t += 1

        self.T_env += self.t

        # TODO: Log stuff

        return self.batch

    def log(self):
        pass
