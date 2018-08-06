from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.test_rewards = []
        self.test_env_stats = []

        self.log_train_stats_t = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        # TODO: Remove these if the runner doesn't need them
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env.get_env_info()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # TODO: Return episode limit from the environment separately from env_info
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        if not test_mode:
            self.t_env += self.t

        # TODO: Sort out sc2/env stats logging
        env_stats = self.env.get_stats()

        if test_mode:
            # Always log testing stats
            self.test_rewards.append(episode_return)
            self.test_env_stats.append(env_stats)
            if len(self.test_rewards) == self.args.test_nepisode:
                # Finished testing for test_nepisodes, log stats about it for convenience
                self.logger.log_stat("mean_test_return", np.mean(self.test_rewards), self.t_env)
                self.logger.log_stat("std_test_return", np.std(self.test_rewards), self.t_env)
                self.test_rewards = []

                for k, v in self.env.get_agg_stats([env_stats]).items():
                    self.logger.log_stat("test_mean_" + k, v, self.t_env)
                self.test_env_stats = []
            self.logger.log_stat("test_return", episode_return, self.t_env)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            # Only log training stats if enough time has passed
            self.logger.log_stat("train_return", episode_return, self.t_env)
            self.logger.log_stat("ep_length", self.t, self.t_env)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            # Log the env stats
            for k, v in self.env.get_agg_stats([env_stats]).items():
                self.logger.log_stat(k, v, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch
