# from .multiagentenv import MultiAgentEnv

import torch as th
import numpy as np
import pygame
from random import random

'''
A simple environment for integration tests
'''


class Env(): #MultiAgentEnv):

    def __init__(self, **kwargs):
        self.actions = np.asarray([[0, 1],
                                   [1, 0],
                                   [0, -1],
                                   [-1, 0],
                                   [0, 0]])
        self.n_actions = self.actions.shape[0]
        self.args = kwargs["env_args"]
        self.n_agents = self.args["n_agents"]
        self.t_terminal = 2 # env will return terminated=True when this time-step has been reached
        self.obs_size = 5
        self.state_size = 10
        self.is_tensor = True
        self.is_cuda = False

        self.episode_limit = self.t_terminal  # downward-compatibility

        # TODO: adapt for use with subproc_id, thread_id and loop_id
        self.bs_id = kwargs.get("bs_id", 0)

        self.t = 0
        self.time_reward = -0.1
        self.ttype = (th.cuda.FloatTensor if self.is_cuda else th.FloatTensor) if self.is_tensor else np.ndarray
        self.reset()

    def reset(self):
        self.t = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        self.t += 1

        reward = -0.1*(self.bs_id+1)

        # Check whether env has terminated
        terminated = False
        info = {"episode_limit": False}
        if self.t == self.bs_id+1: # arbitrary termination condition
            terminated = True
        elif self.t >= self.t_terminal:
            terminated = True
            info = {"episode_limit": True}

        return reward, terminated, info

    def get_total_actions(self):
        return self.n_actions

    def get_obs_agent(self, agent_id):
        arr = self.ttype([agent_id]*self.obs_size)
        arr2 = self.ttype([self.t*0.1]*self.obs_size)
        arr3 = self.ttype([self.bs_id*0.01] * self.obs_size)
        return arr + arr2 + arr3

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        arr2 = self.ttype([self.t*1]*self.state_size)
        arr3 = self.ttype([self.bs_id*0.1] * self.state_size)
        return arr2 + arr3

    def get_avail_agent_actions(self, agent_id):
        return [1 for _ in range(self.n_actions)]

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.state_size

    def get_stats(self):
        pass

    def close(self):
        pass

    def render_array(self):
        pass

    def render(self):
        pass

