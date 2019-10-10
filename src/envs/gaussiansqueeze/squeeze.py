# Using code from author's implementation: https://github.com/Sonkyunghwan/QTRAN/blob/master/Others/envs/environment.py,
# Class MultiAgentSimpleEnv4 (at the bottom)
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np


class GaussianSqueeze(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Define the agents and actions
        self.n_agents = 2
        self.n_actions = 11
        self.episode_limit = 1

        # Paper says 0.2, but the author's code uses 2.
        self.max_s = 2 # Using 2
        self.state = np.random.uniform(0, self.max_s, self.n_agents)

        self.mu = 8

    def reset(self):
        """ Returns initial observations and states"""
        self.state = np.random.uniform(0, self.max_s, self.n_agents)
        return self.state, self.state

    def step(self, actions):
        """ Returns reward, terminated, info """
        r = np.sum(np.array(actions) * self.state)/self.n_agents

        # Regular Gaussian Squeeze
        reward = r * np.exp(-np.square(r-self.mu) / 0.25) # paper says 1 but their code uses 0.25.

        # Hacky Debugging
        # reward = 1

        info = {}
        terminated = True
        info["episode_limit"] = False

        return reward, terminated, info

    def get_obs(self):
        return [self.state for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.get_state_size()

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.state)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def get_stats(self):
        raise NotImplementedError

