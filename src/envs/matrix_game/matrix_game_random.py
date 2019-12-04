# Using code from author's implementation: https://github.com/Sonkyunghwan/QTRAN/blob/master/Others/envs/environment.py,
# Class MultiAgentSimpleEnv4 (at the bottom)
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np


class RandomMatrixgame(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Define the agents and actions
        self.n_agents = getattr(args, "n_agents", 2)
        self.n_actions = getattr(args, "n_actions", 2)
        self.episode_limit = 1

        shape = tuple([self.n_actions for _ in range(self.n_agents)])
        self.payoff_matrix = (np.random.random_sample(size=shape))
        self.payoff_matrix = self.payoff_matrix / self.payoff_matrix.max()
        self.payoff_matrix = self.payoff_matrix * 10

        self.state = np.ones(5)

    def reset(self):
        """ Returns initial observations and states"""
        return self.state, self.state

    def step(self, actions):
        """ Returns reward, terminated, info """
        reward = self.payoff_matrix[tuple(actions)]

        info = {}
        terminated = True
        info["episode_limit"] = False
        info["correct_argmax"] = np.isclose(reward, 10)

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

