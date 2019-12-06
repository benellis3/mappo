from envs.multiagentenv import MultiAgentEnv
import numpy as np
import random


class RandomEnv(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):

        # Define the agents and actions
        self.n_agents = 2
        self.n_actions = 10
        self.episode_limit = 10

        self.state = np.ones(5)

        # self.allowed = np.random.randint(self.n_actions)
        self.allowed = random.randint(0, self.n_actions-1) # Numpy random.randint returns same value for every env
        self.avail = np.zeros(shape=(self.n_actions))
        self.avail[self.allowed] = 1

        self.t = 0
        print("ALLOWED ACTION:", self.allowed)


    def reset(self):
        """ Returns initial observations and states"""
        self.t = 0
        return self.state, self.state

    def step(self, actions):
        """ Returns reward, terminated, info """
        reward = 1
        self.t += 1

        for a in actions:
            if a != self.allowed:
                raise Exception("Incorrect action!")

        info = {}
        terminated = False if self.t<=4 else bool(np.random.randint(2))
        info["episode_limit"] = True if self.t == self.episode_limit and not terminated else False
        terminated = terminated or self.t == self.episode_limit

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
        return self.avail

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def get_stats(self):
        return {}

