from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np


class NormalFormMatrixGame(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Define the agents
        self.n_agents = 2

        self.episode_limit = 1

        # Define the internal state
        self.steps = 0

        self.p_common = args.p_common
        self.p_observation = args.p_observation
        self.common_knowledge = 0
        self.matrix_id = 0

        self.payoff_values = []

        self.payoff_values.append(np.array([  # payoff values
            [5, 0, 0, 2, 0],
            [0, 1, 2, 4, 2],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 5],
        ], dtype=np.float32) * 0.1)
        self.payoff_values.append(np.array([  # payoff values
            [0, 0, 1, 0, 5],
            [0, 0, 2, 0, 0],
            [1, 2, 4, 2, 1],
            [0, 0, 2, 0, 0],
            [5, 0, 1, 0, 0],
        ], dtype=np.float32) * 0.1)

        self.n_actions = len(self.payoff_values[0])

        self.state = self._get_state()
        self.obs = self._get_obs()


    # ---------- INTERACTION METHODS -----------------------------------------------------------------------------------

    def reset(self):
        """ Returns initial observations and states"""
        self.steps = 0
        self.state = self._get_state()
        self.obs = self._get_obs()
        return self.obs, self.state

    def step(self, actions):
        """ Returns reward, terminated, info """
        reward = self.payoff_values[self.matrix_id][actions[0], actions[1]].astype(np.float64)

        info = {}
        self.steps += 1
        terminated = True
        info["episode_limit"] = True

        return reward, terminated, info

    def _get_obs(self):
        """ Returns all agent observations in a list """
        if self.common_knowledge == 1:
            observations = np.repeat(self.matrix_id, 2)
        else:
            observations = np.ones(self.n_agents) * 2  # -1: unobserved

            for a in range(self.n_agents):
                if np.random.random() < self.p_observation:
                    observations[a] += self.matrix_id + 1
        return observations.astype(np.float64)

    def get_obs(self):
        return self.obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return 2

    def _get_state(self):
        if np.random.random() < self.p_common:
            self.common_knowledge = 1
        else:
            self.common_knowledge = 0
        self.matrix_id = np.random.randint(0, 2)

        return np.array([self.matrix_id, self.common_knowledge], dtype=np.float32)

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return 2

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

    # --------- RENDER METHODS -----------------------------------------------------------------------------------------

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError
