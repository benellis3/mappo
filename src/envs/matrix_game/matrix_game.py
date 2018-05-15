from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np


class NormalFormMatrixGame(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Downwards compatibility of batch_mode
        self.batch_mode = batch_size is not None
        self.batch_size = batch_size if self.batch_mode else 1

        # Define the agents
        # self.actions = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=int_type)
        # self.action_names = ["right", "down", "left", "up", "stay"]
        # self.n_actions = self.actions.shape[0]
        self.n_agents = args.n_agents
        # self.agent_obs_dim = np.asarray(self.agent_obs, dtype=int_type)
        self.obs_size = args.obs_size
        self.state_size = args.state_size

        # Define the episode and rewards
        self.episode_limit = args.episode_limit

        # Define the internal state
        # self.agents = np.zeros((self.n_agents, self.batch_size, 2), dtype=int_type)
        self.steps = 0

        self.p_common = args.p_common
        self.p_observation = args.p_observation
        self.common_knowledge = 0
        self.matrix_id = 0

        self.reset()

    # ---------- INTERACTION METHODS -----------------------------------------------------------------------------------

    def reset(self):
        """ Returns initial observations and states"""
        self.payoff_values = []

        self.payoff_values.append(np.array([  # payoff values
            [5, 0, 0, 2, 0],
            [0, 1, 2, 4, 2],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 5],
        ], dtype=np.float32) * 0.1
                                  )
        self.payoff_values.append(np.array([  # payoff values
            [0, 0, 1, 0, 5],
            [0, 0, 2, 0, 0],
            [1, 2, 4, 2, 1],
            [0, 0, 2, 0, 0],
            [5, 0, 1, 0, 0],
        ], dtype=np.float32) * 0.1
                                  )

        self.step = 0
        self.n_actions = len(self.payoff_values[0])

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Returns reward, terminated, info """

        self.reward = self.payoff_values[self.matrix_id][actions]

        info = {}
        self.steps += 1
        terminated = False
        if self.steps >= self.episode_limit:
            terminated = [True for _ in range(self.batch_size)]
            info["episode_limit"] = True
        else:
            info["episode_limit"] = False

        return self.reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        if self.common_knowledge == 1:
            observations = np.repeat(self.matrix_id, 2)
        else:
            observations = np.ones(self.n_agents) * 3  # -1: unobserved

            for a in range(self.n_agents):
                if np.random.random() < self.p_observation:
                    observations[a] += self.matrix_id

        return observations[agent_id].astype(float)

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.obs_size

    def get_state(self):
        if np.random.random() < self.p_common:
            self.common_knowledge = 1
        else:
            self.common_knowledge = 0
        #     print(common_knowledge)
        self.matrix_id = np.random.randint(0, 2)

        return np.array([self.matrix_id, self.common_knowledge], dtype=np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.state_size

    def get_avail_actions(self):
        return np.ones(self.n_actions)

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

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