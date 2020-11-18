import gym
import ma_gym
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np

class MaGymWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        env_name = kwargs["args"].env
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Debug testing
        # ma_env = gym.make("Checkers-v0")
        # Just under 40 seems to be optimal for Checkers-v0
        # ma_env = gym.make("Combat-v0")
        # ma_env = gym.make("Switch4-v0")
        # ma_env = gym.make("PredatorPrey7x7-v0")
        # ma_env = gym.make("PredatorPrey7x7P-2-v0")
        ma_env = gym.make(env_name)
        self.ma_env = ma_env

        self.r_func = lambda x: sum(x) # Sum the rewards
        if "PredatorPrey" in env_name:
            self.r_func = lambda x: x[0]

        # Define the agents and actions
        self.n_agents = ma_env.n_agents
        self.n_actions = ma_env.action_space[0].n
        self.episode_limit = ma_env._max_steps

        self.obs_size = ma_env._obs_low.shape[0]

        self.obs = None
        self.state = None

        self.ma_env.reset()
        self._build_mapping()

        self.debug_render = False
        # self.debug_render = True

    def _build_mapping(self):
        x = 0
        mapping = {}
        flat_state = np.array(self.ma_env._full_obs).flatten()
        for elem in flat_state:
            if elem not in mapping:
                mapping[elem] = x
                x += 1
        self.mapping = mapping
        for k in self.mapping.keys():
            self.mapping[k] = self.mapping[k] / (x-1)

    def _convert_state_to_num(self, s):
        return np.array([self.mapping[x] for x in s])

    def reset(self):
        """ Returns initial observations and states"""
        self.ma_env.reset()
        self.dones = [False for _ in range(self.n_agents)]
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Returns reward, terminated, info """
        obs, rewards, dones, infos = self.ma_env.step(actions)
        self.dones = dones
        reward = self.r_func(rewards)

        terminated = all(dones)
        info = {}
        for k in infos.keys():
            if k == "health":
                continue
            elif k == "prey_alive":
                info[k] = sum(infos[k])
            elif type(infos[k]) == dict:
                info = {**info, **infos[k]}
            else:
                info[k] = infos[k]
        if terminated and self.ma_env._step_count >= self.ma_env._max_steps:
            info["episode_limit"] = False

        if self.debug_render:
            self.ma_env.render()

        return reward, terminated, info

    def get_obs(self):
        return self.ma_env.get_agent_obs()

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.obs_size

    def get_state(self):
        return self._convert_state_to_num(np.array(self.ma_env._full_obs).flatten())

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        if self.dones[agent_id] is False:
            return np.ones(self.n_actions)
        else:
            # Cache this?
            return np.array([1 if meaning == "NOOP" else 0 for meaning in self.ma_env.get_action_meanings()[agent_id]])


    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def get_stats(self):
        return None

