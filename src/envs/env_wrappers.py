import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque

from smac.env import MultiAgentEnv, StarCraft2Env

# Reference: https://github.com/Denys88/rl_games/blob/master/common/wrappers.py
#            baselines.common.atari_wrappers.LazyFrames

class FrameStackStartCraft2Env():
    def __init__(self, **kwargs):
        """
        Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        kwargs_copy = kwargs.copy()

        self.k = kwargs_copy.pop("framestack_num")
        self.frames_obs = deque([], maxlen=self.k)
        self.frames_state = deque([], maxlen=self.k)
        self.frames_avail_actions = deque([], maxlen=self.k)
        self.transpose = kwargs_copy.pop("framestack_transpose")
        self.flatten = kwargs_copy.pop("framestack_flatten")

        self.sc2_env = StarCraft2Env(**kwargs_copy)

    def reset(self):
        self.sc2_env.reset()
        ob = self.sc2_env.get_obs()
        state = self.sc2_env.get_state()
        avail_actions = self.sc2_env.get_avail_actions()
        for _ in range(self.k):
            self.frames_obs.append(ob)
            self.frames_state.append(state)
            self.frames_avail_actions.append(avail_actions)
    
    def _transform(self, _input_frames):
        input_frames = np.array(_input_frames)
        shape = np.shape(input_frames)
        frames = np.transpose(input_frames, (1, 0, 2))
        frames = np.reshape(frames, (shape[1], shape[0] * shape[2]))
        return frames

    def step(self, actions):
        self.sc2_env.step(actions)
        # update the state
        new_state_frame = self.sc2_env.get_state()
        self.frames_state.append(new_state_frame)
        # update the observation
        new_obs_frame = self.sc2_env.get_obs()
        self.frames_obs.append(new_obs_frame)
        # update avail actions
        new_avail_actions_frame = self.sc2_env.get_avail_actions()
        self.frames_avail_actions.append(new_avail_actions_frame)

    def get_state(self):
        assert len(self.frames_state) == self.k
        frames_state = np.array(self.frames_state)
        return frames_state.flatten()

    def get_obs(self):
        assert len(self.frames_obs) == self.k
        return self._transform(self.frames_obs)
        # return self.frames_obs

    def get_avail_actions(self):
        assert len(self.frames_avail_actions) == self.k
        return self._transform(self.frames_avail_actions)
        # return self.frames_avail_actions

    def get_env_info(self):
        return self.sc2_env.get_env_info()
    
    def close(self):
        return self.sc2_env.close()

    def get_stats(self):
        return self.sc2_env.get_stats()
