from envs.multiagentenv import MultiAgentEnv
import numpy as np
import pygame
from utils.dict2namedtuple import convert

'''
A simple grid-world game for N agents trying to capture M prey and M' hares. 
No two entities can occupy the same position. The world can be either toroidal or bounded.

INTERSECTION OF OBSERVATIONS:
This version of predator prey also includes the observation method get_obs_intersection(agent_ids, global_view=False),
which returns a numpy array either as of size batch_size*state_dim (if global_view=True) or of size 
len(agent_ids)*batch_size*observation_dim (if global_view=False), where batch_size can be omitted if it is 
not specified in the constructor. The array contains the intersection of the observation of all agents 
in the list of agent_ids, either as state or as agent observation (centered around the respective agent),
but with the difference that the agent and prey (ID+1) are given, instead of just their presence (1).
Standard agent observations can be recovered by masking array>0.   

MOVEMENTS
Both predators and prey can move to the 4 adjacent states or remain in the current one. Movement is executed 
sequentially: first the predators move in a random order, then the prey chooses a random available action 
(i.e. an action that would not lead to a collision with another entity). 
A prey is captured if it cannot move (i.e. if 4 agents block all 4 adjacent fields).

REWARDS 
A captured prey is removed and yields a collaborative reward of +50. 
Forcing the a prey to move (scaring it off), by moving into the same field yields no additional reward. 
Collisions between agents is not punished, and each movement costs additional -0.1 reward. 
An episode ends if all prey have been captured.  

OBSERVATIONS
Prey only react to blocked movements (after the predators have moved), but predator agents observe all positions 
in a square of obs_size=(2*agent_obs+1) centered around the agent's current position. The observations are reshaped 
into a 1d vector of size (2*obs_size), including all predators and prey the agent can observe.

State output is a list of length 2, giving location of all agents and all targets.

TODO: Fine tune the reward to allow fast learning. 
'''

int_type = np.int16
float_type = np.float32


class StagHunt(MultiAgentEnv):

    # ---------- CONSTRUCTOR -------------------------------------------------------------------------------------------
    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Downwards compatibility of batch_mode
        self.batch_mode = batch_size is not None
        self.batch_size = batch_size if self.batch_mode else 1

        # Define the environment grid
        self.observe_ids = getattr(args, "observe_ids", False)
        self.intersection_global_view = getattr(args, "intersection_global_view", False)
        self.intersection_unknown = getattr(args, "intersection_unknown", False)
        self.observe_walls = getattr(args, "observe_walls", True)
        self.observe_one_hot = getattr(args, "observe_one_hot", False)
        self.n_feats = 5 if self.observe_one_hot else 3
        self.toroidal = args.toroidal
        shape = args.world_shape
        self.x_max, self.y_max = shape
        self.state_size = self.x_max * self.y_max * self.n_feats
        self.env_max = np.asarray(shape, dtype=int_type)
        self.grid_shape = np.asarray(shape, dtype=int_type)
        self.grid = np.zeros((self.batch_size, self.x_max, self.y_max, self.n_feats), dtype=float_type)
        # 0=agents, 1=stag, 2=hare, [3=wall, 4=unknown]

        # Define the agents
        self.actions = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=int_type)
        self.action_names = ["right", "down", "left", "up", "stay"]
        self.n_actions = self.actions.shape[0]
        self.n_agents = args.n_agents
        self.n_stags = args.n_stags
        self.p_stags_rest = args.p_stags_rest
        self.n_hare = args.n_hare
        self.p_hare_rest = args.p_hare_rest
        self.n_prey = self.n_stags + self.n_hare
        self.agent_obs = args.agent_obs
        self.agent_obs_dim = np.asarray(self.agent_obs, dtype=int_type)
        self.obs_size = self.n_feats*(2*args.agent_obs[0]+1)*(2*args.agent_obs[1]+1)

        # Define the episode and rewards
        self.episode_limit = args.episode_limit
        self.time_reward = getattr(args, "reward_time", -0.1)
        self.collision_reward = getattr(args, "reward_collision", 0.0)
        self.capture_hare_reward = getattr(args, "reward_hare", 1.0)
        self.capture_stag_reward = getattr(args, "reward_stag", 2.0)
        self.capture_terminal = getattr(args, "capture_terminal", True)

        # Define the internal state
        self.agents = np.zeros((self.n_agents, self.batch_size, 2), dtype=int_type)
        self.prey = np.zeros((self.n_prey, self.batch_size, 2), dtype=int_type)
        self.prey_alive = np.zeros((self.n_prey, self.batch_size), dtype=int_type)
        self.prey_type = np.ones((self.n_prey, self.batch_size), dtype=int_type)    # fill with stag (1)
        self.prey_type[self.n_stags:, :] = 2    # set hares to 2
        self.steps = 0
        self.reset()

        self.made_screen = False
        self.scaling = 5

    # ---------- INTERACTION METHODS -----------------------------------------------------------------------------------
    def reset(self):
        # Reset old episode
        self.prey_alive.fill(1)
        self.steps = 0

        # Clear the grid
        self.grid.fill(0.0)

        # Place n_agents and n_preys on the grid
        self._place_actors(self.agents, 0)
        self._place_actors(self.prey[:self.n_stags, :, :], 1)
        self._place_actors(self.prey[self.n_stags:, :, :], 2)

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Execute a*bs actions in the environment. """
        if not self.batch_mode:
            actions = np.expand_dims(np.asarray(actions, dtype=int_type), axis=1)
        assert len(actions.shape) == 2 and actions.shape[0] == self.n_agents and actions.shape[1] == self.batch_size, \
            "improper number of agents and/or parallel environments!"
        actions = actions.astype(dtype=int_type)

        # Initialise returned values and grid
        reward = np.ones(self.batch_size, dtype=float_type) * self.time_reward
        terminated = [False for _ in range(self.batch_size)]

        # Move the agents sequentially in random order
        for b in range(self.batch_size):
            for a in np.random.permutation(self.n_agents):
                self.agents[a, b, :], collide = self._move_actor(self.agents[a, b, :], actions[a, b], b,
                                                                 np.asarray([0], dtype=int_type), 0)
                if collide:
                    reward[b] = reward[b] + self.collision_reward

        # Move the prey
        for b in range(self.batch_size):
            for p in np.random.permutation(self.n_prey):
                if self.prey_alive[p, b] > 0:
                    # Collect all allowed actions for the prey
                    possible = []
                    # Run through all potential actions (without actually moving), except stay(4)
                    for u in range(self.n_actions-1):
                        _, c = self._move_actor(self.prey[p, b, :], u, b, np.asarray([0, 1, 2], dtype=int_type))
                        if not c:
                            possible.append(u)
                    # Stags are caught when they cannot move to any adjacent position
                    captured = (self.prey_type[p, b] == 1) and (len(possible) <= 0)
                    # Hares play dead when they would have only one position left and are caught
                    captured = captured or ((self.prey_type[p, b] == 2) and (len(possible) <= 1))
                    # If the prey is captured, remove it from the grid and terminate episode if specified
                    if captured:
                        self.prey_alive[p, b] = 0
                        self.grid[b, self.prey[p, b, 0], self.prey[p, b, 1], self.prey_type[p, b]] = 0
                        terminated[b] = terminated[b] or self.capture_terminal
                        reward[b] += self.capture_stag_reward if self.prey_type[p, b] == 1 else 0
                        reward[b] += self.capture_hare_reward if self.prey_type[p, b] == 2 else 0
                    else:
                        # If not, check if the prey can rest and if so determine randomly whether it wants to
                        rest = (self.grid[b, self.prey[p, b, 0], self.prey[p, b, 1], 0] == 0) and \
                               (np.random.rand() < (self.p_stags_rest if self.prey_type[p] == 1 else self.p_hare_rest))
                        # If the prey decides not to rest, choose a movement action randomly
                        if not rest:
                            u = possible[np.random.randint(len(possible))]
                            self.prey[p, b, :], _ = self._move_actor(self.prey[p, b, :], u, b,
                                                                     np.asarray([0, 1, 2], dtype=int_type),
                                                                     self.prey_type[p, b])
            terminated[b] = terminated[b] or sum(self.prey_alive[:, b]) == 0

        # Terminate if episode_limit is reached
        info = {}
        self.steps += 1
        if self.steps >= self.episode_limit:
            terminated = [True for _ in range(self.batch_size)]
            info["episode_limit"] = True
        else:
            info["episode_limit"] = False

        if self.batch_mode:
            return reward, terminated, info
        else:
            return np.asscalar(reward[0]), int(terminated[0]), info

    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    def get_obs_agent(self, agent_id, batch=0):
        #return self._get_obs_from_grid(self.grid, agent_id, batch)
        obs, _ = self._observe([agent_id])
        return obs

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        if self.batch_mode:
            return self.grid.copy().reshape(self.state_size)
        else:
            return self.grid[0, :, :, :].reshape(self.state_size)

    def get_obs_intersect_pair_size(self):
        return 2 * self.get_obs_size()

    def get_obs_intersect_all_size(self):
        return self.n_agents * self.get_obs_size()

    def get_obs_intersection(self, agent_ids):
        return self._observe(agent_ids)

    # ---------- GETTERS -----------------------------------------------------------------------------------------------
    def get_total_actions(self):
        return self.n_actions

    def get_avail_agent_actions(self, agent_id):
        """ Currently runs only with batch_size==1. """
        if self.toroidal:
            return [1 for _ in range(self.n_actions)]
        else:
            new_pos = self.agents[agent_id, 0, :] + self.actions
            allowed = np.logical_and(new_pos >= 0, new_pos < self.grid_shape).all(axis=1)
            assert np.any(allowed), "No available action in the environment: this should never happen!"
            return [int(allowed[a]) for a in range(self.n_actions)]

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

    # --------- RENDER METHODS -----------------------------------------------------------------------------------------
    def close(self):
        if self.made_screen:
            pygame.quit()
        print("Closing Multi-Agent Navigation")

    def render_array(self):
        # Return an rgb array of the frame
        return None

    def render(self):
        # TODO!
        pass

    def seed(self):
        raise NotImplementedError

# ---------- PRIVATE METHODS ---------------------------------------------------------------------------------------
    def _place_actors(self, actors: np.ndarray, type_id: int):
        for b in range(self.batch_size):
            for a in range(actors.shape[0]):
                is_free = False
                while not is_free:
                    # Draw actors's position randomly
                    actors[a, b, 0] = np.random.randint(self.env_max[0])
                    actors[a, b, 1] = np.random.randint(self.env_max[1])
                    # Check if position is valid
                    is_free = np.sum(self.grid[b, actors[a, b, 0], actors[a, b, 1], :]) == 0
                self.grid[b, actors[a, b, 0], actors[a, b, 1], type_id] = 1

    def print_grid(self, batch=0, grid=None):
        if grid is None:
            grid = self.grid
        grid = grid[batch, :, :, :].squeeze().copy()
        for i in range(grid.shape[2]):
            grid[:, :, i] *= i + 1
        grid = np.sum(grid, axis=2)
        print(grid)

    def print_agents(self, batch=0):
        obs = np.zeros((self.grid_shape[0], self.grid_shape[1]))
        for a in range(self.n_agents):
            obs[self.agents[a, batch, 0], self.agents[a, batch, 1]] = a + 1
        for p in range(self.n_prey):
            if self.prey_alive[p]:
                obs[self.prey[p, batch, 0], self.prey[p, batch, 1]] = -p - 1
        print(obs)

    def _env_bounds(self, positions: np.ndarray):
        # positions is 2 dimensional
        if self.toroidal:
            positions = positions % self.env_max
        else:
            positions = np.minimum(positions, self.env_max - 1)
            positions = np.maximum(positions, 0)
        return positions

    def _move_actor(self, pos: np.ndarray, action: int, batch: int, collision_mask: np.ndarray, move_type=None):
        # compute hypothetical next position
        new_pos = self._env_bounds(pos + self.actions[action])
        # check for a collision with anything in the collision_mask
        found_at_new_pos = self.grid[batch, new_pos[0], new_pos[1], :]
        collision = np.sum(found_at_new_pos[collision_mask]) > 0
        if collision:
            # No change in position
            new_pos = pos
        elif move_type is not None:
            # change the agent's state and position on the grid
            self.grid[batch, pos[0], pos[1], move_type] = 0
            self.grid[batch, new_pos[0], new_pos[1], move_type] = 1
        return new_pos, collision

    def _is_visible(self, agents, target):
        """ agents are plural and target is singular. """
        target = target.reshape(1, 2).repeat(agents.shape[0], 0)
        # Determine the Manhattan distance of all agents to the target
        if self.toroidal:
            # Account for the environment wrapping around in a toroidal fashion
            lower = np.minimum(agents, target)
            higher = np.maximum(agents, target)
            d = np.abs(np.minimum(higher - lower, lower - higher + self.grid_shape))
        else:
            # Account for the environment being bounded
            d = np.abs(agents - target)
        # Return true if all targets are visible by all agents
        return np.all(d <= self.agent_obs)

    def _intersect_targets(self, grid, agent_ids, targets, batch=0, target_id=0, targets_alive=None, offset=0):
        """" Helper for get_obs_intersection(). """
        for a in range(targets.shape[0]):
            marker = a + 1 if self.observe_ids else 1
            if targets_alive is None or targets_alive[a, batch]:
                # If the target is visible to all agents
                if self._is_visible(self.agents[agent_ids, batch, :], targets[a, batch, :]):
                    # include the target in all observations (in relative positions)
                    for o in range(len(agent_ids)):
                        grid[batch, targets[a, batch, 0] + offset, targets[a, batch, 1] + offset, target_id] = marker

    def _observe(self, agent_ids):
        # Compute available actions
        if len(agent_ids) == 1:
            avail_all = self.get_avail_agent_actions(agent_ids[0])
        elif len(agent_ids) == 2:
            a_a1 = np.reshape(np.array(self.get_avail_agent_actions(agent_ids[0])), [-1, 1])
            a_a2 = np.reshape(np.array(self.get_avail_agent_actions(agent_ids[1])), [1, -1])
            avail_actions = a_a1.dot(a_a2)
            avail_all = avail_actions * 0 + 1
        else:
            avail_all = []
        # Create over-sized grid
        ashape = np.array(self.agent_obs)
        ushape = self.grid_shape + 2 * ashape
        grid = np.zeros((self.batch_size, ushape[0], ushape[1], self.n_feats), dtype=float_type)
        # Make walls
        if self.observe_walls:
            wall_dim = 3 if self.observe_one_hot else 0
            wall_id = 1 if self.observe_one_hot else -1
            grid[:, :ashape[0], :, wall_dim] = wall_id
            grid[:, (self.grid_shape[0]+ashape[0]):, :, wall_dim] = wall_id
            grid[:, :, :ashape[1], wall_dim] = wall_id
            grid[:, :, (self.grid_shape[1] + ashape[1]):, wall_dim] = wall_id
        # Mark the grid with all intersected entities
        noinformation = False
        for b in range(self.batch_size):
            if all([self._is_visible(self.agents[agent_ids, b, :], self.agents[agent_ids[a], b, :])
                    for a in range(len(agent_ids))]):
                # Every agent sees other intersected agents
                self._intersect_targets(grid, agent_ids, targets=self.agents, batch=b, target_id=0, offset=ashape)
                # Every agent sees intersected stags
                self._intersect_targets(grid, agent_ids, targets=self.prey[:self.n_stags, :, :], batch=b, target_id=1,
                                        targets_alive=self.prey_alive, offset=ashape)
                # Every agent sees intersected hares
                self._intersect_targets(grid, agent_ids, targets=self.prey[self.n_stags:, :, :], batch=b, target_id=2,
                                        targets_alive=self.prey_alive, offset=ashape)
            else:
                noinformation = True
        # Mask out all unknown if specified
        if self.intersection_unknown:
            for b in range(self.batch_size):
                for a in agent_ids:
                    self._mask_agent(grid, self.agents[a, b, :] + ashape, ashape)

        if self.intersection_global_view:
            # In case of the global view
            obs = grid[:, ashape[0]:(ashape[0] + self.grid_shape[0]), ashape[1]:(ashape[1] + self.grid_shape[1]), :]
            obs = obs.reshape((1, self.batch_size, self.state_size))
        else:
            # otherwise local view
            obs = np.zeros((len(agent_ids), self.batch_size, 2*ashape[0]+1, 2*ashape[1]+1, self.n_feats),
                           dtype=float_type)
            for b in range(self.batch_size):
                for i, a in enumerate(agent_ids):
                    obs[i, b, :, :, :] = grid[b, (self.agents[a, b, 0]):(self.agents[a, b, 0] + 2*ashape[0] + 1),
                                              (self.agents[a, b, 1]):(self.agents[a, b, 1] + 2*ashape[1] + 1), :]
            obs = obs.reshape(len(agent_ids), self.batch_size, -1)

        # Final check: if not all agents can see each other, the mutual knowledge is empty
        if noinformation:
            if self.intersection_unknown:
                obs = obs.reshape(obs.shape[0], obs.shape[1], obs.shape[2] // self.n_feats, self.n_feats)
                unknown_dim = 4 if self.observe_one_hot else 1
                unknown_id = 1 if self.observe_one_hot else -1
                obs.fill(0.0)
                obs[:, :, :, unknown_dim] = unknown_id
                obs = obs.reshape(obs.shape[0], obs.shape[1], self.n_feats * obs.shape[2])
            else:
                obs = 0 * obs

        # Return considering batch-mode
        if self.batch_mode:
            return obs, avail_all
        else:
            return obs[:, 0, :].squeeze(), avail_all

    def _mask_agent(self, grid, pos, ashape):
        unknown_dim = 4 if self.observe_one_hot else 1
        unknown_id = 1 if self.observe_one_hot else -1
        grid[:, :(pos[0] - ashape[0]), :, :].fill(0.0)
        grid[:, :(pos[0] - ashape[0]), :, unknown_dim] = unknown_id
        grid[:, (pos[0] + ashape[0] + 1):, :, :].fill(0.0)
        grid[:, (pos[0] + ashape[0] + 1):, :, unknown_dim] = unknown_id
        grid[:, :, :(pos[1] - ashape[1]), :].fill(0)
        grid[:, :, :(pos[1] - ashape[1]), unknown_dim] = unknown_id
        grid[:, :, (pos[1] + ashape[1] + 1):, :].fill(0.0)
        grid[:, :, (pos[1] + ashape[1] + 1):, unknown_dim] = unknown_id

    def _get_obs_from_grid(self, grid, agent_id, batch=0):
        """ OBSOLETE! """
        if self.toroidal:
            return self._get_obs_from_grid_troidal(grid, agent_id, batch)
        else:
            return self._get_obs_from_grid_bounded(grid, agent_id, batch)

    def _get_obs_from_grid_bounded(self, grid, agent_id, batch=0):
        """ Return a bounded observation for other agents' locations and targets, the size specified by observation
            shape, centered on the agent. Values outside the bounds of the grid are set to 0.
            OBSOLETE! """
        # Create the empty observation grid
        agent_obs = np.zeros((2*self.agent_obs[0]+1, 2*self.agent_obs[1]+1, 3), dtype=float_type)
        # Determine the unbounded limits of the agent's observation
        ul = self.agents[agent_id, batch, :] - self.agent_obs
        lr = self.agents[agent_id, batch, :] + self.agent_obs
        # Bound the limits to the grid
        bul = np.maximum(ul, [0, 0])
        blr = np.minimum(lr, self.grid_shape - 1)
        # Compute the ranges in x/y direction in the agent's observation
        bias = bul - ul
        aoy = [bias[0], blr[0] - bul[0] + bias[0]]
        aox = [bias[1], blr[1] - bul[1] + bias[1]]
        # Copy the bounded observations
        agent_obs[aoy[0]:(aoy[1]+1), aox[0]:(aox[1]+1), :] = grid[batch, bul[0]:(blr[0]+1), bul[1]:(blr[1]+1), :]
        return np.reshape(agent_obs, self.obs_size)

    def _get_obs_from_grid_troidal(self, grid, agent_id, batch=0):
        """ Return a wrapped observation for other agents' locations and targets, the size specified by observation
            shape, centered on the agent.
            OBSOLETE! """
        a_x, a_y = self.agents[agent_id, batch, :]
        o_x, o_y = self.agent_obs
        x_range = range((a_x - o_x), (a_x + o_x + 1))
        y_range = range((a_y - o_y), (a_y + o_y + 1))
        ex_grid = grid[batch, :, :, :].astype(dtype=float_type)
        agent_obs = ex_grid.take(x_range, 0, mode='wrap').take(y_range, 1, mode='wrap')
        return np.reshape(agent_obs, self.obs_size)

    def _get_obs_intersection_old(self, agent_ids):
        """ Returns the intersection of the all of agent_ids agents' observations.
            OBSOLETE, only maintained for legacy issues! """
        # Create grid
        grid = np.zeros((self.batch_size, self.grid_shape[0], self.grid_shape[0], 3), dtype=float_type)

        a_a1 = np.reshape( np.array(self.get_avail_agent_actions(agent_ids[0])),[-1,1])
        a_a2 = np.reshape( np.array(self.get_avail_agent_actions(agent_ids[1])),[1,-1])
        avail_actions = a_a1.dot(a_a2)
        avail_all = avail_actions * 0 + 1
        # If all agent_ids can see each other (otherwise the observation is empty)
        for b in range(self.batch_size):
            if all([self._is_visible(self.agents[agent_ids, b, :], self.agents[agent_ids[a], b, :])
                    for a in range(len(agent_ids))]):
                # Every agent sees other intersected agents
                self._intersect_targets(grid, agent_ids, targets=self.agents, batch=b, target_id=0)
                # Every agent sees intersected prey
                self._intersect_targets(grid, agent_ids, targets=self.prey, batch=b, target_id=1,
                                        targets_alive=self.prey_alive)
                avail_all = avail_actions
        # Return 0-1 encoded intersection if necessary
        if not self.observe_ids:
            grid = (grid != 0.0).astype(np.float32)
        # The intersection grid is constructed, now we have to generate the observations from it
        if self.intersection_global_view:
            # Return the intersection as a state
            if self.batch_mode:
                return grid.reshape((self.batch_size, self.state_size)), avail_all
            else:
                return grid[0, :, :, :].reshape(self.state_size), avail_all
        else:
            # Return the intersection as individual observations
            obs = np.zeros((len(agent_ids), self.batch_size, self.obs_size),
                           dtype=float_type)
            for b in range(self.batch_size):
                for a in range(len(agent_ids)):
                    obs[a, b, :] = self._get_obs_from_grid(grid, a, b)
            if self.batch_mode:
                return obs, avail_all
            else:
                return obs[:, 0, :], avail_all


# ######################################################################################################################
if __name__ == "__main__":
    env_args = {
        'world_shape': (6, 6),
        'toroidal': False,
        'observe_walls': False,
        'observe_ids': True,
        'observe_one_hot': True,
        'intersection_global_view': False,
        'intersection_unknown': True,
        'reward_hare': 1,
        'reward_stag': 10,
        'reward_collision': 0.0,
        'reward_time': -0.1,
        'capture_terminal': True,
        'episode_limit': 50,
        'n_stags': 1,
        'p_stags_rest': 0.1,
        'n_hare': 1,
        'p_hare_rest': 0.5,
        'n_agents': 3,
        'agent_obs': (2, 2),
    }
    env_args = convert(env_args)
    print(env_args)

    env = StagHunt(env_args=env_args)
    [all_obs, state] = env.reset()
    print("Env is ", "batched" if env.batch_mode else "not batched")

    if False:
        print(state)
        for i in range(env.n_agents):
            print(all_obs[i])

        acts = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]]).transpose()
        env.step(acts[:, 0])

        env.print_grid()
        obs = []
        for i in range(4):
            obs.append(np.expand_dims(env.get_obs_agent(i), axis=1))
        print(np.concatenate(obs, axis=1))

    if True:
        # Test observation with local view
        print("STATE:\n")
        env.print_agents()
        print()
        state_shape = (env_args.world_shape[0], env_args.world_shape[1], 2)
        obs_shape = (2*env_args.agent_obs[0] + 1, 2*env_args.agent_obs[1] + 1)
        obs = env.get_obs()

        print("\n\nOBSERVATIONS of", env.n_agents, " agents:\n")
        for a in range(env.n_agents):
            obs[a] = obs[a].reshape(obs_shape[0], obs_shape[1], env.n_feats)
            visualisation = obs[a][:, :, 0] + 10 * obs[a][:, :, 1] + 100 * obs[a][:, :, 2]
            visualisation -= 0 if not env.observe_one_hot else obs[a][:, :, 3] + 10 * obs[a][:, :, 4]
            print(visualisation, "\n")

    if False:
        # Test intersection with local view
        print("STATE:\n")
        env.print_agents()
        print()
        state_shape = (env_args.world_shape[0], env_args.world_shape[1], 2)
        obs_shape = (2*env_args.agent_obs[0] + 1, 2*env_args.agent_obs[1] + 1)
        agent_ids = [0, 1]
        iobs, _ = env.get_obs_intersection(agent_ids)
        iobs = iobs.reshape(len(agent_ids), obs_shape[0], obs_shape[1], env.n_feats)

        print("\n\nINTERSECTIONS of", agent_ids, "\n")
        for a in range(len(agent_ids)):
            visualisation = iobs[a, :, :, 0] + 10 * iobs[a, :, :, 1] + 100 * iobs[a, :, :, 2]
            visualisation -= 0 if not env.observe_one_hot else iobs[a, :, :, 3] + 10 * iobs[a, :, :, 4]
            print(visualisation, "\n")

    if False:
        # Test intersection with global view
        print("STATE:\n")
        env.print_agents()
        print()
        state_shape = (env_args.world_shape[0], env_args.world_shape[1])
        obs_shape = (2*env_args.agent_obs[0] + 1, 2*env_args.agent_obs[1] + 1)
        agent_ids = [0, 1]
        iobs, _ = env.get_obs_intersection(agent_ids)
        iobs = iobs.reshape(state_shape[0], state_shape[1], 3)

        print("\n\nINTERSECTION of", agent_ids, "\n")
        print(iobs[:, :, 0].reshape(state_shape) + 10 * iobs[:, :, 1].reshape(state_shape)
              + 100 * iobs[:, :, 2].reshape(state_shape), "\n")

    if False:
        env.print_agents()
        print(env.get_avail_actions())

    if False:
        env.print_agents()
        print()
        for _ in range(10):
            acts = (np.random.rand(3)*5) // 1
            print(acts)
            env.step(acts)
            env.print_agents()
            for a in range(env.n_agents):
                print(env.get_avail_agent_actions(a))
            print()
