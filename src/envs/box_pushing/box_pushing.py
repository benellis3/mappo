from envs.multiagentenv import MultiAgentEnv
import numpy as np
import torch
import pygame
from utils.dict2namedtuple import convert

int_type = np.int16
float_type = np.float32

'''
A simple grid-world game for N agents trying to capture M prey. No two entities can occupy the same position. 
The world can be either toroidal or bounded.

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


# if torch.cuda.is_available():
#     Tensor = torch.cuda.FloatTensor
#     lTensor = torch.cuda.LongTensor
# else:
Tensor = torch.FloatTensor
lTensor = torch.LongTensor


class CoopBoxPushing(MultiAgentEnv):

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
        self.intersection_id_coded = getattr(args, "intersection_id_coded", False)
        self.intersection_global_view = getattr(args, "intersection_global_view", False)
        self.intersection_unknown = getattr(args, "intersection_unknown", False)
        self.toroidal = args.predator_prey_toroidal

        shape = args.predator_prey_shape
        self.x_max, self.y_max = shape
        self.state_size = self.x_max * self.y_max * 3
        self.env_max = lTensor(shape)
        self.grid_shape = np.asarray(shape)
        # channels are 0: agents, 1: small box, 2: big box NB: no goal in state, add it if endzone changes
        # 0: num agents in each square, 1-2: ID of box in location
        self.grid = Tensor(self.batch_size, self.x_max, self.y_max, 3).zero_()

        # Define the agents
        self.actions = lTensor([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]])
        self.action_names = ["right", "down", "left", "up", "stay"]
        self.n_actions = self.actions.shape[0]
        self.n_agents = args.n_agents
        self.n_small = 2
        self.n_big = 1
        self.agent_obs = np.asarray(args.agent_obs, dtype=int_type)
        # self.agent_obs = torch.from_numpy(np.asarray(args.agent_obs))
        self.obs_size = 3*(2*args.agent_obs[0]+1)*(2*args.agent_obs[1]+1)

        # Define the episode and rewards
        self.episode_limit = args.episode_limit
        self.time_reward = -0.05
        self.collision_reward = 0.0 #-0.1
        self.scare_off_reward = 0.0
        #self.capture_rewards = [20, 1]
        self.r_small = 1
        self.r_big = 10
        self.r_fail = -0.5

        # Define the internal state
        self.agents = np.zeros((self.n_agents, self.batch_size, 2), dtype=int_type)
        self.n_small = 2
        self.small_boxes = np.zeros((self.n_small, self.batch_size, 2), dtype=int_type)
        self.small_boxes_done = np.zeros((self.n_small, self.batch_size), dtype=int_type)
        self.n_big = 1
        self.big_boxes = np.zeros((self.n_big, self.batch_size, 2), dtype=int_type)
        self.big_boxes_done = np.zeros((self.n_big, self.batch_size), dtype=int_type)
        self.steps = 0
        self.reset()

        self.made_screen = False
        self.scaling = 5

    # ---------- PRIVATE METHODS ---------------------------------------------------------------------------------------
    def _place_actors(self, actors: lTensor, type_id: int):
        for b in range(self.batch_size):
            for a in range(actors.shape[0]):
                is_free = False
                while not is_free:
                    # Draw actors's position randomly (left wall for agents, first column for box)
                    actors[a, b, 0] = 0 if type_id == 0 else 1
                    actors[a, b, 1] = np.random.randint(self.env_max[1])
                    # Check if position is valid
                    if type_id == 0:
                        is_free = True # agents can (must be able to) overlap
                    else:
                        is_free = torch.sum(self.grid[b, actors[a, b, 0], actors[a, b, 1], :]) == 0
                if type_id == 0:
                    self.grid[b, actors[a, b, 0], actors[a, b, 1], type_id] += 1
                else:
                    self.grid[b, actors[a, b, 0], actors[a, b, 1], type_id] = a + 1

    def print_grid(self, batch=0, grid=None):
        if grid is None:
            grid = self.grid
        grid = grid[batch, :, :, :].squeeze().clone()
        for i in range(grid.shape[2]):
            grid[:, :, i].mul_(i+1)
        grid = torch.sum(grid, dim=2)
        print(grid)

    def _env_bounds(self, positions: lTensor):
        # positions is a*bs*2
        if self.toroidal:
            positions = positions % self.env_max.expand_as(positions)
        else:
            positions = torch.min(positions, (self.env_max-1).expand_as(positions))
            positions = torch.max(positions, lTensor(1).zero_().expand_as(positions))
        return positions

    def _move_actor(self, pos: lTensor, action: int, batch: int, collision_mask: lTensor, move_type=None):
        # compute hypothetical next position
        new_pos = self._env_bounds(pos + self.actions[action])
        # check for a collision with anything in the collision_mask
        # found_at_new_pos = self.grid[batch, new_pos[0], new_pos[1], :]
        # collision = torch.sum(found_at_new_pos[collision_mask]) > 0
        # if collision:
            # No change in position
            # new_pos = pos
        # elif move_type is not None:
            # change the agent's state and position on the grid
        self.grid[batch, pos[0], pos[1], move_type] -= 1
        self.grid[batch, new_pos[0], new_pos[1], move_type] = 1
        return new_pos, False

    # ---------- INTERACTION METHODS -----------------------------------------------------------------------------------
    def reset(self):
        # Reset old episode
        self.small_boxes_done.fill(0)
        self.big_boxes_done.fill(0)
        self.steps = 0

        # Clear the grid
        self.grid.fill_(0.0)

        # Place n_agents and n_preys on the grid
        self._place_actors(self.agents, 0)
        self._place_actors(self.small_boxes, 1)
        self._place_actors(self.big_boxes, 2)

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Execute a*bs actions in the environment. """
        if not self.batch_mode:
            actions = lTensor(actions).unsqueeze(1)
        assert len(actions.shape) == 2 and actions.shape[0] == self.n_agents and actions.shape[1] == self.batch_size, \
            "improper number of agents and/or parallel environments!"
        actions = actions.long()

        # Initialise returned values and grid
        reward = np.ones(self.batch_size, dtype=float_type) * self.time_reward
        terminated = [False for _ in range(self.batch_size)]



        # Move the agents sequentially in random order
        for b in range(self.batch_size):
            for a in np.random.permutation(self.n_agents):
                pos = self.agents[a, b, :]
                action_id = actions[a, b]
                move_pos = self._env_bounds(pos + self.actions[action_id])

                bb_id = int(self.grid[b, move_pos[0], move_pos[1], 2]) - 1
                if (bb_id >= 0):
                    # try to move big box
                    if self.grid[b, pos[0], pos[1], 0] == self.n_agents and (actions[:, b] == action_id).all():
                        # all pushing same big box same way
                        for a_ in np.arange(0, self.n_agents):
                            box_move_pos = self._env_bounds(move_pos + self.actions[action_id])

                            if sum(self.grid[b, box_move_pos[0], box_move_pos[1], :]) > 0:
                                # box move is blocked
                                break
                            else:
                                self.agents[a_, b, :] = move_pos
                                self.grid[b, pos[0], pos[1], 0] -= 1
                                self.grid[b, move_pos[0], move_pos[1], 0] += 1
                                self.big_boxes[bb_id, b, :] = box_move_pos
                                self.grid[b, move_pos[0], move_pos[1], 2] = 0
                                self.grid[b, box_move_pos[0], box_move_pos[1], 2] = bb_id + 1
                        # have resolved all actions now, so break main agent loop
                        break
                    else:
                        reward[b] += self.r_fail

                sb_id = int(self.grid[b, move_pos[0], move_pos[1], 1]) - 1
                if (sb_id >= 0):
                    # move into small box
                    box_move_pos = self._env_bounds(move_pos + self.actions[action_id])

                    if sum(self.grid[b, box_move_pos[0], box_move_pos[1], :]) > 0:
                        # box move is blocked
                        break
                    else:
                        self.agents[a, b, :] = move_pos
                        self.grid[b, pos[0], pos[1], 0] -= 1
                        self.grid[b, move_pos[0], move_pos[1], 0] += 1

                        self.small_boxes[sb_id, b, :] = box_move_pos
                        self.grid[b, move_pos[0], move_pos[1], 1] = 0
                        self.grid[b, box_move_pos[0], box_move_pos[1], 1] = sb_id + 1
                    continue

                if (sb_id < 0) and (bb_id < 0):
                    # no boxes, just move
                    self.agents[a, b, :] = move_pos
                    self.grid[b, pos[0], pos[1], 0] -= 1
                    self.grid[b, move_pos[0], move_pos[1], 0] += 1

            for sb in range(self.n_small):
                if self.small_boxes_done[sb, b] == 0 and self.small_boxes[sb, b, 0] == self.env_max[0] - 1:
                    reward[b] += self.r_small
                    self.small_boxes_done[sb, b] = 1

            for bb in range(self.n_big):
                if self.big_boxes_done[bb, b] == 0 and self.big_boxes[bb, b, 0] == self.env_max[0] - 1:
                    reward[b] += self.r_big
                    self.big_boxes_done[bb, b] = 1

            terminated[b] = (sum(1 - self.big_boxes_done[:, b]) == 0) or (sum(1 - self.small_boxes_done[:, b]) == 0)

        # Terminate if episode_limit is reached
        info = {}
        self.steps += 1
        if self.steps >= self.episode_limit:
            terminated = [True for _ in range(self.batch_size)]
            info["episode_limit"] = True
        else:
            info["episode_limit"] = False

        # if (reward > 0).any():
        #     print("reward")

        if self.batch_mode:
            return reward, terminated, info
        else:
            return np.asscalar(reward[0]), int(terminated[0]), info

    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    # def get_obs_agent(self, agent_id, batch=0):
    #     """ Return a wrapped observation for other agents' locations and targets, the size specified by observation
    #         shape, centered on the agent. """
    #     # TODO: implement observations for self.toroidal=False
    #     a_x, a_y = self.agents[agent_id, batch, :]
    #     o_x, o_y = self.agent_obs
    #     x_range = range((a_x - o_x), (a_x + o_x + 1))
    #     y_range = range((a_y - o_y), (a_y + o_y + 1))
    #     ex_grid = self.grid[batch, :, :, :].cpu().numpy().astype(dtype=np.float32)
    #     agent_obs = ex_grid.take(x_range, 0, mode='wrap').take(y_range, 1, mode='wrap')
    #     return np.resize(agent_obs, self.obs_size)


# ----- COPY FROM predator_prey_intersect.py
    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    def get_obs_agent(self, agent_id, batch=0):
        return self._get_obs_from_grid(self.grid, agent_id, batch)

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
        if self.toroidal or not self.intersection_unknown:
            return self.get_obs_intersection1(agent_ids)
        else:
            return self.get_obs_intersection2(agent_ids)

    def get_obs_intersection2(self, agent_ids):
        # Compute available actions
        a_a1 = np.reshape(np.array(self.get_avail_agent_actions(agent_ids[0])), [-1, 1])
        a_a2 = np.reshape(np.array(self.get_avail_agent_actions(agent_ids[1])), [1, -1])
        avail_actions = a_a1.dot(a_a2)
        avail_all = avail_actions * 0 + 1
        # Create oversized grid
        ashape = np.array(self.agent_obs)
        ushape = self.grid_shape + 2 * ashape
        grid = np.zeros((self.batch_size, ushape[0], ushape[1], 3), dtype=float_type)
        # Make walls
        grid[:, :ashape[0], :, 0] = -1
        grid[:, (self.grid_shape[0]+ashape[0]):, :, 0] = -1
        grid[:, :, :ashape[1], 0] = -1
        grid[:, :, (self.grid_shape[1] + ashape[1]):, 0] = -1
        # Mark the grid with all intersected entities
        noinformation = False
        for b in range(self.batch_size):
            if all([self._is_visible(self.agents[agent_ids, b, :], self.agents[agent_ids[a], b, :])
                    for a in range(len(agent_ids))]):
                # Every agent sees other intersected agents
                self._intersect_targets(grid, agent_ids, targets=self.agents, batch=b, target_id=0, offset=ashape)
                # Every agent sees intersected prey
                self._intersect_targets(grid, agent_ids, targets=self.small_boxes, batch=b, target_id=1,
                                        targets_alive=1-self.small_boxes_done, offset=ashape)
                self._intersect_targets(grid, agent_ids, targets=self.big_boxes, batch=b, target_id=2,
                                        targets_alive=1 - self.big_boxes_done, offset=ashape)
            else:
                noinformation = True
        # Mask out all unknown
        for b in range(self.batch_size):
            for a in agent_ids:
                self._mask_agent(grid, self.agents[a, b, :] + ashape, ashape)

        if self.intersection_global_view:
            # In case of the global view
            obs = grid[:, ashape[0]:(ashape[0] + self.grid_shape[0]), ashape[1]:(ashape[1] + self.grid_shape[1]), :]
            obs = obs.reshape((1, self.batch_size, self.state_size))

        else:
            # otherwise local view
            obs = np.zeros((len(agent_ids), self.batch_size, 2*ashape[0]+1, 2*ashape[1]+1, 2), dtype=float_type)
            for b in range(self.batch_size):
                for i, a in enumerate(agent_ids):
                    obs[i, b, :, :, :] = grid[b, (self.agents[a, b, 0]):(self.agents[a, b, 0] + 2*ashape[0] + 1),
                                              (self.agents[a, b, 1]):(self.agents[a, b, 1] + 2*ashape[1] + 1), :]
            obs = obs.reshape(len(agent_ids), self.batch_size, -1)

        # Final check: if not all agents can see each other, the mutual knowledge is empty
        if noinformation:
            obs = obs.reshape(obs.shape[0], obs.shape[1], obs.shape[2] // 3, 3)
            obs[:, :, :, 0] = 0
            obs[:, :, :, 1] = -1
            obs[:, :, :, 2] = -1
            obs = obs.reshape(obs.shape[0], obs.shape[1], 3 * obs.shape[2])

        # Return considering batch-mode
        if self.batch_mode:
            return obs, avail_all
        else:
            return obs[:, 0, :].squeeze(), avail_all

    def _mask_agent(self, grid, pos, ashape):
        grid[:, :(pos[0] - ashape[0]), :, 0] = 0
        grid[:, :(pos[0] - ashape[0]), :, 1] = -1
        grid[:, (pos[0] + ashape[0]+1):, :, 0] = 0
        grid[:, (pos[0] + ashape[0]+1):, :, 1] = -1
        grid[:, :, :(pos[1] - ashape[1]), 0] = 0
        grid[:, :, :(pos[1] - ashape[1]), 1] = -1
        grid[:, :, (pos[1] + ashape[1] + 1):, 0] = 0
        grid[:, :, (pos[1] + ashape[1] + 1):, 1] = -1

    def get_obs_intersection1(self, agent_ids):
        """ Returns the intersection of the all of agent_ids agents' observations. """
        # Create grid
        grid = np.zeros((self.batch_size, self.grid_shape[0], self.grid_shape[1], 3), dtype=float_type)
        noinformation = False

        # Compute available actions
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
                self._intersect_targets(grid, agent_ids, targets=self.small_boxes, batch=b, target_id=1,
                                        targets_alive=1 - self.small_boxes_done)
                self._intersect_targets(grid, agent_ids, targets=self.big_boxes, batch=b, target_id=2,
                                        targets_alive=1 - self.big_boxes_done)
                avail_all = avail_actions
            else:
                noinformation = True

        # Unknown positions are encoded [0, -1]
        if self.intersection_unknown:
           # All agents remove the grid-positions they cannot see
            for b in range(self.batch_size):
                for a in range(len(agent_ids)):
                    # kill upper quadrant
                    if self.agents[a, b, 0] > self.agent_obs[0]:
                        grid[b, :(self.agents[a, b, 0] - self.agent_obs[0]), :, 0] = 0
                        grid[b, :(self.agents[a, b, 0] - self.agent_obs[0]), :, 1] = -1
                    # Kill the lower quadrant
                    if self.agents[a, b, 0] < self.grid_shape[0] - self.agent_obs[0] - 1:
                        grid[b, (self.agents[a, b, 0] + self.agent_obs[0] + 1):, :, 0] = 0
                        grid[b, (self.agents[a, b, 0] + self.agent_obs[0] + 1):, :, 1] = -1
                        # kill left quadrant
                    if self.agents[a, b, 1] > self.agent_obs[1]:
                        grid[b, :, :(self.agents[a, b, 1] - self.agent_obs[0]), 0] = 0
                        grid[b, :, :(self.agents[a, b, 1] - self.agent_obs[0]), 1] = -1
                    # Kill the right quadrant
                    if self.agents[a, b, 1] < self.grid_shape[1] - self.agent_obs[1] - 1:
                        grid[b, :, (self.agents[a, b, 1] + self.agent_obs[1] + 1):, 0] = 0
                        grid[b, :, (self.agents[a, b, 1] + self.agent_obs[1] + 1):, 1] = -1

        # The intersection grid is constructed, now we have to generate the observations from it
        if self.intersection_global_view:
            obs = grid.reshape((1, self.batch_size, self.state_size))
        else:
            # Return the intersection as individual observations
            obs = np.zeros((len(agent_ids), self.batch_size, self.obs_size),
                           dtype=float_type)
            for b in range(self.batch_size):
                for a in range(len(agent_ids)):
                    obs[a, b, :] = self._get_obs_from_grid(grid, a, b)

        # Return 0-1 encoded (including negative 1) intersection if necessary
        if not self.intersection_id_coded:
            obs = (obs > 0).astype(np.float32) - (obs < 0).astype(np.float32)

        # Final check: if not all agents can see each other, the mutual knowledge is empty
        if noinformation:
            obs = obs.reshape(obs.shape[0], obs.shape[1], obs.shape[2]//3, 3)
            obs[:, :, :, 0] = 0
            obs[:, :, :, 1] = -1
            obs[:, :, :, 2] = -1
            obs = obs.reshape(obs.shape[0], obs.shape[1], 3*obs.shape[2])

        # Return considering batch-mode
        if self.batch_mode:
            return obs, avail_all
        else:
            return obs[:, 0, :].squeeze(), avail_all

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
            if targets_alive is None or targets_alive[a, batch]:
                # If the target is visible to all agents
                if self._is_visible(self.agents[agent_ids, batch, :], targets[a, batch, :]):
                    # include the target in all observations (in relative positions)
                    for o in range(len(agent_ids)):
                        grid[batch, targets[a, batch, 0] + offset, targets[a, batch, 1] + offset, target_id] = a + 1

    def _get_obs_from_grid(self, grid, agent_id, batch=0):
        if self.toroidal:
            return self._get_obs_from_grid_troidal(grid, agent_id, batch)
        else:
            return self._get_obs_from_grid_bounded(grid, agent_id, batch)

    def _get_obs_from_grid_bounded(self, grid, agent_id, batch=0):
        """ Return a bounded observation for other agents' locations and targets, the size specified by observation
            shape, centered on the agent. Values outside the bounds of the grid are set to [-1, 0]. """
        # Create the empty observation grid
        agent_obs = np.zeros((2 * self.agent_obs[0] + 1, 2 * self.agent_obs[1] + 1, 3), dtype=float_type)
        agent_obs[:, :, 0] = -1
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
        agent_obs[aoy[0]:(aoy[1] + 1), aox[0]:(aox[1] + 1), :] = grid[batch, bul[0]:(blr[0] + 1),
                                                                 bul[1]:(blr[1] + 1), :]
        return np.reshape(agent_obs, self.obs_size)

    def _get_obs_from_grid_troidal(self, grid, agent_id, batch=0):
        """ Return a wrapped observation for other agents' locations and targets, the size specified by observation
            shape, centered on the agent. """
        a_x, a_y = self.agents[agent_id, batch, :]
        o_x, o_y = self.agent_obs
        x_range = range((a_x - o_x), (a_x + o_x + 1))
        y_range = range((a_y - o_y), (a_y + o_y + 1))
        ex_grid = grid[batch, :, :, :].astype(dtype=float_type)
        agent_obs = ex_grid.take(x_range, 0, mode='wrap').take(y_range, 1, mode='wrap')
        return np.reshape(agent_obs, self.obs_size)

# ----- END COPY


    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        if self.batch_mode:
            return self.grid.clone().view(self.state_size)
        else:
            return self.grid[0, :, :, :].cpu().view(self.state_size).numpy()

    # ---------- GETTERS -----------------------------------------------------------------------------------------------
    def get_total_actions(self):
        return self.n_actions

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


# ######################################################################################################################
if __name__ == "__main__":
    env_args = {
        # SC2
        'map_name': '5m_5m',
        'n_enemies': 5,
        'move_amount': 2,
        'step_mul': 8,  # lets you skip observations and actions
        'difficulty': "3",
        'reward_only_positive': True,
        'reward_negative_scale': 0.5,
        'reward_death_value': 10,
        'reward_win': 200,
        'reward_scale': True,
        'reward_scale_rate': 2,
        'state_last_action': True,
        'heuristic_function': False,
        'measure_fps': True,
        # Payoff
        'payoff_game': "monotonic",
        # Predator Prey
        'prey_movement': "escape",
        'predator_prey_shape': (4, 4),
        'predator_prey_toroidal': True,
        'nagent_capture_enabled': False,
        # Stag Hunt
        'stag_hunt_shape': (3, 3),
        'stochastic_reward_shift_optim': None,  # e.g. (1.0, 4) = (p, Delta_r)
        'stochastic_reward_shift_mul': None,  # e.g. (0.5, 2) = (p, Factor_r)
        'global_reward_scale_factor': 1.0,
        'state_variant': "grid",  # comma-separated string
        'n_prey': 1,
        'agent_obs': (1, 1),
        'episode_limit': 20,
        'n_agents': 4,
    }

    env = CoopBoxPushing(env_args=env_args)
    print("Env is ", "batched" if env.batch_mode else "not batched")

    [all_obs, state] = env.reset()
    print(state)
    for i in range(env.n_agents):
        print(all_obs[i])

    acts = lTensor([[0, 1, 2, 3], [3, 2, 1, 0]]).t()
    env.step(acts[:, 0].unsqueeze(1))

    env.print_grid()
    obs = []
    for i in range(4):
        obs.append(np.expand_dims(env.get_obs_agent(i), axis=1))
    print(np.concatenate(obs, axis=1))