from ..multiagentenv import MultiAgentEnv
import numpy as np
import pygame
from random import random
from utils.dict2namedtuple import convert

'''
A simple toroidal gridworld game for N agents where agents receive a collective reward of +1 for each agent that reaches
one of N randomised target locations and a reward of -1 for each agent that collides with another agent

Observation output is a 1d vector of size (2 x obs_size), giving location of all agents and targets within a vicinity
of agent_obs in two appended 1d vectors of size obs_size.

State output is a list of length 2, giving location of all agents and all targets

TODO: Check rendering
'''

class PredPrey(MultiAgentEnv):

    def __init__(self, n_agents=2, **kwargs):
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)
        #, n_prey = 1, agent_obs = (1, 1), episode_limit = 20

        self.nagent_capture_enabled = getattr(args, "nagent_capture_enabled", None)
        self.stochastic_reward_shift_mul = getattr(args, "stochastic_reward_shift_mul", None)
        self.stochastic_reward_shift_optim = getattr(args, "stochastic_reward_shift_optim", None)

        shape = args.predator_prey_shape
        self.state_variant = set(getattr(args, "state_variant", "grid").split(","))
        self.x_max, self.y_max = shape
        self.grid_shape = np.asarray(shape)
        self.actions = np.asarray([[0, 1],
                                   [1, 0],
                                   [0, -1],
                                   [-1, 0],
                                   [0, 0]])
        self.n_actions = self.actions.shape[0]
        self.n_agents = args.n_agents
        self.n_prey = args.n_prey
        self.agent_obs = args.agent_obs
        self.episode_limit = args.episode_limit
        self.obs_size = 2*(2*args.agent_obs[0]+1)*(2*args.agent_obs[1]+1) + 1

        self.state_size = 0
        if "grid" in self.state_variant:
            self.state_size += shape[0] * shape[1] * 2
        if "multigrid" in self.state_variant:
            self.state_size += shape[0] * shape[1] * (self.n_agents + self.n_prey)
        if "list" in self.state_variant:
            self.state_size += self.n_agents*2 + self.n_prey*2

        if self.stochastic_reward_shift_mul is not None:
            self.state_size += 1
        if self.stochastic_reward_shift_optim is not None:
            self.state_size += 1

        self.agents = []
        self.prey = []
        self.grid = np.zeros([self.x_max, self.y_max, 2], dtype=np.float32) #channels are 0: agents, 1: prey
        self.steps = 0
        self.prey_move = args.prey_movement # [random, escape]
        self.time_reward = -0.1

        self.reset()

        self.made_screen = False
        self.scaling = 5


    def step(self, actions):

        assert len(actions) == self.n_agents, "improper number of actions!"
        # Move the agents and update the grid
        reward = self.time_reward
        for a_id, action in enumerate(actions):
            old_pos = self.agents[a_id]
            new_pos = (old_pos + self.actions[action]) % self.grid_shape
            self.agents[a_id] = new_pos

        # Update the grid
        self.grid[:,:,:] = 0
        for agent in self.agents:
            self.grid[agent[0], agent[1], 0] = 1

        if self.prey_move == "random":
            prey_actions = np.random.randint(0, self.n_actions, len(self.prey))
            for p_id, action in enumerate(prey_actions):
                prey_pos = self.prey[p_id]
                new_pos = (prey_pos + self.actions[action]) % self.grid_shape
                self.prey[p_id] = new_pos

                if self.grid[new_pos[0], new_pos[1], 0] > 0:
                    reward += 1
                    self.prey[p_id] = None # Delete the prey

        elif self.prey_move == "escape":

            # The prey should move so that it is not captured
            for p_id in range(len(self.prey)):
                nr_safe_positions = 0
                p_action = np.random.randint(0, self.n_actions - 1)
                old_pos = self.prey[p_id]
                for act_inc in range(self.n_actions - 1):
                    new_pos = (old_pos + self.actions[:-1][(p_action + act_inc) % (self.n_actions - 1)]) % self.grid_shape
                    if self.grid[new_pos[0], new_pos[1], 0] < 1:
                        nr_safe_positions += 1

                prey_captured = False
                if self.nagent_capture_enabled:
                    if nr_safe_positions == 1:
                        reward += 1
                        # prey_captured = True # makes it hard tpo ever capture prey
                    elif nr_safe_positions == 0:
                        reward += 10
                        prey_captured = True
                elif nr_safe_positions == 0:
                    reward += 10
                    prey_captured = True

                if prey_captured:
                    self.prey[p_id] = None
                else:
                    self.prey[p_id] = new_pos

        # Get rid of the deleted preys
        self.prey = [prey for prey in self.prey if prey is not None]

        # Update the prey positions
        for prey_pos in self.prey:
            self.grid[prey_pos[0], prey_pos[1], 1] = 1

        terminated = False
        info = {}
        self.steps += 1
        if self.steps >= self.episode_limit:
            terminated = True
            info["episode_limit"] = True
        if self.prey == []:
            terminated = True
            info["episode_limit"] = False

        return reward, terminated, info

    def _print_location(self): # for debugging only

        loc = np.copy(self.grid[:, :, 0])
        for i in range(len(self.agents)):
            a = self.agents[i]
            loc[a[0], a[1]] = i + 1
        for i in range(len(self.prey)):
            a = self.prey[i]
            loc[a[0], a[1]] = -1*(i + 1)
        print(loc)


    def get_total_actions(self):
        return self.n_actions

    def get_obs_agent(self, agent_id):

        # Return a wrapped observation for other agents' locations and targets, the size specified by observation shape,
        #  centered on the agent
        a_x, a_y = self.agents[agent_id]
        o_x, o_y = self.agent_obs
        x_range = range((a_x-o_x),(a_x+o_x+1))
        y_range = range((a_y-o_y),(a_y+o_y+1))
        agent_obs = self.grid.take(x_range, 0, mode='wrap').take(y_range, 1, mode='wrap')

        # Flatten observations and return a single vector
        env_timestep = np.array([self.steps / self.episode_limit])
        agent_obs = np.append(agent_obs.flatten(), env_timestep)
        # Turn it into a float instead of long
        agent_obs = agent_obs.astype(dtype=np.float32)

        return agent_obs

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        state = np.array([])
        for item in self.state_variant:
            if item == "grid":
                state = np.append(state, self.grid.flatten())
            if item == "list":
                state = np.append(state, np.array(self.agents).flatten())
                state = np.append(state, np.array(self.prey).flatten())
            if item == "multigrid":
                multigrid = np.zeros((self.grid_shape[0], self.grid_shape[0], len(self.agents)+len(self.prey)))
                for i, pos in enumerate(self.agents):
                    multigrid[pos[0], pos[1], i] = 1
                for i, pos in enumerate(self.prey):
                    multigrid[pos[0], pos[1], i + len(self.agents)] = 1
                state = np.append(state, multigrid.flatten())

        if self.stochastic_reward_shift_optim is not None:
            state = np.append(state, 1 if self.stochastic_reward_shift_optim_is_lucky else 0)
            #return np.zeros((self.state_size,)) #DEBUG
        if self.stochastic_reward_shift_mul is not None:
            state = np.append(state, 1 if self.stochastic_reward_shift_mul_is_lucky else 0)
        return state

    def get_avail_agent_actions(self, agent_id):
        return [1 for _ in range(self.n_actions)]

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def reset(self):
        agent_locs = np.random.choice(self.x_max * self.y_max, self.n_agents, replace=False)
        xs, ys = np.unravel_index(agent_locs, [self.x_max, self.y_max])

        self.agents = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            self.agents.append(np.asarray([x, y]))

        self.grid.fill(0)
        for agent in self.agents:
            self.grid[agent[0], agent[1], 0] = 1

        prey_locs = np.random.choice(self.x_max * self.y_max, self.n_prey, replace=False)
        xs, ys = np.unravel_index(prey_locs, [self.x_max, self.y_max])

        self.prey = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            while self.grid[x, y, 0] > 0 or self.grid[x, y, 1] > 0:
                x = np.random.randint(0, self.x_max)
                y = np.random.randint(0, self.y_max)
            self.prey.append(np.asanyarray([x, y]))
            self.grid[x, y, 1] = len(self.prey) # should contain prey id

        self.steps = 0

        # Lucky modes
        if self.stochastic_reward_shift_mul is not None:
            self.stochastic_reward_shift_mul_is_lucky = (random() < self.stochastic_reward_shift_mul[0])

        # a = self.grid[:,:,0]
        # p = self.grid[:,:,1]
        if self.stochastic_reward_shift_optim is not None:
            self.stochastic_reward_shift_optim_is_lucky =  (random() < self.stochastic_reward_shift_optim[0])

        return self.get_obs(), self.get_state()

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.state_size

    def get_stats(self):
        pass

    def close(self):
        if self.made_screen:
            pygame.quit()
        print("Closing Multi-Agent Navigation")

    def render_array(self):
        # Return an rgb array of the frame
        frame = np.zeros((self.grid.shape[0] * self.scaling, self.grid.shape[1] * self.scaling, 3))
        agent_colours = [(b,b,255) for b in [255/(1+n) for n in range(self.n_agents)]]
        for color, agent_pos in zip(agent_colours, self.agents):
            x, y = agent_pos
            frame[(x * self.scaling):( (x+1) * self.scaling), (y * self.scaling):( (y+1) * self.scaling)] = color
        prey_colours = [(255,0,0) for b in [255/(1+n) for n in range(len(self.prey))]]
        for color, agent_pos in zip(prey_colours, self.prey):
            x, y = agent_pos
            frame[(x * self.scaling):( (x+1) * self.scaling), (y * self.scaling):( (y+1) * self.scaling)] = color

        return frame

    def render(self):
        if not self.made_screen:
            pygame.init()
            self.scaling = 20
            screen_size = ((self.grid.shape[1]) * self.scaling, (self.grid.shape[0]) * self.scaling)
            screen = pygame.display.set_mode(screen_size)
            self.screen = screen
            self.made_screen = True
        self.screen.fill((0, 0, 0))
        grid = self.grid
        # for x in range(grid.shape[0]):
        #     for y in range(grid.shape[1]):
        #         color = (0, 0, 0)
        #         if grid[x, y, 0] == 1:
        #             color = (0, 255, 0)
        #         pygame.draw.rect(self.screen, color, (y * self.scaling, x * self.scaling, self.scaling, self.scaling))
        #         color = (0, 0, 0)
        #         if grid[x, y, 1] == 1:
        #             color = (255, 0, 0)
        #         pygame.draw.rect(self.screen, color, (y * self.scaling + 1, x * self.scaling + 1, self.scaling - 2, self.scaling - 2))
        agent_colours = [(b,b,255) for b in [255/(1+n) for n in range(self.n_agents)]]
        for color, agent_pos in zip(agent_colours, self.agents):
            x, y = agent_pos
            pygame.draw.rect(self.screen, color, (y * self.scaling + 3, x * self.scaling + 3, self.scaling - 6, self.scaling - 6))
        prey_colours = [(255,0,0) for b in [255/(1+n) for n in range(len(self.prey))]]
        for color, agent_pos in zip(prey_colours, self.prey):
            x, y = agent_pos
            pygame.draw.rect(self.screen, color, (y * self.scaling + 3, x * self.scaling + 3, self.scaling - 6, self.scaling - 6))

        pygame.display.update()

