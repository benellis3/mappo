from ..multiagentenv import MultiAgentEnv
import torchcraft as tc
import torchcraft.Constants as tcc

import numpy as np
import sys
import os
import subprocess
import math
import time
from operator import attrgetter
from copy import deepcopy
from copy import copy

import socket, errno

from utils.dict2namedtuple import convert

'''
StarCraft I: Brood War
'''

# map parameter registry
map_param_registry = {"m5v5_c_far": {"n_agents": 5, "n_enemies": 5}}


class SC1(MultiAgentEnv):

    def __init__(self, **kwargs):
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Read arguments
        self.map_name = args.map_name
        self.n_agents = map_param_registry[self.map_name]["n_agents"]
        self.n_enemies = map_param_registry[self.map_name]["n_enemies"]
        self.episode_limit = args.episode_limit
        self._move_amount = args.move_amount
        self._step_mul = args.step_mul
        # self.difficulty = args.difficulty
        self.state_last_action = args.state_last_action

        # Rewards args
        self.reward_only_positive = args.reward_only_positive
        self.reward_negative_scale = args.reward_negative_scale
        self.reward_death_value = args.reward_death_value
        self.reward_win = args.reward_win
        self.reward_scale = args.reward_scale
        self.reward_scale_rate = args.reward_scale_rate

        # Other
        self.seed = args.seed
        self.heuristic_function = args.heuristic_function
        self.measure_fps = args.measure_fps

        self.bs_id = kwargs["bs_id"]
        self.hostname = args.hostname
        self.port = args.port + self.bs_id
        self.port_in_use = False
        self.debug_inputs = False
        self.debug_rewards = False

        print(self.bs_id)

        self.n_actions_no_attack = 6
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        if sys.platform == 'linux':
            # self.game_version = "1.4.0"
            os.environ['SC1PATH'] = os.path.join(os.getcwd(), os.pardir, '3rdparty', 'StarCraftI')
            self.env_file_type = 'so'
        if sys.platform == 'darwin':
            # self.game_version = "1.4.0"
            os.environ['SC1PATH'] = os.path.join(os.getcwd(), os.pardir, '3rdparty', 'StarCraftI')
            self.env_file_type = 'dylib'
            # self.stalker_id = 1885
            # self.zealot_id = 1886
        # else:
        #     self.game_version = "4.1.2"  # latest release, uses visualisations
        #     self.stalker_id = 1922
        #     self.zealot_id = 1923

        if self.map_name == 'm5v5_c_far':
            self._agent_race = "Terran"  # Protoss
            self._bot_race = "Terran"  # Protoss

            # TODO: add more specs for other maps
            self.unit_health_max_m = 40

            self.max_reward = 5 * self.unit_health_max_m + 5 * self.unit_health_max_m + self.n_enemies * self.reward_death_value + self.reward_win
        else:
            self._agent_race = "Terran"  # Terran
            self._bot_race = "Terran"  # Terran
            self.unit_health_max = 45

            self.max_reward = self.n_enemies * (self.unit_health_max + self.reward_death_value) + self.reward_win

        # Check if server has already been launched
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((socket.gethostbyname(socket.gethostname()), self.port))
            self.port_in_use = False
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                # Port is already in use
                self.port_in_use = True
            else:
                # something else raised the socket.error exception
                print(e)
        s.close()

        # Launch the server
        if not self.port_in_use:
            self._launch_server()

        # Launch the game
        self._launch_client()

        self.map_x = self._obs.map_size[0]
        self.map_y = self._obs.map_size[1]
        self.map_play_area_min = [int(0), int(0)]
        self.map_play_area_max = [self.map_x, self.map_y]
        self.max_distance_x = self.map_x
        self.max_distance_y = self.map_y

        self._episode_count = -1
        self._total_steps = 0

        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0

        # self.last_action = tc.zeros((self.n_agents, self.n_actions))

    def _launch_server(self):
        my_env = {"OPENBW_ENABLE_UI": '0',
                  "BWAPI_CONFIG_AI__RACE": '{}'.format(self._bot_race),
                  "BWAPI_CONFIG_AI__AI": '{}/bwapi/build/lib/BWEnv.{}'.format(os.environ['SC1PATH'], self.env_file_type),
                  "BWAPI_CONFIG_AUTO_MENU__AUTO_MENU": "SINGLE_PLAYER",
                  "BWAPI_CONFIG_AUTO_MENU__MAP": '{}/envs/starcraft1/maps/{}.scm'.format(os.getcwd(), self.map_name),
                  # "BWAPI_CONFIG_AUTO_MENU__GAME_TYPE": "USE MAP SETTINGS",
                  "TORCHCRAFT_PORT": '{}'.format(self.port)}
        launcher_path = '{}/bwapi/build/bin'.format(os.environ['SC1PATH'])
        launcher = './BWAPILauncher'

        subprocess.Popen([launcher], cwd=launcher_path, env=my_env)

    def _launch_client(self):
        self.controller = tc.Client()
        self.controller.connect(self.hostname, self.port)
        self._obs = self.controller.init(micro_battles=True)

    def reset(self):
        """Start a new episode."""

        if self.debug_inputs or self.debug_rewards:
            print('------------>> RESET <<------------')

        self._episode_steps = 0
        if self._episode_count > 0:
            # No need to restart for the first episode.
            self._restart()

        self._episode_count += 1

        # naive way to measure FPS
        if self.measure_fps:
            if self._episode_count == 10:
                self.start_time = time.time()

            if self._episode_count == 20:
                elapsed_time = time.time() - self.start_time
                print('-------------------------------------------------------------')
                print("Took %.3f seconds for %s steps with step_mul=%d: %.3f fps" % (
                    elapsed_time, self._total_steps, self._step_mul,
                    (self._total_steps * self._step_mul) / elapsed_time))
                print('-------------------------------------------------------------')

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_agent_units = None
        self.previous_enemy_units = None

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        try:
            self._obs = self.controller.recv()
            self.init_units()
        except:
            # self.full_restart()
            pass

        return self.get_obs(), self.get_state()

    def _restart(self):
        # self.controller.restart() # restarts the game after reloading the map

        # Kill and restore all units
        try:
            self.kill_all_units()

            # self.restore_units()
            self._obs = self.controller.init(micro_battles=True)
        except:
            # self.full_restart()
            pass

    def full_restart(self):
        # End episode and restart a new one
        self.controller.close()
        # self._launch_server()
        self._launch_client()
        self.force_restarts += 1

    def one_hot(self, data, nb_classes):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def step(self, actions):
        """ Returns reward, terminated, info """
        self.last_action = self.one_hot(actions, self.n_actions)

        sc_actions = []
        for a_id, action in enumerate(actions):
            agent_action = self.get_agent_action(a_id, action)
            if agent_action:
                sc_actions.append(agent_action)
        # for a_id, action in enumerate(actions):
        #     if not self.heuristic_function:
        #         agent_action = self.get_agent_action(a_id, action)
        #     else:
        #         agent_action = self.get_agent_action_heuristic(a_id, action)
        #     if agent_action:
        #       sc_actions.append(agent_action)

        # Send actions
        sent = self.controller.send(sc_actions)

        if sent:
            self._obs = self.controller.recv()
        else:
            self.full_restart()
            return 0, True, {"episode_limit": True}

        # try:
        #     res_actions = self.controller.actions(req_actions)
        #     # Make step in SC2, i.e. apply actions
        #     self.controller.step(self._step_mul)
        #     # Observe here so that we know if the episode is over.
        #     self._obs = self.controller.observe()
        # except protocol.ProtocolError:
        #     self.full_restart()
        #     return 0, True, {"episode_limit": True}
        # except protocol.ConnectionError:
        #     self.full_restart()
        #     return 0, True, {"episode_limit": True}

        self._total_steps += 1
        self._episode_steps += 1

        # Update what we know about units
        end_game = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {}

        if end_game is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if end_game == 1:
                self.battles_won += 1
                reward += self.reward_win

        elif self.episode_limit > 0 and self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug_inputs or self.debug_rewards:
            print("Total Reward = %.f \n ---------------------" % reward)

        if self.reward_scale:
            reward /= (self.max_reward / self.reward_scale_rate)

        # if reward > 0:
        #     print(reward)
        # if terminated:
        #     print(terminated)
        #     print(reward)

        return reward, terminated, info

    def get_agent_action(self, a_id, action):

        unit = self.get_unit_by_id(a_id)
        x = unit.x
        y = unit.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op chosen but the agent's unit is not dead"
            if self.debug_inputs:
                print("Agent %d: Dead" % a_id)
            return None

        elif action == 1:
            # stop
            sc_action = [tcc.command_unit_protected, a_id, tcc.unitcommandtypes.Stop]
            if self.debug_inputs:
                print("Agent %d: Stop" % a_id)

        elif action == 2:
            # north
            sc_action = [tcc.command_unit_protected,
                         a_id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x),
                         int(y + self._move_amount)]
            if self.debug_inputs:
                print("Agent %d: North" % a_id)

        elif action == 3:
            # south
            sc_action = [tcc.command_unit_protected,
                         a_id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x),
                         int(y - self._move_amount)]
            if self.debug_inputs:
                print("Agent %d: South" % a_id)

        elif action == 4:
            # east
            sc_action = [tcc.command_unit_protected,
                         a_id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x + self._move_amount),
                         int(y)]
            if self.debug_inputs:
                print("Agent %d: East" % a_id)

        elif action == 5:
            # west
            sc_action = [tcc.command_unit_protected,
                         a_id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x - self._move_amount),
                         int(y)]
            if self.debug_inputs:
                print("Agent %d: West" % a_id)
        else:
            # attack units that are in range
            enemy_id = action - self.n_actions_no_attack
            sc_action = [tcc.command_unit_protected, a_id, tcc.unitcommandtypes.Attack_Unit, enemy_id]
            if self.debug_inputs:
                print("Agent %d attacks enemy # %d" % (a_id, enemy_id))

        return sc_action

    # TODO: DO this
    def get_agent_action_heuristic(self, a_id, action):

        # unit = self.get_unit_by_id(a_id)
        # tag = unit.tag
        #
        # enemy_tag = 0
        # for unit in self.enemies.values():
        #     if unit.health > 0:
        #         enemy_tag = unit.tag
        #
        # # attack units that are in range
        # cmd = r_pb.ActionRawUnitCommand(ability_id = action_atack_id,
        #         target_unit_tag = enemy_tag,
        #         unit_tags = [tag],
        #         queue_command = False)
        #
        # sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        # return sc_action
        pass

    def reward_battle(self):
        #  delta health - delta enemies + delta deaths where value:
        #   if enemy unit dies, add reward_death_value per dead unit
        #   if own unit dies, subtract reward_death_value per dead unit

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        if self.debug_rewards:
            for al_id in range(self.n_agents):
                print("Agent %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_agent_units[al_id].health \
                                                                      - self.agents[al_id].health,
                                                                      self.previous_agent_units[al_id].shield \
                                                                      - self.agents[al_id].shield))
            print('---------------------')
            for al_id in range(self.n_enemies):
                print("Enemy %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_enemy_units[al_id].health \
                                                                      - self.enemies[al_id].health,
                                                                      self.previous_enemy_units[al_id].shield \
                                                                      - self.enemies[al_id].shield))

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = self.previous_agent_units[al_id].health + self.previous_agent_units[al_id].shield
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += (prev_health - al_unit.health - al_unit.shield) * neg_scale

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = self.previous_enemy_units[e_id].health + self.previous_enemy_units[e_id].shield
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:

            if self.debug_rewards:
                print("--------------------------")
                print("Delta enemy: ", delta_enemy)
                print("Delta deaths: ", delta_deaths)
                print("Reward: ", delta_enemy + delta_deaths)
                print("--------------------------")

            reward = delta_enemy + delta_deaths

        else:
            if self.debug_rewards:
                print("--------------------------")
                print("Delta enemy: ", delta_enemy)
                print("Delta deaths: ", delta_deaths)
                print("Delta ally: ", - delta_ally)
                print("Reward: ", delta_enemy + delta_deaths)
                print("--------------------------")

            reward = delta_enemy + delta_deaths - delta_ally

        return reward

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        return tcc.orders.AttackFixedRange  # 11

    def unit_sight_range(self, agent_id):
        unit = self.get_unit_by_id(agent_id)
        return tcc.staticvalues["sightRange"][unit.type]

    # def unit_max_cooldown(self, agent_id):
    #     # These are the biggest I've seen, there is no info about this
    #     # if self.map_name == '2d_3z' or self.map_name == '3d_5z':
    #     #     return 15
    #
    #     unit = self.get_unit_by_id(agent_id)
    #     if unit.unit_type == self.stalker_id: # Protoss's Stalker
    #         return 35
    #     if unit.unit_type == self.zealot_id: # Protoss's Zaelot
    #         return 22

    def unit_max_shield(self, unit_id, ally):
        # These are the biggest I've seen, there is no info about this
        if ally:
            unit = self.agents[unit_id]
        else:
            unit = self.enemies[unit_id]

        return tcc.staticvalues["maxShields"][unit.type]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """

        unit = self.get_unit_by_id(agent_id)

        nf_al = 4
        nf_en = 4

        move_feats = np.zeros(self.n_actions_no_attack - 2, dtype=np.float32)  # exclude no-op & stop
        enemy_feats = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        ally_feats = np.zeros((self.n_agents - 1, nf_al), dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.x
            y = unit.y
            sight_range = self.unit_sight_range(agent_id)

            avail_actions = self.get_avail_agent_actions(agent_id)

            for m in range(self.n_actions_no_attack - 2):
                move_feats[m] = avail_actions[m + 2]

            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.x
                e_y = e_unit.y
                dist = self.distance(x, y, e_x, e_y)

                if dist < sight_range and e_unit.health > 0:  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y

                    # if self.map_name == '2d_3z' or self.map_name == '3d_5z':
                    #     type_id = e_unit.unit_type - 73  # id(Stalker) = 74, id(Zealot) = 73
                    #     enemy_feats[e_id, 4 + type_id] = 1

            # place the features of the agent himself always at the first place
            al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.x
                al_y = al_unit.y
                dist = self.distance(x, y, al_x, al_y)

                if dist < sight_range and al_unit.health > 0:  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

        agent_obs = np.concatenate((move_feats.flatten(),
                                    enemy_feats.flatten(),
                                    ally_feats.flatten()))

        agent_obs = agent_obs.astype(dtype=np.float32)

        if self.debug_inputs:
            print("***************************************")
            print("Agent: ", agent_id)
            print("Available Actions\n", self.get_avail_agent_actions(agent_id))
            print("Move feats\n", move_feats)
            print("Enemy feats\n", enemy_feats)
            print("Ally feats\n", ally_feats)
            print("***************************************")

        return agent_obs

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        nf_al = 4
        nf_en = 3

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))
        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.x
                y = al_unit.y
                # max_cd = self.unit_max_cooldown(al_id)

                ally_state[al_id, 0] = al_unit.health / al_unit.max_health  # health
                ally_state[al_id, 1] = al_unit.airCD / al_unit.maxCD  # cooldown for airCD = groundCD
                ally_state[al_id, 2] = (x - center_x) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (y - center_y) / self.max_distance_y  # relative Y

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.x
                y = e_unit.y

                enemy_state[e_id, 0] = e_unit.health / e_unit.max_health  # health
                enemy_state[e_id, 1] = (x - center_x) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (y - center_y) / self.max_distance_y  # relative Y

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        state = state.astype(dtype=np.float32)

        if self.debug_inputs:
            print("------------ STATE ---------------")
            print("Ally state\n", ally_state)
            print("Enemy state\n", enemy_state)
            print("Last action\n", self.last_action)
            print("----------------------------------")

        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        nf_al = 4
        nf_en = 3

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions

        return size

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot do no-op as alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if unit.y + self._move_amount < self.map_play_area_max[1]:
                avail_actions[2] = 1
            if unit.y - self._move_amount > self.map_play_area_min[1]:
                avail_actions[3] = 1
            if unit.x + self._move_amount < self.map_play_area_max[0]:
                avail_actions[4] = 1
            if unit.x - self._move_amount > self.map_play_area_min[0]:
                avail_actions[5] = 1

            # can attack only those who is alive
            # and in the shooting range

            shoot_range = self.unit_shoot_range(agent_id)

            for e_id, e_unit in self.enemies.items():
                if e_unit.health > 0:
                    dist = self.distance(unit.x, unit.y, e_unit.x, e_unit.y)
                    if dist <= shoot_range:
                        avail_actions[e_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):

        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_obs_size(self):
        """ Returns the shape of the observation """
        nf_al = 4
        nf_en = 4

        move_feats = self.n_actions_no_attack - 2
        enemy_feats = self.n_enemies * nf_en
        ally_feats = (self.n_agents - 1) * nf_al

        return move_feats + enemy_feats + ally_feats

    def close(self):
        print("Closing StarCraftII")
        self.controller.close()

    def render(self):
        pass

    def save_units(self):
        # called after initialising the map to remember the locations of units
        self.agents_orig = {}
        self.enemies_orig = {}

        self._obs = self.controller.recv()
        # self._obs = self.controller.observe()

        for unit in self._obs.units[0]:
            self.agents_orig[len(self.agents_orig)] = unit
        for unit in self._obs.units[1]:
            self.enemies_orig[len(self.enemies_orig)] = unit

        # for unit in self._obs.observation.raw_data.units:
        #     if unit.owner == 1: # agent
        #         self.agents_orig[len(self.agents_orig)] = unit
        #     else:
        #         self.enemies_orig[len(self.enemies_orig)] = unit

        assert len(self.agents_orig) == self.n_agents, "Incorrect number of agents: " + str(len(self.agents_orig))
        assert len(self.enemies_orig) == self.n_enemies, "Incorrect number of enemies: " + str(len(self.enemies_orig))

    def restore_units(self):
        # restores the original state of the game
        pass

        # for unit in self.enemies_orig.values():
        #     pos = unit.pos
        #     cmd = d_pb.DebugCommand(create_unit =
        #             d_pb.DebugCreateUnit(
        #                 unit_type = unit.unit_type,
        #                 owner = 2,
        #                 pos = sc_common.Point2D(x = pos.x, y = pos.y),
        #                 quantity = 1))
        #
        #     debug_create_command.append(cmd)
        #
        # self.controller.debug(sc_pb.RequestDebug(debug = debug_create_command))

    def kill_all_units(self):

        # TODO: updated this for SC1
        # units_alive = [unit.tag for unit in self.agents.values() if unit.health > 0] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
        # debug_command = [d_pb.DebugCommand(kill_unit = d_pb.DebugKillUnit(tag = units_alive))]
        # self.controller.debug(sc_pb.RequestDebug(debug = debug_command))
        pass

    def init_units(self):

        counter = 1
        while True:

            self.agents = {}
            self.enemies = {}

            ally_units = [unit for unit in self._obs.units[0]]
            # ally_units = [unit for unit in self._obs.observation.raw_data.units if unit.owner == 1]
            ally_units_sorted = sorted(ally_units, key=attrgetter('type', 'x', 'y'), reverse=False)
            # ally_units_sorted = sorted(ally_units, key=attrgetter('unit_type', 'x', 'y'), reverse=False)

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug_inputs:
                    print("Unit %d is %d, x = %.1f, y = %1.f" % (
                        len(self.agents), self.agents[i].type, self.agents[i].x, self.agents[i].y))

            for unit in self._obs.units[1]:
                # for unit in self._obs.observation.raw_data.units:
                # if unit.owner == 2: # agent
                self.enemies[len(self.enemies)] = unit

            if self.agents == {}:
                counter += 1

            if len(self.agents) == self.n_agents and len(self.enemies) == self.n_enemies:
                # All good
                # print('Spawned agents after iteration {}'.format(counter))
                return

            # Might happen very rarely, just gonna do an additional environmental step
            # to give time for the units to spawn
            # as usual in the try brackets

            # Send actions
            sent = self.controller.send([])

            if sent:
                self._obs = self.controller.recv()
            else:
                self.full_restart()

            # try:
            #     self.controller.step(1)
            #     self._obs = self.controller.recv()
            # except protocol.ProtocolError:
            #     # iffy way, but would not thraw an error for sure
            #     self.full_restart()
            #     self.reset()
            # except protocol.ConnectionError:
            #     # iffy way, but would not thraw an error for sure
            #     self.full_restart()
            #     self.reset()
            # assert len(self.agents) == self.n_agents, "Incorrect number of agents: " + str(len(self.agents))
            # assert len(self.enemies) == self.n_enemies, "Incorrect number of enemies: " + str(len(self.enemies))

    def update_units(self):
        # TODO optimise this

        # This function assumes that self._obs is up-to-date
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_agent_units = copy(
            self.agents)  # TODO: "copy" is just temporary--need to find alternative solution here
        self.previous_enemy_units = copy(
            self.enemies)  # TODO: "copy" is just temporary--need to find alternative solution here

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.units[0]:
                if al_unit.id == unit.id:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # means dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.units[1]:
                if e_unit.id == unit.id:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # means dead
                e_unit.health = 0

        if n_ally_alive == 0 and n_enemy_alive > 0:
            return -1  # loss
        if n_ally_alive > 0 and n_enemy_alive == 0:
            return 1  # win
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0  # tie, not sure if this is possible

        return None

    def get_unit_by_id(self, a_id):
        return self.agents[a_id]

    def get_stats(self):
        stats = {}
        stats["battles_won"] = self.battles_won
        stats["battles_game"] = self.battles_game
        stats["win_rate"] = self.battles_won / self.battles_game
        stats["timeouts"] = self.timeouts
        stats["restarts"] = self.force_restarts
        return stats


from components.transforms import _seq_mean


class StatsAggregator():

    def __init__(self):
        self.last_stats = None
        self.stats = []
        pass

    def aggregate(self, stats, add_stat_fn):

        current_stats = {}
        for stat in stats:
            for _k, _v in stat.items():
                if not (_k in current_stats):
                    current_stats[_k] = []
                if _k in ["win_rate"]:
                    continue
                current_stats[_k].append(_v)

        # average over stats
        aggregate_stats = {}
        for _k, _v in current_stats.items():
            if _k in ["win_rate"]:
                aggregate_stats[_k] = np.mean(
                    [(_a - _b) / (_c - _d) for _a, _b, _c, _d in zip(current_stats["battles_won"],
                                                                     [0] * len(current_stats[
                                                                                   "battles_won"]) if self.last_stats is None else
                                                                     self.last_stats["battles_won"],
                                                                     current_stats["battles_game"],
                                                                     [0] * len(current_stats[
                                                                                   "battles_game"]) if self.last_stats is None else
                                                                     self.last_stats["battles_game"])])
            else:
                aggregate_stats[_k] = np.mean(
                    [_a - _b for _a, _b in zip(_v, [0] * len(_v) if self.last_stats is None else self.last_stats[_k])])

        # add stats that have just been produced to tensorboard / sacred
        for _k, _v in aggregate_stats.items():
            add_stat_fn(_k, _v)

        # collect stats for logging horizon
        self.stats.append(aggregate_stats)
        # update last stats
        self.last_stats = current_stats
        pass

    def log(self, log_directly=False):
        assert not log_directly, "log_directly not supported."
        logging_str = " Win rate: {}".format(_seq_mean([stat["win_rate"] for stat in self.stats])) \
                      + " Timeouts: {}".format(_seq_mean([stat["timeouts"] for stat in self.stats])) \
                      + " Restarts: {}".format(_seq_mean([stat["restarts"] for stat in self.stats]))

        # flush stats
        self.stats = []
        return logging_str