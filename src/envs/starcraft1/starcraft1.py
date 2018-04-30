from ..multiagentenv import MultiAgentEnv
import torchcraft as tc
import torchcraft.Constants as tcc

import numpy as np
import pygame
import sys
import os
import math
import time
from operator import attrgetter
from copy import deepcopy

# from utils.dict2namedtuple import convert

# TODO: updated these for SC1
_possible_results = {
    sc_pb.Victory: 1,
    sc_pb.Defeat: -1,
    sc_pb.Tie: 0,
    sc_pb.Undecided: 0,
}

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

action_move_id = 16     #    target: PointOrUnit
action_atack_id = 23    #    target: PointOrUnit
action_stop_id = 4      #    target: None

'''
StarCraft I: Brood War
'''

class SC1(MultiAgentEnv):

    def __init__(self, **kwargs):
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Read arguments
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.map_name = args.map_name
        self.episode_limit = args.episode_limit
        self._move_amount = args.move_amount
        self._step_mul = args.step_mul
        self.difficulty = args.difficulty
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

        self.hostname = args.hostname
        self.port = args.port

        self.debug_inputs = False
        self.debug_rewards = False

        self.n_actions_no_attack = 6
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        # TODO fill in correct values
        if sys.platform == 'linux':
            self.game_version = "1.4.0"
            os.environ['SC1PATH'] = os.path.join(os.getcwd(), '..', '3rdparty', 'StarCraftI')
            self.stalker_id = 1885
            self.zealot_id = 1886
        else:
            self.game_version = "4.1.2"  # latest release, uses visualisations
            self.stalker_id = 1922
            self.zealot_id = 1923

        if self.map_name == '2d_3z' or self.map_name == '3d_5z':
            self._agent_race = "P"  # Protoss
            self._bot_race = "P"  # Protoss

            self.unit_health_max_z = 160
            self.unit_health_max_s = 150

            self.max_reward = 2 * self.unit_health_max_s + 3 * self.unit_health_max_z + self.n_enemies * self.reward_death_value + self.reward_win
        else:
            self._agent_race = "T"  # Terran
            self._bot_race = "T"  # Terran
            self.unit_health_max = 45

            self.max_reward = self.n_enemies * (self.unit_health_max + self.reward_death_value) + self.reward_win

        # Launch the game
        self._launch()

        self._game_info = self.controller.game_info() # CHANGE
        self.map_x = self._game_info.start_raw.map_size.x
        self.map_y = self._game_info.start_raw.map_size.y
        self.map_play_area_min = self._game_info.start_raw.playable_area.p0
        self.map_play_area_max = self._game_info.start_raw.playable_area.p1
        self.max_distance_x = self.map_play_area_max.x - self.map_play_area_min.x
        self.max_distance_y = self.map_play_area_max.y - self.map_play_area_min.y

        self._episode_count = -1
        self._total_steps = 0

        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0

        # self.last_action = tc.zeros((self.n_agents, self.n_actions))

        # self.save_units()

    def _launch(self):
        #CREATE SUBPROCESS - use pid

        self.controller = tc.Client()
        self.controller.connect(self.hostname, self.port)
        self._obs = self.cl.init()

        #self._run_config = run_configs.get()
        self._map = maps.get(self.map_name) # TODO: find out how to set the map in sc1

        # Setting up the interface
        #self.interface = sc_pb.InterfaceOptions(
        #        raw = True, # raw, feature-level data
        #        score = True)

        # TODO: could try to change config file (bwapi.ini), but calling launcher is probably better
        #self._sc2_procs = [self._run_config.start(game_version=self.game_version)]
        #self._controllers = [p.controller for p in self._sc2_procs]

        # All the communication with SC2 will go through the controller
        #self.controller = self._controllers[0] #already set controller above

        # Create the game
        # TODO: need to call launcher, possibly with environment specifications (number of players, map, etc.)
        # TODO: find out how to set seed
        #create = sc_pb.RequestCreateGame(realtime = False,
        #        random_seed = self.seed,
        #        local_map=sc_pb.LocalMap(map_path=self._map.path, map_data=self._run_config.map_data(self._map.path)))
        #create.player_setup.add(type=sc_pb.Participant)
        #create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],
                                difficulty=difficulties[self.difficulty])
        #self.controller.create_game(create)


        #join = sc_pb.RequestJoinGame(race=races[self._agent_race], options=self.interface)
        #self.controller.join_game(join)

    # TODO: find out what exactly happens in _run_config.save_replay() function
    def save_replay(self, replay_dir):

        replay_path = self._run_config.save_replay(self.controller.save_replay(), replay_dir, self.map_name)
        print("Wrote replay to: %s", replay_path)


    def reset(self):
        """Start a new episode."""
        # EMPTY MAP: Reset should spawn units explicitely - don't use map files

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
                    elapsed_time, self._total_steps, self._step_mul, (self._total_steps * self._step_mul) / elapsed_time))
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
        # TODO: figure out what to put here for sc1
        #except protocol.ProtocolError:
        #    self.full_restart()
        #except protocol.ConnectionError:
        #    self.full_restart()

        #print(self.agents[0])
        #print(self.agents[4])
        #print(self.controller.query(q_pb.RequestQuery(abilities=[q_pb.RequestQueryAvailableAbilities(unit_tag=self.agents[0].tag)])))
        #print(self.controller.data_raw())

        return self.get_obs(), self.get_state()

    def _restart(self):
        #self.controller.restart() # restarts the game after reloading the map

        # Kill and restore all units
        try:
            self.kill_all_units()
            #self.restore_units()
            self.controller.step(2)
        # TODO: figure out what to put here for sc1
        except protocol.ProtocolError:
            self.full_restart()
        except protocol.ConnectionError:
            self.full_restart()

    def full_restart(self):
        # End episode and restart a new one
        self.controller.quit()
        self._launch()
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

        terminated = False
        reward = self.reward_battle()
        info = {}

        # Update what we know about units
        end_game = self.update_units()

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

        return reward, terminated, info

    def get_agent_action(self, a_id, action):

        unit = self.get_unit_by_id(a_id)
        x = unit.x
        y = unit.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op chosen but the agent's unit is not dead"
            if self.debug_inputs:
                print("Agent %d: Dead"% a_id)
            return None

        elif action == 1:
            # stop
            sc_action = [tcc.command_unit_protected, a_id, tcc.unitcommandtypes.Stop]
            if self.debug_inputs:
                print("Agent %d: Stop"% a_id)

        elif action == 2:
            # north
            sc_action = [tcc.command_unit_protected,
                         a_id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x),
                         int(y + self._move_amount)]
            if self.debug_inputs:
                print("Agent %d: North"% a_id)

        elif action == 3:
            # south
            sc_action = [tcc.command_unit_protected,
                         a_id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x),
                         int(y - self._move_amount)]
            if self.debug_inputs:
                print("Agent %d: South"% a_id)

        elif action == 4:
            # east
            sc_action = [tcc.command_unit_protected,
                         a_id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x + self._move_amount),
                         int(y)]
            if self.debug_inputs:
                print("Agent %d: East"% a_id)

        elif action == 5:
            # west
            sc_action = [tcc.command_unit_protected,
                         a_id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x - self._move_amount),
                         int(y)]
            if self.debug_inputs:
                print("Agent %d: West"% a_id)
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
                print("Agent %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_agent_units[al_id].health - self.agents[al_id].health, self.previous_agent_units[al_id].shield - self.agents[al_id].shield))
            print('---------------------')
            for al_id in range(self.n_enemies):
                print("Enemy %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_enemy_units[al_id].health - self.enemies[al_id].health, self.previous_enemy_units[al_id].shield - self.enemies[al_id].shield))

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
        return 6 # TODO: update this for SC1

    def unit_sight_range(self, agent_id):
        return 9 # TODO: update this for SC1

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

        if unit.unit == 74 or unit.type == self.stalker_id: # Protoss's Stalker
            return 80 # TODO: update this for SC1
        if unit.unit == 73 or unit.type == self.zealot_id: # Protoss's Zaelot
            return 50 # TODO: update this for SC1

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """

        unit = self.get_unit_by_id(agent_id)

        nf_al = 4
        nf_en = 4

        move_feats = np.zeros(self.n_actions_no_attack - 2, dtype=np.float32) # exclude no-op & stop
        enemy_feats = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        ally_feats = np.zeros((self.n_agents - 1, nf_al), dtype=np.float32)

        if unit.health > 0: # otherwise dead, return all zeros
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

                if dist < sight_range and e_unit.health > 0: # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id] # available
                    enemy_feats[e_id, 1] = dist / sight_range # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range # relative Y

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

                if dist < sight_range and al_unit.health > 0: # visible and alive
                    ally_feats[i, 0] = 1 # visible
                    ally_feats[i, 1] = dist / sight_range # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range # relative Y

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
        nf_al = 4 # TODO: find out what nf stands for
        nf_en = 3 # TODO: find out what nf stands for

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
            if unit.y + self._move_amount < self.map_play_area_max.y:
                avail_actions[2] = 1
            if unit.y - self._move_amount > self.map_play_area_min.y:
                avail_actions[3] = 1
            if unit.x + self._move_amount < self.map_play_area_max.x:
                avail_actions[4] = 1
            if unit.x - self._move_amount > self.map_play_area_min.x:
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

        # if self.map_name == '2d_3z' or self.map_name == '3d_5z':
        #     nf_al += 2
        #     nf_en += 2

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

        debug_create_command = []
        for unit in self.agents_orig.values():
            pos = unit.pos
            cmd = d_pb.DebugCommand(create_unit =
                    d_pb.DebugCreateUnit(
                        unit_type = unit.unit_type,
                        owner = 1,
                        pos = sc_common.Point2D(x = pos.x, y = pos.y),
                        quantity = 1))

            debug_create_command.append(cmd)

        # TODO:update this for SC1
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

    def init_units(self):

        while True:

            self.agents = {}
            self.enemies = {}

            ally_units = [unit for unit in self._obs.units[0]]
            #ally_units = [unit for unit in self._obs.observation.raw_data.units if unit.owner == 1]
            ally_units_sorted = sorted(ally_units, key=attrgetter('type', 'x', 'y'), reverse=False)
            #ally_units_sorted = sorted(ally_units, key=attrgetter('unit_type', 'x', 'y'), reverse=False)

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug_inputs:
                    print("Unit %d is %d, x = %.1f, y = %1.f"  % (len(self.agents), self.agents[i].type, self.agents[i].x, self.agents[i].y))

            for unit in self._obs.units[1]:
            #for unit in self._obs.observation.raw_data.units:
                #if unit.owner == 2: # agent
                self.enemies[len(self.enemies)] = unit

            if len(self.agents) == self.n_agents and len(self.enemies) == self.n_enemies:
                # All good
                return

            # Might happen very rarely, just gonna do an additional environmental step
            # to give time for the units to spawn
            # as usual in the try brackets

            # Send actions
            sent = self.controller.send(1)

            if sent:
                self._obs = self.controller.recv()
            else:
                self.full_restart()
                return 0, True, {"episode_limit": True}

            # try:
            #     self.controller.step(1)
            #     self._obs = self.controller.recv()
            # # TODO: figure out what to put here for SC1
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
        self.previous_agent_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
                updated = False
            for unit in self._obs.units[0]:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated: # means dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.units[1]:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated: # means dead
                e_unit.health = 0

        if n_ally_alive == 0 and n_enemy_alive > 0:
            return -1 # loss
        if n_ally_alive > 0 and n_enemy_alive == 0:
            return 1 # win
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0 # tie, not sure if this is possible

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
