import numpy as np
from components.scheme import Scheme
from components.transforms import _seq_mean
from copy import deepcopy
from itertools import combinations
import torch as th
from torch.distributions import Normal

from components.epsilon_schedules import FlatThenDecaySchedule
from runners import REGISTRY as r_REGISTRY
from utils.xxx import _n_agent_pair_samples, _n_agent_pairings, _ordered_agent_pairings, _n_agent_pairings

NStepRunner = r_REGISTRY["nstep"]

class XXXRunner(NStepRunner):

    def _setup_data_scheme(self, data_scheme):

        scheme_list = [dict(name="observations",
                            shape=(self.env_obs_size,),
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.float32,
                            missing=np.nan,),
                       dict(name="state",
                            shape=(self.env_state_size,),
                            dtype = np.float32,
                            missing=np.nan,
                            size=self.env_state_size),
                       *[dict(name="actions_level1__sample{}".format(_i), # stores ids of pairs that are sampled from
                            shape=(1,),
                            dtype=np.int32,
                            missing=-1,) for _i in range(_n_agent_pair_samples(self.n_agents))],
                       *[dict(name="actions_level2__sample{}".format(_i), # stores joint action for each sampled pair
                              shape=(1,), # i.e. just one number for a pair of actions!
                              dtype=np.int32,
                              missing=-1,) for _i in range(_n_agent_pair_samples(self.n_agents))],
                       dict(name="actions_level2",  # stores action for each agent that was chosen individually
                            shape=(1,),
                            select_agent_ids=range(0, _n_agent_pairings(self.n_agents)),
                            dtype=np.int32,
                            missing=-1, ),
                       dict(name="actions_level3", # stores action for each agent that was chosen individually
                            shape=(1,),
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.int32,
                            missing=-1, ),
                       dict(name="actions", # contains all agent actions - this is what env.step is based on!
                            shape=(1,),
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.int32,
                            missing=-1, ),
                       dict(name="avail_actions",
                            shape=(self.n_actions+1,), # includes no-op
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.int32,
                            missing=-1,),
                       dict(name="reward",
                            shape=(1,),
                            dtype=np.float32,
                            missing=np.nan),
                       dict(name="agent_id",
                            shape=(1,),
                            dtype=np.int32,
                            select_agent_ids=range(0, self.n_agents),
                            missing=-1),
                       dict(name="policies_level1",
                            shape=(_n_agent_pairings(self.n_agents),),
                            dtype=np.float32,
                            missing=np.nan),
                       *[dict(name="policies_level2__sample{}".format(_i),
                              shape=(2+self.n_actions*self.n_actions,), # include delegation and no-op
                              dtype=np.int32,
                              missing=-1,) for _i in range(_n_agent_pair_samples(self.n_agents))],
                       dict(name="policies_level3",
                            shape=(self.n_actions+1,), # includes no-op
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.float32,
                            missing=np.nan),
                       dict(name="terminated",
                            shape=(1,),
                            dtype=np.bool,
                            missing=False),
                       dict(name="truncated",
                            shape=(1,),
                            dtype=np.bool,
                            missing=False),
                       dict(name="reset",
                            shape=(1,),
                            dtype=np.bool,
                            missing=False),
                       # dict(name="xxx_epsilons_level1",
                       #      shape=(1,),
                       #      dtype=np.float32,
                       #      missing=float("nan")),
                       # dict(name="xxx_epsilons_level2",
                       #      shape=(1,),
                       #      dtype=np.float32,
                       #      missing=float("nan")),
                       # dict(name="xxx_epsilons_level3",
                       #      shape=(1,),
                       #      dtype=np.float32,
                       #      missing=float("nan")),
                       dict(name="xxx_epsilons_central_level1",
                            scope="episode",
                            shape=(1,),
                            dtype=np.float32,
                            missing=float("nan")),
                       dict(name="xxx_epsilons_central_level2",
                            scope="episode",
                            shape=(1,),
                            dtype=np.float32,
                            missing=float("nan")),
                       dict(name="xxx_epsilons_central_level3",
                            scope="episode",
                            shape=(1,),
                            dtype=np.float32,
                            missing=float("nan"))
                       ]



        #self.env_state_size if self.args.env_args["intersection_global_view"] else self.n_agents * self.env_obs_size,
        if self.args.xxx_use_obs_intersections:
            obs_intersect_pair_size = self.env_setup_info[0]["obs_intersect_pair_size"]
            obs_intersect_all_size = self.env_setup_info[0]["obs_intersect_all_size"]
            scheme_list.extend([dict(name="obs_intersection_all",
                                     shape=(self.env_state_size if self.args.env_args["intersection_global_view"] else obs_intersect_all_size,),
                                     dtype=np.float32,
                                     missing=np.nan,
                                ),
                                *[dict(name="obs_intersection__pair{}".format(_i),
                                       shape=(self.env_state_size if self.args.env_args["intersection_global_view"] else obs_intersect_pair_size,),
                                       dtype=np.float32,
                                       missing=np.nan,
                                 )
                                for _i in range(_n_agent_pairings(self.n_agents))],
                                *[dict(name="avail_actions__pair{}".format(_i),
                                       shape=(self.n_actions*self.n_actions + 1,), # include no-op
                                       dtype=np.float32,
                                       missing=np.nan,
                                       )
                                  for _i in range(_n_agent_pairings(self.n_agents))]
                                ])

        self.data_scheme = Scheme(scheme_list)
        pass

    def _add_episode_stats(self, T_env):
        super()._add_episode_stats(T_env)

        test_suffix = "" if not self.test_mode else "_test"
        # TODO!
        # self._add_stat("policy_level1_entropy",
        #               self.episode_buffer.get_stat("policy_entropy", policy_label="policies_level1"),
        #               T_env=T_env,
        #               suffix=test_suffix)
        # self._add_stat("policy_level2_entropy_",
        #               self.episode_buffer.get_stat("policy_entropy", policy_label="policies_level2"),
        #               T_env=T_env,
        #               suffix=test_suffix)
        # self._add_stat("policy_level3_entropy_",
        #               self.episode_buffer.get_stat("policy_entropy", policy_label="policies_level3"),
        #               T_env=T_env,
        #               suffix=test_suffix)


        actions_level2, _ = self.episode_buffer.get_col(col="actions_level2__sample{}".format(0))
        delegation_rate = th.sum(actions_level2==0.0) / actions_level2.contiguous().view(-1).shape[0]
        self._add_stat("level2_delegation_rate",
                       delegation_rate,
                       T_env=T_env,
                       suffix=test_suffix)


        # TODO: Policy entropy across levels! (Use suffix)
        return

    def reset(self):
        super().reset()

        # if no test_mode, calculate fresh set of epsilons/epsilon seeds and update epsilon variance
        if not self.test_mode:
            ttype = th.cuda.FloatTensor if self.episode_buffer.is_cuda else th.FloatTensor
            # calculate XXX_epsilon_schedules
            if not hasattr(self, "xxx_epsilon_decay_schedule_level1"):
                 self.xxx_epsilon_decay_schedule_level1 = FlatThenDecaySchedule(start=self.args.xxx_epsilon_start_level1,
                                                                                finish=self.args.xxx_epsilon_finish_level1,
                                                                                time_length=self.args.xxx_epsilon_time_length_level1,
                                                                                decay=self.args.xxx_epsilon_decay_mode_level1)

            epsilons = ttype(self.batch_size, 1).fill_(self.xxx_epsilon_decay_schedule_level1.eval(self.T_env))
            self.episode_buffer.set_col(col="xxx_epsilons_central_level1",
                                        scope="episode",
                                        data=epsilons)

            if not hasattr(self, "xxx_epsilon_decay_schedule_level2"):
                 self.xxx_epsilon_decay_schedule_level2 = FlatThenDecaySchedule(start=self.args.xxx_epsilon_start_level2,
                                                                                finish=self.args.xxx_epsilon_finish_level2,
                                                                                time_length=self.args.xxx_epsilon_time_length_level2,
                                                                                decay=self.args.xxx_epsilon_decay_mode_level2)

            epsilons = ttype(self.batch_size, 1).fill_(self.xxx_epsilon_decay_schedule_level2.eval(self.T_env))
            self.episode_buffer.set_col(col="xxx_epsilons_central_level2",
                                        scope="episode",
                                        data=epsilons)

            if not hasattr(self, "xxx_epsilon_decay_schedule_level3"):
                 self.xxx_epsilon_decay_schedule_level3 = FlatThenDecaySchedule(start=self.args.xxx_epsilon_start_level3,
                                                                         finish=self.args.xxx_epsilon_finish_level3,
                                                                         time_length=self.args.xxx_epsilon_time_length_level3,
                                                                         decay=self.args.xxx_epsilon_decay_mode_level3)

            epsilons = ttype(self.batch_size, 1).fill_(self.xxx_epsilon_decay_schedule_level3.eval(self.T_env))
            self.episode_buffer.set_col(col="xxx_epsilons_central_level3",
                                        scope="episode",
                                        data=epsilons)

        pass



    def log(self, log_directly=True):
        stats = self.get_stats()
        self._stats = deepcopy(stats)
        log_str, log_dict = super().log(log_directly=False)
        if not self.test_mode:
            log_str += ", XXX_epsilon_level1={:g}".format(self.xxx_epsilon_decay_schedule_level1.eval(self.T_env))
            log_str += ", XXX_epsilon_level2={:g}".format(self.xxx_epsilon_decay_schedule_level2.eval(self.T_env))
            log_str += ", XXX_epsilon_level3={:g}".format(self.xxx_epsilon_decay_schedule_level3.eval(self.T_env))
            log_str += ", level2_delegation_rate={:g}".format(_seq_mean(stats["level2_delegation_rate"]))
            # log_str += ", policy_level1_entropy={:g}".format(_seq_mean(stats["policy_level1_entropy"]))
            # log_str += ", policy_level2_entropy={:g}".format(_seq_mean(stats["policy_level2_entropy"]))
            # log_str += ", policy_level3_entropy={:g}".format(_seq_mean(stats["policy_level3_entropy"]))
            self.logging_struct.py_logger.info("TRAIN RUNNER INFO: {}".format(log_str))
        else:
            log_str += ", level2_delegation_rate={:g}".format(_seq_mean(stats["level2_delegation_rate_test"]))
            # log_str += ", policy_level1_entropy={:g}".format(_seq_mean(stats["policy_level1_entropy_test"]))
            # log_str += ", policy_level2_entropy={:g}".format(_seq_mean(stats["policy_level2_entropy_test"]))
            # log_str += ", policy_level3_entropy={:g}".format(_seq_mean(stats["policy_level3_entropy_test"]))
            self.logging_struct.py_logger.info("TEST RUNNER INFO: {}".format(log_str))
        return log_str, log_dict

    pass


    @staticmethod
    def _loop_worker(envs,
                     in_queue,
                     out_queue,
                     buffer_insert_fn,
                     subproc_id=None,
                     args=None,
                     msg=None):

        if in_queue is None:
            id, chosen_actions, output_buffer, column_scheme = msg
            env_id = id
        else:
            id, chosen_actions, output_buffer, column_scheme = in_queue.get() # timeout=1)
            env_id_offset = len(envs) * subproc_id  # TODO: Adjust for multi-threading!
            env_id = id - env_id_offset

        _env = envs[env_id]

        if chosen_actions == "SCHEME":
            env_dict = dict(obs_size=_env.get_obs_size(),
                            state_size=_env.get_state_size(),
                            episode_limit=_env.episode_limit,
                            n_agents = _env.n_agents,
                            n_actions=_env.get_total_actions())
            if args.xxx_use_obs_intersections:
                env_dict["obs_intersect_pair_size"]= _env.get_obs_intersect_pair_size()
                env_dict["obs_intersect_all_size"] = _env.get_obs_intersect_all_size()
            # Send results back
            ret_msg = dict(id=id, payload=env_dict)
            if out_queue is None:
                return ret_msg
            out_queue.put(ret_msg)
            return

        elif chosen_actions == "RESET":
            _env.reset() # reset the env!

            # perform environment steps and insert into transition buffer
            observations = _env.get_obs()
            state = _env.get_state()
            avail_actions = [_aa + [0] for _aa in _env.get_avail_actions()]  # add place for noop action
            ret_dict = dict(state=state)  # TODO: Check that env_info actually exists
            for _i, _obs in enumerate(observations):
                ret_dict["observations__agent{}".format(_i)] = observations[_i]
            for _i, _obs in enumerate(observations):
                ret_dict["avail_actions__agent{}".format(_i)] = avail_actions[_i]

            # handle observation intersections
            if args.xxx_use_obs_intersections:
                ret_dict["obs_intersection_all"], _ = _env.get_obs_intersection(tuple(range(_env.n_agents)))
                for _i, (_a1, _a2) in enumerate(_ordered_agent_pairings(_env.n_agents)):
                    ret_dict["obs_intersection__pair{}".format(_i)], \
                    ret_dict["avail_actions__pair{}".format(_i)] = _env.get_obs_intersection((_a1, _a2))
                    ret_dict["avail_actions__pair{}".format(_i)] = ret_dict["avail_actions__pair{}".format(_i)].flatten().tolist() + [0]

            buffer_insert_fn(id=id, buffer=output_buffer, data_dict=ret_dict, column_scheme=column_scheme)

            # Signal back that queue element was finished processing
            ret_msg = dict(id=id, payload=dict(msg="RESET DONE"))
            if out_queue is None:
                return ret_msg
            out_queue.put(ret_msg)
            return

        elif chosen_actions == "STATS":
            env_stats = _env.get_stats()
            env_dict = dict(env_stats=env_stats)
            # Send results back
            ret_msg = dict(id=id, payload=env_dict)
            if out_queue is None:
                return ret_msg
            out_queue.put(ret_msg)
            return

        else:

            reward, terminated, env_info = \
                _env.step([int(_i) for _i in chosen_actions])

            # perform environment steps and add to transition buffer
            observations = _env.get_obs()
            state = _env.get_state()
            avail_actions = [_aa + [0] for _aa in _env.get_avail_actions()]  # add place for noop action
            terminated = terminated
            truncated = terminated and env_info.get("episode_limit", False)
            ret_dict = dict(state=state,
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            )
            for _i, _obs in enumerate(observations):
                ret_dict["observations__agent{}".format(_i)] = observations[_i]
            for _i, _obs in enumerate(observations):
                ret_dict["avail_actions__agent{}".format(_i)] = avail_actions[_i]

            if args.xxx_use_obs_intersections:
                # handle observation intersections
                ret_dict["obs_intersection_all"], _= _env.get_obs_intersection(tuple(range(_env.n_agents)))
                for _i, (_a1, _a2) in enumerate(_ordered_agent_pairings(_env.n_agents)):
                    ret_dict["obs_intersection__pair{}".format(_i)],\
                    ret_dict["avail_actions__pair{}".format(_i)] = _env.get_obs_intersection((_a1, _a2))
                    ret_dict["avail_actions__pair{}".format(_i)] = ret_dict["avail_actions__pair{}".format(_i)].flatten().tolist() + [0]

            buffer_insert_fn(id=id, buffer=output_buffer, data_dict=ret_dict, column_scheme=column_scheme)

            # Signal back that queue element was finished processing
            ret_msg = dict(id=id, payload=dict(msg="STEP DONE", terminated=terminated))
            if out_queue is None:
                return ret_msg
            else:
                out_queue.put(ret_msg)
            return

        return