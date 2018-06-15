import numpy as np
from components.scheme import Scheme
from components.transforms import _seq_mean
from copy import deepcopy
from itertools import combinations
import torch as th
from torch.distributions import Normal

from components.epsilon_schedules import FlatThenDecaySchedule
from runners import REGISTRY as r_REGISTRY
from utils.mackrel import _n_agent_pair_samples, _n_agent_pairings, _ordered_agent_pairings, _n_agent_pairings

NStepRunner = r_REGISTRY["nstep"]

class FLOUNDERLRunner(NStepRunner):

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
                            shape=(self.n_actions + 1,), # include no-op
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
                       # dict(name="policies_",
                       #      shape=(self.n_actions+1,), # includes no-op
                       #      select_agent_ids=range(0, self.n_agents),
                       #      dtype=np.float32,
                       #      missing=np.nan),
                       dict(name="policies_level1",
                            shape=(_n_agent_pairings(self.n_agents),),
                            dtype=np.float32,
                            missing=np.nan),
                       *[dict(name="policies_level2__sample{}".format(_i),
                              shape=(2 + self.n_actions * self.n_actions,),  # includes delegation and no-op
                              dtype=np.int32,
                              missing=-1, ) for _i in range(_n_agent_pair_samples(self.n_agents))],
                       dict(name="policies_level3",
                            shape=(self.n_actions+1,),  # does include no-op
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
                       dict(name="flounderl_epsilons_central",
                            scope="episode",
                            shape=(1,),
                            dtype=np.float32,
                            missing=float("nan")),
                       ]



        #self.env_state_size if self.args.env_args["intersection_global_view"] else self.n_agents * self.env_obs_size,
        if self.args.flounderl_use_obs_intersections:
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
                                       shape=(self.n_actions*self.n_actions + 1,), # do include no-op
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

        #[np.nanmean(np.nansum((-th.log(self["policies__agent{}".format(_aid)][0]) *
        #                       self["{}__agent{}".format(policy_label, _aid)][0]).cpu().numpy(), axis=2))
        # TODO!
        #self._add_stat("policy_level1_entropy",
        #              self.episode_buffer.get_stat("policy_entropy", policy_label="policies_level1"),
        #              T_env=T_env,
        #              suffix=test_suffix)

        tmp = self.episode_buffer["policies_level1"][0]
        entropy1 = np.nanmean(np.nansum((-th.log(tmp)*tmp).cpu().numpy(), axis=2))
        self._add_stat("policy_level1_entropy",
                       entropy1,
                       T_env=T_env,
                       suffix=test_suffix)


        for _i in range(_n_agent_pair_samples(self.n_agents)):
            tmp = self.episode_buffer["policies_level2__sample{}".format(_i)][0]
            entropy2 = np.nanmean(np.nansum((-th.log(tmp) * tmp).cpu().numpy(), axis=2))
            self._add_stat("policy_level2_entropy_sample{}".format(_i),
                           entropy2,
                           T_env=T_env,
                           suffix=test_suffix)

        #entropy3 = np.nanmean(np.nansum((-th.log(tmp) * tmp).cpu().numpy(), axis=2))
        self._add_stat("policy_level3_entropy",
                        self.episode_buffer.get_stat("policy_entropy", policy_label="policies_level3"),
                        T_env=T_env,
                        suffix=test_suffix)

        actions_level2, _ = self.episode_buffer.get_col(col="actions_level2__sample{}".format(0))
        delegation_rate = th.sum(actions_level2==0.0) / (actions_level2.contiguous().view(-1).shape[0] - th.sum(actions_level2!=actions_level2))
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
            # calculate MACKREL_epsilon_schedules
            if not hasattr(self, "flounderl_epsilon_decay_schedule"):
                 self.flounderl_epsilon_decay_schedule = FlatThenDecaySchedule(start=self.args.flounderl_epsilon_start,
                                                                                finish=self.args.flounderl_epsilon_finish,
                                                                                time_length=self.args.flounderl_epsilon_time_length,
                                                                                decay=self.args.flounderl_epsilon_decay_mode)

            epsilons = ttype(self.batch_size, 1).fill_(self.flounderl_epsilon_decay_schedule.eval(self.T_env))
            self.episode_buffer.set_col(col="flounderl_epsilons_central",
                                        scope="episode",
                                        data=epsilons)

        pass

    def run(self, test_mode):
        self.test_mode = test_mode

        # don't reset at initialization as don't have access to hidden state size then
        self.reset()

        terminated = False
        while not terminated:
            # increase episode time counter
            self.t_episode += 1

            # retrieve ids of all envs that have not yet terminated.
            # NOTE: for efficiency reasons, will perform final action selection in terminal state
            ids_envs_not_terminated = [_b for _b in range(self.batch_size) if not self.envs_terminated[_b]]
            ids_envs_not_terminated_tensor = th.cuda.LongTensor(ids_envs_not_terminated) \
                                                if self.episode_buffer.is_cuda \
                                                else th.LongTensor(ids_envs_not_terminated)


            if self.t_episode > 0:

                # flush transition buffer before next step
                self.transition_buffer.flush()

                # get selected actions from last step
                selected_actions, selected_actions_tformat = self.episode_buffer.get_col(col="actions",
                                                                                         t=self.t_episode-1,
                                                                                         agent_ids=list(range(self.n_agents))
                                                                                         )

                ret = self.step(actions=selected_actions[:, ids_envs_not_terminated_tensor.cuda()
                                                             if selected_actions.is_cuda else ids_envs_not_terminated_tensor.cpu(), :, :],
                                ids=ids_envs_not_terminated)

                # retrieve ids of all envs that have not yet terminated.
                # NOTE: for efficiency reasons, will perform final action selection in terminal state
                ids_envs_not_terminated = [_b for _b in range(self.batch_size) if not self.envs_terminated[_b]]
                ids_envs_not_terminated_tensor = th.cuda.LongTensor(ids_envs_not_terminated) \
                    if self.episode_buffer.is_cuda \
                    else th.LongTensor(ids_envs_not_terminated)

                # update which envs have terminated
                for _id, _v in ret.items():
                    self.envs_terminated[_id] = _v["terminated"]

                # insert new data in transition_buffer into episode buffer (NOTE: there's a good reason for why processes
                # don't write directly into the episode buffer)
                self.episode_buffer.insert(self.transition_buffer,
                                           bs_ids=list(range(self.batch_size)),
                                           t_ids=self.t_episode,
                                           bs_empty=[_i for _i in range(self.batch_size) if _i not in ids_envs_not_terminated])

                # update episode time counter
                if not self.test_mode:
                    self.T_env += len(ids_envs_not_terminated)


            # generate multiagent_controller inputs for policy forward pass
            action_selection_inputs, \
            action_selection_inputs_tformat = self.episode_buffer.view(dict_of_schemes=self.multiagent_controller.joint_scheme_dict,
                                                                       to_cuda=self.args.use_cuda,
                                                                       to_variable=True,
                                                                       bs_ids=ids_envs_not_terminated,
                                                                       t_id=self.t_episode,
                                                                       fill_zero=True, # TODO: DEBUG!!!
                                                                       )

            # retrieve avail_actions from episode_buffer
            avail_actions, avail_actions_format = self.episode_buffer.get_col(bs=ids_envs_not_terminated,
                                                                              col="avail_actions",
                                                                              t = self.t_episode,
                                                                              agent_ids=list(range(self.n_agents)))


            # select actions and retrieve related objects
            if isinstance(self.hidden_states, dict):
                hidden_states = {_k:_v[:, ids_envs_not_terminated_tensor, :, :] for _k, _v in self.hidden_states.items()}
            else:
                hidden_states = self.hidden_states[:, ids_envs_not_terminated_tensor, :,:]


            hidden_states, selected_actions, action_selector_outputs, selected_actions_format = \
                self.multiagent_controller.select_actions(inputs=action_selection_inputs,
                                                          avail_actions=avail_actions,
                                                          #tformat=avail_actions_format,
                                                          tformat=action_selection_inputs_tformat,
                                                          info=dict(T_env=self.T_env),
                                                          hidden_states=hidden_states,
                                                          test_mode=test_mode)

            if isinstance(hidden_states, dict):
                for _k, _v in hidden_states.items():
                    self.hidden_states[_k][:, ids_envs_not_terminated_tensor, :, :] = _v
            else:
                self.hidden_states[:, ids_envs_not_terminated_tensor, :, :] = hidden_states

            for _sa in action_selector_outputs:
                self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                            col=_sa["name"],
                                            t=self.t_episode,
                                            agent_ids=_sa.get("select_agent_ids", None),
                                            data=_sa["data"])

            # write selected actions to episode_buffer
            if isinstance(selected_actions, list):
               for _sa in selected_actions:
                   self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                               col=_sa["name"],
                                               t=self.t_episode,
                                               agent_ids=_sa.get("select_agent_ids", None),
                                               data=_sa["data"])
            else:
                self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                            col="actions",
                                            t=self.t_episode,
                                            agent_ids=list(range(self.n_agents)),
                                            data=selected_actions)

            # keep a copy of selected actions explicitely in transition_buffer device context
            #self.selected_actions = selected_actions.cuda() if self.transition_buffer.is_cuda else selected_actions.cpu()

            #Check for termination conditions
            #Check for runner termination conditions
            if self.t_episode == self.max_t_episode:
                terminated = True
            # Check whether all envs have terminated
            if all(self.envs_terminated):
                terminated = True
            # Check whether envs may have failed to terminate
            if self.t_episode == self.env_episode_limit+1 and not terminated:
                assert False, "Envs seem to have failed returning terminated=True, thus not respecting their own episode_limit. Please fix envs."

            pass

        # calculate episode statistics
        self._add_episode_stats(T_env=self.T_env)
        a = self.episode_buffer.to_pd()
        return self.episode_buffer


    def log(self, log_directly=True):
        stats = self.get_stats()
        self._stats = deepcopy(stats)
        log_str, log_dict = super().log(log_directly=False)
        if not self.test_mode:
            log_str += ", MACKREL_epsilon_level1={:g}".format(self.flounderl_epsilon_decay_schedule_level1.eval(self.T_env))
            log_str += ", MACKREL_epsilon_level2={:g}".format(self.flounderl_epsilon_decay_schedule_level2.eval(self.T_env))
            log_str += ", MACKREL_epsilon_level3={:g}".format(self.flounderl_epsilon_decay_schedule_level3.eval(self.T_env))
            log_str += ", level2_delegation_rate={:g}".format(_seq_mean(stats["level2_delegation_rate"]))
            log_str += ", policies_level1_entropy={:g}".format(_seq_mean(stats["policy_level1_entropy"]))
            for _i in range(_n_agent_pair_samples(self.n_agents)):
                log_str += ", policies_level2_entropy_sample{}={:g}".format(_i, _seq_mean(stats["policy_level2_entropy_sample{}".format(_i)]))
            log_str += ", policies_level3_entropy={:g}".format(_seq_mean(stats["policy_level3_entropy"]))
            # log_str += ", policy_level1_entropy={:g}".format(_seq_mean(stats["policy_level1_entropy"]))
            # log_str += ", policy_level2_entropy={:g}".format(_seq_mean(stats["policy_level2_entropy"]))
            # log_str += ", policy_level3_entropy={:g}".format(_seq_mean(stats["policy_level3_entropy"]))
            self.logging_struct.py_logger.info("TRAIN RUNNER INFO: {}".format(log_str))
        else:
            log_str += ", level2_delegation_rate={:g}".format(_seq_mean(stats["level2_delegation_rate_test"]))
            log_str += ", policies_level1_entropy={:g}".format(_seq_mean(stats["policy_level1_entropy_test"]))
            for _i in range(_n_agent_pair_samples(self.n_agents)):
                log_str += ", policies_level2_entropy_sample{}={:g}".format(_i, _seq_mean(stats["policy_level2_entropy_sample{}_test".format(_i)]))
            log_str += ", policies_level3_entropy={:g}".format(_seq_mean(stats["policy_level3_entropy_test"]))
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
            if args.flounderl_use_obs_intersections:
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
            avail_actions = [_aa + [0] for _aa in _env.get_avail_actions()] #_env.get_avail_actions()  # add place for noop action
            ret_dict = dict(state=state)  # TODO: Check that env_info actually exists
            for _i, _obs in enumerate(observations):
                ret_dict["observations__agent{}".format(_i)] = observations[_i]
            for _i, _obs in enumerate(observations):
                ret_dict["avail_actions__agent{}".format(_i)] = avail_actions[_i]

            # handle observation intersections
            if args.flounderl_use_obs_intersections:
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
            avail_actions = [_aa + [0] for _aa in _env.get_avail_actions()] # _env.get_avail_actions()  # add place for noop action
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

            if args.flounderl_use_obs_intersections:
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