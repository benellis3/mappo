import numpy as np
from components.scheme import Scheme
from components.transforms import _seq_mean
from itertools import combinations
import torch as th
from torch.distributions import Normal

from components.epsilon_schedules import FlatThenDecaySchedule
from runners import REGISTRY as r_REGISTRY
from utils.xxx import _n_agent_pair_samples, _n_agent_pairings

NStepRunner = r_REGISTRY["nstep"]

class XXXRunner(NStepRunner):

    def _setup_data_scheme(self, data_scheme):
        self.data_scheme = Scheme([dict(name="observations",
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
                                        shape=(self.n_actions,),
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
                                          shape=(1+self.n_actions*self.n_actions,), # i.e. just one number for a pair of actions!
                                          #select_agent_ids=range(0, _n_agent_pairings(self.n_agents)),
                                          dtype=np.int32,
                                          missing=-1,) for _i in range(_n_agent_pair_samples(self.n_agents))],
                                   dict(name="policies_level3",
                                        shape=(self.n_actions,),
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
                                   ])
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
        log_str, log_dict = super().log(log_directly=False)
        if not self.test_mode:
            log_str += ", XXX_epsilon_level1={:g}".format(self.xxx_epsilon_decay_schedule_level1.eval(self.T_env))
            log_str += ", XXX_epsilon_level2={:g}".format(self.xxx_epsilon_decay_schedule_level2.eval(self.T_env))
            log_str += ", XXX_epsilon_level3={:g}".format(self.xxx_epsilon_decay_schedule_level3.eval(self.T_env))
            # log_str += ", policy_level1_entropy={:g}".format(_seq_mean(stats["policy_level1_entropy"]))
            # log_str += ", policy_level2_entropy={:g}".format(_seq_mean(stats["policy_level2_entropy"]))
            # log_str += ", policy_level3_entropy={:g}".format(_seq_mean(stats["policy_level3_entropy"]))
            self.logging_struct.py_logger.info("TRAIN RUNNER INFO: {}".format(log_str))
        else:
            # log_str += ", policy_level1_entropy={:g}".format(_seq_mean(stats["policy_level1_entropy_test"]))
            # log_str += ", policy_level2_entropy={:g}".format(_seq_mean(stats["policy_level2_entropy_test"]))
            # log_str += ", policy_level3_entropy={:g}".format(_seq_mean(stats["policy_level3_entropy_test"]))
            self.logging_struct.py_logger.info("TEST RUNNER INFO: {}".format(log_str))
        return log_str, log_dict

    pass