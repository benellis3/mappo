import numpy as np
from components.scheme import Scheme
from itertools import combinations
import torch as th
from torch.distributions import Normal

from components.epsilon_schedules import FlatThenDecaySchedule
from runners import REGISTRY as r_REGISTRY

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
                                   dict(name="actions_level1",
                                        shape=(1,),
                                        select_agent_ids=range(0, self.n_agents),
                                        dtype=np.int32,
                                        missing=-1,),
                                   *[dict(name="actions_level2_agents{}:{}".format(_agent_id1, _agent_id2),
                                          shape=(1,),
                                          dtype=np.int32,
                                          missing=-1, ) for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2))],
                                   dict(name="actions_level3",
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
                                   dict(name="policies_level2",
                                        shape=(self.n_actions,),
                                        select_agent_ids=range(0, self.n_agents),
                                        dtype=np.float32,
                                        missing=np.nan),
                                   dict(name="policies_level2",
                                        shape=(self.n_actions,),
                                        select_agent_ids=range(0, self.n_agents),
                                        dtype=np.float32,
                                        missing=np.nan),
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
                                   dict(name="xxx_epsilons_level1",
                                        shape=(1,),
                                        dtype=np.float32,
                                        missing=float("nan")),
                                   dict(name="xxx_epsilons_level2",
                                        shape=(1,),
                                        dtype=np.float32,
                                        missing=float("nan")),
                                   dict(name="xxx_epsilons_level3",
                                        shape=(1,),
                                        dtype=np.float32,
                                        missing=float("nan")),
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
                                   ]).agent_flatten()
        pass

    def _add_episode_stats(self, T_env):
        super()._add_episode_stats(T_env)

        test_suffix = "" if not self.test_mode else "_test"
        #self._add_stat("policy_entropy",
        #               self.episode_buffer.get_stat("policy_entropy"),
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
                 self.xxx_epsilon_decay_schedule = FlatThenDecaySchedule(start=self.args.xxx_epsilon_start_level3,
                                                                         finish=self.args.xxx_epsilon_finish_level3,
                                                                         time_length=self.args.xxx_epsilon_time_length_level3,
                                                                         decay=self.args.xxx_epsilon_decay_mode_level3)

            epsilons = ttype(self.batch_size, 1).fill_(self.xxx_epsilon_decay_schedule_level3.eval(self.T_env))
            self.episode_buffer.set_col(col="xxx_epsilons_central_level3",
                                        scope="episode",
                                        data=epsilons)

        pass

    def log(self, log_directly=True):
        log_str, log_dict = super().log(log_directly=False)
        if not self.test_mode:
            log_str += ", XXX_epsilon_level1={:g}".format(self.xxx_epsilon_decay_schedule_level1.eval(self.T_env))
            log_str += ", XXX_epsilon_level2={:g}".format(self.xxx_epsilon_decay_schedule_level2.eval(self.T_env))
            log_str += ", XXX_epsilon_level3={:g}".format(self.xxx_epsilon_decay_schedule_level3.eval(self.T_env))
            self.logging_struct.py_logger.info("TRAIN RUNNER INFO: {}".format(log_str))
        else:
            self.logging_struct.py_logger.info("TEST RUNNER INFO: {}".format(log_str))
        return log_str, log_dict

    pass