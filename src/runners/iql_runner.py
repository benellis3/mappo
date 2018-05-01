import numpy as np
from components.scheme import Scheme
import torch as th
from torch.distributions import Normal

from components.epsilon_schedules import FlatThenDecaySchedule
from runners import REGISTRY as r_REGISTRY

NStepRunner = r_REGISTRY["nstep"]

class IQLRunner(NStepRunner):

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
                                   dict(name="actions",
                                        shape=(1,),
                                        select_agent_ids=range(0, self.n_agents),
                                        dtype=np.int32,
                                        missing=-1,),
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
                                   dict(name="qvalues",
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
                                   dict(name="iql_epsilons",
                                        scope="episode",
                                        shape=(1,),
                                        dtype=np.float32,
                                        missing=float("nan"))
                                  ]).agent_flatten()
        pass

    def __init__(self,
                 multiagent_controller=None,
                 args=None,
                 logging_struct=None,
                 data_scheme=None,
                 **kwargs):

        super().__init__(multiagent_controller,
                         args,
                         logging_struct,
                         data_scheme,
                         **kwargs)

        # set up epsilon greedy action selector with proper access function to epsilons
        def _get_epsilons():
            ret = self.episode_buffer.get_col(col="iql_epsilons", scope="episode")
            return ret
        self.multiagent_controller.action_selector._get_epsilons = _get_epsilons
        pass

    def _add_episode_stats(self, T_env):
        super()._add_episode_stats(T_env)
        self._add_stat("qvalues_entropy", self.episode_buffer.get_stat("qvalues_entropy"), T_env=T_env)
        return

    def reset(self):
        super().reset()

        # if no test_mode, calculate fresh set of epsilons/epsilon seeds and update epsilon variance
        if not self.test_mode:
            ttype = th.cuda.FloatTensor if self.episode_buffer.is_cuda else th.FloatTensor
            # calculate IQL_epsilon_schedule
            if not hasattr(self, "iql_epsilon_decay_schedule"):
                 self.iql_epsilon_decay_schedule = FlatThenDecaySchedule(start=self.args.iql_epsilon_start,
                                                                          finish=self.args.iql_epsilon_finish,
                                                                          time_length=self.args.iql_epsilon_time_length,
                                                                          decay=self.args.iql_epsilon_decay_mode)

            epsilons = ttype(self.batch_size, 1).fill_(self.iql_epsilon_decay_schedule.eval(self.T_env))
            self.episode_buffer.set_col(col="iql_epsilons",
                                        scope="episode",
                                        data=epsilons)
        pass

    def log(self, log_directly=True):
        log_str, log_dict = super().log(log_directly=False)
        if not self.test_mode:
            log_str += ", IQL_epsilon={:g}".format(self.iql_epsilon_decay_schedule.eval(self.T_env))
            self.logging_struct.py_logger.info("TRAIN RUNNER INFO: {}".format(log_str))
        else:
            self.logging_struct.py_logger.info("TEST RUNNER INFO: {}".format(log_str))
        return log_str, log_dict

    pass