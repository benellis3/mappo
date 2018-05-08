import numpy as np
import torch as th
from torch.distributions import Normal

from components.epsilon_schedules import FlatThenDecaySchedule
from components.scheme import Scheme
from runners import REGISTRY as r_REGISTRY

NStepRunner = r_REGISTRY["nstep"]

class MCCERunner(NStepRunner):

    def _setup_data_scheme(self, data_scheme):
        self.data_scheme = Scheme([dict(name="observations",
                                        shape=(self.env_obs_size,),
                                        select_agent_ids=range(0, self.n_agents),
                                        dtype=np.float32,
                                        missing=np.nan,),
                                   dict(name="state",
                                        shape=(self.env_state_size,),
                                        dtype=np.float32,
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
                                   dict(name="policies",
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
                                   dict(name="mcce_epsilons",
                                        scope="episode",
                                        shape=(self.args.mcce_epsilon_size,),
                                        dtype=np.float32,
                                        missing=float("nan"),
                                        switch=self.args.multiagent_controller in ["mcce_mac"] and \
                                               not self.args.mcce_use_epsilon_seed),
                                   dict(name="mcce_epsilon_seeds",
                                        scope="episode",
                                        shape=(1,),  # TODO: not sure how long a seed is
                                        missing=float("nan"),
                                        switch=self.args.multiagent_controller in ["mcce_mac"] and \
                                               self.args.mcce_use_epsilon_seed),
                                   dict(name="mcce_epsilon_variances",
                                        scope="episode",
                                        shape=(1,),  # TODO: not sure how long a seed is
                                        missing=float("nan"),
                                        switch=self.args.multiagent_controller in ["mcce_mac"] and \
                                               self.args.mcce_use_epsilon_seed),
                                  ]).agent_flatten()
        pass

    def _add_episode_stats(self, T_env):
        super()._add_episode_stats(T_env)

        test_suffix = "" if not self.test_mode else "_test"
        self._add_stat("policy_entropy",
                       self.episode_buffer.get_stat("policy_entropy"),
                       T_env=T_env,
                       suffix=test_suffix)
        return

    def reset(self):
        super().reset()

        # if no test_mode, calculate fresh set of epsilons/epsilon seeds and update epsilon variance
        if not self.test_mode:
            self.mcce_epsilon_mean = 0.0
            ttype = th.cuda.FloatTensor if self.episode_buffer.is_cuda else th.FloatTensor
            # if no test_mode, calculate fresh set of epsilons/epsilon seeds and update epsilon variance
            if not self.test_mode:
                ttype = th.cuda.FloatTensor if self.episode_buffer.is_cuda else th.FloatTensor
                # calculate COMA_epsilon_schedule
                if not hasattr(self, "mcce_epsilon_decay_schedule"):
                    self.mcce_epsilon_decay_schedule = FlatThenDecaySchedule(start=self.args.coma_epsilon_start,
                                                                             finish=self.args.coma_epsilon_finish,
                                                                             time_length=self.args.coma_epsilon_time_length,
                                                                             decay=self.args.coma_epsilon_decay_mode)


            if self.args.mcce_use_epsilon_seed:
                # write epsilon_variances into buffer (use episode-wide one,
                # i.e. one per batch - could vary per batch entry if we wanted [not currently implemented])
                epsilon_variances = ttype(self.batch_size, 1).fill_(self.mcce_epsilon_decay_schedule.eval(self.T_env))
                self.episode_buffer.set_col(col="mcce_epsilon_variances",
                                            scope="episode",
                                            data=epsilon_variances.cuda() if self.episode_buffer.is_cuda else epsilon_variances.cpu())

                epsilon_seeds = ttype(self.batch_size, 1).random_(0, self.args.mcce_random_seed_max)
                self.episode_buffer.set_col(col="mcce_epsilon_seeds",
                                            scope="episode",
                                            data=epsilon_seeds)
            else:
                dist = Normal(self.mcce_epsilon_mean, self.mcce_epsilon_decay_schedule.eval(self.T_env))
                epsilons = dist.sample_n(self.batch_size * self.args.mcce_lambda_size).view(
                    self.batch_size,
                    self.args.mcce_lambda_size).view(self.batch_size, -1)
                self.episode_buffer.set_col(col="mcce_epsilons",
                                            scope="episode",
                                            data=epsilons.cuda() if self.episode_buffer.is_cuda else epsilons.cpu())

        pass



    def log(self, log_directly=True):
        log_str, log_dict = super().log(log_directly=False)
        if not self.test_mode:
            log_str += ", mcce_epsilon_variance={:g}".format(self.mcce_epsilon_decay_schedule.eval(self.T_env))
            self.logging_struct.py_logger.info("TRAIN RUNNER INFO: {}".format(log_str))
        else:
            self.logging_struct.py_logger.info("TEST RUNNER INFO: {}".format(log_str))
        return log_str, log_dict

    pass