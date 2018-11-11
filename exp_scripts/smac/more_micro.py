from run_experiment import extend_param_dicts

server_list = [
    ("dgx1", [2,3,4,5,6,7], 3),
]

label = "more_micro__10_Nov_2018"
config = "qmix_parallel"
env_config = "sc2"

n_repeat = 4

parallel_repeat = 4

param_dicts = []

shared_params = {
    "t_max": 20 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 24,
    "test_greedy": True,
    "env_args.obs_own_health": True, # We want this for SMAC(right?)
    "epsilon_start": 1.0,
    "epsilon_finish": 0.05,
    "epsilon_anneal_time": [20 * 1000],
    "target_update_interval": 200,
    # "env_args.map_name": ["3s_vs_3z", "3s_vs_4z", "3s_vs_5z", "3s_vs_6z",],
    "save_model": True,
    "save_model_interval": 5000 * 1000,
    "test_interval": 20000,
    "log_interval": 25000,
    "runner_log_interval": 25000,
    "learner_log_interval": 25000,
}

for map in ["micro_retarget", "micro_corridor"]:
    for algo in [None, "qmix"]:

        name = algo
        if name is None:
            name = "iql"
        name += "__{}".format(map)
        extend_param_dicts(param_dicts, shared_params,
            {
                "name": name,
                "mixer": algo,
                "env_args.map_name": map
            },
            repeats=parallel_repeat)

