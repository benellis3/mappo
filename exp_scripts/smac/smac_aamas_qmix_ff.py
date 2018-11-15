from run_experiment import extend_param_dicts

server_list = [
    ("savitar", [0,1,2,3,4,5,6,7], 2),
    ("gandalf", [0,1,2,3,4,5,6,7], 1),
]

label = "smac_aamas_ff__14_Nov_2018"
config = "qmix_parallel"
env_config = "sc2"

n_repeat = 3

parallel_repeat = 8

param_dicts = []

shared_params = {
    "t_max": 10 * 1000 * 1000 + 50 * 1000,
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
    "save_model_interval": 2000 * 1000,
    "test_interval": 20000,
    "log_interval": 20000,
    "runner_log_interval": 20000,
    "learner_log_interval": 20000,
    "agent": "ff",
}

maps = []

# Symmetric
maps += ["2s_3z"]

# Asym

# Micro
maps += ["micro_baneling"]
maps += ["3s_vs_3z"]


for map_name in maps:

    name = "feedforward__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name
        },
        repeats=parallel_repeat)

