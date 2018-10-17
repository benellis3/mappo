from run_experiment import extend_param_dicts

server_list = [
    ("mulga", [4,5,6,7], 1),
]

label = "qmix_micro1_test__17_Oct_2018__v1"
config = "qmix_parallel"
env_config = "sc2"

n_repeat = 6

parallel_repeat = 2

param_dicts = []

shared_params = {
    "t_max": 3 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 24,
    "test_greedy": True,
    "env_args.obs_own_health": True, # We want this for SMAC(right?)
    "epsilon_start": 1.0,
    "epsilon_finish": 0.05,
    "epsilon_anneal_time": [20 * 1000, 500 * 1000],
    "target_update_interval": 200,
    "env_args.map_name": "micro1",
    "save_model": True,
    "save_model_interval": 100 * 1000,
}

# QMIX
extend_param_dicts(param_dicts, shared_params,
    {
        "name": "qmix_parallel_micro1",
    },
    repeats=parallel_repeat)

