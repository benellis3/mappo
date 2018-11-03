from run_experiment import extend_param_dicts

server_list = [
    ("dgx1", [1,4,5,6,7], 2),
    ("sauron", [0,1,2,3,4,5,6,7], 2),
    ("gollum", [0,1,2,3,4,5,6,7], 2),
    # ("mulga", [4,5,6,7], 1),
]

label = "asymmetric_test__1_Nov_2018__v2"
config = "qmix_parallel"
env_config = "sc2"

n_repeat = 6

parallel_repeat = 3

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
    "env_args.map_name": ["5m_6m", "8m_9m", "8m_10m", "10m_11m", "3s5z_3s6z", "MMM2", "3s_5z"],
    "save_model": True,
    "save_model_interval": 500 * 1000,
}

# QMIX
extend_param_dicts(param_dicts, shared_params,
    {
        "name": "qmix_parallel",
    },
    repeats=parallel_repeat)

