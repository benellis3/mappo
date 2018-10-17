from run_experiment import extend_param_dicts

server_list = [
    ("brown", [0,1,2,3,4,5], 1),
    ("woma", [0,1,2,5], 1),
    ("brown", [0,1,2,3,4,5], 1),
]

label = "coma_unrestricted_actions__17_Oct_2018__v1"
config = "coma"
env_config = "sc2"

n_repeat = 3

param_dicts = []

shared_params = {
    "t_max": 3 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 24,
    "test_greedy": True,
    "env_args.obs_own_health": True, # Same as SMAC runs
    "mask_before_softmax": True, # All actions are unavailable, masking doesn't make a difference
    "lr": 0.0005,
    "critic_lr": 0.0005,
    "td_lambda": 0.8,
    "epsilon_start": 0.5,
    "epsilon_finish": 0.01,
    "epsilon_anneal_time": 100 * 1000,
    "target_update_interval": 200,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_unrestricted_actions",
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "td_lambda": 0.8,
        "epsilon_start": 0.5,
        "epsilon_finish": [0.01],
        "target_update_interval": 200,
        "env_args.map_name": ["2s_3z", "3m_3m"],
        "env_args.move_amount": 2,
        "env_args.obs_own_health": True,
        "mask_before_softmax": True,
    },
    repeats=8)