from run_experiment import extend_param_dicts

server_list = [
    ("sauron", [0,1,2,3,4,5,6,7], 2),
]

label = "centralv_compare_masking_1e10"
config = "centralV"
env_config = "sc2"

n_repeat = 3

param_dicts = []

shared_params = {
    "t_max": 3000000,
    "test_interval": 2000,
    "test_nepisode": 24,
    "test_greedy": False
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "centralV_sc2_2s_3z",
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "td_lambda": 0.8,
        "epsilon_start": 0.5,
        "epsilon_finish": [0.01],
        "target_update_interval": 200,
        "env_args.map_name": "2s_3z",
        "env_args.move_amount": 2,
        "env_args.obs_own_health": False,
        "mask_before_softmax": [True, False]
    },
    repeats=8)