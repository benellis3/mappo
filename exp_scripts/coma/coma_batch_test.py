from run_experiment import extend_param_dicts

server_list = [
    ("brown", [4,5,6,7], 1),
]

label = "coma_batch_2"
config = "coma_batch"
env_config = "sc2"

n_repeat = 4

param_dicts = []

shared_params = {
    "t_max": 3000000,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_2s_3z_lam0.5",
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "td_lambda": 0.5,
        "env_args.map_name": "2s_3z"
    },
    repeats=2)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_2s_3z_ag_lr1e-4",
        "lr": 0.0001,
        "critic_lr": 0.0005,
        "env_args.map_name": "2s_3z"
    },
    repeats=2)