from run_experiment import extend_param_dicts

server_list = [
    ("brown", [0,1,2,3,4,5,6,7], 1),
]

label = "COMA_fix_2s_3z"
config = "coma"
env_config = "sc2"

n_repeat = 2

param_dicts = []

shared_params = {
    "t_max": 2000000,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_2s_3z",
        "lr": 0.0001,
        "critic_lr": 0.0005,
        "td_lambda": 0.8,
        "env_args.map_name": "2s_3z"
    },
    repeats=8)