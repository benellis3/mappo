from run_experiment import extend_param_dicts

server_list = [
    ("bilbo", [0,1,2,3,4,5,6,7], 1),
]

label = "COMA_refactor_test_1"
config = "coma"
env_config = "sc2"

n_repeat = 10

param_dicts = []

shared_params = {
    "t_max": 1000000,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_3m",
        "env_args.map_name": "3m_3m"
    },
    repeats=2)

for lr in [0.0001, 0.0005, 0.001]:
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": "coma_sc2_5m",
            "lr": lr,
            "critic_lr": lr,
            "env_args.map_name": "5m_5m"
        },
        repeats=2)