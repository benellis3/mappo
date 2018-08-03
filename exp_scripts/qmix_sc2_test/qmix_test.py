from run_experiment import extend_param_dicts

server_list = [
    ("brown", [0,1,2,3,4,5,6,7], 2),
    ("gollum", [0,1,2,3,4,5,6,7], 2),
    ("gandalf", [1,2,3,4,5,6,7], 2),
    ("gimli", [0,3,5,7], 2)
    ("woma", [1,2,3,4,5,6,7], 1)
]

label = "QMIX_Refactor_Test"
config = "qmix"
env_config = "sc2"

n_repeat = 1

param_dicts = []

shared_params = {
    "epsilon_anneal_time": 50000,
    "buffer_size": 2000,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "qmix_sc2_3m",
        "env_args.map_name": "3m_3m"
    },
    repeats=8)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "qmix_sc2_5m",
        "env_args.map_name": "5m_5m"
    },
    repeats=8)
