from run_experiment import extend_param_dicts

server_list = [
    ("gimli", [0,1,2,3,4,5,6,7], 1),
]

label = "qmix_nodamage"
config = "qmix"
env_config = "sc2"

n_repeat = 5

param_dicts = []

shared_params = {
    "t_max": 2000000,
    "epsilon_anneal_time": 50000,
    "buffer_size": 2000,
    "env_args.reward_damage_coef": 0,
    "test_interval": 8000,
    "no-mongo": True
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "qmix_5m_nodamage",
        "mixer": "qmix",
        "env_args.map_name": "5m_5m"
    },
    repeats=4)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "qmix_3m_nodamage",
        "mixer": "qmix",
        "env_args.map_name": "3m_3m"
    },
    repeats=4)
