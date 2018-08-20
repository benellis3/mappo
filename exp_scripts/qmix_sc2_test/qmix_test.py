from run_experiment import extend_param_dicts

server_list = [
    ("gimli", [0,1,2,3,4,5,6,7], 2),
]

label = "QMIX_Refactor_Test_4"
config = "qmix"
env_config = "sc2"

n_repeat = 5

param_dicts = []

shared_params = {
    "t_max": 1000000,
    "epsilon_anneal_time": 50000,
    "buffer_size": 2000,
}

for mixer in [None, "vdn", "qmix"]:
    mixer_name = mixer
    if mixer is None:
        mixer_name = "iql"
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": "{}_sc2_3m".format(mixer_name),
            "mixer": mixer,
            "env_args.map_name": "3m_3m"
        },
        repeats=2)

    extend_param_dicts(param_dicts, shared_params,
        {
            "name": "{}_sc2_5m".format(mixer_name),
            "mixer": mixer,
            "env_args.map_name": "5m_5m"
        },
        repeats=3)
