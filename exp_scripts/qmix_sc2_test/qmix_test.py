from run_experiment import extend_param_dicts

server_list = [
    ("gimli", [0,1,2,3,4], 2),
]

label = "qmix_s_vs_z"
config = "qmix"
env_config = "sc2"

n_repeat = 5

param_dicts = []

shared_params = {
    "t_max": 1000000,
    "epsilon_anneal_time": 50000,
    "buffer_size": 2000,
}

for mixer in [None, "qmix"]:
    mixer_name = mixer
    if mixer is None:
        mixer_name = "iql"
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": "{}_sc2_3s_vs_3z".format(mixer_name),
            "mixer": mixer,
            "env_args.map_name": "3s_vs_3z"
        },
        repeats=2)

    extend_param_dicts(param_dicts, shared_params,
        {
            "name": "{}_sc2_3s_vs_4z".format(mixer_name),
            "mixer": mixer,
            "env_args.map_name": "3s_vs_4z"
        },
        repeats=3)
