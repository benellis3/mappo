from run_experiment import extend_param_dicts

server_list = [
    ("woma", [0,1,2,3,4,5,6,7], 1),
]

label = "qmix_more_tests__20_Aug_2019_v1"
config = "qmix_journal"
env_config = "sc2"

n_repeat = 5 # Just incase some die

parallel_repeat = 1

param_dicts = []

shared_params = {
    "t_max": 2 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 32,
    "test_greedy": True,
    "env_args.obs_own_health": True,
    "save_model": True,
    "save_model_interval": 25 * 1000,
    "test_interval": 10000,
    "log_interval": 10000,
    "runner_log_interval": 10000,
    "learner_log_interval": 10000,
    "buffer_cpu_only": True, # 5k buffer is too big for VRAM!
}

maps = []
maps += ["2s3z"]
maps += ["3s5z"]

for map_name in maps:
    # Longer epsilon anneal time
    name = "qmix__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
           {
               "name": name,
               "env_args.map_name": map_name,
               "skip_connections": [False],
               "gated": False,
               "mixing_embed_dim": [32],
               "hypernet_layers": [2],
               "epsilon_anneal_time": 1 * 1000 * 1000,
           },
           repeats=parallel_repeat)

    name = "vdn__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name,
            "mixer": "vdn",
            "epsilon_anneal_time": 1 * 1000 * 1000,
        },
        repeats=parallel_repeat)

maps = []
maps += ["3s5z"]
maps += ["3s_vs_5z"]

for map_name in maps:
    name = "qmix_ff_{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
               {
                   "name": name,
                   "env_args.map_name": map_name,
                   "skip_connections": [False],
                   "gated": False,
                   "mixing_embed_dim": [32],
                   "hypernet_layers": [2],
                   "agent": "ff",
               },
               repeats=parallel_repeat)

maps = []
maps += ["MMM2"]

for map_name in maps:
    name = "qmix_train_{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
               {
                   "name": name,
                   "env_args.map_name": map_name,
                   "skip_connections": [False],
                   "gated": False,
                   "mixing_embed_dim": [32],
                   "hypernet_layers": [2],
                   "training_iters": 4,
               },
               repeats=parallel_repeat)

    name = "vdn_train_{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name,
            "mixer": "vdn",
            "training_iters": 4,
        },
        repeats=parallel_repeat)