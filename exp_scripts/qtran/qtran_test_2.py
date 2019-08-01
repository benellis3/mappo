from run_experiment import extend_param_dicts

server_list = [
    # ("orion", [0,1,2,3,4,5,6,7], 1),
    ("gandalf", [0,1,2,3,4], 1),
]

label = "qtran_test_2__1_Aug_2019_v2"
config = "qtran"
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
    "save_model_interval": 2000 * 1000,
    "test_interval": 10000,
    "log_interval": 10000,
    "runner_log_interval": 10000,
    "learner_log_interval": 10000,
    "buffer_cpu_only": True, # 5k buffer is too big for VRAM!
}


maps = []

# Easy maps
maps += ["2s3z"]

for map_name in maps:
    name = "qtran__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name,
            "network_size": "big",
            "mixing_embed_dim": 128,
            "opt_loss": [1], # td_error and these 2 optimise disjoint parameters, so only their relative scaling is important
            "nopt_min_loss": [0.1, 1, 10],
        },
        repeats=parallel_repeat)

for map_name in maps:
    name = "qtran__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name,
            "network_size": "small",
            "opt_loss": [1],
            "nopt_min_loss": [0.1, 10], # 1 was already run in qtran_test_1
        },
        repeats=parallel_repeat)
