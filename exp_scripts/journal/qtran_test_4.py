from run_experiment import extend_param_dicts

server_list = [
    # ("orion", [0,1,2,3,4,5,6,7], 1),
    ("dgx1", [0,1,2,3,4,5,6,7], 2),
    # ("savitar", [0,1,2,3,4,5,6,7], 1),
]

label = "qtran_test_3__10_Oct_2019_v1"
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
maps += ["3s5z"]

for map_name in maps:
    name = "qtran__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name,
            "network_size": ["small", "big"],
            "mixing_embed_dim": [64],
            "opt_loss": [1], # td_error and these 2 optimise disjoint parameters, so only their relative scaling is important
            "nopt_min_loss": [0.1, 1, 10],
            "qtran_arch": ["coma_critic", "qtran_paper"]
        },
        repeats=parallel_repeat)

