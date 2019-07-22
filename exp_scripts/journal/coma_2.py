from run_experiment import extend_param_dicts

server_list = [
    ("sauron", [1,1,2,3,4,5,6,7], 1),
]

label = "coma__19_July_2019_v1"
config = "coma_smac"
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
    "buffer_cpu_only": False,
}

maps = []

# Easier maps
maps += ["2s3z"]
maps += ["3s5z"]
maps += ["2s_vs_1sc"]

# Medium difficulty
maps += ["5m_vs_6m"]
maps += ["bane_vs_bane"]

# Hard
maps += ["3s5z_vs_3s6z"]
maps += ["27m_vs_30m"]

for map_name in maps:

    name = "coma__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name
        },
        repeats=parallel_repeat)

