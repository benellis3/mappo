from run_experiment import extend_param_dicts

server_list = [
    ("saruman", [5,6], 1),
    ("sauron", [0,5,6,7], 1),
]

label = "coma__19_July_2019_v1"
config = "coma_smac"
env_config = "sc2"

n_repeat = 2 # Just incase some die

parallel_repeat = 3

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

maps += ["1c3s5z"]
maps += ["10m_vs_11m"]

for map_name in maps:

    name = "coma__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name
        },
        repeats=parallel_repeat)

