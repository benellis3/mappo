from run_experiment import extend_param_dicts

server_list = [
    ("savitar", [0], 1),
]

label = "journal_vdn_runs__5_Mar_2019__v1"
# label = "testing"
config = "vdn_smac"
env_config = "sc2"

n_repeat = 4 # Just incase some die

parallel_repeat = 1

param_dicts = []

shared_params = {
    "t_max": 10 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 24,
    "test_greedy": True,
    "env_args.obs_own_health": True,
    "save_model": True,
    "save_model_interval": 2000 * 1000,
    "test_interval": 20000,
    "log_interval": 20000,
    "runner_log_interval": 20000,
    "learner_log_interval": 20000,
    "buffer_cpu_only": True, # 5k buffer is too big for VRAM!
}

# Just need 4 more VDN 3m seeds
maps = []
maps += ["3m"]

for map_name in maps:

    name = "vdn__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name
        },
        repeats=parallel_repeat)

