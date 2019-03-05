from run_experiment import extend_param_dicts

server_list = [
    # ("saruman", [0,1,2,3,4,5,6], 2),
]

label = "journal_vdn_runs__5_Mar_2019__v1"
config = "vdn_smac"
env_config = "sc2"

n_repeat = 12 # Just incase some die

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

maps = []

# Symmetric (6)
maps += ["3m", "8m", "25m", "2s3z", "3s5z", "MMM"]

# Asymmetric (6)
maps += ["5m_vs_6m", "8m_vs_9m", "10m_vs_11m", "27m_vs_30m"]
maps += ["MMM2", "3s5z_vs_3s6z"]

# Micro (10)
maps += ["3s_vs_3z", "3s_vs_4z", "3s_vs_5z"]
maps += ["2m_vs_1z"]
maps += ["2s_vs_1sc"]
maps += ["6h_vs_8z"]
maps += ["corridor"]
maps += ["bane_vs_bane"]
maps += ["so_many_banelings"]
maps += ["2c_vs_64zg"]

for map_name in maps:

    name = "vdn__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name
        },
        repeats=parallel_repeat)

