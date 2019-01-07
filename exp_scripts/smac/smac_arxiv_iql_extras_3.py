from run_experiment import extend_param_dicts

server_list = [
    ("gollum", [3, 4], 1),
    # ("oc_gpu_2", [0,1], 1),
    # ("oc_gpu_3", [1], 1),
    # ("oc_gpu_3", [0], 3),
    # ("oc_gpu_4", [0,1], 1),
]

label = "smac_arxiv__iql_runs__oracle__28_Dec_2018__v4_extra_3"
config = "qmix_smac"
env_config = "sc2"

n_repeat = 3 # Just incase some die

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

# # Symmetric (6)
# maps += ["3m", "8m", "25m", "2s3z", "3s5z", "MMM"]
#
# # Asymmetric (6)
# maps += ["5m_6m", "8m_9m", "10m_11m", "27m_30m"]
# maps += ["MMM2", "3s5z_3s6z"]
#
# # Micro (10)
# maps += ["3s_vs_3z", "3s_vs_4z", "3s_vs_5z"]
# maps += ["micro_2M_Z"]
# maps += ["micro_baneling"]
# maps += ["micro_colossus"]
# maps += ["micro_corridor"]
# maps += ["micro_focus"]
# maps += ["micro_retarget"]
# maps += ["micro_bane", "micro_bane", "micro_bane"]
# maps += ["10m_11m"]
# maps += ["MMM"]
# maps += ["MMM2"]

# maps += ["25m", "27m_30m", "micro_colossus", "micro_bane"]
# maps += ["8m_9m", "3s5z_3s6z"]

maps += ["25m"]
maps += ["27m_30m"]

for map_name in maps:

    name = "iql__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "mixer": None,
            "env_args.map_name": map_name
        },
        repeats=parallel_repeat)

