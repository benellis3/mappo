from run_experiment import extend_param_dicts

server_list = [
    ("brown", [0,1,2,3,4,5], 1),
    ("dgx1", [6,7], 3),
    ("gandalf", [0,1,2,3,4,5,6], 1),
    ("woma", [1,2,3,5], 1),
    # ("dgx1", [0,1,2,3,4,5], 1),
    ("savitar", [0,1,2,3,4,5,6,7], 2),
    ("gandalf", [7,0,1], 1),
    ("brown", [0,1,2,3,4,5], 1),
    ("gandalf", [2,3,4,5,6,7], 1),
    ("gollum", [0,1,2,3,4,5,6,7], 1),
    ("gollum", [0], 1),
    # ("gollum", [0,1,2,3,4,5,6,7], 1),
    # ("saruman", [0,1,2,3,4,5,6], 2),
]

label = "smac_arxiv__qmix_runs__27_Nov_2018__v3"
config = "qmix_smac"
env_config = "sc2"

n_repeat = 5 # Just incase some die

parallel_repeat = 3

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
maps += ["5m_6m", "8m_9m", "10m_11m", "27m_30m"]
maps += ["MMM2", "3s5z_3s6z"]

# Micro (10)
maps += ["3s_vs_3z", "3s_vs_4z", "3s_vs_5z"]
maps += ["micro_2M_Z"]
maps += ["micro_baneling"]
maps += ["micro_colossus"]
maps += ["micro_corridor"]
maps += ["micro_focus"]
maps += ["micro_retarget"]


for map_name in maps:

    name = "qmix__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name
        },
        repeats=parallel_repeat)

