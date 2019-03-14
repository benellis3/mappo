from run_experiment import extend_param_dicts

server_list = [
    ("smac1", [0], 1),
    ("smac1", [1], 3),
    ("smac1", [0], 1),
    ("smac2", [0,1], 2),
    ("smac3", [0,1], 3),
    ("smac4", [0], 1),
    ("smac4", [0,1], 2),
]

label = "journal_ablation_runs__13_Mar_2019__v1"
# label = "testing"
config = "qmix_lin_smac"
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
# maps += ["micro_bane"]

maps += ["2s3z", "3m", "MMM", "micro_bane", "25m", "micro_retarget"]

for map_name in maps:

    # QMIX NS
    name = "qmix_ns__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name,
            "mixer": "qmix_ns"
        },
        repeats=parallel_repeat)

    # QMIX LIN
    name = "qmix_lin__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name,
            "mixer": "qmix_lin"
        },
        repeats=parallel_repeat)

    # VDN STATE
    name = "vdn_state__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name,
            "mixer": "vdn_state"
        },
        repeats=parallel_repeat)

