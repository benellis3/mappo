from run_experiment import extend_param_dicts

server_list = [
    ("brown", [0,1,2,3,4,5], 2),
    ("woma", [0,1,2,3], 1),
    ("dgx1", [0,1,2,3,4,5,6,7], 3),
    ("savitar", [0,1,2,3,4,5,6,7], 2),
    ("sauron", [0,1,2,3,4,5,6,7], 2),
    ("gollum", [0,1,2,3,4,5,6,7], 1),
    ("gollum", [0,1,2,3], 1),
    # ("gollum", [4,5,6,7], 1),
    # ("gimli", [0,1,2,3,4,5,6,7], 2),
    # ("saruman", [0,1,2,3,4,5,6], 2),
]

label = "smac_aamas__12_Nov_2018"
config = "qmix_parallel"
env_config = "sc2"

n_repeat = 4

parallel_repeat = 6

param_dicts = []

shared_params = {
    "t_max": 10 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 24,
    "test_greedy": True,
    "env_args.obs_own_health": True, # We want this for SMAC(right?)
    "epsilon_start": 1.0,
    "epsilon_finish": 0.05,
    "epsilon_anneal_time": [20 * 1000],
    "target_update_interval": 200,
    # "env_args.map_name": ["3s_vs_3z", "3s_vs_4z", "3s_vs_5z", "3s_vs_6z",],
    "save_model": True,
    "save_model_interval": 2000 * 1000,
    "test_interval": 20000,
    "log_interval": 20000,
    "runner_log_interval": 20000,
    "learner_log_interval": 20000,
}

maps = []

# Symmetric
maps += ["3m_3m", "2s_3z"]
maps += ["MMM", "8m_8m"]

# Asym
maps += ["8m_9m", "MMM2"]

# Micro
maps += ["micro_baneling", "micro_corridor", "micro_retarget"]
maps += ["3s_vs_3z", "3s_vs_4z", "3s_vs_5z"]

maps += ["micro_colossus2", "micro_focus"]


for map_name in maps:

    name = "qmix__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name
        },
        repeats=parallel_repeat)

