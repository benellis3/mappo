from run_experiment import extend_param_dicts

server_list = [
    # ("brown", [0,1,2,3,4,5], 2),
    # ("woma", [0,1,2,3], 1),
    # ("dgx1", [0,1,2,3,4,5,6,7], 3),
    # ("savitar", [0,1,2,3,4,5,6,7], 2),
    # ("sauron", [0,1,2,3,4,5,6,7], 2),
    # ("gollum", [0,1,2,3,4,5,6,7], 1),
    # ("gollum", [0,1,2,3], 1),

    # ("gollum", [4,5,6,7], 1),
    # ("gimli", [0,1,2,3,4,5,6,7], 3),
    
    ("saruman", [0,1,2,3,4,5,6], 3),
]

label = "smac_aamas__12_Nov_2018"
config = "coma"
env_config = "sc2"

n_repeat = 6

parallel_repeat = 3

param_dicts = []

shared_params = {
    "t_max": 10 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 24,
    "test_greedy": True,
    "env_args.obs_own_health": True, # We want this for SMAC(right?)

    "mask_before_softmax": False,  # Better performance...
    "lr": 0.0005,
    "critic_lr": 0.0005,
    "td_lambda": 0.8,
    "epsilon_start": 0.5,
    "epsilon_finish": 0.01,
    "epsilon_anneal_time": 100 * 1000,

    "target_update_interval": 200,
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
maps += ["10m_11m"]

# Micro
maps += ["3s_vs_3z", "micro_2M_Z"]

for map_name in maps:

    name = "coma__{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name
        },
        repeats=parallel_repeat)

