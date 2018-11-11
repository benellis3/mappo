from run_experiment import extend_param_dicts

server_list = [
    ("gimli", [0,1,2,3,4,5,6,7], 2),
    ("saruman", [0,1,2,3,4,5,6], 2),
]

label = "coma_some_maps__10_Nov_2018"
config = "coma"
env_config = "sc2"

n_repeat = 4

parallel_repeat = 4

param_dicts = []

shared_params = {
    "t_max": 20 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 24,
    "test_greedy": True,
    "env_args.obs_own_health": True,  # We want this for SMAC(right?)

    "mask_before_softmax": False,  # Better performance...
    "lr": 0.0005,
    "critic_lr": 0.0005,
    "td_lambda": 0.8,
    "epsilon_start": 0.5,
    "epsilon_finish": 0.01,
    "epsilon_anneal_time": 100 * 1000,

    "target_update_interval": 200,
    # "env_args.map_name": ["3s_vs_3z", "3s_vs_4z", "3s_vs_5z", "3s_vs_6z",],
    "save_model": True,
    "save_model_interval": 5000 * 1000,
    "test_interval": 20000,
    "log_interval": 25000,
    "runner_log_interval": 25000,
    "learner_log_interval": 25000,
}

for map in ["micro_retarget", "8m_8m", "MMM", "micro_baneling", "5m_6m", "3s_5z", "3s_vs_3z"]:

    name = "coma"
    name += "__{}".format(map)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map
        },
        repeats=parallel_repeat)

