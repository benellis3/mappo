from run_experiment import extend_param_dicts

server_list = [
    ("saruman", [0,1,4,5,6], 2),
    ("sauron", [0,1,2,3,4,5,6,7], 2),
    ("woma", [0,1,3,4,5,6,7], 2),
]

label = "smac__10_OCT_2018__ac"
config = "centralV"
env_config = "sc2"

n_repeat = 6

parallel_repeat = 4

param_dicts = []

shared_params = {
    "t_max": 3 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 24,
    "test_greedy": True,
    "env_args.obs_own_health": True, # We want this for SMAC(right?)
    "mask_before_softmax": False, # Better performance...
    "lr": 0.0005,
    "critic_lr": 0.0005,
    "td_lambda": 0.8,
    "epsilon_start": 0.5,
    "epsilon_finish": 0.01,
    "epsilon_anneal_time": 100 * 1000,
    "target_update_interval": 200,
    "env_args.map_name": ["3m_3m", "5m_5m", "2s_3z"],
}

# CentralV
extend_param_dicts(param_dicts, shared_params,
    {
        # Already using centralV config, don't need to change much else
        "name": "centralV",
    },
    repeats=parallel_repeat)

# COMA - Using actor_critic learner
extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_ac",
        "critic_q_fn": "coma",
        "critic_baseline_fn": "coma",
    },
    repeats=parallel_repeat)

# COMA - Using coma learner to check they are the same
extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_coma",
        "learner": "coma_learner",
        "recurrent_critic": False,
        "critic_q_fn": "coma",
        "critic_baseline_fn": "coma",
    },
    repeats=parallel_repeat)

