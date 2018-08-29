from run_experiment import extend_param_dicts

server_list = [
    ("gollum", [4,5,6,7], 1),
    ("gandalf", [0,2,3,6], 1)
]

label = "COMA_refactor_test_4"
config = "coma"
env_config = "sc2"

n_repeat = 5

param_dicts = []

shared_params = {
    "t_max": 1500000,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_lam0.8_eps",
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "td_lambda": 0.8,
        "epsilon_start": .5,
        "epsilon_finish": .02,
        "env_args.map_name": "5m_5m"
    },
    repeats=4)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_agent_lr1e-4",
        "lr": 0.0001,
        "critic_lr": 0.0005,
        "td_lambda": 0.8,
        "env_args.map_name": "5m_5m"
    },
    repeats=4)