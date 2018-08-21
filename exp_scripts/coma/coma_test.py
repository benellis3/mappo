from run_experiment import extend_param_dicts

server_list = [
    ("gandalf", [0,1,2,3,4,5,6,7], 1),
    ("gollum", [0,1,2,3,4,5,6,7], 1),
]

label = "COMA_refactor_test_3"
config = "coma"
env_config = "sc2"

n_repeat = 5

param_dicts = []

shared_params = {
    "t_max": 1000000,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_agent_2e-5",
        "lr": 0.0002,
        "critic_lr": 0.0005,
        "env_args.map_name": "5m_5m"
    },
    repeats=4)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_both_2e-5",
        "lr": 0.0002,
        "critic_lr": 0.0002,
        "env_args.map_name": "5m_5m"
    },
    repeats=4)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_gamble",
        "lr": 0.0002,
        "critic_lr": 0.0002,
        "batch_size": 16,
        "batch_size_run": 16,
        "buffer_size": 16,
        "epsilon_start": 0.05,
        "epsilon_finish": 0.05,
        "env_args.map_name": "5m_5m"
    },
    repeats=3)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_eps05",
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "epsilon_start": 0.05,
        "epsilon_finish": 0.05,
        "env_args.map_name": "5m_5m"
    },
    repeats=3)


extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_target10k",
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "target_update_interval": 10000,
        "env_args.map_name": "5m_5m"
    },
    repeats=2)