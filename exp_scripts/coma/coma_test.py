from run_experiment import extend_param_dicts

server_list = [
    ("gandalf", [0,1,2,3,4,5,6,7], 1),
    ("gollum", [0,1,2,3,4,5,6,7], 1),
]

label = "COMA_refactor_test_2"
config = "coma"
env_config = "sc2"

n_repeat = 5

param_dicts = []

shared_params = {
    "t_max": 1000000,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_critic_lowlr",
        "lr": 0.0005,
        "critic_lr": 0.0001,
        "env_args.map_name": "5m_5m"
    },
    repeats=4)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_agent_lowlr",
        "lr": 0.0001,
        "critic_lr": 0.0005,
        "env_args.map_name": "5m_5m"
    },
    repeats=4)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_bs32",
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "batch_size": 32,
        "batch_size_run": 32,
        "buffer_size": 32,
        "env_args.map_name": "5m_5m"
    },
    repeats=3)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_5m_eps",
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "epsilon_start": 0.02,
        "epsilon_finish": 0.02,
        "env_args.map_name": "5m_5m"
    },
    repeats=3)


extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_sc2_3s_5z",
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "env_args.map_name": "3s_5z"
    },
    repeats=2)