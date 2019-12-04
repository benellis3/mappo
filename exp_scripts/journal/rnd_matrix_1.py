from run_experiment import extend_param_dicts

server_list = [
    ("legolas", [0], 18),
]

label = "random_matrix_games__4_Dec_2019_v1"
config = "qmix_journal"
env_config = "rnd_matrix_game"

n_repeat = 10 # Just incase some die

parallel_repeat = 1

param_dicts = []

shared_params = {
    "t_max": 20 * 1000 + 50 * 10,
    "test_interval": 100,
    "test_nepisode": 1,
    "test_greedy": True,
    "save_model": False,
    "log_interval": 100,
    "runner_log_interval": 100,
    "learner_log_interval": 100,
    "buffer_cpu_only": True, # 5k buffer is too big for VRAM!
    "buffer_size": 500,
    "epsilon_finish": 1,
    "agent": "ff",
    "obs_last_action": False,
}

for agents in [2,3,4]:
    for actions in [2,3,4]:
        name = "qmix"
        extend_param_dicts(param_dicts, shared_params,
            {
                "name": name,
            },
            repeats=parallel_repeat)

        name = "vdn"
        extend_param_dicts(param_dicts, shared_params,
            {
                "name": name,
                "mixer": "vdn",
            },
            repeats=parallel_repeat)

