from run_experiment import extend_param_dicts

server_list = [
    ("brown", [0], 1),
]

LABEL = "Coma_Baseline_PP_Test"

n_repeat = 1

param_dicts = []

exp_name = "coma_baseline_pp"

shared_params = {
    "use_tensorboard": False,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "comabaseline_test"
    },
    repeats=1)
