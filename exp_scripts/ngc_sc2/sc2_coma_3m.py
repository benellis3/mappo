from run_experiment import extend_param_dicts

server_list = "ngc"

# LABEL = "Coma_Baseline_PP_Test"

n_repeat = 1

param_dicts = []

exp_name = "coma_jakob_sc2_3m"

shared_params = {
    "use_tensorboard": False,
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "coma_jakob_sc2_3m_ngc"
    },
    repeats=15)
