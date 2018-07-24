from run_experiment import extend_param_dicts

server_list = [
    ("dgx1", [0,1,2,3,4,5,6,7], 1),
]

LABEL = "QMIX_sc2_test_24July2018"

n_repeat = 4

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
