from run_experiment import extend_param_dicts

server_list = [
    ("brown", [0,1,2,3,4,5,6,7], 2),
]

LABEL = "QMIX_sc2_test_26July2018"

n_repeat = 1

param_dicts = []

exp_name = "qmix/qmix_sc2"

shared_params = {
}

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "qmix_sc2_3m"
    },
    repeats=8)

extend_param_dicts(param_dicts, shared_params,
    {
        "name": "qmix_sc2_5m",
        "env_args.map_name": "5m_5m"
    },
    repeats=8)
