import os
import numpy as np
from itertools import product
import subprocess
import copy
import time
import sys
import importlib
import stat


def gen_params_prod(param_dict):
    keys = list(param_dict.keys())

    for k in param_dict:
        if isinstance(param_dict[k], list):
            param_dict[k] = map(str, param_dict[k])
        else:
            param_dict[k] = [str(param_dict[k])]

    vals = param_dict.values()
    # print(vals)

    val_prod = list(product(*vals))
    # print(val_prod)

    params_prod = []
    for setting in val_prod:
        params = "with "
        for i, e in enumerate(setting):
            params += keys[i] + "=" + e + " "

        params_prod.append(params)

    return params_prod


def build_param_dict(shared_params, exp_params):
    # copy
    result = copy.deepcopy(shared_params)
    # add experiment parameters to shared params
    result.update(exp_params)
    return result


def extend_param_dicts(param_dicts, shared_params, exp_params, repeats=1):
    # copy
    result = copy.deepcopy(shared_params)
    # add experiment parameters to shared params
    result.update(exp_params)
    for i in range(repeats):
        # result.update({"par_id": [i]})
        param_dicts.append(copy.deepcopy(result))
    return param_dicts


def main(config_name, run=False):
    config = importlib.import_module(config_name)
    server_list = config.server_list
    param_dicts = config.param_dicts
    n_repeat = config.n_repeat
    config_name = "--config='{}'".format(config.config)
    config_env = "--env-config='{}'".format(config.env_config)

    all_experiments = [exp for param_dict in param_dicts for exp in gen_params_prod(param_dict)]
    n_experiments = len(all_experiments)

    if config.server_list == "ngc":
        if actually_run:
            print("Going to actually run on ngc")
            time.sleep(10)
        log = []
        print("#!/bin/bash\nshopt -s expand_aliases\nalias ngc=\"python /auto/users/tabhid/.local/lib/python2.7/site-packages/ngccli/ngc.py\"\n")
        for i, params in enumerate(all_experiments):
            log.append(params)
            command = '\'python3 src_new/main.py {} {}\''.format(exp_name, params)
            name = "{}_{}".format(config.exp_name, i)
            if True:
                # subprocess.Popen(["ngc", "batch", "run", "--instance", "ngcv1", "--name", name, "--image", "oxford_ml01/pymarl:v0.2", "--result", "/pymarl/results", "--ace", "nv-us-west-2", "--datasetid", "9929:/pymarl/src_new", "--command", command])
                cmd_str = " ".join(["ngc", "batch", "run", "--instance", "ngcv1", "--name", name, "--image", "oxford_ml01/pymarl:v0.2", "--result", "/pymarl/results",
                              "--ace", "nv-us-west-2", "--datasetid", "9936:/pymarl/src_new", "--command", command])
                print(cmd_str)
            else:
                print(command)
            # time.sleep(2)
        with open("exp_logs/" + config.label + "_log.txt", "w") as f:
            for line in log:
                f.write(line + "\n")
    else:
        n_available = sum([len(x[1])*x[2] for x in server_list])
        print(n_available, "servers available")
        print(n_experiments, "experiments requested")
        assert n_available >= n_experiments
        if actually_run:
            print("Going to actually run on local servers!")
            time.sleep(10)

        log = []
        exp_idx = 0
        for server, gpu_list, exps_per_gpu in server_list:
            for _ in range(exps_per_gpu):
                for gpu in gpu_list:
                    if exp_idx >= n_experiments:
                        break
                    params = all_experiments[exp_idx]
                    # params += "server=" + server +" gpu=" + str(gpu)
                    params = "{} {} {} label={}".format(config_name, config_env, params, config.label)
                    log.append(params)
                    # print(params)
                    if not run:
                        print(server, "GPU:",str(gpu), "Repeats:",str(n_repeat), params)
                    else:
                        subprocess.Popen(["exp_scripts/run_on_server.sh", server, str(gpu), str(n_repeat), params])
                        time.sleep(10)

                    exp_idx += 1

        with open("exp_logs/" + config.label + "_log.txt", "w") as f:
            for line in log:
                f.write(line + "\n\n")

if __name__ == "__main__":
    actually_run = False
    if "--run" in sys.argv:
        actually_run = True
    main(sys.argv[1], run=actually_run)