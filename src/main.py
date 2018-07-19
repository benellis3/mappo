import numpy as np
import os
from os.path import dirname, abspath
import pymongo
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th

from components.transforms import _merge_dicts
from run import run
from utils.logging import get_logger

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("deepmarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

mongo_client = None

def setup_mongodb(conf_str):
    # The central mongodb for our deepmarl experiments
    # You need to set up local port forwarding to ensure this local port maps to the server
    # if conf_str == "":
    # db_host = "localhost"
    # db_port = 27027 # Use a different port from the default mongodb port to avoid a potential clash

    from config.mongodb import REGISTRY as mongo_REGISTRY
    mongo_conf = mongo_REGISTRY[conf_str](None, None)
    db_url = mongo_conf["db_url"]
    db_name = mongo_conf["db_name"]

    client = None
    mongodb_fail = True

    # Try 5 times to connect to the mongodb
    for tries in range(5):
        # First try to connect to the central server. If that doesn't work then just save locally
        maxSevSelDelay = 10000  # Assume 10s maximum server selection delay
        try:
            # Check whether server is accessible
            logger.info("Trying to connect to mongoDB '{}'".format(db_url))
            client = pymongo.MongoClient(db_url, ssl=True, serverSelectionTimeoutMS=maxSevSelDelay)
            client.server_info()
            # If this hasn't raised an exception, we can add the observer
            ex.observers.append(MongoObserver.create(url=db_url, db_name=db_name, ssl=True)) # db_name=db_name,
            logger.info("Added MongoDB observer on {}.".format(db_url))
            mongodb_fail = False
            break
        except pymongo.errors.ServerSelectionTimeoutError:
            logger.warning("Couldn't connect to MongoDB on try {}".format(tries + 1))

    if mongodb_fail:
        logger.error("Couldn't connect to MongoDB after 5 tries!")
        # TODO: Maybe we want to end the script here sometimes?

    return client

@ex.main
def my_main(_run, _config, _log, env_args):
    global mongo_client

    # Setting the random seed throughout the modules
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    env_args['seed'] = _config["seed"]

    # run the framework
    run(_run, _config, _log, mongo_client)

    # force exit
    os._exit()

if __name__ == '__main__':
    import os

    from copy import deepcopy
    params = deepcopy(sys.argv)

    defaults = []
    config_dic = {}

    # manually parse for experiment tags
    del_indices = []
    exp_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--exp_name":
            del_indices.append(_i)
            exp_name = _v.split("=")[1]
            break

    # load experiment config (if there is such as thing)
    exp_dic = None
    if exp_name is not None:
        from config.experiments import REGISTRY as exp_REGISTRY
        assert exp_name in exp_REGISTRY, "Unknown experiment name: {}".format(exp_name)
        exp_dic = exp_REGISTRY[exp_name](None, logger)
        if "defaults" in exp_dic:
            defaults.extend(exp_dic["defaults"].split(" "))
            del exp_dic["defaults"]
        config_dic = deepcopy(exp_dic)

    # check for defaults in command line parameters
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--default_cfgs":
            del_indices.append(_i)
            defaults.extend(_v.split("=")[1].split(" "))
            break

    # load default configs in order
    for _d in defaults:
        from config.defaults import REGISTRY as def_REGISTRY
        def_dic = def_REGISTRY[_d](config_dic, logger)
        config_dic = _merge_dicts(config_dic, def_dic)

    #  finally merge with experiment config
    if exp_name is not None:
        config_dic = _merge_dicts(config_dic, exp_dic)

    # add results path to config
    config_dic["local_results_path"] = results_path

    # now add all the config to sacred
    ex.add_config(config_dic)

    # Check if we don't want to save to sacred mongodb
    no_mongodb = False
    if "--no-mongo" in sys.argv:
        no_mongodb = True

    # delete indices that contain custom experiment tags
    for _i in sorted(del_indices, reverse=True):
        del params[_i]

    if not no_mongodb:
        mongo_client = setup_mongodb(config_dic["mongodb_profile"])

    # Save to disk by default for sacred, even if we are using the mongodb
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

