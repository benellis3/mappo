def get_cfg(existing_cfg, _log):
    """
    generates
    """
    _sanity_check(existing_cfg, _log)
    import ntpath, os, ruamel.yaml as yaml
    with open(os.path.join(os.path.dirname(__file__), "{}.yml".format(ntpath.basename(__file__).split(".")[0])),
              'r') as stream:
        try:
            ret = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            assert "Default config yaml for '{}' not found!".format(os.path.splitext(__file__)[0])

    # turn off coma exploration if it is set!
    if existing_cfg["learner"] == "coma":
        ret["coma_exploration_mode"] = None

    return ret

def _sanity_check(existing_cfg, _log):
    """
    """
    return