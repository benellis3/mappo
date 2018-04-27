import importlib, ntpath, os
REGISTRY = {}

# automatically fill registry from experiment entries
subdirs = [x[0] for x in os.walk(ntpath.dirname(__file__))]
for subdir in subdirs:
    if os.path.isfile(os.path.join(subdir, "config.py")):
        fname=os.path.split(subdir)[-1]
        REGISTRY[fname] = importlib.import_module(".{}.config".format(fname), package="config.defaults").get_cfg
