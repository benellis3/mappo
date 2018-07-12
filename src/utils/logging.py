import logging
from components.episode_buffer import BatchEpisodeBuffer
from sacred.observers import FileStorageObserver
from sacred.commandline_options import CommandLineOption
import os

def append_scalar(run, key, val):
    if key in run.info:
        run.info[key].append(val)
    else:
        run.info[key] = [val]

def log_stats(log, stats):
    if not stats:
        return

    log_string = ""
    for k, v in stats.items():
        log_string += k + "= %0.2f, " % v
    log.info(log_string)

# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

# set up a custom observer
class CustomIdObserver(FileStorageObserver):
    priority = 50  # very high priority

    @classmethod
    def create(cls, basedir, resource_dir=None, source_dir=None,
                template=None, priority=priority):

        if not os.path.exists(basedir):
           os.makedirs(basedir)
        resource_dir = resource_dir or os.path.join(basedir, '_resources')
        source_dir = source_dir or os.path.join(basedir, '_sources')
        if template is not None:
            if not os.path.exists(template):
                raise FileNotFoundError("Couldn't find template file '{}'"
                        .format(template))
        else:
            template = os.path.join(basedir, 'template.html')
            if not os.path.exists(template):
                template = None

        return cls(basedir, resource_dir, source_dir, template, priority)

    def __init__(self, basedir, resource_dir, source_dir, template, priority):
        super(CustomIdObserver, self).__init__(basedir, resource_dir, source_dir, template, priority)

    def started_event(self, ex_info, command, host_info, start_time,  config, meta_info, _id):

        custom_id = config["run_name"]

        if isinstance(custom_id, list):

            length = len(custom_id)
            new_id = ""

            for i in range(length):
                if custom_id[i] not in config:
                    print("Config ERROR: {} not in {}".format(custom_id[i], str(config)))
                    new_id = _id
                    break
                if i != length - 1:
                    new_id += str(config[custom_id[i]]) + '_'
                else:
                    new_id += str(config[custom_id[i]])

        elif isinstance(custom_id, str) and len(custom_id) > 0:
            new_id = custom_id
        else:
            new_id = _id

        return super(CustomIdObserver, self).started_event(ex_info, command, host_info, start_time,  config, meta_info, new_id)

    def get_save_dir(self):
        return self.dir


class ResultDir(CommandLineOption):
    """Add a file-storage observer to the experiment."""

    short_flag = 'R'
    arg = 'BASEDIR'
    arg_description = "Base-directory to write the runs to"

    @classmethod
    def apply(cls, args, run):

        # this overrides the default observer
        observer = CustomIdObserver.create(args)
        run.observers = [ observer ]



class HDFLogger():

    def __init__(self, path, name):
        from tables import open_file
        self.path = path
        self.h5file = open_file("{}.h5".format(name), mode="w", title="Experiment results: {}".format(name))
        group = self.h5file.create_group("/", 'detector', 'Detector information')
        pass

    def log(self, key, item, T_env):

        if isinstance(item, BatchEpisodeBuffer):
            if not hasattr(self.h5file.root, "learner_samples"):
                self.h5file.create_group("/", "learner_samples", 'Learner samples')

            if not hasattr(self.h5file.root, "learner_samples"):
                self.h5file.create_group("/learner_samples/", "{}".format(T_env), 'Learner samples T_env:{}'.format(T_env))

            if not hasattr(getattr(self.h5file.root.learner_samples, "{}".format(T_env)), "_transition"):
                self.h5file.create_group("/learner_samples/{}".format(T_env), "_transition", 'Transition-wide data')

            if not hasattr(getattr(self.h5file.root.learner_samples, "{}".format(T_env)), "_episode"):
                self.h5file.create_group("/learner_samples/{}".format(T_env), "_episode", 'Episode-wide data')

            # if table layout has not been created yet, do it now:
            for _c, _pos in item.columns._transition.items():
                if not hasattr(self.h5file.root.learner_samples, _c):
                    self.h5file.root.learner_samples.create_earray(getattr(self.h5file.root.learner_samples, "{}".format(T_env))._transition,
                                                                   _c, obj=item[_c])
                else:
                    getattr(self.h5file.root.learner_samples._transition, _c).append(item[_c])

            # if table layout has not been created yet, do it now:
            for _c, _pos in item.columns._episode.items():
                if not hasattr(self.h5file.root.learner_samples, _c):
                    self.h5file.root.learner_samples.create_earray(getattr(self.h5file.root.learner_samples, "{}".format(T_env))._episode,
                                                                   _c, obj=item[_c])
                else:
                    getattr(self.h5file.root.learner_samples._episode, _c).append(item[_c])

        else:
            # item needs to be scalar!

            if not hasattr(self.h5file.root, "log_values"):
                self.h5file.create_group("/", "log_values", 'Log Values')

            if not hasattr(self.h5file.root.log_values, key):
                from tables import Float32Atom, IntAtom
                self.h5file.root.learner_samples.create_earray(self.h5file.root.log_values,
                                                               key, atom=Float32Atom, shape=[1])
                self.h5file.root.learner_samples.create_earray(self.h5file.root.log_values,
                                                               "{}_T_env".format(key), atom=IntAtom, shape=[1])
            else:
                getattr(self.h5file.root.log_values, key).append(item)
                getattr(self.h5file.root.log_values, "{}_T_env".format(key)).append(T_env)

