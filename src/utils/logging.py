from collections import defaultdict
import logging


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    # TODO: Setup hdf logger

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))

        # TODO: Limit how much is being logged if it is too much
        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred:
            if key in self.sacred_info:
                self.sacred_info[key].append((t, value))
            else:
                self.sacred_info[key] = [(t, value)]


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

