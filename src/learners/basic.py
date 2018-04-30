import numpy as np

from components.transforms import _underscore_to_cap


class BasicLearner():
    """
    basis class for learners
    """

    def _add_stat(self, name, value, T_global):
        if isinstance(value, np.ndarray) and value.size == 1:
            value = float(value)

        if not hasattr(self, "_stats"):
            self._stats = {}
        if name not in self._stats:
            self._stats[name] = []
            self._stats[name+"_T"] = []
        self._stats[name].append(value)
        self._stats[name+"_T"].append(T_global)

        if hasattr(self, "max_stats_len") and len(self._stats) > self.max_stats_len:
            self._stats[name].pop(0)
            self._stats[name+"_T"].pop(0)

        if hasattr(self, "max_stats_len") and len(self._stats) > self.max_stats_len:
            self._stats[name].pop(0)
            self._stats[name+"_T"].pop(0)

        # log to sacred if enabled
        if hasattr(self.logging_struct, "sacred_log_scalar_fn"):
            self.logging_struct.sacred_log_scalar_fn(key=_underscore_to_cap(name), val=value)

        # log to tensorboard if enabled
        if hasattr(self.logging_struct, "tensorboard_log_scalar_fn"):
            self.logging_struct.tensorboard_log_scalar_fn(_underscore_to_cap(name), value, T_global)