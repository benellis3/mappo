from .iql import IQLLearner, IQLLoss
from models import REGISTRY as model_registry


class VDNLoss(IQLLoss):

    def __init__(self):
        super(VDNLoss, self).__init__()
        self.mixer = model_registry["vdn_mixer"]()


class VDNLearner(IQLLearner):

    def __init__(self, multiagent_controller, logging_struct=None, args=None):
        super(VDNLearner, self).__init__(multiagent_controller, logging_struct, args)

        self.loss_func = VDNLoss
        # TODO: Maybe we want some extra logging for other stuff as well
