from .iql import IQLLearner, IQLLoss
from models import REGISTRY as model_registry
import torch as th
from torch.optim import RMSprop


class QMIXLoss(IQLLoss):

    def __init__(self, state_size, n_agents, mixing_dim):
        super(QMIXLoss, self).__init__()
        self.mixer = model_registry["qmix_mixer"](state_size, n_agents, mixing_dim)

    def parameters(self):
        return self.mixer.parameters()


class QMIXLearner(IQLLearner):

    def __init__(self, multiagent_controller, logging_struct=None, args=None):
        super(QMIXLearner, self).__init__(multiagent_controller, logging_struct, args)

        self.loss_func = QMIXLoss
        self.args = args


    def create_models(self, transition_scheme):

        self.agent_parameters = []
        for agent in self.agents:
            self.agent_parameters.extend(agent.get_parameters())
            if self.args.share_agent_params:
                break

        # Make mixer
        self.state_size = transition_scheme.get_by_name("state")["size"]
        self.loss_func = self.loss_func(self.state_size, self.n_agents, self.args.mixing_network_dim)

        # Move mixer model to gpu if required
        if self.args.use_cuda:
            self.loss_func.mixer.cuda()

        # To make minimal changes its still called agent_optimiser and agent_parameters
        self.agent_parameters.extend(self.loss_func.parameters())

        self.agent_optimiser = RMSprop(self.agent_parameters, lr=self.args.lr_q)

        # calculate a grand joint scheme
        self.joint_scheme_dict = self.multiagent_controller.joint_scheme_dict
        pass

