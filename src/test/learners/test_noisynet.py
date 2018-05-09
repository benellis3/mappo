from utils.blitzz.agents import REGISTRY as ag_REGISTRY
#import utils.blitzz.noisynet
#from utils.blitzz.noisynet import NoisyNStepRunner, NoisyCOMALearner
from utils.blitzz.learners.coma import COMALearner
from utils.dict2namedtuple import convert
from utils.blitzz.scheme import Scheme
from utils.blitzz.runners import NStepRunner
import numpy as np

env_args = {
    # SC2
    'map_name': '5m_5m', 'n_enemies': 5, 'episode_limit': 50, 'move_amount': 2, 'step_mul': 8,
    'difficulty': "3", 'reward_only_positive': True, 'reward_negative_scale': 0.5, 'reward_death_value': 10,
    'reward_win': 200, 'reward_scale': True, 'reward_scale_rate': 2, 'state_last_action': True,
    'heuristic_function': False, 'measure_fps': True,
    # Payoff
    'payoff_game': "monotonic",
    # Predator Prey
    'prey_movement': "escape", 'predator_prey_shape': (3, 3), 'nagent_capture_enabled': False,
    # Stag Hunt
    'stag_hunt_shape': (3, 3), 'stochastic_reward_shift_optim': None, 'stochastic_reward_shift_mul': None,
    'global_reward_scale_factor': 1.0, 'state_variant': "grid"  # comma-separated string
}

conf = {"n_agents": 4, "obs_agent_id": 0, "obs_last_action": True, "obs_epsilon": False,
        "n_actions": 5, "learner": "coma", "env": "pred_prey",
        "t_max": 1000000, "batch_size_run": 32, "env_args": env_args,
        "share_params": True, "epsilon_nn": True, "use_cuda": True, "use_coma": True,
        "action_selector": "multinomial", "epsilon_start": 1.0, "epsilon_finish": 0.05,
        "epsilon_time_length": 1000, "epsilon_decay": "exp",
        "lr": 5e-4,
        "n_critic_learner_reps":1,
        "target_critic_update_interval":800,
        "lr_agent":5e-4,
        "lr_critic":5e-4,
        "td_lambda":0.8,
        "gamma":0.99}

# conf.update({"multiagent_controller": "noisyindependent", "agent": "noisyagent", "agent_model": "DQN"})
conf.update({"multiagent_controller": "independent", "agent": "basic", "agent_model": "DQN"})

args = convert(conf)

runner_obj = NStepRunner(args=args)
print("The model from the Runner: ", runner_obj.multiagent_controller.agents[0].model)
# print("The epsilon after Runner construction: ", runner_obj.epsilon)

learner_obj = COMALearner(multiagent_controller=runner_obj.multiagent_controller, args=args)
learner_obj.create_models(runner_obj.transition_scheme)

batch = runner_obj.run()
print("The batch returned after run(): ", batch)
# print("The epsilon of the batch: ", batch.epsilon)

learner_obj.train(batch)

a = 5
