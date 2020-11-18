from gym.envs.registration import register

for p in [-0.5, -1, -1.5, -2, -3, -4]:
    register(
        id='PredatorPrey7x7P{}-v0'.format(p),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': (7,7), 'n_agents': 4, 'n_preys': 2, 'penalty': p
        }
    )

    register(
        id='PredatorPrey5x5P{}-v0'.format(p),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': (5, 5), 'n_agents': 2, 'n_preys': 1, 'penalty': p
        }
    )