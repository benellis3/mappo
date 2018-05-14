class MultiAgentEnv(object):

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def get_stats(self):
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError












def get_actions(epsilon, epsilon_dec, common_knowledge, observations, independent, test, jal):
    if test:
        epsilon = 0
    if jal:
        epsilon_dec = 0

    actions = []
    q_dec = []

    # joint-action controller:
    if common_knowledge == 0:
        jac_state = 0
    else:
        jac_state = observations[0] + 1
    if np.random.random() < epsilon:
        if np.random.random() < epsilon_dec:
            # force decentralisation action
            a_jac = torch.LongTensor(1).zero_()
        else:
            if jal:
                a_jac = torch.LongTensor(1).random_(0, n_actions ** 2)
                a_jac += 1
            else:
                a_jac = torch.LongTensor(1).random_(0, n_actions ** 2 + 1)
    else:
        if jal:
            a_jac = torch.max(jac_q_vals[jac_state, 1:], dim=0)[1].data
            a_jac += 1
        else:
            a_jac = torch.max(jac_q_vals[jac_state, :], dim=0)[1].data

    q_jac = jac_q_vals[jac_state, a_jac[0]]

    if (((a_jac != 0).all() and (not independent))) or jal:
        a0 = (a_jac - 1) / n_actions
        a1 = (a_jac - 1) % n_actions
        return [a0, a1], q_jac, None

    # chosen to decentralise
    for i in range(n_agents):
        dec_state = observations[i]
        q_temp = dec_q_vals[i][dec_state]

        if np.random.random() < epsilon:
            a = torch.LongTensor(1).random_(0, n_actions)
        else:
            a = torch.max(q_temp, 0)[1]
        actions.append(a)
        q_dec.append(q_temp[actions[-1]])
    return actions, q_jac, q_dec


# def get_joint_action(epsilon, central_state ):
#   if np.random.random() < epsilon:
#     a = torch.LongTensor(1).random_(0, n_actions**2)
#   else:
#     a = torch.max(q_joint[central_state], 0)[1]

#   actions = [a/n_actions, a%n_actions]
#   val = q_joint[central_state][a]
#   return actions, [val]

def get_reward(actions, matrix_id):
    return payoff_values[matrix_id][actions]
