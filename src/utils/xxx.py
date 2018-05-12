from itertools import combinations

def _n_agent_pair_samples(n_agents):
    return n_agents // 2

def _ordered_agent_pairings(n_agents):
    return sorted(combinations(list(range(n_agents)), 2))

def _n_agent_pairings(n_agents):
    return int((n_agents * (n_agents-1)) / 2)

def _joint_actions_2_action_pair(joint_action, n_actions):
    _action1 = joint_action // n_actions
    _action2 = joint_action % n_actions
    return _action1, _action2

def _action_pair_2_joint_actions(action_pair, n_actions):
    return action_pair[0] * n_actions + action_pair[1]

def _pairing_id_2_agent_ids(pairing_id, n_actions):
    _action1 = pairing_id // n_actions
    _action2 = pairing_id % n_actions
    return _action1, _action2

def _agent_ids_2_pairing_id(agent_ids, n_actions):
    return agent_ids[0] * n_actions + agent_ids[1]

# simple tests to establish correctness of encoding functions
if __name__ == "__main__":
    print(_action_pair_2_joint_actions(_joint_actions_2_action_pair(100, 3), 3))
    print(_action_pair_2_joint_actions(_joint_actions_2_action_pair(5, 3), 3))
