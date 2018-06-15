from itertools import combinations
from torch.autograd import Variable
import torch as th


def _n_agent_pair_samples(n_agents):
    return n_agents // 2

def _ordered_agent_pairings(n_agents):
    return sorted(combinations(list(range(n_agents)), 2))

def _excluded_pair_ids(n_agents, sampled_pair_ids):
    pairings = _ordered_agent_pairings(n_agents)
    tmp = [_i for _i, _pair in enumerate(pairings) if not any([{*pairings[_s]} & {*_pair} for _s in sampled_pair_ids ])]
    return tmp

def _n_agent_pairings(n_agents):
    return int((n_agents * (n_agents-1)) / 2)

def _joint_actions_2_action_pair(joint_action, n_actions,use_delegate_action=True):
    if use_delegate_action:
        mask = (joint_action == 0.0)
        joint_action[mask] = 1.0
        _action1 = th.floor((joint_action-1.0) / n_actions)
        _action2 = (joint_action-1.0) % n_actions
        _action1[mask] = float("nan")
        _action2[mask] = float("nan")
    else:
        _action1 = th.floor(joint_action / n_actions)
        _action2 = (joint_action) % n_actions
    return _action1, _action2

def _joint_actions_2_action_pair_aa(joint_action, n_actions, avail_actions1, avail_actions2, use_delegate_action=True):
    if use_delegate_action:
        mask = (joint_action == 0.0)
        joint_action[mask] = 1.0
        _action1 = th.floor((joint_action-1.0) / n_actions)
        _action2 = (joint_action-1.0) % n_actions
        _action1[mask] = float("nan")
        _action2[mask] = float("nan")
    else:
        _action1 = th.floor(joint_action / n_actions)
        _action2 = (joint_action) % n_actions

    aa_m1 = _action1 != _action1
    aa_m2 = _action2 != _action2
    _action1[aa_m1 ] = 0
    _action2[aa_m2] = 0

    aa1 = avail_actions1.data.gather(-1, ( _action1.long() ))
    aa2 = avail_actions2.data.gather(-1, ( _action2.long() ))
    _action1[aa1 == 0] = float("nan")
    _action1[aa2 == 0] = float("nan")
    _action1[aa_m1] = float("nan")
    _action2[aa_m2] = float("nan")
    return _action1, _action2

def _action_pair_2_joint_actions(action_pair, n_actions):
    return action_pair[0] * n_actions + action_pair[1]

def _pairing_id_2_agent_ids(pairing_id, n_agents):
    all_pairings = _ordered_agent_pairings(n_agents)
    return all_pairings[pairing_id]


def _pairing_id_2_agent_ids__tensor(pairing_id, n_agents, tformat):
    assert tformat in ["a*bs*t*v"], "invalid tensor input format"
    pairing_list = _ordered_agent_pairings(n_agents)
    ttype = th.cuda.LongTensor if pairing_id.is_cuda else th.LongTensor
    ids1 = ttype(pairing_list)[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(pairing_id.shape[0], pairing_id.shape[1], pairing_id.shape[2],1)
    ids2 = ttype(pairing_list)[:, 1].unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(pairing_id.shape[0], pairing_id.shape[1], pairing_id.shape[2],1)
    ret0 = ids1.gather(-1, pairing_id.long())
    ret1 = ids2.gather(-1, pairing_id.long())
    return ret0, ret1

def _agent_ids_2_pairing_id(agent_ids, n_agents):
    agent_ids = tuple(agent_ids)
    all_pairings = _ordered_agent_pairings(n_agents)
    assert agent_ids in all_pairings, "agent_ids is not of proper format!"
    return all_pairings.index(agent_ids)

# simple tests to establish correctness of encoding functions
if __name__ == "__main__":
    print(_action_pair_2_joint_actions(_joint_actions_2_action_pair(100, 3), 3))
    print(_action_pair_2_joint_actions(_joint_actions_2_action_pair(5, 3), 3))
