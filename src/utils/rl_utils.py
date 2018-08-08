import torch as th

def build_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    bs = rewards.size(0)
    max_t = rewards.size(1)
    targets = rewards.new(target_qs.size()).zero_()
    running_target = rewards.new(bs, n_agents).zero_()
    terminated = terminated.float()
    for t in reversed(range(max_t)):
        if t == max_t - 1:
            running_target = terminated[:, t] * rewards[:, t]
        else:
            running_target = mask[:, t] * (
                terminated[:, t] * rewards[:, t]
                + (1 - terminated[:, t]) * (
                               td_lambda * (rewards[:, t] + gamma * running_target)
                               + (1 - td_lambda) * (rewards[:, t] + gamma * target_qs[:, t, :])
                                           ))
        targets[:, t, :] = running_target
    return targets