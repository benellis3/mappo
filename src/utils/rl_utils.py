import torch as th

def build_targets(rewards, terminated, mask, target_max_qs, n_agents, gamma, td_lambda):
    bs = rewards.size(0)
    max_t = rewards.size(1)
    targets = rewards.new(target_max_qs.size()).zero_()
    running_target = rewards.new(bs, n_agents).zero_()
    for t in reversed(range(max_t)):
        for b in range(bs):
            if terminated.data[b, t, 0] == 1:
                running_target[b, :] = rewards.data[b, t, 0]
            elif mask.data[b, t, 0] == 0:
                running_target[b, :] = 0
            elif t == max_t - 1:
                running_target[b, :] = rewards.data[b, t, 0] + gamma * target_max_qs.data[b, t, :]
            # elif mask.data[b, t, 0] == 1 and mask.data[b, t + 1, 0] == 0:
            #     running_target[b, :] = rewards.data[b, t, 0] + gamma * target_max_qs.data[b, t, :]
            else:
                running_target[b, :] = (
                (td_lambda) * (rewards.data[b, t, 0] + gamma * running_target[b, :])
                + (1 - td_lambda) * (rewards.data[b, t, 0] + gamma * target_max_qs.data[b, t, :]))
            targets[b, t, :] = running_target[b, :]
    return targets