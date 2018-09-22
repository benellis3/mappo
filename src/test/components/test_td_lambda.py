import numpy.testing as tst
import numpy as np
import torch as th
#from components.episode_buffer_old import BatchEpisodeBuffer
#from components.scheme import Scheme
from utils.rl_utils import build_td_lambda_targets__old as build_td_lambda_targets, build_td_lambda_targets as build_td_lambda_targets__wendelin

def G_t_n(R, V, t, n, gamma, truncated):
    """
    Sutton 2018
    eq. 12.1
    """
    Gtn = 0
    for i in range(1,n+1):
        Gtn += (gamma**(i-1)) * R[t+i]
    if truncated:
        Gtn += (gamma**n) * V[t+n]
    return Gtn

def G_t_n_lambda(G_t_1_to_t_h, t, h, td_lambda):
    """
    Sutton 2018
    eq. 12.9
    """
    Gtnlambda = 0
    for n in range(1, h-t-1 +1):
        Gtnlambda += ((td_lambda)**(n-1)) * G_t_1_to_t_h[t+n]
    Gtnlambda *= (1-td_lambda)
    Gtnlambda += (td_lambda**(h-t-1))*G_t_1_to_t_h[h]
    return Gtnlambda

def G_t_n_lambda_range(G_t_n_lambda_0, R, V, gamma, h, td_lambda, truncated):
    """
    uses efficient update rule for calculating consecutive G_t_n_lambda (see docs)
    """
    G_t_n_arr = [G_t_n_lambda_0]
    for t in range(0, h-1):
        delta_t = R[t+1] + gamma*(V[t+1] if not (t==h-1 and not truncated) else 0.0) - V[t]
        new_G = (V[t+1] if not (t==h-1 and not truncated) else 0.0) - (1 / (gamma*td_lambda))*(V[t] + delta_t - G_t_n_arr[-1])
        G_t_n_arr.append(new_G)
    return G_t_n_arr

def G_t_n_lambda_range_rev(R, V, gamma, h, td_lambda, truncated):
    """
    uses efficient update rule for calculating consecutive G_t_n_lambda in reverse (see docs)
    """
    G_hm1_h = R[h]
    if truncated:
        G_hm1_h += gamma*V[h]

    G_t_n_arr = [G_hm1_h]
    for t in range(h-1,0,-1):
        delta_tm1 = R[t] + gamma*V[t] - V[t-1]
        new_G = gamma*td_lambda*(G_t_n_arr[-1] - V[t]) + (V[t-1] + delta_tm1)
        G_t_n_arr.append(new_G)
    return list(reversed(G_t_n_arr))

def _align_right(tensor, h, lengths):
    for _i, _l in enumerate(lengths):
        if _l < h+1 and _l > 0:
            tensor[:,_i,-_l:,:] = tensor[:,_i,:_l,:]
            try:
                tensor[:,_i, :(h+1-_l), :] = float("nan") # not strictly necessary as will shift back anyway later...
            except:
                pass
    return tensor

def _align_left(tensor, h, lengths):
    for _i, _l in enumerate(lengths):
        if _l < h+1 and _l > 0 :
            tensor[:,_i,:_l,:] = tensor[:,_i,-_l:,:]
            tensor[:, _i, -(h+1-_l):, :] = float("nan") # not strictly necessary as will shift back anyway later...
    return tensor

def G_t_n_lambda_range_rev_batch(R_tensor, V_tensor, seq_lens, gamma, h, td_lambda, truncated_tensor):
    """
    uses efficient update rule for calculating consecutive G_t_n_lambda in reverse (see docs)
    """

    R_tensor = _align_right(R_tensor.clone(), h, seq_lens)
    V_tensor = _align_right(V_tensor.clone(), h, seq_lens)

    # a = R_tensor.numpy()[0,:,:,0]
    # b = V_tensor.numpy()[0,:,:,0]

    G_buffer = th.FloatTensor(*R_tensor.shape)*float("nan")
    G_buffer[:,:,h-1,:] = R_tensor[:,:,h,:]
    #TODO: Catch seq_len==1 tensors!
    G_buffer[:,:,h-1:,:] += gamma*V_tensor[:,:,h:,:] * truncated_tensor

    for t in range(h-1,0,-1):
        delta_tm1 = R_tensor[:,:,t,:] + gamma*V_tensor[:,:,t,:] - V_tensor[:,:,t-1,:]
        new_G = gamma*td_lambda*(G_buffer[:,:,t,:] - V_tensor[:,:,t,:]) + (V_tensor[:,:,t-1,:] + delta_tm1)
        G_buffer[:,:,t-1,:] = new_G

    # j = G_buffer.numpy()[0,:,:,0]
    G_buffer = _align_left(G_buffer, h, seq_lens)

    return G_buffer

def test1():
    """
    Test G_t_n (1)
    """

    R = [1,-1,1,5]
    V = [2,1,-3,9]
    truncated = True
    gamma = 0.9
    td_lambda = 0.5

    ret = G_t_n(R, V, t=1, n=2, gamma=gamma, truncated=truncated)
    tst.assert_almost_equal(ret, 12.79, 10)
    pass

def test2():
    """
    Test G_t_n (2)
    """
    R = [1,-1,1,5]
    V = [2,1,-3,9]
    truncated = False
    gamma = 0.9
    td_lambda = 0.5

    ret = G_t_n(R, V, t=1, n=2, gamma=gamma, truncated=truncated)
    tst.assert_almost_equal(ret, 5.5, 10)
    pass

def test3():
    """
    Test G_t_n_lambda (1)
    """
    R = [1,-1,1,5]
    V = [2,1,-3,9]
    truncated = False # only applicable at end of episode!
    gamma = 0.9
    td_lambda = 0.5

    t = 1
    h = len(R)-1

    Gts = {}
    for i in range(t+1, h+1):
        Gts[i] = G_t_n(R,V,t=t,n=(i-t), gamma=gamma, truncated = True if i < h else truncated)
        #, truncated = True if i < (h-1) else truncated)
    # Gts[h] =
    ret = G_t_n_lambda(Gts, t, h, td_lambda)
    tst.assert_almost_equal(ret, 1.9, 10)

def test4():
    """
    Test G_t_n_lambda (2)
    """
    R = [1,-1,1,5]
    V = [2,1,-3,9]
    truncated = True # only applicable at end of episode!
    gamma = 0.9
    td_lambda = 0.5

    t = 1
    h = len(R)-1

    Gts = {}
    for i in range(t+1, h+1):
        Gts[i] = G_t_n(R,V,t=t,n=(i-t), gamma=gamma, truncated = True if i < h else truncated)
    ret = G_t_n_lambda(Gts, t, h, td_lambda)
    tst.assert_almost_equal(ret, 5.545, 10)


def test5():
    """
    Test BatchEpisodeBuffer (1)
    """
    R = [-1,1,5,0] #[1,-1,1,5]
    V = [2,1,-3,9]
    truncated = False # False: Last state is terminal
    gamma = 0.9
    td_lambda = 0.5

    rewards = th.FloatTensor(R).unsqueeze(0).unsqueeze(2)
    target_qs = th.FloatTensor(V).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    terminated = rewards.clone().fill_(0)
    if not truncated:
        terminated[:,-2,:] = 1.0
    mask = rewards.clone().fill_(1.0)
    if not truncated:
        mask[:,-1,:] = 0.0

    ret = build_td_lambda_targets__wendelin(rewards, terminated, mask, target_qs, None, gamma, td_lambda)
    print("RET:", ret)
    if truncated:
        tst.assert_array_almost_equal(ret.squeeze().numpy(), np.array([1.94525, 5.545, 13.1]), 5)
    else:
        tst.assert_array_almost_equal(ret.squeeze().numpy(), np.array([0.305,1.9,5.0]), 5)

    ret = build_td_lambda_targets(rewards, terminated, mask, target_qs, 1, gamma, td_lambda)
    print("RET:", ret)
    if truncated:
        tst.assert_array_almost_equal(ret.squeeze().numpy(), np.array([1.94525, 5.545, 13.1]), 5)
    else:
        tst.assert_array_almost_equal(ret.squeeze().numpy(), np.array([0.305,1.9,5.0]), 5)
    pass

def test8():
    """
    Test G_t_n_lambda_range (1)
    """
    R = [1,-1,1,5]
    V = [2,1,-3,9]
    truncated = True # only applicable at end of episode!
    gamma = 0.9
    td_lambda = 0.5

    t = 0
    h = len(R) - 1 # horizon has to be set to the index of the last available reward!

    #Gts = {}
    #for i in range(t+1, h+1):
    #    Gts[i] = G_t_n(R,V,t=t,n=(i-t), gamma=gamma, truncated = True if i < h else truncated)
    #ret = G_t_n_lambda(Gts, t, h, td_lambda)
    Gts = {}
    for i in range(t+1, h+1):
        Gts[i] = G_t_n(R,V,t=t,n=(i-t), gamma=gamma, truncated = True if i < h else truncated)
    tst.assert_array_almost_equal(np.array([Gts[i] for i in range(t+1, h+1)]), np.array([-0.1, -2.53, 10.511]), 5)
    G_t_n_lambda_0 = G_t_n_lambda(Gts, t, h, td_lambda)
    ret = G_t_n_lambda_range(G_t_n_lambda_0, R, V, gamma, h, td_lambda, truncated)
    tst.assert_array_almost_equal(np.array(ret), np.array([1.94525, 5.545, 13.1]), 10)

def test9():
    """
    Test G_t_n_lambda_range (2)
    """
    R = [1,-1,1,5]
    V = [2,1,-3,9]
    truncated = False # only applicable at end of episode!
    gamma = 0.9
    td_lambda = 0.5

    t = 0
    h = len(R) - 1 # horizon has to be set to the index of the last available reward!

    Gts = {}
    for i in range(t+1, h+1):
        Gts[i] = G_t_n(R,V,t=t,n=(i-t), gamma=gamma, truncated = True if i < h else truncated)
    tst.assert_array_almost_equal(np.array([Gts[i] for i in range(t+1, h+1)]), np.array([-0.1, -2.53, 3.95]), 5)
    G_t_n_lambda_0 = G_t_n_lambda(Gts, t, h, td_lambda)
    ret = G_t_n_lambda_range(G_t_n_lambda_0, R, V, gamma, h, td_lambda, truncated)
    tst.assert_array_almost_equal(np.array(ret), np.array([0.305,1.9,5.0]), 10)

def test10():
    """
    Test G_t_n_lambda_range_rev (1)
    """
    R = [1,-1,1,5]
    V = [2,1,-3,9]
    truncated = True # only applicable at end of episode!
    gamma = 0.9
    td_lambda = 0.5

    t = 0
    h = len(R) - 1 # horizon has to be set to the index of the last available reward!
    ret = G_t_n_lambda_range_rev(R, V, gamma, h, td_lambda, truncated)
    tst.assert_array_almost_equal(np.array(ret), np.array([1.94525, 5.545, 13.1]), 10)

def test11():
    """
    Test G_t_n_lambda_range_rev (2)
    """
    R = [1,-1,1,5]
    V = [2,1,-3,9]
    truncated = False # only applicable at end of episode!
    gamma = 0.9
    td_lambda = 0.5

    h = len(R) - 1 # horizon has to be set to the index of the last available reward!
    ret = G_t_n_lambda_range_rev(R, V, gamma, h, td_lambda, truncated)
    tst.assert_array_almost_equal(np.array(ret), np.array([0.305,1.9,5.0]), 10)

def test12():
    """
    Test G_t_n_lambda_range_rev_batch
    """
    R = [[1, -1, 1, 5],
         [1, np.nan, np.nan, np.nan],
         [1, -1, np.nan, np.nan],
         [1, -1, 1, np.nan],
         [1, -1, 1, 5]]

    seq_lens = [4,1,2,3,4]

    V = [[2, 1, -3, 9],
         [2, 1, -3, 9],
         [2, 1, -3, 9],
         [2, 1, -3, 9],
         [2, 1, -3, 9]]

    truncated = [False, False, False, True, True] # only applicable at end of episode!
    truncated_tensor = th.LongTensor(truncated).unsqueeze(0).unsqueeze(2).unsqueeze(3).float()

    gamma = 0.9
    td_lambda = 0.5

    R_tensor = th.FloatTensor(R).unsqueeze(0).unsqueeze(3)
    V_tensor = th.FloatTensor(V).unsqueeze(0).unsqueeze(3)

    h = R_tensor.shape[2] - 1
    ret = G_t_n_lambda_range_rev_batch(R_tensor, V_tensor, seq_lens, gamma, h, td_lambda, truncated_tensor)
    ret = ret.numpy()[0,:,:,0]
    a = ret[0, :]
    b = ret[4, :]
    c = R_tensor.numpy()[0,:,:,0]
    tst.assert_array_almost_equal(ret[0,:], np.array([0.305,1.9,5.0, np.nan]), 5)
    tst.assert_array_almost_equal(ret[4,:], np.array([1.94525, 5.545, 13.1, np.nan]), 5)

def test12_new():
    """
    Test G_t_n_lambda_range_rev_batch
    """
    rewards = [[-1, 1, 5, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [-1, 0.0, 0.0, 0.0],
         [-1, 1, 0.0, 0.0],
         [-1, 1, 5, 0.0]]

    seq_lens = [4,1,2,3,4]
    gamma = 0.9
    td_lambda = 0.5

    target_qs = [[2, 1, -3, 9],
         [2, 1, -3, 9],
         [2, 1, -3, 9],
         [2, 1, -3, 9],
         [2, 1, -3, 9]]

    rewards = th.FloatTensor(rewards).unsqueeze(2)
    target_qs = th.FloatTensor(target_qs).unsqueeze(2) #.unsqueeze(3)

    truncated = [False, False, False, True, True] # only applicable at end of episode!
    terminated = rewards.clone().fill_(0)
    for b in range(rewards.shape[0]):
        truncated_b = truncated[b]
        if not truncated_b:
            terminated[b,-2,:] = 1.0

    mask = rewards.clone().fill_(1.0)
    for b in range(rewards.shape[0]):
        truncated_b = truncated[b]
        if not truncated_b:
            mask[b,-1,:] = 0.0

    ret = build_td_lambda_targets__wendelin(rewards, terminated, mask, target_qs, None, gamma, td_lambda)
    tst.assert_array_almost_equal(ret[0].squeeze(), np.array([0.305,1.9,5.0]), 5)
    tst.assert_array_almost_equal(ret[4].squeeze(), np.array([1.94525, 5.545, 13.1]), 5)

    ret = build_td_lambda_targets(rewards, terminated, mask, target_qs, 1, gamma, td_lambda)
    tst.assert_array_almost_equal(ret[0].squeeze(), np.array([0.305,1.9,5.0, 0.0]), 5)
    tst.assert_array_almost_equal(ret[4].squeeze(), np.array([1.94525, 5.545, 13.1, 0.0]), 5)
    pass

def test13():
    """
    Test BatchEpisodeBuffer (2)
    """
    R = [1,-1,1,5]
    V = [2,1,-3,9]
    truncated = False # only applicable at end of episode!
    gamma = 0.9
    td_lambda = 0.5
    n_agents=4
    n_bs=4

    rewards = th.FloatTensor(R).unsqueeze(0).unsqueeze(2)
    target_qs = th.FloatTensor(V).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    terminated = rewards.clone().fill_(0)
    if not truncated:
        terminated[:,-2,:] = 1.0
    mask = rewards.clone().fill_(1.0)
    if not truncated:
        mask[:,-1,:] = 0.0
    ret = build_td_lambda_targets__wendelin(rewards, terminated, mask, target_qs, None, gamma, td_lambda)

    print("RET:", ret)
    if truncated:
        #tst.assert_array_almost_equal(ret.squeeze().numpy(), np.array([1.94525, 5.545, 13.1]), 5)
        assert False
    else:
        tst.assert_array_almost_equal(ret[:,:,0].squeeze().numpy(), np.array([0.305, 1.9, 5.0, np.nan]), 5)


def test14():
    """
    Test BatchEpisodeBuffer (2)
    """
    n_nan = 5
    R = [1,-1,1,5] + [float("nan")]*n_nan
    V = [2,1,-3,9] + [float("nan")]*n_nan
    truncated = False # only applicable at end of episode!
    gamma = 0.9
    td_lambda = 0.5
    n_agents=1
    n_bs=1

    R_tensor = th.FloatTensor(R).unsqueeze(0).unsqueeze(2)
    V_tensor = th.FloatTensor(V).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(n_agents,n_bs,1,1)
    terminated_tensor = th.FloatTensor([0.0]*(len(R)-1-n_nan) + [1.0] + [float("nan")]*n_nan).unsqueeze(0).unsqueeze(2)#.repeat(2,1,1,1)
    truncated_tensor = th.FloatTensor([0.0]*(len(R)-1-n_nan) + [1.0 if truncated else 0.0] + [float("nan")]*n_nan).unsqueeze(0).unsqueeze(2)#.repeat(2,1,1,1)

    scheme = Scheme([dict(name="reward",
                          shape=(1,),
                          dtype=np.float32,
                          missing=np.nan,),
                     dict(name="terminated",
                          shape=(1,),
                          dtype=np.float32,
                          missing=np.nan,),
                     dict(name="truncated",
                          shape=(1,),
                          dtype=np.float32,
                          missing=np.nan,)
                     ])

    b = BatchEpisodeBuffer(data_scheme=scheme,
                               n_bs=n_bs,
                               n_t=len(R),
                               n_agents=n_agents,
                               is_cuda=False,
                               is_shared_mem=False)

    for hist_id in range(n_bs):
        b.set_col(col="reward", data=R_tensor[:,:4,:], bs=hist_id, t=slice(0,4))
        b.set_col(col="terminated", data=terminated_tensor[:,:4,:], bs=hist_id, t=slice(0,4))
        b.set_col(col="truncated", data=truncated_tensor[:,:4,:], bs=hist_id, t=slice(0,4))
    #b.seq_lens = [_s-n_nan for _s in b.seq_lens ]
    b_pd = b.to_pd()

    ret, _ = b.get_stat("td_lambda_targets",
                        bs_ids=None,
                        td_lambda=td_lambda,
                        gamma=gamma,
                        value_function_values=V_tensor,
                        to_variable=False,
                        to_cuda=False)
    # print(ret)
    ret1 = ret[0,:,:,0]
    # ret2 = ret[1, :, :, 0]
    tst.assert_array_almost_equal(ret1[0,:], np.array([0.305,1.9,5.0]+[float("nan")]*(1+n_nan)), 5)

def test15():
    """
    Test BatchEpisodeBuffer (2)
    """
    n_nan = 5
    R = [1,-1,1,5] + [float("nan")]*n_nan
    V = [2,1,-3,9] + [float("nan")]*n_nan
    truncated = False # only applicable at end of episode!
    gamma = 0.9
    td_lambda = 0.5
    n_agents=3
    n_bs=1

    R_tensor = th.FloatTensor(R).unsqueeze(0).unsqueeze(2)
    V_tensor = th.FloatTensor(V).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(1,n_bs,1,1)
    terminated_tensor = th.FloatTensor([0.0]*(len(R)-1-n_nan) + [1.0] + [float("nan")]*n_nan).unsqueeze(0).unsqueeze(2)#.repeat(2,1,1,1)
    truncated_tensor = th.FloatTensor([0.0]*(len(R)-1-n_nan) + [1.0 if truncated else 0.0] + [float("nan")]*n_nan).unsqueeze(0).unsqueeze(2)#.repeat(2,1,1,1)

    scheme = Scheme([dict(name="reward",
                          shape=(1,),
                          dtype=np.float32,
                          missing=np.nan,),
                     dict(name="terminated",
                          shape=(1,),
                          dtype=np.float32,
                          missing=np.nan,),
                     dict(name="truncated",
                          shape=(1,),
                          dtype=np.float32,
                          missing=np.nan,)
                     ])

    b = BatchEpisodeBuffer(data_scheme=scheme,
                               n_bs=n_bs,
                               n_t=len(R),
                               n_agents=n_agents,
                               is_cuda=False,
                               is_shared_mem=False)

    for hist_id in range(n_bs):
        b.set_col(col="reward", data=R_tensor[:,:4,:], bs=hist_id, t=slice(0,4))
        b.set_col(col="terminated", data=terminated_tensor[:,:4,:], bs=hist_id, t=slice(0,4))
        b.set_col(col="truncated", data=truncated_tensor[:,:4,:], bs=hist_id, t=slice(0,4))
    #b.seq_lens = [_s-n_nan for _s in b.seq_lens ]
    b_pd = b.to_pd()

    ret, _ = b.get_stat("td_lambda_targets",
                        bs_ids=None,
                        td_lambda=td_lambda,
                        gamma=gamma,
                        value_function_values=V_tensor,
                        to_variable=False,
                        to_cuda=False,
                        n_agents=1)
    # print(ret)
    ret1 = ret[0,:,:,0]
    # ret2 = ret[1, :, :, 0]
    tst.assert_array_almost_equal(ret1[0,:], np.array([0.305,1.9,5.0]+[float("nan")]*(1+n_nan)), 5)

def main():
    # For td lambda based on wendelin's refined implementation, run the following tests:
    # test5()
    # test12_new()
    test2()
    test3()
    test4()
    #test5()
    # test6()
    # test8()
    # test9()
    # test10()
    # test11()
    # test12()
    # test13()
    # test14()
    # test15()
    test12_new()

if __name__ == "__main__":
    main()