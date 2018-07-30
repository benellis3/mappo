import numpy.testing as tst
import numpy as np
import torch as th
from components.transforms_old import _n_step_return
from components.episode_buffer_old import BatchEpisodeBuffer
from components.scheme import Scheme

def test1():
    """
    Test BatchEpisodeBuffer (2)
    """
    n_nan = 5
    R = [1,-1,1,5] + [float("nan")]*n_nan
    V = [2,1,-3,9] + [float("nan")]*n_nan
    truncated = True # only applicable at end of episode!
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

    ret = _n_step_return(values=V_tensor,
                         rewards=b["reward"][0],
                         terminated=b["terminated"][0],
                         truncated=b["truncated"][0],
                         gamma=gamma,
                         n=1,
                         horizon=b._n_t-1,
                         seq_lens=b.seq_lens)
    # print(ret)
    ret1 = ret[0,:,:,0]
    # ret2 = ret[1, :, :, 0]
    tst.assert_array_almost_equal(ret1[0,:], np.array([-0.1, -1.7, 13.1]+[float("nan")]*(1+n_nan)), 5)
    pass

def test2():
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

    ret = _n_step_return(values=V_tensor,
                         rewards=b["reward"][0],
                         terminated=b["terminated"][0],
                         truncated=b["truncated"][0],
                         gamma=gamma,
                         n=1,
                         horizon=b._n_t - 1,
                         seq_lens=b.seq_lens
                         )
    # print(ret)
    ret1 = ret[0,:,:,0]
    # ret2 = ret[1, :, :, 0]
    tst.assert_array_almost_equal(ret1[0,:], np.array([-0.1, -1.7, 5]+[float("nan")]*(1+n_nan)), 5)
    pass

def test3():
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

    b_pd = b.to_pd()

    ret = _n_step_return(values=V_tensor,
                         rewards=b["reward"][0],
                         terminated=b["terminated"][0],
                         truncated=b["truncated"][0],
                         gamma=gamma,
                         n=2,
                         horizon=b._n_t-1,
                         seq_lens=b.seq_lens)
    # print(ret)
    ret1 = ret[0,:,:,0]
    # ret2 = ret[1, :, :, 0]
    tst.assert_array_almost_equal(ret1[0,:], np.array([-2.53, 5.5, 5.0]+[float("nan")]*(1+n_nan)), 5)
    pass

def main():
    test1()
    test2()
    test3()
    pass

if __name__ == "__main__":
    main()