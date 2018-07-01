import models
models.basic
models.REGISTRY
#from models import REGISTRY

from components.scheme import Scheme
from components.episode_buffer import BatchEpisodeBuffer


import torch as th
import numpy as np

def test1():
    """
    Test the following functions:
    - set_col
    - get_col
    - get_cols
    """
    env_state_size = 10
    env_episode_limit = 7
    buffer_size = 5
    batch_size = 3
    use_cuda = False
    n_agents = 3

    transition_scheme = Scheme([ dict(name="state",
                                   shape=(env_state_size,),
                                   dtype=np.float32,
                                   missing=np.nan,
                                   size=env_state_size),
                                 dict(name="actions",
                                      shape=(1,),
                                      select_agent_ids=range(0, n_agents),
                                      dtype=np.int32,
                                      missing=-1, ),
                              ]).agent_flatten()


    h = BatchEpisodeBuffer(data_scheme=transition_scheme,
                                          n_bs=batch_size,
                                          n_t=env_episode_limit + 1,
                                          n_agents=n_agents,
                                          is_cuda=use_cuda,
                                          is_shared_mem=False)

    # test batch insertion
    range_upper = batch_size
    for i in range(range_upper):
        #h.flush()
        data = th.FloatTensor([[1*(i+1)],[2*(i+1)]]).unsqueeze(2).unsqueeze(3)
        h.set_col(col="actions", data=data, agent_ids=(1, 2), t=i, bs=i)
        h.set_col(col="actions", data=data, agent_ids=(2, 0), t=i, bs=0)
        data2 = data.repeat(1,2,1,1)
        a = data2[0]
        b = data2[1]
        h.set_col(col="actions", data=data2, agent_ids=(2,2), t=i, bs=[0,2])



    # Try writing outside of ranges
    try:
        h.set_col(col="actions", data=data, agent_ids=(2, 0), t=env_episode_limit, bs=0)
    except AssertionError:
        pass
    try:
        h.set_col(col="actions", data=data, agent_ids=(2, 0), t=0, bs=batch_size)
    except AssertionError:
        pass

    # Try writing non-existing column
    try:
        h.set_col(col="actionp", data=data, agent_ids=(2, 0), t=0, bs=0)
    except (AssertionError, KeyError) as e:
        pass

    # test get_col
    dbg_pd = h.to_pd()
    ret1 = h.get_col(col="actions__agent0", agent_ids=None, t=None, stack=True, bs=None)[0].numpy()
    a1 = ret1[0,:,:]
    a2 = ret1[1, :, :]
    a3 = ret1[2, :, :]
    try:
        ret2 = h.get_col(col="actions", agent_ids=[0,1], t=[0,1], stack=False, bs=[2,0])[0].numpy()
    except AssertionError:
        pass
    ret2 = h.get_col(col="actions", agent_ids=[0, 1], t=1, stack=True, bs=[2, 0])[0].numpy()
    np.testing.assert_array_equal(ret2.squeeze(), np.array([[float("nan"), 4.0],[float("nan"),float("nan")]])), "incorrect indexing result"

    # test get_cols
    ret = h.get_cols(cols=["actions__agent0", "actions__agent1"])

    pass

def test1b():
    """
    Does test1 work for buffer size == 1 and multiple overflow? (corner case)
    """
    """
    Test the following functions:
    - view
    """
    env_state_size = 10
    env_episode_limit = 7
    buffer_size = 5
    batch_size = 3
    use_cuda = False
    n_agents = 3

    transition_scheme = Scheme([ dict(name="state",
                                   shape=(env_state_size,),
                                   dtype=np.float32,
                                   missing=np.nan,
                                   size=env_state_size),
                                 dict(name="actions",
                                      shape=(1,),
                                      select_agent_ids=range(0, n_agents),
                                      dtype=np.int32,
                                      missing=-1, ),
                                 dict(name="agent_id",
                                      shape=(1,),
                                      select_agent_ids=range(0, n_agents),
                                      dtype=np.int32,
                                      missing=-1, )
                              ]).agent_flatten()


    h = BatchEpisodeBuffer(data_scheme=transition_scheme,
                                          n_bs=batch_size,
                                          n_t=env_episode_limit + 1,
                                          n_agents=n_agents,
                                          is_cuda=use_cuda,
                                          is_shared_mem=False)
    h.set_col(col="actions__agent0",
              data=(th.arange(env_episode_limit + 1) + 1).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1),
              bs=[1,2,0])
    a = h.to_pd()
    bs_ids = [0,2]
    t_id = 1
    agent_id = 0
    n_actions = 5
    view_scheme = Scheme([dict(name="actions",
                               rename="past_action",
                               select_agent_ids=[agent_id],
                               transforms=[("shift", dict(steps=1, fill=0))],
                                          # ("one_hot", dict(range=(0, n_actions-1)))],
                               switch=True),
                          dict(name="agent_id",
                               #transforms=[("one_hot", dict(range=(0, n_agents - 1)))],
                               select_agent_ids=[agent_id],
                               switch=True)
                          ]).agent_flatten()

    #dat = th.arange(env_episode_limit).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)
    h.set_col(col="actions__agent0", data=(th.arange(env_episode_limit+1)+1).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1))
    # test __setitem__
    h["actions__agent1"] = (th.arange(env_episode_limit+1)*2+2).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1), None
    h.set_col(col="actions__agent2", data=(th.arange(env_episode_limit+1)*3+3).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1))
    h_pd = h.to_pd()


    ret, _ = h.view(dict_of_schemes=dict(scheme1=view_scheme),
                    to_cuda=False,
                    to_variable=True,
                    bs_ids=bs_ids,
                    t_id=t_id)
    ret_pd = ret["scheme1"].to_pd()

    # test __getitem__
    test1 = h[0]
    test2, _ = h["actions__agent1"]

    pass

def test2():
    """
    Test episode-wide data propagation
    """
    """
    Test the following functions:
    - view
    """
    env_state_size = 10
    env_episode_limit = 7
    buffer_size = 5
    batch_size = 3
    use_cuda = False
    n_agents = 3

    transition_scheme = Scheme([ dict(name="state",
                                   shape=(env_state_size,),
                                   dtype=np.float32,
                                   missing=np.nan,
                                   size=env_state_size),
                                 dict(name="actions",
                                      shape=(1,),
                                      select_agent_ids=range(0, n_agents),
                                      dtype=np.int32,
                                      missing=-1, ),
                                 dict(name="agent_id",
                                      shape=(1,),
                                      select_agent_ids=range(0, n_agents),
                                      dtype=np.int32,
                                      missing=-1, ),
                                 dict(name="yeti",
                                      shape=(1,),
                                      scope="episode",
                                      dtype=np.int32,
                                      missing=-1, )
                              ]).agent_flatten()


    h = BatchEpisodeBuffer(data_scheme=transition_scheme,
                           n_bs=batch_size,
                           n_t=env_episode_limit + 1,
                           n_agents=n_agents,
                           is_cuda=use_cuda,
                           is_shared_mem=False)
    h.set_col(col="actions__agent0",
              data=(th.arange(env_episode_limit + 1) + 1).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1),
              bs=[1,2,0])
    a = h.to_pd()
    bs_ids = [0,2]
    t_id = 1
    agent_id = 0
    n_actions = 5
    view_scheme = Scheme([dict(name="actions",
                               rename="past_action",
                               select_agent_ids=[agent_id],
                               transforms=[("shift", dict(steps=1, fill=0))],
                                          # ("one_hot", dict(range=(0, n_actions-1)))],
                               switch=True),
                          dict(name="agent_id",
                               #transforms=[("one_hot", dict(range=(0, n_agents - 1)))],
                               select_agent_ids=[agent_id],
                               switch=True),
                          dict(name="yeti",
                               scope="episode")
                          ]).agent_flatten()

    #dat = th.arange(env_episode_limit).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)
    h.set_col(col="actions__agent0", data=(th.arange(env_episode_limit+1)+1).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1))
    # test __setitem__
    h["actions__agent1"] = (th.arange(env_episode_limit+1)*2+2).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1), None
    h.set_col(col="actions__agent2", data=(th.arange(env_episode_limit+1)*3+3).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1))
    h_pd = h.to_pd()


    ret, _ = h.view(dict_of_schemes=dict(scheme1=view_scheme),
                    to_cuda=False,
                    to_variable=True,
                    bs_ids=bs_ids,
                    t_id=t_id)
    ret_pd = ret["scheme1"].to_pd()

    # test __getitem__
    # test1 = h[0]
    # test2, _ = h["actions__agent1"]

    pass

def test2b():
    """
    Test episode-wide data propagation
    """
    """
    Test the following functions:
    - view
    """
    env_state_size = 10
    env_episode_limit = 7
    buffer_size = 5
    batch_size = 3
    use_cuda = False
    n_agents = 3

    transition_scheme = Scheme([ dict(name="state",
                                   shape=(env_state_size,),
                                   dtype=np.float32,
                                   missing=np.nan,
                                   size=env_state_size),
                                 dict(name="actions",
                                      shape=(1,),
                                      select_agent_ids=range(0, n_agents),
                                      dtype=np.int32,
                                      missing=-1, ),
                                 dict(name="agent_id",
                                      shape=(1,),
                                      select_agent_ids=range(0, n_agents),
                                      dtype=np.int32,
                                      missing=-1, ),
                                 dict(name="yeti",
                                      shape=(1,),
                                      scope="episode",
                                      dtype=np.int32,
                                      missing=-1, )
                              ]).agent_flatten()


    h = BatchEpisodeBuffer(data_scheme=transition_scheme,
                           n_bs=batch_size,
                           n_t=env_episode_limit + 1,
                           n_agents=n_agents,
                           is_cuda=use_cuda,
                           is_shared_mem=False)
    h.set_col(col="actions__agent0",
              data=(th.arange(env_episode_limit + 1) + 1).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1),
              bs=[1,2,0])
    a = h.to_pd()
    bs_ids = [0,2]
    t_id = 1
    agent_id = 0
    n_actions = 5
    view_scheme = Scheme([dict(name="actions",
                               rename="past_action",
                               select_agent_ids=[agent_id],
                               transforms=[("shift", dict(steps=1, fill=0))],
                                          # ("one_hot", dict(range=(0, n_actions-1)))],
                               switch=True),
                          dict(name="agent_id",
                               #transforms=[("one_hot", dict(range=(0, n_agents - 1)))],
                               select_agent_ids=[agent_id],
                               switch=True),
                          dict(name="yeti",
                               scope="episode")
                          ]).agent_flatten()

    #dat = th.arange(env_episode_limit).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)
    h.set_col(col="actions__agent0", data=(th.arange(env_episode_limit+1)+1).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1))
    # test __setitem__
    h["actions__agent1"] = (th.arange(env_episode_limit+1)*2+2).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1), None
    h.set_col(col="actions__agent2", data=(th.arange(env_episode_limit+1)*3+3).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1))
    h_pd = h.to_pd()


    ret, _ = h.view(dict_of_schemes=dict(scheme1=view_scheme),
                    to_cuda=False,
                    to_variable=True,
                    bs_ids=bs_ids,
                    t_id=t_id)
    ret_pd = ret["scheme1"].to_pd()

    pass


def main():
    # test1()
    # test1b()
    # test2()
    test2b()
    pass

if __name__ == "__main__":
    main()