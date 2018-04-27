from components.replay_buffer import ContiguousReplayBuffer
from components.scheme import Scheme
from components.episode_buffer import BatchEpisodeBuffer

import numpy as np

def test1():
    """
    Does Buffer insert elements at right positions?
    In particular, is overflow handled correctly?
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
                              ]).agent_flatten()
    buffer = ContiguousReplayBuffer(data_scheme=transition_scheme,
                                   n_bs=buffer_size,
                                   n_t=env_episode_limit+1,
                                   n_agents=n_agents,
                                   batch_size=batch_size,
                                   is_cuda=use_cuda,
                                   is_shared_mem=use_cuda)

    # test batch insertion
    n_bs = 3
    range_upper = 51
    for i in range(range_upper):
        h = BatchEpisodeBuffer(data_scheme=transition_scheme,
                                   n_bs=n_bs,
                                   n_t=env_episode_limit+1,
                                   n_agents=n_agents,
                                   is_cuda=use_cuda,
                                   is_shared_mem=False)
        h.seq_lens = [i] * n_bs
        h.data._transition[:,:,:] = i
        buffer.put(h)

    assert buffer.buffer.data._transition[ (n_bs * range_upper ) % buffer_size-1, 0, 0] == range_upper-1, "faulty buffer value."
    assert buffer.buffer.seq_lens[(n_bs * range_upper) % buffer_size-1] == range_upper-1, "faulty seq lens."
    pass

def test1b():
    """
    Does test1 work for buffer size == 1 and multiple overflow? (corner case)
    """
    env_state_size = 10
    env_episode_limit = 7
    buffer_size = 1
    batch_size = 3
    use_cuda = False
    n_agents = 3

    transition_scheme = Scheme([ dict(name="state",
                                   shape=(env_state_size,),
                                   dtype=np.float32,
                                   missing=np.nan,
                                   size=env_state_size),
                              ]).agent_flatten()
    buffer = ContiguousReplayBuffer(data_scheme=transition_scheme,
                                   n_bs=buffer_size,
                                   n_t=env_episode_limit+1,
                                   n_agents=n_agents,
                                   batch_size=batch_size,
                                   is_cuda=use_cuda,
                                   is_shared_mem=use_cuda)

    # test batch insertion
    n_bs = 3
    range_upper = 51
    for i in range(range_upper):
        h = BatchEpisodeBuffer(data_scheme=transition_scheme,
                                   n_bs=n_bs,
                                   n_t=env_episode_limit+1,
                                   n_agents=n_agents,
                                   is_cuda=use_cuda,
                                   is_shared_mem=False)
        h.seq_lens = [i] * n_bs
        h.data._transition[:,:,:] = i
        buffer.put(h)

    assert buffer.buffer.data._transition[ (n_bs * range_upper ) % buffer_size-1, 0, 0] == range_upper-1, "faulty buffer value."
    assert buffer.buffer.seq_lens[(n_bs * range_upper) % buffer_size-1] == range_upper-1, "faulty seq lens."
    pass

def test2():
    """
    Does Buffer sample correctly?
    In particular:
        - is sampling only happening from filled area
        - is sampling distribution correct
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
                              ]).agent_flatten()
    buffer = ContiguousReplayBuffer(data_scheme=transition_scheme,
                                   n_bs=buffer_size,
                                   n_t=env_episode_limit+1,
                                   n_agents=n_agents,
                                   batch_size=batch_size,
                                   is_cuda=use_cuda,
                                   is_shared_mem=use_cuda)

    # test batch insertion
    n_bs = 1
    range_upper = 3
    for i in range(range_upper):
        h = BatchEpisodeBuffer(data_scheme=transition_scheme,
                                   n_bs=n_bs,
                                   n_t=env_episode_limit+1,
                                   n_agents=n_agents,
                                   is_cuda=use_cuda,
                                   is_shared_mem=False)
        h.seq_lens = [i] * n_bs
        h.data._transition[:,:,:] = i
        buffer.put(h)

    # buffer is now filled, now test whether sampling follows correct distribution and does not occur outside of fill area
    dic = {}
    upper = 10000
    for i in range(upper):
        s = buffer.sample(2)
        for j in range(len(s)):
            if s.data._transition[j,0,0] not in dic:
                dic[s.data._transition[j,0,0]] = 0
            dic[s.data._transition[j,0,0]] += 1

    for _k, _v in dic.items():
        pred = (1 + _k) / (len(buffer) + sum(buffer.buffer.seq_lens))*upper*2
        actual = dic[_k]
        np.testing.assert_almost_equal(pred/actual, 1.0, 2)
    pass

def main():
    test1()
    test1b()
    test2()
    pass

if __name__ == "__main__":
    main()