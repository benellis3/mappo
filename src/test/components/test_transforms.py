import numpy as np
import ujson as json

from components.episode_buffer import BatchEpisodeBuffer
from components.scheme import Scheme
from components.transforms import _join_dicts, _build_model_inputs

def test1():
    """
    test _join_dicts
    """
    a = {"a":6, "b":{18:"z", "9":[1,2]}}
    b = {"c":"y"}
    target = {"a":6, "b":{18:"z", "9":[1,2]}, "c":"y"}
    ret = _join_dicts(a,b)
    assert json.dumps(ret) == json.dumps(target), "_join_dicts did not join a,b correctly!"

    c = {"b":"y"}
    try:
        ret = _join_dicts(a, c)
    except AssertionError:
        pass
    pass

def test2():
    """
    Test _to_batch and _from_batch
    """
    import torch as th
    from components.transforms import _to_batch, _from_batch
    tformat="a*bs*t*v"
    a = th.randn(4,32,20,10)
    x, params, tformat = _to_batch(a, tformat)
    y = _from_batch(x, params, tformat)

    ret = th.sum(a - y)
    assert sum(a - y), "batchification methods are broken!"
    pass

def test3():
    """
    test whether _build_model_inputs handles episode-wide data correctly
    """

    env_state_size = 10
    env_episode_limit = 7
    buffer_size = 5
    batch_size = 3
    use_cuda = False
    n_agents = 3
    agent_id = 0
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
                                      missing=float("nan"), ),
                                 # dict(name="yeti_expand",
                                 #      shape=(1,),
                                 #      scope="episode",
                                 #      dtype=np.int32,
                                 #      missing=float("nan"), )
                              ]).agent_flatten()


    inputs_beb = BatchEpisodeBuffer(data_scheme=transition_scheme,
                                    n_bs=batch_size,
                                    n_t=env_episode_limit + 1,
                                    n_agents=n_agents,
                                    is_cuda=use_cuda,
                                    is_shared_mem=False)
    #inputs_beb.set_col(col="yeti", scope="episode") = th.FloatTensor()


    view_scheme = Scheme([dict(name="actions",
                               rename="past_action",
                               select_agent_ids=[agent_id],
                               transforms=[("shift", dict(steps=1, fill=0))],
                               switch=True),
                          dict(name="agent_id",
                               select_agent_ids=[agent_id],
                               switch=True),
                          dict(name="yeti",
                               scope="episode"),
                          # dict(name="yeti_expand",
                          #      scope="episode",
                          #      transforms=[("t_repeat", dict(t_dim=env_episode_limit + 1))])
                          ]).agent_flatten()


    inputs, inputs_tformat = inputs_beb.view(dict_of_schemes=dict(input1_columns=view_scheme),
                                             to_cuda=False,
                                             to_variable=True,
                                             bs_ids=None,
                                             t_id=None)

    input_columns = {}
    input_columns["input1_columns"] = {}
    input_columns["input1_columns"]["yeti_input"] = Scheme([dict(name="yeti",
                                                                 scope="episode")])
    input_columns["input1_columns"]["noyeti"] = Scheme([dict(name="past_action",
                                                              select_agent_ids=[0])]).agent_flatten()
    # input_columns["input1_columns"]["yeti_nonyeti"] = Scheme([dict(name="yeti_expand",
    #                                                              scope="episode"),
    #                                                           dict(name="past_action",
    #                                                                select_agent_ids=[0])
    #                                                           ]).agent_flatten()
    model_inputs, model_inputs_tformat = _build_model_inputs(column_dict=input_columns,
                                                             inputs=inputs,
                                                             inputs_tformat=inputs_tformat,
                                                             to_variable=True)
    pass

def main():
    #test1()
    #test2()
    test3()
    pass

if __name__ == "__main__":
    main()