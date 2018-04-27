from components.scheme import Scheme

import numpy as np

def test1():
    """
    Test scheme join
    """
    env_state_size = 10
    env_episode_limit = 7
    buffer_size = 5
    batch_size = 3
    use_cuda = False
    n_agents = 3

    scheme_fn = lambda _agent_id: Scheme([dict(name="actions",
                                               rename="other_agents_actions_{}".format(_agent_id),
                                               select_agent_ids=range(0, n_agents),
                                               transforms=[("mask", dict(select_agent_ids=[_agent_id], fill=0.0))]),
                                         ]).agent_flatten()
    schemes = [scheme_fn(_i) for _i in range(n_agents)]

    # TODO: write scheme tests (if appropriate)

    pass

def test1b():
    """
    Does test1 work for buffer size == 1 and multiple overflow? (corner case)
    """
    pass

def main():
    test1()
    pass

if __name__ == "__main__":
    main()