import torch as th
import numpy as np
from envs import REGISTRY as envs_REGISTRY
# import envs.starcraft1.SC1 as SC1

import sys
print(sys.path)

def test_connection():
    print("testing sc1...")
    kwargs = dict()
    sc1 = envs_REGISTRY["sc1"](kwargs)
    # sc1 = SC1(kwargs)
    pass

def main():
    test_connection()
    pass

if __name__ == "__main__":
    main()