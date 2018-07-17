import os

from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert

MOD = None
class PredatorPreyEnv(MultiAgentEnv):
    global GridEnvModule

    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        from torch.utils.cpp_extension import load
        MOD = load(name="gridenv_cpp", sources=[os.path.join(os.path.dirname(__file__), "gridenv.cpp")], verbose=True)

        self.grid_shape_x, self.grid_shape_y = args.predator_prey_shape
        self.env = MOD.PredatorPreyEnv(5, self.grid_shape_x, self.grid_shape_y)
        try:
            self.env.Test()
        except Exception as e:
            print(e)
            pass
        pass

if __name__ == "__main__":
    kwargs = dict(env_args=dict(predator_prey_shape=(5,5)))
    a = PredatorPreyEnv(**kwargs)
    pass