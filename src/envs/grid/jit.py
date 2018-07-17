from torch.utils.cpp_extension import load
import os
lltm_cpp = load(name="lltm_cpp", sources=[os.path.join(os.path.dirname(__file__), "gridenv.cpp")], verbose=True)
# help(lltm_cpp)

a = lltm_cpp.PredatorPreyEnv(5,5,5,5,5, True, 0, "toroidal")
#a.Test()