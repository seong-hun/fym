import numpy as np

from fym.core import BaseEnv, BaseSystem


env = BaseEnv(
    {
        "main": BaseEnv({
            "main_1": BaseSystem([0, 0, 0]),
            "main_2": BaseSystem([1, 1, 1]),
        }),
        "sub": BaseSystem([2, 2, 2]),
    },
    dt=0.01,
    max_t=10
)
env.reset()
env.systems_dict["main"].state = np.array([-1, -1, -1, -2, -2, -2])
env.systems_dict["main"].systems_dict["main_2"].state = np.array([0, 0, 0])
