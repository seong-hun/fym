import numpy as np

from fym.core import BaseEnv, BaseSystem


class Env(BaseEnv):
    def __init__(self):
        super().__init__(
            {
                "main": BaseEnv({
                    "main_1": BaseSystem(np.zeros((3, 3))),
                    "main_2": BaseSystem(np.ones(4)),
                }),
                "sub": BaseSystem([2, 2, 2]),
            },
            dt=0.01,
            max_t=10
        )

    def step(self, action):
        self.update(action)

    def set_dot(self, time, action):
        self.systems_dict["main"].systems_dict["main_1"].dot = 0.1 * np.ones((3, 3))
        self.systems_dict["main"].systems_dict["main_2"].dot = np.zeros(4)
        self.systems_dict["sub"].dot = -0.1 * np.ones(3)


env = Env()
env.reset()
env.systems_dict["main"].state = np.array(
    [1, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2, -2])
env.systems_dict["main"].systems_dict["main_2"].state = np.array([0, 0, 0, 10])
