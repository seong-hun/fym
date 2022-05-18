"""
refrence: M. Deng, A.Inoue, K. Sekiguchi, L. Jian,
"Two-wheeled mobile robot motion control in dynamic environments," 2009
"""
import numpy as np

from fym.core import BaseSystem


class TwoWheelsRobot3Dof(BaseSystem):
    L = 0.5
    r = 0.3
    Jb = 1
    m = 10
    name = "TwoWheelsRobot"

    def __init__(self, initial_state):
        super().__init__(initial_state)

    def external(self, states, controls):
        return 0

    def deriv(self, state, t, control, external):
        x, y, v, theta = state.ravel()
        T1, T2 = control

        dxdt = v * np.cos(theta)
        dydt = v * np.sin(theta)
        dvdt = (T1 / self.r + T2 / self.r) / self.m
        dthetadt = (-T1 / self.r + T2 / self.r) * self.L / 2

        return np.vstack([dxdt, dydt, dvdt, dthetadt])
