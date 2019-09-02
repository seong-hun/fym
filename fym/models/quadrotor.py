import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R

from fym.core import BaseSystem


class Quadrotor(BaseSystem):
    """
    Prof. Taeyoung Lee's model for quadrotor UAV is used.
    - (https://www.math.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf)

    Description:
        - a DCM matrix is in the state vector.
        - an NED frame is used for the body fixed frame. Hence, `+z` direction
        is downward.

    Variables:
        R: SO(3)
            the rotation matrix from the body-fixed frame to the inertial frame
    """
    J = np.diag([0.0820, 0.0845, 0.1377])  # kg * m^2
    m = 4.34  # kg
    d = 0.0315  # m
    c = 8.004e-4  # m
    g = 9.81  # m/s^2
    control_size = 4  # f1, f2, f3, f4
    name = 'quadrotor'
    state_lower_bound = np.array(-np.inf * np.ones(18))
    state_upper_bound = np.array(np.inf * np.ones(18))
    control_lower_bound = np.array(-np.inf * np.ones(4))
    control_upper_bound = np.array(np.inf * np.ones(4))

    def __init__(self, initial_state: list):
        super().__init__(self.name, initial_state, self.control_size)
        self.state_index = np.cumsum([3, 3, 9, 3])

    def external(self, states, controls):
        state = states['quadrotor']
        return None

    def split(self, ss, index):
        *ss, _ = np.split(ss, index)
        return ss

    def deriv(self, state, t, control, external):
        d, c = self.d, self.c
        f, M1, M2, M3 = np.array(
            [[1, 1, 1, 1],
             [0, -d, 0, d],
             [d, 0, -d, 0],
             [-c, c, -c, c]]
        ).dot(control)
        M = np.array([M1, M2, M3])

        m, g, J = self.m, self.g, self.J
        e3 = np.array([0, 0, 1])

        x, v, R, Omega = self.split(state, self.state_index)
        R = R.reshape(3, 3)

        dx = v
        dv = g*e3 - f*R.dot(e3)/m
        dR = R.dot(hat(Omega))
        dOmega = np.linalg.inv(J).dot(M - np.cross(Omega, J.dot(Omega)))

        return np.hstack((dx, dv, dR.ravel(), dOmega))



def hat(v: list) -> np.ndarray:
    v1, v2, v3 = v
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])


if __name__ == '__main__':
    x0 = [0, 0, 0]
    v0 = [0, 0, 0]
    R0 = R.from_euler('ZYX', [0, 0, 0]).as_dcm()
    dOmega = [0, 0, 0]
    initial_state = np.hstack((x0, v0, R0.ravel(), dOmega))

    system = Quadrotor(initial_state)
    print(system.deriv(initial_state, 0, [0, 0, 0, 0], {}))
