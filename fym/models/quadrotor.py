import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R

from fym.core import BaseEnv, BaseSystem


class Quadrotor(BaseEnv):
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
    name = "quadrotor"
    state_lower_bound = np.array(-np.inf * np.ones(18))
    state_upper_bound = np.array(np.inf * np.ones(18))
    control_lower_bound = np.array(-np.inf * np.ones(4))
    control_upper_bound = np.array(np.inf * np.ones(4))

    def __init__(self, pos, vel, dcm, omega):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.dcm = BaseSystem(dcm)
        self.omega = BaseSystem(omega)

    def deriv(self, pos, vel, dcm, omega, control):
        d, c = self.d, self.c
        F, M1, M2, M3 = np.array(
            [[1, 1, 1, 1],
             [0, -d, 0, d],
             [d, 0, -d, 0],
             [-c, c, -c, c]]
        ).dot(control)
        M = np.vstack((M1, M2, M3))

        m, g, J = self.m, self.g, self.J
        e3 = np.vstack((0, 0, 1))

        dpos = vel
        dvel = g*e3 - F*dcm.dot(e3)/m
        ddcm = dcm.dot(hat(omega))
        domega = np.linalg.inv(J).dot(M - np.cross(omega, J.dot(omega), axis=0))
        return dpos, dvel, ddcm, domega

    def set_dot(self, t, control):
        pos, vel, dcm, omega = self.observe_list()
        dots = self.deriv(pos, vel, dcm, omega, control)
        self.pos.dot, self.vel.dot, self.dcm.dot, self.omega.dot = dots


def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])


if __name__ == '__main__':
    x = np.zeros((3, 1))
    v = np.zeros((3, 1))
    dcm = R.from_euler('ZYX', [0, 0, 0]).as_dcm()
    omega = np.zeros((3, 1))

    system = Quadrotor(x, v, dcm, omega)
    system.set_dot(0, np.zeros((4, 1)))
