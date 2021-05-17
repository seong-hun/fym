import numpy as np

from fym.core import BaseEnv, BaseSystem
from fym.utils import rot


class Quadrotor(BaseEnv):
    """
    Prof. Taeyoung Lee's model for quadrotor UAV is used.
    - (https://www.math.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf)

    Description:
        - a DCM matrix is in the state vector.
        - an NED frame is used for the body fixed frame. Hence, `+z` direction
        is downward.

    Variables:
        dcm: SO(3)
            the rotation matrix from the body-fixed frame to the inertial frame
            dcm = C_{i/b}
    """
    J = np.diag([0.0820, 0.0845, 0.1377])  # kg * m^2
    m = 4.34  # kg
    d = 0.0315  # m
    c = 8.004e-4  # m
    g = 9.81  # m/s^2

    e3 = np.vstack((0, 0, 1))

    name = "quadrotor"

    def __init__(self, pos=np.zeros((3, 1)), vel=np.zeros((3, 1)),
                 dcm=np.eye(3), omega=np.zeros((3, 1))):
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

        m, g, J, e3 = self.m, self.g, self.J, self.e3

        dpos = vel
        dvel = g*e3 - F * dcm.dot(e3)/m
        ddcm = dcm.dot(hat(omega))
        domega = np.linalg.inv(J).dot(M - np.cross(omega, J.dot(omega), axis=0))
        return dpos, dvel, ddcm, domega

    def set_dot(self, t, control):
        pos, vel, dcm, omega = self.observe_list()
        dots = self.deriv(pos, vel, dcm, omega, control)
        self.pos.dot, self.vel.dot, self.dcm.dot, self.omega.dot = dots

    def angle2dcm(self, angle):
        """angle: psi, theta, phi in radian"""
        return rot.angle2dcm(*angle)

    def dcm2angle(self, dcm):
        """angle: psi, theta, phi in radian"""
        return rot.dcm2anglE(dcm)


def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])


class MultiRotor(BaseEnv):
    """
    Prof. Taeyoung Lee's model for quadrotor UAV is used.
    - (https://www.math.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf)

    Description:
        - an NED frame is used for the inertia and body fixed frame.
        Hence, `+z` direction is downward.
        - pos and vel are resolved in the inertial frame,
        whereas dcm and omega are resolved in the body frame

    Variables:
        dcm: SO(3)
            the rotation matrix from the body-fixed frame to the inertial frame
            dcm = C_{i/b}
    """
    g = 9.81  # m/s^2
    e3 = np.vstack((0, 0, 1))

    name = "multirotor"

    def __init__(self,
                 pos=np.zeros((3, 1)),
                 vel=np.zeros((3, 1)),
                 dcm=np.eye(3),
                 omega=np.zeros((3, 1)),
                 J=np.diag([0.0820, 0.0845, 0.1377]),
                 m=4.34,
                 config="Quadrotor",
                 ):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.dcm = BaseSystem(dcm)
        self.omega = BaseSystem(omega)

        self.J = J
        self.m = m

    def deriv(self, pos, vel, dcm, omega, F, M):
        m, g, J, e3 = self.m, self.g, self.J, self.e3

        dpos = vel
        dvel = g * e3 - F * dcm.dot(e3) / m
        ddcm = dcm.dot(hat(omega))
        domega = np.linalg.inv(J).dot(M - np.cross(omega, J.dot(omega), axis=0))
        return dpos, dvel, ddcm, domega

    def set_dot(self, t, F, M):
        pos, vel, dcm, omega = self.observe_list()
        dots = self.deriv(pos, vel, dcm, omega, F, M)
        self.pos.dot, self.vel.dot, self.dcm.dot, self.omega.dot = dots

    def angle2dcm(self, angle):
        """angle: psi, theta, phi in radian"""
        return rot.angle2dcm(*angle)

    def dcm2angle(self, dcm):
        """angle: psi, theta, phi in radian"""
        return rot.dcm2anglE(dcm)
