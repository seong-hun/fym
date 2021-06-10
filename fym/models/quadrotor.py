import numpy as np

from fym.core import BaseEnv, BaseSystem
from fym.utils import rot


def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])


class Quadrotor(BaseEnv):
    """
    Prof. Taeyoung Lee's model for quadrotor UAV is used.
    - https://www.math.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf

    Description:
        - an NED frame is used for the inertia and body fixed frame.
        Hence, `+z` direction is downward.
        - ``pos`` and ``vel`` are resolved in the inertial frame,
        whereas ``R`` and ``omega`` are resolved in the body frame
        - ``fis`` is a vector of thrusts generated by the rotors.

    Variables:
        R: SO(3)
            The rotation matrix from the body-fixed frame to the inertial frame
            R = C_{i/b} = C_{b/i}^T
    """
    g = 9.81  # m/s^2
    e3 = np.vstack((0, 0, 1))
    J = np.diag([0.0820, 0.0845, 0.1377])
    m = 4.34  # Mass
    d = 0.315  # The distance from the center of mass to the center of each rotor
    ctf = 8.004e-4  # The torque coefficient. ``torque_i = (-1)^i ctf f_i``
    B = np.array(
        [[1, 1, 1, 1],
         [0, -d, 0, d],
         [d, 0, -d, 0],
         [-ctf, ctf, -ctf, ctf]]
    )
    Binv = np.linalg.pinv(B)

    name = "quadrotor"

    def __init__(self,
                 pos=np.zeros((3, 1)),
                 vel=np.zeros((3, 1)),
                 R=np.eye(3),
                 omega=np.zeros((3, 1)),
                 config="Quadrotor"):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.R = BaseSystem(R)
        self.omega = BaseSystem(omega)

    def deriv(self, pos, vel, R, omega, fis):
        m, g, J, e3 = self.m, self.g, self.J, self.e3

        f, *M = self.fis2fM(fis)
        M = np.vstack(M)

        dpos = vel
        dvel = g * e3 - f * R @ e3 / m
        dR = R @ hat(omega)
        domega = np.linalg.inv(J).dot(M - np.cross(omega, J.dot(omega), axis=0))
        return dpos, dvel, dR, domega

    def set_dot(self, t, fis):
        pos, vel, R, omega = self.observe_list()
        dots = self.deriv(pos, vel, R, omega, fis)
        self.pos.dot, self.vel.dot, self.R.dot, self.omega.dot = dots

    def fis2fM(self, fis):
        """Convert f_i's to force and moments
        Parameters:
            fis: (4, 1) array
        Return:
            f, M1, M2, M3: (4,) array of force and moments
        """
        return (self.B @ fis).ravel()

    def fM2fis(self, f, M1, M2, M3):
        """Convert force and moments to f_i's
        Parameters:
            f: scalar, the total thrust
            M1, M2, M3: scalars, the moments
        Return:
            fis: (4, 1) array of f_i's
        """
        return self.Binv @ np.vstack((f, M1, M2, M3))

    def angle2R(self, angle):
        """angle: phi, theta, psi in radian"""
        return rot.angle2dcm(*np.ravel(angle)[::-1]).T

    def R2angle(self, R):
        """angle: phi, theta, psi in radian"""
        return rot.dcm2angle(R.T)[::-1]
