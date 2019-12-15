import gym
from gym import spaces
import numpy as np

from fym.core import BaseSystem


class Aircraft3Dof(BaseSystem):
    g = 9.80665
    rho = 1.2215
    m = 8.5
    S = 0.65
    b = 3.44
    CD0 = 0.033
    CD1 = 0.017
    name = 'aircraft'
    control_size = 2  # CL, phi
    state_lower_bound = [-np.inf, -np.inf, -np.inf, 3, -np.inf, -np.inf]
    state_upper_bound = [np.inf, np.inf, -0.01, np.inf, np.inf, np.inf]
    control_lower_bound = [-0.5, np.deg2rad(-70)]
    control_upper_bound = [1.5, np.deg2rad(70)]

    def __init__(self, initial_state, wind):
        super().__init__(self.name, initial_state, self.control_size)
        self.wind = wind
        self.term1 = self.rho*self.S/2/self.m

    def external(self, states, controls):
        state = states['aircraft']
        return dict(wind=self.wind.get(state))

    def deriv(self, state, t, control, external):
        CL, phi = control
        CD = self.CD0 + self.CD1*CL**2
        raw_control = CD, CL, phi
        return self._raw_deriv(state, t, raw_control, external)

    def _raw_deriv(self, state, t, control, external):
        x, y, z, V, gamma, psi = state
        CD, CL, phi = control
        (_, Wy, _), (_, (_, _, dWydz), _) = external['wind']

        dxdt = V*np.cos(gamma)*np.cos(psi)
        dydt = V*np.cos(gamma)*np.sin(psi) + Wy
        dzdt = - V*np.sin(gamma)

        dWydt = dWydz * dzdt

        dVdt = (-self.term1*V**2*CD - self.g*np.sin(gamma)
                - dWydt*np.cos(gamma)*np.sin(psi))
        dgammadt = (self.term1*V*CL*np.cos(phi) - self.g*np.cos(gamma)/V
                    + dWydt*np.sin(gamma)*np.sin(psi)/V)
        dpsidt = (self.term1*V/np.cos(gamma)*CL*np.sin(phi)
                  - dWydt*np.cos(psi)/V/np.cos(gamma))

        return np.array([dxdt, dydt, dzdt, dVdt, dgammadt, dpsidt])


class F16LinearLateral(BaseSystem):
    """
    Reference:
        B. L. Stevens et al. "Aircraft Control and Simulation", 3/e, 2016
        Example 5.3-1: LQR Design for F-16 Lateral Regulator
    """
    A = np.array([
        [-0.322, 0.064, 0.0364, -0.9917, 0.0003, 0.0008, 0],
        [0, 0, 1, 0.0037, 0, 0, 0],
        [-30.6492, 0, -3.6784, 0.6646, -0.7333, 0.1315, 0],
        [8.5396, 0, -0.0254, -0.4764, -0.0319, -0.062, 0],
        [0, 0, 0, 0, -20.2, 0, 0],
        [0, 0, 0, 0, 0, -20.2, 0],
        [0, 0, 0, 57.2958, 0, 0, -1]
    ])
    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [20.2, 0],
        [0, 20.2],
        [0, 0]
    ])
    C = np.array([
        [0, 0, 0, 57.2958, 0, 0, -1],
        [0, 0, 57.2958, 0, 0, 0, 0],
        [57.2958, 0, 0, 0, 0, 0, 0],
        [0, 57.2958, 0, 0, 0, 0, 0]
    ])

    def __init__(self, initial_state=[1, 0, 0, 0, 0, 0, 0]):
        super().__init__(initial_state)

    def deriv(self, x, u):
        return self.A.dot(x) + self.B.dot(u)
