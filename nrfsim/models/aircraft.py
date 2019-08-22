import gym
from gym import spaces
import numpy as np

from nrfsim.core import BaseSystem


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
    state_lower_bound = [-np.inf, -np.inf, -np.inf, 3, -np.inf, -np.inf],
    state_upper_bound = [np.inf, np.inf, -0.01, np.inf, np.inf, np.inf],
    control_lower_bound = [0, -0.5, np.deg2rad(-70)],
    control_upper_bound = [1, 1.5, np.deg2rad(70)],

    def __init__(self, initial_state, wind):
        super().__init__(self.name, initial_state, self.control_size)
        self.wind = wind

    def external(self, states, controls):
        state = states['aircraft']
        return dict(wind=self.wind.get(state))

    def deriv(self, state, control, external, t):
        CL, phi = control
        CD = self.CD0 + self.CD1*CL**2
        raw_control = CD, CL, phi
        return self._raw_deriv(state, raw_control, external, t)

    def _raw_deriv(self, state, control, external, t):
        x, y, z, V, gamma, psi = state
        CD, CL, phi = control
        (_, Wy, _), (_, dWydt, _) = external['wind']

        term1 = self.rho*self.S/2/self.m

        dxdt = V*np.cos(gamma)*np.cos(psi)
        dydt = V*np.cos(gamma)*np.sin(psi) + Wy
        dzdt = - V*np.sin(gamma)

        dVdt = (-term1*V**2*CD - self.g*np.sin(gamma)
                - dWydt*np.cos(gamma)*np.sin(psi))
        dgammadt = (term1*V*CL*np.cos(phi) - self.g*np.cos(gamma)/V
                    + dWydt*np.sin(gamma)*np.sin(psi)/V)
        dpsidt = (term1*V/np.cos(gamma)*CL*np.sin(phi)
                  - dWydt*np.cos(psi)/V/np.cos(gamma))

        return np.array([dxdt, dydt, dzdt, dVdt, dgammadt, dpsidt])
