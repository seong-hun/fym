import gym
from gym import spaces
import numpy as np

from nrfsim.core import BaseSystem


class MissilePlanar(BaseSystem):
    g = 9.80665
    rho = 1.2215
    m = 8.5 # probably the values of m, S, etc. are not appropriate for missiles. 
    S = 0.65
    b = 3.44
    CD0 = 0.033
    CD1 = 0.017
    name = 'missile'
    control_size = 1  # latax. 
    state_lower_bound = [-np.inf, -np.inf, -np.inf, -np.inf],
    state_upper_bound = [np.inf, np.inf, np.inf, np.inf],
    control_lower_bound = [-10*g],
    control_upper_bound = [10*g],

    def __init__(self, initial_state, wind):
        super().__init__(self.name, initial_state, self.control_size)
        self.wind = wind

    def external(self, states, controls):
        state = states['missile']
        return {"wind" : [(0, 0), (0, 0)]} # no external effects

    def deriv(self, state, t, control, external):
        a = control
        CD = self.CD0 + self.CD1*a**2
        raw_control = CD, a
        return self._raw_deriv(state, t, raw_control, external)

    def _raw_deriv(self, state, t, control, external):
        import ipdb; ipdb.set_trace()
        x, y, V, gamma, = state
        CD, a = control
        (_, Wy), (_, dWydt) = external['wind']

        term1 = self.rho*self.S/2/self.m

        dxdt = V*np.cos(gamma)
        dydt = V*np.sin(gamma) + Wy

        dVdt = (-term1*V**2*CD - self.g*np.sin(gamma)
                - dWydt*np.cos(gamma))
        dgammadt = (a - self.g*np.cos(gamma)
                    + dWydt*np.sin(gamma))/V

        return np.array([dxdt, dydt, dVdt, dgammadt])
