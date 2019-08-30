import gym
from gym import spaces
import numpy as np

from fym.core import BaseSystem


class MissilePlanar(BaseSystem):
    R = 288
    g = 9.80665
    S = 1
    t1 = 1.5
    t2 = 8.5
    name = 'missile'
    control_size = 1  # latax. 
    state_lower_bound = [-np.inf, -np.inf, -np.inf, -np.inf],
    state_upper_bound = [np.inf, np.inf, np.inf, np.inf],
    control_lower_bound = [-10*g],
    control_upper_bound = [10*g],

    def __init__(self, initial_state):
        super().__init__(self.name, initial_state, self.control_size)

    def external(self, states, controls):
        state = states['missile']
        return 0
        # return {"wind" : [(0, 0), (0, 0)]} # no external effects

    def deriv(self, state, t, control, external):
        # state and (control) input
        x, y, V, gamma, = state
        a = control
        # temperature
        if y <= 11000:
            Tmp = 288.16 - 0.0065*y
        else:
            Tmp = 216.66
        # Mach number
        M = V/(1.4*self.R*Tmp)**0.5
        # Mass and thrust (Note: guidance loop is closed after t=t1)
        if t < self.t1:
            m = 135 - 14.53*t
            T = 33000
        elif t < self.t2:
            m = 113.205 - 3.331*t
            T = 7500
        else:
            m = 90.035
            T = 0
        # density and dynamic pressure
        rho = (1.15579 - 1.058*1e-4*y + 3.725*1e-9*y**2 
               -6.0*1e-14*y**3)      # y in [0, 20000]
        Q = 0.5*rho*V**2
        # Drag model
        if M < 0.93:
            Cd0 = 0.02
        elif M < 1.03:
            Cd0 = 0.02 +0.2*(M - 0.93)
        elif M < 1.10:
            Cd0 = 0.04 + 0.06*(M - 1.03)
        else:
            Cd0 = 0.0442 - 0.007*(M - 1.10)

        if M < 1.15:
            K = 0.2
        else:
            K = 0.2 + 0.246*(M - 1.15)

        D0 = Cd0*Q*self.S
        Di = K*m**2*a**2/(Q*self.S)
        D = D0 + Di

        dxdt = V*np.cos(gamma)
        dydt = V*np.sin(gamma)
        dVdt = (T - D)/m - self.g*np.sin(gamma)
        dgammadt = (a - self.g*np.cos(gamma))/V

        return np.hstack([dxdt, dydt, dVdt, dgammadt])
