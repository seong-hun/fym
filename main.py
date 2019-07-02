import numpy as np
import numpy.linalg as nla

from core import BaseEnvironment, BaseSystem


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

    def reset(self):
        initial_states = (
            super().reset()
            + [0, 0, 0, 4, 1, 1]*np.random.uniform(-1, 1, self.state_size)
        )
        return initial_states

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

    def _terminal(self):
        s = self.state
        lb, ub = self.state_lower_bound, self.state_upper_bound
        if not np.all([s > lb, s < ub]):
            return True
        else:
            return False


class Wind:
    def __init__(self, Wref=10, href=10, h0=0.03):
        self.Wref = Wref
        self.href = href
        self.h0 = h0

    def get(self, state):
        _, _, z, V, gamma, _ = state
        h = -z

        if h < 0:
            raise ValueError(f'Negative height {h}')

        Wy = self.Wref*np.log(h/self.h0)/np.log(self.href/self.h0)
        dWyds = -self.Wref/h/np.log(self.href/self.h0)

        vel = [0, Wy, 0]
        grad = [0, dWyds, 0]
        return vel, grad


class Environment(BaseEnvironment):
    def _get_reward(self, states, controls, next_states):
        state = states['aircraft'][2:]
        goal_state = [-5, 10, 0, 0]
        error = self.weight_norm(state - goal_state, [0.02, 0.01, 1, 1])
        return error

    def weight_norm(self, v, W):
        if np.asarray(W).ndim == 1:
            W = np.diag(W)
        elif np.asarray(W).ndim > 2:
            raise ValueError("W must have the dimension less than or equal to 2")
        return np.sqrt(np.dot(np.dot(v, W), v))


np.random.seed(1)

wind = Wind(Wref=10, href=10, h0=0.03)
aircraft = Aircraft3Dof(initial_state=[0, 0, -1, 10, -0.5, 0], wind=wind)

env = Environment(systems=[aircraft])

time_step = 0.01
time_series = np.arange(0, 2, time_step)

obs = env.reset()
for i in time_series:
    controls = dict(
        aircraft=np.zeros(2),
    )

    # Need logging here
    next_obs, reward, done, _ = env.step(controls, time_step)

    if done:
        break

    obs = next_obs
