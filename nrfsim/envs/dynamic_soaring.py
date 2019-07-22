import numpy as np
import gym
from gym import spaces

from nrfsim.models.aircraft import Aircraft3Dof
from nrfsim.core import BaseEnv


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


class DynamicSoaringEnv(BaseEnv):
    def __init__(self):
        wind = Wind(Wref=10, href=10, h0=0.03)
        aircraft = Aircraft3Dof(
            initial_state=[0, 0, -1, 10, -0.5, 0],
            wind=wind
        )

        super().__init__(systems=[aircraft], dt=0.01)

        low = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        high = -low
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-0.3, np.deg2rad(-60)]),
            high=np.array([1.5, np.deg2rad(60)]),
            dtype=np.float32,
        )

    def step(self, action):
        lb, ub = self.action_space.low, self.action_space.high
        aircraft_control = (lb + ub)/2 + (ub - lb)/2*np.asarray(action)
        controls = dict(aircraft=aircraft_control)
        return super().step(controls)

    def get_ob(self):
        states = self.states['aircraft']
        return states[2:]

    def terminal(self):
        state = self.states['aircraft']
        system = self.systems['aircraft']
        lb, ub = system.state_lower_bound, system.state_upper_bound
        if not np.all([state > lb, state < ub]):
            return True
        else:
            return False

    def get_reward(self, states, controls):
        state = states['aircraft'][2:]
        goal_state = [-5, 10, 0, 0]
        error = self.weight_norm(state - goal_state, [0.02, 0.01, 1, 1])
        return -error

    def weight_norm(self, v, W):
        if np.asarray(W).ndim == 1:
            W = np.diag(W)
        elif np.asarray(W).ndim > 2:
            raise ValueError("W must have the dimension less than or equal to 2")
        return np.sqrt(np.dot(np.dot(v, W), v))


if __name__ == '__main__':
    env = DynamicSoaringEnv()
