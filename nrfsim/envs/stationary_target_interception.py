import numpy as np
import gym
from gym import spaces

from nrfsim.models.missile import MissilePlanar
from nrfsim.core import BaseEnv


class Wind:
    def __init__(self, Wref=10, href=10, h0=0.03):
        self.Wref = Wref
        self.href = href
        self.h0 = h0

    def get(self, state):
        _, y, V, gamma = state
        h = y

        # if h < 0:
        #     raise ValueError(f'Negative height {h}')
        h = max(h, self.h0)

        Wy = self.Wref*np.log(h/self.h0)/np.log(self.href/self.h0)
        dWyds = -self.Wref/h/np.log(self.href/self.h0)

        vel = [0, Wy]
        grad = [0, dWyds]
        return vel, grad


class StationaryTargetInterceptionEnv(BaseEnv):
    def __init__(self, initial_state, dt=0.01, Wref=10, href=10, h0=0.03):
        wind = Wind(Wref, href, h0)
        missile = MissilePlanar(initial_state=initial_state, wind=wind)

        super().__init__(systems=[missile], dt=dt)

        low = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        high = -low
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([np.deg2rad(-60)]),
            high=np.array([np.deg2rad(60)]),
            dtype=np.float32,
        )

    def reset(self, noise=0):
        super().reset()
        self.states['missile'] += np.random.uniform(-noise, noise)
        return self.get_ob()

    def step(self, action):
        lb, ub = self.action_space.low, self.action_space.high
        missile_control = (lb + ub)/2 + (ub - lb)/2*np.asarray(action)
        controls = dict(missile=missile_control)
        states = self.states.copy()
        next_obs, reward, done, _ = super().step(controls)
        info = {'states': states, 'next_states': self.states}
        return next_obs, reward, done, info

    def get_ob(self):
        states = self.states['missile']
        return states[2:]

    def terminal(self):
        state = self.states['missile']
        system = self.systems['missile']
        lb, ub = system.state_lower_bound, system.state_upper_bound
        # import ipdb; ipdb.set_trace()
        if not np.all([state > lb, state < ub]):
            return True
        else:
            return False

    def get_reward(self, states, controls):
        state = states['missile'][2:]
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
    env = StationaryTargetInterceptionEnv()
