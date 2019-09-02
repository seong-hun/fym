import numpy as np
import gym
from gym import spaces

from fym.models.missile import MissilePlanar
from fym.core import BaseEnv


class StationaryTargetEnv(BaseEnv):
    g = 9.80665

    def __init__(self, initial_state, dt=0.01):
        missile = MissilePlanar(initial_state=initial_state)

        obs_sp = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )
        act_sp = gym.spaces.Box(
            low=np.array([-10*self.g]),
            high=np.array([10*self.g]),
            dtype=np.float32,
        )

        super().__init__(systems=[missile], dt=dt, obs_sp=obs_sp, act_sp=act_sp)

    def reset(self, noise=0):
        super().reset()
        self.states['missile'] += np.random.uniform(-noise, noise)
        return self.get_ob()

    def step(self, action):
        # ----------------------------------------------------------------------
        # These lines will be replaced with
        #   controls = dict(aircraft=action)
        lb, ub = self.action_space.low, self.action_space.high
        missile_control = (lb + ub)/2 + (ub - lb)/2*np.asarray(action)
        controls = dict(missile=missile_control)
        # ----------------------------------------------------------------------

        states = self.states.copy()
        next_obs, reward, done, _ = super().step(controls)
        info = {'states': states, 'next_states': self.states}
        return next_obs, reward, done, info

    def get_ob(self):
        states = self.states['missile']
        return states

    def terminal(self):
        state = self.states['missile']
        system = self.systems['missile']
        lb, ub = system.state_lower_bound, system.state_upper_bound
        if not np.all([state > lb, state < ub]):
            return True
        else:
            return False

    def get_reward(self, controls):
        state = self.states['missile'][:2]
        goal_state = [0, 0]      # target position
        error = self.weight_norm(state - goal_state, [1, 1])
        return -error

    def weight_norm(self, v, W):
        if np.asarray(W).ndim == 1:
            W = np.diag(W)
        elif np.asarray(W).ndim > 2:
            raise ValueError("W must have the dimension less than or equal to 2")
        return np.sqrt(np.dot(np.dot(v, W), v))


if __name__ == '__main__':
    env = StationaryTargetEnv()
