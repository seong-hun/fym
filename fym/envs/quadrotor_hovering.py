import numpy as np
import gym
from gym import spaces
from scipy.spatial.transform import Rotation as R

from fym.models.quadrotor import Quadrotor
from fym.core import BaseEnv


class QuadrotorHoveringEnv(BaseEnv):
    def __init__(self, initial_state, dt=0.01):
        quadrotor = Quadrotor(initial_state=initial_state)

        obs_sp = gym.spaces.Box(
            low=-np.full(6, np.inf),
            high=np.full(6, np.inf),
            dtype=np.float32,
        )
        act_sp = gym.spaces.Box(
            low=-np.full(4, np.inf),
            high=np.full(4, np.inf),
            dtype=np.float32,
        )

        super().__init__(systems=[quadrotor], dt=dt,
                         obs_sp=obs_sp, act_sp=act_sp)

    def reset(self, noise=0):
        super().reset()
        self.states['quadrotor'] += np.random.uniform(-noise, noise)
        return self.get_ob()

    def step(self, action):
        # ----------------------------------------------------------------------
        # These lines will be replaced with
        #   controls = dict(aircraft=action)
        # lb, ub = self.action_space.low, self.action_space.high
        # quadrotor_control = (lb + ub)/2 + (ub - lb)/2*np.asarray(action)
        quadrotor_control = np.asarray(action)
        controls = dict(quadrotor=quadrotor_control)
        # ----------------------------------------------------------------------

        states = self.states.copy()
        next_obs, reward, done, _ = super().step(controls)
        info = {'states': states, 'next_states': self.states}
        return next_obs, reward, done, info

    def get_ob(self):
        state = self.states['quadrotor']
        position = state[:3]
        euler_angles = R.from_dcm(state[6:15].reshape(3, 3)).as_euler('ZYX')
        return np.hstack((position, euler_angles))

    def terminal(self):
        state = self.states['quadrotor']
        system = self.systems['quadrotor']
        lb, ub = system.state_lower_bound, system.state_upper_bound
        if not np.all([state > lb, state < ub]):
            return True
        else:
            return False

    def get_reward(self, controls):
        att = self.states['quadrotor'][3:5]  # goal attitude (roll, pitch)
        att_goal = [0, 0]
        error = self.weight_norm(att - att_goal, [1, 1])
        return -error

    def weight_norm(self, v, W):
        if np.asarray(W).ndim == 1:
            W = np.diag(W)
        elif np.asarray(W).ndim > 2:
            raise ValueError("W must have the dimension less than or equal to 2")
        return np.sqrt(np.dot(np.dot(v, W), v))


if __name__ == '__main__':
    env = QuadrotorHoveringEnv()
