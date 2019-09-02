import numpy as np
import numpy.linalg as nla
import gym
from gym import spaces
from scipy.spatial.transform import Rotation as R

from fym.models.quadrotor import Quadrotor
from fym.core import BaseEnv
from fym.agents.PID import PID


class QuadrotorHoveringEnv(BaseEnv):
    # inner-loop PID control
    pid_height = PID(5 * np.array([1.0, 0.1, 1.0]), windup=10)
    pid_pitch = PID(1.0 * np.array([1.0, 0.1, 1.0]), windup=False)
    pid_roll = PID(1.0 * np.array([1.0, 0.1, 1.0]), windup=False)

    allocation_matrix = nla.inv(
        [[0, 1, 0, -1],
         [-1, 0, 1, 0],
         [1, 1, 1, 1],
         [1, -1, 1, -1]]
    )

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
        y_goal = action

        # inner-loop PID control
        quad = self.systems['quadrotor']
        y = np.array([-1, 1, 1]) * self.get_ob()[[2, 4, 5]]
        e_y = y - y_goal
        f1234_sum = self.pid_height.get(-e_y[0]) + quad.m * quad.g
        f31_diff = self.pid_pitch.get(e_y[1])
        f24_diff = self.pid_roll.get(e_y[2])
    
        quadrotor_control \
                = self.allocation_matrix.dot([f24_diff, f31_diff, f1234_sum, 0])
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
