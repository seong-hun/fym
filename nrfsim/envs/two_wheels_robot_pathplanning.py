import numpy as np
import gym
from gym import spaces
from nrfsim.models.two_wheels_robot import TwoWheelsRobot3Dof
from nrfsim.core import BaseEnv


class TwoWheelsRobotPathPlanningEnv(BaseEnv):
    def __init__(self, initial_state, dt=0.01):
        
        TwoWheelsRobot = TwoWheelsRobot3Dof(initial_state=initial_state)
        #import ipdb; ipdb.set_trace()
        super().__init__(systems=[TwoWheelsRobot], dt=dt)

        low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        high = -low
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.87, -1.87]),
            high=np.array([1.87, 1.87]),
            dtype=np.float32,
        )

    def reset(self, noise=0):
        super().reset()
        #self.states['aircraft'] += np.random.uniform(-noise, noise)
        return self.get_ob()

    def step(self, action):
        lb, ub = self.action_space.low, self.action_space.high
        TwoWheelsRobot_control = np.asarray(action)
        controls = dict(TwoWheelsRobot=TwoWheelsRobot_control)
        states = self.states.copy()
        #import ipdb; ipdb.set_trace()
        next_obs, reward, done, _ = super().step(controls)
        info = {'states': states, 'next_states': self.states}
        return next_obs, reward, done, info

    def get_ob(self):
        states = self.states['TwoWheelsRobot']
        return states

    def terminal(self):
        state = self.states['TwoWheelsRobot']
        system = self.systems['TwoWheelsRobot']
        lb, ub = system.state_lower_bound, system.state_upper_bound
        #import ipdb; ipdb.set_trace()
        if not np.all([state > lb, state < ub]):
            return True
        else:
            return False

    def get_reward(self, states, controls):
        '''
        state = states['electricWheelchair']
        goal_state = [-5, 10, 0, 0]
        error = self.weight_norm(state - goal_state, [0.02, 0.01, 1, 1])
        '''
        error = 1
        return -error

    def weight_norm(self, v, W):
        if np.asarray(W).ndim == 1:
            W = np.diag(W)
        elif np.asarray(W).ndim > 2:
            raise ValueError("W must have the dimension less than or equal to 2")
        return np.sqrt(np.dot(np.dot(v, W), v))


if __name__ == '__main__':
    env = TwoWheelsRobotPathPlanningEnv()
