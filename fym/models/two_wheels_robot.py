'''
refrence: M. Deng, A.Inoue, K. Sekiguchi, L. Jian, 
"Two-wheeled mobile robot motion control in dynamic environments," 2009
'''
import gym
from gym import spaces
import numpy as np
from fym.core import BaseSystem


class TwoWheelsRobot3Dof(BaseSystem):
    L = 0.5
    r = 0.3
    Jb = 1
    m = 10
    name = 'TwoWheelsRobot'
    state_lower_bound = [-np.inf, -np.inf, -np.inf, -np.inf],
    state_upper_bound = [np.inf, np.inf, np.inf, np.inf],
    control_lower_bound = [-1.87, -1.87]  # Left, Right wheel torque
    control_upper_bound = [1.87, 1.87]
    control_size = np.array(control_lower_bound).size

    def __init__(self, initial_state):
        super().__init__(self.name, initial_state, self.control_size)
        
    def external(self, states, controls):
        return 0
        
    def deriv(self, state, t, control, external):
        x, y, v, theta = state
        T1, T2 = control

        dxdt = v*np.cos(theta)
        dydt = v*np.sin(theta)
        dvdt = (T1 / self.r + T2 / self.r) / self.m
        dthetadt = (-T1 / self.r + T2 / self.r) * self.L / 2

        return np.array([dxdt, dydt, dvdt, dthetadt])
