"""
References: E. Porter, S. H. Hamdar, and W. Daamen,
“Pedestrian dynamics at transit stations:
an integrated pedestrian flow modeling approach,”
Transp. A Transp. Sci., vol. 14, no. 5, pp. 468–483, 2018.
"""
import gym
from gym import spaces
import numpy as np
from nrfsim.core import BaseSystem


class Pedestrian(BaseSystem):
    L = 0.5
    r = 0.3
    Jb = 1
    m = 10
    name = 'Pedestrian'
    state_lower_bound = [-np.inf, -np.inf, -1, -1],
    state_upper_bound = [np.inf, np.inf, 1, 1],
    control_lower_bound = [-1.87, -1.87]  # Left, Right wheel torque ###
    control_upper_bound = [1.87, 1.87]
    control_size = np.array(control_lower_bound).size

    def __init__(self, initial_state):
        super().__init__(self.name, initial_state, self.control_size)

    def external(self, states, controls):
        return 0

    def deriv(self, state, t, control, external):
        x, y, vx, vy, theta = state
        Cbi = np.array([[np.cos(theta), np.sin(theta), 0],
                        [-np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        Cib = np.transpose(Cbi)
        T1, T2 = control
        F1b = np.array([0, T1 / self.r, 0])
        F2b = np.array([0, T2 / self.r, 0])
        F1i = Cib.dot(F1b)
        F2i = Cib.dot(F2b)
        Mb = np.cross([-self.L / 2, 0, 0], F1b) + np.cross([self.L / 2, 0, 0],
                                                           F2b)

        dxdt = vx
        dydt = vy
        dvxdt = (F1i[0] + F2i[0]) / self.m
        dvydt = (F1i[1] + F2i[1]) / self.m
        dthetadt = Mb[2] / self.Jb

        return np.array([dxdt, dydt, dvxdt, dvydt, dthetadt])
