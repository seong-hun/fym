import numpy as np
import numpy.linalg as lin
from nrfsim.utils.plotting import PltModule
import matplotlib.pyplot as plt

class SFmodel:
    def __init__(self):
        self.detection_range = 5
        self.detection_angle = 135 * np.pi / 180
        self.attraitve_gain = 2
        self.Aa = 2.1
        self.Ba = 0.35
        self.r = 0.5
        self.Ao = 10
        self.Bo = 0.8
        self.desired_speed = 1

    def obstacle_detection(self, state, state_obstacles):
        """
        Whether social force is on/off between two object.

        :return:
        """
        pos = state[0:2]
        theta = state[3]
        n = np.size(state_obstacles, 0)
        pos_obstacle = []
        for i in np.arange(n):
            relative_pos = state_obstacles[i] - pos
            relative_angle = np.arctan2(relative_pos[1], relative_pos[0])
            if np.abs(relative_angle - theta) < self.detection_angle and \
                    0 < lin.norm(relative_pos) < self.detection_range:
                pos_obstacle.append(state_obstacles[i])
        return pos_obstacle

    def attractive(self, state, goal):
        """
        Attractive force to the goal.

        :return:
        """
        pos = state[0:2]
        v = state[2]
        theta = state[3]
        vel = np.array([v * np.cos(theta), v * np.sin(theta)])
        direction = (goal - pos) / lin.norm(goal - pos)
        desired_vel = self.desired_speed * direction
        #desired_vel = self.hrvo()
        attracitve_force = self.attraitve_gain * (desired_vel - vel)
        return attracitve_force

    def repulsive_agent(self, state, state_agents):
        """
        Repulsive force from other agents.

        :return:
        """
        if state_agents:
            obstacle = state_agents[:, 0:2]
            pos_obstacle = self.obstacle_detection(state, obstacle)
            pos = state[0:2]
            n = np.size(state_agents, 0)
            repulsive_force_agents = []
            for i in np.arange(n):
                d = lin.norm(pos - pos_obstacle[i])
                direction = (pos - pos_obstacle[i]) / d
                repulsive_force = self.Aa * np.exp((self.r - d) / self.Ba) * \
                                  direction
                repulsive_force_agents.append(repulsive_force)
            return sum(repulsive_force_agents)
        else:
            return 0

    def repulsive_object(self, state, obstacles):
        """
        Repulsive force from objects.

        :return:
        """
        pos_obstacle = self.obstacle_detection(state, obstacles)
        pos = state[0:2]
        n = np.size(pos_obstacle, 0)
        repulsive_force_obstacles = []
        for i in np.arange(n):
            d = lin.norm(pos - pos_obstacle[i])
            direction = (pos - pos_obstacle[i]) / d
            repulsive_force = self.Ao * np.exp((self.r - d) / self.Bo) * \
                              direction
            repulsive_force_obstacles.append(repulsive_force)
        return sum(repulsive_force_obstacles)

    def extended_social_force(self, state, state_agents, obstacles, goal,
                              attractive=True, repulsive_agent=True,
                              repulsive_obstacle=True):
        """
        Extended social force

        :return:
        """
        attractive_force = self.attractive(state, goal) if attractive else 0
        repulsive_force_agents = self.repulsive_agent(state, state_agents) \
            if repulsive_agent else 0
        repulsive_force_obstacles = self.repulsive_object(state, obstacles) \
            if repulsive_obstacle else 0

        extended_force = attractive_force + repulsive_force_agents + \
                         repulsive_force_obstacles
        return extended_force


#    def hrvo(self):




