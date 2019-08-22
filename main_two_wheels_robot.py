import numpy as np
from nrfsim.envs.two_wheels_robot_pathplanning import TwoWheelsRobotPathPlanningEnv
from nrfsim.utils.plotting import PltModule
import matplotlib.pyplot as plt
import time

np.random.seed(1)
env = TwoWheelsRobotPathPlanningEnv(initial_state=np.array([0, 0, 0, 0, 0]).astype('float'))
time_step = 0.01
time_series = np.arange(0, 5, time_step)
obs = env.reset()
obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))
start_time = time.time()         # elpased time

for i in time_series:
    controls = np.array([1, 0.5])
    # Need logging here
    next_obs, reward, done, _ = env.step(controls)
    print(obs)
    if done:
        break
    obs = next_obs
    obs_series = np.vstack((obs_series, obs))

elapsed_time = time.time() - start_time
print('simulation time = ', time_series[-1] - time_series[0], '[s]')
print('elpased time = ', elapsed_time, '[s]')

# plot figures (1)
data = {'obs': obs_series, 'traj': obs_series[:, 0:2]}
variables = {'obs': ('x', 'y', 'Vx', 'Vy', 'theta'), 'traj': ('x', 'y')}
quantities = {'obs': ('distance', 'distance', 'speed', 'speed', 'angle'),
              'traj': ('distance', 'distance')}
labels = ('obs', 'traj')
a = PltModule(time_series, data, variables, quantities)
a.plot_time(labels)
a.plot_traj(labels)
plt.show()

