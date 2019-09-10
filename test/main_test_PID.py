import numpy as np
from matplotlib import pyplot as plt

from fym.envs.quadrotor_hovering import QuadrotorHoveringEnv
from fym.models.quadrotor import Quadrotor
from fym.utils.plotting import PltModule
import fym.utils.rotation as rot

np.random.seed(1)

x0 = [0, 0, -5]
v0 = [0, 0, 0]
R0 = rot.angle_to_dcm(np.deg2rad([0.0, 1.0, 1.0]))
dOmega = [0, 0, 0]
initial_state = np.hstack((x0, v0, R0.ravel(), dOmega))

env = QuadrotorHoveringEnv(
    initial_state=initial_state.astype('float'))

time_step = 0.01
time_series = np.arange(0, 50, time_step)

obs = env.reset()
obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))

for i in time_series:
    action = np.array([0, 0, 0])  # desired height, pitch, roll
    next_obs, reward, done, _ = env.step(action)

    # obs_series = np.vstack((obs_series, obs))

    if done:
        break

    obs = next_obs

# time_series = time_series[:obs_series.shape[0]]

# NED2ENU = np.array([[0, 1, 0],
#                     [1, 0, 0],
#                     [0, 0, -1]])

# data = {
#     'traj': obs_series[:, 0:3].dot(NED2ENU),
#     'output': obs_series[:, [2, 4, 5]]
# }

# variables = {
#     'traj': ('x', 'y', 'z'),
#     'output': ('height', 'pitch', 'roll')
# }

# quantities = {
#     'traj': ('distance', 'distance', 'distance'),
#     'output': ('distance', 'angle', 'angle')
# }

# labels = ('traj', 'output')

# a = PltModule(time_series, data, variables, quantities)
# a.plot_time(labels)
# a.plot_traj(labels)
# plt.show()
