import numpy as np
import numpy.linalg as nla
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from numpy.linalg import inv

from fym.envs.quadrotor_hovering import QuadrotorHoveringEnv
from fym.models.quadrotor import Quadrotor
from fym.utils.plotting import PltModule
from fym.agents.PID import PID

np.random.seed(1)

x0 = [0, 0, -5]
v0 = [0, 0, 0]
R0 = R.from_euler('ZYX', (np.pi/180)*np.array([0.0, 1.0, 1.0]), degrees=False).as_dcm()
dOmega = [0, 0, 0]
initial_state = np.hstack((x0, v0, R0.ravel(), dOmega))

env = QuadrotorHoveringEnv(
    initial_state=initial_state.astype('float'))

time_step = 0.01
time_series = np.arange(0, 50, time_step)

obs = env.reset()
obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))
ctrl_series = np.array([], dtype=np.float64).reshape(0, 4)
y_series = np.array([], dtype=np.float64).reshape(0, 3)

pid_roll = PID(1.0 * np.array([1.0, 0.1, 1.0]), windup=False)
pid_pitch = PID(1.0 * np.array([1.0, 0.1, 1.0]), windup=False)
pid_height = PID(5 * np.array([1.0, 0.1, 1.0]), windup=10)
for i in time_series:
    euler = R.from_dcm(obs[6:15].reshape(3,3)).as_euler('zyx', degrees=False)
    y = np.concatenate((euler[[2, 1]], -obs[2]), axis=None)
    y_goal = np.array([0, 0, 0])  # roll, pitch, height
    e_y = y - y_goal
    f24_diff = pid_roll.input(e_y[0])
    f31_diff = pid_pitch.input(e_y[1])
    quad = Quadrotor(initial_state=initial_state.astype('float'))
    f1234_sum = pid_height.input(-e_y[2]) + quad.m * quad.g
    controls = inv(np.array([[0, 1, 0, -1],
                            [-1, 0, 1, 0],
                            [1, 1, 1, 1],
                            [1, -1, 1, -1]])) @ np.transpose(
                                np.array([f24_diff, f31_diff, f1234_sum, 0]))

    # Need logging here
    next_obs, reward, done, _ = env.step(controls)
    # print(obs)

    if done:
        break

    obs = next_obs
    obs_series = np.vstack((obs_series, obs))
    ctrl_series = np.vstack((ctrl_series, controls))
    y_series = np.vstack((y_series, y))

time_series = time_series[:obs_series.shape[0]]

NED2ENU = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]])
data = {'traj': obs_series[:, 0:3] @ NED2ENU, 'input': ctrl_series, 'output': y_series}
variables = {'traj': ('x', 'y', 'z'), 'input': ('f1', 'f2', 'f3', 'f4'),
             'output': ('roll', 'pitch', 'height')}
quantities = {'traj': ('distance', 'distance', 'distance'),
              'input': ('force', 'force', 'force', 'force'),
              'output': ('angle', 'angle', 'distance')}
labels = ('traj', 'input', 'output')
a = PltModule(time_series, data, variables, quantities)
a.plot_time(labels)
a.plot_traj(labels)
plt.show()
