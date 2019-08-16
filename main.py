import numpy as np
import numpy.linalg as nla

from nrfsim.envs.dynamic_soaring import DynamicSoaringEnv
from nrfsim.utils.plotting import PltModule

np.random.seed(1)

env = DynamicSoaringEnv(
    initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'))

time_step = 0.01
time_series = np.arange(0, 2, time_step)

obs = env.reset()
obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))

for i in time_series:
    controls = np.zeros(2)

    # Need logging here
    next_obs, reward, done, _ = env.step(controls)
    print(obs)

    if done:
        break

    obs = next_obs
    obs_series = np.vstack((obs_series, obs))

time_series = time_series[:obs_series.shape[0]]

data = {'traj': obs_series[:, 0:3]}
variables = {'traj': ('x', 'y', 'z')}
quantities = {'traj': ('distance', 'distance', 'distance')}
labels = ('traj',)
a = PltModule(time_series, data, variables, quantities)
a.plot_time(labels)
a.plot_traj(labels)
