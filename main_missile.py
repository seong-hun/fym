import numpy as np
import numpy.linalg as nla
import time

from nrfsim.envs.stationary_target_interception import StationaryTargetInterceptionEnv
from nrfsim.utils.plotting import PltModule


np.random.seed(1)

env = StationaryTargetInterceptionEnv(
    initial_state=np.array([-10e3, 0, 200, np.deg2rad(30)]).astype('float')
)

time_step = 0.01
t0 = 0      # launch time (no guidance loop)
t1 = 1.5    # first boost phase
t2 = 8.5    # second boost phase
tf = 40     # final simulation time
time_series = np.arange(t0, tf, time_step)

obs = env.reset()
obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))
start_time = time.time()         # elpased time

for i in time_series:
    if i < t1:
        controls = np.zeros(1)   # should be zero until the first boost phase
    else:
        controls = np.zeros(1)   # should be the output of controller

    # Need logging here
    next_obs, reward, done, _ = env.step(controls)
    # print(obs)

    if done:
        break
    obs = next_obs
    obs_series = np.vstack((obs_series, obs))

elapsed_time = time.time() - start_time
print('simulation time = ', tf - t0, '[s]')
print('elpased time = ', elapsed_time, '[s]')

# plot figures (1)
data = {'obs': obs_series, 'traj': obs_series[:, 0:2]}
variables = {'obs': ('x', 'y', 'V', 'gamma'), 'traj': ('x', 'y')}
quantities = {'obs': ('distance', 'distance', 'speed', 'angle'), 'traj': ('distance', 'distance')}
labels = ('obs', 'traj')
a = PltModule(time_series, data, variables, quantities)
a.plot_time(labels)
a.plot_traj(labels)
