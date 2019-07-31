import numpy as np
import numpy.linalg as nla

from nrfsim.envs.stationary_target_interception import StationaryTargetInterceptionEnv


np.random.seed(1)

env = StationaryTargetInterceptionEnv(initial_state=np.array([0, 5, 13, 0]).astype('float'))

time_step = 0.01
time_series = np.arange(0, 2, time_step)

obs = env.reset()
for i in time_series:
    controls = np.zeros(1)

    # Need logging here
    next_obs, reward, done, _ = env.step(controls)
    print(obs, done)

    if done:
        break

    obs = next_obs
