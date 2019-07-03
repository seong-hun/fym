import numpy as np
import numpy.linalg as nla

from nrfsim.envs.dynamic_soaring import DynamicSoaringEnv


np.random.seed(1)

env = DynamicSoaringEnv()

time_step = 0.01
time_series = np.arange(0, 2, time_step)

obs = env.reset()
for i in time_series:
    controls = np.zeros(2)

    # Need logging here
    next_obs, reward, done, _ = env.step(controls)
    print(obs)

    if done:
        break

    obs = next_obs
