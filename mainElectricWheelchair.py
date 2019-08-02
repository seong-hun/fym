import numpy as np
import numpy.linalg as nla

from nrfsim.envs.electricWheelchair_pathplanning import ElectricWheelchairEnv


np.random.seed(1)
time_step = 0.01
env = ElectricWheelchairEnv(initial_state=np.zeros((14,)).astype('float'), mload=40, rGBb=np.array([0, 0, 0]))


time_series = np.arange(0, 2, time_step)

obs = env.reset()
for i in time_series:
    controls = np.zeros(2)/10

    # Need logging here
    next_obs, reward, done, _ = env.step(controls)
    print(obs, done)

    if done:
        break

    obs = next_obs
