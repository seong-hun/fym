import numpy as np
import numpy.linalg as nla

from nrfsim.envs.two_wheels_robot_pathplanning import TwoWheelsRobotPathPlanningEnv
import matplotlib.pyplot as plt

np.random.seed(1)

env = TwoWheelsRobotPathPlanningEnv(initial_state=np.array([0, 0, 0, 0, 0]).astype('float'))

time_step = 0.01
time_series = np.arange(0, 5, time_step)
stateHis = np.zeros([np.size(time_series, 0), 5])
j = 0
obs = env.reset()
for i in time_series:
    controls = np.array([1, 0.5])

    # Need logging here
    next_obs, reward, done, _ = env.step(controls)
    print(obs)
    stateHis[j] = np.array(next_obs)

    if done:
        break

    obs = next_obs
    j += 1

plt.figure(1)
plt.plot(time_series, stateHis[:,2], color="y", linewidth=1.5, linestyle="-")
plt.figure(2)
plt.plot(time_series, stateHis[:,3], color="y", linewidth=1.5, linestyle="-")
plt.figure(3)
plt.plot(time_series, stateHis[:,4], color="y", linewidth=1.5, linestyle="-")
plt.figure(4)
plt.plot(stateHis[:,0], stateHis[:,1], color="y", linewidth=1.5, linestyle="-")

