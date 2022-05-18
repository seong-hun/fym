import numpy as np

from fym.envs.two_wheels_robot_pathplanning import \
    TwoWheelsRobotPathPlanningEnv

np.random.seed(1)
env = TwoWheelsRobotPathPlanningEnv(
    initial_state=np.array([0, 0, 0, 0, 0]).astype("float")
)
time_step = 0.01
time_series = np.arange(0, 5, time_step)
obs = env.reset()
for i in time_series:
    controls = np.array([1, 0.5])
    # Need logging here
    next_obs, reward, done, _ = env.step(controls)
    print(obs)
    if done:
        break
    obs = next_obs
