from image_modefy import get_binary_map, rgb_constraint
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as lin
from nrfsim.envs.two_wheels_robot_pathplanning import TwoWheelsRobotPathPlanningEnv
from SFmodel import SFmodel
from nrfsim.models.two_wheels_robot import TwoWheelsRobot3Dof
from nrfsim.utils.linearization import jacob_analytic
from nrfsim.agents.LQR import dlqr
import time
from nrfsim.utils.plotting import PltModule


start_time = time.time()         # elpased time
np.random.seed(1)
map_size = [500, 500]
hospital_map_image = Image.open('snu_hospital.png')
hospital_map_image_red, hospital_image_name = \
    rgb_constraint([100, 0, 0], [255, 40, 40],
                   hospital_map_image=hospital_map_image, save=False)

waypoint_map_image = Image.open('waypoint_map.png')
waypoint_map_image_red, waypoint_image_name = \
    rgb_constraint([100, 0, 0], [255, 40, 40],
                   waypoint_map_image=waypoint_map_image, save=False)

obstacle, hospital_map_image_bin = get_binary_map(hospital_image_name, map_size)
waypoint, waypoint_map_image_bin = get_binary_map(waypoint_image_name, map_size)

waypoint_dot_imag = waypoint
waypoint = waypoint[0:-1:24, :]

initial_state = np.array([waypoint[6, 0], waypoint[6, 1], 0, np.pi / 2])
env = TwoWheelsRobotPathPlanningEnv(initial_state=initial_state.astype('float'))
obs = env.reset()
obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))
control_series = np.array([], dtype=np.float64).reshape(0, 2)
wheelchair_dynamics = TwoWheelsRobot3Dof(initial_state=initial_state)

dfdx = jacob_analytic(wheelchair_dynamics.deriv, 0)
dfdu = jacob_analytic(wheelchair_dynamics.deriv, 2)
Q = np.diag(np.array([0, 0, 1, 1]))
R = 0.1*np.eye(2)

sfmodel = SFmodel()

goal = [waypoint[5], waypoint[3], waypoint[4]]
j = 0
previous_control = np.array([0, 0])
time_step = 0.01
time_series = np.arange(0, 15, time_step)
v_max = 1
vel_r_pre = np.zeros([2])
for i in time_series:
    if lin.norm(goal[j] - obs[0:2]) < 0.1:
        j += 1

    sf = sfmodel.extended_social_force(obs, [], obstacle, goal[j])
    vel_r = vel_r_pre + sf * time_step
    ver_r = np.clip(vel_r, -v_max, v_max)

    v_r = lin.norm(vel_r)
    theta_r = np.arctan2(vel_r[1], vel_r[0])
    linearization_state = np.array([obs[0], obs[1], v_r, theta_r])

    A = dfdx(linearization_state, 0, previous_control, 0)
    B = dfdu(linearization_state, 0, previous_control, 0)

    K_con, X_con, eig_vals_con, eig_vecs_con = dlqr(A, B, Q, R)
    next_control = - K_con.dot(linearization_state)

    # Need logging here
    next_obs, reward, done, _ = env.step(next_control)

    if done:
        break
    obs = next_obs
    previous_state = next_obs
    previous_control = next_control
    obs_series = np.vstack((obs_series, obs))
    control_series = np.vstack((control_series, next_control))


plt.scatter(obstacle[:, 0], obstacle[:, 1], s=np.pi*0.1)
plt.scatter(waypoint_dot_imag[:, 0], waypoint_dot_imag[:, 1], s=np.pi*0.1)

elapsed_time = time.time() - start_time
print('simulation time = ', time_series[-1] - time_series[0], '[s]')
print('elpased time = ', elapsed_time, '[s]')

data = {'obs': obs_series, 'traj': obs_series[:, 0:2]}
variables = {'obs': ('x', 'y', 'V', 'theta'), 'traj': ('x', 'y')}
quantities = {'obs': ('distance', 'distance', 'speed', 'angle'),
              'traj': ('distance', 'distance')}
labels = ('obs', 'traj')
a = PltModule(time_series, data, variables, quantities)
a.plot_time(labels)
a.plot_traj(labels)

plt.show()
