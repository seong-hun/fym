import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import fym.core as core
import fym.models.aircraft as aircraft 
import fym.logging as logging

class Env(core.BaseEnv):
    def __init__(self, main_state, sub_state):
        self.main = aircraft.TransportAugLinear(main_state)
        systems = {
            "main": self.main,
            "sub": core.BaseSystem(sub_state),
            # "dcm": core.BaseSystem(initial) 
        }
        super().__init__(
            systems_dict=systems,
            dt=0.01,
            max_t=100,
        )
    
    def reset(self):
        super().reset()
        return self.observation()
        
    def observation(self):
        return self.main.C.dot(self.main.state)
    
    def step(self, control):
        state = self.main.state
        self.update(control)
        return self.observation(), 0, self.clock.time_over(), {"state": state}

    def derivs(self, time, control):
        main = self.systems_dict["main"]
        sub = self.systems_dict["sub"]
        
        main.set_dot(main.deriv(main.state, control))
        sub.set_dot(control[:2])


env = Env(main_state=[250, 0, 0, 0, 0, 0, 0, 0, 0, 0], sub_state=[0, 0])
logger = logging.Logger(log_dir="data", file_name="tmp.h5")

obs = env.reset()
while True:
    K = np.array([
        [2.598, 0, 0, 0, 0, -0.9927, 0],
        [0, 583.7, -58.33, -2.054, -1.375, 0, 6.1]
    ])
    time = env.clock.get()

    control = np.hstack([-K.dot(obs), np.deg2rad(-2.5)])
    next_obs, reward, done, info = env.step(control)

    logger.record(time=time, **info)

    obs = next_obs

    if done:
        break

logger.close()

data = logging.load(logger.path)

vel, alp, theta, q = data["state"][:, :4].T

plt.rc("font", **{
    "family": "sans-serif",
    # "sans-serif": ["Helvetica"],
})
# plt.rc("text", usetex=True)
plt.rc("lines", linewidth=1.3)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=0.8)


canvas = []
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].set_ylabel(r"$V$ [m/sec]")
axes[1].set_ylabel(r"$\alpha$ [deg]")
axes[1].set_xlabel("Time [sec]")
canvas.append((fig, axes))

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].set_ylabel(r"$\theta$ [deg]")
axes[1].set_ylabel(r"$q$ [deg/sec]")
axes[1].set_xlabel("Time [sec]")
canvas.append((fig, axes))


def plot_single(data, color="k", name=None):
    time = data["time"]
    vel, alp, theta, q = data["state"][:, :4].T
    # aileron, rudder = data["control"].T
    alp, theta, q = np.rad2deg([alp, theta, q])

    canvas[0][1][0].plot(time, vel, color=color, label=name)
    canvas[0][1][1].plot(time, alp, color=color)

    canvas[1][1][0].plot(time, theta, color=color, label=name)
    canvas[1][1][1].plot(time, q, color=color)


plot_single(data)
plt.show()