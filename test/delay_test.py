import numpy as np
from scipy.interpolate import interp1d

import fym
import fym.core
import fym.agents.LQR
import fym.logging as logging


class System(fym.core.BaseSystem):
    A = np.array([
        [-1.01887, 0.90506, -0.00215],
        [0.82225, -1.07741, -0.17555],
        [0, 0, -1]
    ])
    B = np.array([
        [0],
        [0],
        [1]
    ])
    Q = np.diag([10, 10, 1])
    R = np.diag([1])

    def __init__(self, *args):
        super().__init__(*args)
        self.K, *_ = fym.agents.LQR.clqr(
            self.A, self.B, self.Q, self.R
        )

    def deriv(self, x, u):
        return self.A.dot(x) + self.B.dot(u)

    def get_control(self, x):
        return - self.K.dot(x)


class Env(fym.core.BaseEnv):
    def __init__(self, T, **kwargs):
        self.system = System([1, -1, 0])

        super().__init__(
            systems_dict={"main": self.system},
            **kwargs
        )

        self.set_delay([self.system], T)

    def reset(self):
        super().reset()
        return self.observation()

    def observation(self):
        return self.observe_flat()

    def step(self, action):
        done = self.clock.time_over()
        time = self.clock.get()
        state = self.system.state
        control = self.system.get_control(state)

        if self.delay.available():
            self.delay.set_states(time)
            d_state = self.system.d_state
            d_control = self.system.get_control(d_state)
        else:
            d_state = np.zeros_like(state)
            d_control = np.zeros_like(control)

        self.update(action)
        info = {
            "time": time,
            "state": state,
            "control": control,
            "delayed_state": d_state,
            "delayed_control": d_control,
        }

        return self.observation(), 0, done, info

    def derivs(self, time, action):
        if self.delay.available():
            self.delay.set_states(time)

        x = self.system.state
        u = self.system.get_control(x)
        self.system.set_dot(self.system.deriv(x, u))


def run(env):
    logger = logging.Logger(log_dir="data", file_name="tmp.h5")
    env.reset()
    while True:
        action = 0
        next_obs, reward, done, info = env.step(action)

        logger.record(**info)

        if done:
            break

    env.close()
    logger.close()
    return logger.path


T = 0.5
env = Env(T=T, dt=0.1, max_t=10, ode_step_len=4)
savepath = run(env)


import matplotlib.pyplot as plt

data = logging.load(savepath)

canvas = []
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].set_ylabel(r"$x_1$")
axes[1].set_ylabel(r"$x_2$")
axes[2].set_ylabel(r"$x_3$")
axes[2].set_xlabel("Time [sec]")
canvas.append((fig, axes))

fig, ax = plt.subplots(1, 1)
ax.set_ylabel(r"$u$")
ax.set_xlabel("Time [sec]")
canvas.append((fig, ax))

time = data["time"]

axes = canvas[0][1]
x1, x2, x3 = data["state"].T
axes[0].plot(time, x1, color="k", label="True")
axes[1].plot(time, x2, color="k")
axes[2].plot(time, x3, color="k")

xd1, xd2, xd3 = data["delayed_state"].T
dindex = time >= T
axes[0].plot(time[dindex], xd1[dindex], color="r", label="Delayed")
axes[1].plot(time[dindex], xd2[dindex], color="r")
axes[2].plot(time[dindex], xd3[dindex], color="r")

trans_time = time[time >= T]
axes[0].plot(trans_time, x1[:len(trans_time)], "k--", label="Trans")
axes[1].plot(trans_time, x2[:len(trans_time)], "k--")
axes[2].plot(trans_time, x3[:len(trans_time)], "k--")

axes[0].legend(*axes[0].get_legend_handles_labels())

ax = canvas[1][1]
u = data["control"]
ax.plot(time, u, color="k", label="True")

ud = data["delayed_control"]
ax.plot(time[dindex], ud[dindex], color="r", label="Delayed")

ax.plot(trans_time, u[:len(trans_time)], "k--", label="Trans")

plt.show()
