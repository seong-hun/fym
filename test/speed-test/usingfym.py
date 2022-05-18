from collections import deque
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import fym.logging
from fym.agents.LQR import clqr
from fym.core import BaseEnv, BaseSystem, Delay
from fym.utils import parser


class Env(BaseEnv):
    def __init__(self):
        super().__init__(**parser.decode(cfg.env))
        self.plant = BaseSystem(cfg.initial_states.plant)
        self.W1 = BaseSystem(cfg.initial_states.W1)
        self.W2 = BaseSystem(cfg.initial_states.W2)
        self.J = BaseSystem()

    def observe(self):
        x = self.plant.state
        return self.Phi(x)

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        x = self.plant.state
        W1 = self.W1.state
        W2 = self.W2.state
        J = self.J.state

        x_prev = x
        J_prev = J

        rho = J - J_prev
        dphi = self.Phi(x) - self.Phi(x_prev)
        dPhi = self.dPhi(x)
        D = dPhi @ cfg.B @ cfg.Rinv @ cfg.B.T @ dPhi.T
        m = dphi / (dphi.T @ dphi + 1) ** 2

        u = -0.5 * cfg.Rinv @ cfg.B.T @ dPhi.T @ W2
        n = (
            0.5
            * np.exp(-0.1 * t)
            * np.sum(
                [
                    np.sin(5 * t) ** 2 * np.cos(t),
                    np.sin(8 * t) ** 2 * np.cos(0.1 * t),
                    np.sin(-2 * t) ** 2 * np.cos(0.5 * t),
                    np.sin(t) ** 5,
                ]
            )
        )
        u = u + n

        self.plant.dot = cfg.A @ x + cfg.B @ u
        self.W1.dot = -cfg.agent.alp1 * m @ (dphi.T @ W1 + rho)
        self.W2.dot = -cfg.agent.alp2 * (
            cfg.agent.F2 * W2 - cfg.agent.F1 * W1 - 0.25 * D @ W2 @ m.T @ W1
        )
        self.J.dot = self.Q(x) + 0.25 * W2.T @ D @ W2

        w1, w2, w3 = W1.ravel()
        P = 0.5 * np.array([[2 * w1, w2], [w2, 2 * w3]])

        w1, w2, w3 = W2.ravel()
        K = 0.5 * cfg.Rinv @ cfg.B.T @ np.array([[2 * w1, w2], [w2, 2 * w3]])

        return dict(
            t=t,
            x=np.rad2deg(x).ravel(),
            u=np.rad2deg(u).ravel(),
            P=P.ravel(),
            K=K.ravel(),
        )

    def Q(self, x):
        return x.T @ cfg.Q @ x

    def Phi(self, x):
        x1, x2 = x.ravel()
        return np.vstack((x1**2, x1 * x2, x2**2))

    def dPhi(self, x):
        x1, x2 = x.ravel()
        return np.array([[2 * x1, 0], [x2, x1], [0, 2 * x2]])


np.random.seed(0)

# Parameter setup
cfg = parser.parse(
    {
        "env": {
            "max_t": 20,
            "dt": 0.0025,
        },
        "initial_states": {
            "plant": np.deg2rad(np.vstack((5, -5))),
            "W1": 0.01 * np.random.rand(3, 1),
            "W2": 0.01 * np.random.rand(3, 1),
        },
        "Q": np.diag([0.1, 0.1]),
        "R": np.diag([0.1]),
        "agent": {
            "T": 0.01,
            "F1": 1,
            "F2": 1,
            "alp1": 1e4,
            "alp2": 1,
        },
    }
)

aerodata = scipy.io.loadmat("Data.mat", squeeze_me=True)
parser.update(
    cfg,
    {
        "A": np.array([[aerodata["Za"], 1], [aerodata["Ma"], aerodata["Mq"]]]),
        "B": np.vstack([aerodata["Zdp"], aerodata["Mdp"]]),
    },
)

cfg.Rinv = np.linalg.inv(cfg.R)

# Calculate the optimal
cfg.K, cfg.P = clqr(cfg.A, cfg.B, cfg.Q, cfg.R)


# Run simulation
def run():
    env = Env()
    env.logger = fym.logging.Logger("data.h5")
    env.logger.set_info(cfg=cfg)

    env.reset()

    while True:
        # env.render()
        done = env.step()

        if done:
            break

    env.close()
