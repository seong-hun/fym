import numpy as np
import scipy.io

import matplotlib.pyplot as plt

from fym.utils import parser
from fym.agents.LQR import clqr


def RK4(func, t, y, **kwargs):
    k1 = func(t, y, **kwargs)
    k2 = func(t + cfg.env.dt/2, y + cfg.env.dt/2 * k1, **kwargs)
    k3 = func(t + cfg.env.dt/2, y + cfg.env.dt/2 * k2, **kwargs)
    k4 = func(t + cfg.env.dt, y + cfg.env.dt * k3, **kwargs)
    return y + cfg.env.dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def fx(t, x, u):
    return cfg.A @ x + cfg.B @ u


def fJ(t, J, x, u):
    return x.T @ cfg.Q @ x + u.T @ cfg.R @ u


def Phi(x):
    x1, x2 = x.ravel()
    return np.vstack((x1**2, x1 * x2, x2**2))


def dPhi(x):
    x1, x2 = x.ravel()
    return np.array([
        [2 * x1, 0],
        [x2, x1],
        [0, 2 * x2]
    ])


np.random.seed(0)

# Parameter setup
cfg = parser.parse({
    "env": {
        "max_t": 20,
        "dt": 0.0025,
    },
    "initial": {
        "X": np.deg2rad(np.vstack((5, -5))),
        "W1": 0.01 * np.random.rand(3, 1),
        "W2": 0.01 * np.random.rand(3, 1),
    },
    "Q": np.diag([0.1, 0.1]),
    "R": np.diag([0.1]),
    "T": 1,
    "agent": {
        "F1": 1,
        "F2": 1,
        "alp1": 1e3,
        "alp2": 1,
    },
})

aerodata = scipy.io.loadmat("Data.mat", squeeze_me=True)
parser.update(cfg, {
    "A": np.array([
        [aerodata["Za"], 1],
        [aerodata["Ma"], aerodata["Mq"]]
    ]),
    "B": np.vstack([aerodata["Zdp"], aerodata["Mdp"]]),
})

cfg.Rinv = np.linalg.inv(cfg.R)

# Calculate the optimal
cfg.K, cfg.P = clqr(cfg.A, cfg.B, cfg.Q, cfg.R)


# Simulation
def run():
    tspan = np.arange(0, cfg.env.max_t + cfg.env.dt, cfg.env.dt)
    X = np.zeros(tspan.shape + (2, 1))
    J = np.zeros(tspan.shape + (1, 1))
    W1 = np.zeros(tspan.shape + (3, 1))
    W2 = np.zeros(tspan.shape + (3, 1))
    U = np.zeros(tspan.shape + (1, 1))
    phi = np.zeros(tspan.shape + (3, 1))

    # Initial state
    X[0] = cfg.initial.X
    J[0] = 0
    W1[0] = cfg.initial.W1
    W2[0] = cfg.initial.W2

    for k in range(len(tspan) - 1):
        t = tspan[k]
        x = X[k]
        w1 = W1[k]
        w2 = W2[k]

        phi[k] = Phi(x)

        if k >= cfg.T:
            rho = J[k] - J[k - cfg.T]
            dphi = phi[k] - phi[k - cfg.T]
            D = dPhi(x) @ cfg.B @ cfg.Rinv @ cfg.B.T @ dPhi(x).T
            m = dphi / (dphi.T @ dphi + 1)**2

            dW1 = - cfg.agent.alp1 * m @ (dphi.T @ w1 + rho)
            dW2 = - cfg.agent.alp2 * (
                cfg.agent.F2 * w2 - cfg.agent.F1 * w1
                - 0.25 * D @ w2 @ m.T @ w1
            )
        else:
            dW1 = 0
            dW2 = 0

        u = - 0.5 * cfg.Rinv @ cfg.B.T @ dPhi(x).T @ w2
        n = 0.5 * np.exp(-0.1*t) * np.sum([
            np.sin(5*t)**2 * np.cos(t),
            np.sin(8*t)**2 * np.cos(0.1*t),
            np.sin(-2*t)**2 * np.cos(0.5*t),
            np.sin(t)**5
        ])
        u = u + n

        U[k] = u

        X[k + 1] = RK4(fx, t, x, u=u)
        J[k + 1] = RK4(fJ, t, J[k], x=x, u=u)
        W1[k + 1] = w1 + dW1 * cfg.env.dt
        W2[k + 1] = w2 + dW2 * cfg.env.dt
