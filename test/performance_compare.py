import numpy as np
from scipy.integrate import odeint
import time

import fym
import fym.core as core
import fym.models.aircraft as aircraft
from fym.agents.LQR import clqr


class FastEnv(core.BaseEnv):
    def __init__(self, dt=0.01, max_t=10, rand_init=True, ode_step_len=2):
        self.main = aircraft.F16LinearLateral()
        self.aux = aircraft.F16LinearLateral()
        self.rand_init = rand_init
        self.observation_space = core.infinite_box((7,))
        self.action_space = core.infinite_box((2,))
        self.clock = core.Clock(dt=dt, max_t=max_t)
        self.t_span = np.linspace(0, dt, ode_step_len + 1)
        self.logging_off = True

    def derivs(self, t, x, u):
        x1, x2 = x[:7], x[7:]
        u1, u2 = u[:2], u[2:]
        dot_x1 = self.main.deriv(x1, u1)
        dot_x2 = self.aux.deriv(x2, u2)
        return np.hstack([dot_x1, dot_x2])

    def reset(self):
        self.state = np.hstack([self.main.initial_state] * 2)
        self.clock.reset()
        return self.state

    def step(self, action):
        state = self.state
        time = self.clock.get()

        ode_hist = odeint(
            func=self.derivs,
            y0=state,
            t=time + self.t_span,
            args=(action,),
            tfirst=True
        )
        next_state = ode_hist[-1]

        reward = 0

        self.state = next_state
        self.clock.tick()

        return next_state, reward, time > self.clock.max_t, {}


class OriginalEnv(core.BaseEnv):
    def __init__(self, logging_off=False):
        super().__init__(
            systems_dict={
                "main": aircraft.F16LinearLateral(),
                "aux": core.BaseSystem([1, 0, 0, 0, 0, 0, 0]),
            },
            dt=0.01,
            max_t=10,
            logging_off=logging_off,
        )

    def reset(self):
        super().reset()
        return self.observation()

    def observation(self):
        return self.observe_flat()

    def step(self, action):
        done = self.clock.time_over()
        self.update(action)
        return self.observation(), 0, done, {}

    def derivs(self, time, action):
        u1, u2 = action[:2], action[2:]

        main, aux = self.systems
        main.set_dot(main.deriv(main.state, u1))
        aux.set_dot(main.A.dot(aux.state) + main.B.dot(u2))


class Lqr:
    Q = np.diag([50, 100, 100, 50, 0, 0, 1])
    R = np.diag([0.1, 0.1])

    def __init__(self, sys):
        self.K = clqr(sys.A, sys.B, self.Q, self.R)[0]

    def get_action(self, x):
        x1, x2 = x[:7], x[7:]
        return np.hstack([-self.K.dot(x1), -self.K.dot(x2)])


def run(env, agent=None, number=1, text=""):
    t0 = time.time()
    for _ in range(number):
        obs = env.reset()
        while True:
            if agent is None:
                action = 0
            else:
                action = agent.get_action(obs)

            next_obs, _, done, _ = env.step(action)
            obs = next_obs

            if done:
                break

    t1 = time.time()

    env.close()

    print("\t".join([
        f"{text}:",
        f"{number} Runs,",
        f"Total: {t1 - t0:.4} sec",
    ]))

    return t1 - t0


def test_linear_system():
    number = 20

    env = OriginalEnv()
    agent = Lqr(env.systems_dict["main"])
    t0 = run(env, agent, number, "Original Env (logging on)")

    env = OriginalEnv(logging_off=True)
    agent = Lqr(env.systems_dict["main"])
    t1 = run(env, agent, number, "Original Env (logging off)")

    env = FastEnv()
    agent = Lqr(env.main)
    t2 = run(env, agent, number, "Fast Env (Euler integral)")

    print("\t".join([
        f"{t0/t2:5.2f}",
        f"{t1/t2:5.2f}",
    ]))


if __name__ == "__main__":
    test_linear_system()
