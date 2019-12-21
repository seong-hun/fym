import numpy as np
import time

import fym
import fym.core as core
import fym.models.aircraft as aircraft
from fym.agents.LQR import clqr


class FastEnv(core.BaseEnv):
    def __init__(self, dt=0.01, max_t=10, rand_init=True):
        self.system = aircraft.F16LinearLateral()
        self.rand_init = rand_init
        self.observation_space = core.infinite_box((7,))
        self.action_space = core.infinite_box((2,))
        self.clock = core.Clock(dt=dt, max_t=max_t)

    def reset(self):
        self.state = self.system.initial_state
        self.clock.reset()

        if self.rand_init:
            self.state = (
                np.array([1, 20, 20, 6, 80, 80, 0])
                * np.random.uniform(-1.5, 1.5)
            )

        return self.state

    def step(self, action):
        state = self.state
        time = self.clock.get()

        next_state = state + self.system.deriv(state, action) * self.clock.dt

        # Q = np.array([50, 100, 100, 50, 0, 0, 1])
        # R = np.array([0.1])
        # reward = -(Q * states["main"]**2).sum() - (R * action**2).sum()
        reward = 0

        info = dict(
            time=time,
            state=state,
            control=action,
            reward=reward
        )

        self.state = next_state
        self.clock.tick()

        return next_state, reward, time > self.clock.max_t, info


class NewEnv(core.BaseEnv):
    def __init__(self, logging_off=False):
        system = aircraft.F16LinearLateral()
        super().__init__(
            systems={
                "main": system,
            },
            dt=0.01,
            max_t=10,
            logging_off=logging_off,
        )

    def reset(self):
        return self.observation(super().reset())

    def step(self, action):
        states = self.states
        time = self.clock.get()
        done = self.clock.time_over()
        next_states, _ = self.get_next_states(time, states, action)

        self.states = next_states
        self.clock.tick()
        return self.observation(next_states), 0, done, {}

    def derivs(self, time, states, action):
        x, = states.values()
        return {"main": self.systems["main"].deriv(x, action)}

    def observation(self, states):
        return states["main"]


class OriginalEnv(core.BaseEnv):
    def __init__(self, logging_off=False):
        system = aircraft.F16LinearLateral()
        super().__init__(
            systems={
                "main": system,
            },
            dt=0.01,
            max_t=10,
            logging_off=logging_off,
        )

    def reset(self):
        return self.observation(super().reset())

    def step(self, action):
        states = self.states
        time = self.clock.get()
        done = self.clock.time_over()
        next_states, _ = self.get_next_states(time, states, action)

        self.states = next_states
        self.clock.tick()
        return self.observation(next_states), 0, done, {}

    def derivs(self, time, states, action):
        x, = states.values()
        return {"main": self.systems["main"].deriv(x, action)}

    def observation(self, states):
        return states["main"]


class Lqr:
    Q = np.diag([50, 100, 100, 50, 0, 0, 1])
    R = np.diag([0.1, 0.1])

    def __init__(self, sys):
        self.K = clqr(sys.A, sys.B, self.Q, self.R)[0]

    def get_action(self, x):
        return -self.K.dot(x)


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

    print("\t".join([
        f"{text}:",
        f"{number} Runs,",
        f"Total: {t1 - t0:.4} sec",
    ]))


def test_linear_system():
    number = 10

    env = OriginalEnv()
    agent = Lqr(env.systems["main"])
    run(env, agent, number, "Original Env (logging on)")

    env = OriginalEnv(logging_off=True)
    agent = Lqr(env.systems["main"])
    run(env, agent, number, "Original Env (logging off)")

    env = FastEnv()
    agent = Lqr(env.system)
    run(env, agent, number, "Fast Env (Euler integral)")


def test_nonlinear_system():
    pass


if __name__ == "__main__":
    test_linear_system()
