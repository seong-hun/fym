import time

import numpy as np
from scipy.integrate import odeint

import fym
import fym.core as core
import fym.models.aircraft as aircraft
from fym.agents.LQR import clqr


class OriginalEnv(core.BaseEnv):
    def __init__(self, logging_off=False):
        super().__init__(
            systems={
                "main": aircraft.F16LinearLateral(),
                "aux": aircraft.F16LinearLateral(),
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
        info = {"states": self.observe_dict()}
        self.update(action)
        return self.observation(), 0, done, info

    def derivs(self, time, action):
        x1, x2 = [system.state for system in self.systems.values()]
        u1, u2 = action[:2], action[2:]

        main = self.systems["main"]
        aux = self.systems["aux"]
        main.dot = main.deriv(x1, u1)
        aux.dot = aux.deriv(x2, u2)

        # x, y = [self.states[ind] for ind in self.index.values()]
        # x = self.states[self.index["main"]]

        # states_dot = np.zeros_like(self.states)
        # states_dot[self.index["main"]] = self.main.deriv(x, action)
        # return states_dot


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

    print(
        "\t".join(
            [
                f"{text}:",
                f"{number} Runs,",
                f"Total: {t1 - t0:.4} sec",
            ]
        )
    )

    return t1 - t0


def test_validate():
    env = OriginalEnv()
    agent = Lqr(env.systems["main"])
    run(env, agent, 1, "Original Env (logging on)")


if __name__ == "__main__":
    test_validate()
