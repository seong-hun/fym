from fym.core import BaseEnv, BaseSystem


class Env(BaseEnv):
    def __init__(self):
        initial_state = 1
        system = ScalarSystem(initial_state=initial_state)

        super().__init__(systems={"main": system}, dt=0.1, max_t=3)

    def step(self, action):
        time = self.clock.get()
        states = self.states

        next_states, _ = self.get_next_states(time, states, action)

        done = self.is_terminal()

        self.states = next_states
        self.clock.tick()

        return states, None, done, None

    def is_terminal(self):
        if self.clock.get() > self.clock.max_t:
            return True
        else:
            return False

    def derivs(self, t, states, action):
        x, = states.values()
        u = - 3*x
        xdot = {
            "main": self.systems["main"].deriv(x, u)
        }
        return xdot


class ScalarSystem(BaseSystem):
    def __init__(self, initial_state):
        super().__init__(initial_state=initial_state)

    def deriv(self, x, u):
        xdot = 2*x + u
        return xdot


env = Env()
obs = env.reset()

while True:
    env.render()

    next_obs, reward, done, info = env.step(0)

    obs = next_obs

    if done:
        break

env.close()
