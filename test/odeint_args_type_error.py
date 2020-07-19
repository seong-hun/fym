import numpy as np
from fym.core import BaseEnv, BaseSystem
import fym.logging as logging
import os


class MyEnv(BaseEnv):
    def __init__(self, x0, **kwargs):
        super().__init__(**kwargs)
        # system
        self.sys = BaseSystem(x0)
        self.A = -np.eye(len(x0))

    def observation(self):
        return self.sys.state

    def reset(self):
        super().reset()
        return self.observation()

    def step(self):
        self.update()
        next_x = self.observation()
        done = self.clock.time_over()
        return next_x, 0., done, {}

    def set_dot(self, t):
        x = self.observation()
        self.sys.dot = self.A @ x


if __name__ == "__main__":
    import time

    def _sample(env, agent, log_dir, file_name):
        logger = logging.Logger(
            log_dir=log_dir, file_name=file_name, max_len=1000
        )
        obs = env.reset()
        while True:
            if agent is None:
                pass
            else:
                pass
            next_obs, reward, done, info = env.step()
            logger.record(**info)
            # obs = next_obs
            if done:
                break
        env.close()
        logger.close()

    x0 = np.zeros(3)
    env = MyEnv(
        x0,
        solver="odeint",
        # solver="rk4",
        max_t=100.,
        ode_step_len=1,
        dt=0.01,
    )
    log_dir = "tmp"
    file_name = "test.h5"
    t0 = time.time()
    _sample(env, None, log_dir, file_name)
    t1 = time.time()
    print(t1-t0)
    # file_path = os.path.join(log_dir, file_name)
    # data = logging.load(file_path)
