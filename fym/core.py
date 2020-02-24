import itertools
import functools
import os

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import tqdm

import gym

import fym.logging as logging


class BaseEnv(gym.Env):
    def __init__(self, systems_dict, dt=0.01, max_t=1,
                 logging_path=os.path.join("data", "tmp.h5"), logging_off=True,
                 solver="rk4", ode_step_len=1, ode_option={},
                 name=None):
        self.name = name
        self.systems_dict = systems_dict
        self.systems = systems_dict.values()
        self.state_shape = (sum([
            functools.reduce(lambda a, b: a * b, system.state_shape)
            for system in self.systems
        ]),)
        self.indexing()

        if not hasattr(self, 'observation_space'):
            self.observation_space = infer_obs_space(systems_dict)
            print(
                "Observation space is inferred using the initial states "
                f"of the systems: {self.systems_dict.keys()}"
            )

        if not hasattr(self, 'action_space'):
            raise NotImplementedError('The action_space is not defined.')

        self.clock = Clock(dt=dt, max_t=max_t)

        self.logging_off = logging_off
        if not logging_off:
            self.logger = logging.Logger(
                log_dir=tmp_dir, file_name='history.h5'
            )

        # ODE Solver
        if solver == "odeint":
            self.solver = odeint
        elif solver == "rk4":
            self.solver = rk4

        self.ode_func = self.ode_wrapper(self.set_dot)
        self.ode_option = ode_option
        self.tqdm_bar = None

        self.t_span = np.linspace(0, dt, ode_step_len + 1)

        self.delay = None

    def __repr__(self, base=[]):
        name = self.name or self.__class__.__name__
        base = base + [name]
        result = [
            f"<{' - '.join(base)}>",
            "state:",
            f"{self.state}"
        ]
        if hasattr(self, "dot"):
            result.append("dot:"
                          f"{self.dot}")
        result.append("")

        for system in self.systems:
            v_str = system.__repr__(base=base)
            result.append(v_str)
        return "\n".join(result)

    @property
    def state(self):
        return self.observe_flat()

    @state.setter
    def state(self, state):
        for system in self.systems:
            system.state = state[system.flat_index].reshape(system.state_shape)

    @property
    def dot(self):
        return np.hstack([np.ravel(system.dot) for system in self.systems])

    @dot.setter
    def dot(self, dot):
        for system in self.systems:
            system.dot = dot[system.flat_index].reshape(system.state_shape)

    def indexing(self):
        start = 0
        for system in self.systems:
            size = functools.reduce(lambda a, b: a * b, system.state_shape)
            system.flat_index = slice(start, start + size)
            start += size

    def reset(self):
        for system in self.systems:
            system.reset()
        self.clock.reset()

    def observe_list(self, state=None):
        if state is None:
            return [
                system.state for system in self.systems
            ]
        else:
            return [state[system.flat_index] for system in self.systems]

    def observe_dict(self):
        return {
            name: system.state
            for name, system in self.systems_dict.items()
        }

    def observe_flat(self):
        return np.hstack([
            system.state.ravel() for system in self.systems
        ])

    def update(self, *args):
        t_span = self.clock.get() + self.t_span
        ode_hist = self.solver(
            func=self.ode_func,
            y0=self.observe_flat(),
            t=t_span,
            args=args,
            **self.ode_option
        )

        # Update the systems' state
        y = ode_hist[-1]
        for system in self.systems:
            system.state = y[system.flat_index].reshape(system.state_shape)

        # Log the inner history of states
        if not self.logging_off:
            for t, y in zip(t_span[:-1], ode_hist[:-1]):
                state_dict = {
                    name: y[system.flat_index].reshape(system.state_shape)
                    for name, system in self.systems_dict.items()
                }
                self.logger.record(time=t, state=state_dict, action=action)

        self.clock.tick()

        if self.delay:
            self.delay.update(t_span, ode_hist)

    def ode_wrapper(self, func):
        @functools.wraps(func)
        def wrapper(y, t, *args):
            for system in self.systems:
                system.state = y[system.flat_index].reshape(system.state_shape)
            func(t, *args)
            return self.dot
        return wrapper

    def set_dot(self, time, *args):
        """
        Overwrite this method with a custom method.
        Note that ``*args`` are fixed during integration.
        If you want to time-varying variables,
        i.e. state feedback control inputs, or time-varying commands,
        you should use exogeneous methods taking time or states
        where the states can be obatined by ``self.system.state``.

        Sample code:
            ```python
            def set_dot(self, time, action):
                system = self.main_system
                state = system.state
                system.dot = system.A.dot(state) + system.B.dot(action)
            ```
        """
        raise NotImplementedError

    def append_systems(self, systems):
        self.systems_dict.update(systems)
        self.indexing()
        self.observation_space = infer_obs_space(self.systems_dict)

    def step(self, action):
        raise NotImplementedError

    def close(self):
        if not self.logging_off:
            self.logger.close()
        if self.tqdm_bar is not None:
            self.tqdm_bar.close()

    def render(self, mode="tqdm", desc=None, **kwargs):
        if mode == "tqdm":
            if self.tqdm_bar is None or self.clock.get() == 0:
                self.tqdm_bar = tqdm.tqdm(
                    total=self.clock.max_len,
                    **kwargs
                )

            self.tqdm_bar.update(1)
            if desc:
                self.tqdm_bar.set_description(desc)

    def set_delay(self, systems: list, T):
        self.delay = Delay(self.clock, systems, T)


class BaseSystem:
    def __init__(self, initial_state, name=None):
        self.initial_state = initial_state
        self.state = self.initial_state
        self.state_shape = self.initial_state.shape
        self.name = name

    def __repr__(self, base=[]):
        name = self.name or self.__class__.__name__
        base = base + [name]
        result = [
            f"<{' - '.join(base)}>",
            "state:",
            f"{self.state}"
        ]
        if hasattr(self, "dot"):
            result.append("dot:"
                          f"{self.dot}")
        result.append("")
        return "\n".join(result)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, val):
        self._initial_state = np.atleast_1d(val)

    @property
    def dot(self):
        return self._dot

    @dot.setter
    def dot(self, dot):
        self._dot = dot

    def reset(self):
        self.state = self.initial_state
        return self.state


class Clock:
    def __init__(self, dt, max_t=10):
        self.dt = dt
        self.max_t = max_t
        self.max_len = int(max_t / dt) + 1

    def reset(self):
        self.t = 0

    def tick(self):
        self.t += self.dt

    def get(self):
        return self.t

    def time_over(self):
        return self.get() >= self.max_t


class Delay:
    def __init__(self, clock, systems: list, T, fill_value="extrapolate"):
        if clock.dt > T:
            raise ValueError("Time step should be smaller than the delay")

        self.clock = clock
        self.systems = systems
        self.T = T
        self.fill_value = fill_value

        self.memory = []

    def available(self):
        return self.clock.get() >= self.T

    def set_states(self, time):
        if time > self.memory_dump.x[-1] + self.T:
            fit = self.memory[0]
        else:
            fit = self.memory_dump

        y = fit(time - self.T)

        for system in self.systems:
            system.d_state = y[system.flat_index].reshape(system.state_shape)

    def update(self, t_hist, state_hist):
        self.memory.append(
            interp1d(t_hist, state_hist, axis=0, fill_value=self.fill_value)
        )

        if self.clock.get() >= self.T:
            self.memory_dump = self.memory.pop(0)


def pack(flat_state, indices):
    """
    Pack states from the flattened state using ``indices``.
    The ``indices`` is a list or a tuple which must have the equal length
    to the ``flat_state``.
    """

    packed = []
    tmp = 0
    for index in indices:
        mult = functools.reduce(lambda a, b: a * b, index)
        packed.append(
            flat_state[tmp:tmp + mult].reshape(index)
        )
        tmp += mult

    return packed


def deep_flatten(arg):
    if arg == []:
        return arg
    if isinstance(arg, (list, tuple)):
        return deep_flatten(arg[0]) + deep_flatten(arg[1:])
    elif isinstance(arg, (np.ndarray, float, int)):
        return [np.asarray(arg).ravel()]


def infinite_box(shape):
    return gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)


def infer_obs_space(systems):
    """
    Infer the gym observation space from the ordered dictionary ``systems``,
    and return a gym.spaces.Dict
    """
    obs_space = gym.spaces.Dict(
        {k: infinite_box(s.state_shape) for k, s in systems.items()})
    return obs_space


def rk4(func, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = func(y[i], t[i], *args)
        k2 = func(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = func(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = func(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y
