import itertools
import functools

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import tqdm

import gym

import fym.logging as logging


class BaseEnv(gym.Env):
    def __init__(self, systems_dict, dt, max_t,
                 tmp_dir='data/tmp', logging_off=True,
                 ode_step_len=2, odeint_option={}):
        self.systems_dict = systems_dict
        self.systems = systems_dict.values()
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

        self.ode_func = self.ode_wrapper(self.derivs)
        self.odeint_option = odeint_option
        self.tqdm_bar = None

        if not isinstance(ode_step_len, int):
            raise ValueError("ode_step_len should be integer.")

        self.t_span = np.linspace(0, dt, ode_step_len + 1)

        self.delay = None

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

    def observe_dict(self):
        return {
            name: system.state
            for name, system in self.systems_dict.items()
        }

    def observe_flat(self):
        return np.hstack([
            system.state.ravel() for system in self.systems
        ])

    def update(self, action, *args):
        t_span = self.clock.get() + self.t_span
        ode_hist = odeint(
            func=self.ode_func,
            y0=self.observe_flat(),
            t=t_span,
            args=(action,) + args,
            tfirst=True
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
        def wrapper(t, y, *args):
            for system in self.systems:
                system.state = y[system.flat_index].reshape(system.state_shape)
            self.derivs(t, *args)
            return np.hstack([system._dot for system in self.systems])
        return wrapper

    def derivs(self, time, *args):
        """
        Overwrite this method with a custom method.
        Note that ``*args`` are fixed during integration.
        If you want to time-varying variables,
        i.e. state feedback control inputs, or time-varying commands,
        you should use exogeneous methods taking time or states
        where the states can be obatined by ``self.system.state``.

        Sample code:
            ```python
            def derivs(self, time, action):
                system = self.main_system
                state = system.state
                system.set_dot(
                    system.A.dot(state) + system.B.dot(action)
                )
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

    def render(self, mode="tqdm"):
        if mode == "tqdm":
            if self.tqdm_bar is None:
                self.tqdm_bar = tqdm.tqdm(
                    total=self.clock.max_len,
                    desc="Time"
                )

            self.tqdm_bar.update(1)

    def set_delay(self, systems: list, T):
        self.delay = Delay(self.clock, systems, T)


class BaseSystem:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.state_shape = self.initial_state.shape

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, val):
        self._initial_state = np.atleast_1d(val)

    def deriv(self):
        raise NotImplementedError("deriv method is not defined in the system.")

    def reset(self):
        self.state = self.initial_state
        return self.state

    def set_dot(self, deriv):
        self._dot = deriv


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
