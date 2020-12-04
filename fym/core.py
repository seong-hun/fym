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
    def __init__(self, dt=0.01, max_t=1, eager_stop=None,
                 logger=None, logger_callback=None,
                 solver="rk4", ode_step_len=1, ode_option={},
                 name=None):
        self._name = name or self.__class__.__name__
        self._systems = dict()
        self.systems = self._systems.values()

        self._delays = dict()
        self.delays = self._delays.values()

        self.indexing()

        self.eager_stop = eager_stop

        if not hasattr(self, 'observation_space'):
            self.set_obs_space()
            print(
                "Observation space is inferred using the initial states "
                f"of the systems: {self._systems.keys()}"
            )

        if not hasattr(self, 'action_space'):
            raise NotImplementedError('The action_space is not defined.')

        if not isinstance(ode_step_len, int):
            raise ValueError("ode_step_len should be integer.")

        self.clock = Clock(dt=dt, ode_step_len=ode_step_len, max_t=max_t)

        self.logger = logger
        if logger_callback is not None or not hasattr(self, "logger_callback"):
            self.logger_callback = logger_callback

        # ODE Solver
        if solver == "odeint":
            self.solver = odeint
        elif solver == "rk4":
            self.solver = rk4

        self.ode_func = self.ode_wrapper(self.set_dot)
        self.ode_option = ode_option
        self.tqdm_bar = None

    def __getattr__(self, name):
        if "_systems" in self.__dict__:
            systems = self.__dict__["_systems"]
            if name in systems:
                return systems[name]

        if "_delays" in self.__dict__:
            delays = self.__dict__["_delays"]
            if name in delays:
                return delays[name]

        return super().__getattribute__(name)

        raise AttributeError(
            f"{type(self).__name__} object has no attribute {name}")

    def __setattr__(self, name, value):
        if isinstance(value, (BaseSystem, BaseEnv)):
            systems = self.__dict__.get("_systems")
            if systems is None:
                raise AttributeError(
                    "cannot assign system before BaseEnv.__init__() call")
            systems[name] = value
            if isinstance(value, BaseEnv) or value._name is None:
                value._name = name
            # if isinstance(value, BaseSystem):
            self.indexing()
            self.set_obs_space()
        elif isinstance(value, Delay):
            delays = self.__dict__.get("_delays")
            if delays is None:
                raise AttributeError(
                    "cannot assign delays before BaseEnv.__init__() call")
            delays[name] = value
        else:
            super().__setattr__(name, value)

    def set_obs_space(self):
        self.observation_space = infinite_box(self.state_shape)

    def __repr__(self, base=[]):
        name = self._name or self.__class__.__name__
        base = base + [name]
        # result = [
        #     f"<{' - '.join(base)}>",
        #     "state:",
        #     f"{self.state}"
        # ]
        # if hasattr(self, "dot"):
        #     result += ["dot:", f"{self.dot}"]
        # result.append("")
        result = []

        for system in self.systems:
            v_str = system.__repr__(base=base)
            result.append(v_str)
        return "\n".join(result)

    @property
    def state(self):
        return self.observe_vec()

    @state.setter
    def state(self, state):
        for system in self.systems:
            system.state = state[system.flat_index].reshape(system.state_shape)

    @property
    def dot(self):
        dot = []
        for system in self.systems:
            if system.dot is not None:
                dot.append(np.reshape(system.dot, (-1, 1)))
        return np.vstack(dot) if dot != [] else dot

    @dot.setter
    def dot(self, dot):
        for system in self.systems:
            system.dot = dot[system.flat_index].reshape(system.state_shape)

    def indexing(self):
        start = 0
        for system in self.systems:
            system.state_size = functools.reduce(
                lambda a, b: a * b, system.state_shape)
            system.flat_index = slice(start, start + system.state_size)
            start += system.state_size

        self.state_shape = (sum([
            system.state_size for system in self.systems
        ]), 1)

    def reset(self):
        for system in self.systems:
            system.reset()
        self.clock.reset()

    def observe_list(self, state=None):
        res = []
        if state is None:
            for system in self.systems:
                if isinstance(system, BaseSystem):
                    res.append(system.state)
                elif isinstance(system, BaseEnv):
                    res.append(system.observe_list())
        else:
            for system in self.systems:
                if isinstance(system, BaseSystem):
                    res.append(
                        state[system.flat_index].reshape(system.state_shape))
                elif isinstance(system, BaseEnv):
                    res.append(system.observe_list(state[system.flat_index]))
        return res

    def observe_dict(self, state=None):
        res = {}
        if state is None:
            for name, system in self._systems.items():
                if isinstance(system, BaseSystem):
                    res[name] = system.state
                elif isinstance(system, BaseEnv):
                    res[name] = system.observe_dict()
        else:
            for name, system in self._systems.items():
                if isinstance(system, BaseSystem):
                    res[name] = state[system.flat_index].reshape(
                        system.state_shape)
                elif isinstance(system, BaseEnv):
                    res[name] = system.observe_dict(state[system.flat_index])
        return res

    def observe_vec(self, state=None):
        res = []
        if state is None:
            res = [system.state.reshape(-1, 1) for system in self.systems]
        else:
            for system in self.systems:
                if isinstance(system, BaseSystem):
                    res.append(state[system.flat_index].reshape(-1, 1))
                elif isinstance(system, BaseEnv):
                    res.append(system.observe_vec(state[system.flat_index]))
        return np.vstack(res) if res != [] else []

    def observe_flat(self):
        flat = []
        for system in self.systems:
            if system.state is not None:
                flat.append(np.ravel(system.state))
        return np.hstack(flat) if flat != [] else flat

    def update(self, **kwargs):
        t_hist = self.clock.get_thist()
        ode_hist = self.solver(
            func=self.ode_func,
            y0=self.observe_flat(),
            t=t_hist,
            args=tuple(kwargs.values()),
            **self.ode_option
        )

        done = False
        if self.eager_stop:
            t_hist, ode_hist, done = self.eager_stop(t_hist, ode_hist)

        tfinal, yfinal = t_hist[-1], ode_hist[-1]
        # Update the systems' state
        for system in self.systems:
            system.state = yfinal[system.flat_index].reshape(system.state_shape)

        self.update_delays(t_hist, ode_hist)

        # Log the inner history of states
        if self.logger:
            if not self.logger_callback:
                for t, y in zip(t_hist[:-1], ode_hist[:-1]):
                    state_dict = self.observe_dict(y)
                    if kwargs:
                        self.logger.record(time=t, state=state_dict, **kwargs)
                    else:
                        self.logger.record(time=t, state=state_dict)
            else:
                for i, (t, y) in enumerate(zip(t_hist[:-1], ode_hist[:-1])):
                    self.logger.record(
                        **self.logger_callback(i, t, y, t_hist, ode_hist))

        self.clock.set(tfinal)

        return t_hist, ode_hist, done or self.clock.time_over()

    def update_delays(self, t_hist, ode_hist):
        for delay in self.delays:
            delay.update(t_hist, ode_hist)

        for system in self.systems:
            if isinstance(system, BaseEnv):
                system.update_delays(t_hist, ode_hist)

    def ode_wrapper(self, func):
        @functools.wraps(func)
        def wrapper(y, t, *args):
            for system in self.systems:
                system.state = y[system.flat_index].reshape(system.state_shape)
            func(t, *args)
            return self.dot.ravel()
        return wrapper

    def set_dot(self, time, *args):
        """Overwrite this method with a custom method.
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

    def step(self):
        """Sample code:
            ```python
            def step(self):
                self.update()
                done = self.clock.time_over()
                return self.observe_dict(), None, done, None
            ```
        """
        raise NotImplementedError

    def close(self):
        if self.logger:
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


class BaseSystem:
    def __init__(self, initial_state=None, shape=(1, 1), name=None):
        if initial_state is None:
            initial_state = np.zeros(shape)
        self.initial_state = initial_state
        # self.state = self.initial_state
        self.state_shape = self.initial_state.shape
        self._name = name

        self.has_delay = False

    def __repr__(self, base=[]):
        name = self._name or self.__class__.__name__
        base = base + [name]
        result = [
            f"<{' - '.join(base)}>",
            "state:",
            f"{self.state}"
        ]
        if hasattr(self, "dot"):
            result += ["dot:", f"{self.dot}"]
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
        self.state = self._initial_state

    @property
    def dot(self):
        return self._dot

    @dot.setter
    def dot(self, dot):
        self._dot = dot

    def reset(self):
        self.state = self.initial_state
        return self.state

    def set_delay(self, T):
        self.delay = Delay(self, T)
        self.has_delay = True

    def update_delays(self, t_hist, ode_hist):
        if self.has_delay:
            self.delay.update(t_hist, ode_hist)


class Sequential(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for arg in args:
            assert isinstance(arg, (BaseEnv, BaseSystem))
            setattr(self, arg.name, arg)

        for k, v in kwargs.items():
            assert isinstance(v, (BaseEnv, BaseSystem))
            setattr(self, k, v)


class Clock:
    def __init__(self, dt, ode_step_len, max_t=10):
        self.dt = dt
        self.max_t = max_t
        self.max_len = int(max_t / dt) + 1
        self.thist = np.linspace(0, dt, ode_step_len + 1)

    def reset(self, t=0.):
        self.t = t
        self.max_len = int((self.max_t - t) / self.dt) + 1

    def tick(self):
        self.t += self.dt

    def set(self, t):
        self.t = t

    def get(self):
        return self.t

    def time_over(self, t=None):
        if t is None:
            return self.get() >= self.max_t
        else:
            return t >= self.max_t

    def get_thist(self):
        thist = self.get() + self.thist
        if self.time_over(thist[-1]):
            index = np.where(thist > self.max_t)[0]
            if index.size == 0:
                return thist
            else:
                return thist[:index[0] + 1]
        else:
            return thist


class Delay:
    def __init__(self, system, T, fill_value="extrapolate"):
        self.system = system
        self.T = T
        self.fill_value = fill_value

        self.memory = []
        self.memory_dump = None

    def available(self, t):
        return t >= self.T

    def get(self, t):
        if self.memory_dump is not None:
            if t > self.memory_dump.x[-1] + self.T:
                fit = self.memory[0]
            else:
                fit = self.memory_dump

            y = fit(t - self.T)

            return y[self.system.flat_index].reshape(self.system.state_shape)
        else:
            return self.system.state

    def update(self, t_hist, state_hist):
        self.memory.append(
            interp1d(t_hist, state_hist, axis=0, fill_value=self.fill_value)
        )

        if t_hist[-1] >= self.T:
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
    return gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64)


def infer_obs_space(obj):
    """
    Infer the gym observation space from the ordered dictionary ``systems``,
    and return a gym.spaces.Dict
    """
    obs_space = infinite_box(obj.state_shape)
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
