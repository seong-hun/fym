import itertools
import functools
import os

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import tqdm

import fym.logging as logging


class BaseEnv:
    def __init__(self, dt=0.01, max_t=1, eager_stop=None,
                 logger=None, logger_callback=None,
                 solver="rk4", ode_step_len=1, ode_option={},
                 name=None):
        self._name = name or self.__class__.__name__
        self._systems_dict = dict()
        self._systems_list = self._systems_dict.values()

        self._delays = dict()
        self.delays = self._delays.values()

        self.indexing()

        if eager_stop is not None or not hasattr(self, "eager_stop"):
            self.eager_stop = eager_stop

        if not isinstance(ode_step_len, int):
            raise ValueError("ode_step_len should be integer.")

        self.clock = Clock(dt=dt, max_t=max_t, ode_step_len=ode_step_len)

        self.logger = logger
        self._log_set_dot = True

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
        self._registered = False

    def __getattr__(self, name):
        val = self.__dict__.get("_systems_dict", {}).get(name, None)
        if val:
            return val

        val = self.__dict__.get("_delays", {}).get(name, None)
        if val:
            return val

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if isinstance(value, (BaseSystem, BaseEnv)) and not value._registered:
            systems = self.__dict__.get("_systems_dict")
            if systems is None:
                raise AttributeError(
                    "cannot assign system before BaseEnv.__init__() call")
            systems[name] = value
            value._registered = True
            if isinstance(value, BaseEnv) or value._name is None:
                value._name = name
            self.indexing()
            return
        elif isinstance(value, Delay):
            delays = self.__dict__.get("_delays")
            if delays is None:
                raise AttributeError(
                    "cannot assign delays before BaseEnv.__init__() call")
            delays[name] = value
            return
        elif isinstance(value, logging.Logger):
            value._inner = True

        super().__setattr__(name, value)

    def __repr__(self, base=[]):
        name = self._name or self.__class__.__name__
        base = base + [name]
        result = []

        for system in self._systems_list:
            v_str = system.__repr__(base=base)
            result.append(v_str)
        return "\n".join(result)

    @property
    def systems(self):
        return self._systems_list

    @property
    def state(self):
        return self._state.copy()

    @state.setter
    def state(self, state):
        self._state[:] = state

    @property
    def initial_state(self):
        res = [system.initial_state.reshape(-1, 1) for system in self._systems_list]
        return np.vstack(res) if res != [] else []

    @initial_state.setter
    def initial_state(self, state):
        for system in self._systems_list:
            initial_state = state[system.flat_index].reshape(system.state_shape)
            system.initial_state = initial_state

    @property
    def dot(self):
        return self._dot

    @dot.setter
    def dot(self, dot):
        self._dot[:] = dot

    def indexing(self):
        start = 0
        for system in self._systems_list:
            system.state_size = functools.reduce(
                lambda a, b: a * b, system.state_shape)
            system.flat_index = slice(start, start + system.state_size)
            start += system.state_size

        self.state_shape = (sum([
            system.state_size for system in self._systems_list
        ]), 1)

        self._state = np.empty(self.state_shape)
        self._dot = np.empty(self.state_shape)
        self.distributing()

    def distributing(self):
        for system in self._systems_list:
            system._state, system.state = self._state[system.flat_index].reshape(
                system.state_shape), system._state
            system._dot = self._dot[system.flat_index].reshape(
                system.state_shape)
            system.distributing()

    def reset(self):
        for system in self._systems_list:
            system.reset()
        self.clock.reset()

    def observe_list(self, state=None):
        res = []
        if state is None:
            for system in self._systems_list:
                if isinstance(system, BaseSystem):
                    res.append(system._state)
                elif isinstance(system, BaseEnv):
                    res.append(system.observe_list())
        else:
            for system in self._systems_list:
                if isinstance(system, BaseSystem):
                    res.append(
                        state[system.flat_index].reshape(system.state_shape))
                elif isinstance(system, BaseEnv):
                    res.append(system.observe_list(state[system.flat_index]))
        return res

    def observe_dict(self, state=None):
        res = {}
        if state is None:
            for name, system in self._systems_dict.items():
                if isinstance(system, BaseSystem):
                    res[name] = system._state
                elif isinstance(system, BaseEnv):
                    res[name] = system.observe_dict()
        else:
            for name, system in self._systems_dict.items():
                if isinstance(system, BaseSystem):
                    res[name] = state[system.flat_index].reshape(
                        system.state_shape)
                elif isinstance(system, BaseEnv):
                    res[name] = system.observe_dict(state[system.flat_index])
        return res

    def observe_vec(self, state=None):
        if state is None:
            res = self._state
        else:
            res = []
            for system in self._systems_list:
                if isinstance(system, BaseSystem):
                    res.append(state[system.flat_index].reshape(-1, 1))
                elif isinstance(system, BaseEnv):
                    res.append(system.observe_vec(state[system.flat_index]))
            res = np.vstack(res)
        return res

    def observe_flat(self):
        return self._state.ravel()

    def update(self, **kwargs):
        t_hist = self.clock._get_interval_span()
        ode_hist = self.solver(
            func=self.ode_func,
            y0=self._state.ravel(),
            t=t_hist,
            args=tuple(kwargs.values()),
            **self.ode_option
        )

        done = False
        if self.eager_stop:
            t_hist, ode_hist, done = self.eager_stop(t_hist, ode_hist)

        self.update_delays(t_hist, ode_hist)

        # Log the inner history of states
        if self.logger:
            for t, y in zip(t_hist[:-1], ode_hist[:-1]):
                self._record(t, y, **kwargs)
                self.clock._tick_minor()

        # Update the systems' state
        self.clock._tick_major()
        self._state[:] = ode_hist[-1][:, None]

        done = done or self.clock.time_over()
        if done:
            self._record(self.clock.get(), self.state.ravel())

        return t_hist, ode_hist, done or self.clock.time_over()

    def _record(self, t, y, **kwargs):
        self._state[:] = y[:, None]
        data = {}
        if self._log_set_dot:
            data.update(self.set_dot(t, **kwargs) or {})
            if not data:
                self._log_set_dot = False
        if self.logger_callback:
            data.update(self.logger_callback(t, **kwargs))
        self.logger._record(**(data or dict(t=t, **self.observe_dict())))

    def ode_wrapper(self, func):
        @functools.wraps(func)
        def wrapper(y, t, *args):
            self._state[:] = y[:, None]
            func(t, *args)
            return self._dot.ravel()
        return wrapper

    def update_delays(self, t_hist, ode_hist):
        for delay in self.delays:
            delay.update(t_hist, ode_hist)

        for system in self._systems_list:
            if isinstance(system, BaseEnv):
                system.update_delays(t_hist, ode_hist)

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
        self.state_shape = self.initial_state.shape
        self._name = name

        self.has_delay = False
        self._registered = False

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
        return self._state.copy()

    @state.setter
    def state(self, state):
        self._state[:] = state

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, val):
        self._initial_state = np.atleast_1d(val)
        self._state = self._initial_state.copy()

    @property
    def dot(self):
        return self._dot

    @dot.setter
    def dot(self, dot):
        self._dot[:] = dot

    def distributing(self):
        pass

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
        nargs = len(str(len(args)))
        for i, arg in enumerate(args):
            assert isinstance(arg, (BaseEnv, BaseSystem))
            setattr(self, f"{arg.name}_{i:0{nargs}d}", arg)

        for k, v in kwargs.items():
            assert isinstance(v, (BaseEnv, BaseSystem))
            setattr(self, k, v)


class Clock:
    def __init__(self, dt, max_t, ode_step_len=1):
        self.dt = dt
        self.max_t = max_t
        self._interval = ode_step_len
        interval_step = self.dt / self._interval
        self.tspan = np.arange(0, self.max_t + interval_step, interval_step)
        self.tspan = self.tspan[self.tspan <= max_t]
        self.index = 0
        self.max_len = int(np.ceil(max_t / dt))
        self._max_index = len(self.tspan) - 1

    def reset(self, t=0.):
        self.index = np.flatnonzero(self.tspan == t)[0].item()

    def _tick_major(self):
        self._major_index += 1
        self._minor_index = 0
        if self.index > self._max_index:
            self.index = self._max_index

    def _tick_minor(self):
        assert self._minor_index < self._interval
        self._minor_index += 1

    def _tick(self, step=1):
        self.index += step

    @property
    def index(self):
        return self._major_index * self._interval + self._minor_index

    @index.setter
    def index(self, ind):
        self._major_index = ind // self._interval
        self._minor_index = ind % self._interval

    def get(self):
        return self.tspan[self.index]

    def time_over(self, t=None):
        if t is None:
            return self.index == self._max_index
        else:
            return t >= self.max_t

    def _get_interval_span(self):
        return self.tspan[self.index:self.index + self._interval + 1]


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
            return self.system._state

    def update(self, t_hist, state_hist):
        self.memory.append(
            interp1d(t_hist, state_hist, axis=0, fill_value=self.fill_value)
        )

        if t_hist[-1] >= self.T:
            self.memory_dump = self.memory.pop(0)


def rk4(func, y0, t, args=()):
    n = len(t)
    y = np.empty((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = func(y[i], t[i], *args)
        k2 = func(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = func(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = func(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y
