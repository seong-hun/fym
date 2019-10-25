from collections import OrderedDict
from itertools import chain
import functools

import numpy as np
from scipy.integrate import odeint
import gym

import fym.logging import logging


class BaseEnv(gym.Env):
    def __init__(self, systems, dt, max_t,
                 tmp_dir='data/tmp', ode_step_len=2, odeint_option={}):
        self.systems = OrderedDict(systems)
        self.state_index = indexing(self.systems)

        if not hasattr(self, 'observation_space'):
            self.observation_space = self.infer_obs_space(self.systems)

        # Necessary properties for gym.Env
        if not hasattr(self, 'observation_space'):
            raise NotImplementedError('The observation_space is not defined.')

        if not hasattr(self, 'action_space'):
            raise NotImplementedError('The action_space is not defined.')

        self.clock = Clock(dt=dt, max_t=max_t)
        self.logger = logging.Logger(log_dir=tmp_dir, file_name='history.h5')
        self.odeint_option = odeint_option

        if not isinstance(ode_step_len, int):
            raise ValueError("ode_step_len should be integer.")

        self.t_span = np.linspace(0, dt, ode_step_len + 1)

    def reset(self):
        initial_states = {
            k: system.reset() for k, system in self.systems.items()
        }
        self.states = initial_states
        self.clock.reset()
        return self.states

    def get_next_states(self, t, states, action):
        xs = self.unpack_state(states)

        t_span = t + self.t_span
        func = self.ode_wrapper(self.derivs)
        ode_hist = odeint(func, xs, t_span, args=(action,), tfirst=True)

        packed_hist = [self.pack_state(_) for _ in ode_hist]
        next_states = packed_hist[-1]

        # Log the inner history of states
        for t, s in zip(t_span[:-1], packed_hist[:-1]):
            self.logger.record(time=t, state=s, action=action)

        return next_states, packed_hist

    def ode_wrapper(self, func):
        @functools.wraps(func)
        def wrapper(t, y, *args):
            states = self.pack_state(y)
            return func(t, states, *args)
        return wrapper

    def derivs(self, t, states, action):
        """
        It is recommened to override this method by a user-defined ``derivs``.

        Sample:
        ```python
        controls = self.pack_action(us, self.control_index)
        derivs = [system.deriv(t, states, controls) for system in self.systems]
        return np.hstack(derivs)
        ```
        """
        raise NotImplementedError

    def pack_state(self, flat_state):
        packed = OrderedDict(
            zip(self.systems.keys(), pack(flat_state, self.state_index)))
        return packed

    def unpack_state(self, states):
        unpacked = flatten(states.values())
        return np.hstack(unpacked)

    def step(self, action):
        raise NotImplementedError


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
        self._initial_state = np.asarray(val)

    def deriv(self):
        raise NotImplementedError("deriv method is not defined in the system.")

    def reset(self):
        return self.initial_state


class Clock:
    def __init__(self, dt, max_t=10):
        self.dt = dt
        self.max_t = max_t

    def reset(self):
        self.t = 0

    def tick(self):
        self.t += self.dt

    def get(self):
        return self.t


def pack(flat_state, indices):
    """
    Pack states from the flattened state using ``indices``.
    The ``indices`` is a list or a tuple which must have the equal length
    to the ``flat_state``.
    """
    div_points = [0] + np.cumsum(
        [np.prod(index) for index in indices]).tolist()

    packed = []
    for i in range(len(indices)):
        packed.append(
            flat_state[div_points[i]:div_points[i+1]].reshape(indices[i]))
    return packed


def indexing(systems):
    index = [system.state_shape for system in systems.values()]
    return index


def deep_flatten(arg):
    if arg == []:
        return arg
    if isinstance(arg, (list, tuple)):
        return deep_flatten(arg[0]) + deep_flatten(arg[1:])
    elif isinstance(arg, (np.ndarray, float, int)):
        return [np.asarray(arg).ravel()]


def flatten(arglist):
    return [np.asarray(arg).ravel() for arg in arglist]


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
