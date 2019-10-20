from collections import OrderedDict
from itertools import chain
import functools

import numpy as np
from scipy.integrate import odeint
import gym

from fym.utils import logger


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


class BaseEnv(gym.Env):
    def __init__(self, systems: list, dt, max_t, infer_obs_space=True,
                 log_dir=None, file_name='state_history.h5',
                 ode_step_len=2, odeint_option={}):
        self.systems = OrderedDict({s.name: s for s in systems})

        if infer_obs_space and not hasattr(self, 'observation_space'):
            self.observation_space = self.infer_obs_space(self.systems)

        # Indices for packing
        self.state_index = [system.state_shape for system in systems]

        # Necessary properties for gym.Env
        if not hasattr(self, 'observation_space'):
            raise NotImplementedError('The observation_space is not defined.')

        if not hasattr(self, 'action_space'):
            raise NotImplementedError('The action_space is not defined.')

        self.clock = Clock(dt=dt, max_t=max_t)
        self.logger = logger.Logger(file_name=file_name)
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
        return self.observation(self.states)

    def get_next_states(self, t, states, action):
        xs = self.unpack_state(states)

        t_span = t + self.t_span
        func = self.ode_wrapper(self.derivs)
        ode_hist = odeint(func, xs, t_span, args=(action,), tfirst=True)

        packed_hist = [self.pack_state(_) for _ in ode_hist]
        next_states = packed_hist[-1]

        # Log the inner history of states
        for t, s in zip(t_span[:-1], packed_hist[:-1]):
            self.logger.log_dict(time=t, state=s, action=action)

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
    def __init__(self, name, initial_state, control_size=0, deriv=None):
        self.name = name
        self.initial_state = initial_state
        self.state_shape = self.initial_state.shape
        self.control_size = control_size

        if callable(deriv):
            self.deriv = deriv

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
