from collections import OrderedDict
import functools
import numpy as np
from scipy.integrate import odeint
import gym


def infer_obs_space(systems):
    """
    Infer the gym observation space from the ordered dictionary ``systems``,
    and return a gym.spaces.Dict
    """
    obs_space = gym.spaces.Dict(
        {k: infinite_box(s.state_shape) for k, s in systems.items()})
    return obs_space


class BaseEnv(gym.Env):
    def __init__(self, systems: list, dt: float, infer_obs_space=True):
        self.systems = OrderedDict({s.name: s for s in systems})

        if infer_obs_space and not hasattr(self, 'observation_space'):
            self.observation_space = self.infer_obs_space(self.systems)

        # Indices for packing
        self.state_index = [system.state_shape for system in systems]
        self.control_index = [system.control_shape for system in systems]

        # Necessary properties for gym.Env
        if not hasattr(self, 'observation_space'):
            raise NotImplementedError('The observation_space is not defined.')

        if not hasattr(self, 'action_space'):
            raise NotImplementedError('The action_space is not defined.')

        self.clock = Clock(dt=dt)

    def reset(self):
        initial_states = {
            k: system.reset() for k, system in self.systems.items()
        }
        self.states = initial_states
        self.clock.reset()
        return self.get_ob()

    def step(self, action):
        xs = np.hstack(list(self.states.values()))
        t_span = self.clock + np.array([0, self.dt])

        func = self.ode_wrapper(self.derivs, args=(action,))
        nxs = odeint(func, t_span, xs, tfirst=True, **self.ode_args)

        nxs = nxs[-1]
        next_states = self.pack_state(nxs)

        # Reward and terminal
        reward = self.reward()
        terminal = self.terminal()

        # Update internal state and clock
        self.states = next_states
        self.clock = t_span[-1]

        return (self.observation(next_states), reward, terminal, {})

    def ode_wrapper(self, func, args):
        @functools.wraps
        def wrapper(t, y):
            states = self.pack_state(y)
            return func(t, states, *args)
        return wrapper

    def derivs(self, t, states, action):
        """
        It is recommened to override this method by a user-defined ``derivs``.
        """
        controls = self.pack_action(us, self.control_index)
        derivs = [system.deriv(t, states, controls) for system in self.systems]
        return np.hstack(derivs)

    def pack_state(self, flat_state):
        unpacked = OrderedDict(
            zip(self.systems.keys(), pack(flat_state, self.state_index)))
        return unpacked

    def pack_action(self, action):
        unpacked = OrderedDict(
            zip(self.systems.keys(), pack(action, self.control_index)))
        return unpacked

    def resolve(self, ss, index):
        *ss, _ = np.split(ss, index)
        some = OrderedDict(zip(self.systems.keys(), ss))
        return some

    def reward(self, controls: dict) -> float:
        raise NotImplementedError("Reward function is not defined in the Env.")

    def terminal(self) -> bool:
        raise NotImplementedError("Terminal is not defined in the Env.")

    def observation(self, observation):
        raise NotImplementedError


class BaseSystem:
    def __init__(self, name, initial_state, control_size=0, deriv=None):
        self.name = name
        self.initial_state = initial_state
        self.state_shape = self.initial_state.shape()
        self.control_size = control_size

        if callable(deriv):
            self.deriv = deriv

    @property
    def initial_state(self):
        return _initial_state

    @initial_state.setter
    def initial_state(self, val):
        self._initial_stat = np.asarray(val)

    def deriv(self, state, t, control, external):
        raise NotImplementedError("deriv method is not defined in the system.")

    def reset(self):
        return self.initial_state


class Clock:
    def __init__(self, dt, max_t=None):
        self.dt = dt

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
