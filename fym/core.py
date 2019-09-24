from collections import OrderedDict
import numpy as np
import gym


class BaseEnv(gym.Env):
    def __init__(self, systems: list, dt: float,
                 obs_sp: gym.spaces.Space, act_sp: gym.spaces.Space,
                 rk4_steps: int=2):
        self.systems = OrderedDict({s.name: s for s in systems})
        self.state_index = np.cumsum([system.state_size for system in systems])
        self.control_index = np.cumsum([system.control_size for system in systems])

        # Necessary properties for gym.Env
        self.observation_space = obs_sp
        self.action_space = act_sp

        self.rk4_steps = rk4_steps
        self.dt = dt

    def reset(self):
        initial_states = {
            k: system.reset() for k, system in self.systems.items()
        }
        self.states = initial_states
        self.clock = 0
        return self.get_ob()

    def step(self, controls):
        xs = np.hstack(list(self.states.values()))
        us = np.hstack(list(controls.values()))
        t = self.clock

        nxs = rk4(
            self.derivs,
            xs,
            t + np.linspace(0, self.dt, self.rk4_steps), us
        )

        nxs = nxs[-1]
        next_states = self.resolve(nxs, self.state_index)

        # Reward and terminal

        reward = self.get_reward(controls)
        terminal = self.terminal()

        # Update internal state and clock
        self.states = next_states
        self.clock = t + self.dt

        return (self.get_ob(), reward, terminal, {})

    def get_reward(self, controls: dict) -> float:
        raise NotImplementedError("Reward function is not defined in the Env.")

    def terminal(self) -> bool:
        raise NotImplementedError("Terminal is not defined in the Env.")

    def get_ob(self) -> np.ndarray:
        raise NotImplementedError("Observation is not defined in the Env.")

    def resolve(self, ss, index):
        *ss, _ = np.split(ss, index)
        some = OrderedDict(zip(self.systems.keys(), ss))
        return some

    def derivs(self, xs, t, us):
        """
        Returns:
            *xs*: ndarray
                An array of aggregated states.
            *t*: float
                The time when the derivatives calculated.
            *us*: ndarray
                An array of aggregated control inputs.
        """
        states = self.resolve(xs, self.state_index)
        controls = self.resolve(us, self.control_index)

        derivs = []
        for s, x, u in zip(*map(dict.values, [self.systems, states, controls])):
            derivs.append(s.deriv(x, t, u, s.external(states, controls)))

        return np.hstack(derivs)


class BaseSystem:
    def __init__(self, name, initial_state, control_size=0, deriv=None):
        self.name = name
        self.initial_state = initial_state
        self.state_size = len(initial_state)
        self.control_size = control_size
        if callable(deriv):
            self.deriv = deriv

    def deriv(self, state, t, control, external):
        raise NotImplementedError("deriv method is not defined in the system.")

    def reset(self):
        return self.initial_state


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
