# Fym

**Fym** is a general perpose dynamical simulator based on
[Python](https://www.python.org). The origin of **Fym** is a flight simulator
that requires highly accurate integration (e.g. [Runge-Kutta-Fehlberg
method](https://en.wikipedia.org/wiki/Runge–Kutta–Fehlberg_method) or simply
`rk45`), and a set of components that interact each other. For the integration,
**Fym** supports various [Scipy
integration](https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html)
methods in addition with own fixed-step solver such as `rk4`. Also, **Fym** has
a novel structure that provides modular design of each component of systems,
which is much simiar to
[Simulink](https://kr.mathworks.com/products/simulink.html).

The **Fym** project began with the development of accurate flight simulators
that aerospace engineers could use with [OpenAI Gym](https://gym.openai.com) to
study reinforcement learning. This is why the package name is **Fym** (Flight +
Gym). Although it is now a general purpose dynamical simulator, many codes and
ideas have been devised in the OpenAI Gym.

For more information, see:
* [Documentation](https://seong-hun.github.io/fym-pages/build/html/index.html#)

# Installation

There are two ways to install **Fym**.

## Manual installation (recommended)

As **Fym** is the ongoing project, many changes are expected in the short time.
We periodically upload stable versions to [PyPi](https://pypi.org), but if you
want to use the latest features of **Fym** development, we recommend installing
**Fym** manually, as follows.

```bash
git clone https://github.com/fdcl-nrf/fym.git
cd fym
pip install -e .
```

Note that `master` branch contains all the latest features that can be used
immediately as the default development branch.

## Install with PyPi

If you want to install the most stable version of **Fym** uploaded in PyPi, you
can do it.

```bash
pip install fym
```

# Basic Usage

## Simulation template

The basic usage of **Fym** is very similar to Simulink (conceptual only, of
course). A simulation is executed through a following basic template.

```python
env = Env()
env.reset()

while True:
    done = env.step()
    if done:
        break

env.close()
```

As you can see, this is the legacy of the OpenAI Gym.

`Env` is like a [Blank
Model](https://www.mathworks.com/help/simulink/gs/create-a-simple-model.html#bu3nd7o-1)
in Simulink. Every setup including dynamics, simulation time-step, final time,
integration method should be defined in this main class. The `Env` class is
initialized with the following structure.

```python
from fym.core import BaseEnv, BaseSystem

class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=10)
```

The arguments of `super().__init__` establishes the integration method
consisting of a time step (`dt`), a final time (`max_t`), etc.

## Registration of `BaseSystem`

Now, you can add dynamical systems as follows. From now on, dynamical system
means a system that requires an integration in the simulation, denoted by
`BaseSystem`.

```python
import numpy as np

class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=10)
        self.plant = BaseSystem(shape=(3, 2))
        self.load = BaseSystem(np.vstack((0, 0, -1)))
        self.actuator = BaseSystem()
```

There are three ways to initalize `BaseSystem` to the main Blank Model, `Env`.
The first way is give it a shape as `BaseSystem(shape=(3, 2))`. This
initializes the system with a numpy zeros with a shape `(3, 2)`. Another way is
directly give it an initial state as `BaseSystem(np.vstack((0, 0, -1)))`.
Finally, it can be initialized without any argument, where it's default initial
state is a numpy zeros with a shape `(1, 1)`.

## States of `BaseSystem`

Because `BaseSystem` is a dynamical system, it has a state. The state is
initialized with the [registration of the
instance](https://www.notion.so/63f0df1d92a340299a10d64434c03c43) in
`BaseEnv.__init__` method. It is basically a list or a numpy array with any
shape. After the registration, states of each `BaseSystem` can be accessed in
anywhere, with `BaseSystem.state` variable.

```python
env = Env()
print(env.plant.state)
```

## Setup the dynamics in `BaseEnv.set_dot`

Every dynamcal systems, i.e., `BaseSystem`, has its own dynamics, or a
derivative. For example, there might be a stable linear system: `ẋ = - x`.
Since [we've define](https://www.notion.so/63f0df1d92a340299a10d64434c03c43) an
initial value in `BaseEnv.__init__` method, only defining the derivative
completes the [ordinary differential
equation](https://en.wikipedia.org/wiki/Ordinary_differential_equation), as it
is an [initial value
problem](https://en.wikipedia.org/wiki/Initial_value_problem). All the
derivatives should be defined in `BaseEnv.set_dot` method by simply assiging
the derivative to `BaseSystem.dot` variable.

```python
class Env(BaseEnv):
    """..."""
    def step(self, **action):
        *_, done = self.update(**action)
        return done

    def set_dot(self, t, **action):
        self.plant.dot = - self.plant.state
        """self.load.dot, self.actuator.dot, ... """
```

The method `BaseEnv.step` defines how the `BaseEnv` communicates with outer
world as in the [simulation
template](https://www.notion.so/63f0df1d92a340299a10d64434c03c43). The input
and output is free to define, but there must be a `self.update` method, which
will actually perform the integration. Fortunately, you don't need to define
`BaseEnv.update` method. Everything complicated, such as integration, is done
automatically by the **Fym** module.

## Define the interaction APIs for `BaseEnv`

### How Numerical Integration Works

**The most important thing that you must be aware of is that how the numerical
integration works.** **Fym** implements a continuous-time ODE solver. Inside
the `BaseEnv.set_dot` method, time `t` and `BaseSystem.state`'s are varying
continuously. Hence, if you want to design a continuous-time feedback
controller, you must define it inside the `BaseEnv.set_dot` method.

```python
class Env(BaseEnv):
    """..."""
    def set_dot(self, t):
        x = self.plant.state
        u = - K @ x  # a linear feedback input
        r = t > 1  # a Heviside function
        self.plant.dot = A @ x + B @ u + r
```

### Define the `BaseEnv.step` method with ZoH inputs

There is another type of input that requires the [zero-order
hold](https://en.wikipedia.org/wiki/Zero-order_hold) (ZoH) which is useful for
command signals or agent actions in reinforcement learning (of course, the
command signal can be performed within the `BaseEnv.set_dot` method using a
function of time). To conform to reinforcement learning practice (e.g., OpenAI
Gym), one can define a step method and call `update` method with keyword
arguments which are the ZoH inputs.

```python
class Env(BaseEnv):
    """..."""
    def set_dot(self, t, action):
        """..."""

    def step(self, action):
        *_, done = self.update(action=action)
        """Construct next_obs, action, done, info etc."""
        return next_obs, action, done, info
```

In the above example, the key of ZoH input is `action`, and it must be set to
an argument of `set_dot` method with the same key name. The `update` method
returns three object: `t_hist`, `ode_hist` and `done`, although only the last
`done` is useful for typical situations.

`update` method is actually do the numerical integration internally. Hence,
after calling `update`, every states contained in the `BaseEnv` will be
updated.

For the simulations that do not require the ZoH inputs, just call `update` and `set_dot` only with time `t`, like this:

```python
class Env(BaseEnv):
    """..."""
    def set_dot(self, t):
        """..."""

    def step(self):
        *_, done = self.update()
        return done
```
