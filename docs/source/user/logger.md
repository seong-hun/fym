# {any}`fym.logging` module

```{currentmodule} fym
```

There are several way to save a experimental data in `fym`.
The {any}`Logger` class provides several convient features to record a numerical
data into a [`HDF5` format](https://www.hdfgroup.org/solutions/hdf5/) file
using [`h5py` module](https://docs.h5py.org/en/stable/).

## Inner logger

### Basic usage

As long as a {any}`Logger` instance is registered with the {any}`BaseEnv`
instance as a `logger` attribute, the {any}`Logger` instance becomes the inner
logger, and is ready to record the inner states.

```python
env = Env()
env.logger = fym.Logger(path="data.h5")

"""..."""

env.close()
```

Note that you should {any}`BaseEnv.close` the BaseEnv to ensure all the data is
properly recorded into the file.

There are three ways to record {any}`BaseEnv`'s inner states.

1. Simply do nothing. The inner logger automatically records the time and all
	 the inner states.
2. Define the {any}`BaseEnv.logger_callback` method that returns a `dict`, or make
	 {any}`BaseEnv.set_dot` method return a `dict`. The returned dict will be
	 recorded.
3. Make both in 2. The two `dict` is going to be merged where
	 {any}`BaseEnv.logger_callback` has priority. (i.e.,
	 `set_dot_returned.update(logger_callback_returned)`)

### {any}`BaseEnv.logger_callback`

Sometimes, the data users want to record can be simply state variables within
the {any}`BaseEnv` and/or {any}`BaseSystem` and data derived from them (i.e.
control inputs). When using `fym`, there is one major {any}`BaseEnv` that runs
the simulation with {any}`BaseEnv.update` method. The
{any}`BaseEnv.logger_callback` method defines information to be recorded in
every time step when the environment updates. See the example below.

```python
from fym.core import BaseEnv, BaseSystem
import fym.logging

class Env(BaseEnv):
    """Other methods"""
    def logger_callback(self, t):
        ud = self.controller.get(self.plant)
        return dict(t=t, **self.observe_dict(), ud=ud)

class Controller(BaseSystem):
    """Other methods"""
    def get(self, plant):
        x = plant.state
        k = self.state
        return - k * x[0]

env = Env()
env.logger = fym.logging.Logger("data.h5")
env.reset()

while True:
    env.render()
    done = env.step()

    if done:
        break

env.close()

data = fym.logging.load("data.h5")
print(data)
```

### {any}`BaseEnv.set_dot`

If all the variables to be recorded is already calculated in {any}`BaseEnv.set_dot`,
simply return a dictionary of them is going to be recorded automatically. It
would reduce several lines of code and saves time. For this, you must remove
the {any}`BaseEnv.logger_callback` method as follows.

```python
class Env(BaseEnv):
    """Other methods without logger_callback"""
		def set_dot(self, t):
				x = self.plant.state
				u = self.actuator.state
				ud = self.controller.get(self.plant)

				self.plant.dot = - x
				self.actuator.dot = - (u - ud)
				self.controller.set_dot()

				return dict(t=t, **self.observe_dict(), ud=ud)
```

### When both returns are defined

- {any}`BaseEnv.set_dot` has returned value
	+ `BaseEnv.logger_callback` is defined: record
	both (priority check: `dict.update(set_dot, logger_callback)`)
- {any}`BaseEnv.set_dot` doesn't have returned value
  + `BaseEnv.logger_callback` is defined:
	record `BaseEnv.logger_callback`
- {any}`BaseEnv.set_dot` has returned value + `BaseEnv.logger_callback`
	is not defined: record {any}`BaseEnv.set_dot`
- {any}`BaseEnv.set_dot` doesn't have returned value
	+ `BaseEnv.logger_callback` is not defined:
	record `t` and {any}`BaseEnv.observe_dict()`
