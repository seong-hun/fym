# {any}`fym.parser` module

```{currentmodule} fym
```

This module assists setting up the configuration of each module and scripts. It
parses JSON files with dot notation such as `settings.json` in [Visual Studio
Code](https://code.visualstudio.com).

## Basic Usage

### {any}`parser.parse`

{any}`parser.parse` parses a JSON-like `dict` object to a nested
[`types.SimpleNamespace`](https://docs.python.org/3/library/types.html#types.SimpleNamespace)
object like following. It encodes a dotted key into a nested structure to make
ease of grouping the configuration.

```python
from fym import parser, BaseEnv, BaseSystem

json_dict = {
    "env.kwargs": dict(dt=0.01, max_t=10),
    "multicopter.nrotor": 6,
    "multicopter.m": 3.,
    "multicopter.LQRGain": {
        "Q": [1, 1, 1, 1],
        "R": [1, 1],
    },
    "actuator.tau": 1e-1,
}

cfg = parser.parse(json_dict)
```

### {any}`parser.update`

Because the {any}`parser.parse` method returns a `SimpleNamespace` instance,
the object should be updated or modified following the `SimpleNamespace`'s
rule. To make ease of usage, {any}`parser` provides a convenient method,
{any}`parser.update`, which can take a JSON-like `dict` with dotted keys or
another `SimpleNamespace` instance. See the following example.

```python
cfg = parser.parse()

def load_config():
    parser.update(cfg, {
        "env.kwargs": dict(dt=0.01, max_t=10),
        "agent.memory_size": 1000,
        "agent.minibatch_size": 32,
    })

load_config()
cfg.env.kwargs.dt = 0.001
```

### The `prune` keyword

If you pass a `prune=True` keyword argument to `update` method, it will remove
the items that exist only in the first argument. It is extremely useful when
one need to reset a configuration without losing the reference.

- `config.py`

    ```python
    from fym import parser

    default_cfg = parser.parse({"k1": 1, "k2": 2})
    cfg = parser.copy(default_cfg)
    ```

- `foo.py`

    ```python
    from fym import parser
    import config

    cfg = config.cfg

    def bar():
        print(cfg)
    ```

- `main.py`

    ```python
    from fym import parser
    import foo
    import config

    cfg = config.cfg

    def main():
        # With prune=False (default)
        parser.update(cfg, {"k1": 2, "k3": 3})
        foo.bar()  # k1: 2, k2: 2, k3: 3

        parser.update(cfg, config.default_cfg)
        foo.bar()  # k1: 1, k2: 2, k3: 3

        # With prune=True
        parser.update(cfg, {"k1": 2, "k3": 3})
        parser.update(cfg, config.default_cfg, prune=True)
        foo.bar()  # k1: 1, k2: 2

    if __name__ == "__main__":
        main()
    ```

As you can see, the reference in `foo.cfg` is not broken. This concept can be
used to setup one default configuration, and several per-experiment,
use-defined configurations.

### {any}`parser.decode`

Note that all the child elements in the returned `SimpleNamespace` is also a
`SimpleNamespace`. A common method to conert it into a `dict` is `vars`,
although it cannot handle the nested `dict`. {any}`parser` module provides a
{any}`parser.decode` method that converts the nested `SimpleNamespace` object
into a nested `dict` which may useful to deal with keyword arguments.

```python
cfg = parser.parse()
parser.update(cfg, {"env.kwargs": dict(dt=0.01, max_t=10)})
env = BaseEnv(**parser.decode(cfg.env.kwargs))
```

### {any}`parser.merge`

This is a recursively merging method for `dict`-like objects.

```python
a = parser.parse({"env.kwargs": dict(dt=0.01, max_t=10)})
b = parser.parse({"env.kwargs.dt": 0.05, "env.loggerPath": "data.h5"})
```

Now, two `SimpleNamespace`s, `a` and `b`, can be merged through:

```python
c = parser.merge(a, b)
```

The result is given by

```python
{
  env: {
    kwargs: {
      dt: 0.05,
      max_t: 10,
    },
    loggerPath: data.h5,
  },
}
```

### {any}`parser.copy`

This is a nothing but a `deepcopy` of the dict-like object. Hence, it breaks
the references.

```python
def copy(d):
    return deepcopy(parse(d))
```
