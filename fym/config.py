from functools import reduce

import fym

default_settings = fym.parser.parse()
settings = fym.parser.parse(default_settings)


def register(d, base=None):
    """Register a configuration.

    .. versionadded:: 1.2.2

    Parameters
    ----------
    d : dict_like
        A configuration dictionary to be registered.
    base : str, optional
        Base path to register ``d``. The path is separated by dots (``.``),
        i.e., like ``"model.quadrotor"``.

    Examples
    --------
    At each module, the configuration can be registered as follows.
    In ``moduleA.submoduleB``:

    ::

        import fym

        fym.config.register(
            {
                "param1": 1,
                "paramSet.config1": 2,
            },
            base=__name__,
        )

    Then, from any other scripts, the configuration can be called
    with `load`.

    >>>  print(fym.config.load())
    {
      moduleA: {
        submoduleB: {
          param1: 1,
          paramSet: {
            config1: 2,
          },
        },
      },
    }

    """
    if base is not None and isinstance(base, str):
        d = {base: d}
    fym.parser.update(default_settings, d)
    fym.parser.update(settings, d)


# def decode(sn):
#     sn = sn.__dict__

#     for key, val in sn.items():
#         if isinstance(val, fym.parser.FymNamespace):
#             sn[key] = decode(val)

#     return sn


def load(path=None, as_dict=False):
    """Load the configuration.

    The configuration is empty `FymNamespace` unless any module loaded
    by the running script registers the configuration, or the script updates
    the configuration.

    .. versionadded:: 1.2.2

    Parameters
    ----------
    path : str, optional
        Absolute path where the requested configuration starts with.
        If the original configuration is defined as
        ``{"path.to.conf.requsted.conf": 1}``, then ``load("path.to.conf")``
        returns ``{"requested.conf: 1}``.
        If not specified, then the whole configuration is returned.
    as_dict : boolean, optional
        Return the configuration as a ``dict``, or `FymNamespace`.
        (Default: False)

    Returns
    -------
    cfg : dict or `FymNamespace`
        Child configuration under ``path``.

    """
    if isinstance(path, str):
        chunks = path.split(".")
        cfg = reduce(lambda v, k: v.__dict__[k], chunks, settings)
    else:
        cfg = settings

    if as_dict:
        return fym.parser.decode(cfg)
    else:
        return cfg


def update(d, base=None):
    """Update the configuration.

    The registered configuration does not change with this method.
    It only changes the configuration when the script running,
    i.e., this makes a local configuration.

    .. versionadded:: 1.2.2

    Parameters
    ----------
    d : dict_like
        New configuration that will replace or be added to the
        registered configuration.
    base : str, optional
        Base path to be updated using ``d``. The path is separated
        by dots (``.``), i.e., like ``"model.quadrotor"``.

    """
    if base is not None and isinstance(base, str):
        d = {base: d}
    fym.parser.update(settings, d)


def reset():
    fym.parser.update(settings, default_settings, prune=True)
