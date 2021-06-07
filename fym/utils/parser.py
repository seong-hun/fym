import numpy as np
from types import SimpleNamespace as SN
from functools import reduce
import re


class PrettySN(SN):
    def __repr__(self, indent=0):
        items = ["{"]
        indent += 1
        for key, val in self.__dict__.items():
            item = "  " * indent + str(key) + ": "
            if isinstance(val, SN):
                item += val.__repr__(indent)
            elif isinstance(val, np.ndarray):
                if val.ndim > 1 and val.shape[0] > 1:
                    item += np.array2string(
                        val, prefix=" " * len(item))
            else:
                item += str(val)
            item += ","
            items.append(item)
        indent -= 1
        items.append("  " * indent + "}")
        return "\n".join(items)


def _make_clean(string):
    """From https://stackoverflow.com/a/3305731"""
    return re.sub(r"\W+|^(?=\d)", "_", string)


def _put(a, b):
    for k, v in b.items():
        if k in a and isinstance(v, dict) and isinstance(a[k], dict):
            _put(a[k], b[k])
        else:
            a[k] = b[k]


def unwind_nested_dict(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = unwind_nested_dict(v)
        d = reduce(lambda d, k: {k: d}, k.split(".")[::-1], v)
        _put(result, d)
    return result


def encode(d):
    """Encode a dict to a SimpleNamespace"""
    if isinstance(d, SN):
        d = SN.__dict__
    elif not isinstance(d, dict):
        return d

    out = PrettySN()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(out, _make_clean(k), encode(v))
        else:
            setattr(out, _make_clean(k), v)
    return out


def decode(sn):
    """Decode a SimpleNamespace or a dict to a nested dict"""
    if isinstance(sn, SN):
        sn = sn.__dict__
    elif not isinstance(sn, dict):
        return sn
    out = {}
    for key, val in sn.items():
        if isinstance(val, (SN, dict)):
            val = decode(val)
        out.update({key: val})
    return out


def parse(d={}):
    return encode(unwind_nested_dict(decode(d)))


def update(sn, d):
    """Update a SimpleNamespace with a dict or a SimpleNamespace"""
    if isinstance(sn, SN):
        sn = vars(sn)
    d = unwind_nested_dict(decode(d))
    for k, v in d.items():
        if k in sn and isinstance(v, (dict, SN)) and isinstance(sn[k], (dict, SN)):
            update(sn[k], d[k])
        else:
            sn[k] = encode(d[k])


if __name__ == "__main__":
    # ``parser.parse``
    import fym.utils.parser as parser
    from fym.core import BaseEnv, BaseSystem

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

    # ``parser.update``
    cfg = parser.parse()

    def load_config():
        parser.update(cfg, {
            "env.kwargs": dict(dt=0.01, max_t=10),
            "agent.memory_size": 1000,
            "agent.minibatch_size": 32,
        })

    load_config()
    cfg.env.kwargs.dt = 0.001

    # ``parser.decode``
    cfg = parser.parse()
    parser.update(cfg, {"env.kwargs": dict(dt=0.01, max_t=10)})
    env = BaseEnv(**parser.decode(cfg.env.kwargs))
