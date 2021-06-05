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
            else:
                item += str(val)
            item += ","
            items.append(item)
        indent -= 1
        items.append("  " * indent + "}")
        return "\n".join(items)


def make_clean(string):
    """From https://stackoverflow.com/a/3305731"""
    return re.sub(r"\W+|^(?=\d)", "_", string)


def put(d, k, v):
    chunks = k.split(".", 1)
    root_key = chunks[0]
    if not root_key:
        raise KeyError("empty root key")
    if len(chunks) == 1:
        d[root_key] = v
    else:
        if root_key not in d:
            d[root_key] = {}
        if not isinstance(d[root_key], dict):
            raise KeyError("sub document is not a dictinary")
        put(d[root_key], chunks[1], v)


def unwind_nested_dict(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = unwind_nested_dict(v)
        put(result, k, v)
    return result


def merge_dict(a, b):
    for k, v in b.items():
        if k in a and isinstance(v, dict) and isinstance(a[k], dict):
            merge_dict(a[k], b[k])
        else:
            a[k] = b[k]


def dotdict2nestdict(data):
    out = {}
    for keystring, val in data.items():
        if isinstance(val, dict):
            val = dotdict2nestdict(val)
        d = reduce(lambda d, k: {k: d}, keystring.split(".")[::-1], val)
        merge_dict(out, d)
    return out


def encode(d):
    """Encode a dict to a SimpleNamespace"""
    if isinstance(d, SN):
        d = SN.__dict__
    elif not isinstance(d, dict):
        return d

    out = PrettySN()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(out, make_clean(k), encode(v))
        else:
            setattr(out, make_clean(k), v)
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
    import numpy as np
    import fym.utils.parser as parser

    cfg = parser.parse()
    agents_cfg = {}

    def load_config():
        parser.update(cfg, {
            "env.kwargs.dt": 0.01,
            "env.kwargs.max_t": 10,
            "env.solver": "rk4",
        })

        parser.update(agents_cfg, {
            "CommonAgent.memory_len": 4000,
            "CommonAgent.batch_size": 2000,
        })

    load_config()

    cfg.env.kwargs.dt = 0.02
