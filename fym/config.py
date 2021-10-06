from functools import reduce
import fym

default_settings = fym.parser.parse()
settings = fym.parser.parse(default_settings)


def register(d, base=None):
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


def load(key=None, as_dict=False):
    if isinstance(key, str):
        chunks = key.split(".")
        res = reduce(lambda v, k: v.__dict__[k], chunks, settings)
    else:
        res = settings

    if as_dict:
        return fym.parser.decode(res)
    else:
        return res


def update(d, base=None):
    if base is not None and isinstance(base, str):
        d = {base: d}
    fym.parser.update(settings, d)


def reset():
    fym.parser.update(settings, default_settings, prune=True)
