from types import SimpleNamespace as SN
from functools import reduce
import re


def make_clean(string):
    """From https://stackoverflow.com/a/3305731"""
    return re.sub(r"\W+|^(?=\d)", "_", string)


class SimpleSN(SN):
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


def nested_merge(a, b):
    for k, v in b.items():
        if k in a and isinstance(v, dict) and isinstance(a[k], dict):
            nested_merge(a[k], b[k])
        else:
            a[k] = b[k]


def dotdict2nestdict(data):
    out = {}
    for keystring, val in data.items():
        if isinstance(val, dict):
            val = dotdict2nestdict(val)
        d = reduce(lambda d, k: {k: d}, keystring.split(".")[::-1], val)
        nested_merge(out, d)
    return out


def nestdict2sn(d):
    out = SimpleSN()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(out, make_clean(k), nestdict2sn(v))
        else:
            setattr(out, make_clean(k), v)
    return out


def parse(json_dict):
    return nestdict2sn(dotdict2nestdict(json_dict))


if __name__ == "__main__":
    import json

    settings = {
        "diffEditor.codeLens": False,
        "diffEditor.ignoreTrimWhitespace": True,
        "diffEditor.maxComputationTime": 5000,
        "diffEditor.renderIndicators": True,
        "diffEditor.renderSideBySide": True,
        "diffEditor.wordWrap": "inherit",
        "editor.acceptSuggestionOnCommitCharacter": True,
        "editor.acceptSuggestionOnEnter": "on",
        "editor.rulers": [],
        "editor.scrollBeyondLastColumn": 5,
        "editor.scrollBeyondLastLine": True,
        "workbench.colorCustomizations": {
            "editor.background": "#000088",
            "editor.selectionBackground": "#008800",
            "Custom setup": 1,
            "3": 3,
        }
    }
    f = json.dumps(settings)
    config_json = json.loads(f)
    parsing_from_json = parse(config_json)
    parsing_from_dict = parse(settings)
