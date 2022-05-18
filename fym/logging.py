import os
import pickle

import h5py
import numpy as np


class Logger:
    def __init__(self, path=None, max_len=1e3, mode="flush"):
        assert mode in ["flush", "deque", "stop"]

        if path is None:
            assert mode != "flush"
        else:
            # To overwite if a file of the same path exists
            self.path = path
            with h5py.File(self.path, "w"):
                pass

        self.max_len = int(max_len)
        self.mode = mode
        self._info = {}
        self._inner = False
        self.clear()

    def len(self):
        return self._ind + 1

    def __len__(self):
        return self.len()

    @property
    def buffer(self):
        return self._rec_get(self._buf, self._ind)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if path is not None:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

        self._path = path

    def clear(self):
        self._buf = {}
        self._ind = -1

    def record(self, **kw):
        assert not self._inner, "Inner loggers are not allowed to record directly"
        self._record(**kw)

    def _record(self, **kw):
        """Record a dictionary or a numeric data preserving the structure."""

        if self.len() == self.max_len:  # index reaches the last one
            if self.mode == "flush":
                self.flush()  # make ``_ind`` 0 and ``_full`` False
                self._rec_update(self._buf, kw, self._ind + 1)
                self._ind += 1
            elif self.mode == "rotate":
                self._rec_update(self._buf, kw, self._ind, rotate=True)
            else:  # mode == "stop"
                pass
        else:
            self._rec_update(self._buf, kw, self._ind + 1)
            self._ind += 1

    def flush(self):
        with h5py.File(self.path, "r+") as h5file:
            _rec_save(h5file, "/", self._buf, self._ind)
        self.clear()

    def close(self):
        self.flush()
        with h5py.File(self.path, "r+") as h5file:
            _info_save(h5file, self._info)

    def reopen(self):
        with h5py.File(self.path, "r+"):
            pass

    def set_info(self, **kw):
        self._info.update(kw)
        with h5py.File(self.path, "r+") as h5file:
            _info_save(h5file, self._info)

    def _rec_update(self, base_dict, input_dict, index, rotate=False):
        """Recursively update ``base_dict`` with ``input_dict``.

        The ``input_dict`` will be added in the ``base_dict[...][index]``.
        """
        for key, val in input_dict.items():
            if isinstance(val, dict):
                if key not in base_dict:
                    base_dict[key] = {}
                self._rec_update(base_dict[key], val, index)
            elif not isinstance(val, str):
                val = np.asarray(val)
                if key not in base_dict:
                    base_dict[key] = np.empty(
                        (self.max_len,) + val.shape,
                        dtype=val.dtype,
                    )

                if rotate:
                    base_dict[key][:-1] = base_dict[key][1:]

                base_dict[key][index] = np.copy(val)
            else:
                raise ValueError("Unsupported data type")

    def _rec_get(self, base_dict, index):
        out_dict = {}
        for key, val in base_dict.items():
            if isinstance(val, dict):
                out_dict[key] = self._rec_get(val, index)
            elif not isinstance(val, str):
                out_dict[key] = val[: index + 1]
            else:
                raise ValueError("Unsupported data type")
        return out_dict


def save(h5file, dic, mode="w", info=None):
    if not isinstance(h5file, h5py.File):
        try:
            dirname = os.path.dirname(h5file)

            if dirname:
                os.makedirs(dirname, exist_ok=True)

            with h5py.File(h5file, mode) as h5file:
                _rec_save(h5file, "/", dic)
                _info_save(h5file, info)
        except ValueError:
            raise ValueError(f"Cannot save into {type(h5file)} type")
    else:
        _rec_save(h5file, "/", dic)
        _info_save(h5file, info)


def _info_save(h5file, info=None):
    if info is not None:
        ser = pickle.dumps(info)
        h5file.attrs.update(_info=np.void(ser))


def _rec_save(h5file, path, dic, index=None):
    """Recursively save the ``dic`` into the HDF5 file."""
    for key, val in dic.items():
        if isinstance(val, (np.ndarray, list)):
            if isinstance(val, list):
                val = np.stack(val)
            if index is not None:
                val = val[: index + 1]
            if path + key not in h5file:
                dset = h5file.create_dataset(
                    path + key,
                    shape=val.shape,
                    maxshape=(None,) + val.shape[1:],
                    dtype=val.dtype,
                )
                dset[:] = val
            else:
                dset = h5file[path + key]
                dset.resize(dset.shape[0] + len(val), axis=0)
                dset[-len(val) :] = val
        elif isinstance(val, dict):
            _rec_save(h5file, path + key + "/", val, index)
        else:
            raise ValueError(f"Cannot save {type(val)} type")


def load(path, with_info=False):
    with h5py.File(path, "r") as h5file:
        ans = _rec_load(h5file, "/")
        if with_info:
            return ans, pickle.loads(h5file.attrs["_info"].tostring())
        else:
            return ans


def _rec_load(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _rec_load(h5file, path + key + "/")
    return ans


if __name__ == "__main__":
    logger = Logger("sample", max_len=5)

    data = {
        "state": {
            "main_system": np.array([0.0, 0.0, 0.0, 0.0]),
            "reference_system": np.array([1.0, 1.0, 1.0, 1.0]),
            "adaptive_system": np.array(
                [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]
            ),
        },
        "action": {
            "M": np.array(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ),
            "N": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        },
    }

    for _ in range(10):
        logger.log_dict(**data)

    logger.close()
