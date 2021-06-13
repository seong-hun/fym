import pickle
import h5py
import numpy as np
import os
from datetime import datetime
from .utils import parser


class Logger:
    def __init__(self, path=None, log_dir=None, file_name="data.h5",
                 max_len=1e3, mode="w"):
        if path is None:
            if log_dir is None:
                log_dir = os.path.join(
                    'log', datetime.today().strftime('%Y%m%d-%H%M%S'))
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, file_name)

        self.path = path
        with h5py.File(self.path, mode):
            pass
        self.mode = mode
        self.max_len = int(max_len)
        self._info = {}

        self.clear()

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
        self.buffer = {}
        self.index = 0

    def record(self, **kwargs):
        """Record a dictionary or a numeric data preserving the structure."""
        self._rec_update(self.buffer, kwargs)
        self.index += 1

        if self.index >= self.max_len:
            self.flush()

    def flush(self):
        with h5py.File(self.path, "r+") as h5file:
            _rec_save(h5file, '/', self.buffer, self.index)
        self.clear()

    def close(self):
        self.flush()
        with h5py.File(self.path, "r+") as h5file:
            _info_save(h5file, self._info)

    def set_info(self, **kwargs):
        parser.update(self._info, kwargs)
        with h5py.File(self.path, "r+") as h5file:
            _info_save(h5file, self._info)

    def _rec_update(self, base_dict, input_dict):
        """Recursively update ``base_dict`` with ``input_dict``."""
        for key, val in input_dict.items():
            if isinstance(val, dict):
                if key not in base_dict:
                    base_dict[key] = {}
                self._rec_update(base_dict[key], val)
            elif not isinstance(val, str):
                if key not in base_dict:
                    base_dict[key] = np.empty((self.max_len,) + np.shape(val))
                base_dict[key][self.index] = np.copy(val)
            else:
                raise ValueError("Unsupported data type")


def save(h5file, dic, mode="w", info=None):
    if not isinstance(h5file, h5py.File):
        if isinstance(h5file, str):
            dirname = os.path.dirname(h5file)

            if dirname:
                os.makedirs(dirname, exist_ok=True)

            with h5py.File(h5file, mode) as h5file:
                _rec_save(h5file, '/', dic)
                _info_save(h5file, info)
        else:
            raise ValueError(f'Cannot save into {type(h5file)} type')
    else:
        _rec_save(h5file, '/', dic)
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
                val = val[:index]
            if path + key not in h5file:
                dset = h5file.create_dataset(
                    path + key,
                    shape=val.shape,
                    maxshape=(None,) + val.shape[1:],
                    dtype=float,
                )
                dset[:] = val
            else:
                dset = h5file[path + key]
                dset.resize(dset.shape[0] + len(val), axis=0)
                dset[-len(val):] = val
        elif isinstance(val, dict):
            _rec_save(h5file, path + key + '/', val, index)
        else:
            raise ValueError(f'Cannot save {type(val)} type')


def load(path, with_info=False):
    with h5py.File(path, 'r') as h5file:
        ans = _rec_load(h5file, '/')
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
            ans[key] = _rec_load(h5file, path + key + '/')
    return ans


if __name__ == '__main__':
    logger = Logger('sample', max_len=5)

    data = {
        'state': {
            'main_system': np.array([0., 0., 0., 0.]),
            'reference_system': np.array([1., 1., 1., 1.]),
            'adaptive_system': np.array([
                [2., 2.],
                [2., 2.],
                [2., 2.],
                [2., 2.],
                [2., 2.]])
        },
        'action': {
            'M': np.array([
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.]]),
            'N': np.array([
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.]])
        }
    }

    for _ in range(10):
        logger.log_dict(**data)

    logger.close()
