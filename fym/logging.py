import h5py
import numpy as np
import os
from datetime import datetime


def save(h5file, dic):
    if not isinstance(h5file, h5py.File):
        if not os.path.exists(os.path.dirname(h5file)):
            os.makedirs(os.path.dirname(h5file), exist_ok=True)
        with h5py.File(h5file, 'w') as h5file:
            _rec_save(h5file, '/', dic)
    else:
        _rec_save(h5file, '/', dic)


def _rec_save(h5file, path, dic):
    """Recursively save the ``dic`` into the HDF5 file."""
    for key, val in dic.items():
        if isinstance(val, (np.ndarray, list)):
            val = np.stack(val)
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
            _rec_save(h5file, path + key + '/', val)
        else:
            raise ValueError(f'Cannot save {type(val)} type')


def load(path):
    with h5py.File(path, 'r') as h5file:
        return _rec_load(h5file, '/')


def _rec_load(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _rec_load(h5file, path + key + '/')
    return ans


def _rec_update(base_dict, input_dict):
    """Recursively update ``base_dict`` with ``input_dict``."""
    for key, val in input_dict.items():
        if isinstance(val, dict):
            _rec_update(base_dict.setdefault(key, {}), val)
        elif not isinstance(val, str):
            base_dict.setdefault(key, []).append(val)
        else:
            raise ValueError("Unsupported data type")


class Logger:
    def __init__(self, log_dir=None, file_name='data.h5', max_len=1e2):
        if log_dir is None:
            log_dir = os.path.join(
                'log',
                datetime.today().strftime('%Y%m%d-%H%M%S')
            )

        os.makedirs(log_dir, exist_ok=True)
        self.basename = file_name
        self.path = os.path.join(log_dir, file_name)
        self.max_len = max_len
        self.h5file = h5py.File(self.path, 'w')
        self.clear()

    def clear(self):
        self.buffer = {}
        self.len = 0

    def record(self, **kwargs):
        """Record a dictionary or a numeric data preserving the structure."""
        _rec_update(self.buffer, kwargs)
        self.len += 1

        if self.len >= self.max_len:
            self.flush()

    def flush(self):
        save(self.h5file, self.buffer)
        self.clear()

    def close(self):
        self.flush()
        self.h5file.close()


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
