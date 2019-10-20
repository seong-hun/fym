import h5py
import numpy as np
import os
from datetime import datetime


def save_dict_to_hdf5(h5file, dic):
    recursively_save(h5file, '/', dic)


def recursively_save(h5file, path, dic):
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
            recursively_save(h5file, path + key + '/', val)
        else:
            raise ValueError(f'Cannot save {type(val)} type')


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans


def recursively_update_dict(base_dict, input_dict):
    for key, val in input_dict.items():
        if isinstance(val, dict):
            recursively_update_dict(base_dict.setdefault(key, {}), val)
        elif not isinstance(val, str):
            base_dict.setdefault(key, []).append(val)
        else:
            raise ValueError("Unsupported data type")


class Logger:
    def __init__(self, file_name, max_len=1e2, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join(
                'log',
                datetime.today().strftime('%Y%m%d-%H%M%S')
            )

        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, file_name)

        self.f = h5py.File(self.path, 'w')
        self.max_len = max_len
        self.reset()

    def reset(self):
        self.buffer = {}
        self.len = 0

    def flush(self):
        save_dict_to_hdf5(self.f, self.buffer)
        self.reset()

    def log_dict(self, **kwargs):
        recursively_update_dict(self.buffer, kwargs)
        self.len += 1

        if self.len >= self.max_len:
            self.flush()

    def close(self):
        self.flush()
        self.f.close()


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
