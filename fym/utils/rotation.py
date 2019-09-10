"""
This module is a set of functions which are python version of matlab functions
``dcm2angle`` and ``angle2dcm``.
Reference:
    https://github.com/jimmyDunne/kinematicToolbox/blob/master/spatialMath/dcm2angle.m
"""
import numpy as np
from numpy import sin, cos


def three_axis_rotation(r11, r12, r21, r31, r32):
    r1 = np.arctan2(r11, r12)
    r2 = np.arcsin(r21)
    r3 = np.arctan2(r31, r32)
    return np.hstack((r1, r2, r3))


def dcm_to_angle(dcm, rtype='zyx'):
    """
    Parameters
    ----------
    dcm: array_like
        A DCM matrix with dimension (3, 3) or (N, 3, 3).

    rtype: string, optional
        'zyx' (default)

    Returns
    -------
    r1, r2, r3: ndarray
    """
    dcm = np.asarray(dcm)

    assert(dcm.shape[-2:] == (3, 3))

    if dcm.ndim == 2:
        dcm = dcm.reshape(1, 3, 3)

    if rtype == 'zyx':
        return three_axis_rotation(
            dcm[:, 0, 1], dcm[:, 0, 0], -dcm[:, 0, 2],
            dcm[:, 1, 2], dcm[:, 2, 2]
        )


def angle_to_dcm(r, rtype='ZYX'):
    if rtype == 'ZYX':
        return tx(r[2]).dot(ty(r[1])).dot(tz(r[0]))


def tx(a):
    return np.array([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]])


def ty(a):
    return np.array([[cos(a), 0, -sin(a)], [0, 1, 0], [sin(a), 0, cos(a)]])


def tz(a):
    return np.array([[cos(a), sin(a), 0], [-sin(a), cos(a), 0], [0, 0, 1]])


if __name__ == '__main__':
    angle = np.deg2rad([10, 20, 30])
    dcm = angle_to_dcm(angle)
    angle2 = dcm_to_angle(dcm)
    assert(np.all(angle == angle2))
