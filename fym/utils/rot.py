import numpy.linalg as nla
import scipy.linalg as sla
import numpy as np
from numpy import sin, cos


def quat2dcm(q):
    q = np.squeeze(q)
    q0, q1, q2, q3 = q / nla.norm(q)

    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
        [2*(q1*q2 - q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 + q0*q1)],
        [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
    ])


def dcm2quat(dcm):
    tr = dcm[0, 0]+dcm[1, 1]+dcm[2, 2]
    if tr > 0:
        q0 = 0.5 * np.sqrt(np.abs(1+tr))
        q1 = -(dcm[2, 1] - dcm[1, 2])/(4*q0)
        q2 = -(dcm[0, 2] - dcm[2, 0])/(4*q0)
        q3 = -(dcm[1, 0] - dcm[0, 1])/(4*q0)
    elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
        q1 = 0.5 * np.sqrt(1 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
        q0 = -(dcm[2, 1] - dcm[1, 2])/(4*q1)
        q2 = (dcm[1, 0] + dcm[0, 1])/(4*q1)
        q3 = (dcm[0, 2] + dcm[2, 0])/(4*q1)
    elif dcm[1, 1] > dcm[2, 2]:
        q2 = 0.5 * np.sqrt(1 - dcm[0, 0] + dcm[1, 1] - dcm[2, 2])
        q0 = -(dcm[0, 2] - dcm[2, 0])/(4*q2)
        q1 = (dcm[1, 0] + dcm[0, 1])/(4*q2)
        q3 = (dcm[2, 1] + dcm[1, 2])/(4*q2)
    else:
        q3 = 0.5 * np.sqrt(1 - dcm[0, 0] - dcm[1, 1] + dcm[2, 2])
        q0 = -(dcm[1, 0] - dcm[0, 1])/(4*q3)
        q1 = (dcm[0, 2] + dcm[2, 0])/(4*q3)
        q2 = (dcm[2, 1] + dcm[1, 2])/(4*q3)
    return np.vstack((q0, q1, q2, q3))


def angle2quat(yaw, pitch, roll):
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)

    q = np.zeros((4, 1))
    q[0, 0] = cy * cr * cp + sy * sr * sp
    q[1, 0] = cy * sr * cp - sy * cr * sp
    q[2, 0] = cy * cr * sp + sy * sr * cp
    q[3, 0] = sy * cr * cp - cy * sr * sp
    return q


def quat2angle(quat):
    """Output: yaw, pitch, roll"""
    qin = (quat / nla.norm(quat)).squeeze()

    r11 = 2 * (qin[1] * qin[2] + qin[0] * qin[3])
    r12 = qin[0]**2 + qin[1]**2 - qin[2]**2 - qin[3]**2
    r21 = - 2 * (qin[1] * qin[3] - qin[0] * qin[2])
    r31 = 2 * (qin[2] * qin[3] + qin[0] * qin[1])
    r32 = qin[0]**2 - qin[1]**2 - qin[2]**2 + qin[3]**2

    return np.arctan2(r11, r12), np.arcsin(r21), np.arctan2(r31, r32)


def angle2dcm(a1, a2, a3):
    return np.array([
        [cos(a2) * cos(a1), cos(a2) * sin(a1), - sin(a2)],
        [- cos(a3) * sin(a1) + sin(a3) * sin(a2) * cos(a1),
         cos(a3) * cos(a1) + sin(a3) * sin(a2) * sin(a1), sin(a3) * cos(a2)],
        [sin(a3) * sin(a1) + cos(a3) * sin(a2) * cos(a1),
         - sin(a3) * cos(a1) + cos(a3) * sin(a2) * sin(a1), cos(a3) * cos(a2)]
    ])


def velocity2polar(vel):
    norm = sla.norm(vel)
    chi = np.arctan2(vel[1], vel[0])
    gamma = np.arcsin(- vel[2] / norm)
    return np.array([norm, chi, gamma])
