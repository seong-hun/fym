import numpy as np
import numpy.linalg as nla
import scipy.linalg as lin


def clqr(
    A: np.array, B: np.array, Q: np.array, R: np.array, with_eigs=False
) -> np.array:
    P = lin.solve_continuous_are(A, B, Q, R)
    if np.size(R) == 1:
        K = (np.transpose(B).dot(P)) / R
    else:
        K = nla.inv(R).dot((np.transpose(B).dot(P)))

    eig_vals, eig_vecs = nla.eig(A - B.dot(K))

    if with_eigs:
        return K, P, eig_vals, eig_vecs
    else:
        return K, P


def dlqr(
    A: np.array, B: np.array, Q: np.array, R: np.array, with_eigs=False
) -> np.array:
    P = lin.solve_discrete_are(A, B, Q, R)
    if np.size(R) == 1:
        K = (np.transpose(B).dot(P)) / R
    else:
        K = nla.inv(R).dot((np.transpose(B).dot(P)))

    eig_vals, eig_vecs = nla.eig(A - B.dot(K))

    if with_eigs:
        return K, P, eig_vals, eig_vecs
    else:
        return K, P
