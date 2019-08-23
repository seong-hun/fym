import numpy as np
import scipy.linalg as lin
import numpy.linalg as nla


# from nrfsim.~ import Linearization


class LQR:
    def __init__(self):
        pass

    def clqr(self, a: np.array, b: np.array, q: np.array, r: np.array) \
            -> np.array:
        x = lin.solve_continuous_are(a, b, q, r)
        if np.size(r) == 1:
            k = (np.transpose(b).dot(x)) / r
        else:
            k = nla.inv(r).dot((np.transpose(b).dot(x)))
        eig_vals, eig_vecs = nla.eig(a - b.dot(k))
        return k, x, eig_vals, eig_vecs

    def dlqr(self, a: np.array, b: np.array, q: np.array, r: np.array) \
            -> np.array:
        x = lin.solve_discrete_are(a, b, q, r)
        if np.size(r) == 1:
            k = (np.transpose(b).dot(x)) / r
        else:
            k = nla.inv(r).dot((np.transpose(b).dot(x)))
        eig_vals, eig_vecs = nla.eig(a - b.dot(k))
        return k, x, eig_vals, eig_vecs


'''
    def nlqr(f, g, Q: array, R: array) -> array, array, array, array:
        X = lin.solve_discrete_are(A, B, Q, R)
        K = lin.inv(R)*(np.transpose(B).dot(X))
        eigVals, eivVecs = lin.eig(A-B.dot(K))
        return K, X, eigVals, eigVecs        
'''
