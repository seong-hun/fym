import numpy as np
import scipy.linalg as lin
import numpy.linalg as nla
#from nrfsim.~ import Linearization

class LQR():
    def __init__(self):
        pass

    def clqr(self, A: np.array, B: np.array, Q: np.array, R: np.array) \
        -> np.array:
        X = lin.solve_continuous_are(A, B, Q, R)
        if np.size(R) == 1:
            K = (np.transpose(B).dot(X))/R
        else:
            K = nla.inv(R).dot((np.transpose(B).dot(X)))
        eig_vals, eig_vecs = nla.eig(A-B.dot(K))
        return K, X, eig_vals, eig_vecs

    def dlqr(self, A: np.array, B: np.array, Q: np.array, R: np.array) \
        -> np.array:
        X = lin.solve_discrete_are(A, B, Q, R)
        if np.size(R) == 1:
            K = (np.transpose(B).dot(X))/R
        else:
            K = nla.inv(R).dot((np.transpose(B).dot(X)))
        eig_vals, eig_vecs = nla.eig(A-B.dot(K))
        return K, X, eig_vals, eig_vecs

'''
    def nlqr(f, g, Q: array, R: array) -> array, array, array, array:
        X = lin.solve_discrete_are(A, B, Q, R)
        K = lin.inv(R)*(np.transpose(B).dot(X))
        eigVals, eivVecs = lin.eig(A-B.dot(K))
        return K, X, eigVals, eigVecs        
'''
