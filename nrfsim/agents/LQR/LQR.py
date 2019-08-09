import numpy as np
import scipy.linalg as lin
#from nrfsim.~ import Linearization

class LQR():
    def __init__(self):
        pass

    def clqr(A: array, B: array, Q: array, R: array) -> array, array, array, array:
        X = lin.solve_continuous_are(A, B, Q, R)
        K = lin.inv(R)*(np.transpose(B).dot(X))
        eigVals, eigVecs = lin.eig(A-B.dot(K))
        return K, X, eigVals, eigVecs

    def dlqr(A: array, B: array, Q: array, R: array) -> array, array, array, array:
        X = lin.solve_discrete_are(A, B, Q, R)
        K = lin.inv(R)*(np.transpose(B).dot(X))
        eigVals, eivVecs = lin.eig(A-B.dot(K))
        return K, X, eigVals, eigVecs
'''
    def nlqr(f, g, Q: array, R: array) -> array, array, array, array:
        X = lin.solve_discrete_are(A, B, Q, R)
        K = lin.inv(R)*(np.transpose(B).dot(X))
        eigVals, eivVecs = lin.eig(A-B.dot(K))
        return K, X, eigVals, eigVecs        
'''
