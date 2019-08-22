import numpy as np
from nrfsim.agents.LQR import clqr, dlqr

A = np.array([[1, 2], [2, -1]])
B = np.array([[1], [2]])
Q = np.eye(2)
R = 1


K_con, X_con, eig_vals_con, eig_vecs_con = clqr(A, B, Q, R)
K_dis, X_dis, eig_vals_dis, eig_vecs_dis = dlqr(A, B, Q, R)
