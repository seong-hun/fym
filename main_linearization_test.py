'''
function f: function f has stats, control input, external input as positional argument
function h: function h has stats, control inputas positional argument
function g: function g has stats as positional argument
function ext: function ext has stats, external input as positional argument
'''


from nrfsim.utils.linearization import jacob_analytic
from nrfsim.utils.linearization import jacob_numerical
import numpy as np

x = np.array([2, 2])
u = np.array([1, 1])
e = 3


def f(state, input, external):
    x1 = state[0]
    x2 = state[1]
    u1 = input[0]
    u2 = input[1]
    f1 = (x1 ** 2) * u1 + (x2 ** 3) * u2 * external
    f2 = x1 + external * u2 - x2 ** 2
    return np.array([f1, f2])


dfdx = jacob_analytic(f, 0)
A_f = dfdx(x, u, e)
print('A_f =', A_f)

A_f_n = jacob_numerical(f, 0, x, u, e)
print('A_f_num =', A_f_n)

dfdu = jacob_analytic(f, 1)
B_f = dfdu(x, u, e)
print('B_f =', B_f)

B_f_n = jacob_numerical(f, 1, x, u, e)
print('B_f_num =', B_f_n)


def h(state, input):
    x1 = state[0]
    x2 = state[1]
    u1 = input[0]
    u2 = input[1]
    f1 = (x1 ** 2) * u1 + (x2 ** 3) * u2
    f2 = x1 + u2 - x2 ** 2
    return np.array([f1, f2])


dhdx = jacob_analytic(h, 0)
A_h = dhdx(x, u)
print('A_h =', A_h)

A_h_n = jacob_numerical(h, 0, x, u)
print('A_h_num =', A_h_n)

dhdu = jacob_analytic(h, 1)
B_h = dhdu(x, u)
print('B_h =', B_h)

B_h_n = jacob_numerical(h, 1, x, u)
print('B_h_num =', B_h_n)


def g(state):
    k1 = state[0]
    k2 = state[1]
    f1 = k1**2 - k2**3
    f2 = k1
    f = np.array([f1, f2])
    return f


dgdx = jacob_analytic(g, 0)
A_g = dgdx(x)
print('A_g =', A_g)

A_g_n = jacob_numerical(g, 0, x)
print('A_g_num =', A_g_n)


def ext(state, external):
    k1 = state[0]
    k2 = state[1]
    f1 = k1 ** 2 - k2 ** 3
    f2 = k1*external
    return np.array([f1, f2])


dextdx = jacob_analytic(ext, 0)
A_ext = dextdx(x, e)
print('A_ext =', A_ext)

A_ext_n = jacob_numerical(ext, 0, x, e)
print('A_ext_n =', A_ext_n)

