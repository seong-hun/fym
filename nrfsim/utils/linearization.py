import numdifftools as nd
import numpy as np

'''
-argument(input)
function: A function of wanting to get Jacobian function. This function has to have output as np.array
i: What will we get Jacobian function for? 0: first input of function(usually x), 1: second input of function(usually u)
-return(output)
Jacobian function
'''


def jacob_analytic(functions, i):
    if i == 0:
        jacob_fnc = nd.Jacobian(functions)
    else:
        def g(u, x, *e):
            return functions(x, u, *e)
        jacob_fnc_temp = nd.Jacobian(g)

        def jacob_fnc(x, u, *e):
            return jacob_fnc_temp(u, x, *e)
    return jacob_fnc


'''
-argument(input)
function: A function of wanting to get Jacobian function. This function has to have output as np.array
i: What will we get Jacobian function for? 0: first input of function(usually x), 1: second input of function(usually u)
x: state
*args: control input or external input of function. If the function has both, control input must come before external
-return(output)
Jacobian maxtrix
'''


def jacob_numerical(fnc, i, x, *args):
    ptrb = 1e-9
    if len(args) == 2:
        u = args[0]
        e = args[1]
        dx = fnc(x, u, e)
        if i == 0:
            n = np.size(x)
            dfdx = np.zeros([n, n])
            for j in np.arange(n):
                ptrbvec = np.zeros(n)
                ptrbvec[j] = ptrb
                dx_ptrb = fnc(x + ptrbvec, u, e)
                dfdx[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdx)
        else:
            m = np.size(u)
            dfdu = np.zeros([m, m])
            for j in np.arange(m):
                ptrbvec = np.zeros(m)
                ptrbvec[j] = ptrb
                dx_ptrb = fnc(x, u + ptrbvec, e)
                dfdu[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdu)
    elif len(args) == 1:
        u = args[0]
        dx = fnc(x, u)
        if i == 0:
            n = np.size(x)
            dfdx = np.zeros([n, n])
            for j in np.arange(n):
                ptrbvec = np.zeros(n)
                ptrbvec[j] = ptrb
                dx_ptrb = fnc(x + ptrbvec, u)
                dfdx[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdx)
        else:
            m = np.size(u)
            dfdu = np.zeros([m, m])
            for j in np.arange(m):
                ptrbvec = np.zeros(m)
                ptrbvec[j] = ptrb
                dx_ptrb = fnc(x, u + ptrbvec)
                dfdu[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdu)
    else:
        dx = fnc(x)
        n = np.size(x)
        dfdx = np.zeros([n, n])
        for j in np.arange(n):
            ptrbvec = np.zeros(n)
            ptrbvec[j] = ptrb
            dx_ptrb = fnc(x + ptrbvec)
            dfdx[j] = (dx_ptrb - dx) / ptrb
        return np.transpose(dfdx)
