import numdifftools as nd
import numpy as np
#from inspect import signature


def jacob_analytic(functions, i):
    """
    jacob_analytic is used for obtaining analytic jacobian function

    Parameters
    ----------
    funcions : callable
        ``functions`` is what we want to get jacobian function.
        ``functions`` function that takes at least one positional arguments and
        at most three arguments including positional, arbitrary, keyword args.
        The order of arguments should be
        'state', *'control input', *'the other inputs', *'time'.

        -``x``: state (`float` or `int`). It must be taken
        -``u``: control input (`float` or `int`). arbitraty argumnets
        -``e``: external input (`float` or `int`). arbitrary argumnets
        -``t``: time (`float`). arbitrary arguments
    i : int or float
        ``i`` means what will we get Jacobian function for.
        0: first arguments of ``functions``(usually x)
        1: second argument of ``functions``(usually u)

    Return
    ------
    jacob_fnc : callable
        jacobian function of ``functions``
    """
    if i == 0:
        jacob_fnc = nd.Jacobian(functions)
    else:
        def g(u, x, *e):
            return functions(x, u, *e)
        jacob_fnc_temp = nd.Jacobian(g)

        def jacob_fnc(x, u, *e):
            return jacob_fnc_temp(u, x, *e)
    return jacob_fnc


def jacob_numerical(functions, i, x, *args):
    """
    jacob_numerical is used for obtaining jacobian matrix numerically

    Parameters
    ----------
    funcions : callable
        ``functions`` is what we want to get jacobian function.
        ``functions`` function that takes at least one positional arguments and
        at most three arguments including positional, arbitrary, keyword args.
        The order of arguments should be
        'state', *'control input', *'external input', *'time'.

        -``x``: state (`float` or `int`). It must be taken
        -``u``: control input (`float` or `int`). arbitrary argumnets
        -``e``: external input (`float` or `int`). arbitrary argumnets
        -``t``: time. arbitrary arguments
    i : int or float
        ``i`` means what will we get Jacobian function for.
        0: first arguments of ``functions``(usually ``x``)
        1: second argument of ``functions``(usually ``u``)
    x : int or float
        ``x`` is state where we want to get jacobian matrix of ``functions``.
    *args : int or float
        ``*args`` can be 'control input' or 'external input' or both of them.
        If both 'control input' and 'external input' are included,
        'control input' must come befor 'exernal input'.

    Return
    ------
    dfdx or dfdu: numpy.ndarray
        jacobian matrix of ``functions`` for ``x``, ``u`` respectively.
    """
    ptrb = 1e-9
    if len(args) == 3:
        u = args[0]
        e = args[1]
        t = args[2]
        dx = functions(x, u, e, t)
        if i == 0:
            n = np.size(x)
            dfdx = np.zeros([n, n])
            for j in np.arange(n):
                ptrbvec = np.zeros(n)
                ptrbvec[j] = ptrb
                dx_ptrb = functions(x + ptrbvec, u, e, t)
                dfdx[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdx)
        else:
            m = np.size(u)
            dfdu = np.zeros([m, m])
            for j in np.arange(m):
                ptrbvec = np.zeros(m)
                ptrbvec[j] = ptrb
                dx_ptrb = functions(x, u + ptrbvec, e, t)
                dfdu[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdu)
    elif len(args) == 2:
        u = args[0]
        e = args[1]
        dx = functions(x, u, e)
        if i == 0:
            n = np.size(x)
            dfdx = np.zeros([n, n])
            for j in np.arange(n):
                ptrbvec = np.zeros(n)
                ptrbvec[j] = ptrb
                dx_ptrb = functions(x + ptrbvec, u, e)
                dfdx[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdx)
        else:
            m = np.size(u)
            dfdu = np.zeros([m, m])
            for j in np.arange(m):
                ptrbvec = np.zeros(m)
                ptrbvec[j] = ptrb
                dx_ptrb = functions(x, u + ptrbvec, e)
                dfdu[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdu)
    elif len(args) == 1:
        u = args[0]
        dx = functions(x, u)
        if i == 0:
            n = np.size(x)
            dfdx = np.zeros([n, n])
            for j in np.arange(n):
                ptrbvec = np.zeros(n)
                ptrbvec[j] = ptrb
                dx_ptrb = functions(x + ptrbvec, u)
                dfdx[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdx)
        else:
            m = np.size(u)
            dfdu = np.zeros([m, m])
            for j in np.arange(m):
                ptrbvec = np.zeros(m)
                ptrbvec[j] = ptrb
                dx_ptrb = functions(x, u + ptrbvec)
                dfdu[j] = (dx_ptrb - dx) / ptrb
            return np.transpose(dfdu)
    else:
        dx = functions(x)
        n = np.size(x)
        dfdx = np.zeros([n, n])
        for j in np.arange(n):
            ptrbvec = np.zeros(n)
            ptrbvec[j] = ptrb
            dx_ptrb = functions(x + ptrbvec)
            dfdx[j] = (dx_ptrb - dx) / ptrb
        return np.transpose(dfdx)
