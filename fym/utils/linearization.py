import numdifftools as nd
import numpy as np


def jacob_analytic(function, i=0):
    """
    jacob_analytic is used for obtaining analytic jacobian function

    Parameters
    ----------
    funcion : callable
        ``function`` is what we want to get jacobian function.
    i : int or float
        ``i`` means i-th argument of ``function`` and
        what will we get Jacobian function for.
        for example,
        if you want to get Jacobian function of ``function`` for first argument
        of ``function``, set i=0.

    Return
    ------
    jacob_fnc : callable
        jacobian function of ``function``
    """
    def new_fun(*argument):
        argument = list(argument)
        argument[0], argument[i] = argument[i], argument[0]
        return function(*argument)
    jacob_temp = nd.Jacobian(new_fun)

    def jacob_fnc(*argument):
        argument = list(argument)
        argument[0], argument[i] = argument[i], argument[0]
        return jacob_temp(*argument)
    return jacob_fnc


def jacob_numerical(function, i, *args):
    """
    jacob_numerical is used for obtaining jacobian matrix numerically

    Parameters
    ----------
    funcion : callable
        ``function`` is what we want to get jacobian function.
    i : int or float
        ``i`` means i-th argument of ``function`` and
        what will we get Jacobian function for.
        for example,
        if you want to get Jacobian function of ``function`` for first argument
        of ``function``, set i=0.
    *args : int or float
        Values of ``function``'s arguments which we want to get Jacobian at.

    Return
    ------
    dfdx or dfdu: numpy.ndarray
        jacobian matrix of ``functions`` for ``x``, ``u``, respectively,
        at *args.
    """
    ptrb = 1e-9
    n = np.size(args[i])
    dx = function(*args)
    dfdx = np.zeros([n, n])
    args_save = args[i]
    for j in np.arange(n):
        args = list(args)
        ptrbvec = np.zeros(n)
        ptrbvec[j] = ptrb
        args[i] = args[i] + ptrbvec
        dx_ptrb = function(*args)
        dfdx[j] = (dx_ptrb - dx) / ptrb
        args[i] = args_save

    return np.transpose(dfdx)
