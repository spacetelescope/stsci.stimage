# This module contains functions to do general non-linear
# least squares fits.
#
# Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 2006-11-23
#
# Modified import statements to use just a portion of the package
# and drop support for non-numpy array packages. Also added an
# additional return to the leastSquaresFit routine.
# V. Laidler, 2007-02-05.

"""
Non-linear least squares fitting

Examples
--------
Usage example::

    from Scientific.N import exp

    def f(param, t):
        return param[0]*exp(-param[1]/t)

    data_quantum = [(100, 3.445e+6),(200, 2.744e+7),
                    (300, 2.592e+8),(400, 1.600e+9)]
    data_classical = [(100, 4.999e-8),(200, 5.307e+2),
                      (300, 1.289e+6),(400, 6.559e+7)]

    print leastSquaresFit(f, (1e13,4700), data_classical)

    def f2(param, t):
        return 1e13*exp(-param[0]/t)

    print leastSquaresFit(f2, (3000.,), data_quantum)
"""
#------------------------------------------------------
#Original import statements
#------------------------------------------------------
#from Scientific import N, LA
#from FirstDerivatives import DerivVar
#from Scientific import IterationCountExceededError
#------------------------------------------------------
from __future__ import division
import SP_numpy as N
import numpy.oldnumeric.linear_algebra as LA
from SP_FirstDerivatives import DerivVar
class IterationCountExceededError(ValueError):
    pass


def _chiSquare(model, parameters, data):
    n_param = len(parameters)
    chi_sq = 0.
    alpha = N.zeros((n_param, n_param))
    for point in data:
        sigma = 1
        if len(point) == 3:
            sigma = point[2]
        f = model(parameters, point[0])
        chi_sq = chi_sq + ((f-point[1])/sigma)**2
        d = N.array(f[1])/sigma
        alpha = alpha + d[:,N.NewAxis]*d
    return chi_sq, alpha

def leastSquaresFit(model, parameters, data, max_iterations=None,
                    stopping_limit = 0.005):
    """General non-linear least-squares fit using the
    X{Levenberg-Marquardt} algorithm and X{automatic differentiation}.

    model : function
        the function to be fitted. It will be called
        with two parameters: the first is a tuple containing all fit
        parameters, and the second is the first element of a data point (see
        below). The return value must be a number.  Since automatic
        differentiation is used to obtain the derivatives with respect to the
        parameters, the function may only use the mathematical functions known
        to the module FirstDerivatives.
    parameters : tuple of numbers
        a tuple of initial values for the
        fit parameters

    data : list
        a list of data points to which the model
        is to be fitted. Each data point is a tuple of length two or
        three. Its first element specifies the independent variables
        of the model. It is passed to the model function as its first
        parameter, but not used in any other way. The second element
        of each data point tuple is the number that the return value
        of the model function is supposed to match as well as possible.
        The third element (which defaults to 1.) is the statistical
        variance of the data point, i.e. the inverse of its statistical
        weight in the fitting procedure.

    Returns
    --------
    fitlist : list
        a list containing the optimal parameter values
    chisq : float
        chi-squared value describing the quality of the fit

    """
    n_param = len(parameters)
    p = ()
    i = 0
    for param in parameters:
        p = p + (DerivVar(param, i),)
        i = i + 1
    id = N.identity(n_param)
    l = 0.001
    chi_sq, alpha = _chiSquare(model, p, data)
    niter = 0
    itertrace=[p]
    while 1:
        delta = LA.solve_linear_equations(alpha+l*N.diagonal(alpha)*id,
                                          -0.5*N.array(chi_sq[1]))
        next_p = map(lambda a,b: a+b, p, delta)
        next_chi_sq, next_alpha = _chiSquare(model, next_p, data)
        if next_chi_sq > chi_sq:
            l = 10.*l
        else:
            l = 0.1*l
            if chi_sq[0] - next_chi_sq[0] < stopping_limit: break
            p = next_p
            chi_sq = next_chi_sq
            alpha = next_alpha
        niter = niter + 1
        if max_iterations is not None and niter == max_iterations:
            raise IterationCountExceededError
    return [p[0] for p in next_p], next_chi_sq[0], itertrace

#
# The special case of n-th order polynomial fits
# was contributed by David Ascher. Note: this could also be
# done with linear least squares, e.g. from LinearAlgebra.
#
def _polynomialModel(params, t):
    r = 0.0
    for i in range(len(params)):
        r = r + params[i]*N.power(t, i)
    return r

def polynomialLeastSquaresFit(parameters, data):
    """
    Least-squares fit to a polynomial whose order is defined by
    the number of parameter values.

    .. note::

        This could also be done with a linear least squares fit
        from L{LinearAlgebra}

    Parameters
    ----------
    parameters : tuple
        a tuple of initial values for the polynomial
        coefficients
    data : list
        the data points, as for L{leastSquaresFit}

    """
    return leastSquaresFit(_polynomialModel, parameters, data)


# Test code

if __name__ == '__main__':

    from Scientific.N import exp

    def f(param, t):
        return param[0]*exp(-param[1]/t)

    data_quantum = [(100, 3.445e+6),(200, 2.744e+7),
                    (300, 2.592e+8),(400, 1.600e+9)]
    data_classical = [(100, 4.999e-8),(200, 5.307e+2),
                      (300, 1.289e+6),(400, 6.559e+7)]

    print leastSquaresFit(f, (1e13,4700), data_classical)

    def f2(param, t):
        return 1e13*exp(-param[0]/t)

    print leastSquaresFit(f2, (3000.,), data_quantum)
