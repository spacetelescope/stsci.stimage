# This module contains functions to do general non-linear
# least squares fits.
#
# Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 2002-11-18
#
# This file was taken almost intact from the ScientificPython
# library by Karl Hinsen. The exception class defined in the
# Scientific __init__ file was copied into this file for local
# use, and a slightly crippled version of the FirstDerivatives
# is imported instead of the original, to avoid further dependence
# on the rest of the library.
# .......... V. G. Laidler, 11 Dec 2003

import Numeric, LinearAlgebra
LA = LinearAlgebra
from SP_FirstDerivatives import DerivVar
#from FirstDerivatives import DerivVar
#from Scientific import IterationCountExceededError

#
# New exception class
#
class IterationCountExceededError(ValueError):

    pass


def _chiSquare(model, parameters, data):
    n_param = len(parameters)
    chi_sq = 0.
    alpha = Numeric.zeros((n_param, n_param))
    for point in data:
	sigma = 1
	if len(point) == 3:
	    sigma = point[2]
	f = model(parameters, point[0])
	chi_sq = chi_sq + ((f-point[1])/sigma)**2
	d = Numeric.array(f[1])/sigma
	alpha = alpha + d[:,Numeric.NewAxis]*d
    return chi_sq, alpha

def leastSquaresFit(model, parameters, data, max_iterations=None):
    """General non-linear least-squares fit using the
    Levenberg-Marquardt algorithm and automatic derivatives.

    The parameter |model| specifies the function to be fitted. It will be
    called with two parameters: the first is a tuple containing all fit
    parameters, and the second is the first element of a data point (see
    below). The return value must be a number.  Since automatic
    differentiation is used to obtain the derivatives with respect to the
    parameters, the function may only use the mathematical functions known
    to the module FirstDerivatives.

    The parameter |parameter| is a tuple of initial values for the
    fit parameters.

    The parameter |data| is a list of data points to which the model
    is to be fitted. Each data point is a tuple of length two or
    three. Its first element specifies the independent variables
    of the model. It is passed to the model function as its first
    parameter, but not used in any other way. The second element
    of each data point tuple is the number that the return value
    of the model function is supposed to match as well as possible.
    The third element (which defaults to 1.) is the statistical
    variance of the data point, i.e. the inverse of its statistical
    weight in the fitting procedure.

    The function returns a list containing the optimal parameter values
    and the chi-squared value describing the quality of the fit.
    """
    n_param = len(parameters)
    p = ()
    i = 0
    for param in parameters:
	p = p + (DerivVar(param, i),)
	i = i + 1
    id = Numeric.identity(n_param)
    l = 0.001
    chi_sq, alpha = _chiSquare(model, p, data)
    niter = 0
    while 1:
	delta = LA.solve_linear_equations(alpha+l*Numeric.diagonal(alpha)*id,
					  -0.5*Numeric.array(chi_sq[1]))
	next_p = map(lambda a,b: a+b, p, delta)
	next_chi_sq, next_alpha = _chiSquare(model, next_p, data)
	if next_chi_sq > chi_sq:
	    l = 10.*l
	else:
	    l = 0.1*l
	    if chi_sq[0] - next_chi_sq[0] < 0.005: break
	    p = next_p
	    chi_sq = next_chi_sq
	    alpha = next_alpha
        niter = niter + 1
        if max_iterations is not None and niter == max_iterations:
            raise IterationCountExceededError
    return map(lambda p: p[0], next_p), next_chi_sq[0]

#
# The important special case of n-th order polynomial fits
# was contributed by David Ascher:
#
def polynomialModel(params, t):
    r = 0.0
    for i in range(len(params)):
        r = r + params[i]*Numeric.power(t, i)
    return r

def polynomialLeastSquaresFit(parameters, data):
    """Least-squares fit to a polynomial whose order is defined by
    the number of parameter values."""
    return leastSquaresFit(polynomialModel, parameters, data)


# Test code

if __name__ == '__main__':

    from Numeric import exp

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
