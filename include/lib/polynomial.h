/*
Copyright (C) 2008-2010 Association of Universities for Research in Astronomy (AURA)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of AURA and its representatives may not be used to
      endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef _STIMAGE_POLYNOMIAL_H_
#define _STIMAGE_POLYNOMIAL_H_

#include "lib/util.h"

/* was tgs_1devpoly */

/**
Evaluate a 1D polynomial

@param order Order of the polynomial, 1 = constant

@param coeff EV array of coefficients

@param ncoord Number of points to be evaluated

@param axis The axis number to use (0 = x, 1 = y)

@param ref The reference points (length ncoord)

@param zfit The fitted values (length ncoord)

@param error

@return non-zero on failure
 */
int
eval_1dpoly(
        const int order,
        const double* const coeff,
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        /* Output */
        double* const zfit,
        stimage_error_t* const error);

/* was tgs_1devcheb */

/**
Evaluate a 1D Chebyshev polynomial, assuming that the coefficients
have been calculated.

@param order Order of the polynomial, 1 = constant

@param coeff EV array of coefficients

@param ncoord Number of points to be evaluated

@param axis The axis number to use (0 = x, 1 = y)

@param ref The reference points (length ncoord)

@param k1 Normalizing constant

@param k2 Normalizing constant

@param zfit The fitted values (length ncoord)

@param error

@return non-zero on failure
 */
int
eval_1dchebyshev(
        const int order,
        const double* const coeff,
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const double k1,
        const double k2,
        /* Output */
        double* const zfit,
        stimage_error_t* const error);

/**
Evaluate a 1D Legendre polynomial, assuming that the coefficients
have been calculated.

@param order Order of the polynomial, 1 = constant

@param coeff EV array of coefficients

@param ncoord Number of points to be evaluated

@param axis The axis number to use (0 = x, 1 = y)

@param ref The reference points (length ncoord)

@param k1 Normalizing constant

@param k2 Normalizing constant

@param zfit The fitted values (length ncoord)

@param error

@return non-zero on failure
 */
int
eval_1dlegendre(
        const int order,
        const double* const coeff,
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const double k1,
        const double k2,
        /* Output */
        double* const zfit,
        stimage_error_t* const error);

/* was tgs_evpoly */

/**
Evaluate a polynomial.

@param Order of the polynomial in x

@param Order of the polynomial in y

@param coeff 1D array of coefficients

@param ncoord Number of points to be evaluated

@param ref Reference points to be evaluated

@param xterms Type of cross terms

@param k1x Normalizing constant

@param k2x Normalizing constant

@param k1y Normalizing constant

@param k2y Normalizing constant

@param zfit The fitted points

@param error

@return non-zero on failure
 */
int
eval_poly(
        const int xorder,
        const int yorder,
        const double* const coeff,
        const size_t ncoord,
        const coord_t* const ref,
        const xterms_e xterms,
        const double k1x,
        const double k2x,
        const double k1y,
        const double k2y,
        /* Output */
        double* const zfit,
        stimage_error_t* const error);

/* was tgs_evcheb */

/**
Evaluate a Chebyshev polynomial, assuming that the coefficients have
been calculated.

@param Order of the polynomial in x

@param Order of the polynomial in y

@param coeff 1D array of coefficients

@param ncoord Number of points to be evaluated

@param ref Reference points to be evaluated

@param xterms Type of cross terms

@param k1x Normalizing constant

@param k2x Normalizing constant

@param k1y Normalizing constant

@param k2y Normalizing constant

@param zfit The fitted points

@param error

@return non-zero on failure
 */
int
eval_chebyshev(
        const int xorder,
        const int yorder,
        const double* const coeff,
        const size_t ncoord,
        const coord_t* const ref,
        const xterms_e xterms,
        const double k1x,
        const double k2x,
        const double k1y,
        const double k2y,
        /* Output */
        double* const zfit,
        stimage_error_t* const error);

/**
Evaluate a Legendre polynomial, assuming that the coefficients have
been calculated.

@param Order of the polynomial in x

@param Order of the polynomial in y

@param coeff 1D array of coefficients

@param ncoord Number of points to be evaluated

@param ref Reference points to be evaluated

@param xterms Type of cross terms

@param k1x Normalizing constant

@param k2x Normalizing constant

@param k1y Normalizing constant

@param k2y Normalizing constant

@param zfit The fitted points

@param error

@return non-zero on failure
 */
int
eval_legendre(
        const int xorder,
        const int yorder,
        const double* const coeff,
        const size_t ncoord,
        const coord_t* const ref,
        const xterms_e xterms,
        const double k1x,
        const double k2x,
        const double k1y,
        const double k2y,
        /* Output */
        double* const zfit,
        stimage_error_t* const error);

int
basis_poly(
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const int order,
        const double k1,
        const double k2,
        double* const basis,
        stimage_error_t* const error);

int
basis_chebyshev(
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const int order,
        const double k1,
        const double k2,
        double* const basis,
        stimage_error_t* const error);

int
basis_legendre(
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const int order,
        const double k1,
        const double k2,
        double* const basis,
        stimage_error_t* const error);

#endif

