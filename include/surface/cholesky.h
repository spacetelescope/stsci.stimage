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

/*
 Author: Michael Droettboom
         mdroe@stsci.edu
*/

#ifndef _STIMAGE_SURFACE_CHOLESKY_H_
#define _STIMAGE_SURFACE_CHOLESKY_H_

#include "surface/fit.h"
#include "surface/surface.h"

/* was dgschofac */

/**
Calculate the Cholesky factorization of a symmetric, positive
semi-definite banded matrix.

@param nbands Number of bands

@param nrows Number of rows

@param matrix Data matrix [nbands, nrows]

@param matfac Cholesky factorization [nbands, nrows]

@param error_type error code

@param error

@return Non-zero on error
 */
int
cholesky_factorization(
        const size_t nbands,
        const size_t nrows,
        const double* const matrix,
        /* Output */
        double* const matfac,
        surface_fit_error_e* const error_type,
        stimage_error_t* const error);

/* was dgschoslv */

/**
Solve the matrix whose Cholesky factorization was calculated in
cholesky_factorization for the coefficients.

@param nbands Number of bands

@param nrows Number of rows

@param matfac Cholesky factorization [nbands, nrows]

@param vector Right side of the matrix equation [nrows]

@param coeff Coefficients [nrows]

@param error

@return Non-zero on error
*/
int
cholesky_solve(
        const size_t nbands,
        const size_t nrows,
        const double* const matfac,
        const double* const vector,
        /* Output */
        double* const coeff,
        stimage_error_t* const error);

#endif
