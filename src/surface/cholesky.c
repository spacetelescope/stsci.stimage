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

#include <assert.h>

#include "surface/cholesky.h"

int
cholesky_factorization(
        const size_t nbands,
        const size_t nrows,
        const double* const matrix,
        /* Output */
        double* const matfac,
        surface_fit_error_e* const error_type,
        stimage_error_t* const error) {

    #define MATRIX(j, i) (matrix[(i)*nbands+(j)])
    #define MATFAC(j, i) (matfac[(i)*nbands+(j)])

    size_t i, n, j;
    int imax, jmax;
    double ratio;

    assert(matrix);
    assert(matfac);
    assert(error_type);
    assert(error);

    if (nrows == 1) {
        if (MATRIX(0, 0) > 0.0) {
            MATFAC(0, 0) = 1.0 / MATRIX(0, 0);
        }
        return 0;
    }

    /* Copy matrix into matfac */
    for (n = 0; n < nrows; ++n) {
        for (j = 0; j < nbands; ++j) {
            assert(n < nbands && j < nrows);
            MATFAC(j, n) = MATRIX(j, n);
        }
    }

    for (n = 0; n < nrows; ++n) {
        /* Test to see if matrix is singular */
        if (((MATFAC(0, n) + MATRIX(0, n)) - MATRIX(0, n)) <=
            1000.0 / MAX_DOUBLE) {
            for (j = 0; j < nbands; ++j) {
                assert(n < nbands && j < nrows);
                MATFAC(j, n) = 0.0;
            }
            *error_type = surface_fit_error_singular;
            continue;
        }

        assert(MATFAC(0, n) != 0.0);
        MATFAC(0, n) = 1.0 / MATFAC(0, n);
        imax = MIN(nbands - 1, nrows - n);
        if (imax < 0) {
            continue;
        }

        jmax = imax;
        for (i = 0; i < (size_t)imax; ++i) {
            assert(n < nbands && i+1 < nrows);
            ratio = MATFAC(i+1, n) * MATFAC(0, n);
            for (j = 0; j < (size_t)jmax; ++j) {
                assert(n+i < nbands && j < nrows && j+i < nrows);
                MATFAC(j, n+i) = MATFAC(j, n+i) - MATFAC(j+i, n) * ratio;
            }
            --jmax;
            assert(n < nbands && i+1 < nrows);
            MATFAC(i+1, n) = ratio;
        }
    }

    return 0;

    #undef MATRIX
    #undef MATFAC
}

int
cholesky_solve(
        const size_t nbands,
        const size_t nrows,
        const double* const matfac,
        const double* const vector,
        /* Output */
        double* const coeff,
        stimage_error_t* const error) {

    #define MATFAC(j, i) (matfac[(i)*nbands+(j)])

    size_t i, j, jmax, nbands_m1;
    int n;

    assert(matfac);
    assert(vector);
    assert(coeff);
    assert(error);
    assert(nbands >= 1);
    assert(nrows >= 1);

    if (nrows == 1) {
        coeff[0] = vector[0] * MATFAC(0, 0);
        return 0;
    }

    /* Copy vector to coefficients */
    for (i = 0; i < nrows; ++i) {
        coeff[i] = vector[i];
    }

    /* Forward substitution */
    nbands_m1 = nbands - 1;
    for (n = 0; n < (int)nrows; ++n) {
        jmax = MIN(nbands_m1, nrows - n);
        if (jmax >= 1) {
            for (j = 0; j < jmax; ++j) {
                coeff[j+n] -= MATFAC(j+1, n) * coeff[n];
            }
        }
    }

    /* Back substitution */
    for (n = (int)nrows - 1; n >= 0; --n) {
        coeff[n] *= MATFAC(0, n);
        jmax = MIN(nbands_m1, nrows - n);
        if (jmax >= 1) {
            for (j = 0; j < jmax; ++j) {
                coeff[n] -= MATFAC(j+1, n) * coeff[j+n];
            }
        }
    }

    return 0;

    #undef MATFAC
}

