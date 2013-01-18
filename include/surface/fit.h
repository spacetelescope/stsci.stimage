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

#ifndef _STIMAGE_SURFACE_FIT_H_
#define _STIMAGE_SURFACE_FIT_H_

#include "surface/surface.h"

typedef enum {
    surface_fit_error_ok,
    surface_fit_error_singular,
    surface_fit_error_no_degrees_of_freedom,
    surface_fit_error_undefined,
    surface_fit_error_LAST
} surface_fit_error_e;

typedef enum {
    surface_fit_weight_uniform,
    surface_fit_weight_spacing,
    surface_fit_weight_user,
    surface_fit_weight_LAST
} surface_fit_weight_e;

/* was: dgsfit */

/**
Solve the normal equations for a surface.  The inner products of the
basis functions are calculated and accumulated into the s->ncoeff ** 2
matrix.  The main diagonal of the matrix is stored in the first row of
matrix followed by the remaining non-zero diagonals.  The inner
product of the basis functions and the data ordinates are stored in
the sf->ncoeff vector. The Cholesky factorization of matrix is
calculated and stored in s->chofac. Forward and back substitution is
used to solve for the s->ncoeff-vector coeff.

@param s Surface descriptor

@param ncoord Number of data points

@param coord Data points

@param x data array

@param w weights array

@param weight_type type of weights

@param error_type

@param error
*/
int
surface_fit(
        surface_t* const s,
        const size_t ncoord,
        const coord_t* const coord,
        const double* const z,
        double* const w,
        const surface_fit_weight_e weight_type,
        /* Output */
        surface_fit_error_e* const error_type,
        stimage_error_t* const error);

#endif
