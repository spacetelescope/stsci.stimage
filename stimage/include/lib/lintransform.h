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

#ifndef _STIMAGE_LINTRANSFORM_H_
#define _STIMAGE_LINTRANSFORM_H_

#include "lib/util.h"

typedef struct {
    double a;
    double b;
    double c;
    double d;
    double e;
    double f;
} lintransform_t;

/**
Compute linear transformation coefficients.

@param in The origin of the input coordinates

@param mag Scale

@param rot Rotation (in degrees)

@param out The origin of the output coordinates

@param coeffs The output set of coefficients
*/
void
compute_lintransform(
    const coord_t in,
    const coord_t mag,
    const coord_t rot,
    const coord_t out,
    lintransform_t* coeffs);

/**
Apply a linear transformation to a list of coordinates.

@param coeffs A set of coeffs, for example created by compute_lintransform

@param ncoords The number of coordinates in the list

@param input The input set of coordinates

@param output The output set of coordinates.  May be equal to input.
*/
void
apply_lintransform(
    const lintransform_t* const coeffs,
    size_t ncoords,
    const coord_t* const input, /* [ncoords] */
    coord_t* output);

#endif /* _STIMAGE_LINTRANSFORM_H_ */
