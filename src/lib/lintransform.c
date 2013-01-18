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

#include <assert.h>
#define _USE_MATH_DEFINES       /* needed for MS Windows to define M_PI */ 
#include <math.h>

#include "lib/lintransform.h"

void
compute_lintransform(
    const coord_t in,
    const coord_t mag,
    const coord_t rot,
    const coord_t out,
    lintransform_t* coeffs) {

    assert(coeffs);

    assert(coord_is_finite(&in));
    assert(coord_is_finite(&mag));
    assert(coord_is_finite(&rot));
    assert(coord_is_finite(&out));

    coeffs->a = mag.x * cos(DEGTORAD(rot.x));
    coeffs->b = -mag.y * sin(DEGTORAD(rot.y));
    coeffs->c = out.x - coeffs->a * in.x - coeffs->b * in.y;

    coeffs->d = mag.x * sin(DEGTORAD(rot.x));
    coeffs->e = mag.y * cos(DEGTORAD(rot.y));
    coeffs->f = out.y - coeffs->d * in.x - coeffs->e * in.y;
}

void
apply_lintransform(
    const lintransform_t* const coeffs,
    size_t ncoords,
    const coord_t* const input, /* [ncoords] */
    coord_t* output) {

    size_t i;
    double x, y;

    assert(coeffs);
    assert(input);
    assert(output);

    for (i = 0; i < ncoords; ++i) {
        assert(coord_is_finite(input + i));

        x = input[i].x;
        y = input[i].y;

        output[i].x = coeffs->a * x + coeffs->b * y + coeffs->c;
        output[i].y = coeffs->d * x + coeffs->e * y + coeffs->f;
    }
}
