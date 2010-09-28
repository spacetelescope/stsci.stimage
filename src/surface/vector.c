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

#include "lib/polynomial.h"
#include "surface/vector.h"

int
surface_vector(
        const surface_t* const s,
        const size_t ncoord,
        const coord_t* const ref,
        /* Output */
        double* const zfit,
        stimage_error_t* const error) {

    int status;

    assert(s);
    assert(ref);
    assert(zfit);
    assert(error);

    switch (s->type) {
    case surface_type_polynomial:
        if (s->xorder == 1) {
            status = eval_1dpoly(
                    s->yorder, s->coeff, ncoord, 1, ref, zfit, error);
        } else if (s->yorder == 1) {
            status = eval_1dpoly(
                    s->xorder, s->coeff, ncoord, 0, ref, zfit, error);
        } else {
            status = eval_poly(
                    s->xorder, s->yorder, s->coeff,
                    ncoord, ref, s->xterms,
                    s->xmaxmin, s->xrange,
                    s->ymaxmin, s->yrange,
                    zfit, error);
        }
        break;

    case surface_type_chebyshev:
        if (s->xorder == 1) {
            status = eval_1dchebyshev(
                    s->yorder, s->coeff, ncoord, 1, ref,
                    s->ymaxmin, s->yrange, zfit, error);
        } else if (s->yorder == 1) {
            status = eval_1dchebyshev(
                    s->xorder, s->coeff, ncoord, 0, ref,
                    s->xmaxmin, s->xrange, zfit, error);
        } else {
            status = eval_chebyshev(
                    s->xorder, s->yorder, s->coeff,
                    ncoord, ref, s->xterms,
                    s->xmaxmin, s->xrange,
                    s->ymaxmin, s->yrange,
                    zfit, error);
        }
        break;

    case surface_type_legendre:
        if (s->xorder == 1) {
            status = eval_1dlegendre(
                    s->yorder, s->coeff, ncoord, 1, ref,
                    s->ymaxmin, s->yrange, zfit, error);
        } else if (s->yorder == 1) {
            status = eval_1dlegendre(
                    s->xorder, s->coeff, ncoord, 0, ref,
                    s->xmaxmin, s->xrange, zfit, error);
        } else {
            status = eval_legendre(
                    s->xorder, s->yorder, s->coeff,
                    ncoord, ref, s->xterms,
                    s->xmaxmin, s->xrange,
                    s->ymaxmin, s->yrange,
                    zfit, error);
        }
        break;

    default:
        stimage_error_set_message(error, "Unknown surface function");
        return 1;
    }

    return status;
}
