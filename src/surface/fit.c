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
#include <stdio.h>

#include "surface/cholesky.h"
#include "surface/fit.h"
#include "lib/polynomial.h"

static double
vector_dot_product(
    const size_t n,
    const double* const a,
    const double* const b) {

    size_t i;
    double sum = 0.0;

    for (i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

/* was dgsacpts */
static int
surface_fit_add_points(
        surface_t* const s,
        const size_t ncoord,
        const coord_t* const coord,
        const double* const z,
        double* const w,
        const surface_fit_weight_e weight_type,
        stimage_error_t* const error) {

    size_t i, j, k, l, ii, jj, ll;
    double* byw = NULL;
    double* bw = NULL;
    double* xbasis = NULL;
    double* ybasis = NULL;
    double* vzp;
    double* mzp;
    double* bxp;
    double* byp;
    double* vindex;
    double* mindex;
    double* bbyp;
    double* bbxp;
    int xorder;
    int xxorder;
    int maxorder;
    size_t ntimes;
    int status = 1;

    assert(s);
    assert(coord);
    assert(z);
    assert(w);
    assert(error);
    assert(s->vector);
    assert(s->matrix);

    /* Increment the number of points */
    s->npoints += ncoord;

    /* Calculate weights */
    switch (weight_type) {
    case surface_fit_weight_spacing:
        if (ncoord == 1) {
            w[0] = 1.0;
        } else {
            w[0] = ABS(coord[1].x - coord[0].x);
        }

        for (i = 1; i < ncoord - 1; ++i) {
            w[i] = ABS(coord[i+1].x - coord[i-1].x);
        }

        if (ncoord == 1) {
            w[ncoord-1] = 1.0;
        } else {
            w[ncoord-1] = ABS(coord[ncoord-1].x - coord[ncoord-2].x);
        }
        break;
    case surface_fit_weight_user:
        /* User supplied-weights: don't touch the w vector */
        break;
    default:
        for (i = 0; i < ncoord; ++i) {
            w[i] = 1.0;
        }
        break;
    }

    xbasis = malloc_with_error(ncoord * s->xorder * sizeof(double), error);
    if (xbasis == NULL) goto exit;
    ybasis = malloc_with_error(ncoord * s->yorder * sizeof(double), error);
    if (ybasis == NULL) goto exit;

    /* Calculate the non-zero basis functions */
    switch (s->type) {
    case surface_type_polynomial:
        if (basis_poly(
                    ncoord, 0, coord, s->xorder, s->xmaxmin, s->xrange,
                    xbasis, error)) goto exit;
        if (basis_poly(
                    ncoord, 1, coord, s->yorder, s->ymaxmin, s->yrange,
                    ybasis, error)) goto exit;
        break;
    case surface_type_chebyshev:
        if (basis_chebyshev(
                    ncoord, 0, coord, s->xorder, s->xmaxmin, s->xrange,
                    xbasis, error)) goto exit;
        if (basis_chebyshev(
                    ncoord, 1, coord, s->yorder, s->ymaxmin, s->yrange,
                    ybasis, error)) goto exit;
        break;
    case surface_type_legendre:
        if (basis_legendre(
                    ncoord, 0, coord, s->xorder, s->xmaxmin, s->xrange,
                    xbasis, error)) goto exit;
        if (basis_legendre(
                    ncoord, 1, coord, s->yorder, s->ymaxmin, s->yrange,
                    ybasis, error)) goto exit;
        break;
    default:
        stimage_error_set_message(error, "Illegal curve type");
        goto exit;
    }

    /* Allocate temporary space for matrix accumulation */
    byw = malloc_with_error(ncoord * sizeof(double), error);
    if (byw == NULL) goto exit;
    bw = malloc_with_error(ncoord * sizeof(double), error);
    if (bw == NULL) goto exit;

    vzp = s->vector - 1;
    mzp = s->matrix;
    bxp = xbasis;
    byp = ybasis;

    maxorder = MAX(s->xorder + 1, s->yorder + 1);
    xorder = s->xorder;
    ntimes = 0;
    for (l = 1; l <= s->yorder; ++l) {
        for (i = 0; i < ncoord; ++i) {
            byw[i] = w[i] * byp[i];
        }

        bxp = xbasis;

        for (k = 1; k <= s->xorder; ++k) {
            for (i = 0; i < ncoord; ++i) {
                bw[i] = byw[i] * bxp[i];
            }

            vindex = vzp + k;
            assert(vindex - s->vector < s->ncoeff);
            *vindex += vector_dot_product(ncoord, bw, z);
            bbyp = byp;
            bbxp = bxp;
            xxorder = xorder;
            jj = k;
            ll = l;
            ii = 0;
            for (j = k + ntimes; j <= s->ncoeff; ++j) {
                mindex = mzp + ii;
                assert(mindex - s->matrix < s->ncoeff * s->ncoeff);
                assert((bbxp - xbasis) + ncoord - 1 < ncoord * s->xorder);
                assert((bbyp - ybasis) + ncoord - 1 < ncoord * s->yorder);
                for (i = 0; i < ncoord; ++i) {
                    *mindex += bw[i] * bbxp[i] * bbyp[i];
                }
                if (jj % xxorder == 0) {
                    jj = 1;
                    ++ll;
                    bbxp = xbasis;
                    bbyp += ncoord;
                    switch (s->xterms) {
                    case xterms_none:
                        xxorder = 1;
                        break;
                    case xterms_half:
                        if ((ll + s->xorder) > maxorder) {
                            --xxorder;
                        }
                        break;
                    default:
                        break;
                    }
                } else {
                    ++jj;
                    bbxp += ncoord;
                }
                ++ii;
            }
            mzp += s->ncoeff;
            bxp += ncoord;
        }

        vzp += xorder;
        ntimes += xorder;

        switch (s->xterms) {
        case xterms_none:
            xorder = 1;
            break;
        case xterms_half:
            if ((l + s->xorder + 1) > maxorder) {
                --xorder;
            }
            break;
        default:
            break;
        }
        byp += ncoord;
    }

    status = 0;

    surface_print(s);

 exit:

    free(byw);
    free(bw);
    free(xbasis);
    free(ybasis);

    return status;
}

static int
surface_fit_solve(
        surface_t* const s,
        /* Output  */
        surface_fit_error_e* const error_type,
        stimage_error_t* const error) {

    int nfree;

    assert(s);
    assert(error_type);
    assert(error);
    assert(s->matrix);
    assert(s->vector);
    assert(s->cholesky_fact);
    assert(s->coeff);

    *error_type = surface_fit_error_ok;

    nfree = (int)s->npoints - (int)s->ncoeff;
    if (nfree < 0) {
        *error_type = surface_fit_error_no_degrees_of_freedom;
        return 0;
    }

    switch (s->type) {
    case surface_type_polynomial:
    case surface_type_chebyshev:
    case surface_type_legendre:
        if (cholesky_factorization(
                    s->ncoeff, s->ncoeff, s->matrix, s->cholesky_fact,
                    error_type, error)) return 1;
        if (cholesky_solve(
                    s->ncoeff, s->ncoeff, s->cholesky_fact, s->vector,
                    s->coeff, error)) return 1;
        break;

    default:
        stimage_error_set_message(error, "Illegal surface type");
        return 1;
    }

    surface_print(s);

    return 0;
}

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
        stimage_error_t* const error) {

    assert(s);
    assert(coord);
    assert(z);
    assert(w);
    assert(error);

    if (surface_zero(s, error) ||
        surface_fit_add_points(s, ncoord, coord, z, w, weight_type, error) ||
        surface_fit_solve(s, error_type, error)) {
        return 1;
    }

    return 0;
}
