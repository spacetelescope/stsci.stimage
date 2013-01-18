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
#include <string.h>

#include "lib/polynomial.h"

typedef int (*basis_function_t)(
        const size_t,
        const size_t,
        const coord_t* const,
        const int,
        const double,
        const double,
        double* const,
        stimage_error_t* const);

int
eval_1dpoly(
        const int order,
        const double* const coeff,
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        double* const zfit,
        stimage_error_t* const error) {

    size_t        i      = 0;
    size_t        j      = 0;
    const double* x      = (double *)ref + axis;
    double*       tmp    = NULL;
    int           status = 1;

    assert(coeff);
    assert(ref);
    assert(zfit);
    assert(error);

    for (i = 0; i < ncoord; ++i) {
        zfit[i] = coeff[0];
    }

    if (order == 1) {
        return 0;
    }

    for (i = 0; i < ncoord; ++i) {
        zfit[i] += (x[i<<1] * coeff[1]);
    }

    if (order == 2) {
        return 0;
    }

    tmp = malloc_with_error(ncoord * sizeof(double), error);
    if (tmp == NULL) goto exit;

    for (i = 0; i < ncoord; ++i) {
        tmp[i] = x[i<<1];
    }

    for (j = 2; j < order; ++j) {
        for (i = 0; i < ncoord; ++i) {
            tmp[i] *= x[i<<1];
            zfit[i] += tmp[i] * coeff[j];
        }
    }

    status = 0;

 exit:

    free(tmp);

    return 0;
}

int
eval_1dchebyshev(
        const int order,
        const double* const coeff,
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const double k1,
        const double k2,
        double* const zfit,
        stimage_error_t* const error) {

    size_t        i      = 0;
    size_t        j      = 0;
    const double* x      = (double *)ref + axis;
    double        c1     = 0.0;
    double        c2     = 0.0;
    double*       sx     = NULL;
    double*       pn     = NULL;
    double*       pnm1   = NULL;
    double*       pnm2   = NULL;
    int           status = 1;

    assert(coeff);
    assert(ref);
    assert(zfit);
    assert(error);

    for (i = 0; i < ncoord; ++i) {
        zfit[i] = coeff[0];
    }

    if (order == 1) {
        return 0;
    }

    c1 = k2 * coeff[1];
    c2 = c1 * k1 + coeff[0];
    for (i = 0; i < ncoord; ++i) {
        zfit[i] = x[i<<1] * c1 + c2;
    }

    if (order == 2) {
        return 0;
    }

    sx = malloc_with_error(ncoord * sizeof(double), error);
    if (sx == NULL) goto exit;

    pn = malloc_with_error(ncoord * sizeof(double), error);
    if (pn == NULL) goto exit;

    pnm1 = malloc_with_error(ncoord * sizeof(double), error);
    if (pnm1 == NULL) goto exit;

    pnm2 = malloc_with_error(ncoord * sizeof(double), error);
    if (pnm2 == NULL) goto exit;

    for (i = 0; i < ncoord; ++i) {
        pnm2[i] = 1.0;
        pnm1[i] = sx[i] = (x[i<<1] + k1) * k2;
        sx[i] *= 2;
    }

    for (j = 2; j < order; ++j) {
        for (i = 0; i < ncoord; ++i) {
            pn[i] = (sx[i] * pnm1[i]) - pnm2[i];
        }

        if (j < order - 1) {
            for (i = 0; i < ncoord; ++i) {
                pnm2[i] = pnm1[i];
                pnm1[i] = pn[i];
            }
        }

        for (i = 0; i < ncoord; ++i) {
            pn[i] *= coeff[j];
            zfit[i] += pn[i];
        }
    }

    status = 0;

 exit:

    free(sx);
    free(pn);
    free(pnm1);
    free(pnm2);

    return 0;
}

int
eval_1dlegendre(
        const int order,
        const double* const coeff,
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const double k1,
        const double k2,
        double* const zfit,
        stimage_error_t* const error) {

    size_t        i      = 0;
    size_t        j      = 0;
    const double* x      = (double *)ref + axis;
    double        ri     = 0.0;
    double        ri1    = 0.0;
    double        ri2    = 0.0;
    double*       sx     = NULL;
    double*       pn     = NULL;
    double*       pnm1   = NULL;
    double*       pnm2   = NULL;
    int           status = 1;

    assert(coeff);
    assert(ref);
    assert(zfit);
    assert(error);

    for (i = 0; i < ncoord; ++i) {
        zfit[i] = coeff[0];
    }

    if (order == 1) {
        return 0;
    }

    ri1 = k2 * coeff[1];
    ri2 = ri1 * k1 + coeff[0];
    for (i = 0; i < ncoord; ++i) {
        zfit[i] = (x[i<<1] * ri1) + ri2;
    }

    if (order == 2) {
        return 0;
    }

    sx = malloc_with_error(ncoord * sizeof(double), error);
    if (sx == NULL) goto exit;

    pn = malloc_with_error(ncoord * sizeof(double), error);
    if (pn == NULL) goto exit;

    pnm1 = malloc_with_error(ncoord * sizeof(double), error);
    if (pnm1 == NULL) goto exit;

    pnm2 = malloc_with_error(ncoord * sizeof(double), error);
    if (pnm2 == NULL) goto exit;

    for (i = 0; i < ncoord; ++i) {
        pnm2[i] = 1.0;
        pnm1[i] = sx[i] = (x[i<<1] + k1) * k2;
    }

    for (j = 2; j < order; ++j) {
        ri = (double)j + 1.0;
        ri1 = (2.0 * ri - 3.0) / (ri - 1.0);
        ri2 = -(ri - 2.0) / (ri - 1.0);

        for (i = 0; i < ncoord; ++i) {
            pn[i] = sx[i] * pnm1[i];
            pn[i] = pn[i] * ri1 + pnm2[i] * ri2;
        }

        if (j < order - 1) {
            for (i = 0; i < ncoord; ++i) {
                pnm2[i] = pnm1[i];
                pnm1[i] = pn[i];
            }
        }

        for (i = 0; i < ncoord; ++i) {
            pn[i] *= coeff[j];
            zfit[i] += pn[i];
        }
    }

    status = 0;

 exit:

    free(sx);
    free(pn);
    free(pnm1);
    free(pnm2);

    return status;
}

int
basis_poly(
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const int order,
        const double k1, /* Ignored */
        const double k2, /* Ignored */
        double* const basis,
        stimage_error_t* const error) {

    size_t              i  = 0;
    size_t              k  = 0;
    const double* const x  = (double*)ref + axis;
    double*             bp = basis;

    assert(ref);
    assert(basis);
    assert(error);

    for (k = 0; k < order; ++k) {
        assert((bp - basis) >= 0);

        if (k == 0) {
            for (i = 0; i < ncoord; ++i) {
                bp[i] = 1.0;
            }
        } else if (k == 1) {
            for (i = 0; i < ncoord; ++i) {
                bp[i] = x[i<<1];
            }
        } else {
            for (i = 0; i < ncoord; ++i) {
                assert(((bp - basis) + i - ncoord) > 0);
                assert(((bp - basis) + i - ncoord) < ncoord * order);
                bp[i] = x[i<<1] * bp[i - ncoord];
            }
        }

        bp += ncoord;
    }

    return 0;
}

int
basis_chebyshev(
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const int order,
        const double k1,
        const double k2,
        double* const basis,
        stimage_error_t* const error) {

    size_t              i  = 0;
    size_t              k  = 0;
    const double* const x  = (double*)ref + axis;
    double*             bp = basis;

    assert(ref);
    assert(basis);
    assert(error);

    for (k = 0; k < order; ++k) {
        if (k == 0) {
            for (i = 0; i < ncoord; ++i) {
                bp[i] = 1.0;
            }
        } else if (k == 1) {
            for (i = 0; i < ncoord; ++i) {
                bp[i] = (x[i<<1] + k1) * k2;
            }
        } else {
            for (i = 0; i < ncoord; ++i) {
                assert(((bp - basis) + i - ncoord) >= 0);
                assert(((bp - basis) + i - ncoord) < ncoord * order);
                assert(((bp - basis) + i - (2 * ncoord)) >= 0);
                assert(((bp - basis) + i - (2 * ncoord)) < ncoord * order);
                bp[i] = (basis[ncoord+i] * bp[i-ncoord]);
                bp[i] *= 2.0;
                bp[i] -= bp[i-(2 * ncoord)];
            }
        }

        bp += ncoord;
    }

    return 0;
}

int
basis_legendre(
        const size_t ncoord,
        const size_t axis,
        const coord_t* const ref,
        const int order,
        const double k1,
        const double k2,
        double* const basis,
        stimage_error_t* const error) {

    size_t              i   = 0;
    size_t              k   = 0;
    const double* const x   = (double*)ref + axis;
    double*             bp  = basis;
    double              ri  = 0.0;
    double              ri1 = 0.0;
    double              ri2 = 0.0;

    assert(ref);
    assert(basis);
    assert(error);

    for (k = 0; k < order; ++k) {
        if (k == 0) {
            for (i = 0; i < ncoord; ++i) {
                bp[i] = 1.0;
            }
        } else if (k == 1) {
            for (i = 0; i < ncoord; ++i) {
                bp[i] = (x[i<<1] + k1) * k2;
            }
        } else {
            assert(((bp - basis) + i - ncoord) >= 0);
            assert(((bp - basis) + i - ncoord) < ncoord * order);
            assert(((bp - basis) + i - (2 * ncoord)) >= 0);
            assert(((bp - basis) + i - (2 * ncoord)) < ncoord * order);

            ri = k + 1;
            ri1 = (2.0 * ri - 3.0) / (ri - 1.0);
            ri2 = -(ri - 2.0) / (ri - 1.0);
            for (i = 0; i < ncoord; ++i) {
                bp[i] = (basis[ncoord+i] * bp[i-ncoord]);
                bp[i] = bp[i] * ri1 + bp[i-(2 * ncoord)] * ri2;
            }
        }

        bp += ncoord;
    }

    return 0;
}

static int
eval_poly_generic(
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
        basis_function_t basis_function,
        /* Output */
        double* const zfit,
        stimage_error_t* const error) {

    size_t       i        = 0;
    size_t       j        = 0;
    size_t       k        = 0;
    double*      xb       = NULL;
    double*      yb       = NULL;
    double*      accum    = NULL;
    size_t       cp       = 0;
    const size_t maxorder = MAX(xorder + 1, yorder + 1);
    size_t       xincr    = 0;
    double*      xbp      = xb;
    double*      ybp      = yb;
    int          status   = 1;

    assert(coeff);
    assert(ref);
    assert(zfit);
    assert(error);

    /* Fit a constant */
    if (xorder == 1 && yorder == 1) {
        for (i = 0; i < ncoord; ++i) {
            zfit[i] = coeff[0];
        }

        return 0;
    }

    /* Fit first order in x and y */
    if (xorder == 2 && yorder == 1) {
        for (i = 0; i < ncoord; ++i) {
            zfit[i] += ref[i].x * coeff[1];
        }

        return 0;
    }

    if (yorder == 2 && xorder == 1) {
        for (i = 0; i < ncoord; ++i) {
            zfit[i] += ref[i].y * coeff[1];
        }

        return 0;
    }

    if (yorder == 2 && xorder == 2 && xterms == xterms_none) {
        for (i = 0; i < ncoord; ++i) {
            zfit[i] += ref[i].x * coeff[1] + ref[i].y * coeff[2];
        }

        return 0;
    }

    xb = malloc_with_error(xorder * ncoord * sizeof(double), error);
    if (xb == NULL) goto exit;
    yb = malloc_with_error(yorder * ncoord * sizeof(double), error);
    if (yb == NULL) goto exit;
    accum = malloc_with_error(ncoord * sizeof(double), error);
    if (accum == NULL) goto exit;

    /* Calculate basis functions */
    if (basis_function(ncoord, 0, ref, xorder, k1x, k2x, xb, error)) goto exit;
    if (basis_function(ncoord, 1, ref, yorder, k1y, k2y, yb, error)) goto exit;

    /* Accumulate the output vector */
    for (i = 0; i < ncoord; ++i) {
        zfit[i] = 0.0;
    }

    if (xterms != xterms_none) {
        xincr = xorder;
        ybp = yb;
        for (j = 0; j < yorder; ++j) {
            for (i = 0; i < ncoord; ++i) {
                accum[i] = 0.0;
            }
            xbp = xb;
            for (k = 0; k < xincr; ++k) {
                for (i = 0; i < ncoord; ++i) {
                    accum[i] += xbp[i] * coeff[cp+k];
                }
            }
            xbp += ncoord;
        }

        for (i = 0; i < ncoord; ++i) {
            zfit[i] += accum[i] * ybp[i];
        }

        cp += xincr;
        ybp += ncoord;

        if (xterms == xterms_half) {
            if ((j + xorder + 1) > maxorder) {
                xincr -= 1;
            }
        }
    } else { /* xterms == surface_xterms_none */
        xbp = xb;
        for (k = 0; k < xorder; ++k) {
            for (i = 0; i < ncoord; ++i) {
                zfit[i] += xbp[i] * coeff[k];
            }

            xbp += ncoord;
        }

        ybp = yb + ncoord;
        for (k = 0; k < yorder - 1; ++k) {
            for (i = 0; i < ncoord; ++i) {
                zfit[i] += ybp[i] * coeff[xorder+k];
            }

            ybp += ncoord;
        }
    }

    status = 0;

 exit:
    free(xb);
    free(yb);
    free(accum);

    return status;
}

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
        stimage_error_t* const error) {

    return eval_poly_generic(
            xorder, yorder, coeff, ncoord, ref, xterms, k1x, k2x, k1y, k2y,
            &basis_poly, zfit, error);
}

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
        stimage_error_t* const error) {

    return eval_poly_generic(
            xorder, yorder, coeff, ncoord, ref, xterms, k1x, k2x, k1y, k2y,
            &basis_chebyshev, zfit, error);
}

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
        stimage_error_t* const error) {

    return eval_poly_generic(
            xorder, yorder, coeff, ncoord, ref, xterms, k1x, k2x, k1y, k2y,
            &basis_legendre, zfit, error);
}
