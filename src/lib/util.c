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
#include <stdlib.h>

#include "lib/util.h"

void *
malloc_with_error(
        size_t size,
        stimage_error_t* error) {

    void *result = NULL;

    assert(error);

    result = malloc(size);
    if (result == NULL) {
        stimage_error_format_message(error, "Error allocating %u bytes", size);
    }
    return result;
}

void *
calloc_with_error(
        size_t nmemb,
        size_t size,
        stimage_error_t* error) {

    void *result = NULL;

    assert(error);

    result = calloc(nmemb, size);
    if (result == NULL) {
        stimage_error_format_message(error, "Error allocating %u bytes", size);
    }
    return result;
}

STIMAGE_Int64
factorial(
        size_t n) {

    STIMAGE_Int64 fac = 1;

    while (n > 0)
        fac *= n--;

    return fac;
}

/*
Though this function is defined as:

   n! / ((n - ngroup)! * ngroup!)

However, doing so directly would cause overflow pretty quickly for
even small values of n.

Therefore we do some things based on a simplified version of the
equation.
 */
size_t
combinatorial(
        size_t n,
        size_t ngroup) {

    size_t fac;
    size_t gfac;
    size_t i;

    assert(n > ngroup);
    assert(ngroup > 0);
    assert(n < 2346);

    if (n == 0) {
        return 1;
    }

    fac = n;
    for (i = n - 1; i > n - 3; --i) {
        fac *= i;
    }

    gfac = ngroup;
    for (i = ngroup - 1; i > 1; --i) {
        gfac *= i;
    }

    return fac / gfac;
}

static int
double_compare(
        const void* ap,
        const void* bp) {

    const double a = *(const double*)ap;
    const double b = *(const double*)bp;

    if (a < b) {
        return -1;
    } else if (a > b) {
        return 1;
    } else {
        return 0;
    }
}

void
sort_doubles(
        const size_t n,
        double* const a) {

    assert(a);

    qsort(a, n, sizeof(double), &double_compare);
}

void
double_normalize(
        const double x,
        double* const normx,
        int* const expon) {

    const double tol = EPS_DOUBLE * 10.0;
    double absx = ABS(x);

    *expon = 0;

    if (absx > 0.0) {
        while (absx < (1.0 - tol)) {
            absx *= 10.0;
            --(*expon);
            if (absx == 0.0) {
                *normx = 0.0;
                *expon = 0;
                return;
            }
        }

        while (absx >= (10.0 + tol)) {
            absx /= 10.0;
            ++(*expon);
        }
    }

    if (x < 0) {
        *normx = -absx;
    } else {
        *normx = absx;
    }
}

int
double_approx_equal(
        const double x,
        const double y) {

    double normx, normy;
    double x1, x2;
    int ex, ey;
    const double tol = EPS_DOUBLE * 32.0;

    /* Obvious first */
    if (x == y) {
        return 1;
    }

    /* We can't normalize zero, so handle the zero operand cases
       first */
    if (x == 0.0 || y == 0.0) {
        return 0;
    }

    /* Normalize the operands and do an epsilon compare */
    double_normalize(x, &normx, &ex);
    double_normalize(y, &normy, &ey);

    if (ex != ey) {
        return 0;
    } else {
        x1 = 1.0 + ABS(normx - normy);
        x2 = 1.0 + tol;
        return (x1 <= x2);
    }
}

double
compute_mode(
        const size_t n,
        const double* const a,
        const size_t min,
        const double range,
        const double bin,
        const double step) {

    int x1, x2, x3, nmax;
    double y1, y2, mode;

    assert(a);

    if (n == 1) {
        return a[0];
    }

    /* If there are too few points, return the median */
    if (n < min) {
        if ((n & 0x1) == 1) {
            return a[(n>>1)];
        } else {
            return (a[(n>>1)] + a[(n>>1)+1]) * 0.5;
        }
    }

    /* Compute the data range that will be used to do the mode search.
       If the data has no range, then the constant value will be
       returned. */
    x1 = MAX(0, (int)((double)(n) * (1.0 - range) * 0.5));
    x3 = MIN(n-1, (int)((double)(n) * (1.0 + range) * 0.5));
    if (a[x1] == a[x3]) {
        return a[x1];
    }

    /* Compute the bin and step size.  The bin size is based on the
       data range over a fraction of the pixels around the median and
       a bin step which may be smaller than the bin size. */
    nmax = 0;
    x2 = x1;
    for (y1 = a[x1]; x2 < x3; y1 += step) {
        for (; a[x1] < y1; ++x1) {
            /* empty */
        }
        y2 = y1 + bin;
        for (; (x2 < x3) && (a[x2] < y2); ++x2) {
            /* empty */
        }
        if (x2 - x1 > nmax) {
            nmax = x2 - x1;
            if (((x2 + x1) & 0x1) == 0) {
                assert(((x2+x1)>>1) < n);
                mode = a[((x2+x1)>>1) - 1];
            } else {
                assert(((x2+x1)>>1) + 1 < n);
                mode = (a[(x2+x1)>>1] + a[((x2+x1)>>1)+1]) * 0.5;
            }
        }
    }

    return mode;
}

double
compute_mean(
        const size_t n,
        const double* const a) {

    size_t i = 0;
    double sum = 0.0;

    assert(a);

    for (i = 0; i < n; ++i) {
        sum += a[i];
    }

    return sum / (double)n;
}

void
compute_mean_coord(
        const size_t n,
        const coord_t* const a,
        coord_t* out) {

    size_t i = 0;
    coord_t sum = {0.0, 0.0};

    assert(a);

    for (i = 0; i < n; ++i) {
        sum.x += a[i].x;
        sum.y += a[i].y;
    }

    out->x = sum.x / (double)n;
    out->y = sum.y / (double)n;
}

