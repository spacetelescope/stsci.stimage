/*
Copyright (C) 2008-2025 Association of Universities for Research in Astronomy (AURA)

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
*/

#ifndef _XY_COORD_MATCH_UTIL_H_
#define _XY_COORD_MATCH_UTIL_H_

#include <math.h>
#include <stdlib.h>

#include "lib/error.h"

/********************************************************************************
 MACROS
*/
#define DEGTORAD(a) (a * (M_PI / 180.0))
#define RADTODEG(a) (a * (180.0 / M_PI))
#if !defined(MIN)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#if !defined(MAX)
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define CLAMP_ABOVE(x, low)  (((x) < low) ? (low) : (x))
#define CLAMP_BELOW(x, high) (((x) > high) ? (high) : (x))

#define ABS(x) (((x) < 0) ? (-x) : (x))

#define MAX_DOUBLE 1.7976931348623158e+308
#define MIN_DOUBLE 2.2250738585072014e-308
#define EPS_DOUBLE 2.22e-16

#if defined(_MSC_VER)
typedef __int64 XY_COORD_MATCH_Int64;
#else
#if defined(_ISOC99_SOURCE)
typedef int64_t XY_COORD_MATCH_Int64;
#else
typedef long long XY_COORD_MATCH_Int64;
#endif
#endif

#if !defined(U64)
#define U64(u) (*(XY_COORD_MATCH_Int64 *) &(u))
#endif /* U64 */

#if !defined(isnan64)
#if !defined(_MSC_VER)
#define isnan64(u)                                                \
    (((U64(u) & 0x7ff0000000000000LL) == 0x7ff0000000000000LL) && \
     ((U64(u) & 0x000fffffffffffffLL) != 0))                      \
        ? 1                                                       \
        : 0
#else
#define isnan64(u)                                                  \
    (((U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64) && \
     ((U64(u) & 0x000fffffffffffffi64) != 0))                       \
        ? 1                                                         \
        : 0
#endif
#endif /* isnan64 */

#if !defined(isinf64)
#if !defined(_MSC_VER)
#define isinf64(u)                                                \
    (((U64(u) & 0x7ff0000000000000LL) == 0x7ff0000000000000LL) && \
     ((U64(u) & 0x000fffffffffffffLL) == 0))                      \
        ? 1                                                       \
        : 0
#else
#define isinf64(u)                                                  \
    (((U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64) && \
     ((U64(u) & 0x000fffffffffffffi64) == 0))                       \
        ? 1                                                         \
        : 0
#endif
#endif /* isinf64 */

#if !defined(isfinite64)
#if !defined(_MSC_VER)
#define isfinite64(u) (((U64(u) & 0x7ff0000000000000LL) != 0x7ff0000000000000LL)) ? 1 : 0
#else
#define isfinite64(u) (((U64(u) & 0x7ff0000000000000i64) != 0x7ff0000000000000i64)) ? 1 : 0
#endif
#endif /* isfinite64 */

#if !defined(notisfinite64)
#if !defined(_MSC_VER)
#define notisfinite64(u) (((U64(u) & 0x7ff0000000000000LL) == 0x7ff0000000000000LL)) ? 1 : 0
#else
#define notisfinite64(u) (((U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)) ? 1 : 0
#endif
#endif /* notisfinite64 */

/********************************************************************************
 STRUCTS
*/
typedef struct {
    double x;
    double y;
} coord_t;

typedef struct {
    const coord_t *l;
    const coord_t *r;
} coord_match_t;

typedef enum { xterms_none, xterms_half, xterms_full, xterms_LAST } xterms_e;

static inline int
coord_is_finite(const coord_t *const c)
{
    return isfinite(c->x) && isfinite(c->y);
}

void *
malloc_with_error(size_t size, xy_coord_match_error_t *error);

/**
Compute the combinatorial function which is defined as
   n! / ((n - ngroup)! * ngroup!)

The result will overflow 32 bits with n == 2346 and ngroup == 3
(though surely an allocation requesting that much memory would fail
much sooner).  It is up to the caller to ensure n is within range
*/
size_t
combinatorial(size_t n, size_t ngroup);

/**
Calculate the square of the Euclidean distance between two points.
*/
static inline double
euclid_distance2(const coord_t *const a, const coord_t *const b)
{
    double dx, dy;
    dx = b->x - a->x;
    dy = b->y - a->y;
    return dx * dx + dy * dy;
}

/**
Sort an array of doubles
*/
void
sort_doubles(
    const size_t n,
    /* Input/output */
    double *const a);

/**
Compute the mode of an array.  The mode is found by binning with a bin
size based on the data range over a fraction of the pixels about the
median and a bin step which may be smaller than the bin size.  If
there are too few points, the median is returned.  The input array
must be sorted.

@param n The size of the array

@param a An array of doubles.  Must be sorted.

@param min The minimum number of points

@param range Fraction of pixels around median to use.

@param bin Bin size for the mode search.

@param step Step size for the mode search.
*/
double
compute_mode(
    const size_t n, const double *const a, const size_t min, const double range, const double bin,
    const double step);

#endif /* _XY_COORD_MATCH_UTIL_H_ */
