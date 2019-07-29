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

#ifndef _STIMAGE_UTIL_H_
#define _STIMAGE_UTIL_H_

#ifdef _WIN32
	/*
	* Make inline go away on Windows (for now).
	* The syntax is subtly different from gcc in some way.  Mike
	* says he did the inline because of benchmarking, but it isn't
	* all that important.
	*	- Mark S 2012-02-14
	*/
#define inline
#endif

#include <math.h>
#include <stdlib.h>

#include "lib/error.h"

/********************************************************************************
 MACROS
*/
#define DEGTORAD(a) (a * (M_PI / 180.0))
#define RADTODEG(a) (a * (180.0 / M_PI))
#if !defined(MIN)
  #define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#endif

#if !defined(MAX)
  #define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#endif

#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define CLAMP_ABOVE(x, low)  (((x) < low) ? (low) : (x))
#define CLAMP_BELOW(x, high)  (((x) > high) ? (high) : (x))

#define ABS(x) (((x) < 0) ? (-x) : (x))

#define MAX_DOUBLE 1.7976931348623158e+308
#define MIN_DOUBLE 2.2250738585072014e-308
#define EPS_DOUBLE 2.22e-16

#if defined(_MSC_VER)
    typedef __int64                  STIMAGE_Int64;
#else
    #if defined(_ISOC99_SOURCE)
        typedef int64_t                  STIMAGE_Int64;
    #else
        typedef long long                STIMAGE_Int64;
    #endif
#endif

#if !defined(U64)
    #define U64(u) (* (STIMAGE_Int64 *) &(u) )
#endif /* U64 */

#if !defined(isnan64)
    #if !defined(_MSC_VER)
        #define isnan64(u) \
            ( (( U64(u) & 0x7ff0000000000000LL)  == 0x7ff0000000000000LL)  && ((U64(u) &  0x000fffffffffffffLL) != 0)) ? 1:0
    #else
        #define isnan64(u) \
            ( (( U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)  && ((U64(u) & 0x000fffffffffffffi64) != 0)) ? 1:0
    #endif
#endif /* isnan64 */

#if !defined(isinf64)
    #if !defined(_MSC_VER)
        #define isinf64(u) \
            ( (( U64(u) & 0x7ff0000000000000LL)  == 0x7ff0000000000000LL)  && ((U64(u) &  0x000fffffffffffffLL) == 0)) ? 1:0
    #else
        #define isinf64(u) \
            ( (( U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)  && ((U64(u) & 0x000fffffffffffffi64) == 0)) ? 1:0
    #endif
#endif /* isinf64 */

#if !defined(isfinite64)
    #if !defined(_MSC_VER)
        #define isfinite64(u) \
            ( (( U64(u) & 0x7ff0000000000000LL)  != 0x7ff0000000000000LL)) ? 1:0
    #else
        #define isfinite64(u) \
            ( (( U64(u) & 0x7ff0000000000000i64) != 0x7ff0000000000000i64)) ? 1:0
    #endif
#endif /* isfinite64 */

#if !defined(notisfinite64)
    #if !defined(_MSC_VER)
        #define notisfinite64(u) \
            ( (( U64(u) & 0x7ff0000000000000LL)  == 0x7ff0000000000000LL)) ? 1:0
    #else
        #define notisfinite64(u) \
            ( (( U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)) ? 1:0
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
    const coord_t* l;
    const coord_t* r;
} coord_match_t;

typedef enum {
    xterms_none,
    xterms_half,
    xterms_full,
    xterms_LAST
} xterms_e;

static inline int
coord_is_finite(
    const coord_t* const c) {
    return isfinite(c->x) && isfinite(c->y);
}

void *
malloc_with_error(
        size_t size,
        stimage_error_t* error);

void *
calloc_with_error(
        size_t nmemb,
        size_t size,
        stimage_error_t* error);

/**
Compute the factorial of n.

This function will overflow for n >= 21, and it is up to the caller to
ensure n is in the proper range.
*/
STIMAGE_Int64
factorial(
        size_t n);

/**
Compute the combinatorial function which is defined as
   n! / ((n - ngroup)! * ngroup!)

The result will overflow 32 bits with n == 2346 and ngroup == 3
(though surely an allocation requesting that much memory would fail
much sooner).  It is up to the caller to ensure n is within range
*/
size_t
combinatorial(
        size_t n,
        size_t ngroup);

/**
Calculate the square of the Euclidean distance between two points.
*/
static inline double
euclid_distance2(
        const coord_t* const a,
        const coord_t* const b) {
    double dx, dy;
    dx = b->x - a->x;
    dy = b->y - a->y;
    return dx*dx + dy*dy;
}

/**
Sort an array of doubles
*/
void
sort_doubles(
        const size_t n,
        /* Input/output */
        double* const a);

/**
Normalize a double precision number x to the value normx, in the
range [1-10).  expon is returned such that

    x = normx * (10.0d0 ** expon).
*/
void
double_normalize(
        const double x,
        /* Output */
        double* const normx,
        int* const expon);

/**
Compare two double precision numbers for equality to within the
machine precision for doubles.  A simple comparison of the difference
of the two numbers with the machine epsilon does not suffice unless
the numbers are first normalized to near 1.0, the constant used to
compute the machine epsilon (epsilon is the smallest number such that
1.0 + epsilon > 1.0).
*/
int
double_approx_equal(
        const double x,
        const double y);

/**
Compute the mean of an array
*/
double
compute_mean(
        const size_t n,
        const double* const a);

/**
Compute the mean values of an array of coordinates
*/
void
compute_mean_coord(
        const size_t n,
        const coord_t* const a,
        coord_t* const out);

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
        const size_t n,
        const double* const a,
        const size_t min,
        const double range,
        const double bin,
        const double step);

#endif /* _STIMAGE_UTIL_H_ */
