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

#ifndef _STIMAGE_SURFACE_H_
#define _STIMAGE_SURFACE_H_

#include "lib/xybbox.h"
#include "lib/error.h"
#include "lib/util.h"

typedef enum {
    surface_type_polynomial,
    surface_type_legendre,
    surface_type_chebyshev,
    surface_type_LAST
} surface_type_e;

typedef struct {
    surface_type_e   type;
    size_t           xorder;
    size_t           yorder;
    size_t           nxcoeff;
    size_t           nycoeff;
    xterms_e         xterms;
    size_t           ncoeff;
    double           xrange;
    double           xmaxmin;
    double           yrange;
    double           ymaxmin;
    bbox_t           bbox;
    double*          matrix;        /* [ncoeff ** 2] */
    double*          cholesky_fact; /* [ncoeff ** 2] */
    double*          vector;        /* [ncoeff] */
    double*          coeff;         /* [ncoeff] */
    size_t           npoints;
} surface_t;

/**
Initialize a surface_t object.

@param s A pointer to a surface object

@param function The surface type

@param xorder x-order of the surface to be fit

@param yorder y-order of the surface to be fit

@param xterms presence of cross terms (when xterms != 0)

@param bbox Bounding box

@param error Error object
*/
int
surface_init(
        surface_t* const s,
        const surface_type_e function,
        const int xorder,
        const int yorder,
        const xterms_e xterms,
        const bbox_t* const bbox,
        stimage_error_t* const error);

/**
 Simply mark a surface object as uninitialized.
*/
int
surface_new(
        surface_t* const s);

/**
Free the allocated memory in a surface object.
*/
void
surface_free(
        surface_t* const s);

/**
Copy the surface into a new struct
*/
int
surface_copy(
        const surface_t* const s,
        surface_t* const d,
        stimage_error_t* const error);

/**
Zero the accumulators before doing a new fit in accumulate mode.  The
inner products of the basis functions are accumulated in the s->ncoeff
** 2 array matrix, while the inner products of the basis functions and
the data ordinates are accumulated in the s->ncoeff vector.
*/
int
surface_zero(
        surface_t* const s,
        stimage_error_t* const error);

/**
 Print the contents of a surface structure
*/
void
surface_print(
        const surface_t* const s);

#endif
