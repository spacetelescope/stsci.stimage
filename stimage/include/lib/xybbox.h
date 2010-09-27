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

#ifndef _STIMAGE_XYBBOX_H_
#define _STIMAGE_XYBBOX_H_

#include "lib/util.h"

typedef struct {
    coord_t min;
    coord_t max;
} bbox_t;

/**
 Initializes a bbox by filling it with NaNs
*/
void
bbox_init(
        bbox_t* const bbox);

/**
 Print the value of the bbox to stdout
*/
void
bbox_print(
        const bbox_t* const bbox);

/**
Check that the bbox is valid, that is min <= max
*/
static inline int
bbox_is_valid(
    const bbox_t* const b) {
    return (b->min.x <= b->max.x &&
            b->min.y <= b->max.y);
}

/**
Copies the contents of one bbox to another
*/
static inline void
bbox_copy(
    const bbox_t* const src,
    bbox_t* const dest) {
    dest->min.x = src->min.x;
    dest->min.y = src->min.y;
    dest->max.x = src->max.x;
    dest->max.y = src->max.y;
}

/**
Determines if a coordinate is within a bbox.
*/
static inline int
coord_in_bbox(
    const coord_t* const c,
    const bbox_t* const b) {
    return (c->x >= b->min.x && c->x <= b->max.x &&
            c->y >= b->min.y && c->y <= b->max.y);
}

/**
Given parallels lists of input and reference coordinates, returns new
lists containing only the pairs where the reference coordinate is
inside the given bounding box.

input_in_bbox and ref_in_bbox should be pre-allocated to ncoord
coordinates.
 */
size_t
limit_to_bbox(
    size_t ncoord,
    const coord_t* const input,
    const coord_t* const ref,
    const bbox_t* const bbox,
    coord_t* const input_in_bbox,
    coord_t* const ref_in_bbox);

/**
Determines the that contains the given set of coordinates.

Any finite values in the bbox will be maintained, so to use this from
scratch, you will usually want to set all values to NaN.
 */
void
determine_bbox(
    size_t n,
    const coord_t* const a,
    bbox_t* const bbox);

/**
Makes the bbox non-singular
 */
void
bbox_make_nonsingular(
    bbox_t* const bbox);

#endif /* _STIMAGE_XYBBOX_H_ */
