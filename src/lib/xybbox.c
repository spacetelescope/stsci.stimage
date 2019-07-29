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
#include <math.h>
#include <stdio.h>

#include "lib/xybbox.h"

void
bbox_init(
        bbox_t* const bbox) {

    double my_nan;

    assert(bbox);

    my_nan = fmod(1.0, 0.0);

    bbox->min.x = my_nan;
    bbox->min.y = my_nan;
    bbox->max.x = my_nan;
    bbox->max.y = my_nan;
}

void
bbox_print(
        const bbox_t* const bbox) {

    printf("(%f, %f)--(%f, %f)",
           bbox->min.x, bbox->min.y,
           bbox->max.x, bbox->max.y);
}

/* was geo_rdxyd */
size_t
limit_to_bbox(
        size_t ncoord,
        const coord_t* const input,
        const coord_t* const ref,
        const bbox_t* const bbox,
        coord_t* const input_in_bbox,
        coord_t* const ref_in_bbox) {

    size_t i = 0;
    size_t nout = 0;

    assert(input);
    assert(ref);
    assert(bbox);
    assert(input_in_bbox);
    assert(ref_in_bbox);
    assert(bbox_is_valid(bbox));

    for (i = 0; i < ncoord; ++i) {
        if (isfinite(bbox->min.x) && ref[i].x < bbox->min.x) {
            continue;
        }
        if (isfinite(bbox->max.x) && ref[i].x > bbox->max.x) {
            continue;
        }
        if (isfinite(bbox->min.y) && ref[i].y < bbox->min.y) {
            continue;
        }
        if (isfinite(bbox->max.y) && ref[i].y > bbox->max.y) {
            continue;
        }

        input_in_bbox[nout].x = input[i].x;
        input_in_bbox[nout].y = input[i].y;
        ref_in_bbox[nout].x   = ref[i].x;
        ref_in_bbox[nout].y   = ref[i].y;
        ++nout;

        assert(nout < ncoord);
    }

    return nout;
}

void
determine_bbox(
        size_t n,
        const coord_t* const a,
        bbox_t* const bbox) {

    size_t i = 0;

    assert(a);
    assert(bbox);

    if (!isfinite(bbox->min.x)) {
        bbox->min.x = MAX_DOUBLE;
    }
    if (!isfinite(bbox->min.y)) {
        bbox->min.y = MAX_DOUBLE;
    }
    if (!isfinite(bbox->max.x)) {
        bbox->max.x = -MAX_DOUBLE;
    }
    if (!isfinite(bbox->max.y)) {
        bbox->max.y = -MAX_DOUBLE;
    }

    for (i = 0; i < n; ++i) {
        if (isfinite(a[i].x)) {
            if (a[i].x < bbox->min.x) {
                bbox->min.x = a[i].x;
            }
            if (a[i].x > bbox->max.x) {
                bbox->max.x = a[i].x;
            }
        }

        if (isfinite(a[i].y)) {
            if (a[i].y < bbox->min.y) {
                bbox->min.y = a[i].y;
            }
            if (a[i].y > bbox->max.y) {
                bbox->max.y = a[i].y;
            }
        }
    }
}

void
bbox_make_nonsingular(
        bbox_t* const bbox) {

    assert(bbox);

    if (double_approx_equal(bbox->min.x, bbox->max.x)) {
        bbox->min.x -= 0.5;
        bbox->max.x += 0.5;
    }

    if (double_approx_equal(bbox->min.y, bbox->max.y)) {
        bbox->min.y -= 0.5;
        bbox->max.y += 0.5;
    }
}
