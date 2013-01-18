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

#include "immatch/lib/tolerance.h"

int
match_tolerance(
        const size_t nref,
        const coord_t* const ref,
        const coord_t* const * const ref_sorted,
        const size_t ninput,
        const coord_t* const input,
        const coord_t* const * const input_sorted,
        const double tolerance,
        coord_match_callback_t* callback,
        void* callback_data,
        stimage_error_t* const error) {

    const double   tolerance2  = tolerance*tolerance;
    size_t         rp          = 0;
    size_t         blp         = 0;
    size_t         lp          = 0;
    size_t         input_index = 0;
    size_t         ref_index   = 0;
    double         dx, dy, rmax2, r2;
    const coord_t* rmatch;
    const coord_t* lmatch;

    assert(ref);
    assert(ref_sorted);
    assert(input);
    assert(input_sorted);
    assert(callback);
    assert(error);

    for (rp = 0; rp < nref; ++rp) {
        /* Compute the start of the search range */
        for (; blp < ninput; ++blp) {
            dy = ref_sorted[rp]->y - input_sorted[blp]->y;
            if (dy < tolerance) {
                break;
            }
        }

        /* Break if the end of the input list is reached */
        if (blp >= ninput) {
            break;
        }

        /* If one is outside the tolerance limits, skip to next
           reference object. */
        if (dy < -tolerance) {
            continue;
        }

        /* Find the closest match to the reference object */
        rmax2 = tolerance2;
        rmatch = NULL;
        lmatch = NULL;
        for (lp = blp; lp < ninput; ++lp) {
            /* Compute the distance between the two points */
            dy = ref_sorted[rp]->y - input_sorted[lp]->y;
            if (dy < -tolerance) {
                break;
            }
            dx = ref_sorted[rp]->x - input_sorted[lp]->x;
            r2 = dx*dx + dy*dy;

            /* A match has been found */
            if (r2 <= rmax2) {
                rmax2 = r2;
                rmatch = ref_sorted[rp];
                lmatch = input_sorted[lp];
            }
        }

        /* A match was found, so write the results to the output array */
        if (rmatch != NULL && lmatch != NULL) {
            ref_index = rmatch - ref;
            input_index = lmatch - input;

            if (callback(callback_data, ref_index, input_index, error)) {
                return 1;
            }
        }
    }

    return 0;
}
