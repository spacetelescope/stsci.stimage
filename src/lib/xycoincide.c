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
#include <string.h>

#include "lib/xycoincide.h"

size_t
xycoincide(
    const size_t ncoords,
    const coord_t* const * const input /*[ncoords]*/,
    const coord_t** const output /*[ncoords]*/,
    const double tolerance) {

    double tolerance2 = tolerance * tolerance;
    size_t nunique = ncoords;
    double distance = 0.0;
    double r2 = 0.0;
    size_t iprev = 0;
    size_t i = 0;

    assert(input);
    assert(output);

    if ((coord_t **)input != (coord_t **)output) {
        memcpy(output, input, sizeof(coord_t *) * ncoords);
    }

    for (iprev = 0; iprev < ncoords; ++iprev) {
        /* Jump to the next object if this one has been deleted,
           since all comparisons are invalid */
        if (output[iprev] == NULL) {
            continue;
        }

        for (i = iprev + 1; i < ncoords; ++i) {
            /* Skip to the next object if this one has been deleted */
            if (output[i] == NULL) {
                continue;
            }

            /* Check the tolerance limit in y and step to the next
               object if the bounds are exceeded */
            distance = output[i]->y - output[iprev]->y;
            r2 = distance * distance;
            if (r2 > tolerance2) {
                break;
            }

            /* Check the tolerance limit in x, and delete if too
               close */
            distance = output[i]->x - output[iprev]->x;
            r2 += distance * distance;
            if (r2 <= tolerance2) {
                /* Delete it */
                output[i] = NULL;
                --nunique;
            }
        }
    }

    /* Compress the array */
    if (nunique < ncoords) {
        iprev = 0;
        for (i = 0; i < ncoords; ++i) {
            if (output[i] != NULL) {
                output[iprev++] = output[i];
            }
        }
    }

    return nunique;
}
