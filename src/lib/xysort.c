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

#include "lib/xysort.h"

/* DIFF: The documentation of the original function (part of rg_sort)
   claims that it sorts in y and then x.  What it in fact does is sort
   on y and then sort x within all equal values of y.  (The net result
   of sorting by x and then y, contrary to the documentation).  We
   should get the same result in a single pass by sorting with y as a
   primary key and x as secondary.

   Whereas the original function sorts the data as well as a set of
   indices, we treat the data as constant and sort pointers to the
   data (which can later be used as indices using pointer
   subtraction).  This allows us to use the C stdlib qsort as-is
   rather than writing our own quicksort algorithm.
*/

static int
xysort_compare(const void* ap, const void* bp) {
    const coord_t* a = *(const coord_t**)ap;
    const coord_t* b = *(const coord_t**)bp;

    if (a->y < b->y) {
        return -1;
    } else if (a->y > b->y) {
        return 1;
    } else {
        if (a->x < b->x) {
            return -1;
        } else if (a->x > b->x) {
            return 1;
        } else {
            return 0;
        }
    }
}

void
xysort(
    const size_t ncoords,
    const coord_t* const coords /* [ncoords] */,
    const coord_t** const coords_ptr /* [ncoords] */) {

    size_t i;

    assert(coords);
    assert(coords_ptr);

    /* Fill the pointer array */
    for (i = 0; i < ncoords; ++i) {
        coords_ptr[i] = (coord_t*)coords + i;
    }

    qsort(coords_ptr, ncoords, sizeof(coord_t**), &xysort_compare);
}
