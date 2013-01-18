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
#include <stdio.h>

#include "immatch/lib/triangles.h"

int
vote_triangle_matches(
        const size_t nleft,
        const coord_t* const left,
        const size_t nright,
        const coord_t* const right,
        const size_t ntriangle_matches,
        const triangle_match_t* const triangle_matches,
        size_t* ncoord_matches,
        const coord_t** const refcoord_matches,
        const coord_t** const inputcoord_matches,
        stimage_error_t* const error) {

    typedef size_t vote_t;

    vote_t*           votes        = NULL;
    vote_t            maxvote      = 0;
    vote_t            half_maxvote = 0;
    vote_t            row_maxvote  = 0;
    vote_t            row_2maxvote = 0;
    vote_t            vote         = 0;
    const triangle_t* r_tri        = NULL;
    const triangle_t* l_tri        = NULL;
    const coord_t*    r_coord      = NULL;
    const coord_t*    l_coord      = NULL;
    size_t            li           = 0;
    size_t            ri           = 0;
    size_t            ri2          = 0;
    size_t            ncount       = 0;
    size_t            i            = 0;
    size_t            j            = 0;
    int               status       = 1;

    assert(triangle_matches);
    assert(ncoord_matches);
    assert(refcoord_matches);
    assert(inputcoord_matches);
    assert(error);

    /* Since the vote tallies are rather sparse, this uses a nested
       map from reference coordinates to a map from input coordinates
       to vote counts. */

    #define VOTE(li, ri) votes[(ri) * nleft + (li)]

    votes = malloc(nleft * nright * sizeof(size_t));
    if (votes == NULL) {
        goto exit;
    }

    for (i = 0; i < nleft * nright; ++i) {
        votes[i] = 0;
    }

    /* Accumulate the votes */
    for (i = 0; i < ntriangle_matches; ++i) {
        r_tri = triangle_matches[i].r;
        l_tri = triangle_matches[i].l;

        for (j = 0; j < 3; ++j) {
            l_coord = l_tri->vertices[j];
            r_coord = r_tri->vertices[j];
            li = l_coord - left;
            assert(li >= 0 && li < nleft);
            ri = r_coord - right;
            assert(ri >= 0 && ri < nright);
            vote = ++VOTE(li, ri);
            if (maxvote < vote) {
                maxvote = vote;
            }
        }
    }

    if (maxvote == 0) {
        *ncoord_matches = 0;
        status = 0;
        goto exit;
    }

    half_maxvote = maxvote >> 1;
    ncount = 0;
    for (ri = 0; ri < nright; ++ri) {
        r_coord = right + ri;

        row_maxvote = 0;
        row_2maxvote = 0;
        l_coord = NULL;
        for (li = 0; li < nleft; ++li) {
            vote = VOTE(li, ri);
            if (vote > row_maxvote) {
                row_2maxvote = row_maxvote;
                row_maxvote = vote;
                l_coord = left + li;
            }
        }

        /* Reject points which

           1. Have no votes, or less than half the number of maximum
              votes (this is handled by the same test, since hmaxvotes
              >= 0),

           2. Have a tie

           3. Which only have a single vote if we expect more
        */
        if (row_maxvote <= half_maxvote ||
            row_maxvote == row_2maxvote ||
            (row_maxvote == 1 && (maxvote > 1 || ntriangle_matches > 1))) {
            continue;
        }

        /* Remove all future matches involving the input coord, so it
           won't be matched twice. */
        for (ri2 = ri; ri2 < nright; ++ri2) {
            li = l_coord - left;
            VOTE(li, ri) = 0;
        }

        #ifndef NDEBUG
            if (ncount >= *ncoord_matches) {
                stimage_error_format_message(
                    error,
                    "Found more coordinate matches than was allocated for\n");
                goto exit;
            }
        #endif

        refcoord_matches[ncount]   = l_coord;
        inputcoord_matches[ncount] = r_coord;
        ++ncount;
    }

    *ncoord_matches = ncount;

    status = 0;

 exit:

    free(votes);

    return status;
}
