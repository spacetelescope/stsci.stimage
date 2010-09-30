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

#include "immatch/xyxymatch.h"
#include "lib/lintransform.h"
#include "lib/xycoincide.h"
#include "lib/xysort.h"
#include "immatch/lib/triangles.h"
#include "immatch/lib/tolerance.h"

typedef struct {
    const coord_t*      ref;
    const coord_t*      input;
    size_t              noutput;
    size_t              outputp;
    xyxymatch_output_t* output;
} xyxymatch_callback_data_t;

static int
xyxymatch_callback(
        void* data,
        size_t ref_index,
        size_t input_index,
        stimage_error_t* error) {

    xyxymatch_callback_data_t* state = (xyxymatch_callback_data_t*)data;
    xyxymatch_output_t* entry;

    if (state->outputp >= state->noutput) {
        stimage_error_format_message(
            error,
            "Number of output coordinates exceeded allocation (%d)",
            state->noutput);
        return 1;
    }

    entry = &(state->output[state->outputp]);

    entry->coord     = state->input[input_index];
    entry->ref       = state->ref[ref_index];
    entry->coord_idx = input_index;
    entry->ref_idx   = ref_index;

    ++(state->outputp);

    return 0;
}

/** DIFF

The original takes lists of input, reference and output files.  This
(for now, until its determined insufficient) only takes a single array
of coordinates for each.  It seems that the original never really took
a list of reference files anyway.

This takes arrays of coordinates, rather than 2-dimensional arrays of
doubles.

    Because of this, there is no flexibility about where the columns
    lie (xcolumn, ycolumn, xrcolumn, yrcolumn parameters).  I am
    assuming that this sort of cleanup can be done more easily with
    Numpy slicing on the Python side.
 */

int
xyxymatch(
        const size_t ninput, const coord_t* const input /*[ninput]*/,
        const size_t nref, const coord_t* const ref /*[nref]*/,
        size_t* noutput, xyxymatch_output_t* const output /*[noutput]*/,
        const coord_t* origin, /* good default: 0.0, 0.0 */
        const coord_t* mag, /* good default: 1.0, 1.0 */
        const coord_t* rotation, /* good default: 0.0, 0.0 */
        const coord_t* ref_origin, /* good default: 0.0, 0.0 */
        const xyxymatch_algo_e algorithm,
        const double tolerance,
        const double separation, /* good default: 9.0 */
        const size_t nmatch,
        const double maxratio,
        const size_t nreject,
        stimage_error_t* const error) {

    static const coord_t      DEFAULT_ORIGIN     = {0.0, 0.0};
    static const coord_t      DEFAULT_MAG        = {1.0, 1.0};
    static const coord_t      DEFAULT_ROTATION   = {0.0, 0.0};
    static const coord_t      DEFAULT_REF_ORIGIN = {0.0, 0.0};
    coord_t*                  input_trans        = NULL;
    const coord_t**           input_trans_sorted = NULL;
    size_t                    ninput_unique      = ninput;
    const coord_t**           ref_sorted         = NULL;
    size_t                    nref_unique        = nref;
    lintransform_t            lintransform;
    xyxymatch_callback_data_t state;
    int                       status             = 1;

    /****************************************
     CHECK ARGUMENTS
    */
    assert(input);
    assert(ref);
    assert(output);
    assert(error);
    assert(*noutput > 0);

    if (ninput == 0) {
        stimage_error_set_message(error, "The input coordinate list is empty");
        goto exit;
    }

    if (nref == 0) {
        stimage_error_set_message(error, "The reference coordinate list is empty");
        goto exit;
    }

    if (algorithm >= xyxymatch_algo_LAST || algorithm < 0) {
        stimage_error_set_message(error, "Invalid algorithm specified");
        goto exit;
    }

    if (origin == NULL) {
        origin = &DEFAULT_ORIGIN;
    }

    if (mag == NULL) {
        mag = &DEFAULT_MAG;
    }

    if (rotation == NULL) {
        rotation = &DEFAULT_ROTATION;
    }

    if (ref_origin == NULL) {
        ref_origin = &DEFAULT_REF_ORIGIN;
    }

    /****************************************
     PREPARE REFERENCE COORDINATES
    */
    ref_sorted = malloc_with_error(nref * sizeof(coord_t*), error);
    if (ref_sorted == NULL) goto exit;

    xysort(nref, ref, ref_sorted);
    nref_unique = xycoincide(nref, ref_sorted, ref_sorted, separation);

    /****************************************
     DETERMINE INITIAL TRANSFORM
    */
    compute_lintransform(*origin, *mag, *rotation, *ref_origin, &lintransform);

    /****************************************
     PREPARE INPUT COORDINATES
    */
    input_trans = malloc_with_error(ninput * sizeof(coord_t), error);
    if (input_trans == NULL) goto exit;

    input_trans_sorted = malloc_with_error(ninput * sizeof(coord_t*), error);
    if (input_trans_sorted == NULL) goto exit;

    apply_lintransform(&lintransform, ninput, input, input_trans);
    xysort(ninput, input_trans, input_trans_sorted);
    ninput_unique = xycoincide(ninput, input_trans_sorted, input_trans_sorted, separation);

    /****************************************
     RUN THE DESIRED ALGORITHM
    */
    state.ref = ref;
    state.input = input;
    state.noutput = *noutput;
    state.outputp = 0;
    state.output = output;

    switch (algorithm) {
    case xyxymatch_algo_tolerance:
        if (match_tolerance(
                nref_unique, ref, ref_sorted,
                ninput_unique, input_trans, input_trans_sorted,
                tolerance,
                xyxymatch_callback, &state,
                error)) goto exit;
        *noutput = state.outputp;
        break;
    case xyxymatch_algo_triangles:
        if (match_triangles(
                nref, nref_unique, ref, ref_sorted,
                ninput, ninput_unique, input_trans, input_trans_sorted,
                nmatch, tolerance, maxratio, nreject,
                &xyxymatch_callback, &state,
                error)) goto exit;
        *noutput = state.outputp;
        break;
    case xyxymatch_algo_LAST:
    default:
        stimage_error_set_message(error, "Invalid algorithm");
        goto exit;
    }

    status = 0;

exit:

    free(ref_sorted);
    free(input_trans_sorted);
    free(input_trans);
    return status;
}



