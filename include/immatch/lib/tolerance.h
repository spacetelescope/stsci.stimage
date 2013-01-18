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

#ifndef _STIMAGE_XYINTERSECT_H_
#define _STIMAGE_XYINTERSECT_H_

#include "lib/util.h"
#include "immatch/lib/match_util.h"

/**
Given two lists of coordinates, finds pairs that are within a certain
tolerance.  For each pair, a callback is called, allowing the caller
to deal with the results in a custom way.

@param nref The number of reference coordinates (specifically, the
length of ref_sorted, not ref).

@param ref A list of reference coordinates

@param ref_sorted A list of pointers to reference coordinates that have
been sorted with xysort and culled with xycoincide.

@param ninput The number of input coordinates (specifically, the
length of input_sorted, not input)

@param input A list of input coordinates

@param input_sorted A list of pointers to input coordinates that have
been sorted with xysort and culled with xycoincide.

@param tolerance The maximum distance to be considered a match

@param callback Called for every matching pair.  Its arguments are
(data, ref_index, input_index, error).  data is always whatever
callback_data is.  ref_index is the index in the original ref array to
the coordinate.  input_index is the index in the original input array
to the coordinate.  error is an error object in case an error needs to
be returned from the callback.

@param callback_data A void* to private data required by the given
callback.

@param error Set to a meaningful message if an error occurred.

@return Non-zero in case of error.
*/
int
match_tolerance(
        const size_t                 nref,
        const coord_t* const         ref,
        const coord_t* const * const ref_sorted,
        const size_t                 ninput,
        const coord_t* const         input,
        const coord_t* const * const input_sorted,
        const double                 tolerance,
        coord_match_callback_t*      callback,
        void*                        callback_data,
        stimage_error_t* const       error);

#endif /* _STIMAGE_XYINTERSECT_H_ */
