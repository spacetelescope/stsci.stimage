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

#ifndef _STIMAGE_XYXYMATCH_H_
#define _STIMAGE_XYXYMATCH_H_

#include "lib/util.h"

typedef struct {
    coord_t coord;
    size_t  coord_idx;
    coord_t ref;
    size_t  ref_idx;
} xyxymatch_output_t;

typedef enum {
    xyxymatch_algo_tolerance,
    xyxymatch_algo_triangles,
    xyxymatch_algo_LAST
} xyxymatch_algo_e;

/**
xyxymatch

@param ninput The number of input coordinates

@param input Array of input coordinates

@param nref The number of reference coordinates

@param ref Array of reference coordinates

@param noutput input: The number of output coordinate pairs allocated
       output: The numbe of output coordinate pairs found

@param output Array of xyxymatch_output_t objects to store the output
       information.  Should be allocated to the same size as the
       number of input coordinates, but it doesn't have to be.  If
       the allocated space is not big enough for all the results, an
       error will be emitted.

@param origin The origin of the input coordinate system.  If NULL,
       assume (0.0, 0.0)

@param mag The scale factor in reference pixels per input pixels.  If
       NULL, assume (1.0, 1.0)

@param rotation The rotation in reference pixels per input pixels.  If
       NULL, assume (0.0, 0.0)

@param ref_origin The origin of the reference coordinate system.  If NULL,
       assume (0.0, 0.0)

@param algorithm The matching algorithm.  The choices are:

    - xyxymatch_algo_tolerance: A linear transformation is applied to
      the input coordinate list, the transformed input list and the
      reference list are sorted, points which are too close together
      are removed, and the input coordinates which most closely match
      the reference coordinates within the user-specified tolerance
      are determined.  The tolerance algorithm requires an initial
      estimate for the linear transformation.  This estimate can be
      derived from a set of tie points, or by setting the parameters
      origin, mag, rotation and ref_origin.  Assuming that well-chosen
      tie points are provided, the tolerance algorithm functions well
      in the presence of any shifts, axis flips, x and y scale
      changes, rotations, and axis skew, between the two coordinate
      systems.  The algorithm is sensitive to higher-order distortion
      terms in the coordinate transformation.

    - xyxymatch_algo_triangles: A linear transformation is applied to
      the input coordinate list, the transformed input list and the
      reference list are sorted, points which are too close together
      are removed, and the input coordinates are matches to the
      reference coordinates using a triangle pattern matching
      technique and the user-specified tolerance parameter.  The
      triangles pattern matching algorithm does not require prior
      knowledge of the linear transformation, although it will use one
      if one is supplied.  The algorithm functions well in the
      presence of any shifts, axis flips, magnification and rotation
      between the two coordinate systems as long as both lists have a
      reasonable number of objects in common and the errors in the
      computed coordinates are small.  However, since the algorithm
      depends on comparisons of similar triangles, it is sensitive to
      differences in the x and y coordinate scales, any skew between
      the x and y axes, and higher order distortion terms in the
      coordinate transformation.

@param tolerance The matching tolerance in pixels.

@param separation The minimum separation for objects in the input and
reference coordinate lists.  Objects closer together than separation
pixels are removed from the input and reference coordinate lists prior
to matching. (9.0)

@param nmatch The maximum number of reference and input coordinates
used by the xyxymatch_algo_triangles pattern matching algorithm.  If
either list contains more coordinates than nmatch, the lists are
subsampled.  nmatch should be kept small as the computation and memory
requirements of the triangles algorithm depend on a high power of
lengths of the respective lists.

@param maxratio The maximum ratio of the longest to shortest side of the
triangles generated by the triangles pattern matching algorithm.
Triangles with computed longest to shortest side ratios > ratio are
rejected from the pattern matching algorithm.  ratio should never be
set higher than 10.0 but may be set as low as 5.0.

@param nreject The maximum number of rejection iterations for the
triangles pattern matching algorithm.

@return Non-zero on error
 */
int
xyxymatch(
    const size_t ninput, const coord_t* const input /*[ninput]*/,
    const size_t nref, const coord_t* const ref /*[nref]*/,
    size_t* noutput, xyxymatch_output_t* const output /*[noutput]*/,
    const coord_t* const origin, /* good default: 0.0, 0.0 */
    const coord_t* const mag, /* good default: 1.0, 1.0 */
    const coord_t* const rotation, /* good default: 0.0, 0.0 */
    const coord_t* const ref_origin, /* good default: 0.0, 0.0 */
    const xyxymatch_algo_e algorithm,
    const double tolerance,
    const double separation, /* good default: 9.0 */
    const size_t nmatch,
    const double maxratio,
    const size_t nreject,
    stimage_error_t* const error);

#endif /* _STIMAGE_XYXYMATCH_H_ */
