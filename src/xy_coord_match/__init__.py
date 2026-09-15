# Copyright (C) 2008-2025 Association of Universities for Research in Astronomy (AURA)

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.

#     3. The name of AURA and its representatives may not be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

from __future__ import absolute_import
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("xy_coord_match")
except PackageNotFoundError:
    # package is not installed
    pass

import xy_coord_match._c_xy_coord_match as _c_xy_coord_match


def xyxymatch(
    input,
    ref,
    origin=(0.0, 0.0),
    mag=(1.0, 1.0),
    rotation=(0.0, 0.0),
    ref_origin=(0.0, 0.0),
    algorithm="tolerance",
    tolerance=1.0,
    separation=9.0,
    nmatch=30,
    maxratio=10.0,
    nreject=10,
):
    """
    Match pixels coordinate lists using various methods.

    `xyxymatch` matches the *x* and *y* coordinates in the reference
    coordinate list reference to the corresponding *x* and *y*
    coordinates in the input coordinate list input to within a user
    specified tolerance *tolerance*, and returns the matched
    coordinates in a structured array.

    `xyxymatch` matches the coordinate lists by:

    1. computing an initial guess at the linear transformation
       required to match the input coordinate system to the reference
       coordinate system

    2. applying the computed transformation to the input coordinates

    3. sorting the reference and input coordinates and removing points
       with a minimum separation specified by the parameter separation
       from both lists

    4. matching the two lists using either the "tolerance" or
       "triangles" algorithm

    5. storing the matched list to the output array

    The coordinate lists are matched using the algorithm specified by
    the *algorithm* parameter.

    - If *algorithm* is "tolerance", `xyxymatch` searches the sorted
      transformed input coordinate list for the object closest to the
      current reference object within the matching tolerance
      *tolerance*. The major advantage of the "tolerance" algorithm is
      that it can deal with *x* and *y* scale differences and axis
      skew in the coordinate transformation. The major disadvantage is
      that the user must supply tie point information in all but the
      simplest case of small *x* and *y* shifts between the input and
      reference coordinate systems.

    - If *algorithm* is "triangles", `xyxymatch` constructs a list of
      triangles using up to *nmatch* reference coordinates and
      transformed input coordinates, and performs a pattern matching
      operation on the resulting triangle lists. If the number of
      coordinates in both lists is less than *nmatch*, the entire list
      is matched using the "triangles" algorithm directly, otherwise
      the "triangles" algorithm is used to estimate a new linear
      transformation, the input coordinate list is transformed using
      the new transformation, and the entire list is matched using the
      "tolerance" algorithm. The major advantage of the "triangles"
      algorithm is that it requires no tie point information from the
      user. The major disadvantages are that it is sensitive to *x*
      and *y* scale differences and axis skews between the input and
      reference coordinate systems and can be computationally
      expensive.

      The "triangles" algorithm uses a sophisticated pattern matching
      technique which requires no tie point information from the
      user. It is expensive computationally and hence is restricted to
      a maximum of *nmatch* objects from the reference and input
      coordinate lists.

      The "triangles" algorithm first generates a list of all the
      possible triangles that can be formed from the points in each
      list. For a list of *nmatch* points, this number is the
      combinatorial factor ``nmatch! / [(nmatch-3)! * 3!]`` or
      ``nmatch * (nmatch-1) * (nmatch-2) / 6``. The length of the
      perimeter, ratio of longest to shortest side, cosine of the
      angle between the longest and shortest side, the tolerances in
      the latter two quantities and the direction of the arrangement
      of the vertices of each triangle are computed and stored in a
      table. Triangles with vertices closer together than *tolerance*
      or with a ratio of the longest to shortest side greater than
      *ratio* are discarded. The remaining triangles are sorted in
      order of increasing ratio. A sort merge algorithm is used to
      match the triangles using the ratio and cosine information, the
      tolerances in these quantities, and the maximum tolerances for
      both lists. Next the ratios of the perimeters of the matched
      triangles are compared to the average ratio for the entire list,
      and triangles which deviate too widely from the mean are
      discarded. The number of triangles remaining are divided into
      the number which match in the clockwise sense and the number
      which match in the counter-clockwise sense. Those in the
      minority category are eliminated. The rejection step can be
      repeated up to *nreject* times or until no more rejections occur
      -- whichever comes first. The last step in the algorithm is a
      voting procedure in which each remaining matched triangle casts
      three votes, one for eached matched pair of vertices. Points
      which have fewer than half the maximum number of votes are
      discarded. The final set of matches are written to the output
      file.

      The "triangles" algorithm functions well when the reference and
      input coordinate lists have a sufficient number of objects
      (~50%, in some cases as low as 25%) of their objects in common,
      any distortions including *x* and *y* scale differences and skew
      between the two systems are small, and the random errors in the
      coordinates are small. Increasing the value of the *tolerance*
      parameter will increase the ability to deal with distortions but
      will also produce more false matches.

    **Parameters:**

    - *input*: Array of input coordinates. (Must be an Nx2 array).

    - *ref*: Array of reference coordinates. (Must be an Nx2 array).

    - *origin*: The origin of the input coordinate system.  Default:
      (0.0, 0.0)

    - *mag*: The scale factor in reference pixels per input pixels.
      Default: (1.0, 1.0)

    - *ref_origin*: The origin of the reference coordinate system.
      Default: (0.0, 0.0)

    - *algorithm*: The matching algorithm.  The choices are:

      - ``'tolerance'``: A linear transformation is applied to the
        input coordinate list, the transformed input list and the
        reference list are sorted, points which are too close together
        are removed, and the input coordinates which most closely
        match the reference coordinates within the user-specified
        tolerance are determined.  The tolerance algorithm requires an
        initial estimate for the linear transformation.  This estimate
        can be derived from a set of tie points, or by setting the
        parameters *origin*, *mag*, *rotation* and *ref_origin*.
        Assuming that well-chosen tie points are provided, the
        tolerance algorithm functions well in the presence of any
        shifts, axis flips, *x* and *y* scale changes, rotations, and
        axis skew, between the two coordinate systems.  The algorithm
        is sensitive to higher-order distortion terms in the
        coordinate transformation.

      - ``'triangles'``: A linear transformation is applied to the
        input coordinate list, the transformed input list and the
        reference list are sorted, points which are too close together
        are removed, and the input coordinates are matches to the
        reference coordinates using a triangle pattern matching
        technique and the user-specified tolerance parameter.  The
        triangles pattern matching algorithm does not require prior
        knowledge of the linear transformation, although it will use
        one if one is supplied.  The algorithm functions well in the
        presence of any shifts, axis flips, magnification and rotation
        between the two coordinate systems as long as both lists have
        a reasonable number of objects in common and the errors in the
        computed coordinates are small.  However, since the algorithm
        depends on comparisons of similar triangles, it is sensitive
        to differences in the *x* and *y* coordinate scales, any skew
        between the *x* and *y* axes, and higher order distortion
        terms in the coordinate transformation.

    - *tolerance*: The matching tolerance in pixels. Default: 1.0

    - *separation*: The minimum separation for objects in the input
      and reference coordinate lists.  Objects closer together than
      *separation* pixels are removed from the input and reference
      coordinate lists prior to matching. Default: 9.0

    - *nmatch*: The maximum number of reference and input coordinates
      used by the ``'triangles'`` pattern matching algorithm.  If
      either list contains more coordinates than *nmatch*, the lists
      are subsampled.  *nmatch* should be kept small as the
      computation and memory requirements of the triangles algorithm
      depend on a high power of lengths of the respective lists.
      Default: 30

    - *maxratio*: The maximum ratio of the longest to shortest side of
      the triangles generated by the triangles pattern matching
      algorithm.  Triangles with computed longest to shortest side.
      Ratios greater than *ratio* are rejected from the pattern
      matching algorithm.  *ratio* should never be set higher than
      10.0 but may be set as low as 5.0.  Default: 10.0

    - *nreject*: The maximum number of rejection iterations for the
      ``'triangles'`` pattern matching algorithm.  Default: 10

    **Returns**: A structured array containing the output
    information.  It has the following columns:

    - *input_x*
    - *input_y*
    - *input_idx*
    - *ref_x*
    - *ref_y*
    - *ref_idx*
    """
    return _c_xy_coord_match.xyxymatch(input, ref, origin, mag, rotation, ref_origin, algorithm, tolerance, separation,
                                   nmatch, maxratio, nreject)
