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

#ifndef _STIMAGE_TRIANGLES_H_
#define _STIMAGE_TRIANGLES_H_

#include "lib/util.h"
#include "immatch/lib/match_util.h"

/**
Compute the intersection of two lists using a pattern matching
algorithm. This algorithm is based on one developed by Edward Groth
1986 A.J. 91, 1244. The algorithm matches pairs of coordinates from
two lists based on the triangles that can be formed from triplets of
points in each list. The algorithm is insensitive to coordinate
translation, rotation, magnification, or inversion and can tolerate
distortions and random errors.

@param nref The number of reference coordinates

@param nref_unique The number of unique reference coordinates
(specifically in ref_sorted)

@param ref The raw array of reference coordinates, used for
determining indices into the original set.

@param ref_sorted An array of pointers reference coordinates in ref.
It is assumed that this array has already been sorted with xysort and
culled with xycoincide.

@param ninput The number of input coordinates

@param ninput_unique The number of unique input coordinates
(specifically in input_sorted)

@param input The raw array of reference coordinates, used for
determining indices into the original set.

@param input_sorted An array of pointers input coordinates in input.
It is assumed that this array has already been sorted with xysort and
culled with xycoincide.

@param nmatch The maximum number of reference and input coordinates to
use in matching.  If either list contains more coordinates than
nmatch, the lists are subsampled.  nmatch should be kept small, as the
computation and memory requirements of the triangles algorithm depend
on a high power of the lengths of the respective lists.

@param tolerance The matching tolerance in pixels.

@param maxratio The maximum ratio of the longest to shortest side of
the triangles used for matching.

@param nreject The maximum number of rejection iteration cycles.

@param callback A callback function that is called with each matching
coordinate pair.  Its arguments are (data, ref_index, input_index,
error).  data is always whatever callback_data is.  ref_index is the
index in the original ref array to the coordinate.  input_index is the
index in the original input array to the coordinate.  error is an
error object in case an error needs to be returned from the callback.

@param callback_data A void* to private data required by the given
callback.

@param error Stores an error string, if an error occurred.
 */
int
match_triangles(
        const size_t nref,
        const size_t nref_unique,
        const coord_t* const ref,
        const coord_t* const * const ref_sorted, /*[nref]*/
        const size_t ninput,
        const size_t ninput_unique,
        const coord_t* const input, /*[ninput]*/
        const coord_t* const * const input_sorted,
        const size_t nmatch,
        const double tolerance,
        const double maxratio,
        const size_t nreject,
        coord_match_callback_t* callback,
        void* callback_data,
        stimage_error_t* const error);

/********************************************************************************
BELOW IS THE SECONDARY API -- SUBJECT TO CHANGE
********************************************************************************/

/**
Stores information about a triangle
*/
typedef struct {
    /** The vertices of the triangle */
    const coord_t* vertices[3];

    /** The log of the perimeter of the triangle */
    double log_perimeter;

    /** The ratio of the longest to shortest side */
    double ratio;

    /** Cosine of angle at vertex 1 */
    double cosine_v1;

    /** Tolerance in the ratio */
    double ratio_tolerance;

    /** Tolerance in the cosine */
    double cosine_tolerance;

    /** Sense of the triangle (clockwise (non-zero) or anti-clockwise
        (zero)) */
    int sense;
} triangle_t;

/**
Pointers to a matching pair of triangles.
*/
typedef struct {
    const triangle_t* l;
    const triangle_t* r;
} triangle_match_t;

/**
Compute the number of possible triangles given the number of
coordinates.
*/
int
max_num_triangles(
        const size_t ncoords,
        const size_t max_ncoords,
        size_t* num_triangles,
        stimage_error_t* const error);

/**
Construct all possible triangles from an input coordinate list.

The triangles are constructed in such a way that the shortest side of
the triangle lies between vertices 1 and 2 and the longest side
between vertices 1 and 3.

@param ncoords The number of coordinates in the coordinate list

@param coords A list of pointers to coordinates.  It is assumed that
these coordinates have already been sorted with xysort and culled with
xycoincide.

@param ntriangles On input, the number of triangles allocated in the
triangle list.  On output, the number of triangles found.  The number
of triangles to allocate should be determined using max_num_triangles.

@param triangles An array of triangle_t structs to store the
triangles.

@param maxnpoints The maximum number of points.

@param tolerance Triangles with vertices closer than tolerance are
rejected.

@param maxratio Triangles with a ratio of longest side to shortest
side greater than maxratio are rejected.

@param error Contains an error message if an error occurred.
 */
int
find_triangles(
        const size_t ncoords,
        const coord_t* const * const coords,
        size_t* ntriangles,
        triangle_t* triangles,
        const size_t maxnpoints,
        const double tolerance,
        const double maxratio,
        stimage_error_t* const error);

/**
Compute the intersection of the two sorted lists of triangles using
the ratio tolerance parameter.

@param nr_triangles The number of reference triangles

@param r_triangles An array of triangles

@param nl_triangles The number of input triangles

@param l_triangles An array of triangles

@param nmatches On input: The number of matches allocated.  On output:
The number of matches found.

@param matches An array to store the match pairs.

@param error
*/
int
merge_triangles(
        const size_t nr_triangles,
        const triangle_t* const r_triangles,
        const size_t nl_triangles,
        const triangle_t* const l_triangles,
        size_t* nmatches,
        triangle_match_t* const matches,
        stimage_error_t* const error);

/**
Remove false matches from the list of matched triangles.

@param nmatches The number of matches

@param matches An array of triangle match pairs

@param nreject The number of rejection iterations to perform

@param error
*/
int
reject_triangles(
        size_t* nmatches,
        triangle_match_t* const matches,
        const size_t nreject,
        stimage_error_t* error);

/**
Count the number a times a particular pair of coordinates is matched
in the set of matched triangles. If a particular pair of points occurs
in many triangles it is much more likely to be a true match than if it
occurs in very few.

@param ntriangle_matches The number of triangle match pairs

@param triangle_matches An array of triangle match pairs

@param ncoord_matches On input: The number of coordinate matches
allocated (the length of refcoord_matches and input_coord_matches
arrays).  On output: The number of matches actually filled in those
arrays.

@param refcoord_matches An array of pointers to coordinates in the
reference set that correspond to the coordinates in
inputcoord_matches.

@param inputcoord_matches An array of pointers to coordinates in the
reference set that correspond to the coordinates in
inputcoord_matches.

@param error
*/
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
        stimage_error_t* const error);

#endif /* _STIMAGE_TRIANGLES_H_ */

