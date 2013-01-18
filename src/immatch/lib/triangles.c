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

#include "immatch/lib/triangles.h"

int
max_num_triangles(
        const size_t ncoords,
        const size_t maxnpoints,
        size_t* num_triangles,
        stimage_error_t* const error) {

    size_t n = MIN(ncoords, maxnpoints);
    if (n >= 2346 || n == 0) {
        stimage_error_set_message(
            error,
            "maxnpoints should be a lower number");
        return 1;
    }

    *num_triangles = combinatorial(n, 3);

    return 0;
}

static const size_t
sides_def [3][2] = {
    { 2, 1 },
    { 1, 0 },
    { 2, 0 }
};

/* Uses as a qsort functor */
static int
triangle_ratio_compare(
        const void* ap,
        const void* bp) {

    const triangle_t* a = (const triangle_t*)ap;
    const triangle_t* b = (const triangle_t*)bp;

    if (a->ratio < b->ratio) {
        return -1;
    } else if (a->ratio > b->ratio) {
        return 1;
    } else {
        return 0;
    }
}

int
find_triangles(
        const size_t ncoords,
        const coord_t* const * const coords,
        size_t* ntriangles,
        triangle_t* triangles,
        const size_t maxnpoints,
        const double tolerance,
        const double maxratio,
        stimage_error_t* const error) {

    const double tol2 = tolerance * tolerance;
    const size_t nsample = MAX(1, ncoords / maxnpoints);
    const size_t npoints = MIN(ncoords, nsample * maxnpoints);
    triangle_t* tri = triangles;
    size_t i, j, k, m;
    size_t ntri = 0;
    double dist_ij, dist_jk, dist_ki;
    double dx[3], dy[3], sides2[3], sides[3];
    double cosc, cosc2, sinc2;
    double ratio, loctol;

    assert(coords);
    assert(ntriangles);
    assert(triangles);
    assert(error);

    if (maxratio > 10.0 || maxratio < 5.0) {
        stimage_error_format_message(
            error,
            "maxratio should be in the range 5.0 - 10.0 (%f)", maxratio);
        return 1;
    }

    for (i = 0; i < npoints - (2 * nsample); i += nsample) {
        for (j = i + nsample; j < npoints - nsample; j += nsample) {
            dist_ij = euclid_distance2(coords[i], coords[j]);
            if (dist_ij <= tol2) {
                continue;
            }

            for (k = j + nsample; k < npoints; k += nsample) {
                dist_jk = euclid_distance2(coords[j], coords[k]);
                if (dist_jk <= tol2) {
                    continue;
                }

                dist_ki = euclid_distance2(coords[k], coords[i]);
                if (dist_ki <= tol2) {
                    continue;
                }

                #ifndef NDEBUG
                    if (ntri >= *ntriangles) {
                        stimage_error_format_message(
                            error,
                            "Found more triangles than were allocated for (%d)\n",
                            *ntriangles);
                        return 1;
                    }
                #endif /* NDEBUG */

                tri = &triangles[ntri];
                /* DIFF: The original stores the index of the
                   triangle.  Do we need to do that? */

                /* Order the vertices with the shortest side of the triangle
                   between vertices 1 and 2 and the intermediate side between
                   vertices 2 and 3.
                */
                if (dist_ij <= dist_jk) {
                    if (dist_ki <= dist_ij) {
                        tri->vertices[0] = coords[k];
                        tri->vertices[1] = coords[i];
                        tri->vertices[2] = coords[j];
                    } else if (dist_ki >= dist_jk) {
                        tri->vertices[0] = coords[i];
                        tri->vertices[1] = coords[j];
                        tri->vertices[2] = coords[k];
                    } else {
                        tri->vertices[0] = coords[j];
                        tri->vertices[1] = coords[i];
                        tri->vertices[2] = coords[k];
                    }
                } else {
                    if (dist_ki <= dist_jk) {
                        tri->vertices[0] = coords[i];
                        tri->vertices[1] = coords[k];
                        tri->vertices[2] = coords[j];
                    } else if (dist_ki >= dist_ij) {
                        tri->vertices[0] = coords[k];
                        tri->vertices[1] = coords[j];
                        tri->vertices[2] = coords[i];
                    } else {
                        tri->vertices[0] = coords[j];
                        tri->vertices[1] = coords[k];
                        tri->vertices[2] = coords[i];
                    }
                }

                /* Compute the lengths of the sides */
                for (m = 0; m < 3; ++m) {
                    dx[m] = tri->vertices[sides_def[m][0]]->x -
                        tri->vertices[sides_def[m][1]]->x;
                    dy[m] = tri->vertices[sides_def[m][0]]->y -
                        tri->vertices[sides_def[m][1]]->y;
                    sides2[m] = dx[m]*dx[m] + dy[m]*dy[m];
                    assert(sides2[m] >= 0.0);
                    sides[m] = sqrt(sides2[m]);
                }

                /* If the ratio of long to short is too high, reject
                   this triangle */
                ratio = sides[2] / sides[1];
                if (ratio > maxratio) {
                    continue;
                }

                /* Compute the cos, cos ** 2 and sin ** 2 of the angle at
                   vertex 1. */
                cosc = (dx[2]*dx[1] + dy[2]*dy[1]) / (sides[2]*sides[1]);
                cosc2 = MAX(0.0, MIN(1.0, cosc*cosc));
                sinc2 = MAX(0.0, MIN(1.0, 1.0 - cosc2));

                /* Determine whether the triangles vertices are
                   arranged clockwise or anti-clockwise */
                tri->sense = ((dx[1]*dy[0] - dy[1]*dx[0]) > 0.0);

                /* Compute the tolerances */
                loctol = (1.0/sides2[2] - cosc/(sides[2]*sides[1]) + 1.0/sides2[1]);
                tri->ratio_tolerance = 2.0*ratio*ratio*tol2*loctol;
                tri->cosine_tolerance = \
                    2.0*sinc2*tol2*loctol +
                    2.0*cosc2*tol2*tol2*loctol*loctol;

                /* Compute the perimeter */
                tri->log_perimeter = log(sides[0] + sides[1] + sides[2]);
                tri->ratio = ratio;
                tri->cosine_v1 = cosc;

                ++ntri;
            }
        }
    }

    *ntriangles = ntri;

    /* Sort the triangles in increasing order of ratio */
    qsort(triangles, ntri, sizeof(triangle_t), &triangle_ratio_compare);

    return 0;
}

int
merge_triangles(
        const size_t nr_triangles,
        const triangle_t* const r_triangles,
        const size_t nl_triangles,
        const triangle_t* const l_triangles,
        size_t* nmatches,
        triangle_match_t* const matches,
        stimage_error_t* const error) {

    size_t i;
    size_t match_iter = 0;
    double rmaxtol, lmaxtol, maxtol;
    size_t blp = 0, rp = 0, lp = 0;
    double dratio, dratio2, dcosine, dcosine2, dtratio, dtcosine;
    const triangle_t* max_tri = NULL;
    const triangle_t* l_tri = NULL;
    const triangle_t* r_tri = NULL;
    double max_dratio2, max_dcosine2;

    assert(nr_triangles);
    assert(r_triangles);
    assert(nl_triangles);
    assert(l_triangles);
    assert(matches);
    assert(error);

    /* Find the maximum tolerance for each list */
    rmaxtol = r_triangles[0].ratio_tolerance;
    for (i = 1; i < nr_triangles; ++i) {
        rmaxtol = MAX(rmaxtol, r_triangles[i].ratio_tolerance);
    }

    lmaxtol = l_triangles[0].ratio_tolerance;
    for (i = 1; i < nl_triangles; ++i) {
        lmaxtol = MAX(lmaxtol, l_triangles[i].ratio_tolerance);
    }

    maxtol = sqrt(rmaxtol + lmaxtol);

    /* Loop over all the triangles in R */
    for (rp = 0; rp < nr_triangles; ++rp) {
        r_tri = r_triangles + rp;

        /* Move to the first triangle in L that satisfies the ratio
           tolerance requirement */
        for ( ; blp < nl_triangles; ++blp) {
            l_tri = l_triangles + blp;
            dratio = r_tri->ratio - l_tri->ratio;
            if (dratio <= maxtol) {
                break;
            }
        }

        /* If the beginning of the search range becomes greater than
           the length of the list, then there are no more matches. */
        if (blp >= nl_triangles) {
            break;
        }

        /* If the first triangle in the list is past the tolerance
           limit, skip to the next reference triangle. */
        if (dratio < -maxtol) {
            continue;
        }

        /* Search through the appropriate range of triangles for the
           closest fit. */

        /* Initialize the tolerances */
        max_tri = NULL;
        max_dratio2 = 0.5 * MAX_DOUBLE;
        max_dcosine2 = 0.5 * MAX_DOUBLE;

        for (lp = blp; lp < nl_triangles; ++lp) {
            l_tri = l_triangles + lp;

            /* Quit the loop if the next triangle is out of match range. */
            dratio = r_tri->ratio - l_tri->ratio;
            if (dratio < -maxtol) {
                break;
            }

            /* Compute the tolerances for the two triangles */
            dratio2 = dratio*dratio;
            dcosine = r_tri->cosine_v1 - l_tri->cosine_v1;
            dcosine2 = dcosine*dcosine;
            dtratio = r_tri->ratio_tolerance + l_tri->ratio_tolerance;
            dtcosine = r_tri->cosine_tolerance + l_tri->cosine_tolerance;

            /* Find the best of all possible matches */
            if (dratio2 <= dtratio && dcosine2 <= dtcosine &&
                (dratio2 + dcosine2) < (max_dratio2 + max_dcosine2)) {
                max_tri = l_tri;
                max_dratio2 = dratio2;
                max_dcosine2 = dcosine2;
            }
        }

        if (max_tri != NULL) {
            #ifndef NDEBUG
                if (match_iter >= *nmatches) {
                    stimage_error_set_message(
                        error,
                        "Found more triangle matches than were allocated for");
                    return 1;
                }
            #endif /* NDEBUG */

            matches[match_iter].l = max_tri;
            matches[match_iter].r = r_tri;
            ++match_iter;
        }
    }

    *nmatches = match_iter;

    return 0;
}

static int
reject_triangles_compute_sigma_mode_factor(
        const size_t nmatches,
        double* const diffp,
        const double sum,
        const double sumsq,
        const size_t nfalse,
        const size_t ntrue,
        double* sigma,
        double* mode,
        double* factor) {

    assert(diffp);
    assert(sigma);
    assert(mode);
    assert(factor);

    /* Compute the mean, mode, and sigma of the log-perimeter
       distribution */
    if (nmatches == 0) {
        *sigma = 0.0;
    } else {
        *sigma = (sumsq - (sum / (double)(nmatches)) * sum) / ((double)(nmatches) - 1);
    }

    if (*sigma <= 0.0) {
        return 1;
    } else {
        *sigma = sqrt(*sigma);
    }

    /* Sort by the diff of the log-perimeters */
    sort_doubles(nmatches, diffp);

    *mode = compute_mode(nmatches, diffp, 10, 1.0, 0.1 * *sigma, 0.01 * *sigma);

    if (nfalse > ntrue) {
        *factor = 1.0;
    } else if ((0.1 * ntrue) > nfalse) {
        *factor = 3.0;
    } else {
        *factor = 2.0;
    }

    return 0;
}

int
reject_triangles(
        size_t* nmatches,
        triangle_match_t* const matches,
        const size_t nreject,
        stimage_error_t* error) {

    size_t            i            = 0;
    double            sum          = 0.0;
    double            sumsq        = 0.0;
    int               nplus        = 0;
    int               nminus       = 0;
    double            diff         = 0.0;
    int               ntrue        = 0;
    int               nfalse       = 0;
    double            sigma        = 0.0;
    double            mode         = 0.0;
    double            factor       = 0.0;
    double            locut        = 0.0;
    double            hicut        = 0.0;
    size_t            ncount       = 0;
    size_t            ncurrmatches = *nmatches;
    size_t            niter        = 0;
    const triangle_t* r_tri        = NULL;
    const triangle_t* l_tri        = NULL;
    double*           diffp        = NULL;
    int               status       = 1;

    assert(nmatches);
    assert(matches);
    assert(error);

    diffp = malloc_with_error(ncurrmatches * sizeof(double), error);
    if (diffp == NULL) goto exit;

    /* Accumulate the number of same-sense and number of
       opposite-sense matches as well as the log perimeter
       statistics. */
    for (i = 0; i < ncurrmatches; ++i) {
        r_tri = matches[i].r;
        l_tri = matches[i].l;

        diff = r_tri->log_perimeter - l_tri->log_perimeter;
        diffp[i] = diff;
        sum += diff;
        sumsq += diff*diff;
        if (r_tri->sense == l_tri->sense) {
            ++nplus;
        }
    }
    nminus = (int)ncurrmatches - nplus;
    ntrue = ABS(nplus - nminus);
    nfalse = ncurrmatches - ntrue;

    if (reject_triangles_compute_sigma_mode_factor(
            ncurrmatches, diffp, sum, sumsq, nfalse, ntrue, &sigma, &mode, &factor)) {
        status = 0;
        goto exit;
    }

    /* Begin the rejection loop */
    for (niter = 0; niter < nreject; ++niter) {
        ncount = 0;
        locut = mode - factor * sigma;
        hicut = mode + factor * sigma;

        for (i = 0; i < ncurrmatches; ++i) {
            r_tri = matches[i].r;
            l_tri = matches[i].l;
            diff = r_tri->log_perimeter - l_tri->log_perimeter;
            if (diff < locut || diff > hicut) {
                sum -= diff;
                sumsq -= diff*diff;
                if (r_tri->sense == l_tri->sense) {
                    --nplus;
                } else {
                    --nminus;
                }
            } else {
                /* This starts writing non-rejected matches to the
                   beginning of the list containing all matches */
                #ifndef NDEBUG
                    if (ncount >= *nmatches) {
                        stimage_error_set_message(
                            error,
                            "Rejection created more matches than it started with.");
                        goto exit;
                    }
                #endif
                diffp[ncount] = diff;
                matches[ncount].r = r_tri;
                matches[ncount].l = l_tri;
                ++ncount;
            }
        }

        /* NOTE: At this point matches[0 --- ncount] contains only
           non-rejected matches.  matches[ncount --- ncurrmatches] is now
           garbage.  Same is true of diffp.  */

        /* No more triangles were rejected, or all the triangles were rejected */
        if (ncurrmatches == ncount || ncount == 0) {
            break;
        }

        ncurrmatches = ncount;

        /* Recompute sigma, mode and factor based on only non-rejected
           values */
        if (reject_triangles_compute_sigma_mode_factor(
                ncurrmatches, diffp, sum, sumsq, nfalse, ntrue, &sigma, &mode, &factor)) {
            break;
        }
    };

    /* One last time through loop to get rid of opposite sense of matches */
    if (ncurrmatches > 0) {
        if (nplus > nminus) {
            ncount = 0;
            for (i = 0; i < ncurrmatches; ++i) {
                r_tri = matches[i].r;
                l_tri = matches[i].l;
                if (r_tri->sense == l_tri->sense) {
                    matches[ncount].r = r_tri;
                    matches[ncount].l = l_tri;
                    ++ncount;
                }
            }
            ncurrmatches = ncount;
        } else {
            ncount = 0;
            for (i = 0; i < ncurrmatches; ++i) {
                r_tri = matches[i].r;
                l_tri = matches[i].l;
                if (r_tri->sense != l_tri->sense) {
                    matches[ncount].r = r_tri;
                    matches[ncount].l = l_tri;
                    ++ncount;
                }
            }
            ncurrmatches = ncount;
        }
    }

    *nmatches = ncurrmatches;

    status = 0;

 exit:

    free(diffp);

    return status;
}

static int
_match_triangles(
        const size_t nref,
        const coord_t* const ref,
        const coord_t* const * const ref_sorted, /*[nref]*/
        const size_t ninput,
        const coord_t* const input, /*[ninput]*/
        const coord_t* const * const input_sorted,
        size_t* ncoord_matches,
        const coord_t** refcoord_matches_,
        const coord_t** inputcoord_matches_,
        const size_t nmatch,
        const double tolerance,
        const double maxratio,
        const size_t nreject,
        size_t* nkeep,
        size_t* nmerge,
        stimage_error_t* const error) {

    const coord_t**   refcoord_matches   = NULL;
    const coord_t**   inputcoord_matches = NULL;
    size_t            nleft              = 0;
    const coord_t*    left               = NULL;
    size_t            nright             = 0;
    const coord_t*    right              = NULL;
    size_t            nref_triangles     = 0;
    triangle_t*       ref_triangles      = NULL;
    size_t            ninput_triangles   = 0;
    triangle_t*       input_triangles    = NULL;
    size_t            ntriangle_matches  = 0;
    triangle_match_t* triangle_matches   = NULL;
    int               status             = 1;

    assert(ref);
    assert(ref_sorted);
    assert(input);
    assert(input_sorted);
    assert(ncoord_matches);
    assert(refcoord_matches_);
    assert(inputcoord_matches_);
    assert(nkeep);
    assert(nmerge);
    assert(error);

    if (nref < 3) {
        stimage_error_set_message(
            error,
            "Too few reference coordinates to do triangle matching");
        goto exit;
    }

    if (ninput < 3) {
        stimage_error_set_message(
            error,
            "Too few input coordinates to do triangle matching");
        goto exit;
    }

    /* Find all the reference triangles */
    if (max_num_triangles(nref, nmatch, &nref_triangles, error)) goto exit;

    ref_triangles = malloc_with_error(
            nref_triangles * sizeof(triangle_t), error);
    if (ref_triangles == NULL) goto exit;

    if (find_triangles(nref, ref_sorted, &nref_triangles, ref_triangles,
                       nmatch, tolerance, maxratio, error)) goto exit;

    if (nref_triangles == 0) {
        stimage_error_set_message(
            error,
            "No valid reference triangles found.");
        goto exit;
    }

    /* Find all the input triangles */
    if (max_num_triangles(ninput, nmatch, &ninput_triangles, error)) goto exit;

    input_triangles = malloc_with_error(
            ninput_triangles * sizeof(triangle_t), error);
    if (input_triangles == NULL) goto exit;

    if (find_triangles(ninput, input_sorted, &ninput_triangles,
                       input_triangles, nmatch, tolerance, maxratio,
                       error)) goto exit;

    if (ninput_triangles == 0) {
        stimage_error_set_message(
            error,
            "No valid input triangles found.");
        goto exit;
    }

    ntriangle_matches = MAX(nref_triangles, ninput_triangles);
    triangle_matches = malloc_with_error(
        ntriangle_matches * sizeof(triangle_match_t), error);
    if (triangle_matches == NULL) goto exit;

    /* Match the triangles in the input list to those in the reference
       list */
    if (nref_triangles <= ninput_triangles) {
        refcoord_matches = inputcoord_matches_;
        inputcoord_matches = refcoord_matches_;
        nleft = ninput;
        left = input;
        nright = nref;
        right = ref;
        if (merge_triangles(
                nref_triangles, ref_triangles,
                ninput_triangles, input_triangles,
                &ntriangle_matches, triangle_matches,
                error)) goto exit;
    } else {
        refcoord_matches = refcoord_matches_;
        inputcoord_matches = inputcoord_matches_;
        nleft = nref;
        left = ref;
        nright = ninput;
        right = input;
        if (merge_triangles(
                ninput_triangles, input_triangles,
                nref_triangles, ref_triangles,
                &ntriangle_matches, triangle_matches,
                error)) goto exit;
    }

    *nmerge = ntriangle_matches;

    if (ntriangle_matches == 0) {
        status = 0;
        goto exit;
    }

    /* Reject triangles */
    if (reject_triangles(&ntriangle_matches, triangle_matches,
                         nreject,
                         error)) {
        goto exit;
    }

    *nkeep = ntriangle_matches;

    if (ntriangle_matches == 0) {
        *ncoord_matches = 0;
        status = 0;
        goto exit;
    }

    /* Match the coordinates */
    if (vote_triangle_matches(
                nleft, left, nright, right,
                ntriangle_matches, triangle_matches,
                ncoord_matches, refcoord_matches, inputcoord_matches,
                error)) {
        goto exit;
    }

    status = 0;

 exit:

    free(ref_triangles);
    free(input_triangles);
    free(triangle_matches);
    return status;
}

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
        stimage_error_t* const error) {

    size_t          ncoord_matches     = nmatch;
    const coord_t** refcoord_matches   = NULL;
    const coord_t** inputcoord_matches = NULL;
    size_t          nkeep              = 0;
    size_t          nmerge             = 0;
    size_t          ncheck             = 0;
    size_t          ref_idx            = 0;
    size_t          input_idx          = 0;
    size_t          i                  = 0;
    int             status             = 1;

    refcoord_matches = malloc_with_error(
            ncoord_matches * sizeof(coord_t*), error);
    if (refcoord_matches == NULL) goto exit;

    inputcoord_matches = malloc_with_error(
            ncoord_matches * sizeof(coord_t*), error);
    if (inputcoord_matches == NULL) goto exit;

    if (_match_triangles(
        nref_unique, ref, ref_sorted,
        ninput_unique, input, input_sorted,
        &ncoord_matches, refcoord_matches, inputcoord_matches,
        nmatch, tolerance, maxratio, nreject,
        &nkeep, &nmerge,
        error)) goto exit;

    if (ncoord_matches == 0 || (ncoord_matches <= 3 && nkeep < nmerge)) {
        status = 0;
        goto exit;
    }

    /* If all the coordinates were not matched then make another pass
       through the triangles matching algorithm. If the number of
       matches decreases as a result of this then all the matches were
       not true matches and declare the list unmatched. */
    if (ncoord_matches < nmatch && ncoord_matches > 2) {
        ncheck = ncoord_matches;
        if (_match_triangles(
                ncoord_matches, ref, refcoord_matches,
                ncoord_matches, input, inputcoord_matches,
                &ncoord_matches, refcoord_matches, inputcoord_matches,
                nmatch, tolerance, maxratio, nreject,
                &nkeep, &nmerge, error)) goto exit;

        if (ncoord_matches < ncheck) {
            ncoord_matches = 0;
        }
    }

    status = 0;

 exit:

    if (status == 0) {
        /* Call the callback with all of the matches */
        for (i = 0; i < ncoord_matches; ++i) {
            ref_idx = refcoord_matches[i] - ref;
            input_idx = inputcoord_matches[i] - input;

            assert(ref_idx < nref);
            assert(input_idx < ninput);

            if (callback(callback_data, ref_idx, input_idx, error)) {
                status = 1;
                break;
            }
        }
    }

    free(refcoord_matches);
    free(inputcoord_matches);

    return status;
}
