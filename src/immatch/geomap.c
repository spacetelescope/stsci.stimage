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

#define _USE_MATH_DEFINES       /* needed for MS Windows to define M_PI */ 
#include <math.h>
#include <stdio.h>

#include "immatch/geomap.h"
#include "lib/xybbox.h"
#include "surface/fit.h"
#include "surface/vector.h"

/** DIFF

The argument 'calctype' to choose between real and double calculations
is gone.  It's always double now.
 */

typedef struct {
    size_t initialized;

    /* Function and order */
    geomap_proj_e       projection;
    geomap_fit_e        fit_geometry;
    surface_type_e      function;
    size_t              xxorder;
    size_t              xyorder;
    xterms_e            xxterms;
    size_t              yxorder;
    size_t              yyorder;
    xterms_e            yxterms;

    /* Rejection parameters */
    double xrms;
    double yrms;
    size_t maxiter;
    double reject;
    size_t nreject;
    int*   rej;

    coord_t oref;
    coord_t oin;
    coord_t refpt;
    bbox_t  bbox;
    size_t  n_zero_weighted;
    size_t  ncoord;
} geomap_fit_t;

/* was geo_minit */
static void
geomap_fit_init(
        geomap_fit_t* fit,
        const geomap_proj_e projection,
        const geomap_fit_e fit_geometry,
        const surface_type_e function,
        const size_t xxorder,
        const size_t xyorder,
        const xterms_e xxterms,
        const size_t yxorder,
        const size_t yyorder,
        const xterms_e yxterms,
        const size_t maxiter,
        const double reject) {

    assert(fit);
    assert(fit_geometry < geomap_fit_LAST);
    assert(function < surface_type_LAST);
    assert(xxterms < xterms_LAST);
    assert(yxterms < xterms_LAST);

    fit->projection   = projection;
    fit->fit_geometry = fit_geometry;
    fit->function     = function;
    fit->xxorder      = xxorder;
    fit->xyorder      = xyorder;
    fit->xxterms      = xxterms;
    fit->yxorder      = yxorder;
    fit->yyorder      = yyorder;
    fit->yxterms      = yxterms;

    fit->xrms    = 0.0;
    fit->yrms    = 0.0;
    fit->maxiter = maxiter;
    fit->reject  = reject;
    fit->nreject = 0;
    fit->rej     = NULL;

    fit->initialized = 1;
}

static void
geomap_fit_new(
        geomap_fit_t* fit) {

    fit->initialized = 0;
    fit->rej = NULL;
}

static void
geomap_fit_free(
        geomap_fit_t* fit) {

    free(fit->rej); fit->rej = NULL;
    fit->initialized = 0;
}

static void
compute_sums(
        const size_t ncoord,
        const coord_t* const input,
        const coord_t* const ref,
        const double* const weights,
        /* Output */
        double* const sw,
        coord_t* const si,
        coord_t* const sr) {

    size_t i = 0;

    assert(input);
    assert(ref);
    assert(weights);
    assert(sw);
    assert(si);
    assert(sr);

    *sw = 0.0;
    si->x = 0.0;
    si->y = 0.0;
    sr->x = 0.0;
    sr->y = 0.0;
    for (i = 0; i < ncoord; ++i) {
        *sw   += weights[i];
        si->x += weights[i] * input[i].x;
        si->y += weights[i] * input[i].y;
        sr->x += weights[i] * ref[i].x;
        sr->y += weights[i] * ref[i].y;
    }
}

static int
compute_surface_coefficients(
        const surface_type_e function,
        const bbox_t* const bbox,
        const coord_t* const i0,
        const coord_t* const r0,
        const coord_t* const cthetac,
        const coord_t* const sthetac,
        surface_t* const sx1,
        surface_t* const sy1,
        stimage_error_t* const error) {

    int       status = 1;

    assert(bbox);
    assert(i0);
    assert(r0);
    assert(cthetac);
    assert(sthetac);
    assert(sx1);
    assert(sy1);
    assert(error);

    /* Compute the x fit coefficients */
    if (surface_init(
                sx1, function, 2, 2, xterms_none, bbox, error)) goto exit;

    if (function == surface_type_polynomial) {
        sx1->coeff[0] = i0->x - (r0->x*cthetac->x + r0->y*sthetac->x);
        sx1->coeff[1] = cthetac->x;
        sx1->coeff[2] = sthetac->x;
    } else {
        sx1->coeff[0] = i0->x - (r0->x*cthetac->x + r0->y*sthetac->x) + \
            cthetac->x*(bbox->max.x + bbox->min.x)/2.0 + \
            sthetac->x*(bbox->max.y + bbox->min.y)/2.0;
        sx1->coeff[1] = cthetac->x * (bbox->max.x - bbox->min.x) / 2.0;
        sx1->coeff[2] = sthetac->x * (bbox->max.y - bbox->min.y) / 2.0;
    }

    /* Compute the y fit coefficients */
    if (surface_init(
                sy1, function, 2, 2, xterms_none, bbox, error)) goto exit;

    if (function == surface_type_polynomial) {
        sy1->coeff[0] = i0->y - (-r0->x*sthetac->y + r0->y*cthetac->y);
        sy1->coeff[1] = -sthetac->y;
        sy1->coeff[2] = cthetac->y;
    } else {
        sy1->coeff[0] = i0->y - (-r0->x*sthetac->y + r0->y*cthetac->y) - \
            sthetac->y*(bbox->max.x + bbox->min.x) / 2.0 +              \
            cthetac->y*(bbox->max.y + bbox->min.y) / 2.0;
        sy1->coeff[1] = -sthetac->y * (bbox->max.x - bbox->min.x) / 2.0;
        sy1->coeff[2] = cthetac->y * (bbox->max.y - bbox->min.y) / 2.0;
    }

    status = 0;

 exit:

    return 0;
}

static int
compute_residuals(
        const surface_t* const sx1,
        const surface_t* const sy1,
        const size_t ncoord,
        const coord_t* const input,
        const coord_t* const ref,
        /* Output */
        double* const residual_x,
        double* const residual_y,
        stimage_error_t* const error) {

    size_t i = 0;

    assert(sx1);
    assert(sy1);
    assert(input);
    assert(ref);
    assert(residual_x);
    assert(residual_y);
    assert(error);

    if (surface_vector(sx1, ncoord, ref, residual_x, error)) return 1;

    if (surface_vector(sy1, ncoord, ref, residual_y, error)) return 1;

    for (i = 0; i < ncoord; ++i) {
        residual_x[i] = input[i].x - residual_x[i];
        residual_y[i] = input[i].y - residual_y[i];
    }

    return 0;
}

static void
compute_rms(
        const size_t ncoord,
        const double* const weights,
        const double* const residual_x,
        const double* const residual_y,
        /* Output */
        double* const xrms,
        double* const yrms) {

    size_t i = 0;

    assert(weights);
    assert(residual_x);
    assert(residual_y);
    assert(xrms);
    assert(yrms);

    /* Compute the X and Y fit rms */
    *xrms = 0.0;
    *yrms = 0.0;
    for (i = 0; i < ncoord; ++i) {
        *xrms += weights[i] * residual_x[i] * residual_x[i];
        *yrms += weights[i] * residual_y[i] * residual_y[i];
    }
}

static size_t
count_zero_weighted(
        const size_t ncoord,
        const double* const weights) {

    size_t count = 0;
    size_t i     = 0;

    assert(weights);

    for (i = 0; i < ncoord; ++i) {
        if (weights[i] <= 0.0) {
            ++count;
        }
    }

    return count;
}

/** DIFF: was geo_fthetad */

/* Compute the shift and rotation angle required to match one set of
   coordinates to another.  The result is stored in the fit
   structure. */
static int
geo_fit_theta(
        geomap_fit_t* const fit,
        surface_t* const sx1,
        surface_t* const sy1,
        const size_t ncoord,
        const coord_t* const input,
        const coord_t* const ref,
        const double* const weights,
        /* Output */
        double* const residual_x,
        double* const residual_y,
        stimage_error_t* error) {

    bbox_t  bbox;
    double  sw      = 0.0;
    coord_t sr      = {0.0, 0.0};
    coord_t si      = {0.0, 0.0};
    coord_t r0      = {0.0, 0.0};
    coord_t i0      = {0.0, 0.0};
    double  syrxi   = 0.0;
    double  sxryi   = 0.0;
    double  sxrxi   = 0.0;
    double  syryi   = 0.0;
    double  num     = 0.0;
    double  denom   = 0.0;
    double  det     = 0.0;
    double  theta   = 0.0;
    double  ctheta  = 0.0;
    double  stheta  = 0.0;
    coord_t cthetac = {0.0, 0.0};
    coord_t sthetac = {0.0, 0.0};
    size_t  i       = 0;
    int     status  = 1;

    assert(fit);
    assert(sx1);
    assert(sy1);
    assert(input);
    assert(ref);
    assert(weights);
    assert(residual_x);
    assert(residual_y);
    assert(error);

    surface_free(sx1);
    surface_free(sy1);

    bbox_copy(&fit->bbox, &bbox);
    bbox_make_nonsingular(&bbox);

    /* Compute the sums required to determine the offsets */
    compute_sums(ncoord, input, ref, weights, &sw, &si, &sr);

    /* Do the fit */
    if (sw < 2.0) {
        if (fit->projection == geomap_proj_none) {
            stimage_error_set_message(
                    error, "Too few data points for X and Y fits.");
            goto exit;
        } else {
            stimage_error_set_message(
                    error, "Too few data points for XI and ETA fits.");
            goto exit;
        }
    }

    /* Compute the sums required to compute the rotation angle */
    r0.x = sr.x / sw;
    r0.y = sr.y / sw;
    i0.x = si.x / sw;
    i0.y = si.y / sw;
    for (i = 0; i < ncoord; ++i) {
        syrxi += weights[i] * (ref[i].y - r0.y) * (input[i].x - i0.x);
        sxryi += weights[i] * (ref[i].x - r0.x) * (input[i].y - i0.y);
        sxryi += weights[i] * (ref[i].x - r0.x) * (input[i].x - i0.x);
        syryi += weights[i] * (ref[i].y - r0.y) * (input[i].y - i0.y);
    }

    /* Compute the rotation angle */
    num = sxrxi * syryi;
    denom = syrxi * sxryi;
    if (double_approx_equal(num, denom)) {
        det = 0.0;
    } else {
        det = num - denom;
    }

    if (det < 0.0) {
        num = syrxi + sxryi;
        denom = -sxrxi + syryi;
    } else {
        num = syrxi - sxryi;
        denom = sxrxi + syryi;
    }

    if (double_approx_equal(num, 0.0) && double_approx_equal(denom, 0.0)) {
        theta = 0.0;
    } else {
        theta = atan2(num, denom);
        if (theta < 0.0) {
            theta += M_PI * 2.0;
        }
    }

    /* Compute the polynomial coefficients */
    ctheta = cos(theta);
    stheta = sin(theta);
    if (det < 0.0) {
        cthetac.x = -ctheta;
        sthetac.y = -stheta;
    } else {
        cthetac.x = ctheta;
        sthetac.y = stheta;
    }
    sthetac.x = stheta;
    cthetac.y = ctheta;

    /* Compute the X and Y fit coefficients */
    if (compute_surface_coefficients(
                fit->function, &bbox, &i0, &r0, &cthetac, &sthetac, sx1, sy1,
                error)) goto exit;

    /* Compute the residuals */
    if (compute_residuals(
                sx1, sy1, ncoord, input, ref, residual_x, residual_y,
                error)) goto exit;

    /* Compute the number of zero-weighted points */
    fit->n_zero_weighted = count_zero_weighted(ncoord, weights);

    /* Compute the rms of the x and y fits */
    compute_rms(
            ncoord, weights, residual_x, residual_y, &fit->xrms, &fit->yrms);

    fit->ncoord = ncoord;

    status = 0;

 exit:

    return status;
}

/* DIFF: was geo_fmagnify */
static int
geo_fit_magnify(
        geomap_fit_t* const fit,
        surface_t* const sx1,
        surface_t* const sy1,
        const size_t ncoord,
        const coord_t* const input,
        const coord_t* const ref,
        const double* const weights,
        double* const residual_x,
        double* const residual_y,
        stimage_error_t* error) {

    bbox_t  bbox;
    double  sw      = 0.0;
    coord_t sr      = {0.0, 0.0};
    coord_t si      = {0.0, 0.0};
    coord_t r0      = {0.0, 0.0};
    coord_t i0      = {0.0, 0.0};
    double  sxrxr   = 0.0;
    double  syryr   = 0.0;
    double  syrxi   = 0.0;
    double  sxryi   = 0.0;
    double  sxrxi   = 0.0;
    double  syryi   = 0.0;
    double  num     = 0.0;
    double  denom   = 0.0;
    double  det     = 0.0;
    double  theta   = 0.0;
    double  ctheta  = 0.0;
    double  stheta  = 0.0;
    double  mag     = 0.0;
    coord_t cthetac = {0.0, 0.0};
    coord_t sthetac = {0.0, 0.0};
    size_t  i       = 0;
    int     status  = 1;

    assert(fit);
    assert(sx1);
    assert(sy1);
    assert(input);
    assert(ref);
    assert(weights);
    assert(residual_x);
    assert(residual_y);

    surface_free(sx1);
    surface_free(sy1);

    bbox_copy(&fit->bbox, &bbox);
    bbox_make_nonsingular(&bbox);

    /* Compute the sums required to determine the offsets */
    compute_sums(ncoord, input, ref, weights, &sw, &si, &sr);

    /* Do the fit */
    if (sw < 2.0) {
        if (fit->projection == geomap_proj_none) {
            stimage_error_set_message(
                    error, "Too few data points for X and Y fits");
        } else {
            stimage_error_set_message(
                    error, "Too few data points for XI and ETA fits");
        }
        goto exit;
    }

    /* Compute the sums */
    r0.x = sr.x / sw;
    r0.y = sr.y / sw;
    i0.x = si.x / sw;
    i0.y = si.y / sw;
    for (i = 0; i < ncoord; ++i) {
        sxrxr += weights[i] * (ref[i].x - r0.x) * (ref[i].x - r0.x);
        syryr += weights[i] * (ref[i].y - r0.y) * (ref[i].y - r0.y);
        syrxi += weights[i] * (ref[i].y - r0.y) * (input[i].x - i0.x);
        sxryi += weights[i] * (ref[i].x - r0.x) * (input[i].y - i0.y);
        sxrxi += weights[i] * (ref[i].x - r0.x) * (input[i].x - i0.x);
        syryi += weights[i] * (ref[i].y - r0.y) * (input[i].y - i0.y);
    }

    /* Compute the rotation angle */
    num = sxrxi * syryi;
    denom = syrxi * sxryi;
    if (double_approx_equal(num, denom)) {
        det = 0.0;
    } else {
        det = num - denom;
    }

    if (det < 0.0) {
        num = syrxi + sxryi;
        denom = -sxrxi + syryi;
    } else {
        num = syrxi - sxryi;
        denom = sxrxi + syryi;
    }

    if (double_approx_equal(num, 0.0) && double_approx_equal(denom, 0.0)) {
        theta = 0.0;
    } else {
        theta = atan2(num, denom);
        if (theta < 0.0) {
            theta += 2.0 * M_PI;
        }
    }

    /* Compute the magnification factor */
    ctheta = cos(theta);
    stheta = sin(theta);
    num = denom * ctheta + num * stheta;
    denom = sxrxr + syryr;
    if (denom <= 0.0) {
        mag = 1.0;
    } else {
        mag = num / denom;
    }

    /* Compute the polynomial coefficients */
    if (det < 0.0) {
        cthetac.x = -mag * ctheta;
        sthetac.y = -mag * stheta;
    } else {
        cthetac.x = mag * ctheta;
        sthetac.y = mag * stheta;
    }
    sthetac.x = mag * stheta;
    cthetac.y = mag * ctheta;

    /* Compute the X fit coefficients */
    if (compute_surface_coefficients(
                fit->function, &bbox, &i0, &r0, &cthetac, &sthetac, sx1, sy1,
                error)) goto exit;

    /* Compute the residuals */
    if (compute_residuals(
                sx1, sy1, ncoord, input, ref, residual_x, residual_y,
                error)) goto exit;

    /* Compute the number of zero-weighted points */
    fit->n_zero_weighted = count_zero_weighted(ncoord, weights);

    /* Compute the rms of the x and y fits */
    compute_rms(
            ncoord, weights, residual_x, residual_y, &fit->xrms, &fit->yrms);

    fit->ncoord = ncoord;

    status = 0;

 exit:

    return status;
}

/* DIFF: was gto_fit_rxyscale */
static int
geo_fit_linear(
        geomap_fit_t* const fit,
        surface_t* const sx1,
        surface_t* const sy1,
        const size_t ncoord,
        const coord_t* const input,
        const coord_t* const ref,
        const double* const weights,
        double* const residual_x,
        double* const residual_y,
        stimage_error_t* error) {

    bbox_t  bbox;
    double  sw      = 0.0;
    coord_t sr      = {0.0, 0.0};
    coord_t si      = {0.0, 0.0};
    coord_t r0      = {0.0, 0.0};
    coord_t i0      = {0.0, 0.0};
    double  sxrxr   = 0.0;
    double  syryr   = 0.0;
    double  syrxi   = 0.0;
    double  sxryi   = 0.0;
    double  sxrxi   = 0.0;
    double  syryi   = 0.0;
    double  num     = 0.0;
    double  denom   = 0.0;
    double  theta   = 0.0;
    double  ctheta  = 0.0;
    double  stheta  = 0.0;
    coord_t cthetac = {0.0, 0.0};
    coord_t sthetac = {0.0, 0.0};
    double  xmag    = 0.0;
    double  ymag    = 0.0;
    size_t  i       = 0;
    int     status  = 1;

    assert(fit);
    assert(sx1);
    assert(sy1);
    assert(input);
    assert(ref);
    assert(weights);
    assert(residual_x);
    assert(residual_y);

    surface_free(sx1);
    surface_free(sy1);

    bbox_copy(&fit->bbox, &bbox);
    bbox_make_nonsingular(&bbox);

    /* Compute the sums required to determine the offsets */
    compute_sums(ncoord, input, ref, weights, &sw, &si, &sr);

    if (sw < 3.0) {
        if (fit->projection == geomap_proj_none) {
            stimage_error_set_message(
                    error, "Too few data points for X and Y fits.");
        } else {
            stimage_error_set_message(
                    error, "Too few data points for XI and ETA fits.");
        }
        goto exit;
    }

    r0.x = sr.x / sw;
    r0.y = sr.y / sw;
    i0.x = si.x / sw;
    i0.y = si.y / sw;
    for (i = 0; i < ncoord; ++i) {
        sxrxr += weights[i] * (ref[i].x - r0.x) * (ref[i].x - r0.x);
        syryr += weights[i] * (ref[i].y - r0.y) * (ref[i].y - r0.y);
        syrxi += weights[i] * (ref[i].y - r0.y) * (input[i].x - i0.x);
        sxryi += weights[i] * (ref[i].x - r0.x) * (input[i].y - i0.y);
        sxrxi += weights[i] * (ref[i].x - r0.x) * (input[i].x - i0.x);
        syryi += weights[i] * (ref[i].y - r0.y) * (input[i].y - i0.y);
    }

    /* Compute the rotation angle */
    num = 2.0 * (sxrxr * syrxi * syryi - syryr * sxrxi * sxryi);
    denom = syryr * (sxrxi - sxryi) * (sxrxi + sxryi) - \
        sxrxr * (syrxi + syryi) * (syrxi - syryi);
    if (double_approx_equal(num, 0.0) && double_approx_equal(denom, 0.0)) {
        theta = 0.0;
    } else {
        theta = atan2(num, denom) / 2.0;
        if (theta < 0.0) {
            theta += M_PI * 2.0;
        }
    }

    ctheta = cos(theta);
    stheta = sin(theta);

    /* Compute the X magnification factor */
    num = sxrxi * ctheta - sxryi * stheta;
    denom = sxrxr;
    if (denom <= 0.0) {
        xmag = 1.0;
    } else {
        xmag = num / denom;
    }

    /* Compute the Y magnification factor */
    num = syrxi * stheta + syryi * ctheta;
    denom = syryr;
    if (denom <= 0.0) {
        ymag = 1.0;
    } else {
        ymag = num / denom;
    }

    /* Compute the polynomial coefficients */
    cthetac.x = xmag * ctheta;
    sthetac.x = ymag * stheta;
    sthetac.y = xmag * stheta;
    cthetac.x = ymag * ctheta;

    /* Compute the X and Y fit coefficients */
    if (compute_surface_coefficients(
                fit->function, &bbox, &i0, &r0, &cthetac, &sthetac, sx1, sy1,
                error)) goto exit;

    /* Compute the residuals */
    if (compute_residuals(
                sx1, sy1, ncoord, input, ref, residual_x, residual_y,
                error)) goto exit;

    /* Compute the number of zero-weighted points */
    fit->n_zero_weighted = count_zero_weighted(ncoord, weights);

    /* Compute the rms of the x and y fits */
    compute_rms(
            ncoord, weights, residual_x, residual_y, &fit->xrms, &fit->yrms);

    fit->ncoord = ncoord;

    status = 0;

 exit:

    return status;
}

static int
_geo_fit_xy_validate_fit_error(
        const surface_fit_error_e error_type,
        const int xfit,
        const geomap_proj_e projection,
        stimage_error_t* error) {

    assert(error);

    switch (error_type) {
    case surface_fit_error_no_degrees_of_freedom:
        if (xfit) {
            if (projection == geomap_proj_none) {
                stimage_error_set_message(
                        error, "Too few data points for X fit.");
            } else {
                stimage_error_set_message(
                        error, "Too few data points for XI fit.");
            }
        } else {
            if (projection == geomap_proj_none) {
                stimage_error_set_message(
                        error, "Too few data points for Y fit.");
            } else {
                stimage_error_set_message(
                        error, "Too few data points for ETA fit.");
            }
        }
        return 1;

    default:
        break;
    }

    return 0;
}

/* was geo_fxyd */
static int
geo_fit_xy(
        geomap_fit_t* const fit,
        surface_t* const sf1,
        surface_t* const sf2,
        const size_t ncoord,
        const int xfit,
        const coord_t* const input,
        const coord_t* const ref,
        /* Output */
        int* has_secondary,
        double* const weights,
        double* const residual,
        stimage_error_t* error) {

    bbox_t              bbox;
    double*             zfit      = NULL;
    const double* const z = (double*)input + (xfit ? 0 : 1);
    surface_t           savefit;
    surface_fit_error_e fit_error = surface_fit_error_ok;
    size_t              i         = 0;
    int                 status    = 1;

    assert(fit);
    assert(sf1);
    assert(sf2);
    assert(ref);
    assert(weights);
    assert(residual);
    assert(has_secondary);
    assert(error);

    surface_new(&savefit);

    surface_free(sf1);
    surface_free(sf2);

    *has_secondary = 1;

    zfit = malloc_with_error(ncoord * sizeof(double), error);
    if (zfit == NULL) goto exit;

    bbox_copy(&fit->bbox, &bbox);
    bbox_make_nonsingular(&bbox);

    if (xfit) {
        switch(fit->fit_geometry) {
        case geomap_fit_shift:
            if (surface_init(
                        &savefit, fit->function, 2, 2, xterms_none, &bbox,
                        error)) goto exit;
            surface_free(sf1);
            if (surface_init(
                        sf1, fit->function, 1, 1, xterms_none, &bbox,
                        error)) goto exit;
            for (i = 0; i < ncoord; ++i) {
                zfit[i] = z[i<<1] - ref[i].x;
            }

            if (surface_fit(
                        sf1, ncoord, ref, zfit, weights,
                        surface_fit_weight_user, &fit_error, error)) goto exit;

            if (fit->function == surface_type_polynomial) {
                savefit.coeff[0] = sf1->coeff[0];
                savefit.coeff[1] = 1.0;
                savefit.coeff[2] = 0.0;
            } else {
                savefit.coeff[0] = sf1->coeff[0] + (bbox.max.x + bbox.min.x) / 2.0;
                savefit.coeff[1] = (bbox.max.x - bbox.min.x) / 2.0;
                savefit.coeff[2] = 0.0;
            }
            surface_free(sf1);
            if (surface_copy(&savefit, sf1, error)) goto exit;
            *has_secondary = 0;
            break;

        case geomap_fit_xyscale:
            if (surface_init(
                        sf1, fit->function, 2, 1, xterms_none, &bbox,
                        error)) goto exit;
            if (surface_fit(
                        sf1, ncoord, ref, z, weights,
                        surface_fit_weight_user, &fit_error, error)) goto exit;
            *has_secondary = 0;
            break;

        default:
            if (surface_init(
                        sf1, fit->function, 2, 2, xterms_none, &bbox,
                        error)) goto exit;
            if (surface_fit(
                        sf1, ncoord, ref, z, weights,
                        surface_fit_weight_user, &fit_error, error)) goto exit;

            if (fit->xxorder > 2 || fit->xyorder > 2 ||
                fit->xxterms == xterms_full) {
                if (surface_init(
                            sf2, fit->function, fit->xxorder, fit->xyorder,
                            fit->xxterms, &fit->bbox, error)) {
                    surface_free(sf1);
                    goto exit;
                }
            } else {
                *has_secondary = 0;
            }
            break;
        }
    } else {
        switch(fit->fit_geometry) {
        case geomap_fit_shift:
            if (surface_init(
                        &savefit, fit->function, 2, 2, xterms_none, &bbox,
                        error)) goto exit;
            surface_free(sf1);
            if (surface_init(
                        sf1, fit->function, 1, 1, xterms_none, &bbox,
                        error)) goto exit;
            for (i = 0; i < ncoord; ++i) {
                zfit[i] = z[i<<1] - ref[i].y;
            }
            if (surface_fit(
                        sf1, ncoord, ref, zfit, weights,
                        surface_fit_weight_user, &fit_error, error)) goto exit;
            if (fit->function == surface_type_polynomial) {
                savefit.coeff[0] = sf1->coeff[0];
                savefit.coeff[1] = 0.0;
                savefit.coeff[2] = 1.0;
            } else {
                savefit.coeff[0] = sf1->coeff[0] + (bbox.min.y + bbox.max.y) / 2.0;
                savefit.coeff[1] = 0.0;
                savefit.coeff[2] = (bbox.max.y - bbox.min.y) / 2.0;
            }
            surface_free(sf1);
            if (surface_copy(&savefit, sf1, error)) goto exit;

            *has_secondary = 0;
            break;

        case geomap_fit_xyscale:
            if (surface_init(
                        sf1, fit->function, 1, 2, xterms_none, &bbox,
                        error)) goto exit;
            if (surface_fit(
                        sf1, ncoord, ref, z, weights,
                        surface_fit_weight_user, &fit_error, error)) goto exit;
            *has_secondary = 0;
            break;

        default:
            if (surface_init(
                        sf1, fit->function, 2, 2, xterms_none, &bbox,
                        error)) goto exit;
            if (surface_fit(
                        sf1, ncoord, ref, z, weights,
                        surface_fit_weight_user, &fit_error, error)) goto exit;
            if (fit->yxorder > 2 || fit->yyorder > 2 ||
                fit->yxterms == xterms_full) {
                if (surface_init(
                            sf2, fit->function, fit->yxorder, fit->yyorder,
                            fit->yxterms, &bbox, error)) goto exit;
            } else {
                *has_secondary = 0;
            }

            break;
        }
    }

    if (_geo_fit_xy_validate_fit_error(
                fit_error, xfit, fit->projection, error)) goto exit;

    if (surface_vector(sf1, ncoord, ref, residual, error)) goto exit;
    for (i = 0; i < ncoord; ++i) {
        residual[i] = z[i<<1] - residual[i];
    }

    /* Calculate the higher-order fit */
    if (*has_secondary) {
        if (surface_fit(
                    sf2, ncoord, ref, residual, weights,
                    surface_fit_weight_user, &fit_error, error)) goto exit;
        if (_geo_fit_xy_validate_fit_error(
                    fit_error, xfit, fit->projection, error)) goto exit;

        if (surface_vector(sf2, ncoord, ref, zfit, error)) goto exit;
        for (i = 0; i < ncoord; ++i) {
            residual[i] = zfit[i] - residual[i];
        }
    }

    /* Compute the number of zero weighted points */
    fit->n_zero_weighted = count_zero_weighted(ncoord, weights);

    /* Calculate the RMS of the fit */
    if (xfit) {
        fit->xrms = 0.0;
        for (i = 0; i < ncoord; ++i) {
            fit->xrms += weights[i] * residual[i] * residual[i];
        }
    } else {
        fit->yrms = 0.0;
        for (i = 0; i < ncoord; ++i) {
            fit->yrms += weights[i] * residual[i] * residual[i];
        }
    }

    fit->ncoord = ncoord;

    status = 0;

 exit:

    surface_free(&savefit);
    free(zfit);

    return status;
}

/* DIFF: was geo_mrejectd */
static int
geo_fit_reject(
        geomap_fit_t* const fit,
        surface_t* const sx1,
        surface_t* const sy1,
        surface_t* const sx2,
        surface_t* const sy2,
        int* const has_sx2,
        int* const has_sy2,
        const size_t ncoord,
        const coord_t* const input,
        const coord_t* const ref,
        const double* const weights,
        double* const residual_x,
        double* const residual_y,
        stimage_error_t* error) {

    double* tweights = NULL;
    size_t  nreject  = 0;
    size_t  niter    = 0;
    double  cutx     = 0.0;
    double  cuty     = 0.0;
    size_t  i        = 0;
    int     status   = 1;

    assert(fit);
    assert(sx1);
    assert(sy1);
    assert(sx2);
    assert(sy2);
    assert(input);
    assert(ref);
    assert(weights);
    assert(residual_x);
    assert(residual_y);
    assert(error);

    tweights = malloc_with_error(ncoord * sizeof(double), error);
    if (tweights == NULL) goto exit;

    if (fit->rej != NULL) {
        free(fit->rej);
    }
    fit->rej = malloc_with_error(ncoord * sizeof(int), error);
    if (fit->rej == NULL) goto exit;

    fit->nreject = 0;

    /* Initialize the temporary weights array and the number of
       rejected points */
    for (i = 0; i < ncoord; ++i) {
        tweights[i] = weights[i];
    }

    do { /* while (niter < fit->maxiter) */
        /* Compute the rejection limits */
        if (ncoord - fit->n_zero_weighted > 1) {
            cutx = fit->reject * \
                sqrt(fit->xrms / (double)(ncoord - fit->n_zero_weighted - 1));
            cuty = fit->reject * \
                sqrt(fit->yrms / (double)(ncoord - fit->n_zero_weighted - 1));
        } else {
            cutx = MAX_DOUBLE;
            cuty = MAX_DOUBLE;
        }

        /* Reject points from the fit */
        for (i = 0; i < ncoord; ++i) {
            if (tweights[i] > 0.0 &&
                ((abs(residual_x[i]) > cutx) || abs(residual_y[i]) > cuty)) {
                tweights[i] = 0.0;
                ++nreject;
                assert(nreject < ncoord);
                fit->rej[nreject] = i;
            }
        }

        if ((long)nreject - (long)fit->nreject <= 0) {
            break;
        }
        fit->nreject = nreject;

        /* Compute the number of deleted points */
        fit->n_zero_weighted = count_zero_weighted(ncoord, weights);

        /* Recompute the X and Y fit */
        switch (fit->fit_geometry) {
        case geomap_fit_rotate:
            if (geo_fit_theta(
                        fit, sx1, sy1, ncoord, input, ref, tweights,
                        residual_x, residual_y, error)) goto exit;
            break;
        case geomap_fit_rscale:
            if (geo_fit_magnify(
                        fit, sx1, sy1, ncoord, input, ref, tweights,
                        residual_x, residual_y, error)) goto exit;
            break;
        case geomap_fit_rxyscale:
            if (geo_fit_linear(
                        fit, sx1, sy1, ncoord, input, ref, tweights,
                        residual_x, residual_y, error)) goto exit;
            break;
        default:
            if (geo_fit_xy(
                        fit, sx1, sx2, ncoord, 1, input, ref, has_sx2, tweights,
                        residual_x, error) ||
                geo_fit_xy(
                        fit, sy1, sy2, ncoord, 0, input, ref, has_sy2, tweights,
                        residual_y, error)) goto exit;
            break;
        }

        /* Compute the X and Y fit rms */
        compute_rms(
                ncoord, tweights, residual_x, residual_y,
                &fit->xrms, &fit->yrms);

        ++niter;
    } while (niter < fit->maxiter);

    status = 0;

 exit:

    free(tweights);

    return status;
}

/* DIFF: was geo_fitd */
static int
geofit(
        geomap_fit_t* const fit,
        surface_t* const sx1,
        surface_t* const sy1,
        surface_t* const sx2,
        surface_t* const sy2,
        int* const has_sx2,
        int* const has_sy2,
        const size_t ncoord,
        const coord_t* const input,
        const coord_t* const ref,
        double* const weights,
        stimage_error_t* error) {

    double* residual_x = NULL;
    double* residual_y = NULL;
    int status = 1;

    assert(fit);
    assert(sx1);
    assert(sy1);
    assert(sx2);
    assert(sy2);
    assert(input);
    assert(ref);
    assert(weights);
    assert(has_sx2);
    assert(has_sy2);
    assert(error);

    *has_sx2 = 0;
    *has_sy2 = 0;

    residual_x = malloc_with_error(ncoord * sizeof(double), error);
    if (residual_x == NULL) goto exit;

    residual_y = malloc_with_error(ncoord * sizeof(double), error);
    if (residual_y == NULL) goto exit;

    switch(fit->fit_geometry) {
    case geomap_fit_rotate:
        if (geo_fit_theta(
                    fit, sx1, sy1, ncoord, input, ref, weights,
                    residual_x, residual_y, error)) goto exit;
        break;
    case geomap_fit_rscale:
        if (geo_fit_magnify(
                    fit, sx1, sy1, ncoord, input, ref, weights,
                    residual_x, residual_y, error)) goto exit;
        break;
    case geomap_fit_rxyscale:
        if (geo_fit_linear(
                    fit, sx1, sy1, ncoord, input, ref, weights,
                    residual_x, residual_y, error)) goto exit;
        break;
    default:
        if (geo_fit_xy(
                    fit, sx1, sx2, ncoord, 0, input, ref, has_sx2, weights,
                    residual_x, error)
            ||
            geo_fit_xy(
                    fit, sy1, sy2, ncoord, 1, input, ref, has_sy2, weights,
                    residual_y, error)) goto exit;
        break;
    }

    if (fit->maxiter <= 0 || !isfinite64(fit->reject)) {
        fit->nreject = 0;
    } else {
        if (geo_fit_reject(
                    fit, sx1, sy1, sx2, sy2, has_sx2, has_sy2, ncoord, input,
                    ref, weights, residual_x, residual_y, error)) goto exit;
    }

    status = 0;

 exit:
    free(residual_x);
    free(residual_y);
    return status;
}

/* DIFF: was geo_evald */
static int
geoeval(
        const surface_t* const sx1,
        const surface_t* const sy1,
        const surface_t* const sx2,
        const surface_t* const sy2,
        const int has_sx2,
        const int has_sy2,
        const size_t ncoord,
        const coord_t* const ref,
        double* const xfit,
        double* const yfit,
        stimage_error_t* const error) {

    double* tmp    = NULL;
    size_t  i      = 0;
    int     status = 1;

    assert(sx1);
    assert(sy1);
    assert(sx2);
    assert(sy2);
    assert(ref);
    assert(xfit);
    assert(yfit);
    assert(error);

    if (has_sx2 || has_sy2) {
        tmp = malloc_with_error(ncoord * sizeof(double), error);
        if (tmp == NULL) goto exit;
    }

    if (surface_vector(sx1, ncoord, ref, xfit, error)) goto exit;
    if (has_sx2) {
        if (surface_vector(sx2, ncoord, ref, tmp, error)) goto exit;
        for (i = 0; i < ncoord; ++i) {
            xfit[i] += tmp[i];
        }
    }

    if (surface_vector(sy1, ncoord, ref, yfit, error)) goto exit;
    if (has_sy2) {
        if (surface_vector(sy2, ncoord, ref, tmp, error)) goto exit;
        for (i = 0; i < ncoord; ++i) {
            yfit[i] += tmp[i];
        }
    }

    status = 0;

 exit:

    free(tmp);

    return status;
}

static int
geo_get_coeff(
        const surface_t* const sx,
        const surface_t* const sy,
        /* Output */
        coord_t* const shift,
        coord_t* const scale,
        coord_t* const rot,
        stimage_error_t* const error) {

    size_t nxxcoeff, nxycoeff, nyxcoeff, nyycoeff;
    double xxrange  = 1.0;
    double xyrange  = 1.0;
    double xxmaxmin = 1.0;
    double xymaxmin = 1.0;
    double yxrange  = 1.0;
    double yyrange  = 1.0;
    double yxmaxmin = 1.0;
    double yymaxmin = 1.0;
    double a, b, c, d;

    assert(sx);
    assert(sy);
    assert(shift);
    assert(scale);
    assert(rot);
    assert(sx->coeff);
    assert(sy->coeff);
    assert(sx->ncoeff >= 3);
    assert(sy->ncoeff >= 3);

    nxxcoeff = nyxcoeff = sx->nxcoeff;
    nxycoeff = nyycoeff = sy->nycoeff;

    /* Get the data range */
    if (sx->type != surface_type_polynomial) {
        xxrange = (sx->bbox.max.x - sx->bbox.min.x) / 2.0;
        xxmaxmin = -(sx->bbox.max.x - sx->bbox.min.x) / 2.0;
        xyrange = (sx->bbox.max.y - sx->bbox.min.y) / 2.0;
        xymaxmin = (sx->bbox.max.y - sx->bbox.min.y) / 2.0;
    }

    if (sy->type != surface_type_polynomial) {
        yxrange = (sy->bbox.max.x - sy->bbox.min.x) / 2.0;
        yxmaxmin = (sy->bbox.max.x - sy->bbox.min.x) / 2.0;
        yyrange = (sy->bbox.max.y - sy->bbox.min.y) / 2.0;
        yymaxmin = (sy->bbox.max.y - sy->bbox.min.y) / 2.0;
    }

    /* Get the shifts */
    shift->x = sx->coeff[0] + \
        sx->coeff[1] * xxmaxmin / xxrange + \
        sx->coeff[2] * xymaxmin / xyrange;
    shift->y = sy->coeff[0] + \
        sy->coeff[1] * yxmaxmin / yxrange +
        sx->coeff[2] * yymaxmin / yyrange;

    /* Get the rotation and scaling parameters */
    if (nxxcoeff > 1) {
        a = sx->coeff[1] / xxrange;
    } else {
        a = 0.0;
    }

    if (nxycoeff > 1) {
        b = sx->coeff[nxxcoeff] / xyrange;
    } else {
        b = 0.0;
    }

    if (nyxcoeff > 1) {
        c = sx->coeff[1] / yxrange;
    } else {
        c = 0.0;
    }

    if (nyycoeff > 1) {
        d = sy->coeff[nyxcoeff] / yyrange;
    } else {
        d = 0.0;
    }

    scale->x = sqrt(a*a + c*c);
    scale->y = sqrt(b*b + d*d);

    if (double_approx_equal(a, 0.0) && double_approx_equal(c, 0.0)) {
        rot->x = 0.0;
    } else {
        rot->x = RADTODEG(atan2(-c, a));
    }
    if (rot->x < 0.0) {
        rot->x += 360.0;
    }

    if (double_approx_equal(b, 0.0) && double_approx_equal(d, 0.0)) {
        rot->y = 0.0;
    } else {
        rot->y = RADTODEG(atan2(b, d));
    }
    if (rot->y < 0.0) {
        rot->y += 360.0;
    }

    return 0;
}

/* Store the results of the coordinate mapping in the result structure */
static int
geo_get_results(
        const geomap_fit_t* const fit,
        const surface_t* const sx1,
        const surface_t* const sy1,
        const surface_t* const sx2,
        const surface_t* const sy2,
        const int has_sx2,
        const int has_sy2,
        /* Output */
        geomap_result_t* const result,
        stimage_error_t* const error) {

    long   ngood  = 0;
    size_t i      = 0;
    int    status = 1;

    assert(fit);
    assert(result);
    assert(error);

    geomap_result_init(result);

    result->fit_geometry = fit->fit_geometry;
    result->function = fit->function;

    ngood = MAX(0, fit->ncoord - fit->n_zero_weighted);

    if (ngood <= 1) {
        result->rms.x = 0.0;
        result->rms.y = 0.0;
    } else {
        result->rms.x = sqrt(fit->xrms / (double)(ngood - 1));
        result->rms.y = sqrt(fit->yrms / (double)(ngood - 1));
    }

    if (geo_get_coeff(
                sx1, sy1, &result->shift, &result->mag, &result->rotation,
                error)) goto exit;

    result->mean_ref.x   = fit->oref.x;
    result->mean_ref.y   = fit->oref.y;
    result->mean_input.x = fit->oin.x;
    result->mean_input.y = fit->oin.y;

    result->nxcoeff = sx1->ncoeff;
    result->xcoeff = malloc_with_error(result->nxcoeff * sizeof(double), error);
    if (result->xcoeff == NULL) goto exit;
    for (i = 0; i < result->nxcoeff; ++i) {
        result->xcoeff[i] = sx1->coeff[i];
    }

    result->nycoeff = sy1->ncoeff;
    result->ycoeff = malloc_with_error(result->nycoeff * sizeof(double), error);
    if (result->ycoeff == NULL) goto exit;
    for (i = 0; i < result->nycoeff; ++i) {
        result->ycoeff[i] = sy1->coeff[i];
    }

    if (has_sx2) {
        result->nx2coeff = sx2->ncoeff;
        result->x2coeff = malloc_with_error(
                result->nx2coeff * sizeof(double), error);
        if (result->x2coeff == NULL) goto exit;
        for (i = 0; i < result->nx2coeff; ++i) {
            result->x2coeff[i] = sx2->coeff[i];
        }

    } else {
        result->nx2coeff = 0;
        result->x2coeff = NULL;
    }

    if (has_sy2) {
        result->ny2coeff = sy2->ncoeff;
        result->y2coeff = malloc_with_error(
                result->ny2coeff * sizeof(double), error);
        if (result->y2coeff == NULL) goto exit;
        for (i = 0; i < result->ny2coeff; ++i) {
            result->y2coeff[i] = sy2->coeff[i];
        }
    } else {
        result->ny2coeff = 0;
        result->y2coeff = NULL;
    }

    status = 0;

 exit:
    if (status != 0) {
        free(result->xcoeff);
        free(result->ycoeff);
        free(result->x2coeff);
        free(result->y2coeff);
    }

    return status;
}

int
geomap(
        const size_t ninput, const coord_t* const input,
        const size_t nref, const coord_t* const ref,
        const bbox_t* const bbox,
        const geomap_fit_e fit_geometry,
        const surface_type_e function,
        const size_t xxorder,
        const size_t xyorder,
        const size_t yxorder,
        const size_t yyorder,
        const xterms_e xxterms,
        const xterms_e yxterms,
        const size_t maxiter,
        const double reject,
        /* Input/Output */
        size_t* const noutput,
        /* Output */
        geomap_output_t* const output, /* [MAX(ninput, nref)] */
        geomap_result_t* const result,
        stimage_error_t* const error) {

    geomap_fit_t     fit;
    bbox_t           tbbox;
    size_t           ninput_in_bbox = ninput;
    size_t           nref_in_bbox   = nref;
    coord_t*         input_in_bbox  = NULL;
    coord_t*         ref_in_bbox    = NULL;
    double*          xfit           = NULL;
    double*          yfit           = NULL;
    double*          weights        = NULL;
    double*          tweights       = NULL;
    geomap_output_t* outi           = NULL;
    surface_t        sx1, sy1, sx2, sy2;
    int              has_sx2        = 0;
    int              has_sy2        = 0;
    size_t           i              = 0;
    double           my_nan         = fmod(1.0, 0.0);
    int              status         = 1;

    assert(input);
    assert(ref);
    assert(error);

    if (ninput != nref) {
        stimage_error_set_message(
            error, "Must have the same number of input and reference coordinates.");
        goto exit;
    }

    surface_new(&sx1);
    surface_new(&sy1);
    surface_new(&sx2);
    surface_new(&sy2);

    geomap_fit_init(
            &fit, geomap_proj_none, fit_geometry, function,
            xxorder, xyorder, xxterms, yxorder, yyorder, yxterms,
            maxiter, reject);

    /* If bbox is NULL, provide a dummy one full of NaNs */
    if (bbox == NULL) {
        bbox_init(&tbbox);
    } else {
        bbox_copy(bbox, &tbbox);
    }

    /* If the bbox is all NaNs, we don't need to reduce the data, saving an
       alloc and copy */
    if (bbox == NULL ||
        (!isfinite64(tbbox.min.x) && !isfinite64(tbbox.min.y) &&
         !isfinite64(tbbox.max.x) && !isfinite64(tbbox.max.y))) {
        /* If we have no bbox, we don't need to allocate and copy */
        input_in_bbox = (coord_t*)input;
        ref_in_bbox = (coord_t*)ref;
        ninput_in_bbox = ninput;
        nref_in_bbox = nref;
    } else {
        input_in_bbox = malloc_with_error(
                ninput * sizeof(coord_t), error);
        if (input_in_bbox == NULL) goto exit;

        ref_in_bbox = malloc_with_error(
                nref * sizeof(coord_t), error);
        if (ref_in_bbox == NULL) goto exit;

        /* Reduce data to only those in the bbox */
        ninput_in_bbox = nref_in_bbox = limit_to_bbox(
                ninput, input, ref, &tbbox, input_in_bbox, ref_in_bbox);
    }

    /* Compute the mean of the reference and input coordinates */
    compute_mean_coord(nref_in_bbox, ref_in_bbox, &fit.oref);
    compute_mean_coord(ninput_in_bbox, input_in_bbox, &fit.oin);

    /* Set the reference point for the projections to undefined */
    fit.refpt.x = my_nan;
    fit.refpt.y = my_nan;

    /* Allocate some memory */
    xfit = malloc_with_error(ninput_in_bbox * sizeof(double), error);
    if (xfit == NULL) goto exit;

    yfit = malloc_with_error(ninput_in_bbox * sizeof(double), error);
    if (yfit == NULL) goto exit;

    /* Compute the weights */
    weights = malloc_with_error(ninput_in_bbox * sizeof(double), error);
    if (weights == NULL) goto exit;

    for (i = 0; i < ninput_in_bbox; ++i) {
        weights[i] = 1.0;
    }

    /* Determine the actual max and min of the coordinates */
    determine_bbox(nref_in_bbox, ref_in_bbox, &tbbox);
    bbox_copy(&tbbox, &fit.bbox);

    if (geofit(
                &fit, &sx1, &sy1, &sx2, &sy2, &has_sx2, &has_sy2,
                ninput_in_bbox, input_in_bbox, ref_in_bbox, weights,
                error)) goto exit;

    /* Compute the fitted x and y values */
    if (geoeval(
                &sx1, &sy1, &sx2, &sy2, has_sx2, has_sy2, ninput_in_bbox,
                ref_in_bbox, xfit, yfit, error)) goto exit;

    if (geo_get_results(
                &fit, &sx1, &sy1, &sx2, &sy2, has_sx2, has_sy2, result,
                error)) goto exit;

    /* DIFF: This section is from geo_plistd */

    /* Copy the results to the output buffer */
    tweights = malloc_with_error(ninput_in_bbox * sizeof(double), error);
    if (tweights == NULL) goto exit;

    for (i = 0; i < ninput_in_bbox; ++i) {
        tweights[i] = weights[i];
    }

    for (i = 0; i < fit.nreject; ++i) {
        assert(fit.rej);
        assert(fit.rej[i] < ninput_in_bbox);
        if (weights[fit.rej[i]] > 0.0) {
            tweights[fit.rej[i]] = 0.0;
        }
    }

    outi = output;
    for (i = 0; i < ninput_in_bbox; ++i, ++outi) {
        outi->ref.x = ref_in_bbox[i].x;
        outi->ref.y = ref_in_bbox[i].y;
        outi->input.x = input_in_bbox[i].x;
        outi->input.y = input_in_bbox[i].y;
        if (tweights[i] > 0.0) {
            outi->fit.x = xfit[i];
            outi->fit.y = yfit[i];
            outi->residual.x = input_in_bbox[i].x - xfit[i];
            outi->residual.y = input_in_bbox[i].y - yfit[i];
        } else {
            outi->fit.x = my_nan;
            outi->fit.y = my_nan;
            outi->residual.x = my_nan;
            outi->residual.y = my_nan;
        }
    }
    *noutput = ninput_in_bbox;

    status = 0;

 exit:

    if (input_in_bbox != input) {
        free(input_in_bbox);
    }
    if (ref_in_bbox != ref) {
        free(ref_in_bbox);
    }
    free(weights);
    free(xfit);
    free(yfit);
    free(tweights);
    surface_free(&sx1);
    surface_free(&sy1);
    surface_free(&sx2);
    surface_free(&sy2);

    return status;
}

void
geomap_result_init(
        geomap_result_t* const r) {

    r->xcoeff = NULL;
    r->ycoeff = NULL;
    r->x2coeff = NULL;
    r->y2coeff = NULL;
}

void
geomap_result_free(
        geomap_result_t* const r) {

    free(r->xcoeff); r->xcoeff = NULL;
    free(r->ycoeff); r->ycoeff = NULL;
    free(r->x2coeff); r->x2coeff = NULL;
    free(r->y2coeff); r->y2coeff = NULL;
}

void
geomap_result_print(
        const geomap_result_t* const r) {

    char*  fit_geometry;
    char*  function;
    size_t i;

    assert(r);

    switch (r->fit_geometry) {
    case geomap_fit_shift:
        fit_geometry = "shift";
        break;

    case geomap_fit_xyscale:
        fit_geometry = "xyscale";
        break;

    case geomap_fit_rotate:
        fit_geometry = "rotate";
        break;

    case geomap_fit_rscale:
        fit_geometry = "rscale";
        break;

    case geomap_fit_rxyscale:
        fit_geometry = "rxyscale";
        break;

    case geomap_fit_general:
        fit_geometry = "general";
        break;

    default:
        fit_geometry = "UNKNOWN";
        break;
    }

    switch (r->function) {
    case surface_type_polynomial:
        function = "polynomial";
        break;

    case surface_type_chebyshev:
        function = "chebyshev";
        break;

    case surface_type_legendre:
        function = "legendre";
        break;

    default:
        function = "UNKNOWN";
        break;
    }

    printf("FIT RESULTS:\n");
    printf("  fit_geometry: %s\n", fit_geometry);
    printf("  function:     %s\n", function);
    printf("  rms:          (%f, %f)\n", r->rms.x, r->rms.y);
    printf("  mean_ref:     (%f, %f)\n", r->mean_ref.x, r->mean_ref.y);
    printf("  mean_input:   (%f, %f)\n", r->mean_input.x, r->mean_input.y);
    printf("  shift:        (%f, %f)\n", r->shift.x, r->shift.y);
    printf("  mag:          (%f, %f)\n", r->mag.x, r->mag.y);
    printf("  rotation:     (%f, %f)\n", r->rotation.x, r->rotation.y);

    if (r->nxcoeff && r->xcoeff) {
        printf("  xcoeff:       ");
        for (i = 0; i < r->nxcoeff; ++i) {
            printf("%f ", r->xcoeff[i]);
        }
        printf("\n");
    }
    if (r->nycoeff && r->ycoeff) {
        printf("  ycoeff:       ");
        for (i = 0; i < r->nycoeff; ++i) {
            printf("%f ", r->ycoeff[i]);
        }
        printf("\n");
    }
    if (r->nx2coeff && r->x2coeff) {
        printf("  x2coeff:       ");
        for (i = 0; i < r->nx2coeff; ++i) {
            printf("%f ", r->x2coeff[i]);
        }
        printf("\n");
    }
    if (r->ny2coeff && r->y2coeff) {
        printf("  y2coeff:       ");
        for (i = 0; i < r->ny2coeff; ++i) {
            printf("%f ", r->y2coeff[i]);
        }
        printf("\n");
    }
    printf("\n");
}
