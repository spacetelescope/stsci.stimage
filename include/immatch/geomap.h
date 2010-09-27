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

#ifndef _STIMAGE_GEOMAP_H_
#define _STIMAGE_GEOMAP_H_

#include "lib/util.h"
#include "lib/xybbox.h"
#include "surface/surface.h"

typedef enum {
    geomap_fit_shift,
    geomap_fit_xyscale,
    geomap_fit_rotate,
    geomap_fit_rscale,
    geomap_fit_rxyscale,
    geomap_fit_general,
    geomap_fit_LAST
} geomap_fit_e;

typedef enum {
    geomap_proj_none,
    geomap_proj_lin,
    geomap_proj_azp,
    geomap_proj_tan,
    geomap_proj_sin,
    geomap_proj_stg,
    geomap_proj_arc,
    geomap_proj_zpn,
    geomap_proj_zea,
    geomap_proj_air,
    geomap_proj_cyp,
    geomap_proj_car,
    geomap_proj_mer,
    geomap_proj_cea,
    geomap_proj_cop,
    geomap_proj_cod,
    geomap_proj_coe,
    geomap_proj_coo,
    geomap_proj_bon,
    geomap_proj_pco,
    geomap_proj_gls,
    geomap_proj_par,
    geomap_proj_ait,
    geomap_proj_mol,
    geomap_proj_csc,
    geomap_proj_qsc,
    geomap_proj_tsc,
    geomap_proj_tnx,
    geomap_proj_zpx,
    geomap_proj_LAST
} geomap_proj_e;

typedef struct {
    coord_t input;
    coord_t ref;
    coord_t fit;
    coord_t residual;
} geomap_output_t;

typedef struct {
    geomap_fit_e fit_geometry;
    surface_type_e function;
    coord_t rms;
    coord_t mean_ref;
    coord_t mean_input;
    coord_t shift;
    coord_t mag;
    coord_t rotation;
    size_t nxcoeff;
    double* xcoeff;
    size_t nycoeff;
    double* ycoeff;
    size_t nx2coeff;
    double* x2coeff;
    size_t ny2coeff;
    double* y2coeff;
} geomap_result_t;

/**
Initialize the geomap_result object.
*/
void
geomap_result_init(
        geomap_result_t* const r);

/**
Free the dynamically allocated arrays in the geomap_result_t object.
*/
void
geomap_result_free(
        geomap_result_t* const r);

/**
`geomap` computes the transformation required to map the reference
coordinate system to the input coordinate system.

@param ninput Number of input coordinates.

@param input Array of input coordinates.

@param nref Number of reference coordinates.

@param ref Array of reference coordinates.

@param bbox The range of reference coordinates over which the computed
       coordinate transformation is valid.

       If the user is working in pixel units, these limits should
       normally be set to the values of the column and row limits of
       the reference image, e.g ``[1.0, 1.0, 512.0, 512.0]`` for a 512
       x 512 image. The minimum and maximum values in *ref* input are
       used if *bbox* is `None` or any of its members are NaN.

@param fit_geometry The fitting geometry to be used.  The options
       are the following:

       - geomap_fit_shift: *x* and *y* shifts are only fit.

       - geomap_fit_xyscale: *x* and *y* shifts and *x* and *y*
         magnification factors are fit.  Axis flips are allowed for.

       - geomap_fit_rotate: *x* and *y* shifts and a rotation angle
         are fit.  Axis flips are allowed for.

       - geomap_fit_rscale: *x* and *x* shifts, a magnification factor
         assumed to be the same in *x* and *y*, and a rotation angle
         are fit. Axis flips are allowed for.

       - geomap_fit_rxyscale: *x* and *y* shifts, *x* and *y*
         magnifications factors, and a rotation angle are fit. Axis
         flips are allowed for.

       - geomap_fit_general: A polynomial of arbitrary order in *x*
         and *y* is fit. A linear term and a distortion term are
         computed separately. The linear term includes an *x* and *y*
         shift, an *x* and *y* scale factor, a rotation and a
         skew. Axis flips are also allowed for in the linear portion
         of the fit. The distortion term consists of a polynomial fit
         to the residuals of the linear term. By default the
         distortion term is set to zero.

       For all the fitting geometries except geomap_fit_general, no
       distortion term is fit, i.e. the *x* and *y* polynomial orders
       are assumed to be 2 and the cross term switches (*xyterms* and
       *yxterms*) are assumed to be xterms_none, regardless of the
       values of the *xxorder*, *xyorder*, *xxterms*, *yxorder*,
       *yyorder* and *yxterms* parameters set by the user.

@param function The type of analytic surface to be fit. The options
       are the following:

       - surface_type_legendre: Legendre polynomials in *x* and *y*.

       - surface_type_chevyshev: Chebyshev polynomials in *x* and *y*.

       - surface_type_polynomial: Power series in *x* and *y*.

@param xxorder
@param xyorder
@param yxorder
@param yyorder The order of the polynomials in *x* and *y* for the *x*
       and *y* fits respectively. The default order and cross term
       settings define the linear term in *x* and *y*, where the 6
       coefficients can be interpreted in terms of an *x* and *y*
       shift, an *x* and *y* scale change, and rotations of the *x*
       and *y* axes. The "shift", "xyscale", "rotation", "rscale", and
       "rxyscale", fitting geometries assume that the polynomial order
       parameters are 2 regardless of the values set by the user. If
       any of the order parameters are higher than 2 and
       *fit_geometry* is geomap_fit_general, then a distortion surface
       is fit to the residuals from the linear portion of the fit.

@param xxterms
@param yxterms The options are:

       - xterms_none: The individual polynomial terms contain powers
         of *x* or powers of *y* but not powers of both.

       - xterms_half: The individual polynomial terms contain powers
         of *x* and powers of *y*, whose maximum combined power is
         ``max(xxorder - 1, xyorder - 1)`` for the *x* fit and
         ``max(yxorder - 1, yyorder - 1)`` for the *y* fit.

       - xterms_full: The individual polynomial terms contain powers
         of *x* and powers of *y*, whose maximum combined power is
         ``max(xxorder - 1 + xyorder - 1)`` for the *x* fit and
         ``max(yxorder - 1 + yyorder - 1)`` for the *y* fit.

       The "shift", "xyscale", "rotation", "rscale", and "rxyscale"
       fitting geometries, assume that the cross term switches are set
       to "none" regardless of the values set by the user. If either
       of the cross terms parameters are set to "half" or "full" and
       *fit_geometry* is "general" then a distortion surface is fit to
       the residuals from the linear portion of the fit.

@param maxiter The maximum number of rejection iterations. 0 means no
       rejection.

@param reject The rejection limit in units of sigma.

@param noutput The number of output records returned

@param output An array of output records matching input and reference
       coordinates with their fit and residual values.

@param result A structure defining the fit that was found.

@param error

@return Non-zero on error
 */
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
        /* Input/output */
        size_t* const noutput,
        /* Output */
        geomap_output_t* const output, /* [MAX(ninput, nref)] */
        geomap_result_t* const result,
        stimage_error_t* const error);

void
geomap_result_print(
        const geomap_result_t* const result);

#endif /* _STIMAGE_GEOMAP_H_ */
