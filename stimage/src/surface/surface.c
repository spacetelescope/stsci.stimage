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

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "surface/surface.h"

int
surface_init(
        surface_t* const s,
        const surface_type_e function,
        const int xorder,
        const int yorder,
        const xterms_e xterms,
        const bbox_t* const bbox,
        stimage_error_t* const error) {

    int order;

    assert(s);
    assert(bbox);
    assert(error);

    /* NULLify pointers first */
    surface_new(s);

    if (xorder < 1 || yorder < 1) {
        stimage_error_set_message(error, "Illegal order");
        goto fail;
    }

    if (bbox->max.x <= bbox->min.x || bbox->max.y <= bbox->min.y) {
        stimage_error_set_message(error, "Invalid bbox");
        goto fail;
    }

    switch (function) {
    case surface_type_chebyshev:
    case surface_type_legendre:
        s->xorder = xorder;
        s->yorder = yorder;
        s->nxcoeff = xorder;
        s->nycoeff = yorder;
        s->xterms = xterms;
        switch (xterms) {
        case xterms_none:
            s->ncoeff = xorder + yorder - 1;
            break;
        case xterms_half:
            order = MIN(xorder, yorder);
            s->ncoeff = xorder * yorder - order * (order - 1) / 2;
            break;
        case xterms_full:
            s->ncoeff = xorder * yorder;
            break;
        default:
            stimage_error_set_message(error, "Invalid surface xterms value");
            goto fail;
        }
        s->xrange = 2.0 / (bbox->max.x - bbox->min.x);
        s->xmaxmin = -(bbox->max.x - bbox->min.x) / 2.0;
        s->yrange = 2.0 / (bbox->max.y - bbox->min.y);
        s->ymaxmin = -(bbox->max.y - bbox->min.y) / 2.0;
        break;

    case surface_type_polynomial:
        s->xorder = xorder;
        s->yorder = yorder;
        s->nxcoeff = xorder;
        s->nycoeff = yorder;
        s->xterms = xterms;
        switch (xterms) {
        case xterms_none:
            s->ncoeff = xorder + yorder - 1;
            break;
        case xterms_half:
            order = MIN(xorder, yorder);
            s->ncoeff = xorder * yorder - order * (order - 1) / 2;
            break;
        case xterms_full:
            s->ncoeff = xorder * yorder;
            break;
        default:
            stimage_error_set_message(error, "Invalid surface xterms value");
            goto fail;
        }
        s->xrange = 1.0;
        s->xmaxmin = 0.0;
        s->yrange = 1.0;
        s->ymaxmin = 0.0;
        break;

    default:
        stimage_error_set_message(error, "Unknown surface type");
        goto fail;
    }

    s->type = function;
    bbox_copy(bbox, &s->bbox);

    s->matrix =
        malloc_with_error(s->ncoeff * s->ncoeff * sizeof(double), error);
    if (s->matrix == NULL) goto fail;
    s->cholesky_fact =
        malloc_with_error(s->ncoeff * s->ncoeff * sizeof(double), error);
    if (s->cholesky_fact == NULL) goto fail;
    s->vector = malloc_with_error(s->ncoeff * sizeof(double), error);
    if (s->vector == NULL) goto fail;
    s->coeff = malloc_with_error(s->ncoeff * sizeof(double), error);
    if (s->coeff == NULL) goto fail;

    if (surface_zero(s, error)) {
        return 1;
    }

    s->npoints = 0;

    return 0;

 fail:
    surface_free(s);

    return 1;
}

int
surface_new(
        surface_t* const s) {
    memset(s, 0, sizeof(surface_t));

    surface_free(s);

    return 0;
}

void
surface_free(
        surface_t* const s) {

    assert(s);

    free(s->matrix); s->matrix = NULL;
    free(s->cholesky_fact); s->cholesky_fact = NULL;
    free(s->vector); s->vector = NULL;
    free(s->coeff); s->coeff = NULL;
}

static int
surface_copy_vector(
        const size_t size,
        const double* const s,
        double** const d,
        stimage_error_t* const error) {

    size_t i;

    if (s != NULL) {
        free(*d);
        *d = malloc_with_error(size * sizeof(double), error);
        if (*d == NULL) return 1;
        for (i = 0; i < size; ++i) {
            (*d)[i] = s[i];
        }
    }

    return 0;
}

int
surface_copy(
        const surface_t* const s,
        surface_t* const d,
        stimage_error_t* const error) {

    assert(s);
    assert(d);
    assert(error);

    surface_new(d);

    d->type    = s->type;
    d->xorder  = s->xorder;
    d->yorder  = s->yorder;
    d->nxcoeff = s->nxcoeff;
    d->nycoeff = s->nycoeff;
    d->xterms  = s->xterms;
    d->ncoeff  = s->ncoeff;
    d->xrange  = s->xrange;
    d->xmaxmin = s->xmaxmin;
    d->yrange  = s->yrange;
    d->ymaxmin = s->ymaxmin;
    d->npoints = s->npoints;

    bbox_copy(&s->bbox, &d->bbox);

    if (surface_copy_vector(
                s->ncoeff * s->ncoeff, s->matrix, &d->matrix, error) ||
        surface_copy_vector(
                s->ncoeff * s->ncoeff, s->cholesky_fact, &d->cholesky_fact, error) ||
        surface_copy_vector(
                s->ncoeff, s->vector, &d->vector, error) ||
        surface_copy_vector(
                s->ncoeff, s->coeff, &d->coeff, error)) {
        goto fail;
    }

    return 0;

 fail:

    surface_free(d);
    return 1;
}

int
surface_zero(
        surface_t* const s,
        stimage_error_t* const error) {

    size_t i;

    assert(s);
    assert(s->vector);
    assert(s->matrix);

    switch (s->type) {
    case surface_type_legendre:
    case surface_type_polynomial:
    case surface_type_chebyshev:
        /* s->npoints = 0; */

        for (i = 0; i < s->ncoeff; ++i) {
            s->vector[i] = 0.0;
        }

        for (i = 0; i < s->ncoeff; ++i) {
            s->coeff[i] = 0.0;
        }

        for (i = 0; i < s->ncoeff * s->ncoeff; ++i) {
            s->matrix[i] = 0.0;
        }

        for (i = 0; i < s->ncoeff * s->ncoeff; ++i) {
            s->cholesky_fact[i] = 0.0;
        }

        break;
    default:
        stimage_error_set_message(error, "Unknown surface type");
        return 1;
    }

    return 0;
}

void
surface_print(
        const surface_t* const s) {

    char*  type;
    char*  xterms;
    size_t i;

    assert(s);

    switch (s->type) {
    case surface_type_polynomial:
        type = "polynomial";
        break;

    case surface_type_chebyshev:
        type = "chebyshev";
        break;

    case surface_type_legendre:
        type = "legendre";
        break;

    default:
        type = "UNKNOWN";
        break;
    }

    switch (s->xterms) {
    case xterms_none:
        xterms = "none";
        break;

    case xterms_half:
        xterms = "half";
        break;

    case xterms_full:
        xterms = "full";
        break;

    default:
        xterms = "UNKNOWN";
        break;
    }

    printf("SURFACE\n");
    printf("  type:        %s\n", type);
    printf("  order:       %lu, %lu\n", s->xorder, s->yorder);
    printf("  ncoeff:      %lu, %lu\n", s->nxcoeff, s->nycoeff);
    printf("  xterms:      %s\n", xterms);
    printf("  ncoeff:      %lu\n", s->ncoeff);
    printf("  range:       %f, %f\n", s->xrange, s->yrange);
    printf("  maxmin:      %f, %f\n", s->xmaxmin, s->ymaxmin);
    printf("  bbox:        ");
    bbox_print(&s->bbox);
    printf("\n");
    printf("  npoints:     %lu\n", s->npoints);

    if (s->matrix) {
        printf("  matrix:      ");
        for (i = 0; i < s->ncoeff * s->ncoeff; ++i) {
            printf("%f ", s->matrix[i]);
        }
        printf("\n");
    }

    if (s->cholesky_fact) {
        printf("  cholesky:    ");
        for (i = 0; i < s->ncoeff * s->ncoeff; ++i) {
            printf("%f ", s->cholesky_fact[i]);
        }
        printf("\n");
    }

    if (s->vector) {
        printf("  vector:      ");
        for (i = 0; i < s->ncoeff; ++i) {
            printf("%f ", s->vector[i]);
        }
        printf("\n");
    }

    if (s->coeff) {
        printf("  coeff:       ");
        for (i = 0; i < s->ncoeff; ++i) {
            printf("%f ", s->coeff[i]);
        }
        printf("\n");
    }

    printf("\n");
}
