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

#define NO_IMPORT_ARRAY

#include "wrap_util.h"

char* SIZE_T_D;

int
to_coord_t(
        const char* const name,
        PyObject* o,
        coord_t* const c) {

    PyObject* array = NULL;

    if (o == NULL || o == Py_None) {
        return 0;
    }

    array = PyArray_FromObject(o, NPY_DOUBLE, 1, 1);
    if (array == NULL) {
        return -1;
    }

    if (PyArray_DIM(array, 0) != 2) {
        Py_DECREF(array);
        PyErr_Format(
                PyExc_ValueError,
                "%s must be a pair",
                name);
        return -1;
    }

    c->x = *((double*)PyArray_GETPTR1(array, 0));
    c->y = *((double*)PyArray_GETPTR1(array, 1));

    Py_DECREF(array);

    return 0;
}

int
from_coord_t(
        const coord_t* const c,
        PyObject** o) {

    npy_intp dims = 2;

    *o = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
    if (*o == NULL) {
        return -1;
    }

    *((double*)PyArray_GETPTR1(*o, 0)) = c->x;
    *((double*)PyArray_GETPTR1(*o, 1)) = c->y;

    return 0;
}

int
to_bbox_t(
        const char* const name,
        PyObject* o,
        bbox_t* const b) {

    PyObject* array;
    double* data;

    if (o == NULL || o == Py_None) {
        return 0;
    }

    array = PyArray_ContiguousFromAny(o, NPY_DOUBLE, 1, 2);
    if (array == NULL) {
        return -1;
    }

    if ((PyArray_NDIM(array) == 1 &&
         PyArray_DIM(array, 0) != 4) ||
        (PyArray_NDIM(array) == 2 &&
         (PyArray_DIM(array, 0) != 2 || PyArray_DIM(array, 1) != 2))) {
        PyErr_Format(
                PyExc_ValueError,
                "%s must be a length-4 or 2x2 sequence",
                name);
        Py_DECREF(array);
        return -1;
    }

    data = (double*)PyArray_DATA(array);
    b->min.x = data[0];
    b->min.y = data[1];
    b->max.x = data[2];
    b->max.y = data[3];

    Py_DECREF(array);

    return 0;
}

int
to_xyxymatch_algo_e(
        const char* const name,
        const char* const s,
        xyxymatch_algo_e* const e) {

    if (s == NULL) {
        return 0;
    }

    if (strcmp(s, "tolerance") == 0) {
        *e = xyxymatch_algo_tolerance;
    } else if (strcmp(s, "triangles") == 0) {
        *e = xyxymatch_algo_triangles;
    } else {
        PyErr_Format(
                PyExc_ValueError,
                "%s must be 'tolerance' or 'triangles'",
                name);
        return -1;
    }

    return 0;
}

int
to_geomap_fit_e(
        const char* const name,
        const char* const s,
        geomap_fit_e* const e) {

    if (s == NULL) {
        return 0;
    }

    if (strcmp(s, "general") == 0) {
        *e = geomap_fit_general;
        return 0;
    } else if (s[0] == 'r') {
        if (strcmp(s, "rotate") == 0) {
            *e = geomap_fit_rotate;
            return 0;
        } else if (strcmp(s, "rscale") == 0) {
            *e = geomap_fit_rscale;
            return 0;
        } else if (strcmp(s, "rxyscale") == 0) {
            *e = geomap_fit_rxyscale;
            return 0;
        }
    } else if (strcmp(s, "shift") == 0) {
        *e = geomap_fit_shift;
        return 0;
    } else if (strcmp(s, "xyscale") == 0) {
        *e = geomap_fit_xyscale;
        return 0;
    }

    PyErr_Format(
            PyExc_ValueError,
            "%s must be 'shift', 'xyscale', 'rotate', 'rscale', 'rxyscale' or 'general'",
            name);
    return -1;
}

int
from_geomap_fit_e(
        const geomap_fit_e e,
        PyObject** o) {

    const char* c;

    switch (e) {
    case geomap_fit_rotate:
        c = "rotate";
        break;
    case geomap_fit_rscale:
        c = "rscale";
        break;
    case geomap_fit_rxyscale:
        c = "rxyscale";
        break;
    case geomap_fit_shift:
        c = "shift";
        break;
    case geomap_fit_xyscale:
        c = "xyscale";
        break;
    case geomap_fit_general:
        c = "general";
        break;
    default:
        PyErr_SetString(
                PyExc_ValueError,
                "Unknown geomap_fit_e value");
        return -1;
    }

#if PY_MAJOR_VERSION >= 3
    *o = PyUnicode_FromString(c);
#else
    *o = PyString_FromString(c);
#endif
    if (*o == NULL) {
        return -1;
    }

    return 0;
}

int
to_surface_type_e(
        const char* const name,
        const char* const s,
        surface_type_e* const e) {

    if (s == NULL) {
        return 0;
    }

    if (strcmp(s, "polynomial") == 0) {
        *e = surface_type_polynomial;
        return 0;
    } else if (strcmp(s, "legendre") == 0) {
        *e = surface_type_legendre;
        return 0;
    } else if (strcmp(s, "chebyshev") == 0) {
        *e = surface_type_chebyshev;
        return 0;
    }

    PyErr_Format(
            PyExc_ValueError,
            "%s must be 'polynomial', 'legendre' or 'chebyshev'",
            name);
    return -1;
}

int
from_surface_type_e(
        const surface_type_e e,
        PyObject** o) {

    const char* c;

    switch (e) {
    case surface_type_polynomial:
        c = "polynomial";
        break;
    case surface_type_legendre:
        c = "legendre";
        break;
    case surface_type_chebyshev:
        c = "chebyshev";
        break;
    default:
        PyErr_SetString(
                PyExc_ValueError,
                "Unknown surface_type_e value");
        return -1;
    }

#if PY_MAJOR_VERSION >= 3
    *o = PyUnicode_FromString(c);
#else
    *o = PyString_FromString(c);
#endif
    if (*o == NULL) {
        return -1;
    }

    return 0;
}

int
to_xterms_e(
        const char* const name,
        const char* const s,
        xterms_e* const e) {

    if (s == NULL) {
        return 0;
    }

    if (strcmp(s, "none") == 0) {
        *e = xterms_none;
        return 0;
    } else if (strcmp(s, "half") == 0) {
        *e = xterms_half;
        return 0;
    } else if (strcmp(s, "full") == 0) {
        *e = xterms_full;
        return 0;
    }

    PyErr_Format(
            PyExc_ValueError,
            "%s must be 'none', 'half', or 'full''",
            name);
    return -1;
}

int
from_xterms_e(
        const xterms_e e,
        PyObject** o) {

    const char* c;

    switch (e) {
    case xterms_none:
        c = "none";
        break;
    case xterms_half:
        c = "half";
        break;
    case xterms_full:
        c = "full";
        break;
    default:
        PyErr_SetString(
                PyExc_ValueError,
                "Unknown xterms_e value");
        return -1;
    }

#if PY_MAJOR_VERSION >= 3
    *o = PyUnicode_FromString(c);
#else
    *o = PyString_FromString(c);
#endif
    if (*o == NULL) {
        return -1;
    }

    return 0;
}
