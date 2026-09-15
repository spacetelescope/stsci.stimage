/*
Copyright (C) 2008-2025 Association of Universities for Research in Astronomy (AURA)

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
         help@stsci.edu
*/

#define NO_IMPORT_ARRAY

#include "wrap_util.h"

char *SIZE_T_D;

int
to_coord_t(const char *const name, PyObject *o, coord_t *const c)
{

    PyArrayObject *array = NULL;

    if (o == NULL || o == Py_None) {
        return 0;
    }

    array = (PyArrayObject *) PyArray_FromObject(o, NPY_DOUBLE, 1, 1);
    if (array == NULL) {
        return -1;
    }

    if (PyArray_DIM(array, 0) != 2) {
        Py_DECREF(array);
        PyErr_Format(PyExc_ValueError, "%s must be a pair", name);
        return -1;
    }

    c->x = *((double *) PyArray_GETPTR1(array, 0));
    c->y = *((double *) PyArray_GETPTR1(array, 1));

    Py_DECREF(array);

    return 0;
}

int
to_xyxymatch_algo_e(const char *const name, const char *const s, xyxymatch_algo_e *const e)
{

    if (s == NULL) {
        return 0;
    }

    if (strcmp(s, "tolerance") == 0) {
        *e = xyxymatch_algo_tolerance;
    } else if (strcmp(s, "triangles") == 0) {
        *e = xyxymatch_algo_triangles;
    } else {
        PyErr_Format(PyExc_ValueError, "%s must be 'tolerance' or 'triangles'", name);
        return -1;
    }

    return 0;
}