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

#include <Python.h>
#include "wrap_util.h"

#include "immatch/xyxymatch.h"

PyObject*
py_xyxymatch(PyObject* self, PyObject* args, PyObject* kwds) {
    PyObject* input_obj      = NULL;
    PyObject* ref_obj        = NULL;
    PyObject* origin_obj     = NULL;
    PyObject* mag_obj        = NULL;
    PyObject* rotation_obj   = NULL;
    PyObject* ref_origin_obj = NULL;
    char*     algorithm_str  = NULL;
    double    tolerance      = 1.0;
    double    separation     = 9.0;
    size_t    nmatch         = 30;
    double    maxratio       = 10.0;
    size_t    nreject        = 10;

    PyObject*        input_array = NULL;
    PyObject*        ref_array   = NULL;
    coord_t          origin      = {0.0, 0.0};
    coord_t          mag         = {1.0, 1.0};
    coord_t          rotation    = {0.0, 0.0};
    coord_t          ref_origin  = {0.0, 0.0};
    xyxymatch_algo_e algorithm   = xyxymatch_algo_tolerance;

    PyObject*           result     = NULL;
    size_t              noutput    = 0;
    xyxymatch_output_t* output     = NULL;
    PyObject*           dtype_list = NULL;
    PyArray_Descr*      dtype      = NULL;
    npy_intp            dims;
    stimage_error_t     error;

    const char*    keywords[]    = {
        "input", "ref", "origin", "mag", "rotation", "ref_origin", "algorithm",
        "tolerance", "separation", "nmatch", "maxratio", "nreject", NULL
    };

    stimage_error_init(&error);

    if (!PyArg_ParseTupleAndKeywords(
                args, kwds, "OO|OOOOsddndn:xyxymatch",
                (char **)keywords,
                &input_obj, &ref_obj, &origin_obj, &mag_obj, &rotation_obj,
                &ref_origin_obj, &algorithm_str, &tolerance, &separation,
                &nmatch, &maxratio, &nreject)) {
        return NULL;
    }

    input_array = (PyObject*)PyArray_ContiguousFromAny(
            input_obj, NPY_DOUBLE, 2, 2);
    if (input_array == NULL) {
        goto exit;
    }
    if (PyArray_DIM(input_array, 1) != 2) {
        PyErr_SetString(PyExc_TypeError, "input array must be an Nx2 array");
        goto exit;
    }

    ref_array = (PyObject*)PyArray_ContiguousFromAny(
            ref_obj, NPY_DOUBLE, 2, 2);
    if (ref_array == NULL) {
        goto exit;
    }
    if (PyArray_DIM(ref_array, 1) != 2) {
        PyErr_SetString(PyExc_TypeError, "ref array must be an Nx2 array");
        goto exit;
    }

    if (to_coord_t("origin", origin_obj, &origin) ||
        to_coord_t("mag", mag_obj, &mag) ||
        to_coord_t("rotation", rotation_obj, &rotation) ||
        to_coord_t("ref_origin", ref_origin_obj, &ref_origin) ||
        to_xyxymatch_algo_e("algorithm", algorithm_str, &algorithm)) {
        goto exit;
    }

    noutput = PyArray_DIM(input_array, 0);
    output = malloc(noutput * sizeof(xyxymatch_output_t));
    if (output == NULL) {
        result = PyErr_NoMemory();
        goto exit;
    }
    if (xyxymatch(
                PyArray_DIM(input_array, 0), (coord_t*)PyArray_DATA(input_array),
                PyArray_DIM(ref_array, 0), (coord_t*)PyArray_DATA(ref_array),
                &noutput, output,
                &origin, &mag, &rotation, &ref_origin,
                algorithm, tolerance, separation, nmatch, maxratio, nreject,
                &error)) {
        PyErr_SetString(PyExc_RuntimeError, stimage_error_get_message(&error));
        goto exit;
    }

    dtype_list = Py_BuildValue(
            "[(ss)(ss)(ss)(ss)(ss)(ss)]",
            "input_x", "f8",
            "input_y", "f8",
            "input_idx", SIZE_T_D,
            "ref_x", "f8",
            "ref_y", "f8",
            "ref_idx", SIZE_T_D);
    if (dtype_list == NULL) {
        goto exit;
    }
    if (!PyArray_DescrConverter(dtype_list, &dtype)) {
        goto exit;
    }
    Py_DECREF(dtype_list);
    dims = (npy_intp)noutput;
    result = PyArray_NewFromDescr(
            &PyArray_Type, dtype, 1, &dims, NULL, output, NPY_OWNDATA, NULL);

 exit:

    Py_DECREF(input_array);
    Py_DECREF(ref_array);
    if (result == NULL) {
        free(output);
    }

    return result;
}
