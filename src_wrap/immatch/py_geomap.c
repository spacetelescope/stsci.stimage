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

#include "immatch/geomap.h"

PyObject*
py_geomap(PyObject* self, PyObject* args, PyObject* kwds) {
    PyObject* input_obj        = NULL;
    PyObject* ref_obj          = NULL;
    PyObject* bbox_obj         = NULL;
    char*     fit_geometry_str = NULL;
    char*     surface_type_str = NULL;
    size_t    xxorder          = 2;
    size_t    xyorder          = 2;
    size_t    yxorder          = 2;
    size_t    yyorder          = 2;
    char*     xxterms_str      = NULL;
    char*     yxterms_str      = NULL;
    size_t    maxiter          = 0;
    double    reject           = 0.0;

    size_t         ninput       = 0;
    PyObject*      input_array  = NULL;
    size_t         nref         = 0;
    PyObject*      ref_array    = NULL;
    bbox_t         bbox;
    geomap_fit_e   fit_geometry = geomap_fit_general;
    surface_type_e surface_type = surface_type_polynomial;
    xterms_e       xxterms      = xterms_half;
    xterms_e       yxterms      = xterms_half;

    static PyObject* class        = NULL;
    geomap_result_t  fit;
    PyObject*        fit_obj      = NULL;
    PyObject*        tmp          = NULL;
    npy_intp         dims         = 0;
    size_t           i            = 0;
    size_t           noutput      = 0;
    geomap_output_t* output       = NULL;
    PyObject*        dtype_list   = NULL;
    PyArray_Descr*   dtype        = NULL;
    PyObject*        result       = NULL;
    PyObject*        output_array = NULL;
    stimage_error_t  error;

    const char*    keywords[]    = {
        "input", "ref", "bbox", "fit_geometry", "function",
        "xxorder", "xyorder", "yxorder", "yyorder", "xxterms",
        "yxterms", "maxiter", "reject", NULL
    };

    bbox_init(&bbox);
    geomap_result_init(&fit);
    stimage_error_init(&error);

    if (!PyArg_ParseTupleAndKeywords(
                args, kwds, "OO|Ossnnnnssnd:geomap",
                (char **)keywords,
                &input_obj, &ref_obj, &bbox_obj, &fit_geometry_str,
                &surface_type_str, &xxorder, &xyorder, &yxorder, &yyorder,
                &xxterms_str, &yxterms_str, &maxiter, &reject)) {
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

    if (to_bbox_t("bbox", bbox_obj, &bbox) ||
        to_geomap_fit_e("fit_geometry", fit_geometry_str, &fit_geometry) ||
        to_surface_type_e("surface_type", surface_type_str, &surface_type) ||
        to_xterms_e("xxterms", xxterms_str, &xxterms) ||
        to_xterms_e("yxterms", yxterms_str, &yxterms)) {
        goto exit;
    }

    ninput = PyArray_DIM(input_array, 0);
    nref = PyArray_DIM(ref_array, 0);
    noutput = MAX(ninput, nref);
    output = malloc(noutput * sizeof(geomap_output_t));
    if (output == NULL) {
        result = PyErr_NoMemory();
        goto exit;
    }

    if (geomap(
                ninput, (coord_t*)PyArray_DATA(input_array),
                nref, (coord_t*)PyArray_DATA(ref_array),
                &bbox, fit_geometry, surface_type,
                xxorder, xyorder, yxorder, yyorder,
                xxterms, yxterms,
                maxiter, reject,
                &noutput, output, &fit,
                &error)) {
        PyErr_SetString(PyExc_RuntimeError, stimage_error_get_message(&error));
        goto exit;
    }

    dtype_list = Py_BuildValue(
            "[(ss)(ss)(ss)(ss)(ss)(ss)(ss)(ss)]",
            "input_x", "f8",
            "input_y", "f8",
            "ref_x", "f8",
            "ref_y", "f8",
            "fit_x", "f8",
            "fit_y", "f8",
            "resid_x", "f8",
            "resid_y", "f8");
    if (dtype_list == NULL) {
        goto exit;
    }
    if (!PyArray_DescrConverter(dtype_list, &dtype)) {
        goto exit;
    }
    Py_DECREF(dtype_list);
    dims = (npy_intp)noutput;
    output_array = PyArray_NewFromDescr(
            &PyArray_Type, dtype, 1, &dims, NULL, output,
            NPY_OWNDATA, NULL);
    if (output_array == NULL) {
        goto exit;
    }

    if (class == NULL) {
        class = PyClass_New(
                NULL, PyDict_New(), PyString_FromString("GeomapResults"));
    }

    fit_obj = PyInstance_New(class, NULL, NULL);

    #define ADD_ATTR(func, member, name) \
        if ((func)((member), &tmp)) goto exit;      \
        PyObject_SetAttrString(fit_obj, (name), tmp);       \
        Py_DECREF(tmp);

    #define ADD_ARRAY(size, member, name) \
        dims = (size); \
        tmp = PyArray_SimpleNew(1, &dims, NPY_DOUBLE); \
        if (tmp == NULL) goto exit; \
        for (i = 0; i < (size); ++i) ((double*)PyArray_DATA(tmp))[i] = (member)[i]; \
        PyObject_SetAttrString(fit_obj, (name), tmp); \
        Py_DECREF(tmp);

    ADD_ATTR(from_geomap_fit_e, fit.fit_geometry, "fit_geometry");
    ADD_ATTR(from_surface_type_e, fit.function, "function");
    ADD_ATTR(from_coord_t, &fit.rms, "rms");
    ADD_ATTR(from_coord_t, &fit.mean_ref, "mean_ref");
    ADD_ATTR(from_coord_t, &fit.mean_input, "mean_input");
    ADD_ATTR(from_coord_t, &fit.shift, "shift");
    ADD_ATTR(from_coord_t, &fit.mag, "mag");
    ADD_ATTR(from_coord_t, &fit.rotation, "rotation");
    ADD_ARRAY(fit.nxcoeff, fit.xcoeff, "xcoeff");
    ADD_ARRAY(fit.nycoeff, fit.ycoeff, "ycoeff");
    ADD_ARRAY(fit.nx2coeff, fit.x2coeff, "x2coeff");
    ADD_ARRAY(fit.ny2coeff, fit.y2coeff, "y2coeff");

    result = Py_BuildValue("OO", fit_obj, output_array);

 exit:

    Py_DECREF(input_array);
    Py_DECREF(ref_array);
    geomap_result_free(&fit);
    if (result == NULL) {
        Py_XDECREF(output_array);
        free(output);
        Py_XDECREF(fit_obj);
    }

    return result;
}
