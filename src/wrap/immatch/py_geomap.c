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
#include "immatch/geomap.h"
#include <structmember.h>

typedef struct {
    PyObject_HEAD PyObject *fit_geometry;
    PyObject *function;
    PyArrayObject *rms;
    PyArrayObject *mean_ref;
    PyArrayObject *mean_input;
    PyArrayObject *shift;
    PyArrayObject *mag;
    PyArrayObject *rotation;
    PyArrayObject *xcoeff;
    PyArrayObject *ycoeff;
    PyArrayObject *x2coeff;
    PyArrayObject *y2coeff;
} geomap_object;

static PyObject *
geomap_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    (void) args;
    (void) kwds;
    return PyType_GenericAlloc(type, 0);
}

static PyArrayObject *
geomap_array_init(void)
{
    PyArrayObject *o = NULL;
    npy_intp dims = 1;

    o = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
    if (o != NULL) {
        *((double *) PyArray_GETPTR1(o, 0)) = 0.0;
    }

    return o;
}

static int
geomap_init(geomap_object *self, PyObject *args, PyObject *kwds)
{
    self->fit_geometry = PyUnicode_FromString("");
    self->function = PyUnicode_FromString("");

    self->rms = geomap_array_init();
    if (self->rms == NULL) {
        return -1;
    }

    self->mean_ref = geomap_array_init();
    if (self->mean_ref == NULL) {
        return -1;
    }

    self->mean_input = geomap_array_init();
    if (self->mean_input == NULL) {
        return -1;
    }

    self->shift = geomap_array_init();
    if (self->shift == NULL) {
        return -1;
    }

    self->mag = geomap_array_init();
    if (self->mag == NULL) {
        return -1;
    }

    self->rotation = geomap_array_init();
    if (self->rotation == NULL) {
        return -1;
    }

    self->xcoeff = geomap_array_init();
    if (self->xcoeff == NULL) {
        return -1;
    }

    self->ycoeff = geomap_array_init();
    if (self->ycoeff == NULL) {
        return -1;
    }

    self->x2coeff = geomap_array_init();
    if (self->x2coeff == NULL) {
        return -1;
    }

    self->y2coeff = geomap_array_init();
    if (self->y2coeff == NULL) {
        return -1;
    }

    return 0;
}

static void
geomap_dealloc(geomap_object *self)
{
    Py_XDECREF(self->fit_geometry);
    Py_XDECREF(self->function);
    Py_XDECREF((PyObject *) self->rms);
    Py_XDECREF((PyObject *) self->mean_ref);
    Py_XDECREF((PyObject *) self->mean_input);
    Py_XDECREF((PyObject *) self->shift);
    Py_XDECREF((PyObject *) self->mag);
    Py_XDECREF((PyObject *) self->rotation);
    Py_XDECREF((PyObject *) self->xcoeff);
    Py_XDECREF((PyObject *) self->ycoeff);
    Py_XDECREF((PyObject *) self->x2coeff);
    Py_XDECREF((PyObject *) self->y2coeff);
    {
        PyTypeObject *tp = Py_TYPE((PyObject *) self);
        freefunc tp_free = (freefunc) PyType_GetSlot(tp, Py_tp_free);
        tp_free((PyObject *) self);
        Py_DECREF(tp);
    }
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-function-type-mismatch"
#endif

/* Table of functions callable from python */
static PyMethodDef geomap_methods[] = {
    {NULL, NULL} /* Sentinel */
};

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

static PyMemberDef geomap_members[] = {
    {"fit_geometry", T_OBJECT_EX, offsetof(geomap_object, fit_geometry), 0, "fit_geometry"},
    {"function", T_OBJECT_EX, offsetof(geomap_object, function), 0, "function"},
    {"rms", T_OBJECT_EX, offsetof(geomap_object, rms), 0, "rms"},
    {"mean_ref", T_OBJECT_EX, offsetof(geomap_object, mean_ref), 0, "mean_ref"},
    {"mean_input", T_OBJECT_EX, offsetof(geomap_object, mean_input), 0, "mean_input"},
    {"shift", T_OBJECT_EX, offsetof(geomap_object, shift), 0, "shift"},
    {"mag", T_OBJECT_EX, offsetof(geomap_object, mag), 0, "mag"},
    {"rotation", T_OBJECT_EX, offsetof(geomap_object, rotation), 0, "rotation"},
    {"xcoeff", T_OBJECT_EX, offsetof(geomap_object, xcoeff), 0, "xcoeff"},
    {"ycoeff", T_OBJECT_EX, offsetof(geomap_object, ycoeff), 0, "ycoeff"},
    {"x2coeff", T_OBJECT_EX, offsetof(geomap_object, x2coeff), 0, "x2coeff"},
    {"y2coeff", T_OBJECT_EX, offsetof(geomap_object, y2coeff), 0, "y2coeff"},
    {0, 0, 0, 0, 0} /* Sentinel */
};

static PyObject *geomap_class = NULL;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types-discards-qualifiers"
#endif

static PyType_Slot geomap_class_slots[] = {
    {Py_tp_doc, PyDoc_STR("geomap result objects")},
    {Py_tp_dealloc, (void *) geomap_dealloc},
    {Py_tp_methods, geomap_methods},
    {Py_tp_members, geomap_members},
    {Py_tp_init, (void *) geomap_init},
    {Py_tp_new, (void *) geomap_new},
    {0, NULL},
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

static PyType_Spec geomap_class_spec = {
    .name = "py_geomap.GeomapResults",
    .basicsize = sizeof(geomap_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = geomap_class_slots,
};

PyObject *
py_geomap(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *input_obj = NULL;
    PyObject *ref_obj = NULL;
    PyObject *bbox_obj = NULL;
    PyObject *fit_obj = NULL;
    char *fit_geometry_str = NULL;
    char *surface_type_str = NULL;
    size_t xxorder = 2;
    size_t xyorder = 2;
    size_t yxorder = 2;
    size_t yyorder = 2;
    char *xxterms_str = NULL;
    char *yxterms_str = NULL;
    size_t maxiter = 0;
    double reject = 0.0;

    size_t ninput = 0;
    PyArrayObject *input_array = NULL;
    size_t nref = 0;
    PyArrayObject *ref_array = NULL;
    bbox_t bbox;
    geomap_fit_e fit_geometry = geomap_fit_general;
    surface_type_e surface_type = surface_type_polynomial;
    xterms_e xxterms = xterms_half;
    xterms_e yxterms = xterms_half;

    geomap_result_t fit;
    PyObject *tmp = NULL;
    PyArrayObject *tmp_arr = NULL;
    npy_intp dims = 0;
    int i = 0;
    size_t noutput = 0;
    geomap_output_t *output = NULL;
    PyObject *dtype_list = NULL;
    PyArray_Descr *dtype = NULL;
    PyObject *result = NULL;
    PyArrayObject *output_array = NULL;
    stimage_error_t error;

    const char *keywords[] = {
        "input",   "ref",     "bbox",    "fit_geometry", "function", "xxorder", "xyorder",
        "yxorder", "yyorder", "xxterms", "yxterms",      "maxiter",  "reject",  NULL};

    bbox_init(&bbox);
    geomap_result_init(&fit);
    stimage_error_init(&error);

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO|Ossnnnnssnd:geomap", (char **) keywords, &input_obj, &ref_obj,
            &bbox_obj, &fit_geometry_str, &surface_type_str, &xxorder, &xyorder, &yxorder, &yyorder,
            &xxterms_str, &yxterms_str, &maxiter, &reject)) {
        return NULL;
    }

    input_array = (PyArrayObject *) PyArray_ContiguousFromAny(input_obj, NPY_DOUBLE, 2, 2);
    if (input_array == NULL) {
        goto exit;
    }
    if (PyArray_DIM(input_array, 1) != 2) {
        PyErr_SetString(PyExc_TypeError, "input array must be an Nx2 array");
        goto exit;
    }

    ref_array = (PyArrayObject *) PyArray_ContiguousFromAny(ref_obj, NPY_DOUBLE, 2, 2);
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

    ninput = (size_t) PyArray_DIM(input_array, 0);
    nref = (size_t) PyArray_DIM(ref_array, 0);
    noutput = MAX(ninput, nref);
    output = malloc(noutput * sizeof(geomap_output_t));
    if (output == NULL) {
        result = PyErr_NoMemory();
        goto exit;
    }

    if (geomap(
            ninput, (coord_t *) PyArray_DATA(input_array), nref,
            (coord_t *) PyArray_DATA(ref_array), &bbox, fit_geometry, surface_type, xxorder,
            xyorder, yxorder, yyorder, xxterms, yxterms, maxiter, reject, &noutput, output, &fit,
            &error)) {
        PyErr_SetString(PyExc_RuntimeError, stimage_error_get_message(&error));
        goto exit;
    }

    dtype_list = Py_BuildValue(
        "[(ss)(ss)(ss)(ss)(ss)(ss)(ss)(ss)]", "input_x", "f8", "input_y", "f8", "ref_x", "f8",
        "ref_y", "f8", "fit_x", "f8", "fit_y", "f8", "resid_x", "f8", "resid_y", "f8");
    if (dtype_list == NULL) {
        goto exit;
    }
    if (!PyArray_DescrConverter(dtype_list, &dtype)) {
        goto exit;
    }
    Py_DECREF(dtype_list);
    dims = (npy_intp) noutput;
    output_array = (PyArrayObject *) PyArray_NewFromDescr(
        &PyArray_Type, dtype, 1, &dims, NULL, output, NPY_ARRAY_OWNDATA, NULL);
    if (output_array == NULL) {
        goto exit;
    }

    fit_obj = geomap_new((PyTypeObject *) geomap_class, NULL, NULL);

#define ADD_ATTR(func, member, name)              \
    if ((func) ((member), &tmp))                  \
        goto exit;                                \
    PyObject_SetAttrString(fit_obj, (name), tmp); \
    Py_DECREF(tmp);

#define ADD_ARR_ATTR(func, member, name)                           \
    if ((func) ((member), &tmp_arr))                               \
        goto exit;                                                 \
    PyObject_SetAttrString(fit_obj, (name), (PyObject *) tmp_arr); \
    Py_DECREF(tmp_arr);

#define ADD_ARRAY(size, member, name)                                    \
    dims = (size);                                                       \
    tmp_arr = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_DOUBLE); \
    if (tmp_arr == NULL)                                                 \
        goto exit;                                                       \
    for (i = 0; i < (size); ++i)                                         \
        ((double *) PyArray_DATA(tmp_arr))[i] = (member)[i];             \
    PyObject_SetAttrString(fit_obj, (name), (PyObject *) tmp_arr);       \
    Py_DECREF(tmp_arr);

    ADD_ATTR(from_geomap_fit_e, fit.fit_geometry, "fit_geometry");
    ADD_ATTR(from_surface_type_e, fit.function, "function");
    ADD_ARR_ATTR(from_coord_t, &fit.rms, "rms");
    ADD_ARR_ATTR(from_coord_t, &fit.mean_ref, "mean_ref");
    ADD_ARR_ATTR(from_coord_t, &fit.mean_input, "mean_input");
    ADD_ARR_ATTR(from_coord_t, &fit.shift, "shift");
    ADD_ARR_ATTR(from_coord_t, &fit.mag, "mag");
    ADD_ARR_ATTR(from_coord_t, &fit.rotation, "rotation");
    ADD_ARRAY((int) fit.nxcoeff, fit.xcoeff, "xcoeff");
    ADD_ARRAY((int) fit.nycoeff, fit.ycoeff, "ycoeff");
    ADD_ARRAY((int) fit.nx2coeff, fit.x2coeff, "x2coeff");
    ADD_ARRAY((int) fit.ny2coeff, fit.y2coeff, "y2coeff");

    result = Py_BuildValue("OO", fit_obj, output_array);

exit:
    Py_XDECREF((PyObject *) input_array);
    Py_XDECREF((PyObject *) ref_array);
    geomap_result_free(&fit);
    if (result == NULL) {
        Py_XDECREF((PyObject *) output_array);
        free(output);
        Py_XDECREF(fit_obj);
    }

    return result;
}

int
_setup_geomap_results_type(PyObject *m)
{
    geomap_class = PyType_FromSpec(&geomap_class_spec);
    if (geomap_class == NULL) {
        return -1;
    }
    Py_INCREF(geomap_class);
    if (PyModule_AddObject(m, "GeomapResults", geomap_class) < 0) {
        Py_DECREF(geomap_class);
        Py_CLEAR(geomap_class);
        return -1;
    }
    return 0;
}
