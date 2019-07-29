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
#include <structmember.h>

#include "wrap_util.h"
#include "immatch/geomap.h"

typedef struct {
    PyObject_HEAD
    PyObject *fit_geometry;
    PyObject *function;
    PyObject *rms;
    PyObject *mean_ref;
    PyObject *mean_input;
    PyObject *shift;
    PyObject *mag;
    PyObject *rotation;
    PyObject *xcoeff;
    PyObject *ycoeff;
    PyObject *x2coeff;
    PyObject *y2coeff;
} geomap_object;

static PyObject *
geomap_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    geomap_object *self;
    self = (geomap_object *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static PyObject *
geomap_array_init() {
    PyObject *o = NULL;
    npy_intp dims = 1;
    
    o = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
    if (o != NULL) {
        *((double*)PyArray_GETPTR1(o, 0)) = 0.0;
    }

    return o;
}

static int
geomap_init(geomap_object *self, PyObject *args, PyObject *kwds)
{
#if PY_MAJOR_VERSION >= 3
    self->fit_geometry = PyUnicode_FromString("");
    self->function = PyUnicode_FromString("");
#else
    self->fit_geometry = PyString_FromString("");
    self->function = PyString_FromString("");
#endif
    
    self->rms = geomap_array_init();
    if (self->rms == NULL) return -1;
    
    self->mean_ref = geomap_array_init();
    if (self->mean_ref == NULL) return -1;
    
    self->mean_input = geomap_array_init();
    if (self->mean_input == NULL) return -1;
    
    self->shift = geomap_array_init();
    if (self->shift == NULL) return -1;
    
    self->mag = geomap_array_init();
    if (self->mag == NULL) return -1;
    
    self->rotation = geomap_array_init();
    if (self->rotation == NULL) return -1;
    
    self->xcoeff = geomap_array_init();
    if (self->xcoeff == NULL) return -1;
 
    self->ycoeff = geomap_array_init();
    if (self->ycoeff == NULL) return -1;

    self->x2coeff = geomap_array_init();
    if (self->x2coeff == NULL) return -1;
 
    self->y2coeff = geomap_array_init();
    if (self->y2coeff == NULL) return -1;

    return 0;
}

static void
geomap_dealloc(geomap_object *self)
{    
    Py_XDECREF(self->fit_geometry);
    Py_XDECREF(self->function);
    Py_XDECREF(self->rms);
    Py_XDECREF(self->mean_ref);
    Py_XDECREF(self->mean_input);
    Py_XDECREF(self->shift);
    Py_XDECREF(self->mag);
    Py_XDECREF(self->rotation);
    Py_XDECREF(self->xcoeff);
    Py_XDECREF(self->ycoeff);
    Py_XDECREF(self->x2coeff);
    Py_XDECREF(self->y2coeff);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyMethodDef geomap_methods[] = {
    {NULL}  /* Sentinel */
};

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
    {NULL}  /* Sentinel */
};

static PyTypeObject geomap_class = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "py_geomap.GeomapResults", /* tp_name */
    sizeof(geomap_object),     /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)geomap_dealloc,/* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "geomap result objects",   /* tp_doc */
    0,		                   /* tp_traverse */
    0,		                   /* tp_clear */
    0,		                   /* tp_richcompare */
    0,		                   /* tp_weaklistoffset */
    0,		                   /* tp_iter */
    0,		                   /* tp_iternext */
    geomap_methods,            /* tp_methods */
    geomap_members,            /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)geomap_init,     /* tp_init */
    0,                         /* tp_alloc */
    geomap_new,                /* tp_new */
};

PyObject*
py_geomap(PyObject* self, PyObject* args, PyObject* kwds) {
    PyObject* input_obj        = NULL;
    PyObject* ref_obj          = NULL;
    PyObject* bbox_obj         = NULL;
    PyObject* fit_obj          = NULL;
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

    geomap_result_t  fit;
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

    fit_obj = geomap_new(&geomap_class, NULL, NULL);
    
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

#if PY_MAJOR_VERSION >= 3

static PyModuleDef geomap_module = {
    PyModuleDef_HEAD_INIT,
    "geomap_results",
    "Python object to hold the results of geomap",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_geomap_results(void) 
{
    PyObject* m;

    geomap_class.tp_new = PyType_GenericNew;
    if (PyType_Ready(&geomap_class) < 0)
        return NULL;

    m = PyModule_Create(&geomap_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&geomap_class);
    PyModule_AddObject(m, "GeomapResults", (PyObject *)&geomap_class);
    return m;
}

#else
PyMODINIT_FUNC
initgeomap_results(void) 
{
    PyObject* m;

    geomap_class.tp_new = PyType_GenericNew;
    if (PyType_Ready(&geomap_class) < 0)
        return;

    m = Py_InitModule("GeomapResults", geomap_methods);

    Py_INCREF(&geomap_class);
    PyModule_AddObject(m, "GeomapResults", (PyObject *)&geomap_class);
}
#endif
