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

#if 0
// This is only available for python versions 3.12 or later.
void print_exception(int line) {
    PyObject *exc = PyErr_GetRaisedException();
    printf("Error - Line %d: ", line);
    PyErr_DisplayException(exc);
    // PyErr_SetRaisedException(exc);
    return;
}
#endif

/*
 * Check an object to see if it has an attribute.
 */
#define check_attr(obj,name) do { \
    if (!PyObject_HasAttrString(obj, name)) { \
        dbg_print("Attribute '%s' doesn't exist yet.\n", name); \
    } else { \
        dbg_print("Attribute '%s' exists.\n", name); \
    } \
} while(0)

// The following three macros may be better as functions.
/*
 * Add an attribute value to PyObject fit_obj attribute 'name'.
 * 'tmp' is a PyObject*.
 */
#define ADD_ATTR(func, member, name) do { \
    if ((func)((member), &tmp)) {err_print("%s - fail\n", name); goto exit;} \
    PyObject_SetAttrString(fit_obj, (name), tmp); \
    EXCEPTION_INFO; \
    Py_DECREF(tmp); \
} while(0)

/*
 * Add an array object to a PyObject fit_obj attribute 'name'.
 * 'tmp' is a PyObject*.
 */
#define ADD_ARR_ATTR(func, member, name) do { \
    if ((func)((member), &tmp_arr)) {err_print("%s - fail\n", name); goto exit;} \
    PyObject_SetAttrString(fit_obj, (name), tmp); \
    EXCEPTION_INFO; \
    Py_DECREF(tmp_arr); \
} while(0)

/*
 * Add an array values to a PyObject fit_obj array attribute 'name'.
 * 'tmp' is a PyObject*.
 */
#define ADD_ARRAY(size, member, name) do { \
    dims = (size); \
    tmp_arr = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_DOUBLE); \
    if (tmp_arr == NULL) goto exit; \
    for (i = 0; i < (size); ++i) ((double*)PyArray_DATA(tmp_arr))[i] = (member)[i]; \
    PyObject_SetAttrString(fit_obj, (name), (PyObject *) tmp_arr); \
    EXCEPTION_INFO; \
    Py_DECREF(tmp_arr); \
} while(0)

// Debugging macro
#if 0
#define EXCEPTION_INFO do { if (PyErr_Occurred()) { print_exception(__LINE__); } }while(0)
#else
#define EXCEPTION_INFO
#endif

#define CHECK_NULL_PYNONE_JUMP(O, L) do { \
    if (NULL==(O)) { \
        err_print("Object is NULL\n"); \
        goto L; \
    } else if (Py_None==(O)) { \
        err_print("Object is Py_None\n"); \
        goto L; \
    } \
} while(0)


// XXX Flesh out comments for each element.
typedef struct {
    PyObject_HEAD
    PyObject *dict;             // will store __dict__
    PyObject *fit_geometry;
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
    geomap_object *self = NULL;

    // An allocation function must be defined, otherwise a segfault occurs.
    if (type->tp_alloc)
    {
        self = (geomap_object *)type->tp_alloc(type, 0);
    }
    // XXX This is NULL, which means it doesn't exist.  Why is it NULL?
    // dbg_print("fit_geometry = %p\n", self->fit_geometry);
    // check_attr(self, "fit_geometry");

    return (PyObject *)self;
}

static PyArrayObject *
geomap_array_init(void)
{
    PyArrayObject *o = NULL;
    const int nd = 1;   // The number of dimensions of array
    npy_intp dims[nd];  // The number of elements in dimension 1.

    dims[0] = 1; // Must be initialized like this.

    // Create a 1-D array with dimension 'dims' of type double.
    o = (PyArrayObject *) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
    if (o != NULL) {
        *((double*)PyArray_GETPTR1(o, 0)) = 0.0;
    }

    return o;
}

#define GEOMAP_ARRAY_INIT(A) do { \
    (A) = geomap_array_init(); \
    if (NULL==(A)) {return -1;} \
} while(0)

static int
geomap_init(geomap_object *self, PyObject *args, PyObject *kwds)
{
    self->fit_geometry = PyUnicode_FromString("");
    self->function = PyUnicode_FromString("");

    GEOMAP_ARRAY_INIT(self->rms);
    GEOMAP_ARRAY_INIT(self->mean_ref);
    GEOMAP_ARRAY_INIT(self->mean_input);
    GEOMAP_ARRAY_INIT(self->shift);
    GEOMAP_ARRAY_INIT(self->mag);
    GEOMAP_ARRAY_INIT(self->rotation);
    GEOMAP_ARRAY_INIT(self->xcoeff);
    GEOMAP_ARRAY_INIT(self->ycoeff);
    GEOMAP_ARRAY_INIT(self->x2coeff);
    GEOMAP_ARRAY_INIT(self->y2coeff);

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

/* 
 # XXX Not sure what these pragmas do.  I commented this out
 *     and everything compiled just fine.
 */
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wmissing-field-initializers"
// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wcast-function-type-mismatch"
static PyMethodDef geomap_methods[] = {
    {NULL, NULL, 0, NULL}  /* Sentinel */
    // {NULL}  /* Sentinel */
};

// https://docs.python.org/3.12/c-api/structures.html#c.PyMemberDef
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

// https://docs.python.org/3.12/c-api/typeobj.html
static PyTypeObject geomap_class = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "py_geomap.GeomapResults",
    .tp_basicsize = sizeof(geomap_object),
    .tp_dealloc = (destructor)geomap_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "geomap result objects",
    .tp_methods = geomap_methods,
    .tp_members = geomap_members,
    .tp_init = (initproc)geomap_init,
    .tp_alloc = PyType_GenericAlloc,
    .tp_new = geomap_new,
    .tp_dictoffset = offsetof(geomap_object, dict), // <--- REQUIRED
};

/* 
 # XXX Not sure what these pragmas do.  I commented this out
 *     and everything compiled just fine.
 */
// #pragma clang diagnostic pop
// #pragma GCC diagnostic pop

/*
 * XXX This function is way too long and does too many things.
 *     It should be refactored into something smaller.
 */
PyObject*
py_geomap(PyObject* self, PyObject* args, PyObject* kwds)
{
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
    PyArrayObject* input_array  = NULL;
    size_t         nref         = 0;
    PyArrayObject* ref_array    = NULL;
    bbox_t         bbox;
    geomap_fit_e   fit_geometry = geomap_fit_general;
    surface_type_e surface_type = surface_type_polynomial;
    xterms_e       xxterms      = xterms_half;
    xterms_e       yxterms      = xterms_half;

    geomap_result_t  fit;
    PyObject*        tmp          = NULL;
    PyArrayObject*   tmp_arr      = NULL;
    npy_intp         dims         = 0;
    size_t           i            = 0;
    size_t           noutput      = 0;
    geomap_output_t* output       = NULL;
    PyObject*        dtype_list   = NULL;
    PyArray_Descr*   dtype        = NULL;
    PyObject*        result       = NULL;
    PyArrayObject*   output_array = NULL;
    stimage_error_t  error;

    // ---------------------------------------------------------------------------
    // XXX Refactor candidate parse_args_from_python_to_c(python_args, c_args);
    //     The refactor would use a struct, instead of the large number of variables
    //     within this function.
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

    // Create Nx2 array, essentially a list of (x,y) points.
    // XXX Refactor candidate input_array = n_by_2_array(input_obj);
    input_array = (PyArrayObject*)PyArray_ContiguousFromAny(input_obj, NPY_DOUBLE, 2, 2);
    if (input_array == NULL) {
        err_print("input_array creation failed (returned NULL).\n");
        goto exit;
    }
    if (PyArray_DIM(input_array, 1) != 2) {
        err_print("input array must be an Nx2 array\n");
        PyErr_SetString(PyExc_TypeError, "input array must be an Nx2 array");
        goto exit;
    }

    // Create Nx2 array, essentially a list of (x,y) points.
    // XXX Refactor candidate ref_array = n_by_2_array(ref_obj);
    ref_array = (PyArrayObject*)PyArray_ContiguousFromAny(ref_obj, NPY_DOUBLE, 2, 2);
    if (ref_array == NULL) {
        err_print("ref_array creation failed (returned NULL).\n");
        goto exit;
    }
    if (PyArray_DIM(ref_array, 1) != 2) {
        err_print("ref array must be an Nx2 array\n");
        PyErr_SetString(PyExc_TypeError, "ref array must be an Nx2 array");
        goto exit;
    }

    if (to_bbox_t("bbox", bbox_obj, &bbox) ||
        to_geomap_fit_e("fit_geometry", fit_geometry_str, &fit_geometry) ||
        to_surface_type_e("surface_type", surface_type_str, &surface_type) ||
        to_xterms_e("xxterms", xxterms_str, &xxterms) ||
        to_xterms_e("yxterms", yxterms_str, &yxterms))
    {
        err_print("");
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
    // XXX End parse_args_from_python_to_c(python_args, c_args);
    // ---------------------------------------------------------------------------

    // dbg_print("Entering geomap\n");
    if (geomap(
                ninput, (coord_t*)PyArray_DATA(input_array),
                nref, (coord_t*)PyArray_DATA(ref_array),
                &bbox, fit_geometry, surface_type,
                xxorder, xyorder, yxorder, yyorder,
                xxterms, yxterms,
                maxiter, reject,
                &noutput, output, &fit,
                &error))
    {
        PyErr_SetString(PyExc_RuntimeError, stimage_error_get_message(&error));
        goto exit;
    }
    // dbg_print("Returned from geomap\n");

    // -----------------------------------------------------
    // XXX Refactor candidate output_array = get_output_array(output);
    // Develop output array
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
    output_array = (PyArrayObject *) PyArray_NewFromDescr(
            &PyArray_Type, dtype, 1, &dims, NULL, output,
            NPY_ARRAY_OWNDATA, NULL);
    if (output_array == NULL) {
        goto exit;
    }
    // -----------------------------------------------------

    // -----------------------------------------------------
    // XXX Refactor candidate fit_obj = get_geomap_result(fit);
    // Develop GeomapResult class
    fit_obj = geomap_new(&geomap_class, NULL, NULL);
    CHECK_NULL_PYNONE_JUMP(fit_obj, exit);

    // dbg_print("ADD_ATTR\n");
    ADD_ATTR(from_geomap_fit_e, fit.fit_geometry, "fit_geometry");
    ADD_ATTR(from_surface_type_e, fit.function, "function");

    // dbg_print("ADD_ARR_ATTR\n");
    ADD_ARR_ATTR(from_coord_t, &fit.rms, "rms");
    ADD_ARR_ATTR(from_coord_t, &fit.mean_ref, "mean_ref");
    ADD_ARR_ATTR(from_coord_t, &fit.mean_input, "mean_input");
    ADD_ARR_ATTR(from_coord_t, &fit.shift, "shift");
    ADD_ARR_ATTR(from_coord_t, &fit.mag, "mag");
    ADD_ARR_ATTR(from_coord_t, &fit.rotation, "rotation");

    // dbg_print("ADD_ARRAY\n");
    ADD_ARRAY(fit.nxcoeff, fit.xcoeff, "xcoeff");
    ADD_ARRAY(fit.nycoeff, fit.ycoeff, "ycoeff");
    ADD_ARRAY(fit.nx2coeff, fit.x2coeff, "x2coeff");
    ADD_ARRAY(fit.ny2coeff, fit.y2coeff, "y2coeff");
    // -----------------------------------------------------

    // The result value is a tuple of length 2u
    // The first element is a GeomapResults class
    // The second element is a array
    result = Py_BuildValue("OO", fit_obj, output_array);

    // PyErr_Clear();  // XXX Remove.  Here only for debugging.
 exit:

    Py_XDECREF(input_array);
    Py_XDECREF(ref_array);
    geomap_result_free(&fit);
    if (result == NULL) {
        Py_XDECREF(output_array);
        free(output);
        Py_XDECREF(fit_obj);
    }

    // PyErr_Clear(); // XXX for debugging only.  Remove!!!
    // dbg_print("    ----> Returning from python\n");
    return result;
}

static PyModuleDef geomap_module = {
    PyModuleDef_HEAD_INIT,
    "geomap_results",
    "Python object to hold the results of geomap",
    -1,
    geomap_methods
    // NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_geomap_results(void)
{
    PyObject* m;

    // geomap_class.tp_new = PyType_GenericNew;
    if (PyType_Ready(&geomap_class) < 0)
    {
        return NULL;
    }

    m = PyModule_Create(&geomap_module);
    if (m == NULL)
    {
        return NULL;
    }

    Py_INCREF(&geomap_class);
    PyModule_AddObject(m, "GeomapResults", (PyObject *)&geomap_class);
#if 0
    // Possibly like this:
    if (PyModule_AddObject(m, "GeomapResults", (PyObject *)&geomap_class) < 0)
    {
        Py_DECREF(&CustomType);
        Py_DECREF(m);
        return NULL;
    }
#endif
    return m;
}
