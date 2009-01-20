#include <Python.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <numpy/arrayobject.h>

#include "gc_coords_private.h"

static PyObject * radec2glatlon(PyObject *obj, PyObject *args)
{
    double ra,dec,glat,glon;

    if (!PyArg_ParseTuple(args,"dd:radec2glatlon", &ra, &dec))
	    return NULL;

    glat = 0;
    glon = 0;

    GC_radec2glatlon(&ra, &dec, &glat, &glon);

    return Py_BuildValue("dd",glat,glon);
}

static PyObject * radec2elatlon(PyObject *obj, PyObject *args)
{
    double ra,dec,elat,elon;

    if (!PyArg_ParseTuple(args,"dd:radec2elatlon", &ra, &dec))
	    return NULL;

    elat = 0;
    elon = 0;

    GC_radec2elatlon(&ra, &dec, &elat, &elon);

    return Py_BuildValue("dd",elat,elon);
}

static PyMethodDef GCcoords_methods[] =
{
    {"radec2glatlon", radec2glatlon, METH_VARARGS, 
        "radec2glatlon(ra,dec)"},
    {"radec2elatlon", radec2elatlon, METH_VARARGS, 
        "radec2elatlon(ra,dec)"},
    {0,            0}                             /* sentinel */
};

void initGCcoords(void) {
	Py_InitModule("GCcoords", GCcoords_methods);
    import_array();
}
