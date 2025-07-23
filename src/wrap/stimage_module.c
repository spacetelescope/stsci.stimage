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
*/

#include "wrap_util.h"

PyObject* py_xyxymatch(PyObject*, PyObject*, PyObject*);
PyObject* py_geomap(PyObject*, PyObject*, PyObject*);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-function-type-mismatch"
static PyMethodDef module_methods[] = {
    {"xyxymatch", (PyCFunction)py_xyxymatch, METH_VARARGS | METH_KEYWORDS, NULL},
    {"geomap", (PyCFunction)py_geomap, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL}  /* Sentinel */
};
#pragma clang diagnostic pop
#pragma GCC diagnostic pop

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_stimage",          /* m_name */
  "Example module that creates an extension type.",  /* m_doc */
  -1,                  /* m_size */
  module_methods,      /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};

PyMODINIT_FUNC
PyInit__stimage(void)
{
    PyObject* m;

    SIZE_T_D = sizeof(size_t) == 8 ? "u8" : "u4";
    m = PyModule_Create(&moduledef);

    import_array();

    /* Check for errors */
    if (PyErr_Occurred()) {
      Py_FatalError("can't initialize module cdrizzle");
    }

    if (m != NULL && _setup_geomap_results_type(m) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
  }


