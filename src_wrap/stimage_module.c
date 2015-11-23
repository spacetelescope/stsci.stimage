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

#include "wrap_util.h"

PyObject* py_xyxymatch(PyObject*, PyObject*, PyObject*);
PyObject* py_geomap(PyObject*, PyObject*, PyObject*);

static PyMethodDef module_methods[] = {
    {"xyxymatch", (PyCFunction)py_xyxymatch, METH_VARARGS | METH_KEYWORDS, NULL},
    {"geomap", (PyCFunction)py_geomap, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
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
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__stimage(void)
#else
init_stimage(void)
#endif
{
    PyObject* m;

    import_array();

    SIZE_T_D = sizeof(size_t) == 8 ? "u8" : "u4";

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
	return m;
#else
    m = Py_InitModule3("_stimage", module_methods,
                       "Example module that creates an extension type.");
	return;
#endif
}
