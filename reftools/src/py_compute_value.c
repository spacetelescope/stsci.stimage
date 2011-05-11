#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Python.h>

#include "imphttab.h"

/* function prototypes */
static int fill_pht_row(PyObject *py_row, PhtRow *row);
static int fill_phot_par(PyObject *py_par, PhotPar *par);

/* wrapper for compute_value C code copied from imphttab */
static PyObject * py_compute_value(PyObject *self, PyObject *args) {
  /* input variables for PhtRow struct */
  PyObject *py_row;
  
  /* input variables for PhotPar struct */
  PyObject *py_par;
  
  /* local variables */
  int npars;
  PhtRow row;
  PhotPar par;
  double result;
  
  /* temporary holder of python object to check for errors */
  PyObject *temp_obj;
  
  char temp_str[SZ_COLNAME];
  
  /* get input arguments */
  if (!PyArg_ParseTuple(args,"OO", &py_row, &py_par)) {
    return NULL;
  }
  
  /* get how many parameterized variables there are */
  temp_obj = PyDict_GetItemString(py_par, "npar");
  if (!temp_obj) {
    PyErr_SetString(PyExc_KeyError, "Key npar not found in par dict.");
    return NULL;
  }
  npars = (int) PyInt_AsLong(temp_obj);
  
  /* fill PhtRow struct (at least the parts we need) */
  if (fill_pht_row(py_row, &row) != 0) {
    ClosePhotRow(&row);
    return NULL;
  }
  
  /* fill PhotPar struct (at least the parts we need) */
  if (fill_phot_par(py_par, &par) != 0) {
    ClosePhotRow(&row);
    FreePhotPar(&par);
    return NULL;
  }
  
  result = ComputeValue(&row, &par);
  
  /* free memory allocated for PhtRow and PhotPar structs */
  ClosePhotRow(&row);
  FreePhotPar(&par);
  
  return Py_BuildValue("f",result);
}

/* utility for filling PhtRow struct from python dictionary. 
 * copied in part from imphttab/getphttab.c: ReadPhotArray.
 * returns 0 if errors occurr, 1 otherwise */
static int fill_pht_row(PyObject *py_row, PhtRow *row) {
  int status = 0;
  
  /* loop variables */
  int i,n;
  
  /* temporary holder of python object to check for errors */
  PyObject *temp_obj;
  
  /* temporary holders of lists pulled from py_row */
  PyObject *names_list, *nelem_list, *res_list, *vals_list, *vals_list2;
  
  /* temporary holder of strings pulled from py_row */
  char *temp_str;
  
  /* get number of parameterized variables */
  temp_obj = PyDict_GetItemString(py_row, "parnum");
  if (!temp_obj) {
    PyErr_SetString(PyExc_KeyError, "Key parnum not found in row dict.");
    return (status = 1);
  }
  row->parnum = (int) PyInt_AsLong(temp_obj);
  
  /* get number of "results" values */
  temp_obj = PyDict_GetItemString(py_row, "telem");
  if (!temp_obj) {
    PyErr_SetString(PyExc_KeyError, "Key telem not found in row dict.");
    return (status = 1);
  }
  row->telem = (int) PyInt_AsLong(temp_obj);
  
  if (row->parnum == 0){
    row->parvals = (double **) malloc(sizeof(double *));
    row->parnames = (char **) malloc(sizeof(char *));
    row->nelem = (int *) malloc(sizeof(int));
    row->results = (double *) malloc(sizeof(double));
    
    temp_obj = PyDict_GetItemString(py_row, "results");
    if (!temp_obj) {
      PyErr_SetString(PyExc_KeyError, "Key results not found in row dict.");
      return (status = 1);
    }
    row->results[0] = PyFloat_AsDouble(temp_obj);
  } else {
    row->nelem = (int *) malloc((row->parnum+1) * sizeof(int));
    row->parvals = (double **) malloc(row->parnum * sizeof(double *));
    row->parnames = (char **) malloc(row->parnum * sizeof(char *));
    row->results = (double *) malloc(row->telem * sizeof(double));
    
    /* copy names and nelem from dict */
    names_list = PyDict_GetItemString(py_row, "parnames");
    if (!names_list) {
      PyErr_SetString(PyExc_KeyError, "Key parnames not found in row dict.");
      return (status = 1);
    }
    
    nelem_list = PyDict_GetItemString(py_row, "nelem");
    if (!nelem_list) {
      PyErr_SetString(PyExc_KeyError, "Key nelem not found in row dict.");
      return (status = 1);
    }
    
    for (i=0, n=1; i < row->parnum; i++, n++) {
      /* copy parameter name strings */
      row->parnames[i] = (char *) malloc(SZ_COLNAME * sizeof(char));
      temp_obj = PyList_GetItem(names_list, (Py_ssize_t) i);
      if (!temp_obj) {
        return (status = 1);
      }
      temp_str = PyString_AsString(temp_obj);
      if (!temp_str) {
        return (status = 1);
      }
      strcpy(row->parnames[i], temp_str);
      
      /* copy nelem numbers */
      temp_obj = PyList_GetItem(nelem_list, (Py_ssize_t) i);
      if (!temp_obj) {
        return (status = 1);
      }
      row->nelem[n] = (int) PyInt_AsLong(temp_obj);
    } /* end copy of names and nelem*/
    
    /* copy results values */
    res_list = PyDict_GetItemString(py_row, "results");
    if (!res_list) {
      PyErr_SetString(PyExc_KeyError, "Key results not found in row dict.");
      return (status = 1);
    }
    
    for (i = 0; i < row->telem; i++) {
      temp_obj = PyList_GetItem(res_list, (Py_ssize_t) i);
      if (!temp_obj) {
        return (status = 1);
      }
      row->results[i] = PyFloat_AsDouble(temp_obj);
    } /* end copy of results values */
    
    /* copy parameter values */
    vals_list = PyDict_GetItemString(py_row, "parvals");
    if (!vals_list) {
      PyErr_SetString(PyExc_KeyError, "Key parvals not found in row dict.");
      return (status = 1);
    }
    
    for (i = 0; i < row->parnum; i++) {
      row->parvals[i] = (double *) malloc(row->nelem[i+1] * sizeof(double));

      vals_list2 = PyList_GetItem(vals_list, (Py_ssize_t) i);
      if (!vals_list2) {
        return (status = 1);
      }
      
      for (n = 0; n < row->nelem[i+1]; n++) {
        temp_obj = PyList_GetItem(vals_list2, (Py_ssize_t) n);
        if (!temp_obj) {
          return (status = 1);
        }
        row->parvals[i][n] = PyFloat_AsDouble(temp_obj);
      }
    } /* end copy of parameter values */
  } /* end else */
  
  return status;
}

/* utility for filling PhotPar struct from python dictionary.
 * returns 0 if errors occurr, 1 otherwise */
static int fill_phot_par(PyObject *py_par, PhotPar *par) {
  int status = 0;
  
  /* loop variable */
  int i;
  
  /* temporary holder of python object to check for errors */
  PyObject *temp_obj;
  
  /* lists of parameter names and values */
  PyObject *names_list, *vals_list;
  
  /* temporary string holder */
  char *temp_str;
  
  /* initialize phot par. string arguments don't matter, they aren't used
   * in ComputeValue. */
  InitPhotPar(par, "name", "pedigree");
  
  /* get number of parameterized variables for this observation */
  temp_obj = PyDict_GetItemString(py_par, "npar");
  if (!temp_obj) {
    PyErr_SetString(PyExc_KeyError, "Key npar not found in par dict.");
    return (status = 1);
  }
  par->npar = (int) PyInt_AsLong(temp_obj);
  
  /* allocate space in PhotPar struct */
  status = AllocPhotPar(par, par->npar);
  if (status == OUT_OF_MEMORY) {
    PyErr_SetString(PyExc_StandardError, 
                    "An error occured allocating memory for PhotPar struct.");
    return status;
  }
  
  /* copy out parameter names and values */
  names_list = PyDict_GetItemString(py_par, "parnames");
  if (!names_list) {
    PyErr_SetString(PyExc_KeyError, "Key parnames not found in par dict.");
    return (status = 1);
  }
  
  vals_list = PyDict_GetItemString(py_par, "parvals");
  if (!vals_list) {
    PyErr_SetString(PyExc_KeyError, "Key parvals not found in par dict.");
    return (status = 1);
  }
  
  for (i = 0; i < par->npar; i++) {
    temp_obj = PyList_GetItem(names_list, (Py_ssize_t) i);
    if (!temp_obj) {
      return (status = 1);
    }
    temp_str = PyString_AsString(temp_obj);
    if (!temp_str) {
      return (status = 1);
    }
    strcpy(par->parnames[i], temp_str);
    
    temp_obj = PyList_GetItem(vals_list, (Py_ssize_t) i);
    if (!temp_obj) {
      return (status = 1);
    }
    par->parvalues[i] = PyFloat_AsDouble(temp_obj);
  }
  
  return status;
}

static PyMethodDef computephotpars_methods[] = 
{
  {"compute_value", py_compute_value, METH_VARARGS, "Compute photometry parameters."},
  {NULL, NULL, 0, NULL} /* sentinel */
};

PyMODINIT_FUNC init_computephotpars(void) {
  (void) Py_InitModule("_computephotpars", computephotpars_methods);
}
