#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <numpy/arrayobject.h>


# define Pix(a,i,j)      a[(j)*(PyArray_DIM(a,0)) + (i)]

/* This routine determines the appropriate array index i and weights
   p and q for linear interpolation.  If the array is called a, and ai
   is the independent variable (in units of the array index), then
   the interpolated value may be computed as:  p * a[i] + q * a[i+1].
*/

static void InterpInfo (float ai, int npts, int *i, float *p, float *q) {

/* arguments:
float ai        i: independent variable in same units as i
int npts        i: size of array within which *i is an index
int *i          o: array index close to ai
float *p, *q    o: weights for linear interpolation
*/

	*i = (int) ai;
	*i = (*i < 0) ? 0 : *i;
	*i = (*i >= npts - 1) ? (npts - 2) : *i;
	*q = ai - *i;
	*p = 1.0F - *q;
}

/* This routine determines which array indexes i1 and i2 to use for
   ORing the data quality information.
*/

int unbin2d (PyObject *a, PyObject *b) {

/* arguments:
PyArrayObject *a        i: input data
PyArrayObject *b        o: output data
*/

	float p, q, r, s;	/* for interpolating */
	float xoffset, yoffset;	/* for getting location of output in input */
	float ai, aj;		/* location in input corresponding to m,n */
	float value;		/* interpolated value */
	int inx, iny;		/* size of input array */
	int onx, ony;		/* size of output array */
	int binx, biny;		/* number of output pixels per input pixel */
	int m, n;		/* pixel index in output array */
	int i, j;		/* pixel index in input array */
    float *dataa;
	float *datab;

	dataa = PyArray_DATA(a);
	datab = PyArray_DATA(b);

	inx = PyArray_DIM(a,0);
	iny = PyArray_DIM(a,1);
	onx = PyArray_DIM(b,0);
	ony = PyArray_DIM(b,1);

	binx = onx / inx;
	biny = ony / iny;
	if (binx * inx != onx || biny * iny != ony) {
	    printf ("ERROR    (unbin2d) bin ratio is not an integer.\n");
	    exit(1);
	}

	xoffset = (float)(binx - 1) / 2.0F;
	yoffset = (float)(biny - 1) / 2.0F;

	if (binx == 1 && biny == 1) {

	    /* Same size, so just copy. */

	    /* Copy the science data. */
	    for (n = 0;  n < ony;  n++)
		for (m = 0;  m < onx;  m++)
		    Pix (datab, m, n) = Pix (dataa, m, n);

	} else if (binx == 1) {

	    /* Interpolate in Y. */

	    /* Science data array. */
	    for (n = 0;  n < ony;  n++) {
		aj = ((float)n - yoffset) / (float)biny;
		InterpInfo (aj, iny, &j, &r, &s);
		for (m = 0;  m < onx;  m++) {
		    value = r * Pix (dataa, m, j) +
			    s * Pix (dataa, m, j+1);
		    Pix (datab, m, n) = value;
		}
	    }

	} else if (biny == 1) {

	    /* Interpolate in X. */

	    /* Science data array. */
	    for (n = 0;  n < ony;  n++) {
		for (m = 0;  m < onx;  m++) {
		    ai = ((float)m - xoffset) / (float)binx;
		    InterpInfo (ai, inx, &i, &p, &q);
		    value = p * Pix (dataa, i, n) +
			    q * Pix (dataa, i+1, n);
		    Pix (datab, m, n) = value;
		}
	    }

	} else {
	    /* Science data array. */
	    for (n = 0;  n < ony;  n++) {
			aj = ((float)n - yoffset) / (float)biny;
			InterpInfo (aj, iny, &j, &r, &s);
			for (m = 0;  m < onx;  m++) {
					ai = ((float)m - xoffset) / (float)binx;
					InterpInfo (ai, inx, &i, &p, &q);
					value = p * r * Pix (dataa, i,   j) +
						q * r * Pix (dataa, i+1, j) +
						p * s * Pix (dataa, i,   j+1) +
						q * s * Pix (dataa, i+1, j+1);
					Pix (datab, m, n) = value;
			}
		}
	}

	return (0);
}

static PyObject * bilininterp(PyObject *obj, PyObject *args)
{
    PyObject *oinput, *ooutput;
    PyArrayObject *input, *output;
    int status=0;

    if (!PyArg_ParseTuple(args,"OO:bilininterp",&oinput,&ooutput))
	    return NULL;

    input = (PyArrayObject *)PyArray_ContiguousFromObject(oinput, PyArray_FLOAT32, 1, 2);

    if (!input) return NULL;

	output = (PyArrayObject *)PyArray_ContiguousFromObject(ooutput, PyArray_FLOAT32, 1, 2);

    if (!output) return NULL;
    
    status = unbin2d((PyObject *)input,(PyObject *)output);

    Py_XDECREF(input);
    Py_XDECREF(output);

    return Py_BuildValue("i",status);
}

static PyMethodDef bilininterp_methods[] =
{
    {"bilininterp",  bilininterp, METH_VARARGS, 
        "bilininterp(input, output)"},
    {0,            0}                             /* sentinel */
};

void initbilininterp(void) {
    Py_InitModule("bilininterp", bilininterp_methods);
    import_array();
}



