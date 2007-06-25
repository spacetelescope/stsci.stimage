#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <numpy/arrayobject.h>

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

int unbin2d (float *a, float *b, int inx, int iny, int onx, int ony) {

/* arguments:
PyArrayObject *a        i: input data
PyArrayObject *b        o: output data
*/

	float p, q, r, s;	/* for interpolating */
	float xoffset, yoffset;	/* for getting location of output in input */
	float ai, aj;		/* location in input corresponding to m,n */
	float value;		/* interpolated value */
	int binx, biny;		/* number of output pixels per input pixel */
	int m, n;		/* pixel index in output array */
	int i, j;		/* pixel index in input array */

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
			for (m = 0;  m < onx;  m++) {
				b[n*onx+m] = a[n*inx+m];
		     }

	} else if (binx == 1) {
	    /* Interpolate in Y. */

	    /* Science data array. */
	    for (n = 0;  n < ony;  n++) {
		aj = ((float)n - yoffset) / (float)biny;
		InterpInfo (aj, iny, &j, &r, &s);
		for (m = 0;  m < onx;  m++) {
		    value = r * a[j*inx+m] +
			    s * a[(j+1)*inx+m];
		    b[n*onx+m] = value;
		}
	    }

	} else if (biny == 1) {
	    /* Interpolate in X. */

	    /* Science data array. */
	    for (n = 0;  n < ony;  n++) {
		for (m = 0;  m < onx;  m++) {
		    ai = ((float)m - xoffset) / (float)binx;
		    InterpInfo (ai, inx, &i, &p, &q);
		    value = p * a[n*inx+i] +
			    q * a[n*inx+(i+1)];
		    b[n*onx+m] = value;
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
					value = p * r * a[j*inx+i] +
						q * r * a[j*inx+(i+1)] +
						p * s * a[(j+1)*inx+i] +
						q * s * a[(j+1)*inx+(i+1)];
					b[n*onx+m] = value;
			}
		}
	}
	return (1);
}

static PyObject * bilinearinterp(PyObject *obj, PyObject *args)
{
    PyObject *input, *output;
	PyArrayObject *dataa, *datab;
	int inx, iny, onx, ony;
	
    int status=0;

    if (!PyArg_ParseTuple(args,"OO:bilinearinterp",&input,&output))
	    return NULL;
    
    dataa = (PyArrayObject *)PyArray_ContiguousFromAny(input, PyArray_FLOAT, 1, 2);
	datab = (PyArrayObject *)PyArray_ContiguousFromAny(output, PyArray_FLOAT, 1, 2);

	inx = PyArray_DIM(input,0);
	iny = PyArray_DIM(input,1);
	onx = PyArray_DIM(output,0);
	ony = PyArray_DIM(output,1);

    status = unbin2d((float *)dataa->data,(float *)datab->data, inx, iny, onx, ony);

	Py_XDECREF(dataa);
    Py_XDECREF(datab);

    return Py_BuildValue("i",status);
}

static PyMethodDef bilinearinterp_methods[] =
{
    {"bilinearinterp",  bilinearinterp, METH_VARARGS, 
        "bilinearinterp(input, output)"},
    {0,            0}                             /* sentinel */
};

void initbilinearinterp(void) {
    Py_InitModule("bilinearinterp", bilinearinterp_methods);
    import_array();
}



