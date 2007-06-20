
int main(void) 
{
    return 0;
    
}

/* This file contains the following:
	unbin2d
	InterpInfo
	InterpDQInfo
*/

# include <stdio.h>
# include <stdlib.h>		/* calloc */
# include <string.h>		/* strncmp */
# include <math.h>		/* sqrt */

# include <c_iraf.h>
# include <hstio.h>

# include "../stis.h"
# include "../stiserr.h"
# include "../stisdef.h"

static void InterpInfo (float, int, int *, float *, float *);
static void InterpDQInfo (float, int, int *, int *, int *);

/* This routine takes an input data array and expands it by linear
   interpolation, writing to the output array.  The calling routine
   must allocate the output SingleGroup (setting its size) and free
   it when done.

   The coordinate keywords in the output extension headers will be updated.

   Note that, in contrast to bin2d, this routine does not include
   any offset (i.e. xcorner, ycorner) in either input or output.

   Note that the errors are *not* increased by the factor
   sqrt (binx * biny), which would be necessary in order to make
   this routine the true inverse of bin2d.

   Phil Hodge, 1998 Oct 5:
	Change status values to GENERIC_ERROR_CODE or HEADER_PROBLEM.
*/

/* The computation of errors is not what one would normally do for
   linear interpolation, but it's reasonable in this context, which is
   that unbin2d should be the inverse of bin2d (except for the factor
   sqrt (binx*biny) mentioned above).
*/

int unbin2d (SingleGroup *a, SingleGroup *b) {

/* arguments:
SingleGroup *a        i: input data
SingleGroup *b        o: output data
*/

	int status;

	double block[2];	/* number of input pixels for one output */
	double offset[2] = {0., 0.};	/* offset of binned image */
	float p, q, r, s;	/* for interpolating */
	float xoffset, yoffset;	/* for getting location of output in input */
	float ai, aj;		/* location in input corresponding to m,n */
	float value;		/* interpolated value */
	int inx, iny;		/* size of input array */
	int onx, ony;		/* size of output array */
	int binx, biny;		/* number of output pixels per input pixel */
	int m, n;		/* pixel index in output array */
	int i, j;		/* pixel index in input array */

	inx = a->sci.data.nx;
	iny = a->sci.data.ny;
	onx = b->sci.data.nx;
	ony = b->sci.data.ny;

	binx = onx / inx;
	biny = ony / iny;
	if (binx * inx != onx || biny * iny != ony) {
	    printf ("ERROR    (unbin2d) bin ratio is not an integer.\n");
	    return (GENERIC_ERROR_CODE);
	}

	xoffset = (float)(binx - 1) / 2.0F;
	yoffset = (float)(biny - 1) / 2.0F;

	if (binx == 1 && biny == 1) {

	    /* Same size, so just copy. */

	    /* Copy the science data. */
	    for (n = 0;  n < ony;  n++)
		for (m = 0;  m < onx;  m++)
		    Pix (b->sci.data, m, n) = Pix (a->sci.data, m, n);

	} else if (binx == 1) {

	    /* Interpolate in Y. */

	    /* Science data array. */
	    for (n = 0;  n < ony;  n++) {
		aj = ((float)n - yoffset) / (float)biny;
		InterpInfo (aj, iny, &j, &r, &s);
		for (m = 0;  m < onx;  m++) {
		    value = r * Pix (a->sci.data, m, j) +
			    s * Pix (a->sci.data, m, j+1);
		    Pix (b->sci.data, m, n) = value;
		}
	    }

	} else if (biny == 1) {

	    /* Interpolate in X. */

	    /* Science data array. */
	    for (n = 0;  n < ony;  n++) {
		for (m = 0;  m < onx;  m++) {
		    ai = ((float)m - xoffset) / (float)binx;
		    InterpInfo (ai, inx, &i, &p, &q);
		    value = p * Pix (a->sci.data, i, n) +
			    q * Pix (a->sci.data, i+1, n);
		    Pix (b->sci.data, m, n) = value;
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
					value = p * r * Pix (a->sci.data, i,   j) +
						q * r * Pix (a->sci.data, i+1, j) +
						p * s * Pix (a->sci.data, i,   j+1) +
						q * s * Pix (a->sci.data, i+1, j+1);
					Pix (b->sci.data, m, n) = value;
			}
		}
	}

	return (0);
}








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

