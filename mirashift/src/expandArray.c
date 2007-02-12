/*
    Program:    expandArray.c
    Author:     Warren J. Hack
    Purpose:    Expand a numarray array by a given factor using bilinear
                interpolation (or other interpolations as implemented)

    Version:
            Version 0.1.0, 14-May-2005: Created -- WJH
*/
#include <Python.h>
#include <arrayobject.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <libnumarray.h>
#include <malloc.h>

float *vector(int nl, int nh){
    float *v;
    
    v = (float *) malloc((unsigned) (nh-nl+1)*sizeof(float));
    return v-nl;
}

void free_vector( float *v, int nl, int nh){
    free ((float *) (v+nl));   
}

float polintF32_(float xa[],float ya[], int n, float x, float *dy){
    int i,m,ns=1;
    float den, dif, dift, ho, hp, w;
    float *c, *d;
    float y;
    
    float *vector(int, int);
    void free_vector(float *, int, int);
    
    dif = fabs(x-xa[1]);
    c = vector(1,n);
    d = vector(1,n);
    for (i = 1; i <= n; i++) {
        if (( dift=fabs(x-xa[i])) < dif) {
            ns = i;
            dif = dift;
        }
        c[i] = ya[i];
        d[i] = ya[i];   
    }
    y = ya[ns--];
    
    for (m=1; m < n; m++){
        for (i=1; i <= n-m; i++){
            ho=xa[i] - x;
            hp = xa[i+m] -x;
            w=c[i+1] - d[i];
            if ( (den=ho-hp) == 0.0) {
                    printf("Error in routine polintF32_\n");
                    exit(1);
            }
            den = w/den;
            d[i] = hp*den;
            c[i] = ho*den;
        }
        y += (*dy = (2*ns < (n-m) ? c[ns+1] : d[ns--]));
          
    }
    
    free_vector(d,1,n);
    free_vector(c,1,n);
    return y;
} 

void resample1DF32_(float *image, int inx, float factor, float *newimage, int order)
{
    int i,n;
    float *xa, *ya;
    float dy;
    int x,o,xm,xo;
    
    float *vector(int, int);
    void free_vector(float *, int, int);
    float polintF32_(float *, float *, int, float, float *);

    if (order < 1) order = 1;
    n = order + 1;
    xa = vector(1,n);
    ya = vector(1,n);
    
    /* Interpolate across the input row */
    for (x = 0; x < inx; x++){
        /* Boundary condition in X */
        if (x < inx-order){
            for (o = 1; o <= n; o++){
                ya[o] = image[x+(o-1)];
                xa[o] = (x+(o-1))*factor;
            }
        }
        xo = (int)(x*factor);
        xm = (int)((x+1)*factor);

        for (i = xo; i < xm; i++){
            newimage[i] = polintF32_(xa,ya,n,i,&dy);
        }
    }
    free_vector(xa,1,n);
    free_vector(ya,1,n);

}


void expandF32_(float *image, int inx, int iny, int factor, 
                    float *newimage, int outnx, int outny, int order)
{
    int i,n;
    float *xa, *ya;
    float dy;
    int x,y,o,xm,ym,xo,yo;
    int indx;
    
    float *vector(int, int);
    void free_vector(float *, int, int);
    float polintF32_(float *, float *, int, float, float *);
    
    if (order < 1) order = 1;
    n = order + 1;
    xa = vector(1,n);
    ya = vector(1,n);
    
    /* Interpolate across each input row ... */
    for (y = 0; y < iny; y++) {
        for (x = 0; x < inx; x++){
            /* Boundary condition in X */
            if (x < inx-order){
                for (o = 1; o <= n; o++){
                    ya[o] = image[x+(o-1)+(y*inx)];
                    xa[o] = (x+(o-1))*factor;
                }
            }
            xo = x*factor;
            yo = y*factor;
            xm = (x+1)*factor;
            for (i = xo; i < xm; i++){
                newimage[i+(yo*outnx)] = polintF32_(xa,ya,n,i,&dy);
            }
        }
    }
    /* Now, interpolate along each output column */
    for (x = 0; x < outnx; x++){
        for (y = 0; y < iny; y++) {
            /* Boundary condition in X */
            if (y < iny-order){
                for (o = 1; o <= n; o++){
                    indx = x+ ((y+(o-1))*outnx*factor);
                    ya[o] = newimage[indx];
                    xa[o] = (y+(o-1))*factor;
                }
            }
            xo = x*factor;
            yo = y*factor;
            ym = (y+1)*factor;
            for (i = yo; i < ym; i++){
                newimage[x+(i*outnx)] = polintF32_(xa,ya,n,i,&dy);
            }
        }
    }
    
    free ((float *) (xa+1));   
    free ((float *) (ya+1));   
}

void collapse_(float *image, short *mask, int outnx, int outny){
    int i,j;
    int zpix0,zpix1;
    int mask_pix;
    
    for (j=0; j<outny; j++){
        zpix0 = 0;
        zpix1 = 0;
        mask_pix = 0;

        for (i = 0; i < outnx; i++){
            if (image[i + j*outnx] != 0.0) {
                mask_pix = 1;
                zpix1++;
            } else if (mask_pix == 0) {
                zpix0++;
                if (zpix0 == outnx -1) {
                    zpix0 = outnx;
                    mask_pix = 1;
                }
            } 
        }/* End loop over each row (x) */

        /* Write out results to mask array for row j*/
        mask[3*j] = zpix0;
        mask[3*j + 1] = zpix1;
        mask[3*j + 2] = outnx - (zpix0 + zpix1) ;

    } /* End loop over rows (y) */           
}
void inflate_(short *mask, char *image, int outnx, int outny) {
    int i,j;
    int start, end;
    
    for (j=0;j<outny;j++){
        start = mask[3*j];
        end = start + mask[1 + 3*j];
        for (i=start; i < end; i++) {
            image[i + j*outnx] = 1.0;
        }
    }   
}
static PyObject * collapseMask(PyObject *obj, PyObject *args)
{
    PyObject *oimage;
    PyArrayObject *image, *mask;
    int outnx, outny;
    
    void collapse_(float* ,short *, int, int);    

    if (!PyArg_ParseTuple(args,"O:collapseMask", &oimage))
	    return NULL;

    image = (PyArrayObject *)NA_InputArray(oimage, tFloat32, C_ARRAY);
    if (!image) return NULL;
        
    /* Compute output size for array */
    outny = image->dimensions[0];
    outnx = image->dimensions[1];
    
    if (!(mask = NA_NewArray(NULL, tInt16, 2, outny, 3)))
      return NULL;    
    
    collapse_(NA_OFFSETDATA(image), NA_OFFSETDATA(mask), outnx, outny);

    Py_DECREF(image);

    return Py_BuildValue("O",mask);
}
static PyObject * inflateMask(PyObject *obj, PyObject *args)
{
    PyObject *omask;
    PyArrayObject *eimage, *mask;
    int outnx, outny;
    Int64 *mask_row = malloc(3*sizeof(Int64));
    int i;
    
    void inflate_(short * ,char *, int, int);    

    if (!PyArg_ParseTuple(args,"O:inflateMask", &omask))
	    return NULL;

    mask = (PyArrayObject *)NA_InputArray(omask, tInt16, C_ARRAY);
    if (!mask) return NULL;    
    
    /* Compute output size for array */
    outny = mask->dimensions[0];
    
    outnx = 0;
    NA_get1D_Int64(mask,NA_get_offset(mask,1,1),3,mask_row);
    for (i=0;i<3;i++) outnx += mask_row[i];
    free(mask_row);
    
    if (!(eimage = NA_NewArray(NULL, tUInt8, 2, outny, outnx)))
      return NULL;    
    
    inflate_(NA_OFFSETDATA(mask), NA_OFFSETDATA(eimage), outnx, outny);
    Py_DECREF(mask);

    return Py_BuildValue("O",eimage);
}

static PyObject * expandArrayF32(PyObject *obj, PyObject *args)
{
    PyObject *oimage;
    PyArrayObject *image, *eimage;
    int inx, iny;
    int outnx, outny;
    int factor, order;
    
    void expandF32_(float *, int, int, int, float *, int, int, int );    

    if (!PyArg_ParseTuple(args,"Ollii:expandArrayF32", &oimage, &factor, &order, &outnx, &outny))
	    return NULL;

    image = (PyArrayObject *)NA_InputArray(oimage, tFloat32, C_ARRAY);
    if (!image) return NULL;
    
    
    /* Compute output size for array */
    iny = image->dimensions[0];
    inx = image->dimensions[1];
    
    if (outnx == 0 && outny == 0){
        outnx = inx * factor;
        outny = iny * factor;
    }    iny = image->dimensions[0];
    if (!(eimage = NA_NewArray(NULL, tFloat32, 2, outny, outnx)))
          return NULL;    
    
    if (factor > 1) {
        expandF32_(NA_OFFSETDATA(image), inx, iny, factor, 
                            NA_OFFSETDATA(eimage), outnx, outny, order);

    } else {
        NA_copyArray( eimage, image); 
    }
    Py_DECREF(image);

    return Py_BuildValue("O",eimage);
}

static PyObject * resample1DF32(PyObject *obj, PyObject *args)
{
    PyObject *oimage;
    PyArrayObject *image, *eimage;
    int outnx;
    int inx;
    float factor;
    int order;
    
    void resample1DF32_(float *, int, float, float *, int );    

    if (!PyArg_ParseTuple(args,"Ofli:resample1DF32", &oimage, &factor, &order, &outnx))
	    return NULL;

    image = (PyArrayObject *)NA_InputArray(oimage, tFloat32, C_ARRAY);
    if (!image) return NULL;
    
    
    /* Compute output size for array */
    inx = image->dimensions[0];
    if (factor == 0 && outnx == 0) 
        return NULL;
    
    if (factor == 0 && outnx > 0) {
        factor = (float)outnx / inx;
    } 
    if (outnx == 0 && factor != 0.){
        outnx = inx * factor;
    }

    if (!(eimage = NA_NewArray(NULL, tFloat32, 1, outnx)))
          return NULL;    

    if (factor != 0.) {
        resample1DF32_(NA_OFFSETDATA(image), inx, factor, 
                            NA_OFFSETDATA(eimage), order);

    } else {
        NA_copyArray( eimage, image); 
    }
    Py_DECREF(image);

    return Py_BuildValue("O",eimage);
}

static PyMethodDef expandArray_methods[] =
{
    {"inflateMask",  inflateMask, METH_VARARGS, 
        "inflateMask(mask)"},
    {"collapseMask", collapseMask, METH_VARARGS, 
        "collapseMask(image)"},
    {"expandArrayF32",  expandArrayF32, METH_VARARGS, 
        "expandArrayF32(image, factor, order, outnx, outny)"},
    {"resample1DF32",  resample1DF32, METH_VARARGS, 
        "resample1DF32(image, factor, order, outnx)"},
    {0,            0}                             /* sentinel */
};

void initexpandArray(void) {
	Py_InitModule("expandArray", expandArray_methods);
	import_libnumarray();
}

