/*
    Program:    findedges.c
    Author:     Warren J. Hack
    Purpose:    Find the zero-crossing edges from an image

    Version:
            Version 0.1.0, 27-May-2005: Created -- WJH
*/
#include <Python.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>

#include <numpy/arrayobject.h>
#include <numpy/libnumarray.h>

#define PI 3.1415926535897931

void convolve1DF32_(int ksizex, float *kernel, 
     int dsizex, float *data, float *convolved) 
{ 
  int xc; int halfk = ksizex/2;

  for(xc=0; xc<halfk; xc++)
      convolved[xc] = data[xc];
  
  for(xc=halfk; xc<dsizex-halfk; xc++) {
      int xk;
      float temp = 0;
      for (xk=0; xk<ksizex; xk++)
         temp += kernel[xk]*data[xc-halfk+xk];
      convolved[xc] = temp;
  }
  
  for(xc=dsizex-halfk; xc<dsizex; xc++)
     convolved[xc] = data[xc];
  
}

float compute_chain_match_val_ (float *acode, float *bcode, int lena, int lenb, 
        int order, float *kernel, int klen) {

    float val, lratio, ilenb;
    int arrlen;
    float factor;
    float *resampled, *convolved;
    
    extern void resample1DF32_(float *, int, float, float *, int );   
    float compute_chain_match_(float *, float *, int);
    void convolve1DF32_(int, float *, int ,float *, float *); 
    
    lratio = ((float)(lena - lenb) / (float)lenb);
    ilenb = 1.0/(float)lenb;
    factor = (float) lena / lenb;
    if (factor > 1.0) factor = 1./factor;
        
    if (lena > lenb) {
        arrlen = lenb;
    } else {
        arrlen = lena;
    }
    
    if (fabs(lratio) > 0.1) {
        val = 0.0;
        
    } else {
        if (fabs(lratio) > ilenb) {
            resampled = (float *) calloc(arrlen, sizeof(float));
            convolved = (float *) calloc(arrlen, sizeof(float));
            if (lratio < 0) {                
                
                resample1DF32_(bcode, lenb, factor, resampled, order);                
                convolve1DF32_(klen, kernel, lena, resampled, convolved);

                val = compute_chain_match_(acode, convolved, lena);

            } else { 
                     
                resample1DF32_(acode, lena, factor, resampled, order);
                convolve1DF32_(klen, kernel, lenb, resampled, convolved);

                val = compute_chain_match_(bcode, convolved, lenb);
            }
            free (convolved);
            free (resampled);

        } else {
            val = compute_chain_match_(acode, bcode, lena);
        }
    }
    
    return val;     
           
}

void computemomentsmatrix_(float *moments1, float *moments2, int len1, int len2, float *matrix)
{
    int k,i,j;
    int nmoments, maxlen;
    float diff, ratio;
        
    nmoments = 8;
    for (i=0;i<len1;i++) {
        for (j=0;j<len2;j++){
            maxlen = (moments1[7 + i*nmoments] > moments2[7+j*nmoments]) ? moments1[7 + i*nmoments] : moments2[7+j*nmoments];
            ratio = (float)abs(moments1[7 + i*nmoments] - moments2[7+j*nmoments]) / (float)maxlen;                

            if (ratio < 0.35) { 
                diff = 0.0;
                for(k=0;k<7;k++){
                    diff += pow((moments1[k + i*nmoments] - moments2[k + j*nmoments]),2);
                }
                matrix[j + i*len2] = sqrt(diff);
            } else {
                matrix[j + i*len2] = 1.0;
            }
        }
    }
}

void getchaincode_(float *image, int inx, int iny, int npix, char *cpix)
{
    int i,x,y;
    int found_pix, newval;
    int x_start,y_start,x_pix,y_pix;
    char *shift_pix;
    
    /* Find the starting point for the contour ... */
    x_start = 0;
    y_start = 0;
    found_pix = 0;
    for (y = 0; y < iny; y++) {
        for (x = 0; x < inx; x++){
            if (image[x + y*inx] > 0) {
                x_start = x;
                y_start = y;  
                found_pix = 1;
                break;
            }
        }
        if (found_pix == 1) break;
    }
    x_pix = x_start;
    y_pix = y_start;
    
    /* Trace contour from this starting position */
    for (i=0; i < npix; i++) {
            if (i > 0) {
                image[x_pix + y_pix*inx] = 0;  
            }
            if (((x_pix+1)< inx) && (image[(x_pix+1) + (y_pix)*inx] > 0)){
                x_pix += 1;
                cpix[i] = 0;
                continue;
            } else if ((((x_pix+1)<inx) && (y_pix+1)<iny)&& (image[(x_pix+1) + (y_pix+1)*inx] > 0)) {
                x_pix += 1;
                y_pix += 1;
                cpix[i] = 1;
                continue;
           
 } else if ( ((y_pix+1)<iny) && (image[(x_pix) + (y_pix+1)*inx] > 0)) {
                y_pix += 1;
                cpix[i] = 2;
                continue;
            } else if ((((x_pix-1)>= 0) && (y_pix+1)<iny)&& (image[(x_pix-1) + (y_pix+1)*inx] > 0)) {
                x_pix -= 1;
                y_pix += 1;
                cpix[i] = 3;
                continue;
            } else if (((x_pix-1) >= 0) && (image[(x_pix-1) + (y_pix)*inx] > 0)) {
                x_pix -= 1;
                cpix[i] = 4;
                continue;
            } else if ((((x_pix-1) >= 0) && (y_pix-1) >= 0)&& (image[(x_pix-1) + (y_pix-1)*inx] > 0)) {
                x_pix -= 1;
                y_pix -= 1;
                cpix[i] = 5;
                continue;
            } else if (((y_pix-1) >= 0)&& (image[(x_pix) + (y_pix-1)*inx] > 0)) {
                y_pix -= 1;
                cpix[i] = 6;
                continue;
            } else if ((((x_pix+1)<inx) && (y_pix-1)>= 0)){
                x_pix += 1;
                y_pix -= 1;
                cpix[i] = 7;
            }
    } /* Finished extracting raw chain-code */
    /* Now, SHIFT the chain-code to prevent wraparound. */
    shift_pix = (char *)calloc (npix, sizeof(char));
    shift_pix[0] = cpix[0];
    for (i = 1; i < npix; i++){
        newval = (cpix[i] + 8)%8;
        if (abs(shift_pix[i-1] - newval) < abs(shift_pix[i-1] - cpix[i]) ) {
            shift_pix[i] = newval;
        } else {
            shift_pix[i] = cpix[i];
        }         
    }
    
    /* Transfer new values back to cpix array as return values*/
    for (i = 0; i < npix; i++) {
        cpix[i] = shift_pix[i];
    }
    free(shift_pix);
}        

float compute_chain_mean_(float *chain, int clen){

    float cmean;
    int l;
    
    cmean = 0.;
    for (l=0;l<clen;l++) {
        cmean += chain[l];
    }
    cmean /= clen;
    
    return cmean;        
}

float compute_chain_match_(float *code_a, float *code_b, int n) {
    float s_ij, s;
    int l,k, lmodk;
    float sum_a,sum_b, avg_a, avg_b;

    float compute_chain_mean_(float *, int);
    
    sum_a = 0.0;
    sum_b = 0.0;
    /* Initialize return value to -999 since no valid value
        will be less than -1, due to use of cos(). 
    */
    s_ij = -999.0;
    
    /* Compute mean for these chain codes */
    avg_a = compute_chain_mean_(code_a,n);
    avg_b = compute_chain_mean_(code_b,n); 
    
    for (k=0; k< n; k++){
        /* Reset s for next iteration */
        s = 0.0;
        for (l=0; l<n; l++) {
            lmodk = (l+k)%n;
            
            s += cos((PI/4.) * ( (code_a[l]-avg_a) - (code_b[lmodk] - avg_b)));
        }
        s /= n;
        if (s > s_ij) s_ij = s;
    }
    return s_ij;
}


/*

Code for computing various moments of images or x/y positions.

*/
int get_moments_(float *image, int inx, int iny, float xcen, float ycen, float *moment1, float *moment2, float *rmin, float *rmax)
{
    int x,y;
    float sum0,sum1,sum2,sum3;
    double dx, dy, r;
    float rlow, rup;
    int npix;
    
    npix = 0;
    rlow = pow(inx+1,2) + pow(iny+1,2);
    rup = -1;
    /* Find the starting point for the contour ... */
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;

    for (y = 0; y < iny; y++) {
        for (x = 0; x < inx; x++){
            if (image[x + y*inx] > 0) {
                dx = pow(x-xcen,2);
                dy = pow(y-ycen,2);

                r = sqrt( dx + dy);
                if (r > rup) rup = (float)r;
                if (r < rlow) rlow = (float)r;

                sum0 += (dx + dy);
                sum1 += dx;
                sum2 += dy;
                sum3 += (x-xcen)*(y-ycen);
                npix++;
            }
        }
    }

    if (npix > 0.){
        *moment1 = sum0/pow(npix,2.);
        *moment2 = (pow((sum1 - sum2),2)/pow(npix,4)) + (4./pow(npix,4) * pow(sum3,2));
    }
    *rmin = rlow;
    *rmax = rup;
    return npix;    
}


float get_first_moment_(float *image, int inx, int iny, float xcen, float ycen)
{
    int x,y;
    float moment;
    int npix;
    
    npix = 0;
    /* Find the starting point for the contour ... */
    moment = 0.;
    for (y = 0; y < iny; y++) {
        for (x = 0; x < inx; x++){
            if (image[x + y*inx] > 0) {
                moment += (pow(x-xcen,2) + pow(y-ycen,2));
                npix++;
            }
        }
    }

    if (npix > 0.){
        moment /= pow(npix,2.);
    }
    return moment;    
}

float get_second_moment_(float *image, int inx, int iny, float xcen, float ycen)
{
    int x,y;
    float moment;
    int npix;
    float sum1,sum2,sum3;
    
    npix = 0;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    /* Find the starting point for the contour ... */
    moment = 0.;
    for (y = 0; y < iny; y++) {
        for (x = 0; x < inx; x++){
            if (image[x + y*inx] > 0) {
                sum1 += pow(x-xcen,2);
                sum2 += pow(y-ycen,2);
                sum3 += (x-xcen)*(y-ycen);
                npix++;
            }
        }
    }

    if (npix > 0.){
        moment = (pow((sum1 - sum2),2)/pow(npix,4)) + (4./pow(npix,4) * pow(sum3,2));
    }
    return moment;    
}

float get_binary_moment_pq_(float *x, float *y, int nxy, int p, int q)
{
    /* The input position arrays, x and y, should already have been 
        subtracted by the mean x,y positions, and they both need to
        be the same length. 
    */
    int i;
    float moment;
    int npix;
    
    npix = 0;
    moment = 0.;

    for (i = 0; i < nxy; i++){
        moment += (pow(x[i],p) * pow(y[i],q));
    }

    return moment;    
}
    
float get_moment_pq_(float *fimage, float xcen, float ycen, int nx, int ny, int p, int q)
{
    /* The input array, fimage, should already have been masked only
        leaving those pixels with non-zero fluxes that correspond to the
        area inside the detected zero-edge contour. 
        The xcen/ycen value should be computed separately and input.
    */
    int x,y;
    float moment;
    
    moment = 0.;

    for (y = 0; y < ny; y++) {
        for (x = 0; x < nx; x++){
            moment += (pow((x-xcen),p) * pow((y-ycen),q) * fimage[x + y*nx]);
        }
    }

    return moment;    
}

static PyObject * getMoments(PyObject *obj, PyObject *args)
{
    PyObject *oimage;
    PyArrayObject *image;
    int inx, iny;
    float xc, yc;
    float moment1,moment2,rmin,rmax;
    int npix;
    
    int get_moments_(float *, int, int, float, float, float *, float *, float *, float *);

    if (!PyArg_ParseTuple(args,"Off:getFirstMoment", &oimage, &xc, &yc))
	    return NULL;

    image = (PyArrayObject *)NA_InputArray(oimage, tFloat32, C_ARRAY);
    if (!image) return NULL;
    
    
    /* Compute output size for array */
    iny = image->dimensions[0];
    inx = image->dimensions[1];
    
    npix = get_moments_(NA_OFFSETDATA(image), inx, iny, xc, yc, &moment1, &moment2, &rmin, &rmax);

    Py_DECREF(image);

    return Py_BuildValue("ffffi",moment1,moment2,rmin,rmax,npix);
}

static PyObject * getFirstMoment(PyObject *obj, PyObject *args)
{
    PyObject *oimage;
    PyArrayObject *image;
    int inx, iny;
    float xc, yc;
    float moment1;
    
    float get_first_moment_(float *, int, int, float, float);

    if (!PyArg_ParseTuple(args,"Off:getFirstMoment", &oimage, &xc, &yc))
	    return NULL;

    image = (PyArrayObject *)NA_InputArray(oimage, tFloat32, C_ARRAY);
    if (!image) return NULL;
    
    
    /* Compute output size for array */
    iny = image->dimensions[0];
    inx = image->dimensions[1];
    
    moment1 = get_first_moment_(NA_OFFSETDATA(image), inx, iny, xc, yc);

    Py_DECREF(image);

    return Py_BuildValue("f",moment1);
}

static PyObject * getSecondMoment(PyObject *obj, PyObject *args)
{
    PyObject *oimage;
    PyArrayObject *image;
    int inx, iny;
    float xc, yc;
    float moment2;
    
    float get_second_moment_(float *, int, int, float, float);

    if (!PyArg_ParseTuple(args,"Off:getSecondMoment", &oimage, &xc, &yc))
	    return NULL;

    image = (PyArrayObject *)NA_InputArray(oimage, tFloat32, C_ARRAY);
    if (!image) return NULL;
    
    
    /* Compute output size for array */
    iny = image->dimensions[0];
    inx = image->dimensions[1];
    
    moment2 = get_second_moment_(NA_OFFSETDATA(image), inx, iny, xc, yc);

    Py_DECREF(image);

    return Py_BuildValue("f",moment2);
}

static PyObject * getChainCode(PyObject *obj, PyObject *args)
{
    PyObject *oimage;
    PyArrayObject *image, *cpix;
    int npix;
    int inx, iny;
    
    
    void getchaincode_(float *, int, int, int, char *);

    if (!PyArg_ParseTuple(args,"Oi:getChainCode", &oimage, &npix))
	    return NULL;

    image = (PyArrayObject *)NA_InputArray(oimage, tFloat32, C_ARRAY);
    if (!image) return NULL;
    
    
    /* Compute output size for array */
    iny = image->dimensions[0];
    inx = image->dimensions[1];
    
    if (!(cpix = NA_NewArray(NULL, tInt8, 1, npix)))
          return NULL;    

    getchaincode_(NA_OFFSETDATA(image), inx, iny, npix, NA_OFFSETDATA(cpix));

    Py_DECREF(image);

    return Py_BuildValue("O",cpix);
}

static PyObject * compute_binary_moment_pq(PyObject *obj, PyObject *args)
{
    PyObject *oxpos,*oypos;
    PyArrayObject *xpos, *ypos;
    int nxy;
    int p,q;
    float moment_pq;
    
    float get_binary_moment_pq_(float *, float *, int, int, int);

    if (!PyArg_ParseTuple(args,"OOii:compute_moment_pq", &oxpos, &oypos, &p, &q))
	    return NULL;

    xpos = (PyArrayObject *)NA_InputArray(oxpos, tFloat32, C_ARRAY);
    if (!xpos) return NULL;
    ypos = (PyArrayObject *)NA_InputArray(oypos, tFloat32, C_ARRAY);
    if (!ypos) return NULL;
    
    /* Compute output size for array */
    nxy = xpos->dimensions[0];
    
    moment_pq = get_binary_moment_pq_(NA_OFFSETDATA(xpos),NA_OFFSETDATA(ypos), nxy, p,q);

    Py_DECREF(xpos);
    Py_DECREF(ypos);
    
    return Py_BuildValue("f",moment_pq);
}

static PyObject * compute_moment_pq(PyObject *obj, PyObject *args)
{
    PyObject *oimage;
    PyArrayObject *fimage;
    int nx,ny;
    int p,q;
    float xcen,ycen;
    float moment_pq;
    
    float get_moment_pq_(float *, float, float, int, int, int, int);

    if (!PyArg_ParseTuple(args,"Offii:compute_moment_pq", &oimage, &xcen, &ycen, &p, &q))
	    return NULL;

    fimage = (PyArrayObject *)NA_InputArray(oimage, tFloat32, C_ARRAY);
    if (!fimage) return NULL;
    
    /* Compute output size for array */
    ny = fimage->dimensions[0];
    nx = fimage->dimensions[1];
    
    moment_pq = get_moment_pq_(NA_OFFSETDATA(fimage),xcen,ycen, nx,ny, p,q);

    Py_DECREF(fimage);
    
    return Py_BuildValue("f",moment_pq);
}

static PyObject * computeChainMatch(PyObject *obj, PyObject *args)
{
    PyObject *oa,*ob, *okernel;
    PyArrayObject *a, *b, *k;
    float Lccode;
    
    int lena, lenb, klen;
    float match_coeff, lratio;
    int order;
    
    float compute_chain_match_val_ (float *, float *, int , int , int , float *, int); 

    if (!PyArg_ParseTuple(args,"OOlOf:computeChainMatch", &oa, &ob, &order, &okernel, &Lccode))
	    return NULL;

    a = (PyArrayObject *)NA_InputArray(oa, tFloat32, C_ARRAY);
    if (!a) return NULL;
    b = (PyArrayObject *)NA_InputArray(ob, tFloat32, C_ARRAY);
    if (!b) return NULL;
    k = (PyArrayObject *)NA_InputArray(okernel, tFloat32, C_ARRAY);
    if (!k) return NULL;
    
    /* Compute output size for array */
    lena = a->dimensions[0];
    lenb = b->dimensions[0];
    klen = k->dimensions[0];
    
    lratio = (float)((lenb - lena)/lenb);
    
    if (abs(lratio) > Lccode) {
        match_coeff = 0.0;
    } else {
        match_coeff = compute_chain_match_val_(NA_OFFSETDATA(a),NA_OFFSETDATA(b), 
                lena, lenb, order, NA_OFFSETDATA(k), klen);
    }
    Py_DECREF(a);
    Py_DECREF(b);
    Py_DECREF(k);
    
    return Py_BuildValue("f",match_coeff);
}

static PyObject * computeChainMatchCoeff(PyObject *obj, PyObject *args)
{
    PyObject *oa,*ob;
    PyArrayObject *a, *b;
    int nxy;
    float match_coeff;
    
    float compute_chain_match_(float *, float *, int);

    if (!PyArg_ParseTuple(args,"OO:computeChainMatch", &oa, &ob))
	    return NULL;

    a = (PyArrayObject *)NA_InputArray(oa, tFloat32, C_ARRAY);
    if (!a) return NULL;
    b = (PyArrayObject *)NA_InputArray(ob, tFloat32, C_ARRAY);
    if (!b) return NULL;
    
    /* Compute output size for array */
    nxy = a->dimensions[0];
    
    match_coeff = compute_chain_match_(NA_OFFSETDATA(a),NA_OFFSETDATA(b), nxy);

    Py_DECREF(a);
    Py_DECREF(b);
    
    return Py_BuildValue("f",match_coeff);
}

static PyObject * getMomentMatrix(PyObject *obj, PyObject *args)
{
    PyObject *omoments1, *omoments2;
    PyArrayObject *moments1, *moments2, *matrix;
    int len1, len2;
    
    
    void computemomentsmatrix_(float *, float *, int, int, float *);

    if (!PyArg_ParseTuple(args,"OO:getMomentMatrix", &omoments1, &omoments2))
	    return NULL;

    moments1 = (PyArrayObject *)NA_InputArray(omoments1, tFloat32, C_ARRAY);
    if (!moments1) return NULL;
    moments2 = (PyArrayObject *)NA_InputArray(omoments2, tFloat32, C_ARRAY);
    if (!moments2) return NULL;
    
    /* Compute size for output matrix array */
    len1 = moments1->dimensions[0];
    len2 = moments2->dimensions[0];
    
    if (!(matrix = NA_NewArray(NULL, tFloat32, 2, len1, len2)))
          return NULL;    

    computemomentsmatrix_(NA_OFFSETDATA(moments1),NA_OFFSETDATA(moments2), len1,len2, NA_OFFSETDATA(matrix));
    Py_DECREF(moments1);
    Py_DECREF(moments2);

            
    return Py_BuildValue("O",matrix);
}


static PyMethodDef chainMoments_methods[] =
{
    {"getChainCode", getChainCode, METH_VARARGS,
            "getChainCode(image, npix)"},
    {"computeChainMatchCoeff", computeChainMatch, METH_VARARGS,
            "computeChainMatchCoeff(a, b)"},
    {"computeChainMatch", computeChainMatch, METH_VARARGS,
            "computeChainMatch(a, b, order, kernel)"},
    {"getMomentMatrix", getMomentMatrix, METH_VARARGS,
            "getMomentMatrix(moments1, moments2)"},
    {"compute_binary_moment_pq", compute_binary_moment_pq, METH_VARARGS,
            "compute_moment_pq(xpos,ypos,p,q)"},
    {"compute_moment_pq", compute_moment_pq, METH_VARARGS,
            "compute_moment_pq(img,xcen,ycen,p,q)"},
    {"getMoments", getMoments, METH_VARARGS,
            "getMoments(image,xcen,ycen)"},
    {"getFirstMoment", getFirstMoment, METH_VARARGS,
            "getFirstMoment(image,xcen,ycen)"},
    {"getSecondMoment", getSecondMoment, METH_VARARGS,
            "getSecondMoment(image,xcen,ycen)"},
    {0,            0}                             /* sentinel */
};

void initchainMoments(void) {
	Py_InitModule("chainMoments", chainMoments_methods);
	import_libnumarray();
}
