#include <math.h>
#include <string.h>
#include <Python.h>

#include <numpy/arrayobject.h>

#include <numpy/libnumarray.h>

#include "f2c.h"

static PyObject *gl_Error;
 
static PyObject *ErrorReturn(char *message)
{
  PyErr_SetString(gl_Error,message);
  return NULL;
}

static PyObject *
tdriz(PyObject *obj, PyObject *args)
{

    PyObject *oimg, *owei, *oout, *owht, *owcsin, *ocon;
    PyArrayObject *img=NULL, *wei=NULL, *out=NULL, *wht=NULL, *con=NULL, 
*wcsin=NULL;
    PyObject *opxg, *opyg;
    PyArrayObject *pxg=NULL, *pyg=NULL;
    double xsh, ysh, rot, scale, pfract;
    double xsh2, ysh2, rot2, xscale, yscale;
    double expin;
    char *shfr2;
    int xmin, ymin, uniqid, ystart;
    long nmiss, nskip;
    char *align, *kernel;
    char *coeffs, *shiftfr, *shiftun, *inun;
    char vers[80];
    int vflag;
    float istat;
    int nx,ny,onx,ony,dny;
    int xgdim, ygdim;
    float wtscl;
    char *fillstr;
    long align_len, kernel_len, coeffs_len;
    long shiftfr_len, shiftun_len, inun_len;
    long vers_len, fillstr_len;
    long shfr2_len;
    double alpha, beta;

    
	extern doublereal tdriz_(real *data, real *wei, real *ndat, real *ncou, integer *
	ncon, integer *uniqid, integer *ystart, integer *xmin, integer *ymin, 
	integer *nx, integer *ny, integer *dny, integer *onx, integer *ony, 
	doublereal *xsh, doublereal *ysh, char *shftfr, char *shftun, 
	doublereal *drot, doublereal *scale,
    doublereal *xsh2,doublereal *ysh2, doublereal *xscale,doublereal *yscale,
    doublereal *rot2, char *shfr2, real *pxg, real *pyg, integer *xgdim, 
    integer *ygdim, char *align, doublereal *pfract, char *kernel, 
	char *coeffs, char *inun, doublereal *expin, real *wtscl, char *filstr, 
	doublereal *wcs, integer *vflag, integer *clen, integer *nmiss, 
	integer *nskip, char *vers, doublereal *alpha, doublereal *beta,
	ftnlen shftfr_len, ftnlen shftun_len,
    ftnlen shfr2_len, ftnlen align_len, ftnlen kernel_len, 
    ftnlen coeffs_len, ftnlen inun_len, ftnlen filstr_len, ftnlen vers_len);
	
    if (!PyArg_ParseTuple(args,"OOOOOlllllddssdddddddsOOsdsssffsOllldd",
            &oimg,&owei,&oout,&owht,&ocon,&uniqid, &ystart,&xmin,&ymin,&dny,
            &xsh,&ysh, &shiftfr,&shiftun, &rot,&scale, 
            &xsh2, &ysh2, &xscale, &yscale, &rot2, &shfr2, &opxg, &opyg, 
            &align,&pfract, &kernel,&coeffs, &inun, &expin,&wtscl, &fillstr,
            &owcsin, &nmiss, &nskip, &vflag, &alpha, &beta)){
         return PyErr_Format(gl_Error, "arrdriz.tdriz: Invalid Parameters.");
    }
    
    img = (PyArrayObject *)NA_InputArray(oimg, tFloat32, C_ARRAY);
    if (!img) goto _exit;

    wei = (PyArrayObject *)NA_InputArray(owei, tFloat32, C_ARRAY);
    if (!wei) goto _exit;
    out = (PyArrayObject *)NA_IoArray(oout, tFloat32, 0);
    if (!out) goto _exit;
    wht = (PyArrayObject *)NA_IoArray(owht, tFloat32, 0);
    if (!wht) goto _exit;
    con = (PyArrayObject *)NA_IoArray(ocon, tInt32, 0);
    if (!con) goto _exit;
    wcsin  = (PyArrayObject *)NA_IoArray(owcsin, tFloat64, 0);
    if (!wcsin) goto _exit;
    pxg = (PyArrayObject *)NA_InputArray(opxg, tFloat32, C_ARRAY);    
    if (!pxg) goto _exit;
    pyg = (PyArrayObject *)NA_InputArray(opyg, tFloat32, C_ARRAY);
    if (!pyg) goto _exit;


                                                                   
      

    nx = img->dimensions[1];
    ny = img->dimensions[0];
    onx = out->dimensions[1];
    ony = out->dimensions[0];   
    xgdim = pxg->dimensions[1];
    ygdim = pxg->dimensions[0];
    
    align_len = 8;
    kernel_len = 8;
    coeffs_len = strlen(coeffs)+1;
    shiftfr_len = 8;
    shiftun_len = 8;
    shfr2_len = 8;
    inun_len = 8;
    vers_len = 44;
    fillstr_len = strlen(fillstr) + 1;
        
    istat = tdriz_(NA_OFFSETDATA(img), NA_OFFSETDATA(wei), 
                    NA_OFFSETDATA(out),NA_OFFSETDATA(wht),
                    NA_OFFSETDATA(con), &uniqid, &ystart,
                    &xmin, &ymin, &nx,&ny, &dny, &onx,&ony, 
                    &xsh,&ysh, shiftfr, shiftun, &rot,&scale,
                    &xsh2,&ysh2, &xscale,&yscale, &rot2, shfr2,                    
                    NA_OFFSETDATA(pxg),NA_OFFSETDATA(pyg),&xgdim, &ygdim,
                    align, &pfract, kernel, coeffs, inun, 
                    &expin, &wtscl, fillstr, 
                    NA_OFFSETDATA(wcsin), &vflag, &coeffs_len, 
                    &nmiss, &nskip, vers,&alpha,&beta, 
                    shiftfr_len, shiftun_len, shfr2_len, align_len, 
                    kernel_len, coeffs_len, inun_len, 
                    fillstr_len, vers_len); 
    
    Py_DECREF(img);
    Py_DECREF(wei);
    Py_DECREF(out);
    Py_DECREF(wht);

    Py_DECREF(con);
    Py_DECREF(wcsin);
    Py_DECREF(pxg);
    Py_DECREF(pyg);

    return Py_BuildValue("s#ii",vers,vers_len,nmiss,nskip);
/*    return Py_BuildValue("f",istat); */

 _exit:
    Py_XDECREF(img);
    Py_XDECREF(wei);
    Py_XDECREF(out);
    Py_XDECREF(wht);
    Py_XDECREF(con);
    Py_XDECREF(wcsin);
    Py_XDECREF(pxg);
    return NULL;             
}

/*
static PyObject *
twdriz(PyObject *obj, PyObject *args)
{

    PyObject *oimg, *owei, *oout, *owht, *owcsin, *owcsout;
    PyArrayObject *img, *wei, *out, *wht, *wcsin, *wcsout;
    PyObject *opxg, *opyg;
    PyArrayObject *pxg, *pyg;
    double pfract;
    long xmin, ymin, ystart;
    long nmiss, nskip;
    char *kernel;
    char *coeffs;
    char vers[80];
    long vflag;
    float istat;
    long nx,ny,onx,ony,dny;
    long xgdim, ygdim;
    char *fillstr;
    long kernel_len, coeffs_len;
    long vers_len, fillstr_len;

	extern doublereal twdriz_(real *data, real *wei, real *ndat, real *ncou,
    integer *ystart, integer *nx, integer *ny, integer *dny, integer *onx, integer *ony, 
	doublereal *wcs, doublereal *wcsout, real *pxg, real *pyg, integer *xgdim, 
    integer *ygdim, doublereal *pfract, char *kernel, 
	char *coeffs, char *filstr, integer *vflag, integer *clen, integer *nmiss, 
	integer *nskip, char *vers, ftnlen kernel_len, 
    ftnlen coeffs_len, ftnlen filstr_len, ftnlen vers_len);
	
    if (!PyArg_ParseTuple(args,"OOOOllllOOOOdssslll",
            &oimg,&owei,&oout,&owht, &ystart,&xmin,&ymin,&dny, &owcsin, &owcsout,
            &opxg, &opyg, &pfract, &kernel,&coeffs, &fillstr,
            &nmiss, &nskip, &vflag)){
         return PyErr_Format(gl_Error, "arrdriz.twdriz: Invalid Parameters.");
    }
    
    img = (PyArrayObject *)NA_InputArray(oimg, tFloat32, C_ARRAY);
    wei = (PyArrayObject *)NA_InputArray(owei, tFloat32, C_ARRAY);
    out = (PyArrayObject *)NA_IoArray(oout, tFloat32, 0);
    wht = (PyArrayObject *)NA_IoArray(owht, tFloat32, 0);

    wcsin   = (PyArrayObject *)NA_IoArray(owcsin, tFloat64, 0);
    wcsout  = (PyArrayObject *)NA_IoArray(owcsin, tFloat64, 0);
    pxg = (PyArrayObject *)NA_InputArray(opxg, tFloat32, C_ARRAY);    
    pyg = (PyArrayObject *)NA_InputArray(opyg, tFloat32, C_ARRAY);

    nx = img->dimensions[1];
    ny = img->dimensions[0];
    onx = out->dimensions[1];
    ony = out->dimensions[0];   
    xgdim = pxg->dimensions[1];
    ygdim = pxg->dimensions[0];
    
    kernel_len = 8;
    coeffs_len = strlen(coeffs)+1;
    vers_len = 44;
    fillstr_len = strlen(fillstr) + 1;
        
    istat = twdriz_(NA_OFFSETDATA(img), NA_OFFSETDATA(wei), 
                    NA_OFFSETDATA(out),NA_OFFSETDATA(wht),
                    &ystart,&nx,&ny, &dny, &onx,&ony,
                    NA_OFFSETDATA(wcsin), NA_OFFSETDATA(wcsout),  
                    NA_OFFSETDATA(pxg),NA_OFFSETDATA(pyg),&xgdim, &ygdim,
                    &pfract, kernel, coeffs, fillstr, 
                    &vflag, &coeffs_len, 
                    &nmiss, &nskip, vers, 
                    kernel_len, coeffs_len, 
                    fillstr_len, vers_len); 
    
    Py_DECREF(img);
    Py_DECREF(wei);
    Py_DECREF(out);
    Py_DECREF(wht);

    Py_DECREF(wcsout);
    Py_DECREF(wcsin);
    Py_DECREF(pxg);
    Py_DECREF(pyg);

    return Py_BuildValue("s#ii",vers,vers_len,nmiss,nskip);
}*/


static PyObject *
tblot(PyObject *obj, PyObject *args)
{

    PyObject *oimg, *oout, *opxg, *opyg;
    PyArrayObject *img=NULL, *out=NULL, *pxg=NULL, *pyg=NULL;
    double xsh, ysh, rot, scale;
    double xsh2, ysh2, rot2, xscale, yscale;
    char *shfr2;
    int xmin,xmax,ymin,ymax; 
    float ef;
    char *align, *interp;
    char *coeffs;
    int vflag,istat;
    float misval, sinscl, kscale;
    int nx,ny,onx,ony;
    int xgdim, ygdim;
    long align_len, interp_len, coeffs_len;
    long shfr2_len;
    double alpha,beta;
    /*
    extern float tblot_(float *, float *, int *, int *, int *, int *, 
    int *, int *, int *, int *, 
    double *, double *, double *, double *, float *,
    float *, float*, int *, int*, 
    char *, char *, char *, float *,
    float *, float *, int *, int *, int *, int *, int *);
	*/
	extern int tblot_(real *data, real *ndat, integer *xmin, integer *
	xmax, integer *ymin, integer *ymax, integer *dnx, integer *dny, 
	integer *onx, integer *ony, doublereal *xsh, doublereal *ysh, 
	doublereal *drot, doublereal *scale, real *kscale, 
    doublereal *xsh2, doublereal *ysh2, 
	doublereal *xscale, doublereal *yscale, doublereal *rot2, char *shfr2,  
    real *pxg, real *pyg, 
    integer *xgdim, integer *ygdim, char *align, char *interp, char *
	coeffs, real *ef, real *misval, real *sinscl, integer *clen, integer *
	vflag, doublereal *alpha, doublereal *beta,
    ftnlen shfr2_len, ftnlen align_len, ftnlen interp_len, ftnlen coeffs_len);
    
	if (!PyArg_ParseTuple(args,"OOllllddddfdddddsOOsssfffldd",&oimg,&oout,
           &xmin,&xmax,&ymin,&ymax,
           &xsh,&ysh,&rot,&scale,&kscale,
           &xsh2,&ysh2,&xscale,&yscale,&rot2,&shfr2,
           &opxg,&opyg, &align,&interp,&coeffs,&ef,&misval,
           &sinscl,&vflag,&alpha,&beta)){
         return PyErr_Format(gl_Error, "arrdriz.tblot: Invalid Parameters.");
    }
    
    img = (PyArrayObject *)NA_InputArray(oimg, tFloat32, C_ARRAY);
    if (!img) goto _exit;
    pxg = (PyArrayObject *)NA_InputArray(opxg, tFloat32, C_ARRAY);
    if (!pxg) goto _exit;
    pyg = (PyArrayObject *)NA_InputArray(opyg, tFloat32, C_ARRAY);
    if (!pyg) goto _exit;
    out = (PyArrayObject *)NA_IoArray(oout, tFloat32, 0);
    if (!out) goto _exit;



    nx = img->dimensions[1];
    ny = img->dimensions[0];
    onx = out->dimensions[1];
    ony = out->dimensions[0];
    xgdim = pxg->dimensions[1];
    ygdim = pxg->dimensions[0];
    
    shfr2_len = 8;
    align_len = 8;
    interp_len = 7;
    coeffs_len = strlen(coeffs)+1;
    
    istat = tblot_(NA_OFFSETDATA(img), NA_OFFSETDATA(out), 
                    &xmin,&xmax, &ymin,&ymax,
                    &nx,&ny, &onx,&ony, &xsh,&ysh, &rot,&scale, 
                    &kscale, &xsh2,&ysh2,&xscale,&yscale,&rot2, shfr2,
                    NA_OFFSETDATA(pxg),NA_OFFSETDATA(pyg),
                    &xgdim,&ygdim,
                    align, interp, coeffs, &ef, &misval, &sinscl, 
                    &coeffs_len, &vflag, &alpha,&beta,shfr2_len,
                    align_len, interp_len, coeffs_len); 
    
    Py_DECREF(img);
    Py_DECREF(pxg);
    Py_DECREF(pyg);
    Py_DECREF(out);
    
    return Py_BuildValue("i",istat);
 _exit:
    Py_XDECREF(img);
    Py_XDECREF(pxg);
    Py_XDECREF(pyg);
    Py_XDECREF(out);
    return NULL;
	
}

static PyMethodDef arrdriz_methods[] =
{
    {"tdriz",  tdriz, METH_VARARGS, "triz(image, weight, output, outweight, xsh, ysh, rot, scale, align, pfract, kernel, coeffs, vflag,alpha,beta)"},
    /*{"twdriz",  tdriz, METH_VARARGS, "triz(image, weight, output, outweight, ystart, xmin, ymin, dny, wcsin, wcsout,pxg,pyg,pfract, kernel, coeffs, fillstr,nmiss,nskip,vflag)"},*/
    {"tblot",  tblot, METH_VARARGS, "tblot(image, output, xsh, ysh, rot, scale, align, interp, coeffs, misval, sinscl, vflag,alpha,beta)"},
    {0,            0}                             /* sentinel */
};

void initarrdriz() {
    
    Py_InitModule("arrdriz", arrdriz_methods);
    import_libnumarray();
}

