#include <math.h>
#include <string.h>
#include <Python.h>
#include <arrayobject.h>
#include <libnumarray.h>
/*#include "numarray.h" */
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
    PyArrayObject *img, *wei, *out, *wht, *con, *wcsin;
    PyObject *opxg, *opyg;
    PyArrayObject *pxg, *pyg;
    double xsh, ysh, rot, scale, pfract;
    long xmin, ymin, uniqid, ystart;
    long nmiss, nskip;
    char *align, *kernel;
    char *coeffs, *shiftfr, *shiftun, *inun;
    char vers[80];
    long vflag;
    float istat;
    long nx,ny,onx,ony,dny;
    long xgdim, ygdim;
    float expin, wtscl;
    char *fillstr;
    long align_len, kernel_len, coeffs_len;
    long shiftfr_len, shiftun_len, inun_len;
    long vers_len, fillstr_len;
/*    
    extern float tdriz_(float *, float *, float *, float *, int *,
    int *, int *, int *, int *, int *, int *, int *, int *, int *, 
    double *, double *, char *, char *, double *, double *, 
    float *, float *, int *, int *,
    char *, double *, char *, char *,
    char *, float *, float *, char *, double *, int *, int *, 
    int *, int *, char *,
    int *, int *, int *, int *, int *, int *, int *, int *);
*/
	extern doublereal tdriz_(real *data, real *wei, real *ndat, real *ncou, integer *
	ncon, integer *uniqid, integer *ystart, integer *xmin, integer *ymin, 
	integer *nx, integer *ny, integer *dny, integer *onx, integer *ony, 
	doublereal *xsh, doublereal *ysh, char *shftfr, char *shftun, 
	doublereal *drot, doublereal *scale, real *pxg, real *pyg, integer *
	xgdim, integer *ygdim, char *align, doublereal *pfract, char *kernel, 
	char *coeffs, char *inun, real *expin, real *wtscl, char *filstr, 
	doublereal *wcs, integer *vflag, integer *clen, integer *nmiss, 
	integer *nskip, char *vers, ftnlen shftfr_len, ftnlen shftun_len, 
	ftnlen align_len, ftnlen kernel_len, ftnlen coeffs_len, ftnlen 
	inun_len, ftnlen filstr_len, ftnlen vers_len);
	
    if (!PyArg_ParseTuple(args,"OOOOOlllllddssddOOsdsssffsOlll",
            &oimg,&owei,&oout,&owht,&ocon,&uniqid, &ystart,&xmin,&ymin,&dny,
            &xsh,&ysh, &shiftfr,&shiftun, &rot,&scale, &opxg, &opyg, 
            &align,&pfract, &kernel,&coeffs, &inun, &expin,&wtscl, &fillstr,
            &owcsin, &nmiss, &nskip, &vflag)){
         return PyErr_Format(gl_Error, "arrdriz.tdriz: Invalid Parameters.");
    }
    
    img = (PyArrayObject *)NA_InputArray(oimg, tFloat32, C_ARRAY);
    wei = (PyArrayObject *)NA_InputArray(owei, tFloat32, C_ARRAY);
    out = (PyArrayObject *)NA_IoArray(oout, tFloat32, 0);
    wht = (PyArrayObject *)NA_IoArray(owht, tFloat32, 0);
    con = (PyArrayObject *)NA_IoArray(ocon, tInt32, 0);
    wcsin  = (PyArrayObject *)NA_IoArray(owcsin, tFloat64, 0);
    pxg = (PyArrayObject *)NA_InputArray(opxg, tFloat32, C_ARRAY);    
    pyg = (PyArrayObject *)NA_InputArray(opyg, tFloat32, C_ARRAY);

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
    inun_len = 8;
    vers_len = 47;
    fillstr_len = strlen(fillstr) + 1;
        
    istat = tdriz_(NA_OFFSETDATA(img), NA_OFFSETDATA(wei), 
                    NA_OFFSETDATA(out),NA_OFFSETDATA(wht),
                    NA_OFFSETDATA(con), &uniqid, &ystart,
                    &xmin, &ymin, &nx,&ny, &dny, &onx,&ony, 
                    &xsh,&ysh, shiftfr, shiftun, &rot,&scale,
                    NA_OFFSETDATA(pxg),NA_OFFSETDATA(pyg),&xgdim, &ygdim,
                    align, &pfract, kernel, coeffs, inun, 
                    &expin, &wtscl, fillstr, 
                    NA_OFFSETDATA(wcsin), &vflag, &coeffs_len, 
                    &nmiss, &nskip, vers, 
                    shiftfr_len, shiftun_len, align_len, 
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
}
static PyObject *
tblot(PyObject *obj, PyObject *args)
{

    PyObject *oimg, *oout, *opxg, *opyg;
    PyArrayObject *img, *out, *pxg, *pyg;
    double xsh, ysh, rot, scale;
    long xmin,xmax,ymin,ymax; 
    float ef;
    char *align, *interp;
    char *coeffs;
    long vflag,istat;
    float misval, sinscl, kscale;
    long nx,ny,onx,ony;
    long xgdim, ygdim;
    long align_len, interp_len, coeffs_len;

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
	doublereal *drot, doublereal *scale, real *kscale, real *pxg, real *
	pyg, integer *xgdim, integer *ygdim, char *align, char *interp, char *
	coeffs, real *ef, real *misval, real *sinscl, integer *clen, integer *
	vflag, ftnlen align_len, ftnlen interp_len, ftnlen coeffs_len);
    
	if (!PyArg_ParseTuple(args,"OOllllddddfOOsssfffl",&oimg,&oout,
           &xmin,&xmax,&ymin,&ymax,
           &xsh,&ysh,&rot,&scale,&kscale,&opxg,&opyg,
           &align,&interp,&coeffs,&ef,&misval,
           &sinscl,&vflag)){
         return PyErr_Format(gl_Error, "arrdriz.tblot: Invalid Parameters.");
    }
    
    img = (PyArrayObject *)NA_InputArray(oimg, tFloat32, C_ARRAY);
    pxg = (PyArrayObject *)NA_InputArray(opxg, tFloat32, C_ARRAY);
    pyg = (PyArrayObject *)NA_InputArray(opyg, tFloat32, C_ARRAY);
    out = (PyArrayObject *)NA_IoArray(oout, tFloat32, 0);
    
    nx = img->dimensions[1];
    ny = img->dimensions[0];
    onx = out->dimensions[1];
    ony = out->dimensions[0];
    xgdim = pxg->dimensions[1];
    ygdim = pxg->dimensions[0];
    
    align_len = 8;
    interp_len = 7;
    coeffs_len = strlen(coeffs)+1;
    
    istat = tblot_(NA_OFFSETDATA(img), NA_OFFSETDATA(out), 
                    &xmin,&xmax, &ymin,&ymax,
                    &nx,&ny, &onx,&ony, &xsh,&ysh, &rot,&scale, 
                    &kscale,NA_OFFSETDATA(pxg),NA_OFFSETDATA(pyg),
                    &xgdim,&ygdim,
                    align, interp, coeffs, &ef, &misval, &sinscl, 
                    &coeffs_len, &vflag, 
                    align_len, interp_len, coeffs_len); 
    
    Py_DECREF(img);
    Py_DECREF(pxg);
    Py_DECREF(pyg);
    Py_DECREF(out);
    
    return Py_BuildValue("i",istat);
}

static PyMethodDef arrdriz_methods[] =
{
    {"tdriz",  tdriz, METH_VARARGS, "triz(image, weight, output, outweight, xsh, ysh, rot, scale, align, pfract, kernel, coeffs, vflag)"},
    {"tblot",  tblot, METH_VARARGS, "tblot(image, output, xsh, ysh, rot, scale, align, interp, coeffs, misval, sinscl, vflag)"},
    {0,            0}                             /* sentinel */
};

void initarrdriz() {
    
    Py_InitModule("arrdriz", arrdriz_methods);
    import_libnumarray();
}

