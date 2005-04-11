/* tblot.f -- translated by f2c (version 20031025).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Common Block Declarations */

struct {
    logical verbose;
} verbose_;

#define verbose_1 verbose_

/* Table of constant values */

static integer c__1 = 1;
static integer c__0 = 0;
static integer c__2 = 2;
static integer c__100 = 100;

/* Subroutine */ int tblot_(real *data, real *ndat, integer *xmin, integer *
	xmax, integer *ymin, integer *ymax, integer *dnx, integer *dny, 
	integer *onx, integer *ony, doublereal *xsh, doublereal *ysh, 
	doublereal *drot, doublereal *scale, real *kscale, doublereal *xsh2, 
	doublereal *ysh2, doublereal *xscale, doublereal *yscale, doublereal *
	rot2, char *shfr2, real *pxg, real *pyg, integer *xgdim, integer *
	ygdim, char *align, char *interp, char *coeffs, real *ef, real *
	misval, real *sinscl, integer *clen, integer *vflag, ftnlen shfr2_len,
	 ftnlen align_len, ftnlen interp_len, ftnlen coeffs_len)
{
    /* System generated locals */
    address a__1[2];
    integer data_dim1, data_offset, ndat_dim1, ndat_offset, pxg_dim1, 
	    pxg_offset, pyg_dim1, pyg_offset, i__1[2];
    char ch__1[47];

    /* Builtin functions */
    /* Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen), s_cat(char *,
	     char **, integer *, integer *, ftnlen);
    integer s_cmp(char *, char *, ftnlen, ftnlen);

    /* Local variables */
    static integer idd;
    static doublereal lam, xco[100], yco[100], xin[10000], yin[10000], rot;
    static char vers[45];
    static integer coty;
    static doublereal xout[10000], yout[10000];
    static logical rotf2, disim, incps;
    static integer conum;
    static doublereal wcsin[8];
    static integer istat;
    static char geomod[8];
    extern /* Subroutine */ int getgeo_(char *, integer *, doublereal *, 
	    integer *, integer *, integer *, doublereal *, doublereal *, 
	    integer *, integer *, ftnlen);
    static logical secpar;
    extern /* Subroutine */ int doblot_(real *, real *, integer *, real *, 
	    real *, real *, integer *, integer *, integer *, integer *, 
	    integer *, integer *, logical *, real *, char *, integer *, 
	    integer *, integer *, integer *, doublereal *, doublereal *, 
	    logical *, real *, real *, integer *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, logical *, doublereal *,
	     doublereal *, char *, logical *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, char *, logical *, 
	    ftnlen, ftnlen, ftnlen);
    static logical rotfir;
    static integer intype;
    static logical usewcs;
    static doublereal wcsout[8];
    extern /* Subroutine */ int umsput_(char *, integer *, integer *, integer 
	    *, ftnlen);

/* ++ */

/* TBLOT.F "Reverse drizzling" */
/* IDL-callable version. */

/*   Richard Hook, ST-ECF/STScI, 23rd September 2002 */

/* Added MISVAL parameter - September 2003 */
/* -- */
/* F2PY control codes added by WJH, 22 Nov 2002 */

/* f2py intent(in,c) data */
/* f2py intent(in,out) ndat */
/* f2py real intent(in) :: xsh, ysh, drot, scale */
/* f2py character*8 intent(in) :: align */
/* f2py character*7 intent(in) :: interp */
/* f2py character*80 :: coeffs */
/* f2py real intent(in) :: misval */
/* f2py integer intent(in) :: vflag */
/* f2py integer intent(hide),depend(data) :: dnx = shape(data,0) */
/* f2py integer intent(hide),depend(data) :: dny = shape(data,1) */
/* f2py integer intent(hide),depend(ndat) :: onx = shape(ndat,0) */
/* f2py integer intent(hide),depend(ndat) :: ony = shape(ndat,1) */

/* Local variables */
/* The main data and output arrays */
/* The distortion image arrays, and their size */
/* These arrays should be 2x2 blank arrays if no images are used */
/* Buffers */
/* Geometrical parameters, the standard set */
/* Constants */
/* Logical flags */
/* Verbose common */
/* -- Start of executable code */
/* Keep quiet */
    /* Parameter adjustments */
    data_dim1 = *xmax - *xmin + 1;
    data_offset = 1 + data_dim1;
    data -= data_offset;
    ndat_dim1 = *onx;
    ndat_offset = 1 + ndat_dim1;
    ndat -= ndat_offset;
    pyg_dim1 = *xgdim;
    pyg_offset = 1 + pyg_dim1;
    pyg -= pyg_offset;
    pxg_dim1 = *xgdim;
    pxg_offset = 1 + pxg_dim1;
    pxg -= pxg_offset;

    /* Function Body */
    if (*vflag == 1) {
	verbose_1.verbose = TRUE_;
    } else {
	verbose_1.verbose = FALSE_;
    }
/* First announce the version */
    s_copy(vers, "Callable BLOT Version 0.5 (4th April 2005)", (ftnlen)45, (
	    ftnlen)42);
/* Writing concatenation */
    i__1[0] = 2, a__1[0] = "+ ";
    i__1[1] = 45, a__1[1] = vers;
    s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)47);
    umsput_(ch__1, &c__1, &c__0, &istat, (ftnlen)47);
    usewcs = FALSE_;
/*      ROTFIR=.FALSE. */
    rotfir = TRUE_;
    secpar = FALSE_;
    incps = TRUE_;
/*      EF=1.0 */
/* Convert the rotation to radians */
    rot = *drot * 3.1415926536f / 180.f;
/* Convert the secondary parameters into suitable units */
    *rot2 = *rot2 * 3.1415926536f / 180.;
    if (s_cmp(shfr2, "input", (ftnlen)8, (ftnlen)5) == 0) {
	rotf2 = FALSE_;
    } else {
	rotf2 = TRUE_;
    }
/* Give a warning message if secondary parameters will have an effect */
    if (*xscale != 1. || *yscale != 1. || *xsh2 != 0. || *ysh2 != 0. || *rot2 
	    != 0.) {
	umsput_("! Warning, secondary geometric transform is being used", &
		c__1, &c__0, &istat, (ftnlen)54);
	secpar = TRUE_;
    } else {
	secpar = FALSE_;
    }
/* Check for invalid scale */
    if (*scale == 0.f) {
	umsput_("! Invalid scale", &c__1, &c__0, &istat, (ftnlen)15);
	goto L99;
    }
/* Get the geometric distortion coefficient information */

    getgeo_(coeffs, &idd, &lam, &coty, &c__100, &conum, xco, yco, clen, &
	    istat, (ftnlen)80);
    if (istat != 0) {
	umsput_("! Error, failed to get geometric distortion coefficients", &
		c__1, &c__0, &istat, (ftnlen)56);
	goto L99;
    }
/* Set DISIM logical based on whether distortion images have */
/*  been passed in for use or not. */
    if (*xgdim == 2 && *ygdim == 2) {
	disim = FALSE_;
    } else {
	disim = TRUE_;
    }
/* Before we start also get the interpolation type and convert */
/* to a numerical code */
    if (s_cmp(interp, "nearest", (ftnlen)7, (ftnlen)7) == 0) {
	intype = 1;
    } else if (s_cmp(interp, "linear", (ftnlen)6, (ftnlen)6) == 0) {
	intype = 2;
    } else if (s_cmp(interp, "poly3", (ftnlen)5, (ftnlen)5) == 0) {
	intype = 3;
    } else if (s_cmp(interp, "poly5", (ftnlen)5, (ftnlen)5) == 0) {
	intype = 4;
/*      ELSE IF(INTERP(1:7).EQ.'spline3') THEN */
/*         INTYPE=5 */
    } else if (s_cmp(interp, "sinc", (ftnlen)4, (ftnlen)4) == 0) {
	intype = 6;
    } else if (s_cmp(interp, "lsinc", (ftnlen)5, (ftnlen)5) == 0) {
	intype = 7;
    } else if (s_cmp(interp, "lan3", (ftnlen)4, (ftnlen)4) == 0) {
	intype = 100;
    } else if (s_cmp(interp, "lan5", (ftnlen)4, (ftnlen)4) == 0) {
	intype = 105;
    } else {
	umsput_("! Invalid interpolant specified", &c__1, &c__0, &istat, (
		ftnlen)31);
	umsput_(interp, &c__1, &c__0, &istat, (ftnlen)7);
	goto L99;
    }
/* Now do the actual combination using interpolation */
    doblot_(&data[data_offset], &ndat[ndat_offset], &intype, sinscl, kscale, 
	    misval, xmin, xmax, ymin, ymax, dnx, dny, &rotfir, ef, align, onx,
	     ony, &coty, &conum, xco, yco, &disim, &pxg[pxg_offset], &pyg[
	    pyg_offset], xgdim, ygdim, xin, yin, xout, yout, scale, &rot, xsh,
	     ysh, &usewcs, wcsin, wcsout, geomod, &secpar, xsh2, ysh2, rot2, 
	    xscale, yscale, shfr2, &rotf2, (ftnlen)8, (ftnlen)8, (ftnlen)8);
L99:
    return 0;
} /* tblot_ */

