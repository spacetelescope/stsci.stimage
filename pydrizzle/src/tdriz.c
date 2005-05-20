/* tdriz.f -- translated by f2c (version 20031025).
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

extern struct {
    logical verbose;
} verbose_;

#define verbose_1 verbose_

/* Table of constant values */

static integer c__1 = 1;
static integer c__0 = 0;
static integer c__100 = 100;
static integer c__4 = 4;

doublereal tdriz_(real *data, real *wei, real *ndat, real *ncou, integer *
	ncon, integer *uniqid, integer *ystart, integer *xmin, integer *ymin, 
	integer *nx, integer *ny, integer *dny, integer *onx, integer *ony, 
	doublereal *xsh, doublereal *ysh, char *shftfr, char *shftun, 
	doublereal *drot, doublereal *scale, doublereal *xsh2, doublereal *
	ysh2, doublereal *xscale, doublereal *yscale, doublereal *rot2, char *
	shfr2, real *pxg, real *pyg, integer *xgdim, integer *ygdim, char *
	align, doublereal *pfract, char *kernel, char *coeffs, char *inun, 
	real *expin, real *wtscl, char *filstr, doublereal *wcs, integer *
	vflag, integer *clen, integer *nmiss, integer *nskip, char *vers, 
	ftnlen shftfr_len, ftnlen shftun_len, ftnlen shfr2_len, ftnlen 
	align_len, ftnlen kernel_len, ftnlen coeffs_len, ftnlen inun_len, 
	ftnlen filstr_len, ftnlen vers_len)
{
    /* System generated locals */
    integer data_dim1, data_offset, wei_dim1, wei_offset, ndat_dim1, 
	    ndat_offset, ncou_dim1, ncou_offset, ncon_dim1, ncon_offset, 
	    pxg_dim1, pxg_offset, pyg_dim1, pyg_offset, i__1, i__2;
    real ret_val;
    icilist ici__1;

    /* Builtin functions */
    /* Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen);
    integer s_cmp(char *, char *, ftnlen, ftnlen), s_rsli(icilist *), do_lio(
	    integer *, integer *, char *, ftnlen), e_rsli(void);

    /* Local variables */
    static doublereal xi[400000]	/* was [100000][4] */, yi[400000]	
	    /* was [100000][4] */, xo[400000]	/* was [100000][4] */, yo[
	    400000]	/* was [100000][4] */;
    static integer idd;
    static doublereal lam;
    static logical con;
    static integer nen;
    static doublereal xib[100000], yib[100000], xob[100000], xco[100], yco[
	    100], yob[100000], rot;
    static integer done[1]	/* was [1][1] */;
    static logical fill;
    static integer xmax, ymax, coty;
    static logical rotf2;
    static integer intab[10000]	/* was [100][100] */;
    static logical disim, incps;
    extern /* Subroutine */ int dobox_(real *, real *, real *, real *, 
	    integer *, integer *, integer *, integer *, integer *, integer *, 
	    integer *, integer *, integer *, integer *, logical *, char *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *,
	     integer *, integer *, integer *, doublereal *, doublereal *, 
	    logical *, real *, real *, integer *, integer *, real *, char *, 
	    logical *, real *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, logical *,
	     logical *, doublereal *, doublereal *, doublereal *, doublereal *
	    , doublereal *, char *, logical *, logical *, logical *, integer *
	    , integer *, integer *, integer *, integer *, logical *, logical *
	    , logical *, integer *, integer *, integer *, ftnlen, ftnlen, 
	    ftnlen);
    static integer conum, istat;
    extern /* Subroutine */ int upwcs_(doublereal *, doublereal *, integer *, 
	    integer *, integer *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, char *, logical *, logical *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, char *, logical *, logical *, integer *, integer *, 
	    doublereal *, doublereal *, logical *, real *, real *, integer *, 
	    integer *, ftnlen, ftnlen), getgeo_(char *, integer *, doublereal 
	    *, integer *, integer *, integer *, doublereal *, doublereal *, 
	    integer *, integer *, ftnlen);
    static real filval;
    static logical bitcon, update, secpar, usewei;
    extern /* Subroutine */ int putfil_(real *, real *, integer *, integer *, 
	    real *);
    static logical rotfir, noover, usewcs;
    static doublereal wcsout[8];
    extern /* Subroutine */ int umsput_(char *, integer *, integer *, integer 
	    *, ftnlen);


/* Call Drizzle without use of IRAF interfaces. */

/* Supplied: */

/*   DATA - a 2d REAL array with dimensions (NX,NY): the input */
/*          data. */

/*   WEI - a 2d REAL array with dimensions (NX,NY): the input */
/*          weight array. */

/*   NDAT - a 2d REAL array with dimensions (ONX,ONY): the output */
/*          data array which is updated. It should be initialised */
/*          the first time around. */

/*   NCOU - a 2d REAL array with dimensions (ONX,ONY): the output */
/*          weight array which is updated. It should be initialised */
/*          the first time around. */

/*   NCON - a 2d INTEGER array with dimensions (ONX,ONY): the output */
/*          context array which is updated. It should be initialised */
/*          the first time around. */

/*   UNIQID - INTEGER: the unique processing ID for the input data */

/*   YSTART - INTEGER: index of the starting line for input data array */
/*                    relative to the entire image. */

/*   NX,NY  - INTEGER: dimensions of the full input image */

/*   DNY     - INTEGER: *Actual* number of rows in the input data array */

/*   ONX,ONY - INTEGERs: the dimensions of the output array */

/*   PXG,PYG - 2d REAL arrays with dimensions (XGDIM,YGDIM): the */
/*           distortion image arrays. If no images are used, they */
/*           should default to blank 2x2 arrays. */

/*   XSH,YSH,DROT,SCALE - REALs: the standard linear drizzle user */
/*                        parameters. DROT is in degrees. */

/*   SHFTFR - CHARACTER*8 f77 character string: WCS frame for shifts: */
/*            "input" (default) or "output" */

/*   SHFTUN - CHARACTER*8 f77 character string: specifies whether the */
/*            shift units are in terms of */
/*            "input" (default) or "output" pixels */

/*   ALIGN  - CHARACTER*8 f77 character string: takes the values */
/*            'corner' or 'center' to determine the position of the */
/*            reference point for rotation. */

/*   PFRACT - REAL: the drizzle "pixfrac" parameter. */

/*   KERNEL - CHARACTER*8 f77 character string: the type of kernel */
/*            used to distribute weight onto the output. Currently */
/*            supports "square","gaussian","tophat","point","turbo", */
/*            "lanczos2" and "lanczos3". */

/*   COEFFS - CHARACTER*80 f77 character string: the name of the */
/*            text file containing distortion correction coefficients. */

/*   INUN   - CHARACTER*8 f77 character string: the units for the input */
/*            image: "cps" or "counts" (default) */

/*   EXPIN  - REAL: exposure time for input image, to allow scaling */
/*            of data to counts/sec. */

/*   WTSCL  - REAL: scaling factor applied to input weighting image */

/*   FILSTR - CHARACTER*80: pixel value to use for filling empty output pixels */
/*            'INDEF' (default) or str(float value).  If 'INDEF', do not fill. */

/*   WCS    - DOUBLE PRECISION: input WCS (corrected in-place by UPWCS) */

/*   VFLAG  - INTEGER: 1=verbose, anything else not. */

/* Returned: */

/*    a REAL status is returned: the status return of the DOBOX */
/*    routine */

/* This was developed as a simple interface to the lower-level */
/* drizzle utilities which could be invoked without the need for */
/* an IRAF environment. */

/* If it is intended to be called from IDL the following applies: */

/* All parameters are passed from IDL using the "call external" */
/* mechanism. The intermediate file is idldriz.f. The linking looks */
/* something like this on Solaris 8: */

/*  f77 -c idldriz.f tdriz.f drutil.f drcall.f -pic -G */
/*  f77 -o idldriz.so drcall.o drutil.o idldriz.o tdriz.o -G -lF77 -lM77 -lsunmath -lm -lc */

/* If just being called directly from some f77 application it is */
/* only necessary to link in the following: */

/*  f77 -o app app.f tdriz.f drutil.f drcall.f */

/* where "app" is the top-level application. */

/* Notes: */

/*  - Some parameters, not included in the calling sequence, are hardcoded */
/* below in the section marked below. */

/*  - Secondary paramFILSTReters are not supported at present */

/*  - All the weighting information must be supplied in the WEI array. */
/*    Ie, wt_scl=1.0. */

/*  - Context images FILSTRare not currently supported */

/*  Test version, Richard Hook, ST-ECF, STScI, September 2002 */
/*  Improved documentation, Richard Hook, ST-ECF, STScI, November 2002 */

/* Modified to run under Linux, Richard Hook, ST-ECF, STScI, April 2003 */

/* F2PY control codes added by WJH, 18 Dec 2003 */


/* f2py integer intent(hide),depend(data) :: nx = shape(data,0) */
/* f2py integer intent(hide),depend(data) :: ny = shape(data,1) */
/* f2py integer intent(hide),depend(ndat) :: onx = shape(ndat,0) */
/* f2py integer intent(hide),depend(ndat) :: ony = shape(ndat,1) */
/* f2py real intent(in,c) :: data, wei */
/* f2py real intent(in,out):: ndat,ncou */
/* f2py real intent(in) :: xsh, ysh, drot, scale, pfract */
/* f2py real intent(in) :: wtscl, expin */
/* f2py character*8 intent(in) :: align */
/* f2py character*8 intent(in) :: kernel */
/* f2py character*80 :: coeffs */
/* f2py integer intent(in) :: vflag */

/* Distortion coefficients and related */
/* Maximum size (in X) of the input image */
/* Distortion image arrays */
/* Context related things */
/* We give these minimal dimensions */
/* Secondary geometrical parameters, added in V1.5 */
/* Keep quiet */
    /* Parameter adjustments */
    wei_dim1 = *nx;
    wei_offset = 1 + wei_dim1;
    wei -= wei_offset;
    data_dim1 = *nx;
    data_offset = 1 + data_dim1;
    data -= data_offset;
    ncon_dim1 = *onx;
    ncon_offset = 1 + ncon_dim1;
    ncon -= ncon_offset;
    ncou_dim1 = *onx;
    ncou_offset = 1 + ncou_dim1;
    ncou -= ncou_offset;
    ndat_dim1 = *onx;
    ndat_offset = 1 + ndat_dim1;
    ndat -= ndat_offset;
    pyg_dim1 = *xgdim;
    pyg_offset = 1 + pyg_dim1;
    pyg -= pyg_offset;
    pxg_dim1 = *xgdim;
    pxg_offset = 1 + pxg_dim1;
    pxg -= pxg_offset;
    --wcs;

    /* Function Body */
    if (*vflag == 1) {
	verbose_1.verbose = TRUE_;
    } else {
	verbose_1.verbose = FALSE_;
    }
/* Define Version ID */
    s_copy(vers, "Callable DRIZZLE Version 0.7 (4th Apr 2005)", (ftnlen)50, (
	    ftnlen)43);
/* Announce */
    umsput_(vers, &c__1, &c__0, &istat, (ftnlen)50);
/* Get geometric distortion coefficients */

    idd = 0;
    getgeo_(coeffs, &idd, &lam, &coty, &c__100, &conum, xco, yco, clen, &
	    istat, (ftnlen)80);
    if (istat != 0) {
	return ret_val;
    }
/* Set DISIM logical based on whether distortion images have */
/*  been passed in for use or not. */
    if (*xgdim == 2 && *ygdim == 2) {
	disim = FALSE_;
    } else {
	disim = TRUE_;
    }
/* Setup reasonable defaults for the drizzling */
/*      XMIN=1 */
    xmax = *onx;
/*      YMIN=1 */
    ymax = *ony;
    noover = FALSE_;
/* All weighting is done by the weight array */
/*      WTSCL=1.0 */
/*      INCPS=.FALSE. */
/* Check the switch value for whether to read in CPS or */
/* counts (latter is the default) */
    if (s_cmp(inun, "cps", (ftnlen)3, (ftnlen)3) == 0) {
	incps = TRUE_;
    } else {
	incps = FALSE_;
    }
/* Input exposure assumed to be 1.0s */
/*      EXPIN=1.0 */
/*      ROTFIR=.TRUE. */
/* Check the parameter specifying which frames shifts are applied */
    if (s_cmp(shftfr, "input", (ftnlen)5, (ftnlen)5) == 0) {
	rotfir = FALSE_;
    } else {
	rotfir = TRUE_;
    }
/* Convert the shift units if necessary */
    if (s_cmp(shftun, "output", (ftnlen)6, (ftnlen)6) == 0) {
	*xsh *= *scale;
	*ysh *= *scale;
    }
/* Convert the rotation to radians */
    rot = *drot * 3.141592653f / 180.f;
/* Convert the secondary parameters into suitable units */
    *rot2 = *rot2 * 3.14159265358979 / 180.;
    if (s_cmp(shfr2, "input", (ftnlen)5, (ftnlen)5) == 0) {
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
    update = TRUE_;
    usewei = TRUE_;
    usewcs = FALSE_;
    con = TRUE_;
    bitcon = TRUE_;
/* Do the drizzling */
    dobox_(&data[data_offset], &wei[wei_offset], &ndat[ndat_offset], &ncou[
	    ncou_offset], &ncon[ncon_offset], done, nx, ny, dny, ystart, xmin,
	     &xmax, ymin, &ymax, &noover, kernel, xi, xo, yi, yo, xib, xob, 
	    yib, yob, onx, ony, &coty, &conum, xco, yco, &disim, &pxg[
	    pxg_offset], &pyg[pyg_offset], xgdim, ygdim, wtscl, align, &incps,
	     expin, pfract, scale, &rot, xsh, ysh, &wcs[1], wcsout, &rotfir, &
	    secpar, xsh2, ysh2, rot2, xscale, yscale, shfr2, &rotf2, &con, &
	    bitcon, intab, &c__100, &c__100, &nen, uniqid, &update, &usewei, &
	    usewcs, &istat, nmiss, nskip, (ftnlen)8, (ftnlen)8, (ftnlen)8);
/* Check for meaningful values */
    if (s_cmp(filstr, "INDEF", (ftnlen)5, (ftnlen)5) != 0 && s_cmp(filstr, 
	    "indef", (ftnlen)5, (ftnlen)5) != 0) {
	ici__1.icierr = 1;
	ici__1.iciend = 1;
	ici__1.icirnum = 1;
	ici__1.icirlen = 80;
	ici__1.iciunit = filstr;
	ici__1.icifmt = 0;
	istat = s_rsli(&ici__1);
	if (istat != 0) {
	    goto L100001;
	}
	istat = do_lio(&c__4, &c__1, (char *)&filval, (ftnlen)sizeof(real));
	if (istat != 0) {
	    goto L100001;
	}
	istat = e_rsli();
L100001:
	if (istat != 0) {
	    umsput_("! Invalid filling value specified", &c__1, &c__0, &istat,
		     (ftnlen)33);
	    fill = FALSE_;
	} else {
	    fill = TRUE_;
	}
    } else {
	fill = FALSE_;
    }
/* Put in the fill values (if defined) */
    if (fill) {
	i__1 = xmax - *xmin + 1;
	i__2 = ymax - *ymin + 1;
	putfil_(&ndat[ndat_offset], &ncou[ncou_offset], &i__1, &i__2, &filval)
		;
    }
/* The arrays NDAT and NCOU will have been updated */
/* Update the WCS, if it needs to be updated. */
/*  Only need to do once per image, not once per section. */
    if (*ystart == 0) {
	upwcs_(&wcs[1], &wcs[1], nx, ny, onx, ony, xsh, ysh, &rot, scale, 
		align, &rotfir, &secpar, xsh2, ysh2, rot2, xscale, yscale, 
		shfr2, &rotf2, &usewcs, &coty, &conum, xco, yco, &disim, &pxg[
		pxg_offset], &pyg[pyg_offset], xgdim, ygdim, (ftnlen)8, (
		ftnlen)8);
    }
/* Return the value of status */
    ret_val = (real) istat;
    return ret_val;
} /* tdriz_ */

