/* twdriz.f -- translated by f2c (version 20060506).
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
static integer c__100 = 100;
static integer c__4 = 4;

doublereal twdriz_(real *data, real *wei, real *ndat, real *ncou, integer *
	ystart, integer *nx, integer *ny, integer *dny, integer *onx, integer 
	*ony, doublereal *wcs, doublereal *wcsout, real *pxg, real *pyg, 
	integer *xgdim, integer *ygdim, real *pfract, char *kernel, char *
	coeffs, char *filstr, integer *vflag, integer *clen, integer *nmiss, 
	integer *nskip, char *vers, ftnlen kernel_len, ftnlen coeffs_len, 
	ftnlen filstr_len, ftnlen vers_len)
{
    /* System generated locals */
    integer pxg_dim1, pxg_offset, pyg_dim1, pyg_offset, data_dim1, 
	    data_offset, wei_dim1, wei_offset, ndat_dim1, ndat_offset, 
	    ncou_dim1, ncou_offset, i__1, i__2;
    real ret_val;
    icilist ici__1;

    /* Builtin functions */
    /* Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen);
    integer s_cmp(char *, char *, ftnlen, ftnlen), s_rsli(icilist *), do_lio(
	    integer *, integer *, char *, ftnlen), e_rsli(void);

    /* Local variables */
    static real xi[400000]	/* was [100000][4] */, yi[400000]	/* 
	    was [100000][4] */, xo[400000]	/* was [100000][4] */, yo[
	    400000]	/* was [100000][4] */;
    static integer idd;
    static real lam;
    static logical con;
    static integer nen;
    static real xib[100000], yib[100000], xob[100000], xco[100], yco[100], 
	    yob[100000], xsh, ysh, rot;
    static doublereal xsh2, ysh2, rot2, beta;
    static integer done[1]	/* was [1][1] */;
    static logical fill;
    static integer ncon[1]	/* was [1][1] */;
    static real drot;
    static integer xmin, ymin, xmax, ymax, coty;
    static char shfr2[8];
    static logical rotf2;
    static doublereal alpha;
    static real scale;
    static char align[8];
    static integer intab[10000]	/* was [100][100] */;
    static logical disim, incps;
    extern /* Subroutine */ int dobox_(real *, real *, real *, real *, 
	    integer *, integer *, integer *, integer *, integer *, integer *, 
	    integer *, integer *, integer *, integer *, logical *, char *, 
	    real *, real *, real *, real *, real *, real *, real *, real *, 
	    integer *, integer *, integer *, integer *, real *, real *, 
	    logical *, real *, real *, integer *, integer *, real *, char *, 
	    logical *, real *, real *, real *, real *, real *, real *, 
	    doublereal *, doublereal *, logical *, logical *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, char *, 
	    logical *, logical *, logical *, integer *, integer *, integer *, 
	    integer *, integer *, logical *, logical *, logical *, integer *, 
	    integer *, integer *, doublereal *, doublereal *, ftnlen, ftnlen, 
	    ftnlen);
    static integer conum;
    static real expin;
    static integer istat;
    static real wtscl;
    extern /* Subroutine */ int getgeo_(char *, integer *, real *, integer *, 
	    integer *, integer *, real *, real *, integer *, integer *, 
	    ftnlen);
    static real filval;
    static logical bitcon, secpar, update;
    static doublereal xscale, yscale;
    static integer uniqid;
    static logical usewei;
    extern /* Subroutine */ int putfil_(real *, real *, integer *, integer *, 
	    real *);
    static logical rotfir, noover, usewcs;
    extern /* Subroutine */ int umsput_(char *, integer *, integer *, integer 
	    *, ftnlen);


/* Call Wdrizzle without use of IRAF interfaces. */

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

/*   NX,NY,ONX,ONY - INTEGERs: the dimensions of the arrays described */
/*                   above. */

/*   WCS,WCSOUT - double precision arrays, 8 elements in each. */
/*                Input and output WCS. */

/*   PFRACT - REAL: the drizzle "pixfrac" parameter. */

/*   KERNEL - CHARACTER*8 f77 character string: the type of kernel */
/*            used to distribute weight onto the output. Currently */
/*            supports "square","gaussian","tophat","point","turbo", */
/*            "lanczos2" and "lanczos3". */

/*   COEFFS - CHARACTER*80 f77 character string: the name of the */
/*            text file containing distortion correction coefficients. */

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

/*  f77 -c idlwdriz.f tdriz.f drutil.f drcall.f -pic -G */
/*  f77 -o idlwdriz.so drcall.o drutil.o idldriz.o tdriz.o -G -lF77 -lM77 -lsunmath -lm -lc */

/* If just being called directly from some f77 application it is */
/* only necessary to link in the following: */

/*  f77 -o app app.f twdriz.f drutil.f drcall.f */

/* where "app" is the top-level application. */

/* Notes: */

/*  - Some parameters, not included in the calling sequence, are hardcoded */
/* below in the section marked below. */

/*  - Secondary parameters are not supported at present */

/*  - All the weighting information must be supplied in the WEI array. */
/*    Ie, wt_scl=1.0. */

/*  - Context images are not currently supported */

/*  Test version, Richard Hook, ST-ECF, STScI, September 2002 */
/*  Improved documentation, Richard Hook, ST-ECF, STScI, November 2002 */

/* Modified to run under Linux, Richard Hook, ST-ECF, STScI, April 2003 */

/* Wdrizzle version, ST-ECF, STScI, September 2003 */
/* Added VFLAG for verbosity, STScI/ECF, October 2003 */

/* Distortion coefficients and related */
/* Distortion image arrays */
/* Verbose flag */
/* Maximum size (in X) of the input image */
/* Context related things */
/* We give these minimal dimensions */
/* Secondary geometrical parameters, added in V1.5 */
/* Check for verbose */
    /* Parameter adjustments */
    wei_dim1 = *nx;
    wei_offset = 1 + wei_dim1;
    wei -= wei_offset;
    data_dim1 = *nx;
    data_offset = 1 + data_dim1;
    data -= data_offset;
    ncou_dim1 = *onx;
    ncou_offset = 1 + ncou_dim1;
    ncou -= ncou_offset;
    ndat_dim1 = *onx;
    ndat_offset = 1 + ndat_dim1;
    ndat -= ndat_offset;
    --wcs;
    --wcsout;
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
/* Define Version ID */
    s_copy(vers, "Callable WDRIZZLE Version 0.5 (25rd June 2004)", (ftnlen)50,
	     (ftnlen)46);
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
    xmin = 1;
    xmax = *onx;
    ymin = 1;
    ymax = *ony;
    noover = FALSE_;
/* All weighting is done by the weight array */
    wtscl = 1.f;
    incps = TRUE_;
/* Input exposure assumed to be 1.0s */
    expin = 1.f;
    rotfir = FALSE_;
/* Convert the rotation to radians */
    rot = drot * 3.141592653f / 180.f;
/* Secondary parameters are not currently supported */
    secpar = FALSE_;
    update = TRUE_;
    usewei = TRUE_;
/* Wdrizzle, hence... */
    usewcs = TRUE_;
    con = FALSE_;
    uniqid = 1;
/* Do the drizzling */
    dobox_(&data[data_offset], &wei[wei_offset], &ndat[ndat_offset], &ncou[
	    ncou_offset], ncon, done, nx, ny, dny, ystart, &xmin, &xmax, &
	    ymin, &ymax, &noover, kernel, xi, xo, yi, yo, xib, xob, yib, yob, 
	    onx, ony, &coty, &conum, xco, yco, &disim, &pxg[pxg_offset], &pyg[
	    pyg_offset], xgdim, ygdim, &wtscl, align, &incps, &expin, pfract, 
	    &scale, &rot, &xsh, &ysh, &wcs[1], &wcsout[1], &rotfir, &secpar, &
	    xsh2, &ysh2, &rot2, &xscale, &yscale, shfr2, &rotf2, &con, &
	    bitcon, intab, &c__100, &c__100, &nen, &uniqid, &update, &usewei, 
	    &usewcs, &istat, nmiss, nskip, &alpha, &beta, (ftnlen)8, (ftnlen)
	    8, (ftnlen)8);
/* The arrays NDAT and NCOU will have been updated */
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
	i__1 = xmax - xmin + 1;
	i__2 = ymax - ymin + 1;
	putfil_(&ndat[ndat_offset], &ncou[ncou_offset], &i__1, &i__2, &filval)
		;
    }
/* Return the value of status */
    ret_val = (real) istat;
    return ret_val;
} /* twdriz_ */

