/* doblot.f -- translated by f2c (version 20031025).
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

/* Table of constant values */

static integer c__3 = 3;
static integer c__2048 = 2048;
static real c_b4 = .01f;
static integer c__5 = 5;
static integer c__1 = 1;
static integer c__0 = 0;
static integer c__4 = 4;
static logical c_false = FALSE_;
static logical c_true = TRUE_;

/* Subroutine */ int doblot_(real *data, real *ndat, integer *intype, real *
	sinscl, real *kscale, real *misval, integer *xmin, integer *xmax, 
	integer *ymin, integer *ymax, integer *dnx, integer *dny, logical *
	rotfir, real *ef, char *align, integer *onx, integer *ony, integer *
	coty, integer *conum, doublereal *xco, doublereal *yco, logical *
	disim, real *pxg, real *pyg, integer *pxdim, integer *pydim, 
	doublereal *xin, doublereal *yin, doublereal *xout, doublereal *yout, 
	doublereal *scale, doublereal *rot, doublereal *xsh, doublereal *ysh, 
	logical *usewcs, doublereal *wcsin, doublereal *wcsout, char *geomod, 
	logical *secpar, doublereal *xsh2, doublereal *ysh2, doublereal *rot2,
	 doublereal *xscale, doublereal *yscale, char *shfr2, logical *rotf2, 
	ftnlen align_len, ftnlen geomod_len, ftnlen shfr2_len)
{
    /* System generated locals */
    integer data_dim1, data_offset, ndat_dim1, ndat_offset, pxg_dim1, 
	    pxg_offset, pyg_dim1, pyg_offset, i__1, i__2;
    real r__1;
    doublereal d__1;

    /* Builtin functions */
    integer s_cmp(char *, char *, ftnlen, ftnlen);
    double sqrt(doublereal);
    integer i_nint(real *), s_wsfi(icilist *), do_fio(integer *, char *, 
	    ftnlen), e_wsfi(void);

    /* Local variables */
    static integer i__, j;
    static real v, s2;
    static doublereal dx, dy;
    static integer lx, ly;
    static real xo, yo;
    static doublereal yv;
    static real ks2, spk;
    static doublereal xcen, ycen;
    static integer nbox;
    static doublereal scall;
    static char chars[80];
    static doublereal scdis;
    static integer istat, nmiss;
    extern /* Subroutine */ int filalu_(integer *, integer *, real *, real *),
	     drival_(doublereal *, doublereal *, integer *, integer *, 
	    integer *, integer *, integer *, logical *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, char *, logical *, 
	    logical *, doublereal *, doublereal *, doublereal *, doublereal *,
	     doublereal *, char *, logical *, logical *, doublereal *, 
	    doublereal *, integer *, integer *, doublereal *, doublereal *, 
	    logical *, real *, real *, integer *, integer *, doublereal *, 
	    doublereal *, ftnlen, ftnlen);
    extern doublereal ginter_(real *, real *, real *, integer *, integer *, 
	    real *, integer *, real *, integer *, real *), nrievl_(real *, 
	    real *, real *, integer *, integer *, integer *, integer *, real *
	    );
    static real lanlut[2048];
    extern /* Subroutine */ int bupwcs_(doublereal *, doublereal *, integer *,
	     integer *, integer *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, char *, logical *, logical *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, char *, logical *, logical *, integer *, integer *, 
	    doublereal *, doublereal *, logical *, real *, real *, integer *, 
	    integer *, ftnlen, ftnlen), umsput_(char *, integer *, integer *, 
	    integer *, ftnlen);

    /* Fortran I/O blocks */
    static icilist io___23 = { 0, chars, 0, "('! Warning, ',I7,' points were"
	    " outside ','the output image.')", 80, 1 };



/* This routine does the interpolation of the input array */
/* In version 1.0 and later this is done using the standard DRIVAL */
/* routine in drutil.f. */

/* Note - when running in WBLOT mode this routine uses the WCS */
/*        values but has to supply them the other way around to DRIVAL. */

/* Standard geometrical parameters set */
/* Space for Lanczos-style look-up-tables */
/* -- */
/* Some initial settings - note that the reference pixel */
/* is offset from the "centre": */

    /* Parameter adjustments */
    data_dim1 = *xmax - *xmin + 1;
    data_offset = 1 + data_dim1;
    data -= data_offset;
    --yout;
    --xout;
    --yin;
    --xin;
    ndat_dim1 = *onx;
    ndat_offset = 1 + ndat_dim1;
    ndat -= ndat_offset;
    --yco;
    --xco;
    pyg_dim1 = *pxdim;
    pyg_offset = 1 + pyg_dim1;
    pyg -= pyg_offset;
    pxg_dim1 = *pxdim;
    pxg_offset = 1 + pxg_dim1;
    pxg -= pxg_offset;
    --wcsin;
    --wcsout;

    /* Function Body */
    nmiss = 0;
/* If we need a LUT calculate one here first */
    if (*intype == 100) {
	filalu_(&c__3, &c__2048, &c_b4, lanlut);
    } else if (*intype == 105) {
	filalu_(&c__5, &c__2048, &c_b4, lanlut);
    }
/* In the WCS case we can't use the scale to calculate the */
/* Jacobian so we need to do it */

/* Note that we use the centre of the image rather than the */
/* reference pixel as the reference here */

/* This is taken from DOBOX except for the inversion of the image */
/* order */
/* This section applies in WBLOT mode and now contains the */
/* addition correction to separate the distortion induced scale change */
    if (*usewcs) {
	umsput_("USEWCS true...\n", &c__1, &c__0, &istat, (ftnlen)15);
	if (s_cmp(align, "corner", (ftnlen)8, (ftnlen)6) == 0) {
	    xcen = (doublereal) (*onx / 2) + .5f;
	    ycen = (doublereal) (*ony / 2) + .5f;
	} else {
	    xcen = (doublereal) (*onx / 2) + 1.f;
	    ycen = (doublereal) (*ony / 2) + 1.f;
	}
	xin[1] = xcen;
	xin[2] = xcen;
	xin[3] = xcen + 1.;
	xin[4] = xcen + 1.;
	yin[1] = ycen;
	yin[2] = ycen + 1.;
	yin[3] = ycen + 1.;
	yin[4] = ycen;
	drival_(&xin[1], &yin[1], &c__4, onx, ony, dnx, dny, &c_false, xsh, 
		ysh, rot, scale, align, rotfir, secpar, xsh2, ysh2, rot2, 
		xscale, yscale, shfr2, rotf2, usewcs, &wcsout[1], &wcsin[1], 
		coty, conum, &xco[1], &yco[1], disim, &pxg[pxg_offset], &pyg[
		pyg_offset], pxdim, pydim, &xout[1], &yout[1], (ftnlen)8, (
		ftnlen)8);
	scall = sqrt(1. / (d__1 = ((xout[2] - xout[4]) * (yout[1] - yout[3]) 
		- (xout[1] - xout[3]) * (yout[2] - yout[4])) * .5, abs(d__1)))
		;
/* Now calculate how much of this is from the geometric distortion */
	*xsh = 0.;
	*ysh = 0.;
	*rot = 0.;
	*scale = 1.;
	*secpar = FALSE_;
	*usewcs = FALSE_;
	drival_(&xin[1], &yin[1], &c__4, onx, ony, dnx, dny, &c_false, xsh, 
		ysh, rot, scale, align, rotfir, secpar, xsh2, ysh2, rot2, 
		xscale, yscale, shfr2, rotf2, usewcs, &wcsout[1], &wcsin[1], 
		coty, conum, &xco[1], &yco[1], disim, &pxg[pxg_offset], &pyg[
		pyg_offset], pxdim, pydim, &xout[1], &yout[1], (ftnlen)8, (
		ftnlen)8);
	scdis = sqrt(1. / (d__1 = ((xout[2] - xout[4]) * (yout[1] - yout[3]) 
		- (xout[1] - xout[3]) * (yout[2] - yout[4])) * .5, abs(d__1)))
		;
	*usewcs = TRUE_;
	*scale = scall / scdis;
    }
/* Image subset size */
    lx = *xmax - *xmin + 1;
    ly = *ymax - *ymin + 1;
/* Offsets */
    dx = (doublereal) (*xmin - 1);
    dy = (doublereal) (*ymin - 1);
/* Recalculate the area scaling factor */
    s2 = (real) (*scale * *scale);
/* Some useful numbers */
    spk = *kscale * .01f;
    if (*intype == 100) {
	r__1 = 3.f / *kscale;
	nbox = i_nint(&r__1);
    } else if (*intype == 105) {
	r__1 = 5.f / *kscale;
	nbox = i_nint(&r__1);
    }
    if (*intype >= 100) {
	ks2 = 1.f / (*kscale * *kscale);
    }
/* Outer loop over output image pixels (X,Y) */
    i__1 = *ony;
    for (j = 1; j <= i__1; ++j) {
	yv = (doublereal) j;
/* Set the X and Y start positions */
	xin[1] = 1.;
	yin[1] = yv;
	xin[2] = 0.;
	yin[2] = 0.;
/* Transform this vector */
	drival_(&xin[1], &yin[1], onx, onx, ony, dnx, dny, &c_true, xsh, ysh, 
		rot, scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, 
		yscale, shfr2, rotf2, usewcs, &wcsout[1], &wcsin[1], coty, 
		conum, &xco[1], &yco[1], disim, &pxg[pxg_offset], &pyg[
		pyg_offset], pxdim, pydim, &xout[1], &yout[1], (ftnlen)8, (
		ftnlen)8);
/* Loop through the output positions and do the interpolation */
	i__2 = *onx;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    xo = (real) (xout[i__] - dx);
	    yo = (real) (yout[i__] - dy);
/* Check it is on the input image */
	    if (xo >= 1.f && xo <= (real) lx && yo >= 1.f && yo <= (real) ly) 
		    {
/* Check for look-up-table interpolation */
		if (*intype >= 100) {
		    v = ginter_(&xo, &yo, &data[data_offset], &lx, &ly, 
			    lanlut, &c__2048, &spk, &nbox, misval) * ks2;
		} else {
/* Do the interpolation */
/* and allow for stretching because of scale change */
/* Also, use callable version of from INTER2D.F. */

		    v = nrievl_(&xo, &yo, &data[data_offset], &lx, &ly, &lx, 
			    intype, sinscl);
		}
		ndat[i__ + j * ndat_dim1] = v * *ef / s2;
	    } else {
/* If there is nothing for us then set the output to missing */
/* value flag */
		ndat[i__ + j * ndat_dim1] = *misval;
/* Count cases where the pixel is off the output image */
		++nmiss;
	    }
	}
    }
/* Report number of points which were outside the input frame */
    if (nmiss > 0) {
	s_wsfi(&io___23);
	do_fio(&c__1, (char *)&nmiss, (ftnlen)sizeof(integer));
	e_wsfi();
	umsput_(chars, &c__1, &c__0, &istat, (ftnlen)80);
    }
/* Finally we need to update the WCS using the standard routine */
/* modified to work in the "blot direction" */

/* Not needed for WBLOT */
    if (! (*usewcs)) {
	bupwcs_(&wcsin[1], &wcsout[1], dnx, dny, onx, ony, xsh, ysh, rot, 
		scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, 
		yscale, shfr2, rotf2, usewcs, coty, conum, &xco[1], &yco[1], 
		disim, &pxg[pxg_offset], &pyg[pyg_offset], pxdim, pydim, (
		ftnlen)8, (ftnlen)8);
    }
    return 0;
} /* doblot_ */

