/* drutil.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "f2c.h"

/* Common Block Declarations */

struct {
    real memr[1];
} mem_;

#define mem_1 mem_

/* Table of constant values */

static integer c__5 = 5;
static integer c__1 = 1;
static integer c__3 = 3;
static integer c__0 = 0;
static integer c__4 = 4;
static logical c_false = FALSE_;
static integer c__16 = 16;
static integer c__2 = 2;
static integer c__512 = 512;
static real c_b231 = .01f;
static logical c_true = TRUE_;

/* DRUTIL.F */

/* Utility routines for Drizzle and Blot */

/* These routines should call only themselves, without reliance on */
/* IRAF/STSDAS f77/vos.  Those routines will be found in 'ioutil.f'. */
/*   Exceptions: UIGL2R (in DOBOX) */

/* We need to revise GETGEO to read xgeoim/ygeoim images using */
/*   non-IRAF f77/vos routines as needed for callable version. */
/*   The 'wcs' option for coeffs (reading Spitzer-style coeffs from */
/*   header) also needs to be implemented without using f77/vos */
/*   routines.  This will probably require writing replacements for */
/*   the f77/vos routines to be put in 'drcall.f' that do not use f77/vos. */

/* Original version for DRIZZLE 3.3 and BLOT 3.3, December 2003 */
/* Richard Hook, ST-ECF/ESO */

/* This version modified for callable and STSDAS DRIZZLE simultaneously */
/* Warren Hack, STScI, 17-June-2004 */

/* History: */

/* The module GOLDH was transferred from drizzle as it is also */
/* now needed by blot. */

/* WCS routines added (XY2RD & RD2XY), September 1998 */

/* Modified GTCOEF routine to also handle STIS and NICMOS, */
/*    January 1999 */

/* Added additional code for reversing the cubic transformation added */
/*  June 1999 */

/* Major extensions and some routines transferred from the EIS utilities */
/* collection (for context handling), October 2000 */

/* Extensive modifications for a more flexible geometric coefficients */
/* scheme, changes in calling sequences, December 2000 */

/* April 2003 - added code for checking overlaps with output images */
/* and minimising unnecessary drizzling (CHOVER). */

/* Added double precision version of several routines for more precision. */
/*  Richard Hook, ST-ECF/ESO/STScI, June 2003 */

/* Full conversion to double precision (for geometry, not data values) */
/*   Richard Hook, ST-ECF/ESO/STScI, October 2003 */

/* Modifications to support "refpix" contruct in distortion files */
/*   Richard Hook, ST-ECF/ESO/STScI, December 2003 */

/* Subroutine */ int setim_(real *a, integer *nx, integer *ny, real *v)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;


/* Set a 2D array to a specified value */

/* SINGLE PRECISION routine. */

    /* Parameter adjustments */
    a_dim1 = *nx;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */
    i__1 = *ny;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *nx;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = *v;
	}
    }
    return 0;
} /* setim_ */

/* Subroutine */ int mulc_(real *a, integer *nx, integer *ny, real *v)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;


/* Multiply a 2D REAL array by a specified value */

/* SINGLE PRECISION routine. */

    /* Parameter adjustments */
    a_dim1 = *nx;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */
    i__1 = *ny;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *nx;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] *= *v;
	}
    }
    return 0;
} /* mulc_ */

/* Subroutine */ int copyim_(real *in, real *out, integer *nx, integer *ny)
{
    /* System generated locals */
    integer in_dim1, in_offset, out_dim1, out_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;


/* Simply copy a REAL array into another */

/* SINGLE PRECISION routine. */

    /* Parameter adjustments */
    out_dim1 = *nx;
    out_offset = 1 + out_dim1 * 1;
    out -= out_offset;
    in_dim1 = *nx;
    in_offset = 1 + in_dim1 * 1;
    in -= in_offset;

    /* Function Body */
    i__1 = *ny;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *nx;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    out[i__ + j * out_dim1] = in[i__ + j * in_dim1];
	}
    }
    return 0;
} /* copyim_ */

doublereal eval3_(doublereal *x, doublereal *y, doublereal *co)
{
    /* System generated locals */
    doublereal ret_val;


/* Evaluate a cubic geometric distortion in 2d */

    /* Parameter adjustments */
    --co;

    /* Function Body */
    ret_val = co[1] + co[2] * *x + co[3] * *y + co[4] * *x * *x + co[5] * *x *
	     *y + co[6] * *y * *y + co[7] * *x * *x * *x + co[8] * *x * *x * *
	    y + co[9] * *x * *y * *y + co[10] * *y * *y * *y;
    return ret_val;
} /* eval3_ */

doublereal eval4_(doublereal *x, doublereal *y, doublereal *co)
{
    /* System generated locals */
    doublereal ret_val;


/* Evaluate a 4th order (quartic) geometric distortion in 2d */

    /* Parameter adjustments */
    --co;

    /* Function Body */
    ret_val = co[1] + co[2] * *x + co[3] * *y + co[4] * *x * *x + co[5] * *x *
	     *y + co[6] * *y * *y + co[7] * *x * *x * *x + co[8] * *x * *x * *
	    y + co[9] * *x * *y * *y + co[10] * *y * *y * *y + co[11] * *x * *
	    x * *x * *x + co[12] * *x * *x * *x * *y + co[13] * *x * *x * *y *
	     *y + co[14] * *x * *y * *y * *y + co[15] * *y * *y * *y * *y;
    return ret_val;
} /* eval4_ */

doublereal eval5_(doublereal *x, doublereal *y, doublereal *co)
{
    /* System generated locals */
    doublereal ret_val;


/* Evaluate a 5th order geometric distortion in 2d */

    /* Parameter adjustments */
    --co;

    /* Function Body */
    ret_val = co[1] + co[2] * *x + co[3] * *y + co[4] * *x * *x + co[5] * *x *
	     *y + co[6] * *y * *y + co[7] * *x * *x * *x + co[8] * *x * *x * *
	    y + co[9] * *x * *y * *y + co[10] * *y * *y * *y + co[11] * *x * *
	    x * *x * *x + co[12] * *x * *x * *x * *y + co[13] * *x * *x * *y *
	     *y + co[14] * *x * *y * *y * *y + co[15] * *y * *y * *y * *y + 
	    co[16] * *x * *x * *x * *x * *x + co[17] * *x * *x * *x * *x * *y 
	    + co[18] * *x * *x * *x * *y * *y + co[19] * *x * *x * *y * *y * *
	    y + co[20] * *x * *y * *y * *y * *y + co[21] * *y * *y * *y * *y *
	     *y;
    return ret_val;
} /* eval5_ */

doublereal evaln_(doublereal *x, doublereal *y, doublereal *co, integer *
	order)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublereal ret_val;

    /* Builtin functions */
    double pow_di(doublereal *, integer *);

    /* Local variables */
    static integer m, n;
    static doublereal t;
    static integer nc;


/* Evaluate the value of a general 2d polynomial in X and Y */
/* complete with "half" cross-terms. */

/* For orders lower than 7 it is slightly more efficient to */
/* use the specific (poly4 etc) routines instead. */

/* Richard Hook, ST-ECF, February 2002 */

    /* Parameter adjustments */
    --co;

    /* Function Body */
    t = 0.f;
    nc = 1;
    i__1 = *order + 1;
    for (n = 1; n <= i__1; ++n) {
	i__2 = n;
	for (m = 1; m <= i__2; ++m) {
	    i__3 = n - m;
	    i__4 = m - 1;
	    t += co[nc] * pow_di(x, &i__3) * pow_di(y, &i__4);
	    ++nc;
	}
    }
    ret_val = t;
    return ret_val;
} /* evaln_ */

/* Subroutine */ int rad3_(doublereal *x, doublereal *y, doublereal *co, 
	doublereal *xo, doublereal *yo)
{
    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static doublereal f, r__;


/* Evaluate a 3rd order radial geometric distortion in 2d */
/* X version. Note that there is no zero order coefficient */
/* as this is physically meaningless. */

    /* Parameter adjustments */
    --co;

    /* Function Body */
    r__ = sqrt(*x * *x + *y * *y);
    f = co[1] + 1. + co[2] * r__ + co[3] * r__ * r__;
    *xo = f * *x;
    *yo = f * *y;
    return 0;
} /* rad3_ */

/* Subroutine */ int getco_(integer *lun, doublereal *lam, integer *coty, 
	integer *comax, integer *conum, doublereal *xco, doublereal *yco, 
	integer *istat)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1;

    /* Builtin functions */
    integer s_cmp(char *, char *, ftnlen, ftnlen), s_rsli(icilist *), do_lio(
	    integer *, integer *, char *, ftnlen), e_rsli(void), s_wsfi(
	    icilist *), do_fio(integer *, char *, ftnlen), e_wsfi(void);

    /* Local variables */
    static char line[80];
    static doublereal a, b, c__;
    static integer i__, j;
    static doublereal n;
    extern /* Subroutine */ int bfill_(integer *, char *, integer *, ftnlen);
    static char chars[80];
    static doublereal xdref, ydref;
    static char buffer[1024];
    extern /* Subroutine */ int ufglin_(integer *, char *, integer *, ftnlen);
    static logical newref;
    extern /* Subroutine */ int umsput_(char *, integer *, integer *, integer 
	    *, ftnlen);
    extern doublereal mgf2_(doublereal *);

    /* Fortran I/O blocks */
    static icilist io___17 = { 1, line, 1, 0, 80, 1 };
    static icilist io___22 = { 1, buffer, 1, 0, 1024, 1 };
    static icilist io___24 = { 1, buffer, 1, 0, 1024, 1 };
    static icilist io___25 = { 1, buffer, 1, 0, 1024, 1 };
    static icilist io___26 = { 0, line+4, 0, 0, 76, 1 };
    static icilist io___27 = { 1, buffer, 1, 0, 1024, 1 };
    static icilist io___28 = { 1, buffer, 1, 0, 1024, 1 };
    static icilist io___29 = { 1, line+6, 1, 0, 74, 1 };
    static icilist io___33 = { 0, chars, 0, "('-Using distortion reference p"
	    "oint: [',                       F10.4,',',F10.4,']')", 80, 1 };



/* Read geometric distortion coefficients in free format */
/* from a text file which has already been opened and return them. */

/* If there is a problem reading the numbers then the status flag will be */
/* returned as 1. If all is well it will be returned */
/* as 0. */

/* This routine caused problems with SUN SPARC SC4 compilers and was */
/* modified to avoid them. March 1997 */

/* Modified to handle higher-order terms in addition. */
/*   December 2000 */

/* Modified to also support the "refpix" specification at the start of the */
/* coefficients files to allow the reference point for the distortion to */
/* be decoupled from the center of the image. */
/*   December 2003 */

/* Different reference pixel support */
/* By default no different reference point */
    /* Parameter adjustments */
    --yco;
    --xco;

    /* Function Body */
    newref = FALSE_;
/* Skip blank lines and those beginning with # until we find a label */
    while(TRUE_) {
	ufglin_(lun, line, istat, (ftnlen)80);
	if (*istat != 0) {
	    goto L99;
	}
	if (s_cmp(line, " ", (ftnlen)80, (ftnlen)1) != 0 && *(unsigned char *)
		line != '#') {
/* First "Trauger" style coefficients - 60 of them */
	    if (s_cmp(line, "trauger", (ftnlen)7, (ftnlen)7) == 0) {
/* Now we need to calculate the coefficients which are a */
/* function of wavelength */
		n = mgf2_(lam);
/* Now we loop through extracting the coefficients, 3 per line, */
/* and calculating the wavelength dependence */
		j = 1;
		while(j <= 20) {
		    ufglin_(lun, line, istat, (ftnlen)80);
		    if (*istat != 0) {
			goto L99;
		    }
		    if (s_cmp(line, " ", (ftnlen)80, (ftnlen)1) != 0 && *(
			    unsigned char *)line != '#') {
			*istat = s_rsli(&io___17);
			if (*istat != 0) {
			    goto L100001;
			}
			*istat = do_lio(&c__5, &c__1, (char *)&a, (ftnlen)
				sizeof(doublereal));
			if (*istat != 0) {
			    goto L100001;
			}
			*istat = do_lio(&c__5, &c__1, (char *)&b, (ftnlen)
				sizeof(doublereal));
			if (*istat != 0) {
			    goto L100001;
			}
			*istat = do_lio(&c__5, &c__1, (char *)&c__, (ftnlen)
				sizeof(doublereal));
			if (*istat != 0) {
			    goto L100001;
			}
			*istat = e_rsli();
L100001:
			if (*istat != 0) {
			    goto L99;
			}
			if (j <= 10) {
/* Computing 2nd power */
			    d__1 = n - 1.5f;
			    xco[j] = a + b * (n - 1.5f) + c__ * (d__1 * d__1);
			} else {
/* Computing 2nd power */
			    d__1 = n - 1.5f;
			    yco[j - 10] = a + b * (n - 1.5f) + c__ * (d__1 * 
				    d__1);
			}
			++j;
		    }
		}
		*coty = 3;
		*conum = 10;
		*istat = 0;
		goto L99;
/* Next "cubic" style - 20 coefficients */
	    } else if (s_cmp(line, "cubic", (ftnlen)5, (ftnlen)5) == 0 || 
		    s_cmp(line, "poly3", (ftnlen)5, (ftnlen)5) == 0) {
/* Copy the rest of the file into a buffer ready for a free-format */
/* read operation */
		bfill_(lun, buffer, istat, (ftnlen)1024);
		*conum = 10;
		if (*istat == 0) {
		    *istat = s_rsli(&io___22);
		    if (*istat != 0) {
			goto L100002;
		    }
		    i__1 = *conum;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			*istat = do_lio(&c__5, &c__1, (char *)&xco[i__], (
				ftnlen)sizeof(doublereal));
			if (*istat != 0) {
			    goto L100002;
			}
		    }
		    i__2 = *conum;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			*istat = do_lio(&c__5, &c__1, (char *)&yco[i__], (
				ftnlen)sizeof(doublereal));
			if (*istat != 0) {
			    goto L100002;
			}
		    }
		    *istat = e_rsli();
L100002:
		    ;
		}
		if (*istat != 0) {
		    goto L99;
		}
		*coty = 3;
		*istat = 0;
		goto L99;
/* Now 4th order - 30 coefficients */
	    } else if (s_cmp(line, "poly4", (ftnlen)5, (ftnlen)5) == 0 || 
		    s_cmp(line, "quartic", (ftnlen)7, (ftnlen)7) == 0) {
		bfill_(lun, buffer, istat, (ftnlen)1024);
		*conum = 15;
		if (*istat == 0) {
		    *istat = s_rsli(&io___24);
		    if (*istat != 0) {
			goto L100003;
		    }
		    i__1 = *conum;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			*istat = do_lio(&c__5, &c__1, (char *)&xco[i__], (
				ftnlen)sizeof(doublereal));
			if (*istat != 0) {
			    goto L100003;
			}
		    }
		    i__2 = *conum;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			*istat = do_lio(&c__5, &c__1, (char *)&yco[i__], (
				ftnlen)sizeof(doublereal));
			if (*istat != 0) {
			    goto L100003;
			}
		    }
		    *istat = e_rsli();
L100003:
		    ;
		}
		if (*istat != 0) {
		    goto L99;
		}
		*coty = 4;
		*istat = 0;
		goto L99;
/* Now 5th order - 42 coefficients */
	    } else if (s_cmp(line, "poly5", (ftnlen)5, (ftnlen)5) == 0 || 
		    s_cmp(line, "quintic", (ftnlen)7, (ftnlen)7) == 0) {
		bfill_(lun, buffer, istat, (ftnlen)1024);
		*conum = 21;
		if (*istat == 0) {
		    *istat = s_rsli(&io___25);
		    if (*istat != 0) {
			goto L100004;
		    }
		    i__1 = *conum;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			*istat = do_lio(&c__5, &c__1, (char *)&xco[i__], (
				ftnlen)sizeof(doublereal));
			if (*istat != 0) {
			    goto L100004;
			}
		    }
		    i__2 = *conum;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			*istat = do_lio(&c__5, &c__1, (char *)&yco[i__], (
				ftnlen)sizeof(doublereal));
			if (*istat != 0) {
			    goto L100004;
			}
		    }
		    *istat = e_rsli();
L100004:
		    ;
		}
		if (*istat != 0) {
		    goto L99;
		}
		*coty = 5;
		*istat = 0;
		goto L99;
/* Higher orders (>5) */
	    } else if (s_cmp(line, "poly", (ftnlen)4, (ftnlen)4) == 0) {
/* First find the order by reading the rest of the line */
		s_rsli(&io___26);
		do_lio(&c__3, &c__1, (char *)&(*coty), (ftnlen)sizeof(integer)
			);
		e_rsli();
		*conum = (*coty + 1) * (*coty + 2) / 2;
		bfill_(lun, buffer, istat, (ftnlen)1024);
		if (*istat == 0) {
		    *istat = s_rsli(&io___27);
		    if (*istat != 0) {
			goto L100005;
		    }
		    i__1 = *conum;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			*istat = do_lio(&c__5, &c__1, (char *)&xco[i__], (
				ftnlen)sizeof(doublereal));
			if (*istat != 0) {
			    goto L100005;
			}
		    }
		    i__2 = *conum;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			*istat = do_lio(&c__5, &c__1, (char *)&yco[i__], (
				ftnlen)sizeof(doublereal));
			if (*istat != 0) {
			    goto L100005;
			}
		    }
		    *istat = e_rsli();
L100005:
		    ;
		}
		if (*istat != 0) {
		    goto L99;
		}
		*istat = 0;
		goto L99;
/* Next third-order radial polynomial */
	    } else if (s_cmp(line, "radial", (ftnlen)6, (ftnlen)6) == 0 || 
		    s_cmp(line, "rad3", (ftnlen)4, (ftnlen)4) == 0) {
/* Copy the rest of the file into a buffer ready for a free-format */
/* read operation */
		bfill_(lun, buffer, istat, (ftnlen)1024);
		*conum = 3;
		if (*istat == 0) {
		    *istat = s_rsli(&io___28);
		    if (*istat != 0) {
			goto L100006;
		    }
		    i__1 = *conum;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			*istat = do_lio(&c__5, &c__1, (char *)&xco[i__], (
				ftnlen)sizeof(doublereal));
			if (*istat != 0) {
			    goto L100006;
			}
		    }
		    *istat = e_rsli();
L100006:
		    ;
		}
		if (*istat != 0) {
		    goto L99;
		}
		*coty = -3;
		*istat = 0;
		goto L99;
/* Optionally there may be a line with a different centre position */
	    } else if (s_cmp(line, "refpix", (ftnlen)6, (ftnlen)6) == 0) {
		*istat = s_rsli(&io___29);
		if (*istat != 0) {
		    goto L100007;
		}
		*istat = do_lio(&c__5, &c__1, (char *)&xdref, (ftnlen)sizeof(
			doublereal));
		if (*istat != 0) {
		    goto L100007;
		}
		*istat = do_lio(&c__5, &c__1, (char *)&ydref, (ftnlen)sizeof(
			doublereal));
		if (*istat != 0) {
		    goto L100007;
		}
		*istat = e_rsli();
L100007:
		if (*istat != 0) {
		    umsput_("! Invalid reference pixel specification in coef"
			    "ficients file", &c__1, &c__0, istat, (ftnlen)60);
		    *istat = 1;
		    goto L99;
		} else {
		    newref = TRUE_;
		    s_wsfi(&io___33);
		    do_fio(&c__1, (char *)&xdref, (ftnlen)sizeof(doublereal));
		    do_fio(&c__1, (char *)&ydref, (ftnlen)sizeof(doublereal));
		    e_wsfi();
		    umsput_(chars, &c__1, &c__0, istat, (ftnlen)80);
		}
	    } else {
		umsput_("! Unknown coefficient set type", &c__1, &c__0, istat,
			 (ftnlen)30);
		*istat = 1;
		goto L99;
	    }
	}
    }
L99:
/* If we have a modified reference pixel we encode this in the standard */
/* coefficients by adding an offset and incrementing the number of coefficients */
/* by one */
    if (newref) {
	*coty += 100;
	++(*conum);
	xco[*conum] = xdref;
	yco[*conum] = ydref;
    }
    return 0;
} /* getco_ */

doublereal mgf2_(doublereal *lam)
{
    /* System generated locals */
    doublereal ret_val;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static doublereal sig;


/* Calculate the refractive index of MgF2 for a given */
/* wavelength (in nm) using the formula given by Trauger (1995) */

    sig = 1e7 / *lam;
    ret_val = sqrt(2.590355e10 / (5.312993e10 - sig * sig) + 1.f + 
	    4454370800. / (1.117083e10 - sig * sig) + 408388.97 / (176636.1 - 
	    sig * sig));
    return ret_val;
} /* mgf2_ */

/* Subroutine */ int putfil_(real *dat, real *cou, integer *onx, integer *ony,
	 real *filval)
{
    /* System generated locals */
    integer dat_dim1, dat_offset, cou_dim1, cou_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;


/* Put in the missing filling value where the weight is */
/* zero */

/* SINGLE PRECISION routine. */

    /* Parameter adjustments */
    cou_dim1 = *onx;
    cou_offset = 1 + cou_dim1 * 1;
    cou -= cou_offset;
    dat_dim1 = *onx;
    dat_offset = 1 + dat_dim1 * 1;
    dat -= dat_offset;

    /* Function Body */
    i__1 = *ony;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *onx;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (cou[i__ + j * cou_dim1] == 0.f) {
		dat[i__ + j * dat_dim1] = *filval;
	    }
	}
    }
    return 0;
} /* putfil_ */

/* Subroutine */ int inmat_(doublereal *p, doublereal *q, doublereal *r__, 
	doublereal *s)
{
    static doublereal a, b, c__, d__, det;


/* Invert a 2 by 2 double precision matrix */
/* in place. It assumes that the matrix is not */
/* singular. */

    det = *p * *s - *q * *r__;
    a = *s / det;
    b = -(*q) / det;
    c__ = -(*r__) / det;
    d__ = *p / det;
    *p = a;
    *q = b;
    *r__ = c__;
    *s = d__;
    return 0;
} /* inmat_ */

/* Subroutine */ int xy2rd_(doublereal *x, doublereal *y, doublereal *r__, 
	doublereal *d__, doublereal *wcs)
{
    /* System generated locals */
    doublereal d__1, d__2;

    /* Builtin functions */
    double sin(doublereal), cos(doublereal), atan2(doublereal, doublereal), 
	    sqrt(doublereal);

    /* Local variables */
    static doublereal xi, ra0, eta, dec0;


/* XY2RD - convert a pixel position to an equatorial (RA,DEC) */
/*         position assuming a TAN projection and WCS with the */
/*         standard 8 elements. */

/* It is based on the xy2rd task in STSDAS */

/* Output angles are in degrees */

/* Richard Hook, STScI, 11th August 1998 */

/* First convert pixel coordinates to tangent plane in radians */
    /* Parameter adjustments */
    --wcs;

    /* Function Body */
    xi = (wcs[5] * (*x - wcs[1]) + wcs[7] * (*y - wcs[3])) * 
	    .017453292519943295;
    eta = (wcs[6] * (*x - wcs[1]) + wcs[8] * (*y - wcs[3])) * 
	    .017453292519943295;
/* And convert the reference point on the sky to radians */
    ra0 = wcs[2] * .017453292519943295;
    dec0 = wcs[4] * .017453292519943295;
/* Now go to equatorial from tangent plane */
    *r__ = atan2(xi, cos(dec0) - eta * sin(dec0)) + ra0;
/* Computing 2nd power */
    d__1 = cos(dec0) - eta * sin(dec0);
/* Computing 2nd power */
    d__2 = xi;
    *d__ = atan2(eta * cos(dec0) + sin(dec0), sqrt(d__1 * d__1 + d__2 * d__2))
	    ;
/* Convert back to degrees and check the range */
    *r__ /= .017453292519943295;
    *d__ /= .017453292519943295;
    if (*r__ < 0.f) {
	*r__ += 360.f;
    }
    return 0;
} /* xy2rd_ */

/* Subroutine */ int rd2xy_(doublereal *r__, doublereal *d__, doublereal *x, 
	doublereal *y, doublereal *wcs, integer *istat)
{
    /* Builtin functions */
    double sin(doublereal), cos(doublereal);

    /* Local variables */
    static doublereal cdinv[4]	/* was [2][2] */, ra, xi, ra0, bottom, dec, 
	    eta, det, dec0;


/* RD2XY - convert an equatorial (RA,DEC) position to a pixel position */
/*         assuming a TAN projection and WCS with the */
/*         standard 8 elements. */

/* It is based on the rd2xy task in STSDAS */

/* Input angles are in degrees */

/* Richard Hook, STScI, 13th August 1998 */

/* First invert the CD matrix */
    /* Parameter adjustments */
    --wcs;

    /* Function Body */
    det = wcs[5] * wcs[8] - wcs[7] * wcs[6];
    if (det == 0.f) {
	*istat = 1;
	return 0;
    }
    cdinv[0] = wcs[8] / det;
    cdinv[2] = -wcs[7] / det;
    cdinv[1] = -wcs[6] / det;
    cdinv[3] = wcs[5] / det;
/* Translate from RA,Dec to X,Y */
    ra0 = wcs[2] * .017453292519943295;
    dec0 = wcs[4] * .017453292519943295;
    ra = *r__ * .017453292519943295;
    dec = *d__ * .017453292519943295;
    bottom = sin(dec) * sin(dec0) + cos(dec) * cos(dec0) * cos(ra - ra0);
    if (bottom == 0.) {
	*istat = 1;
	return 0;
    }
/* Calculate tangent plane position and convert to degrees */
    xi = cos(dec) * sin(ra - ra0) / bottom / .017453292519943295;
    eta = (sin(dec) * cos(dec0) - cos(dec) * sin(dec0) * cos(ra - ra0)) / 
	    bottom / .017453292519943295;
/* Convert back to pixels using the inverse of the CD matrix */
    *x = cdinv[0] * xi + cdinv[2] * eta + wcs[1];
    *y = cdinv[1] * xi + cdinv[3] * eta + wcs[3];
    *istat = 0;
    return 0;
} /* rd2xy_ */

/* Subroutine */ int wcslin_(doublereal *wcsin, doublereal *wcsout, 
	doublereal *xcen, doublereal *ycen, integer *coty, integer *conum, 
	doublereal *xco, doublereal *yco, logical *disim, real *xg, real *yg, 
	integer *nx, integer *ny, doublereal *xc, doublereal *yc, doublereal *
	xs, doublereal *ys, doublereal *xt, doublereal *yt)
{
    /* System generated locals */
    integer xg_dim1, xg_offset, yg_dim1, yg_offset;
    doublereal d__1, d__2;

    /* Local variables */
    static doublereal xout[4], yout[4];
    extern doublereal eval3_(doublereal *, doublereal *, doublereal *), 
	    eval4_(doublereal *, doublereal *, doublereal *), eval5_(
	    doublereal *, doublereal *, doublereal *);
    static doublereal a, b, c__, d__;
    extern /* Subroutine */ int xy2rd_(doublereal *, doublereal *, doublereal 
	    *, doublereal *, doublereal *);
    static integer i__;
    extern /* Subroutine */ int rd2xy_(doublereal *, doublereal *, doublereal 
	    *, doublereal *, doublereal *, integer *);
    static doublereal x[4], y[4], xdoff, ydoff, xdref, ydref;
    extern doublereal evaln_(doublereal *, doublereal *, doublereal *, 
	    integer *);
    static integer istat, scoty;
    static doublereal x0, y0, ra;
    extern /* Subroutine */ int fitlin_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     doublereal *, doublereal *, doublereal *, doublereal *, integer *
	    );
    static logical newref;
    extern /* Subroutine */ int umsput_(char *, integer *, integer *, integer 
	    *, ftnlen);
    static doublereal dec, xin[4], yin[4];
    extern /* Subroutine */ int rad3_(doublereal *, doublereal *, doublereal *
	    , doublereal *, doublereal *);


/* WCSLIN - derive best linear transformation coefficients to map */
/*          pixel positions from one world coordinate system to another. */

/* Supplied: */

/*  WCSIN - double precision 8 element array. The WCS of the input image. */
/*          This includes CRPIX1/2, CRVAL1/2 and a CD matrix. */

/*  WCSOUT - double precision 8 element array. The WCS of the output image. */
/*          This includes CRPIX1/2, CRVAL1/2 and a CD matrix. */

/*  XCEN,YCEN - double precision, the centre of the chip, for reference */

/*  COTY,CONUM - coefficients type code and number of coefficients */

/*  XCO,YCO - double arrays, the non-linear distortion coefficients */

/* DISIM,XG,YG,NX,NY - distortion image support */

/* Returned: */

/* XC,YC,XS,YS,XT,YT - doubles, linear transformation coefficients */

/*   Richard Hook, 18th September 1998 */
/*   Modified for more general coefficients and changed calling */
/*   sequence, December 2000 */

/*  Further modification to try to handle ACS images with big distortions */
/*  and offsets. January 2001. */

/* Modified for better accuracy by greater use of double precision */
/* values. November 2002. */

/* Modified to ignore distortion images - ie, to assume that they do */
/* not have a global effect. */
/*    Richard Hook, STScI/ECF, September 2003 */

/* Added support for "refpix" offsets, */
/*    Richard Hook, STScI/ECF, December 2003 */

/* Used a smaller range of offset for double precision version */
/* to minimise effects of nonlinear terms. */
/*   Richard Hook, STScI/ECF, January 2004 */

/* Input parameters */
/* Output parameters */
/* Local variables */
/* First check for the presence of "refpix" additional information */
/* in the coefficients */
/* If it is set a flag and offset again */
    /* Parameter adjustments */
    --wcsin;
    --wcsout;
    --yco;
    --xco;
    yg_dim1 = *nx;
    yg_offset = 1 + yg_dim1 * 1;
    yg -= yg_offset;
    xg_dim1 = *nx;
    xg_offset = 1 + xg_dim1 * 1;
    xg -= xg_offset;

    /* Function Body */
    if (*coty > 50) {
	newref = TRUE_;
	*coty += -100;
	xdref = xco[*conum];
	ydref = yco[*conum];
	--(*conum);
    } else {
	newref = FALSE_;
    }
/* Set up a square at the reference pixel of the input */
/* image (WCSIN(1)=CRPIX1 and WCSIN(3)=CRPIX2) */
    xin[0] = wcsin[1];
    xin[1] = wcsin[1];
    xin[2] = wcsin[1] + 1.;
    xin[3] = wcsin[1] + 1.;
    yin[0] = wcsin[3];
    yin[1] = wcsin[3] + 1.;
    yin[2] = wcsin[3] + 1.;
    yin[3] = wcsin[3];
/* Transform these points onto the sky and then back out again */
/* using the target WCS - all double precision */
    for (i__ = 1; i__ <= 4; ++i__) {
	xy2rd_(&xin[i__ - 1], &yin[i__ - 1], &ra, &dec, &wcsin[1]);
	rd2xy_(&ra, &dec, &xout[i__ - 1], &yout[i__ - 1], &wcsout[1], &istat);
    }
/* Check for different reference pixel */
    if (newref) {
	xdoff = *xcen - xdref;
	ydoff = *ycen - ydref;
    } else {
	xdoff = 0.;
	ydoff = 0.;
    }
/* Now we apply the geometric distortion to the input points so that */
/* the linear transformation which we derive is appropriate after */
/* the distortion is corrected */

/* Just use LINEAR terms */
    scoty = *coty;
    if (*coty > 1) {
	*coty = 1;
    }
    for (i__ = 1; i__ <= 4; ++i__) {
	if (*coty == 3) {
	    d__1 = xin[i__ - 1] - *xcen + xdoff;
	    d__2 = yin[i__ - 1] - *ycen + ydoff;
	    x[i__ - 1] = eval3_(&d__1, &d__2, &xco[1]) - xdoff;
	    d__1 = xin[i__ - 1] - *xcen + xdoff;
	    d__2 = yin[i__ - 1] - *ycen + ydoff;
	    y[i__ - 1] = eval3_(&d__1, &d__2, &yco[1]) - ydoff;
	} else if (*coty == 4) {
	    d__1 = xin[i__ - 1] - *xcen + xdoff;
	    d__2 = yin[i__ - 1] - *ycen + ydoff;
	    x[i__ - 1] = eval4_(&d__1, &d__2, &xco[1]) - xdoff;
	    d__1 = xin[i__ - 1] - *xcen + xdoff;
	    d__2 = yin[i__ - 1] - *ycen + ydoff;
	    y[i__ - 1] = eval4_(&d__1, &d__2, &yco[1]) - ydoff;
	} else if (*coty == 5) {
	    d__1 = xin[i__ - 1] - *xcen + xdoff;
	    d__2 = yin[i__ - 1] - *ycen + ydoff;
	    x[i__ - 1] = eval5_(&d__1, &d__2, &xco[1]) - xdoff;
	    d__1 = xin[i__ - 1] - *xcen + xdoff;
	    d__2 = yin[i__ - 1] - *ycen + ydoff;
	    y[i__ - 1] = eval5_(&d__1, &d__2, &yco[1]) - ydoff;
	} else if (*coty >= 6 || *coty == 1 || *coty == 2) {
	    d__1 = xin[i__ - 1] - *xcen + xdoff;
	    d__2 = yin[i__ - 1] - *ycen + ydoff;
	    x[i__ - 1] = evaln_(&d__1, &d__2, &xco[1], coty) - xdoff;
	    d__1 = xin[i__ - 1] - *xcen + xdoff;
	    d__2 = yin[i__ - 1] - *ycen + ydoff;
	    y[i__ - 1] = evaln_(&d__1, &d__2, &yco[1], coty) - ydoff;
	} else if (*coty == -3) {
	    d__1 = xin[i__ - 1] - *xcen + xdoff;
	    d__2 = yin[i__ - 1] - *ycen + ydoff;
	    rad3_(&d__1, &d__2, &xco[1], &x[i__ - 1], &y[i__ - 1]);
	    x[i__ - 1] -= xdoff;
	    y[i__ - 1] -= ydoff;
	} else {
	    x[i__ - 1] = xin[i__ - 1] - *xcen;
	    y[i__ - 1] = yin[i__ - 1] - *ycen;
	}
    }
/* Restore order */
    *coty = scoty;
/* Now we have the inputs and outputs and can derive the linear */
/* transform between them */
/* This is now done in a general way using least squares */
/* Double precision version */
    fitlin_(xout, yout, x, y, &c__4, &x0, &y0, &a, &b, &c__, &d__, &istat);
    if (istat != 0) {
	umsput_("! Failed to determine mapping from WCS", &c__1, &c__0, &
		istat, (ftnlen)38);
	istat = 1;
	return 0;
    }
/* We change a sign here to fit in with convention later */
    b = -b;
/* And now the linear offset */
    *xt = xout[0] - a * x[0] + b * y[0];
    *yt = yout[0] - c__ * x[0] - d__ * y[0];
    *xc = a;
    *ys = b;
    *xs = c__;
    *yc = d__;
/* Before returning reset the offsets, if there are any */
    if (newref) {
	*coty += 100;
	++(*conum);
    }
    return 0;
} /* wcslin_ */

/* Subroutine */ int setwcs_(doublereal *outscl, doublereal *orient, 
	doublereal *crpix1, doublereal *crval1, doublereal *crpix2, 
	doublereal *crval2, doublereal *wcs)
{
    /* Builtin functions */
    double cos(doublereal), sin(doublereal);

    /* Local variables */
    static doublereal dsc, ror;


/* SETWCS - convert a scale and orientation, along with */
/*          reference pixel and position on sky, to a WCS */
/*          including a CD matrix. */

/* Note: OUTSCL is in arcsecs/pix, ORIENT and CRVAL1/2 */
/*       are in degrees. */

/* Richard Hook, 22nd September 1998 */

/* Full double precision version - Richard Hook,ST-ECF/STScI June 2003 */

/* Supplied: */
/* Returned: */
/* Local variables */
/* Convert to radians for orientation */
    /* Parameter adjustments */
    --wcs;

    /* Function Body */
    ror = *orient * .017453292519943278;
/* and to degrees for scale */
    dsc = *outscl / 3600.;
/* Set reference points */
    wcs[1] = *crpix1;
    wcs[3] = *crpix2;
    wcs[2] = *crval1;
    wcs[4] = *crval2;
/* and CD matrix (no skew, equal X,Y scales) */
    wcs[5] = -dsc * cos(ror);
    wcs[6] = dsc * sin(ror);
    wcs[7] = wcs[6];
    wcs[8] = -wcs[5];
    return 0;
} /* setwcs_ */

/* Subroutine */ int copy1d_(real *in, real *out, integer *n)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__;


/* Copy a 1d real array from one place to another. */

/* SINGLE PRECISION routine. */

    /* Parameter adjustments */
    --out;
    --in;

    /* Function Body */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	out[i__] = in[i__];
    }
    return 0;
} /* copy1d_ */

/* Subroutine */ int lcorn_(integer *dnx, integer *dny, integer *onx, integer 
	*ony, doublereal *xsh, doublereal *ysh, doublereal *rot, doublereal *
	scale, char *align, logical *rotfir, logical *secpar, doublereal *
	xsh2, doublereal *ysh2, doublereal *rot2, doublereal *xscale, 
	doublereal *yscale, char *shfr2, logical *rotf2, logical *usewcs, 
	doublereal *wcsin, doublereal *wcsout, integer *coty, integer *conum, 
	doublereal *xco, doublereal *yco, logical *disim, real *pxg, real *
	pyg, integer *xgdim, integer *ygdim, integer *xmin, integer *xmax, 
	integer *ymin, integer *ymax, integer *istat, ftnlen align_len, 
	ftnlen shfr2_len)
{
    /* System generated locals */
    integer pxg_dim1, pxg_offset, pyg_dim1, pyg_offset;
    doublereal d__1;

    /* Builtin functions */
    integer i_dnnt(doublereal *);

    /* Local variables */
    static doublereal xout[16], yout[16];
    static integer i__;
    extern /* Subroutine */ int drival_(doublereal *, doublereal *, integer *,
	     integer *, integer *, integer *, integer *, logical *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, char *, 
	    logical *, logical *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, char *, logical *, logical *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    doublereal *, logical *, real *, real *, integer *, integer *, 
	    doublereal *, doublereal *, ftnlen, ftnlen);
    static doublereal xma, yma, xmi, xin[16], yin[16], ymi;


/* LCORN - transform the range of input pixel coordinates covered */
/*         by an image onto the output and find the extreme values. */

/* Modified to also pass the secondary geometric parameters, */
/*  Richard Hook, STScI, July 2000 */

/* Added SECPAR parameter, Sept 2000 */

/* Added check of the border and just 5 pixel margin, October 2000 */

/* Added more general coefficient support, December 2000 */

/* Secondary geometrical parameters, added in V1.5 */
    /* Parameter adjustments */
    --wcsin;
    --wcsout;
    --yco;
    --xco;
    pyg_dim1 = *xgdim;
    pyg_offset = 1 + pyg_dim1 * 1;
    pyg -= pyg_offset;
    pxg_dim1 = *xgdim;
    pxg_offset = 1 + pxg_dim1 * 1;
    pxg -= pxg_offset;

    /* Function Body */
    xin[0] = 1.f;
    xin[1] = 1.f;
    xin[2] = (doublereal) (*dnx);
    xin[3] = (doublereal) (*dnx);
    yin[0] = 1.f;
    yin[1] = (doublereal) (*dny);
    yin[2] = (doublereal) (*dny);
    yin[3] = 1.f;
/* Transform onto output coordinate system */
    drival_(xin, yin, &c__4, dnx, dny, onx, ony, &c_false, xsh, ysh, rot, 
	    scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, yscale, 
	    shfr2, rotf2, usewcs, &wcsin[1], &wcsout[1], coty, conum, &xco[1],
	     &yco[1], disim, &pxg[pxg_offset], &pyg[pyg_offset], xgdim, ygdim,
	     xout, yout, (ftnlen)8, (ftnlen)8);
/* Calculate the extreme values */
/* Computing MAX */
    d__1 = max(xout[0],xout[1]), d__1 = max(d__1,xout[2]);
    xma = max(d__1,xout[3]);
/* Computing MAX */
    d__1 = max(yout[0],yout[1]), d__1 = max(d__1,yout[2]);
    yma = max(d__1,yout[3]);
/* Computing MIN */
    d__1 = min(xout[0],xout[1]), d__1 = min(d__1,xout[2]);
    xmi = min(d__1,xout[3]);
/* Computing MIN */
    d__1 = min(yout[0],yout[1]), d__1 = min(d__1,yout[2]);
    ymi = min(d__1,yout[3]);
/* Now check a few more points on the edges */
    for (i__ = 1; i__ <= 4; ++i__) {
	xin[i__ - 1] = (doublereal) i__ * (doublereal) (*dnx) / 5.;
	yin[i__ - 1] = 1.;
	xin[i__ + 3] = xin[i__ - 1];
	yin[i__ + 3] = (doublereal) (*dny);
	xin[i__ + 7] = 1.f;
	yin[i__ + 7] = (doublereal) i__ * (doublereal) (*dny) / 5.;
	xin[i__ + 11] = (doublereal) (*dnx);
	yin[i__ + 11] = yin[i__ + 7];
    }
    drival_(xin, yin, &c__16, dnx, dny, onx, ony, &c_false, xsh, ysh, rot, 
	    scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, yscale, 
	    shfr2, rotf2, usewcs, &wcsin[1], &wcsout[1], coty, conum, &xco[1],
	     &yco[1], disim, &pxg[pxg_offset], &pyg[pyg_offset], xgdim, ygdim,
	     xout, yout, (ftnlen)8, (ftnlen)8);
    for (i__ = 1; i__ <= 16; ++i__) {
	if (xout[i__ - 1] > xma) {
	    xma = xout[i__ - 1];
	}
	if (yout[i__ - 1] > yma) {
	    yma = yout[i__ - 1];
	}
	if (xout[i__ - 1] < xmi) {
	    xmi = xout[i__ - 1];
	}
	if (yout[i__ - 1] < ymi) {
	    ymi = yout[i__ - 1];
	}
    }
/* Calculate limits allowing for a 5 pixel margin of error */
    d__1 = xma + 5.;
    *xmax = i_dnnt(&d__1);
    d__1 = xmi - 5.;
    *xmin = i_dnnt(&d__1);
    d__1 = yma + 5.;
    *ymax = i_dnnt(&d__1);
    d__1 = ymi - 5.;
    *ymin = i_dnnt(&d__1);
    *istat = 0;
    return 0;
} /* lcorn_ */

/* Subroutine */ int drival_(doublereal *xin, doublereal *yin, integer *n, 
	integer *dnx, integer *dny, integer *onx, integer *ony, logical *reg, 
	doublereal *xsh, doublereal *ysh, doublereal *rot, doublereal *scale, 
	char *align, logical *rotfir, logical *secpar, doublereal *xsh2, 
	doublereal *ysh2, doublereal *rot2, doublereal *xscale, doublereal *
	yscale, char *shfr2, logical *rotf2, logical *usewcs, doublereal *
	wcsin, doublereal *wcsout, integer *coty, integer *conum, doublereal *
	xco, doublereal *yco, logical *disim, real *xg, real *yg, integer *
	xgdim, integer *ygdim, doublereal *xout, doublereal *yout, ftnlen 
	align_len, ftnlen shfr2_len)
{
    /* System generated locals */
    integer xg_dim1, xg_offset, yg_dim1, yg_offset, i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    integer s_cmp(char *, char *, ftnlen, ftnlen);
    double sin(doublereal), cos(doublereal);

    /* Local variables */
    static doublereal xcen, ycen, xoff, yoff;
    extern doublereal eval3_(doublereal *, doublereal *, doublereal *), 
	    eval4_(doublereal *, doublereal *, doublereal *), eval5_(
	    doublereal *, doublereal *, doublereal *);
    static integer i__;
    static doublereal x, y, xdoff, ydoff;
    extern doublereal evaln_(doublereal *, doublereal *, doublereal *, 
	    integer *);
    static doublereal xdref, ydref, costh, sinth, xcorn, ycorn, costh2, 
	    sinth2, xc, yc, xd, xf, yf;
    static integer ix, iy;
    static doublereal yd, xp, yp, xs, ys, xt, yt;
    static logical newref;
    extern /* Subroutine */ int wcslin_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    doublereal *, logical *, real *, real *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *);
    static doublereal xc2, yc2, xp2, yp2, xs2, ys2, xt2, yt2, xut, yut;
    extern /* Subroutine */ int rad3_(doublereal *, doublereal *, doublereal *
	    , doublereal *, doublereal *);


/* DRIVAL - apply the standard Drizzle transformation */
/*          from input to output pixel coordinates. */
/*          This may optional include a polynomial */
/*          distortion solution or be specified using */
/*          input and output WCS. It replaces the */
/*          logic originally in the DOBOX routine within Drizzle. */

/*  Richard Hook, ST-ECF, 5th October 1998 */

/* Modified to include the secondary geometric transformation, */
/*  Richard Hook, STScI, July 2000 */

/* Added SECPAR logical flag, Sept 2000 */

/* Added more general coefficients, December 2000 */
/* Used COTY to define the transformation and remove NONLIN */
/* Added EVAL4/5 for higher order options */

/* Added the new "REG" parameter (logical) which decides whether */
/* the input X and Y are simply at steps of 1.0 in X (starting at */
/* the value supplied) and with Y constant at the value supplied. */
/* This is a new feature for Drizzle V2.7 intended to avoid some */
/* numerical precision problems and increase efficiency. */
/*   Richard Hook, ST-ECF, February 2002 */

/* Added distortion image support (XG,YG), */
/*    Richard Hook, ST-ECF/STScI, August 2003 */

/* Added "refpix" support using higher values of COTY, */
/*    Richard Hook, ST-ECF/STScI, December 2003 */

/* Secondary geometrical parameters, added in V1.5 */
/* -- */
    /* Parameter adjustments */
    --yout;
    --xout;
    --yin;
    --xin;
    --wcsin;
    --wcsout;
    --yco;
    --xco;
    yg_dim1 = *xgdim;
    yg_offset = 1 + yg_dim1 * 1;
    yg -= yg_offset;
    xg_dim1 = *xgdim;
    xg_offset = 1 + xg_dim1 * 1;
    xg -= xg_offset;

    /* Function Body */
    if (s_cmp(align, "corner", (ftnlen)8, (ftnlen)6) == 0) {
	xcen = (doublereal) (*dnx / 2) + .5f;
	ycen = (doublereal) (*dny / 2) + .5f;
    } else {
	xcen = (doublereal) (*dnx / 2) + 1.f;
	ycen = (doublereal) (*dny / 2) + 1.f;
    }
/* Calculate some numbers to simplify things later */
    sinth = sin(*rot);
    costh = cos(*rot);
/* The shifts are in units of INPUT pixels. */
    xoff = *xsh / *scale;
    yoff = *ysh / *scale;
    xf = 1. / *scale;
    yf = 1. / *scale;
/* Some more economies */
    xs = xf * sinth;
    xc = xf * costh;
    ys = yf * sinth;
    yc = yf * costh;
/* Secondary ones */
    if (*secpar) {
	sinth2 = sin(*rot2);
	costh2 = cos(*rot2);
	xs2 = sinth2;
	xc2 = costh2;
	ys2 = sinth2;
	yc2 = costh2;
    }
    if (s_cmp(align, "corner", (ftnlen)8, (ftnlen)6) == 0) {
	xp = (doublereal) (*onx / 2) + .5;
	yp = (doublereal) (*ony / 2) + .5;
    } else {
	xp = (doublereal) (*onx / 2) + 1.;
	yp = (doublereal) (*ony / 2) + 1.;
    }
    xt = xoff + xp;
    yt = yoff + yp;
/* Set the secondary ones */
    if (*secpar) {
	xp2 = xp;
	yp2 = yp;
	xt2 = xp2 + *xsh2;
	yt2 = yp2 + *ysh2;
    }
/* Here is some new code to support the WCS option */
/* first we work out a linear transformation from input to */
/* out */
    if (*usewcs) {
	wcslin_(&wcsin[1], &wcsout[1], &xcen, &ycen, coty, conum, &xco[1], &
		yco[1], disim, &xg[xg_offset], &yg[yg_offset], xgdim, ygdim, &
		xc, &yc, &xs, &ys, &xt, &yt);
	*rotfir = TRUE_;
    }
/* Check for the presence of "refpix" additional information */
/* in the coefficients */
/* If it is, set a flag and offset again */
    if (*coty > 50) {
	newref = TRUE_;
	*coty += -100;
	xdref = xco[*conum];
	ydref = yco[*conum];
	xdoff = xcen - xdref;
	ydoff = ycen - ydref;
	--(*conum);
    } else {
	newref = FALSE_;
	xdoff = 0.f;
	ydoff = 0.f;
    }
/* We consider the case of "regular" and not-regular separately. */
/* Regular means that the X positions are spaced at intervals of */
/* one pixel (starting at XIN(1)) and the Y's are all equal to Y(1) */

/* XIN(2) and YIN(2) are used for offsets */
    if (*reg) {
/* The case of secondary parameters is also separated */
/* to avoid excessive calculations */
/* First the case with secondary parameters */
	if (*secpar) {
/* Note that there is an extra -1 here because it is added again */
/* for the first point */
	    x = *xscale * (xin[1] - 1. - xcen);
	    y = *yscale * (yin[1] + yin[2] - ycen);
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		x += *xscale;
		if (*coty == 3) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = eval3_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = eval3_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty == 4) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = eval4_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = eval4_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty == 5) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = eval5_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = eval5_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty >= 6 || *coty == 1 || *coty == 2) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = evaln_(&d__1, &d__2, &xco[1], coty) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = evaln_(&d__1, &d__2, &yco[1], coty) - ydoff;
		} else if (*coty == -3) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    rad3_(&d__1, &d__2, &xco[1], &xcorn, &ycorn);
		    xcorn -= xdoff;
		    ycorn -= ydoff;
		} else {
		    xcorn = x;
		    ycorn = y;
		}
/* If we have a distortion image we add here */
		if (*disim) {
		    ix = (integer) (x + xcen);
		    iy = (integer) (y + ycen);
		    xcorn += (doublereal) xg[ix + iy * xg_dim1];
		    ycorn += (doublereal) yg[ix + iy * yg_dim1];
		}
/* Apply the linear transform */
/* There are two ways this can be done - shift then */
/* rotate or rotate then shift */
		if (*rotfir) {
		    xut = xc * xcorn - ys * ycorn + xt - xp;
		    yut = xs * xcorn + yc * ycorn + yt - yp;
		} else {
		    xut = xc * (xcorn + *xsh) - ys * (ycorn + *ysh);
		    yut = xs * (xcorn + *xsh) + yc * (ycorn + *ysh);
		}
/* Apply the secondary transform */
		if (*rotf2) {
		    xout[i__] = xc2 * xut - ys2 * yut + xt2;
		    yout[i__] = xs2 * xut + yc2 * yut + yt2;
		} else {
		    xout[i__] = xc2 * (xut + *xsh2) - ys2 * (yut + *ysh2) + 
			    xp2;
		    yout[i__] = xs2 * (xut + *xsh2) + yc2 * (yut + *ysh2) + 
			    yp2;
		}
	    }
/* and now without secondary parameters */
	} else {
/* Note again the extra 1.0 here! */
/* In this case there are some tricks to force the result to be */
/* the same as for V1.41 - XIN(1) and YIN(2) are used for the */
/* offsets */

/* Note that XIN and YIN are used differently! */
	    x = -xcen;
	    xd = xin[1] - 1.;
	    y = yin[1] - ycen;
	    yd = yin[2];
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		x += 1.;
		if (*coty == 3) {
		    d__1 = x + xd + xdoff;
		    d__2 = y + yd + ydoff;
		    xcorn = eval3_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xd + xdoff;
		    d__2 = y + yd + ydoff;
		    ycorn = eval3_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty == 4) {
		    d__1 = x + xd + xdoff;
		    d__2 = y + yd + ydoff;
		    xcorn = eval4_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xd + xdoff;
		    d__2 = y + yd + ydoff;
		    ycorn = eval4_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty == 5) {
		    d__1 = x + xd + xdoff;
		    d__2 = y + yd + ydoff;
		    xcorn = eval5_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xd + xdoff;
		    d__2 = y + yd + ydoff;
		    ycorn = eval5_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty >= 6 || *coty == 1 || *coty == 2) {
		    d__1 = x + xd + xdoff;
		    d__2 = y + yd + ydoff;
		    xcorn = evaln_(&d__1, &d__2, &xco[1], coty) - xdoff;
		    d__1 = x + xd + xdoff;
		    d__2 = y + yd + ydoff;
		    ycorn = evaln_(&d__1, &d__2, &yco[1], coty) - ydoff;
		} else if (*coty == -3) {
		    d__1 = x + xd + xdoff;
		    d__2 = y + yd + ydoff;
		    rad3_(&d__1, &d__2, &xco[1], &xcorn, &ycorn);
		    xcorn -= xdoff;
		    ycorn -= ydoff;
		} else {
		    xcorn = x + xd;
		    ycorn = y + yd;
		}
/* If we have a distortion image we add here */
		if (*disim) {
		    ix = (integer) (x + xcen);
		    iy = (integer) (y + ycen);
		    xcorn += (doublereal) xg[ix + iy * xg_dim1];
		    ycorn += (doublereal) yg[ix + iy * yg_dim1];
		}
/* Apply the linear transform */
/* There are two ways this can be done - shift then */
/* rotate or rotate then shift */
		if (*rotfir) {
		    xout[i__] = xc * xcorn - ys * ycorn + xt;
		    yout[i__] = xs * xcorn + yc * ycorn + yt;
		} else {
		    xout[i__] = xc * (xcorn + *xsh) - ys * (ycorn + *ysh) + 
			    xp;
		    yout[i__] = xs * (xcorn + *xsh) + yc * (ycorn + *ysh) + 
			    yp;
		}
	    }
	}
    } else {
/* The case of secondary parameters is also separated */
/* to avoid excessive calculations */
/* First the case with secondary parameters */
	if (*secpar) {
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		x = *xscale * (xin[i__] - xcen);
		y = *yscale * (yin[i__] - ycen);
		if (*coty == 3) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = eval3_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = eval3_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty == 4) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = eval4_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = eval4_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty == 5) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = eval5_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = eval5_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty >= 6 || *coty == 1 || *coty == 2) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = evaln_(&d__1, &d__2, &xco[1], coty) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = evaln_(&d__1, &d__2, &yco[1], coty) - ydoff;
		} else if (*coty == -3) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    rad3_(&d__1, &d__2, &xco[1], &xcorn, &ycorn);
		    xcorn -= xdoff;
		    ycorn -= ydoff;
		} else {
		    xcorn = x;
		    ycorn = y;
		}
/* If we have a distortion image we add here */
		if (*disim) {
		    ix = (integer) (x + xcen);
		    iy = (integer) (y + ycen);
		    xcorn += (doublereal) xg[ix + iy * xg_dim1];
		    ycorn += (doublereal) yg[ix + iy * yg_dim1];
		}
/* Apply the linear transform */
/* There are two ways this can be done - shift then */
/* rotate or rotate then shift */
		if (*rotfir) {
		    xut = xc * xcorn - ys * ycorn + xt - xp;
		    yut = xs * xcorn + yc * ycorn + yt - yp;
		} else {
		    xut = xc * (xcorn + *xsh) - ys * (ycorn + *ysh);
		    yut = xs * (xcorn + *xsh) + yc * (ycorn + *ysh);
		}
/* Apply the secondary transform */
		if (*rotf2) {
		    xout[i__] = xc2 * xut - ys2 * yut + xt2;
		    yout[i__] = xs2 * xut + yc2 * yut + yt2;
		} else {
		    xout[i__] = xc2 * (xut + *xsh2) - ys2 * (yut + *ysh2) + 
			    xp2;
		    yout[i__] = xs2 * (xut + *xsh2) + yc2 * (yut + *ysh2) + 
			    yp2;
		}
	    }
/* and now without secondary parameters */
	} else {
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		x = xin[i__] - xcen;
		y = yin[i__] - ycen;
		if (*coty == 3) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = eval3_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = eval3_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty == 4) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = eval4_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = eval4_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty == 5) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = eval5_(&d__1, &d__2, &xco[1]) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = eval5_(&d__1, &d__2, &yco[1]) - ydoff;
		} else if (*coty >= 6 || *coty == 1 || *coty == 2) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    xcorn = evaln_(&d__1, &d__2, &xco[1], coty) - xdoff;
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    ycorn = evaln_(&d__1, &d__2, &yco[1], coty) - ydoff;
		} else if (*coty == -3) {
		    d__1 = x + xdoff;
		    d__2 = y + ydoff;
		    rad3_(&d__1, &d__2, &xco[1], &xcorn, &ycorn);
		    xcorn -= xdoff;
		    ycorn -= ydoff;
		} else {
		    xcorn = x;
		    ycorn = y;
		}
/* If we have a distortion image we add here */
		if (*disim) {
		    ix = (integer) (x + xcen);
		    iy = (integer) (y + ycen);
		    xcorn += (doublereal) xg[ix + iy * xg_dim1];
		    ycorn += (doublereal) yg[ix + iy * yg_dim1];
		}
/* Apply the linear transform */
/* There are two ways this can be done - shift then */
/* rotate or rotate then shift */
		if (*rotfir) {
		    xout[i__] = xc * xcorn - ys * ycorn + xt;
		    yout[i__] = xs * xcorn + yc * ycorn + yt;
		} else {
		    xout[i__] = xc * (xcorn + *xsh) - ys * (ycorn + *ysh) + 
			    xp;
		    yout[i__] = xs * (xcorn + *xsh) + yc * (ycorn + *ysh) + 
			    yp;
		}
	    }
	}
    }
/* Before returning reset the offsets, if there are any */
    if (newref) {
	*coty += 100;
	++(*conum);
    }
    return 0;
} /* drival_ */

/* Subroutine */ int dericu_(doublereal *x, doublereal *y, doublereal *co, 
	doublereal *dx, doublereal *dy)
{

/* Evaluate the derivatives of a cubic polynomial distortion with */
/* respect to X and Y. This is mainly for inverting the distortion */
/* using Newton-Raphson. */

/* Richard Hook, STScI, May 1999 */

    /* Parameter adjustments */
    --co;

    /* Function Body */
    *dx = co[2] + co[4] * 2. * *x + co[5] * *y + co[7] * 3. * *x * *x + co[8] 
	    * 2. * *x * *y + co[9] * *y * *y;
    *dy = co[3] + co[5] * *x + co[6] * 2. * *y + co[8] * *x * *x + co[9] * 2. 
	    * *x * *y + co[10] * 3. * *y * *y;
    return 0;
} /* dericu_ */

/* Subroutine */ int invecu_(doublereal *xout, doublereal *yout, doublereal *
	xco, doublereal *yco, doublereal *err, doublereal *xin, doublereal *
	yin)
{
    /* System generated locals */
    doublereal d__1, d__2;

    /* Local variables */
    extern doublereal eval3_(doublereal *, doublereal *, doublereal *);
    static doublereal d__, x, y, xo, yo;
    extern /* Subroutine */ int dericu_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *);
    static doublereal dxx, dxy, dyx, dyy;


/* Invert a cubic distortion in 2d using Newton-Raphson */

/* Supplied: */

/* Xout,Yout - double - the output position from the geometric */
/*                     distortion. */

/* Xco,Yco - double arrays - the cubic distortion coefficients */

/* Err - double - the accuracy required of the inversion (in X and Y) */

/* Returned: */

/* Xin, Yin - double - the position which, when distorted, ends up */
/*                   at Xout,Yout to accuracy Err. */

/* This method is interative and relatively slow. */

/* Richard Hook, STScI, May 1999 */

/* First guess */
    /* Parameter adjustments */
    --yco;
    --xco;

    /* Function Body */
    x = *xout * 2. - eval3_(xout, yout, &xco[1]);
    y = *yout * 2. - eval3_(xout, yout, &yco[1]);
    while(TRUE_) {
/* Distort current guesses */
	xo = eval3_(&x, &y, &xco[1]);
	yo = eval3_(&x, &y, &yco[1]);
/* Check for error criterion */
	if ((d__1 = *xout - xo, abs(d__1)) < *err && (d__2 = *yout - yo, abs(
		d__2)) < *err) {
	    goto L999;
	}
/* Calculate derivatives - there are four of these */
	dericu_(&x, &y, &xco[1], &dxx, &dxy);
	dericu_(&x, &y, &yco[1], &dyx, &dyy);
/* Work out the determinant */
	d__ = dxx * dyy - dyx * dxy;
/* Improve guess with Newton-Raphson */
	x += ((*xout - xo) * dyy - (*yout - yo) * dxy) / d__;
	y += ((*yout - yo) * dxx - (*xout - xo) * dyx) / d__;
    }
L999:
    *xin = x;
    *yin = y;
    return 0;
} /* invecu_ */

doublereal over_(integer *i__, integer *j, doublereal *xmin, doublereal *xmax,
	 doublereal *ymin, doublereal *ymax)
{
    /* System generated locals */
    doublereal ret_val, d__1, d__2, d__3, d__4;

    /* Local variables */
    static doublereal dx, dy;


/* OVER - calculate overlap between an arbitrary rectangle, aligned */
/*        with the axes, and a pixel. */

/* This is a simplified version of the BOXER code. */

/* Richard Hook, 6th May 1998 */

/* Computing MIN */
    d__1 = *xmax, d__2 = (doublereal) (*i__) + .5;
/* Computing MAX */
    d__3 = *xmin, d__4 = (doublereal) (*i__) - .5;
    dx = min(d__1,d__2) - max(d__3,d__4);
/* Computing MIN */
    d__1 = *ymax, d__2 = (doublereal) (*j) + .5;
/* Computing MAX */
    d__3 = *ymin, d__4 = (doublereal) (*j) - .5;
    dy = min(d__1,d__2) - max(d__3,d__4);
    if (dx > 0. && dy > 0.) {
	ret_val = dx * dy;
    } else {
	ret_val = 0.;
    }
    return ret_val;
} /* over_ */

/* Subroutine */ int copy1i_(integer *in, integer *out, integer *n)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__;


/* Copy a 1d integer array from one place to another. */

    /* Parameter adjustments */
    --out;
    --in;

    /* Function Body */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	out[i__] = in[i__];
    }
    return 0;
} /* copy1i_ */

/* Subroutine */ int upcon_(integer *ncon, integer *ii, integer *jj, integer *
	oldcon, integer *newcon, integer *done, integer *lx, integer *ly, 
	integer *intab, integer *nen, integer *maxim, integer *maxen, integer 
	*uniqid, integer *istat)
{
    /* System generated locals */
    integer intab_dim1, intab_offset, ncon_dim1, ncon_offset, done_dim1, 
	    done_offset, i__1;

    /* Builtin functions */
    integer s_wsfi(icilist *), do_fio(integer *, char *, ftnlen), e_wsfi(void)
	    ;

    /* Local variables */
    static integer icon, k;
    extern logical match_(integer *, integer *, integer *);
    static char chars[80];
    static integer newma[100];
    extern /* Subroutine */ int csort_(integer *, integer *);
    static integer nn;
    extern /* Subroutine */ int umsput_(char *, integer *, integer *, integer 
	    *, ftnlen);

    /* Fortran I/O blocks */
    static icilist io___148 = { 0, chars, 0, "('--New context #',I5,        "
	    "                              ' | ',I4,' images: ',             "
	    "                                   5I7)", 80, 1 };
    static icilist io___149 = { 0, chars, 0, "('--New context #',I5,        "
	    "                              ' | ',I4,' images: ',             "
	    "                                   5I7,'...')", 80, 1 };



/* Update the context image */

/* This routine is called in the heart of Drizzle when the */
/* context image needs to be updated. */

/* October 2000 (from the EIS version) */
/* Added "bitmask" style context (16 bit), January 2001 */
/* Converted to 32bit integers */

/* Look up the current context value */
    /* Parameter adjustments */
    done_dim1 = *lx;
    done_offset = 1 + done_dim1 * 1;
    done -= done_offset;
    ncon_dim1 = *lx;
    ncon_offset = 1 + ncon_dim1 * 1;
    ncon -= ncon_offset;
    intab_dim1 = *maxim;
    intab_offset = 1 + intab_dim1 * 1;
    intab -= intab_offset;

    /* Function Body */
    icon = ncon[*ii + *jj * ncon_dim1];
/* If it is the same as the last one we don't need to */
/* go further */
    if (icon == *oldcon) {
	ncon[*ii + *jj * ncon_dim1] = *newcon;
	goto L88;
    }
/* Combine with the new one */
    if (icon == 0) {
	nn = 0;
	newma[0] = 1;
	newma[1] = *uniqid;
    } else {
	nn = intab[icon * intab_dim1 + 1];
/* Check for whether this image is already on this pixel */
	i__1 = nn + 2;
	for (k = 3; k <= i__1; ++k) {
	    if (*uniqid == intab[k + icon * intab_dim1]) {
		goto L88;
	    }
	}
/* If not create the new context by adding the current image */
	newma[0] = nn + 1;
	i__1 = nn + 1;
	for (k = 2; k <= i__1; ++k) {
	    newma[k - 1] = intab[k + 1 + icon * intab_dim1];
	}
/* Check for too many images at a given context */
	if (nn > *maxim - 3) {
	    umsput_("! Too many images - context table overloaded", &c__1, &
		    c__0, istat, (ftnlen)44);
	    *istat = 1;
	    return 0;
	}
	newma[nn + 1] = *uniqid;
    }
/* Before matching sort the context array */
    if (nn > 0) {
	i__1 = nn + 1;
	csort_(&i__1, &newma[1]);
    }
/* See whether we have had this context before */
    for (k = *nen; k >= 1; --k) {
	if (match_(newma, &intab[k * intab_dim1 + 1], maxim)) {
	    ncon[*ii + *jj * ncon_dim1] = k;
	    goto L88;
	}
    }
/* No context match found - make a new one */
    ++(*nen);
/* Check for full table */
    if (*nen == *maxen) {
	umsput_("! Context table full", &c__1, &c__0, istat, (ftnlen)20);
	*istat = 1;
	return 0;
    }
    ncon[*ii + *jj * ncon_dim1] = *nen;
    intab[*nen * intab_dim1 + 1] = newma[0];
    intab[*nen * intab_dim1 + 2] = 0;
    i__1 = nn + 3;
    for (k = 3; k <= i__1; ++k) {
	intab[k + *nen * intab_dim1] = newma[k - 2];
    }
/* Tell the user about this new context */
    if (nn <= 4) {
	s_wsfi(&io___148);
	do_fio(&c__1, (char *)&(*nen), (ftnlen)sizeof(integer));
	i__1 = nn + 2;
	for (k = 1; k <= i__1; ++k) {
	    do_fio(&c__1, (char *)&newma[k - 1], (ftnlen)sizeof(integer));
	}
	e_wsfi();
    } else {
	s_wsfi(&io___149);
	do_fio(&c__1, (char *)&(*nen), (ftnlen)sizeof(integer));
	for (k = 1; k <= 6; ++k) {
	    do_fio(&c__1, (char *)&newma[k - 1], (ftnlen)sizeof(integer));
	}
	e_wsfi();
    }
    if (newma[0] == 1) {
	*(unsigned char *)&chars[33] = ' ';
    }
    umsput_(chars, &c__1, &c__0, istat, (ftnlen)80);
L88:
/* Save the old values for quick comparison */
    *oldcon = icon;
    *newcon = ncon[*ii + *jj * ncon_dim1];
/* Lastly we update the counter */
    if (*oldcon != *newcon) {
	if (*oldcon > 0) {
	    --intab[*oldcon * intab_dim1 + 2];
	}
	++intab[*newcon * intab_dim1 + 2];
    }
/* Note that we have been here */
    done[*ii + *jj * done_dim1] = 1;
    *istat = 0;
    return 0;
} /* upcon_ */

/* Subroutine */ int lenstr_(char *string, integer *i1, integer *i2, ftnlen 
	string_len)
{
    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    integer i_len(char *, ftnlen);

    /* Local variables */
    static integer i__;
    static logical in;


/* Find the start and end of a string */

    in = FALSE_;
    i__1 = i_len(string, string_len);
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (*(unsigned char *)&string[i__ - 1] != ' ' && ! in) {
	    *i1 = i__;
	    in = TRUE_;
	}
	if (*(unsigned char *)&string[i__ - 1] == ' ' && in) {
	    *i2 = i__ - 1;
	    goto L99;
	}
    }
L99:
    return 0;
} /* lenstr_ */

/* Subroutine */ int setimi_(integer *a, integer *nx, integer *ny, integer *v)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;


/* Set a 2d integer image to a constant value */

    /* Parameter adjustments */
    a_dim1 = *nx;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */
    i__1 = *ny;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *nx;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = *v;
	}
    }
    return 0;
} /* setimi_ */

logical match_(integer *a, integer *b, integer *n)
{
    /* System generated locals */
    integer i__1;
    logical ret_val;

    /* Local variables */
    static integer i__, an, bn;


/* Match up a context against a context table */
/* Note that after drizzle V0.40 (EIS) the context */
/* table itself has an extra second column for the */
/* counter. */

    /* Parameter adjustments */
    --b;
    --a;

    /* Function Body */
    an = a[1];
    bn = b[1];
    if (an != bn) {
	ret_val = FALSE_;
	return ret_val;
    } else {
	ret_val = TRUE_;
	i__1 = an + 1;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    if (a[i__] != b[i__ + 1]) {
		ret_val = FALSE_;
		return ret_val;
	    }
	}
    }
    return ret_val;
} /* match_ */

/* Subroutine */ int csort_(integer *n, integer *arr)
{
    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    /* Subroutine */ int s_paus(char *, ftnlen);

    /* Local variables */
    static integer temp, a, i__, j, k, l, ir, istack[50], jstack;


/* This routine is modified from the Numerical Recipes one */
/* to work on INTEGER arrays. */

/* It sorts an array of integers in place. */

    /* Parameter adjustments */
    --arr;

    /* Function Body */
    jstack = 0;
    l = 1;
    ir = *n;
L1:
    if (ir - l < 7) {
	i__1 = ir;
	for (j = l + 1; j <= i__1; ++j) {
	    a = arr[j];
	    for (i__ = j - 1; i__ >= 1; --i__) {
		if (arr[i__] <= a) {
		    goto L2;
		}
		arr[i__ + 1] = arr[i__];
/* L11: */
	    }
	    i__ = 0;
L2:
	    arr[i__ + 1] = a;
/* L12: */
	}
	if (jstack == 0) {
	    return 0;
	}
	ir = istack[jstack - 1];
	l = istack[jstack - 2];
	jstack += -2;
    } else {
	k = (l + ir) / 2;
	temp = arr[k];
	arr[k] = arr[l + 1];
	arr[l + 1] = temp;
	if (arr[l + 1] > arr[ir]) {
	    temp = arr[l + 1];
	    arr[l + 1] = arr[ir];
	    arr[ir] = temp;
	}
	if (arr[l] > arr[ir]) {
	    temp = arr[l];
	    arr[l] = arr[ir];
	    arr[ir] = temp;
	}
	if (arr[l + 1] > arr[l]) {
	    temp = arr[l + 1];
	    arr[l + 1] = arr[l];
	    arr[l] = temp;
	}
	i__ = l + 1;
	j = ir;
	a = arr[l];
L3:
	++i__;
	if (arr[i__] < a) {
	    goto L3;
	}
L4:
	--j;
	if (arr[j] > a) {
	    goto L4;
	}
	if (j < i__) {
	    goto L5;
	}
	temp = arr[i__];
	arr[i__] = arr[j];
	arr[j] = temp;
	goto L3;
L5:
	arr[l] = arr[j];
	arr[j] = a;
	jstack += 2;
	if (jstack > 50) {
	    s_paus("NSTACK too small in sort", (ftnlen)24);
	}
	if (ir - i__ + 1 >= j - l) {
	    istack[jstack - 1] = ir;
	    istack[jstack - 2] = i__;
	    ir = j - 1;
	} else {
	    istack[jstack - 1] = j - 1;
	    istack[jstack - 2] = l;
	    l = i__;
	}
    }
    goto L1;
} /* csort_ */

/* Subroutine */ int gtglco_(char *contab, integer *intab, integer *maxim, 
	integer *maxen, integer *nen, integer *istat, ftnlen contab_len)
{
    /* System generated locals */
    integer intab_dim1, intab_offset, i__1, i__2;
    olist o__1;
    cllist cl__1;

    /* Builtin functions */
    integer f_open(olist *), s_rsfe(cilist *), do_fio(integer *, char *, 
	    ftnlen), e_rsfe(void), s_cmp(char *, char *, ftnlen, ftnlen), 
	    s_rsli(icilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_rsli(void), f_clos(cllist *);

    /* Local variables */
    static integer nval, i__, j, k, cn;
    static char buffer[1024];

    /* Fortran I/O blocks */
    static cilist io___168 = { 0, 11, 1, "(A)", 0 };
    static icilist io___170 = { 1, buffer, 1, 0, 1024, 1 };
    static icilist io___173 = { 1, buffer, 1, 0, 1024, 1 };



/* Read a context index table from global context file */

/* Richard Hook, November 1997 */
/* Modified to handle missing context values, March 1999 */

/* Try to open the context file */
    /* Parameter adjustments */
    intab_dim1 = *maxim;
    intab_offset = 1 + intab_dim1 * 1;
    intab -= intab_offset;

    /* Function Body */
    o__1.oerr = 1;
    o__1.ounit = 11;
    o__1.ofnmlen = 80;
    o__1.ofnm = contab;
    o__1.orl = 0;
    o__1.osta = "OLD";
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    *istat = f_open(&o__1);
    if (*istat != 0) {
	return 0;
    }
/* Initialise the context table */
    i__1 = *maxen;
    for (j = 1; j <= i__1; ++j) {
	intab[j * intab_dim1 + 1] = 1;
	i__2 = *maxim;
	for (i__ = 2; i__ <= i__2; ++i__) {
	    intab[i__ + j * intab_dim1] = 0;
	}
    }
/* Work through the lines skipping comments and blank ones */
    while(TRUE_) {
	i__1 = s_rsfe(&io___168);
	if (i__1 != 0) {
	    goto L99;
	}
	i__1 = do_fio(&c__1, buffer, (ftnlen)1024);
	if (i__1 != 0) {
	    goto L99;
	}
	i__1 = e_rsfe();
	if (i__1 != 0) {
	    goto L99;
	}
	if (*(unsigned char *)buffer != '#' && s_cmp(buffer, " ", (ftnlen)
		1024, (ftnlen)1) != 0) {
	    *istat = s_rsli(&io___170);
	    if (*istat != 0) {
		goto L100008;
	    }
	    *istat = do_lio(&c__3, &c__1, (char *)&cn, (ftnlen)sizeof(integer)
		    );
	    if (*istat != 0) {
		goto L100008;
	    }
	    *istat = do_lio(&c__3, &c__1, (char *)&nval, (ftnlen)sizeof(
		    integer));
	    if (*istat != 0) {
		goto L100008;
	    }
	    *istat = e_rsli();
L100008:
	    if (*istat != 0 || nval > 99 || cn > *maxen) {
		goto L100;
	    }
/* Check for empty contexts */
	    if (nval > 0) {
		*istat = s_rsli(&io___173);
		if (*istat != 0) {
		    goto L100009;
		}
		*istat = do_lio(&c__3, &c__1, (char *)&cn, (ftnlen)sizeof(
			integer));
		if (*istat != 0) {
		    goto L100009;
		}
		i__1 = nval + 2;
		for (k = 1; k <= i__1; ++k) {
		    *istat = do_lio(&c__3, &c__1, (char *)&intab[k + cn * 
			    intab_dim1], (ftnlen)sizeof(integer));
		    if (*istat != 0) {
			goto L100009;
		    }
		}
		*istat = e_rsli();
L100009:
		if (*istat != 0) {
		    goto L100;
		}
	    }
	}
    }
L99:
    *istat = 0;
L100:
    cl__1.cerr = 0;
    cl__1.cunit = 11;
    cl__1.csta = 0;
    f_clos(&cl__1);
    *nen = cn;
    return 0;
} /* gtglco_ */

/* Subroutine */ int ptglco_(char *contab, integer *intab, integer *maxim, 
	integer *maxen, integer *nen, integer *istat, ftnlen contab_len)
{
    /* System generated locals */
    integer intab_dim1, intab_offset, i__1, i__2;
    olist o__1;
    cllist cl__1;

    /* Builtin functions */
    integer f_open(olist *), s_wsfe(cilist *), e_wsfe(void), do_fio(integer *,
	     char *, ftnlen), f_clos(cllist *);

    /* Local variables */
    static integer i__, k;

    /* Fortran I/O blocks */
    static cilist io___175 = { 0, 12, 0, "('# GLOBAL context table')", 0 };
    static cilist io___176 = { 0, 12, 0, "('#')", 0 };
    static cilist io___177 = { 0, 12, 0, "('# Context | Nima | Npix | Unique"
	    " image ids...')", 0 };
    static cilist io___179 = { 0, 12, 0, "(I7,I6,I10,3X,200I7)", 0 };



/* Write a context index table to global context file */

/* Richard Hook, November 1997 */
/* Modified not to write empty contexts, March 1999 */

/* Open the global context file with write access - this will work */
/* whether or not it exists */
    /* Parameter adjustments */
    intab_dim1 = *maxim;
    intab_offset = 1 + intab_dim1 * 1;
    intab -= intab_offset;

    /* Function Body */
    o__1.oerr = 1;
    o__1.ounit = 12;
    o__1.ofnmlen = 80;
    o__1.ofnm = contab;
    o__1.orl = 0;
    o__1.osta = 0;
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    *istat = f_open(&o__1);
    if (*istat != 0) {
	return 0;
    }
/* Write a header of sorts */
    s_wsfe(&io___175);
    e_wsfe();
    s_wsfe(&io___176);
    e_wsfe();
    s_wsfe(&io___177);
    e_wsfe();
    i__1 = *nen;
    for (k = 1; k <= i__1; ++k) {
	if (intab[k * intab_dim1 + 2] != 0) {
	    s_wsfe(&io___179);
	    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
	    i__2 = intab[k * intab_dim1 + 1] + 2;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		do_fio(&c__1, (char *)&intab[i__ + k * intab_dim1], (ftnlen)
			sizeof(integer));
	    }
	    e_wsfe();
	}
    }
    cl__1.cerr = 0;
    cl__1.cunit = 12;
    cl__1.csta = 0;
    f_clos(&cl__1);
    return 0;
} /* ptglco_ */

/* Subroutine */ int bfill_(integer *lun, char *buffer, integer *istat, 
	ftnlen buffer_len)
{
    /* Builtin functions */
    /* Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen);
    integer s_cmp(char *, char *, ftnlen, ftnlen), i_len(char *, ftnlen);

    /* Local variables */
    static char line[256];
    extern /* Subroutine */ int ends_(char *, integer *, integer *, ftnlen);
    static integer i1, i2, is;
    extern /* Subroutine */ int ufglin_(integer *, char *, integer *, ftnlen);


/* Read free-format numbers from an open text file and */
/* concatenate them into a character string buffer. */

/* This is mostly to facilitate free-format numerical */
/* reads from text files. */

/* Richard Hook, ST-ECF, December 2000 */

    is = 1;
    s_copy(buffer, " ", buffer_len, (ftnlen)1);
    while(TRUE_) {
	ufglin_(lun, line, istat, (ftnlen)256);
	if (*istat != 0) {
	    goto L88;
	}
	if (s_cmp(line, " ", (ftnlen)256, (ftnlen)1) != 0 && *(unsigned char *
		)line != '#') {
	    ends_(line, &i1, &i2, (ftnlen)256);
	    if (is + i2 - i1 > i_len(buffer, buffer_len)) {
		*istat = 1;
		return 0;
	    } else {
		s_copy(buffer + (is - 1), line + (i1 - 1), is + i2 - i1 - (is 
			- 1), i2 - (i1 - 1));
		is = is + i2 - i1 + 2;
	    }
	}
    }
L88:
    *istat = 0;
    return 0;
} /* bfill_ */

/* Subroutine */ int ends_(char *string, integer *i1, integer *i2, ftnlen 
	string_len)
{
    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    integer i_len(char *, ftnlen);

    /* Local variables */
    static integer i__;


/* Find the start and end of a string */
/* This differs from LENSTR in that internal whitespace */
/* is retained rather than just the first non-white section */
/* being extracted. */

    *i1 = 1;
    *i2 = 1;
    i__1 = i_len(string, string_len);
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (*(unsigned char *)&string[i__ - 1] != ' ') {
	    *i1 = i__;
	    goto L88;
	}
    }
L88:
    for (i__ = i_len(string, string_len); i__ >= 1; --i__) {
	if (*(unsigned char *)&string[i__ - 1] != ' ') {
	    *i2 = i__;
	    goto L99;
	}
    }
L99:
    return 0;
} /* ends_ */

/* Subroutine */ int set1i_(integer *a, integer *n, integer *v)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__;


/* Set a 1d array to a value */

    /* Parameter adjustments */
    --a;

    /* Function Body */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	a[i__] = *v;
    }
    return 0;
} /* set1i_ */

/* Subroutine */ int fitlin_(doublereal *xo, doublereal *yo, doublereal *x, 
	doublereal *y, integer *n, doublereal *x0, doublereal *y0, doublereal 
	*a, doublereal *b, doublereal *c__, doublereal *d__, integer *istat)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Local variables */
    static doublereal xorg, yorg;
    static integer i__, j;
    static doublereal sigxo, sigyo, xoorg, yoorg;
    extern /* Subroutine */ int matinv_(doublereal *, integer *, doublereal *)
	    ;
    static doublereal sigxox, sigxoy, sigyox, sigyoy;
    extern /* Subroutine */ int umsput_(char *, integer *, integer *, integer 
	    *, ftnlen);
    static doublereal det, mat[100]	/* was [10][10] */;


/* Fit a linear transformation (with six parameters, equivalent */
/* to the linear WCS) to two sets of points. */

/* This uses the standard least-squares linear equations method. */

/* Richard Hook, ST-ECF, January 31st 2001 */

/* Fully double precision version, November 2002 */

/* Initialize the matrix (3x3) */
    /* Parameter adjustments */
    --y;
    --x;
    --yo;
    --xo;

    /* Function Body */
    for (j = 1; j <= 3; ++j) {
	for (i__ = 1; i__ <= 3; ++i__) {
	    mat[i__ + j * 10 - 11] = 0.;
	}
    }
/* Also initialise the vectors */
    sigxox = 0.;
    sigxoy = 0.;
    sigxo = 0.;
    sigyox = 0.;
    sigyoy = 0.;
    sigyo = 0.;
/* Take off an offset */
    xorg = x[1];
    yorg = y[1];
    xoorg = xo[1];
    yoorg = yo[1];
/* Setup the normal equations */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__] - xorg;
	mat[0] += d__1 * d__1;
	mat[10] += (x[i__] - xorg) * (y[i__] - yorg);
	mat[20] += x[i__] - xorg;
/* Computing 2nd power */
	d__1 = y[i__] - yorg;
	mat[11] += d__1 * d__1;
	mat[21] += y[i__] - yorg;
	sigxox += (xo[i__] - xoorg) * (x[i__] - xorg);
	sigxoy += (xo[i__] - xoorg) * (y[i__] - yorg);
	sigxo += xo[i__] - xoorg;
	sigyox += (yo[i__] - yoorg) * (x[i__] - xorg);
	sigyoy += (yo[i__] - yoorg) * (y[i__] - yorg);
	sigyo += yo[i__] - yoorg;
    }
/* Use symmetry (the matrix is diagonal) */
    mat[22] = (doublereal) (*n);
    mat[1] = mat[10];
    mat[2] = mat[20];
    mat[12] = mat[21];
/* Invert the matrix (we check it isn't singular) */
    matinv_(mat, &c__3, &det);
    if (det == 0.) {
	umsput_("! Linear transformation matrix is singular", &c__1, &c__0, 
		istat, (ftnlen)42);
	*istat = 1;
	return 0;
    }
/* Multiply the inverse by the vector */
    *a = sigxox * mat[0] + sigxoy * mat[10] + sigxo * mat[20];
    *b = sigxox * mat[1] + sigxoy * mat[11] + sigxo * mat[21];
    *x0 = sigxox * mat[2] + sigxoy * mat[12] + sigxo * mat[22];
    *c__ = sigyox * mat[0] + sigyoy * mat[10] + sigyo * mat[20];
    *d__ = sigyox * mat[1] + sigyoy * mat[11] + sigyo * mat[21];
    *y0 = sigyox * mat[2] + sigyoy * mat[12] + sigyo * mat[22];
/* Note that X0 and Y0 haven't been corrected for the offsets */
/* Normally they are not used */
    *istat = 0;
    return 0;
} /* fitlin_ */

/* SUBROUTINE MATINV.F */

/* SOURCE */
/*   BEVINGTON, PAGES 302-303. */

/* PURPOSE */
/*   INVERT A SYMMETRIC MATRIX AND CALCULATE ITS DETERMINANT */

/* USAGE */
/*   CALL MATINV (ARRAY, NORDER, DET) */

/* DESCRIPTION OF PARAMETERS */
/*   ARRAY  - INPUT MATRIX WHICH IS REPLACED BY ITS INVERSE */
/*   NORDER - DEGREE OF MATRIX (ORDER OF DETERMINANT) */
/*   DET    - DETERMINANT OF INPUT MATRIX */

/* SUBROUTINES AND FUNCTION SUBPROGRAMS REQUIRED */
/*   NONE */

/* COMMENT */
/*   DIMENSION STATEMENT VALID FOR NORDER UP TO 10 */

/* This version all double precision. */

/* Subroutine */ int matinv_(doublereal *array, integer *norder, doublereal *
	det)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal d__1;

    /* Local variables */
    static doublereal amax, save;
    static integer i__, j, k, l, ik[10], jk[10];



    /* Parameter adjustments */
    array -= 11;

    /* Function Body */
/* L10: */
    *det = 1.;
/* L11: */
    i__1 = *norder;
    for (k = 1; k <= i__1; ++k) {

/* FIND LARGEST ELEMENT ARRAY(I,J) IN REST OF MATRIX */

	amax = 0.;
L21:
	i__2 = *norder;
	for (i__ = k; i__ <= i__2; ++i__) {
	    i__3 = *norder;
	    for (j = k; j <= i__3; ++j) {
/* L23: */
		if (abs(amax) - (d__1 = array[i__ + j * 10], abs(d__1)) <= 0.)
			 {
		    goto L24;
		} else {
		    goto L30;
		}
L24:
		amax = array[i__ + j * 10];
		ik[k - 1] = i__;
		jk[k - 1] = j;
L30:
		;
	    }
	}

/* INTERCHANGE ROWS AND COLUMNS TO PUT AMAX IN ARRAY(K,K) */

/* L31: */
	if (amax != 0.) {
	    goto L41;
	} else {
	    goto L32;
	}
L32:
	*det = 0.;
	goto L140;
L41:
	i__ = ik[k - 1];
	if ((i__3 = i__ - k) < 0) {
	    goto L21;
	} else if (i__3 == 0) {
	    goto L51;
	} else {
	    goto L43;
	}
L43:
	i__3 = *norder;
	for (j = 1; j <= i__3; ++j) {
	    save = array[k + j * 10];
	    array[k + j * 10] = array[i__ + j * 10];
/* L50: */
	    array[i__ + j * 10] = -save;
	}
L51:
	j = jk[k - 1];
	if ((i__3 = j - k) < 0) {
	    goto L21;
	} else if (i__3 == 0) {
	    goto L61;
	} else {
	    goto L53;
	}
L53:
	i__3 = *norder;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    save = array[i__ + k * 10];
	    array[i__ + k * 10] = array[i__ + j * 10];
/* L60: */
	    array[i__ + j * 10] = -save;
	}

/* ACCUMULATE ELEMENTS OF INVERSE MATRIX */

L61:
	i__3 = *norder;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    if (i__ - k != 0) {
		goto L63;
	    } else {
		goto L70;
	    }
L63:
	    array[i__ + k * 10] = -array[i__ + k * 10] / amax;
L70:
	    ;
	}
/* L71: */
	i__3 = *norder;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    i__2 = *norder;
	    for (j = 1; j <= i__2; ++j) {
		if (i__ - k != 0) {
		    goto L74;
		} else {
		    goto L80;
		}
L74:
		if (j - k != 0) {
		    goto L75;
		} else {
		    goto L80;
		}
L75:
		array[i__ + j * 10] += array[i__ + k * 10] * array[k + j * 10]
			;
L80:
		;
	    }
	}
/* L81: */
	i__2 = *norder;
	for (j = 1; j <= i__2; ++j) {
	    if (j - k != 0) {
		goto L83;
	    } else {
		goto L90;
	    }
L83:
	    array[k + j * 10] /= amax;
L90:
	    ;
	}
	array[k + k * 10] = 1.f / amax;
/* L100: */
	*det *= amax;
    }

/* RESTORE ORDERING OF MATRIX */

/* L101: */
    i__1 = *norder;
    for (l = 1; l <= i__1; ++l) {
	k = *norder - l + 1;
	j = ik[k - 1];
	if (j - k <= 0) {
	    goto L111;
	} else {
	    goto L105;
	}
L105:
	i__2 = *norder;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    save = array[i__ + k * 10];
	    array[i__ + k * 10] = -array[i__ + j * 10];
/* L110: */
	    array[i__ + j * 10] = save;
	}
L111:
	i__ = jk[k - 1];
	if (i__ - k <= 0) {
	    goto L130;
	} else {
	    goto L113;
	}
L113:
	i__2 = *norder;
	for (j = 1; j <= i__2; ++j) {
	    save = array[k + j * 10];
	    array[k + j * 10] = -array[i__ + j * 10];
/* L120: */
	    array[i__ + j * 10] = save;
	}
L130:
	;
    }
L140:
    return 0;
} /* matinv_ */

/* Subroutine */ int dobox_(real *data, real *wei, real *ndat, real *ncou, 
	integer *ncon, integer *done, integer *dnx, integer *dny, integer *ny,
	 integer *ystart, integer *xmin, integer *xmax, integer *ymin, 
	integer *ymax, logical *noover, char *kernel, doublereal *xi, 
	doublereal *xo, doublereal *yi, doublereal *yo, doublereal *xib, 
	doublereal *xob, doublereal *yib, doublereal *yob, integer *onx, 
	integer *ony, integer *coty, integer *conum, doublereal *xco, 
	doublereal *yco, logical *disim, real *pxg, real *pyg, integer *xgdim,
	 integer *ygdim, real *wtscl, char *align, logical *incps, real *
	expin, doublereal *pfract, doublereal *scale, doublereal *rot, 
	doublereal *xsh, doublereal *ysh, doublereal *wcs, doublereal *wcsout,
	 logical *rotfir, logical *secpar, doublereal *xsh2, doublereal *ysh2,
	 doublereal *rot2, doublereal *xscale, doublereal *yscale, char *
	shfr2, logical *rotf2, logical *con, logical *bitcon, integer *intab, 
	integer *maxim, integer *maxen, integer *nen, integer *uniqid, 
	logical *update, logical *usewei, logical *usewcs, integer *istat, 
	integer *nmiss, integer *nskip, ftnlen kernel_len, ftnlen align_len, 
	ftnlen shfr2_len)
{
    /* System generated locals */
    integer data_dim1, data_offset, wei_dim1, wei_offset, ndat_dim1, 
	    ndat_offset, ncou_dim1, ncou_offset, pxg_dim1, pxg_offset, 
	    pyg_dim1, pyg_offset, xi_dim1, xi_offset, yi_dim1, yi_offset, 
	    xo_dim1, xo_offset, yo_dim1, yo_offset, ncon_dim1, ncon_offset, 
	    done_dim1, done_offset, intab_dim1, intab_offset, i__1, i__2, 
	    i__3, i__4;
    real r__1;
    doublereal d__1, d__2, d__3, d__4;
    icilist ici__1;

    /* Builtin functions */
    integer pow_ii(integer *, integer *), s_cmp(char *, char *, ftnlen, 
	    ftnlen);
    double sqrt(doublereal);
    integer s_rsfi(icilist *), do_fio(integer *, char *, ftnlen), e_rsfi(void)
	    , i_dnnt(doublereal *);
    double exp(doublereal);

    /* Local variables */
    static doublereal jaco, xcen, ycen;
    extern /* Subroutine */ int mulc_(real *, integer *, integer *, real *);
    static integer nhit;
    extern doublereal over_(integer *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *);
    static doublereal xout[4], yout[4];
    static real d__;
    static integer i__, j;
    static real w;
    static doublereal y, ofrac, scall, scdis, dover;
    extern /* Subroutine */ int boxer_(integer *, integer *, doublereal *, 
	    doublereal *, doublereal *), upcon_(integer *, integer *, integer 
	    *, integer *, integer *, integer *, integer *, integer *, integer 
	    *, integer *, integer *, integer *, integer *, integer *);
    static doublereal r2;
    static real s2;
    static integer x1, x2;
    static doublereal ac;
    static real dd;
    static doublereal dh;
    static integer ii, jj;
    static doublereal es;
    static real vc;
    static integer bv;
    static doublereal dx, dy;
    static integer np;
    static doublereal xf, yf;
    static integer ix, lx, ly, iy;
    extern /* Subroutine */ int filalu_(integer *, integer *, real *, real *);
    static doublereal xx, yy;
    static integer oldcon, lanord;
    extern /* Subroutine */ int drival_(doublereal *, doublereal *, integer *,
	     integer *, integer *, integer *, integer *, logical *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, char *, 
	    logical *, logical *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, char *, logical *, logical *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    doublereal *, logical *, real *, real *, integer *, integer *, 
	    doublereal *, doublereal *, ftnlen, ftnlen);
    static integer newcon;
    extern /* Subroutine */ int chover_(doublereal *, integer *, integer *, 
	    integer *, integer *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, char *, logical *, logical *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, char *, logical *, logical *, doublereal *, 
	    doublereal *, integer *, integer *, doublereal *, doublereal *, 
	    logical *, real *, real *, integer *, integer *, doublereal *, 
	    integer *, integer *, ftnlen, ftnlen);
    static real lanlut[512];
    static doublereal pfo;
    static integer nxa, nya, nxi, nyi;
    static doublereal xin[4], yin[4], xxi, xxa, yyi, yya;
    static real dow, sdp;
    static doublereal pfo2, efac;


/* This module does the actual mapping of input flux to output images using */
/* "boxer", a code written by Bill Sparks for FOC geometric */
/* distortion correction, rather than the "drizzling" approximation. */

/* This works by calculating the positions of the four corners of */
/* a quadrilateral on the output grid corresponding to the corners */
/* of the input pixel and then working out exactly how much of each */
/* pixel in the output is covered, or not. */

/* In V1.6 this was simplified to use the DRIVAL routine and also */
/* to include some limited multi-kernel support. */

/* Some things are still single */
/* Context related things */
/* Space for Lanczos-style look-up-tables */
/* Number of sigma to be included for gaussians */
/* Secondary geometrical parameters, added in V1.5 */
/* -- */
/* Some initial settings - note that the reference pixel */
/* position is determined by the value of ALIGN */
    /* Parameter adjustments */
    --yob;
    --yib;
    --xob;
    --xib;
    yo_dim1 = *dnx;
    yo_offset = 1 + yo_dim1 * 1;
    yo -= yo_offset;
    yi_dim1 = *dnx;
    yi_offset = 1 + yi_dim1 * 1;
    yi -= yi_offset;
    xo_dim1 = *dnx;
    xo_offset = 1 + xo_dim1 * 1;
    xo -= xo_offset;
    xi_dim1 = *dnx;
    xi_offset = 1 + xi_dim1 * 1;
    xi -= xi_offset;
    wei_dim1 = *dnx;
    wei_offset = 1 + wei_dim1 * 1;
    wei -= wei_offset;
    data_dim1 = *dnx;
    data_offset = 1 + data_dim1 * 1;
    data -= data_offset;
    done_dim1 = *xmax - *xmin + 1;
    done_offset = 1 + done_dim1 * 1;
    done -= done_offset;
    ncon_dim1 = *xmax - *xmin + 1;
    ncon_offset = 1 + ncon_dim1 * 1;
    ncon -= ncon_offset;
    ncou_dim1 = *xmax - *xmin + 1;
    ncou_offset = 1 + ncou_dim1 * 1;
    ncou -= ncou_offset;
    ndat_dim1 = *xmax - *xmin + 1;
    ndat_offset = 1 + ndat_dim1 * 1;
    ndat -= ndat_offset;
    --yco;
    --xco;
    pyg_dim1 = *xgdim;
    pyg_offset = 1 + pyg_dim1 * 1;
    pyg -= pyg_offset;
    pxg_dim1 = *xgdim;
    pxg_offset = 1 + pxg_dim1 * 1;
    pxg -= pxg_offset;
    --wcs;
    --wcsout;
    intab_dim1 = *maxim;
    intab_offset = 1 + intab_dim1 * 1;
    intab -= intab_offset;

    /* Function Body */
    oldcon = -1;
/* The bitmask - trimmed to the appropriate range */
    np = (*uniqid - 1) / 32 + 1;
    i__1 = *uniqid - 1 - (np - 1 << 5);
    bv = pow_ii(&c__2, &i__1);
/* In the WCS case we can't use the scale to calculate the */
/* Jacobian so we need to do it */

/* Note that we use the centre of the image rather than the */
/* reference pixel as the reference here */
    if (*usewcs) {
	if (s_cmp(align, "corner", (ftnlen)8, (ftnlen)6) == 0) {
	    xcen = (doublereal) (*dnx / 2) + .5f;
	    ycen = (doublereal) (*dny / 2) + .5f;
	} else {
	    xcen = (doublereal) (*dnx / 2) + 1.f;
	    ycen = (doublereal) (*dny / 2) + 1.f;
	}
	xin[0] = xcen;
	xin[1] = xcen;
	xin[2] = xcen + 1.;
	xin[3] = xcen + 1.;
	yin[0] = ycen;
	yin[1] = ycen + 1.;
	yin[2] = ycen + 1.;
	yin[3] = ycen;
	drival_(xin, yin, &c__4, dnx, dny, onx, ony, &c_false, xsh, ysh, rot, 
		scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, 
		yscale, shfr2, rotf2, usewcs, &wcs[1], &wcsout[1], coty, 
		conum, &xco[1], &yco[1], disim, &pxg[pxg_offset], &pyg[
		pyg_offset], xgdim, ygdim, xout, yout, (ftnlen)8, (ftnlen)8);
	scall = sqrt(1.f / (d__1 = ((xout[1] - xout[3]) * (yout[0] - yout[2]) 
		- (xout[0] - xout[2]) * (yout[1] - yout[3])) * .5f, abs(d__1))
		);
/* Now calculate how much of this is from the geometric distortion */
	*xsh = 0.;
	*ysh = 0.;
	*rot = 0.;
	*scale = 1.;
	*secpar = FALSE_;
	*usewcs = FALSE_;
	drival_(xin, yin, &c__4, dnx, dny, onx, ony, &c_false, xsh, ysh, rot, 
		scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, 
		yscale, shfr2, rotf2, usewcs, &wcs[1], &wcsout[1], coty, 
		conum, &xco[1], &yco[1], disim, &pxg[pxg_offset], &pyg[
		pyg_offset], xgdim, ygdim, xout, yout, (ftnlen)8, (ftnlen)8);
	scdis = sqrt(1.f / (d__1 = ((xout[1] - xout[3]) * (yout[0] - yout[2]) 
		- (xout[0] - xout[2]) * (yout[1] - yout[3])) * .5f, abs(d__1))
		);
	*usewcs = TRUE_;
	*scale = scall / scdis;
    }
/* Image subset size */
    lx = *xmax - *xmin + 1;
    ly = *ymax - *ymin + 1;
    xf = 1.f / *scale;
    yf = 1.f / *scale;
    dh = *pfract * .5;
    ac = 1. / (*pfract * *pfract);
/* Recalculate the area scaling factor */
    s2 = *scale * *scale;
/* Offsets */
    dx = (doublereal) (*xmin - 1);
    dy = (doublereal) (*ymin - 1);
/* Half pixfrac on output */
    pfo = *pfract / *scale / 2.;
/* Computing 2nd power */
    d__1 = pfo;
    pfo2 = d__1 * d__1;
/* Some Gaussian related numbers */
    if (s_cmp(kernel, "gaussian", (ftnlen)8, (ftnlen)8) == 0) {
	efac = s2 * 5.5450830399999997f * ac / 2.;
	es = efac / 3.1415926f;
	pfo = *pfract * 2.5 / 2.3548f / *scale;
/* Added in V2.9 - make sure this doesn't get less than 1.2 */
/* divided by the scale so that there are never holes in the output */
	if (pfo < 1.2f / *scale) {
	    pfo = 1.2f / *scale;
	}
    }
/* Set up a look-up-table for Lanczos-style interpolation kernels */
/* It is assumed that the last character is an order (single digit) */
    if (s_cmp(kernel, "lanczos", (ftnlen)7, (ftnlen)7) == 0) {
	ici__1.icierr = 0;
	ici__1.iciend = 0;
	ici__1.icirnum = 1;
	ici__1.icirlen = 1;
	ici__1.iciunit = kernel + 7;
	ici__1.icifmt = "(I1)";
	s_rsfi(&ici__1);
	do_fio(&c__1, (char *)&lanord, (ftnlen)sizeof(integer));
	e_rsfi();
	filalu_(&lanord, &c__512, &c_b231, lanlut);
/* Note - removed the +1 here (RNH, 21/11/2002) */
	pfo = (doublereal) lanord * *pfract / *scale;
	sdp = *scale / .01f / *pfract;
    }
/* We skip all this if there is no overlap */
    if (! (*noover)) {
/* This is the outer loop over all the lines in the input image */
/* Before we start we can fill the X arrays as they don't change */
/* with Y */
	if (s_cmp(kernel, "square", (ftnlen)8, (ftnlen)6) == 0) {
	    xi[xi_dim1 + 1] = 1. - dh;
	    xi[(xi_dim1 << 1) + 1] = dh + 1.;
	    xi[xi_dim1 * 3 + 1] = dh + 1.;
	    xi[(xi_dim1 << 2) + 1] = 1. - dh;
	} else {
	    xib[1] = 1.;
	}
/* If the input image is not in CPS we need to divide by */
/* the exposure */
	if (! (*incps)) {
	    r__1 = 1.f / *expin;
	    mulc_(&data[data_offset], dnx, ny, &r__1);
	}
/* Loop over input lines */
	y = (doublereal) (*ystart);
	i__1 = *ny;
	for (j = 1; j <= i__1; ++j) {
	    y += 1.;
/* Check the overlap with the output */
	    chover_(&y, &c__5, dnx, dny, onx, ony, xsh, ysh, rot, scale, 
		    align, rotfir, secpar, xsh2, ysh2, rot2, xscale, yscale, 
		    shfr2, rotf2, usewcs, &wcs[1], &wcsout[1], coty, conum, &
		    xco[1], &yco[1], disim, &pxg[pxg_offset], &pyg[pyg_offset]
		    , xgdim, ygdim, &ofrac, &x1, &x2, (ftnlen)8, (ftnlen)8);
/* If the line falls completely off the output then skip it */
	    if (ofrac != 0.) {
/* We know there are some misses */
		*nmiss = *nmiss + *dnx - (x2 - x1 + 1);
/* Fill the X arrays as they don't change with Y */
		if (s_cmp(kernel, "square", (ftnlen)6, (ftnlen)6) == 0) {
		    xi[x1 + xi_dim1] = (doublereal) x1 - dh;
		    xi[x1 + (xi_dim1 << 1)] = (doublereal) x1 + dh;
		    xi[x1 + xi_dim1 * 3] = (doublereal) x1 + dh;
		    xi[x1 + (xi_dim1 << 2)] = (doublereal) x1 - dh;
		} else {
		    xib[x1] = (doublereal) x1;
		}
/* At this point we can handle the different kernels separately */
/* First the cases where we just transform a single point rather */
/* than four - every case except the "classic" square-pixel kernel */
		if (s_cmp(kernel, "square", (ftnlen)6, (ftnlen)6) != 0) {
		    yib[x1] = y;
		    yib[x1 + 1] = 0.;
/* Transform onto the output grid */
		    i__2 = x2 - x1 + 1;
		    drival_(&xib[x1], &yib[x1], &i__2, dnx, dny, onx, ony, &
			    c_true, xsh, ysh, rot, scale, align, rotfir, 
			    secpar, xsh2, ysh2, rot2, xscale, yscale, shfr2, 
			    rotf2, usewcs, &wcs[1], &wcsout[1], coty, conum, &
			    xco[1], &yco[1], disim, &pxg[pxg_offset], &pyg[
			    pyg_offset], xgdim, ygdim, &xob[x1], &yob[x1], (
			    ftnlen)8, (ftnlen)8);
/* Now we consider the different cases, first the "point" option */
/* where a single pixel in the output image is affected */
		    if (s_cmp(kernel, "point", (ftnlen)5, (ftnlen)5) == 0) {
/* Offset within the subset */
			i__2 = x2;
			for (i__ = x1; i__ <= i__2; ++i__) {
			    d__1 = xob[i__] - dx;
			    ii = i_dnnt(&d__1);
			    d__1 = yob[i__] - dy;
			    jj = i_dnnt(&d__1);
/* Check it is on the output image */
			    if (ii >= 1 && ii <= lx && jj >= 1 && jj <= ly) {
				vc = ncou[ii + jj * ncou_dim1];
/* Allow for stretching because of scale change */
				d__ = data[i__ + j * data_dim1] * s2;
/* Scale the weighting mask by the scale factor */
/* Note that we DON'T scale by the Jacobian as it hasn't been */
/* calculated */
				dow = wei[i__ + j * wei_dim1] * *wtscl;
/* If we are creating or modifying the context image we */
/* do so here */
				if (*con && dow > 0.f) {
				    if (*bitcon) {
					ncon[ii + jj * ncon_dim1] |= bv;
				    } else {
					if (done[ii + jj * done_dim1] == 0) {
					    upcon_(&ncon[ncon_offset], &ii, &
						    jj, &oldcon, &newcon, &
						    done[done_offset], &lx, &
						    ly, &intab[intab_offset], 
						    nen, maxim, maxen, uniqid,
						     istat);
					}
				    }
				}
/* Just a simple calculation without logical tests */
				if (vc == 0.f) {
				    ndat[ii + jj * ndat_dim1] = d__;
				} else {
				    ndat[ii + jj * ndat_dim1] = (ndat[ii + jj 
					    * ndat_dim1] * vc + dow * d__) / (
					    vc + dow);
				}
				ncou[ii + jj * ncou_dim1] = vc + dow;
			    } else {
				++(*nmiss);
			    }
			}
/* Next "tophat" - a circular kernel giving equal weight to */
/* all points within a certain radius */
		    } else if (s_cmp(kernel, "tophat", (ftnlen)6, (ftnlen)6) 
			    == 0) {
			i__2 = x2;
			for (i__ = x1; i__ <= i__2; ++i__) {
/* Offset within the subset */
			    xx = xob[i__] - dx;
			    yy = yob[i__] - dy;
			    xxi = xob[i__] - dx - pfo;
			    xxa = xob[i__] - dx + pfo;
			    yyi = yob[i__] - dy - pfo;
			    yya = yob[i__] - dy + pfo;
			    nxi = i_dnnt(&xxi);
			    nxa = i_dnnt(&xxa);
			    nyi = i_dnnt(&yyi);
			    nya = i_dnnt(&yya);
			    nhit = 0;
/* Allow for stretching because of scale change */
			    d__ = data[i__ + j * data_dim1] * s2;
/* Scale the weighting mask by the scale factor */
/* and inversely by the Jacobian to ensure conservation */
/* of weight in the output */
			    dow = wei[i__ + j * wei_dim1] * *wtscl;
			    dd = dow * d__;
/* Loop over output pixels which could be affected */
			    i__3 = nya;
			    for (jj = nyi; jj <= i__3; ++jj) {
				i__4 = nxa;
				for (ii = nxi; ii <= i__4; ++ii) {
/* Check it is on the output image */
				    if (ii >= 1 && ii <= lx && jj >= 1 && jj 
					    <= ly) {
/* Radial distance */
/* Computing 2nd power */
					d__1 = xx - (doublereal) ii;
/* Computing 2nd power */
					d__2 = yy - (doublereal) jj;
					r2 = d__1 * d__1 + d__2 * d__2;
/* Weight is one within the specified radius and zero outside */
/* Note: weight isn't conserved in this case */
					if (r2 <= pfo2) {
/* Count the hits */
					    ++nhit;
					    vc = ncou[ii + jj * ncou_dim1];
/* If we are creating or modifying the context image we */
/* do so here */
					    if (*con && dow > 0.f) {
			  if (*bitcon) {
			      ncon[ii + jj * ncon_dim1] |= bv;
			  } else {
			      if (done[ii + jj * done_dim1] == 0) {
				  upcon_(&ncon[ncon_offset], &ii, &jj, &
					  oldcon, &newcon, &done[done_offset],
					   &lx, &ly, &intab[intab_offset], 
					  nen, maxim, maxen, uniqid, istat);
			      }
			  }
					    }
/* Just a simple calculation without logical tests */
					    if (vc == 0.f) {
			  ndat[ii + jj * ndat_dim1] = d__;
					    } else {
			  ndat[ii + jj * ndat_dim1] = (ndat[ii + jj * 
				  ndat_dim1] * vc + dd) / (vc + dow);
					    }
					    ncou[ii + jj * ncou_dim1] = vc + 
						    dow;
					}
				    }
				}
			    }
/* Count cases where the pixel is off the output image */
			    if (nhit == 0) {
				++(*nmiss);
			    }
			}
/* Next a gaussian weighting kernel with FWHM = pixfrac/scale */
		    } else if (s_cmp(kernel, "gaussian", (ftnlen)8, (ftnlen)8)
			     == 0) {
			i__2 = x2;
			for (i__ = x1; i__ <= i__2; ++i__) {
/* Offset within the subset */
			    xx = xob[i__] - dx;
			    yy = yob[i__] - dy;
			    xxi = xob[i__] - dx - pfo;
			    xxa = xob[i__] - dx + pfo;
			    yyi = yob[i__] - dy - pfo;
			    yya = yob[i__] - dy + pfo;
			    nxi = i_dnnt(&xxi);
			    nxa = i_dnnt(&xxa);
			    nyi = i_dnnt(&yyi);
			    nya = i_dnnt(&yya);
			    nhit = 0;
/* Allow for stretching because of scale change */
			    d__ = data[i__ + j * data_dim1] * s2;
/* Scale the weighting mask by the scale factor */
/* and inversely by the Jacobian to ensure conservation */
/* of weight in the output */
			    w = wei[i__ + j * wei_dim1] * *wtscl;
/* Loop over output pixels which could be affected */
			    i__3 = nya;
			    for (jj = nyi; jj <= i__3; ++jj) {
				i__4 = nxa;
				for (ii = nxi; ii <= i__4; ++ii) {
/* Check it is on the output image */
				    if (ii >= 1 && ii <= lx && jj >= 1 && jj 
					    <= ly) {
/* Radial distance */
/* Computing 2nd power */
					d__1 = xx - (doublereal) ii;
/* Computing 2nd power */
					d__2 = yy - (doublereal) jj;
					r2 = d__1 * d__1 + d__2 * d__2;
/* Weight is a scaled gaussian function of radial distance */
					dover = es * exp(-r2 * efac);
/* Count the hits */
					++nhit;
					vc = ncou[ii + jj * ncou_dim1];
					dow = dover * w;
/* If we are creating or modifying the context image we */
/* do so here */
					if (*con && dow > 0.f) {
					    if (*bitcon) {
			  ncon[ii + jj * ncon_dim1] |= bv;
					    } else {
			  if (done[ii + jj * done_dim1] == 0) {
			      upcon_(&ncon[ncon_offset], &ii, &jj, &oldcon, &
				      newcon, &done[done_offset], &lx, &ly, &
				      intab[intab_offset], nen, maxim, maxen, 
				      uniqid, istat);
			  }
					    }
					}
/* Just a simple calculation without logical tests */
					if (vc == 0.f) {
					    ndat[ii + jj * ndat_dim1] = d__;
					} else {
					    ndat[ii + jj * ndat_dim1] = (ndat[
						    ii + jj * ndat_dim1] * vc 
						    + dow * d__) / (vc + dow);
					}
					ncou[ii + jj * ncou_dim1] = vc + dow;
				    }
				}
			    }
/* Count cases where the pixel is off the output image */
			    if (nhit == 0) {
				++(*nmiss);
			    }
			}
/* Next a Lanczos weighting kernel */
		    } else if (s_cmp(kernel, "lanczos", (ftnlen)7, (ftnlen)7) 
			    == 0) {
			i__2 = x2;
			for (i__ = x1; i__ <= i__2; ++i__) {
/* Offset within the subset */
			    xx = xob[i__] - dx;
			    yy = yob[i__] - dy;
			    xxi = xob[i__] - dx - pfo;
			    xxa = xob[i__] - dx + pfo;
			    yyi = yob[i__] - dy - pfo;
			    yya = yob[i__] - dy + pfo;
			    nxi = i_dnnt(&xxi);
			    nxa = i_dnnt(&xxa);
			    nyi = i_dnnt(&yyi);
			    nya = i_dnnt(&yya);
			    nhit = 0;
/* Allow for stretching because of scale change */
			    d__ = data[i__ + j * data_dim1] * s2;
/* Scale the weighting mask by the scale factor */
/* and inversely by the Jacobian to ensure conservation */
/* of weight in the output */
			    w = wei[i__ + j * wei_dim1] * *wtscl;
/* Loop over output pixels which could be affected */
			    i__3 = nya;
			    for (jj = nyi; jj <= i__3; ++jj) {
				i__4 = nxa;
				for (ii = nxi; ii <= i__4; ++ii) {
/* Check it is on the output image */
				    if (ii >= 1 && ii <= lx && jj >= 1 && jj 
					    <= ly) {
/* X and Y offsets */
					d__2 = (d__1 = xx - (doublereal) ii, 
						abs(d__1)) * sdp;
					ix = i_dnnt(&d__2) + 1;
					d__2 = (d__1 = yy - (doublereal) jj, 
						abs(d__1)) * sdp;
					iy = i_dnnt(&d__2) + 1;
/* Weight is product of Lanczos function values in X and Y */
					dover = lanlut[ix - 1] * lanlut[iy - 
						1];
/* Count the hits */
					++nhit;
					vc = ncou[ii + jj * ncou_dim1];
					dow = dover * w;
/* If we are creating or modifying the context image we */
/* do so here */
					if (*con && dow > 0.f) {
					    if (*bitcon) {
			  ncon[ii + jj * ncon_dim1] |= bv;
					    } else {
			  if (done[ii + jj * done_dim1] == 0) {
			      upcon_(&ncon[ncon_offset], &ii, &jj, &oldcon, &
				      newcon, &done[done_offset], &lx, &ly, &
				      intab[intab_offset], nen, maxim, maxen, 
				      uniqid, istat);
			  }
					    }
					}
/* Just a simple calculation without logical tests */
					if (vc == 0.f) {
					    ndat[ii + jj * ndat_dim1] = d__;
					} else {
/* There is an extra check here in the Lanczos case because the weight can */
/* be negative which means that, just occasionally, the new and existing */
/* weights at a given position can be have opposite signs and zero sum... */
					    if (vc + dow == 0.f) {
			  ndat[ii + jj * ndat_dim1] = d__;
					    } else {
			  ndat[ii + jj * ndat_dim1] = (ndat[ii + jj * 
				  ndat_dim1] * vc + dow * d__) / (vc + dow);
					    }
					}
					ncou[ii + jj * ncou_dim1] = vc + dow;
				    }
				}
			    }
/* Count cases where the pixel is off the output image */
			    if (nhit == 0) {
				++(*nmiss);
			    }
			}
/* The "turbo" option with a constant square pixfrac (as used in EIS Drizzle) */
		    } else if (s_cmp(kernel, "turbo", (ftnlen)5, (ftnlen)5) ==
			     0) {
			i__2 = x2;
			for (i__ = x1; i__ <= i__2; ++i__) {
/* Offset within the subset */
			    xxi = xob[i__] - dx - pfo;
			    xxa = xob[i__] - dx + pfo;
			    yyi = yob[i__] - dy - pfo;
			    yya = yob[i__] - dy + pfo;
			    nxi = i_dnnt(&xxi);
			    nxa = i_dnnt(&xxa);
			    nyi = i_dnnt(&yyi);
			    nya = i_dnnt(&yya);
			    nhit = 0;
/* Allow for stretching because of scale change */
			    d__ = data[i__ + j * data_dim1] * s2;
/* Scale the weighting mask by the scale factor */
/* and inversely by the Jacobian to ensure conservation */
/* of weight in the output */
			    w = wei[i__ + j * wei_dim1] * *wtscl;
/* Loop over output pixels which could be affected */
			    i__3 = nya;
			    for (jj = nyi; jj <= i__3; ++jj) {
				i__4 = nxa;
				for (ii = nxi; ii <= i__4; ++ii) {
/* Check it is on the output image */
				    if (ii >= 1 && ii <= lx && jj >= 1 && jj 
					    <= ly) {
/* Calculate the overlap using the simpler "aligned" box routine */
					dover = over_(&ii, &jj, &xxi, &xxa, &
						yyi, &yya);
					if (dover > 0.f) {
/* Correct for pixfrac area factor */
					    dover = dover * s2 * ac;
/* Count the hits */
					    ++nhit;
					    vc = ncou[ii + jj * ncou_dim1];
					    dow = dover * w;
/* If we are creating or modifying the context image we */
/* do so here */
					    if (*con && dow > 0.f) {
			  if (*bitcon) {
			      ncon[ii + jj * ncon_dim1] |= bv;
			  } else {
			      if (done[ii + jj * done_dim1] == 0) {
				  upcon_(&ncon[ncon_offset], &ii, &jj, &
					  oldcon, &newcon, &done[done_offset],
					   &lx, &ly, &intab[intab_offset], 
					  nen, maxim, maxen, uniqid, istat);
			      }
			  }
					    }
/* Just a simple calculation without logical tests */
					    if (vc == 0.f) {
			  ndat[ii + jj * ndat_dim1] = d__;
					    } else {
			  ndat[ii + jj * ndat_dim1] = (ndat[ii + jj * 
				  ndat_dim1] * vc + dow * d__) / (vc + dow);
					    }
					    ncou[ii + jj * ncou_dim1] = vc + 
						    dow;
					}
				    }
				}
			    }
/* Count cases where the pixel is off the output image */
			    if (nhit == 0) {
				++(*nmiss);
			    }
			}
/* End of the "single point" transform case statement */
		    }
/* Next the "classic" drizzle square kernel... */
/* this is different because we have to transform all four corners */
/* of the shrunken pixel */
		} else {
/* Set the start corner positions - only in Y, */
/* X is already done */
		    yi[x1 + yi_dim1] = y;
		    yi[x1 + (yi_dim1 << 1)] = y;
		    yi[x1 + yi_dim1 * 3] = y;
		    yi[x1 + (yi_dim1 << 2)] = y;
		    yi[x1 + 1 + yi_dim1] = dh;
		    yi[x1 + 1 + (yi_dim1 << 1)] = dh;
		    yi[x1 + 1 + yi_dim1 * 3] = -dh;
		    yi[x1 + 1 + (yi_dim1 << 2)] = -dh;
/* Transform onto the output grid */
		    i__2 = x2 - x1 + 1;
		    drival_(&xi[x1 + xi_dim1], &yi[x1 + yi_dim1], &i__2, dnx, 
			    dny, onx, ony, &c_true, xsh, ysh, rot, scale, 
			    align, rotfir, secpar, xsh2, ysh2, rot2, xscale, 
			    yscale, shfr2, rotf2, usewcs, &wcs[1], &wcsout[1],
			     coty, conum, &xco[1], &yco[1], disim, &pxg[
			    pxg_offset], &pyg[pyg_offset], xgdim, ygdim, &xo[
			    x1 + xo_dim1], &yo[x1 + yo_dim1], (ftnlen)8, (
			    ftnlen)8);
		    i__2 = x2 - x1 + 1;
		    drival_(&xi[x1 + (xi_dim1 << 1)], &yi[x1 + (yi_dim1 << 1)]
			    , &i__2, dnx, dny, onx, ony, &c_true, xsh, ysh, 
			    rot, scale, align, rotfir, secpar, xsh2, ysh2, 
			    rot2, xscale, yscale, shfr2, rotf2, usewcs, &wcs[
			    1], &wcsout[1], coty, conum, &xco[1], &yco[1], 
			    disim, &pxg[pxg_offset], &pyg[pyg_offset], xgdim, 
			    ygdim, &xo[x1 + (xo_dim1 << 1)], &yo[x1 + (
			    yo_dim1 << 1)], (ftnlen)8, (ftnlen)8);
		    i__2 = x2 - x1 + 1;
		    drival_(&xi[x1 + xi_dim1 * 3], &yi[x1 + yi_dim1 * 3], &
			    i__2, dnx, dny, onx, ony, &c_true, xsh, ysh, rot, 
			    scale, align, rotfir, secpar, xsh2, ysh2, rot2, 
			    xscale, yscale, shfr2, rotf2, usewcs, &wcs[1], &
			    wcsout[1], coty, conum, &xco[1], &yco[1], disim, &
			    pxg[pxg_offset], &pyg[pyg_offset], xgdim, ygdim, &
			    xo[x1 + xo_dim1 * 3], &yo[x1 + yo_dim1 * 3], (
			    ftnlen)8, (ftnlen)8);
		    i__2 = x2 - x1 + 1;
		    drival_(&xi[x1 + (xi_dim1 << 2)], &yi[x1 + (yi_dim1 << 2)]
			    , &i__2, dnx, dny, onx, ony, &c_true, xsh, ysh, 
			    rot, scale, align, rotfir, secpar, xsh2, ysh2, 
			    rot2, xscale, yscale, shfr2, rotf2, usewcs, &wcs[
			    1], &wcsout[1], coty, conum, &xco[1], &yco[1], 
			    disim, &pxg[pxg_offset], &pyg[pyg_offset], xgdim, 
			    ygdim, &xo[x1 + (xo_dim1 << 2)], &yo[x1 + (
			    yo_dim1 << 2)], (ftnlen)8, (ftnlen)8);
		    i__2 = x2;
		    for (i__ = x1; i__ <= i__2; ++i__) {
/* Offset within the subset */
			xout[0] = xo[i__ + xo_dim1] - dx;
			xout[1] = xo[i__ + (xo_dim1 << 1)] - dx;
			xout[2] = xo[i__ + xo_dim1 * 3] - dx;
			xout[3] = xo[i__ + (xo_dim1 << 2)] - dx;
			yout[0] = yo[i__ + yo_dim1] - dy;
			yout[1] = yo[i__ + (yo_dim1 << 1)] - dy;
			yout[2] = yo[i__ + yo_dim1 * 3] - dy;
			yout[3] = yo[i__ + (yo_dim1 << 2)] - dy;
/* Work out the area of the quadrilateral on the output grid */
/* Note that this expression expects the points to be in */
/* clockwise order */
			jaco = ((xout[1] - xout[3]) * (yout[0] - yout[2]) - (
				xout[0] - xout[2]) * (yout[1] - yout[3])) * 
				.5f;
			nhit = 0;
/* Allow for stretching because of scale change */
			d__ = data[i__ + j * data_dim1] * s2;
/* Scale the weighting mask by the scale factor */
/* and inversely by the Jacobian to ensure conservation */
/* of weight in the output */
			w = wei[i__ + j * wei_dim1] * *wtscl;
/* Loop over output pixels which could be affected */
/* Computing MIN */
			d__2 = min(yout[0],yout[1]), d__2 = min(d__2,yout[2]);
			d__1 = min(d__2,yout[3]);
/* Computing MAX */
			d__4 = max(yout[0],yout[1]), d__4 = max(d__4,yout[2]);
			d__3 = max(d__4,yout[3]);
			i__3 = i_dnnt(&d__3);
			for (jj = i_dnnt(&d__1); jj <= i__3; ++jj) {
/* Computing MIN */
			    d__2 = min(xout[0],xout[1]), d__2 = min(d__2,xout[
				    2]);
			    d__1 = min(d__2,xout[3]);
/* Computing MAX */
			    d__4 = max(xout[0],xout[1]), d__4 = max(d__4,xout[
				    2]);
			    d__3 = max(d__4,xout[3]);
			    i__4 = i_dnnt(&d__3);
			    for (ii = i_dnnt(&d__1); ii <= i__4; ++ii) {
/* Check it is on the output image */
				if (ii >= 1 && ii <= lx && jj >= 1 && jj <= 
					ly) {
/* Call boxer to calculate overlap */
				    boxer_(&ii, &jj, xout, yout, &dover);
				    if (dover > 0.f) {
/* Re-normalise the area overlap using the Jacobian */
					dover /= jaco;
/* Count the hits */
					++nhit;
					vc = ncou[ii + jj * ncou_dim1];
					dow = dover * w;
/* If we are creating or modifying the context image we */
/* do so here */
					if (*con && dow > 0.f) {
					    if (*bitcon) {
			  ncon[ii + jj * ncon_dim1] |= bv;
					    } else {
			  if (done[ii + jj * done_dim1] == 0) {
			      upcon_(&ncon[ncon_offset], &ii, &jj, &oldcon, &
				      newcon, &done[done_offset], &lx, &ly, &
				      intab[intab_offset], nen, maxim, maxen, 
				      uniqid, istat);
			  }
					    }
					}
/* Just a simple calculation without logical tests */
					if (vc == 0.f) {
					    ndat[ii + jj * ndat_dim1] = d__;
					} else {
					    ndat[ii + jj * ndat_dim1] = (ndat[
						    ii + jj * ndat_dim1] * vc 
						    + dow * d__) / (vc + dow);
					}
					ncou[ii + jj * ncou_dim1] = vc + dow;
				    }
				}
			    }
			}
/* Count cases where the pixel is off the output image */
			if (nhit == 0) {
			    ++(*nmiss);
			}
		    }
/* End of the kernel "case" blocks */
		}
	    } else {
/* If we are skipping a line count it */
		++(*nskip);
		*nmiss += *dnx;
	    }
	}
    } else {
/* If there is no overlap at all set appropriate values */
	*nskip = *dny;
	*nmiss = *dnx * *dny;
    }
/* Set good status if we get this far */
    *istat = 0;
    return 0;
} /* dobox_ */

/* Subroutine */ int upwcs_(doublereal *wcsin, doublereal *wcsout, integer *
	dnx, integer *dny, integer *onx, integer *ony, doublereal *xsh, 
	doublereal *ysh, doublereal *rot, doublereal *scale, char *align, 
	logical *rotfir, logical *secpar, doublereal *xsh2, doublereal *ysh2, 
	doublereal *rot2, doublereal *xscale, doublereal *yscale, char *shfr2,
	 logical *rotf2, logical *usewcs, integer *coty, integer *conum, 
	doublereal *xco, doublereal *yco, logical *disim, real *pxg, real *
	pyg, integer *xgdim, integer *ygdim, ftnlen align_len, ftnlen 
	shfr2_len)
{
    /* System generated locals */
    integer pxg_dim1, pxg_offset, pyg_dim1, pyg_offset;

    /* Local variables */
    static doublereal xout[3], yout[3];
    extern /* Subroutine */ int inmat_(doublereal *, doublereal *, doublereal 
	    *, doublereal *);
    static integer istat, scoty;
    static doublereal w5, w6, w7, w8, am, bm, cm, dm;
    extern /* Subroutine */ int drival_(doublereal *, doublereal *, integer *,
	     integer *, integer *, integer *, integer *, logical *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, char *, 
	    logical *, logical *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, char *, logical *, logical *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    doublereal *, logical *, real *, real *, integer *, integer *, 
	    doublereal *, doublereal *, ftnlen, ftnlen), umsput_(char *, 
	    integer *, integer *, integer *, ftnlen);
    static doublereal xin[3], yin[3];


/* Update the WCS to include the drizzling transformations */

/* This is done by applying the transform to a unit square at */
/* the centre pixel in the input whilst retaining the same */
/* reference point on the sky. */

/*    Richard Hook, ST-ECF, September 2000 */

/* Modified to use double precision, Richard Hook, STScI/ST-ECF, June 2003 */

/* Linear terms only used - Richard Hook, STScI/ST-ECF, January 2004 */

/* If we have the WCS already just return */
    /* Parameter adjustments */
    --wcsin;
    --wcsout;
    --yco;
    --xco;
    pyg_dim1 = *xgdim;
    pyg_offset = 1 + pyg_dim1 * 1;
    pyg -= pyg_offset;
    pxg_dim1 = *xgdim;
    pxg_offset = 1 + pxg_dim1 * 1;
    pxg -= pxg_offset;

    /* Function Body */
    if (*usewcs) {
	return 0;
    }
/* Set up a single point at the reference pixel to map the reference point */
    xin[0] = wcsin[1];
    yin[0] = wcsin[3];
/* Transform */
    drival_(xin, yin, &c__1, dnx, dny, onx, ony, &c_false, xsh, ysh, rot, 
	    scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, yscale, 
	    shfr2, rotf2, usewcs, &wcsin[1], &wcsout[1], coty, conum, &xco[1],
	     &yco[1], disim, &pxg[pxg_offset], &pyg[pyg_offset], xgdim, ygdim,
	     xout, yout, (ftnlen)8, (ftnlen)8);
/* We can immediately set the reference point on the sky */
    wcsout[1] = xout[0];
    wcsout[3] = yout[0];
    wcsout[2] = wcsin[2];
    wcsout[4] = wcsin[4];
/* Set up a 1x1 box at the centre pixel (three sides only) */
/* to allow us to update the WCS */
    xin[1] = xin[0] + 1.;
    yin[1] = yin[0];
    xin[2] = xin[0];
    yin[2] = yin[0] + 1.;
/* Transform */
/* Only use LINEAR terms */
    scoty = *coty;
    *coty = 1;
    drival_(xin, yin, &c__3, dnx, dny, onx, ony, &c_false, xsh, ysh, rot, 
	    scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, yscale, 
	    shfr2, rotf2, usewcs, &wcsin[1], &wcsout[1], coty, conum, &xco[1],
	     &yco[1], disim, &pxg[pxg_offset], &pyg[pyg_offset], xgdim, ygdim,
	     xout, yout, (ftnlen)8, (ftnlen)8);
/* Restore order */
    *coty = scoty;
/* Now work out the effective CD matrix of the transformation */
    am = xout[1] - xout[0];
    bm = xout[2] - xout[0];
    cm = yout[1] - yout[0];
    dm = yout[2] - yout[0];
/* Check the determinant for singularity */
    if (am * dm - bm * cm == 0.) {
	umsput_("! Matrix is singular, cannot update WCS", &c__1, &c__0, &
		istat, (ftnlen)39);
    } else {
/* Invert the matrix */
	inmat_(&am, &bm, &cm, &dm);
/* Correct the CD matrix */
	w5 = am * wcsin[5] + cm * wcsin[7];
	w6 = am * wcsin[6] + cm * wcsin[8];
	w7 = bm * wcsin[5] + dm * wcsin[7];
	w8 = bm * wcsin[6] + dm * wcsin[8];
	wcsout[5] = w5;
	wcsout[6] = w6;
	wcsout[7] = w7;
	wcsout[8] = w8;
    }
    return 0;
} /* upwcs_ */

/* Subroutine */ int boxer_(integer *is, integer *js, doublereal *x, 
	doublereal *y, doublereal *darea)
{
    static integer i__;
    extern doublereal sgarea_(doublereal *, doublereal *, doublereal *, 
	    doublereal *, integer *, integer *);
    static doublereal px[4], py[4], sum;


/* BOXER -- compute area of box overlap */

/* Calculate the area common to input clockwise polygon x(n), y(n) with */
/* square (is, js) to (is+1, js+1). */
/* This version is for a quadrilateral. */

/* W.B. Sparks STScI 2-June-1990. */
/* Phil Hodge        20-Nov-1990  Change calling sequence; single precision. */
/* Richard Hook ECF  24-Apr-1996  Change coordinate origin */
/*                                so that centre of pixel has integer position */
/*                   03-Jan-2001  Removed accuracy check */
/* -- */
/* Set up coords relative to unit square at origin */
/* Note that the +0.5s were added when this code was */
/* included in DRIZZLE */
    /* Parameter adjustments */
    --y;
    --x;

    /* Function Body */
    for (i__ = 1; i__ <= 4; ++i__) {
	px[i__ - 1] = x[i__] - *is + .5;
	py[i__ - 1] = y[i__] - *js + .5;
    }
/* For each line in the polygon (or at this stage, input quadrilateral) */
/* calculate the area common to the unit square (allow negative area for */
/* subsequent `vector' addition of subareas). */
    sum = 0.;
    for (i__ = 1; i__ <= 3; ++i__) {
	sum += sgarea_(&px[i__ - 1], &py[i__ - 1], &px[i__], &py[i__], is, js)
		;
    }
    sum += sgarea_(&px[3], &py[3], px, py, is, js);
    *darea = sum;
    return 0;
} /* boxer_ */

doublereal sgarea_(doublereal *x1, doublereal *y1, doublereal *x2, doublereal 
	*y2, integer *is, integer *js)
{
    /* System generated locals */
    doublereal ret_val;

    /* Local variables */
    static doublereal xtop, c__, m;
    static logical negdx;
    static doublereal dx, xhi, yhi, xlo, ylo;


/* To calculate area under a line segment within unit square at origin. */
/* This is used by BOXER */

    dx = *x2 - *x1;
/* Trap vertical line */
    if (dx == 0.) {
	ret_val = 0.;
	goto L80;
    }
/* Order the two input points in x */
    if (*x1 < *x2) {
	xlo = *x1;
	xhi = *x2;
    } else {
	xlo = *x2;
	xhi = *x1;
    }
/* And determine the bounds ignoring y for now */
    if (xlo >= 1.) {
	ret_val = 0.;
	goto L80;
    }
    if (xhi <= 0.) {
	ret_val = 0.;
	goto L80;
    }
    xlo = max(xlo,0.);
    xhi = min(xhi,1.);
/* Now look at y */
/* basic info about the line y = mx + c */
    negdx = dx < 0.;
    m = (*y2 - *y1) / dx;
    c__ = *y1 - m * *x1;
    ylo = m * xlo + c__;
    yhi = m * xhi + c__;
/* Trap segment entirely below axis */
    if (ylo <= 0. && yhi <= 0.) {
	ret_val = 0.;
	goto L80;
    }
/* Adjust bounds if segment crosses axis (to exclude anything below axis) */
    if (ylo < 0.) {
	ylo = 0.;
	xlo = -c__ / m;
    }
    if (yhi < 0.) {
	yhi = 0.;
	xhi = -c__ / m;
    }
/* There are four possibilities: both y below 1, both y above 1 */
/* and one of each. */
    if (ylo >= 1. && yhi >= 1.) {
/* Line segment is entirely above square */
	if (negdx) {
	    ret_val = xlo - xhi;
	} else {
	    ret_val = xhi - xlo;
	}
	goto L80;
    }
    if (ylo <= 1. && yhi <= 1.) {
/* Segment is entirely within square */
	if (negdx) {
	    ret_val = (xlo - xhi) * .5 * (yhi + ylo);
	} else {
	    ret_val = (xhi - xlo) * .5 * (yhi + ylo);
	}
	goto L80;
    }
/* otherwise it must cross the top of the square */
    xtop = (1. - c__) / m;
    if (ylo < 1.) {
	if (negdx) {
	    ret_val = -((xtop - xlo) * .5 * (ylo + 1.) + xhi - xtop);
	} else {
	    ret_val = (xtop - xlo) * .5 * (ylo + 1.) + xhi - xtop;
	}
	goto L80;
    }
    if (negdx) {
	ret_val = -((xhi - xtop) * .5 * (yhi + 1.) + xtop - xlo);
    } else {
	ret_val = (xhi - xtop) * .5 * (yhi + 1.) + xtop - xlo;
    }
L80:
    return ret_val;
} /* sgarea_ */

/* Subroutine */ int bupwcs_(doublereal *wcsin, doublereal *wcsout, integer *
	dnx, integer *dny, integer *onx, integer *ony, doublereal *xsh, 
	doublereal *ysh, doublereal *rot, doublereal *scale, char *align, 
	logical *rotfir, logical *secpar, doublereal *xsh2, doublereal *ysh2, 
	doublereal *rot2, doublereal *xscale, doublereal *yscale, char *shfr2,
	 logical *rotf2, logical *usewcs, integer *coty, integer *conum, 
	doublereal *xco, doublereal *yco, logical *disim, real *pxg, real *
	pyg, integer *pxdim, integer *pydim, ftnlen align_len, ftnlen 
	shfr2_len)
{
    /* System generated locals */
    integer pxg_dim1, pxg_offset, pyg_dim1, pyg_offset;

    /* Builtin functions */
    double cos(doublereal);

    /* Local variables */
    static doublereal xout[3], yout[3], d__[3];
    extern /* Subroutine */ int xy2rd_(doublereal *, doublereal *, doublereal 
	    *, doublereal *, doublereal *);
    static doublereal r__[3];
    extern /* Subroutine */ int drival_(doublereal *, doublereal *, integer *,
	     integer *, integer *, integer *, integer *, logical *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, char *, 
	    logical *, logical *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, char *, logical *, logical *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    doublereal *, logical *, real *, real *, integer *, integer *, 
	    doublereal *, doublereal *, ftnlen, ftnlen);
    static doublereal xin[3], yin[3];


/* Update the WCS to include the drizzling transformations */

/* This version is for BLOT and works the other way around to */
/* UPWCS. */

/* This is done by applying the transform to a unit square at */
/* the reference pixel in the input whilst retaining the same */
/* reference point on the sky. */

/*    Richard Hook, ST-ECF, December 2001 */

/* If we have the WCS already just return */
    /* Parameter adjustments */
    --wcsin;
    --wcsout;
    --yco;
    --xco;
    pyg_dim1 = *pxdim;
    pyg_offset = 1 + pyg_dim1 * 1;
    pyg -= pyg_offset;
    pxg_dim1 = *pxdim;
    pxg_offset = 1 + pxg_dim1 * 1;
    pxg -= pxg_offset;

    /* Function Body */
    if (*usewcs) {
	return 0;
    }
/* Set up a 1x1 box at the centre of the output image (three sides only) */
    xin[0] = (doublereal) (*onx / 2);
    yin[0] = (doublereal) (*ony / 2);
    xin[1] = (doublereal) (*onx / 2) + 1.;
    yin[1] = (doublereal) (*ony / 2);
    xin[2] = (doublereal) (*onx / 2);
    yin[2] = (doublereal) (*ony / 2) + 1.;
/* Transform */
    drival_(xin, yin, &c__3, onx, ony, dnx, dny, &c_false, xsh, ysh, rot, 
	    scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, yscale, 
	    shfr2, rotf2, usewcs, &wcsin[1], &wcsout[1], coty, conum, &xco[1],
	     &yco[1], disim, &pxg[pxg_offset], &pyg[pyg_offset], pxdim, pydim,
	     xout, yout, (ftnlen)8, (ftnlen)8);
/* Convert the output pixel position to a sky position using */
/* the WCS */
    xy2rd_(xout, yout, r__, d__, &wcsin[1]);
    xy2rd_(&xout[1], &yout[1], &r__[1], &d__[1], &wcsin[1]);
    xy2rd_(&xout[2], &yout[2], &r__[2], &d__[2], &wcsin[1]);
/* We can immediately set the reference point on the sky */
    wcsout[1] = xin[0];
    wcsout[3] = yin[0];
    wcsout[2] = r__[0];
    wcsout[4] = d__[0];
/* Now work out the effective CD matrix of the transformation */
/* Note - 6,7 were swapped (I think) originally???? */
    wcsout[5] = cos(d__[0] * 3.141592653589793 / 180.) * (r__[1] - r__[0]) / 
	    1.;
    wcsout[6] = (d__[1] - d__[0]) / 1.;
    wcsout[7] = cos(d__[0] * 3.141592653589793 / 180.) * (r__[2] - r__[0]) / 
	    1.;
    wcsout[8] = (d__[2] - d__[0]) / 1.;
    return 0;
} /* bupwcs_ */

/* Subroutine */ int filalu_(integer *order, integer *npix, real *del, real *
	lanlut)
{
    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    double sin(doublereal);

    /* Local variables */
    static real poff;
    static integer i__;
    static real for__;


/* Fill up a look-up-table of Lanczos interpolation kernel values */
/* for rapid weighting determination for KERNEL='lanczos3' etc. */

/* Supplied: */

/*  ORDER - integer, the order of the kernel. */

/*  NPIX - integer, the size of the look-up-table. */

/*  DEL - real, the spacings of the sampling of the function. */

/* Returned: */

/*  LANLUT - real, 1d array of look-up values. This is a single-sided */
/*           Lanczos function with LANLUT(1) being the central value */

/* Note that no checking is done to see whether the values are sensible. */

/*    Richard Hook, ST-ECF/STScI, August 2002 */

/* Changed to REAL rather than DOUBLE, Richard Hook, ST-ECF/STScI, Feb 2004 */

    /* Parameter adjustments */
    --lanlut;

    /* Function Body */
    for__ = (real) (*order);
/* Set the first value to avoid arithmetic problems */
    lanlut[1] = 1.f;
    i__1 = *npix;
    for (i__ = 2; i__ <= i__1; ++i__) {
	poff = ((real) i__ - 1.f) * 3.141592653f * *del;
	if (poff < for__ * 3.141592653f) {
	    lanlut[i__] = sin(poff) / poff * sin(poff / for__) / (poff / 
		    for__);
	} else {
	    lanlut[i__] = 0.f;
	}
    }
    return 0;
} /* filalu_ */

/* Subroutine */ int chover_(doublereal *y, integer *margin, integer *dnx, 
	integer *dny, integer *onx, integer *ony, doublereal *xsh, doublereal 
	*ysh, doublereal *rot, doublereal *scale, char *align, logical *
	rotfir, logical *secpar, doublereal *xsh2, doublereal *ysh2, 
	doublereal *rot2, doublereal *xscale, doublereal *yscale, char *shfr2,
	 logical *rotf2, logical *usewcs, doublereal *wcsin, doublereal *
	wcsout, integer *coty, integer *conum, doublereal *xco, doublereal *
	yco, logical *disim, real *pxg, real *pyg, integer *xgdim, integer *
	ygdim, doublereal *ofrac, integer *x1, integer *x2, ftnlen align_len, 
	ftnlen shfr2_len)
{
    /* System generated locals */
    integer pxg_dim1, pxg_offset, pyg_dim1, pyg_offset, i__1, i__2;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8;

    /* Local variables */
    static logical logo[21];
    static integer nhit, last;
    static doublereal xval[21];
    static integer step;
    static doublereal yval[21], xout[21], yout[21];
    static integer i__, first, nmiss, np;
    extern /* Subroutine */ int drival_(doublereal *, doublereal *, integer *,
	     integer *, integer *, integer *, integer *, logical *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, char *, 
	    logical *, logical *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, char *, logical *, logical *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    doublereal *, logical *, real *, real *, integer *, integer *, 
	    doublereal *, doublereal *, ftnlen, ftnlen);


/* Check how much of a line will overlap an output image, if any, */
/* after applying the standard drizzle transformation. */

/* This is intended to allow the number of points which are needlessly */
/* drizzled outside the output image to be minimized. */

/* Richard Hook, ST-ECF at STScI, April 2003 */

/* Supplied and returned */
/* Secondary geometrical parameters, added in V1.5 */
/* Local variables */
/* Loop along line */
    /* Parameter adjustments */
    --wcsin;
    --wcsout;
    --yco;
    --xco;
    pyg_dim1 = *xgdim;
    pyg_offset = 1 + pyg_dim1 * 1;
    pyg -= pyg_offset;
    pxg_dim1 = *xgdim;
    pxg_offset = 1 + pxg_dim1 * 1;
    pxg -= pxg_offset;

    /* Function Body */
    if (*dnx < 21) {
	step = 1;
    } else {
	step = *dnx / 10;
    }
    first = 0;
    last = 0;
    np = 0;
    i__1 = *dnx;
    i__2 = step;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	++np;
	xval[np - 1] = (doublereal) i__;
	yval[np - 1] = *y;
    }
/* Check end point */
    if (xval[np - 1] < (doublereal) (*dnx)) {
	xval[np] = (doublereal) (*dnx);
	yval[np] = *y;
	++np;
    }
/* Transform */
    drival_(xval, yval, &np, dnx, dny, onx, ony, &c_false, xsh, ysh, rot, 
	    scale, align, rotfir, secpar, xsh2, ysh2, rot2, xscale, yscale, 
	    shfr2, rotf2, usewcs, &wcsin[1], &wcsout[1], coty, conum, &xco[1],
	     &yco[1], disim, &pxg[pxg_offset], &pyg[pyg_offset], xgdim, ygdim,
	     xout, yout, (ftnlen)8, (ftnlen)8);
/* Check where the overlap starts and ends */
    i__2 = np;
    for (i__ = 1; i__ <= i__2; ++i__) {
	logo[i__ - 1] = FALSE_;
    }
    i__2 = np - 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
	d__1 = xout[i__ - 1], d__2 = xout[i__];
/* Computing MIN */
	d__3 = xout[i__ - 1], d__4 = xout[i__];
/* Computing MAX */
	d__5 = yout[i__ - 1], d__6 = yout[i__];
/* Computing MIN */
	d__7 = yout[i__ - 1], d__8 = yout[i__];
	if (max(d__1,d__2) >= 1.f - *margin && min(d__3,d__4) <= (doublereal) 
		(*onx + *margin) && max(d__5,d__6) >= 1.f - *margin && min(
		d__7,d__8) <= (doublereal) (*ony + *margin)) {
	    logo[i__ - 1] = TRUE_;
	    logo[i__] = TRUE_;
	}
    }
    nhit = 0;
    i__2 = np;
    for (i__ = 1; i__ <= i__2; ++i__) {
	if (logo[i__ - 1]) {
	    ++nhit;
	}
    }
    nmiss = np - nhit;
    if (nhit == 0) {
	*ofrac = 0.;
	*x1 = 0;
	*x2 = 0;
	return 0;
    } else {
	i__2 = np;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (logo[i__ - 1]) {
		first = i__;
		goto L77;
	    }
	}
L77:
	for (i__ = np; i__ >= 1; --i__) {
	    if (logo[i__ - 1]) {
		last = i__;
		goto L88;
	    }
	}
L88:
	*ofrac = (doublereal) nhit / (doublereal) (nhit + nmiss);
    }
    if (first > 1) {
	*x1 = (integer) xval[first - 2];
    } else {
	*x1 = (integer) xval[0];
    }
    if (last < np) {
	*x2 = (integer) xval[last];
    } else {
	*x2 = (integer) xval[last - 1];
    }
    return 0;
} /* chover_ */

doublereal ginter_(real *x, real *y, real *data, integer *nx, integer *ny, 
	real *lut, integer *nlut, real *space, integer *nbox, real *misval)
{
    /* System generated locals */
    integer data_dim1, data_offset, i__1, i__2;
    real ret_val, r__1, r__2;

    /* Builtin functions */
    integer i_nint(real *);

    /* Local variables */
    static integer xoff, yoff;
    static real luty;
    static integer i__, j, ixe, iye, ixs, iys;
    static real sum;


/* General interpolation using a lookup table */

/* The function returns the interpolated value at position X,Y */
/* in the array DATA. */

/* The array LUT is a one-dimensional, over-sampled, look up */
/* table of weights to be used for interpolating. */

/* Experimental version, Richard Hook, February 2004 */

/* Supplied and returned values */
/* Local variables */
/* First check for being close to the edge and if so return the missing */
/* value */
    /* Parameter adjustments */
    data_dim1 = *nx;
    data_offset = 1 + data_dim1 * 1;
    data -= data_offset;
    --lut;

    /* Function Body */
    ixs = i_nint(x) - *nbox;
    ixe = i_nint(x) + *nbox;
    iys = i_nint(y) - *nbox;
    iye = i_nint(y) + *nbox;
    if (ixs < 1 || ixe > *nx || iys < 1 || iye > *ny) {
	ret_val = *misval;
	return ret_val;
    }
/* Loop over the box, which is assumed to be scaled appropriately */
    sum = 0.f;
    i__1 = iye;
    for (j = iys; j <= i__1; ++j) {
	r__2 = (r__1 = (*y - (real) j) / *space, dabs(r__1));
	yoff = i_nint(&r__2) + 1;
	luty = lut[yoff];
	i__2 = ixe;
	for (i__ = ixs; i__ <= i__2; ++i__) {
	    r__2 = (r__1 = (*x - (real) i__) / *space, dabs(r__1));
	    xoff = i_nint(&r__2) + 1;
	    sum += data[i__ + j * data_dim1] * lut[xoff] * luty;
	}
    }
    ret_val = sum;
    return ret_val;
} /* ginter_ */

/* Subroutine */ int set1r_(real *a, real *arr, integer *npix)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__;

/*  This routine replaces the SPP routine 'amovkr', a routine which */
/*  replaces the values of the 1-D array ARR with the value given */
/*  as VAL, such that ARR(I) = A. */
/*   NPIX is the number of elements in the array ARR */

/*  Warren Hack, 17 June 2004 */

/* Supplied: */
/* Local variables: */
    /* Parameter adjustments */
    --arr;

    /* Function Body */
    i__1 = *npix;
    for (i__ = 1; i__ <= i__1; ++i__) {
	arr[i__] = *a;
    }
    return 0;
} /* set1r_ */

/* Subroutine */ int wsumr_(real *a, real *b, real *c__, integer *npix, real *
	w1, real *w2)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__;


/*  WSUM - Weighted sum of 2 real vectors. */
/*        Replaces the call to the SPP routine 'awsur', where */
/*           awsur(a,b,c,npix,k1,k2) : c(i) = k1*a(i) + k2*b(i) */

/*  Warren Hack, 15 June 2004 */

/* Supplied: */
/* Local variables: */
    /* Parameter adjustments */
    --c__;
    --b;
    --a;

    /* Function Body */
    i__1 = *npix;
    for (i__ = 1; i__ <= i__1; ++i__) {
	c__[i__] = a[i__] * *w1 + b[i__] * *w2;
    }
    return 0;
} /* wsumr_ */

