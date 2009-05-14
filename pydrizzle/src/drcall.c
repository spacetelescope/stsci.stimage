/* drcall.f -- translated by f2c (version 20060506).
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

/* Table of constant values */

static integer c__1 = 1;
static integer c__0 = 0;
static integer c__2 = 2;

/* DRCALL - emulation routines for F77/VOS routines found in DRUTIL.F */

/* These routines are needed to allow the callable version of drutil to */
/* link without needing F77/VOS routines. */

/* Richard Hook, ST-ECF/STScI, September 2002 */
/* Subroutine */ int umsput_(char *line, integer *d1, integer *d2, integer *
	istat, ftnlen line_len)
{
    /* Builtin functions */
    integer s_cmp(char *, char *, ftnlen, ftnlen), s_wsfe(cilist *), do_fio(
	    integer *, char *, ftnlen), e_wsfe(void);

    /* Fortran I/O blocks */
    static cilist io___1 = { 1, 6, 0, "(1X,A,/)", 0 };
    static cilist io___2 = { 1, 6, 0, "(1X,A,/)", 0 };



/* Just write a line of text to the screen. The second and */
/* third parameters are ignored. */

/* Always write an error message (starting !), except if */
/* it is a warning */
    if (*(unsigned char *)line == '!' && s_cmp(line + 2, "War", (ftnlen)3, (
	    ftnlen)3) != 0) {
	*istat = s_wsfe(&io___1);
	if (*istat != 0) {
	    goto L100001;
	}
	*istat = do_fio(&c__1, line, line_len);
	if (*istat != 0) {
	    goto L100001;
	}
	*istat = e_wsfe();
L100001:
	;
    }
    return 0;
} /* umsput_ */

/* Subroutine */ int ufglin_(integer *lun, char *line, integer *istat, ftnlen 
	line_len)
{
    /* Builtin functions */
    integer s_rsfe(cilist *), do_fio(integer *, char *, ftnlen), e_rsfe(void);

    /* Fortran I/O blocks */
    static cilist io___3 = { 1, 0, 1, "(A)", 0 };



/* Read a line of text from an already opened text file. */

    io___3.ciunit = *lun;
    *istat = s_rsfe(&io___3);
    if (*istat != 0) {
	goto L100003;
    }
    *istat = do_fio(&c__1, line, line_len);
    if (*istat != 0) {
	goto L100003;
    }
    *istat = e_rsfe();
L100003:
    return 0;
} /* ufglin_ */

/* Subroutine */ int ufopen_(char *file, integer *flag__, integer *lun, 
	integer *istat, ftnlen file_len)
{
    /* System generated locals */
    olist o__1;

    /* Builtin functions */
    integer f_open(olist *);

    /* Local variables */
    extern /* Subroutine */ int umsput_(char *, integer *, integer *, integer 
	    *, ftnlen);


/* Open a text file. The FLAG parameter controls the iomode. */
/* Only 1=readonly and 2=read-write, 4=append and 5=new are */
/* supported. */

    if (*flag__ == 1) {
	o__1.oerr = 1;
	o__1.ounit = 23;
	o__1.ofnmlen = file_len;
	o__1.ofnm = file;
	o__1.orl = 0;
	o__1.osta = "OLD";
	o__1.oacc = 0;
	o__1.ofm = 0;
	o__1.oblnk = 0;
	*istat = f_open(&o__1);
    } else if (*flag__ == 2) {
	o__1.oerr = 1;
	o__1.ounit = 23;
	o__1.ofnmlen = file_len;
	o__1.ofnm = file;
	o__1.orl = 0;
	o__1.osta = "OLD";
	o__1.oacc = 0;
	o__1.ofm = 0;
	o__1.oblnk = 0;
	*istat = f_open(&o__1);
/* CC      ELSE IF(FLAG.EQ.4) THEN */
/* CC         OPEN(23,FILE=FILE,STATUS='OLD',ACCESS='APPEND',IOSTAT=ISTAT) */
    } else if (*flag__ == 5) {
	o__1.oerr = 1;
	o__1.ounit = 23;
	o__1.ofnmlen = file_len;
	o__1.ofnm = file;
	o__1.orl = 0;
	o__1.osta = "NEW";
	o__1.oacc = 0;
	o__1.ofm = 0;
	o__1.oblnk = 0;
	*istat = f_open(&o__1);
    } else {
	umsput_("! Invalid file iomode specified", &c__1, &c__0, istat, (
		ftnlen)31);
    }
    *lun = 23;
    return 0;
} /* ufopen_ */

/* Subroutine */ int ufclos_(integer *lun, integer *istat)
{
    /* System generated locals */
    cllist cl__1;

    /* Builtin functions */
    integer f_clos(cllist *);


/* Close an open file on logical unit LUN */

    cl__1.cerr = 0;
    cl__1.cunit = *lun;
    cl__1.csta = 0;
    f_clos(&cl__1);
    return 0;
} /* ufclos_ */

/* Subroutine */ int getgeo_(char *coeffs, integer *idd, real *lam, integer *
	coty, integer *comax, integer *conum, real *xco, real *yco, integer *
	clen, integer *istat, ftnlen coeffs_len)
{
    /* System generated locals */
    address a__1[2];
    integer i__1, i__2[2];
    char ch__1[108];

    /* Builtin functions */
    integer s_cmp(char *, char *, ftnlen, ftnlen);
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer i__, lun;
    static logical noco;
    extern /* Subroutine */ int getco_(integer *, real *, integer *, integer *
	    , integer *, real *, real *, integer *), ufclos_(integer *, 
	    integer *), ufopen_(char *, integer *, integer *, integer *, 
	    ftnlen), umsput_(char *, integer *, integer *, integer *, ftnlen);


/* Get the geometrical distortion information, either from */
/* a file or the header. */

/* This new routine, October 2000 */
/* Modified in drizzle 2.1 for more flexible coefficients */

/* Interpret the coefficients file name and act appropriately */
    /* Parameter adjustments */
    --yco;
    --xco;

    /* Function Body */
    noco = FALSE_;
    if (s_cmp(coeffs, " ", (ftnlen)80, (ftnlen)1) == 0) {
	noco = TRUE_;
    }
/* Now check for the special string "header" */
    if (s_cmp(coeffs, "header", (ftnlen)80, (ftnlen)6) == 0) {
	umsput_("! This version does not support the header option", &c__1, &
		c__0, istat, (ftnlen)49);
	*istat = 1;
	return 0;
    }
/* Set default values */
    if (noco) {
	i__1 = *comax;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    xco[i__] = 0.f;
	    yco[i__] = 0.f;
	}
	*conum = 1;
	*coty = 0;
    } else {
/* Open the coefficients file */
	ufopen_(coeffs, &c__1, &lun, istat, (ftnlen)80);
	if (*istat != 0) {
	    umsput_("! Unable to open coefficients file", &c__1, &c__0, istat,
		     (ftnlen)34);
	    *istat = 1;
	    return 0;
	} else {
/* Writing concatenation */
	    i__2[0] = 28, a__1[0] = "-Opening coefficients file: ";
	    i__2[1] = *clen, a__1[1] = coeffs;
	    s_cat(ch__1, a__1, i__2, &c__2, (ftnlen)108);
	    umsput_(ch__1, &c__1, &c__0, istat, *clen + 28);
	}
/* Read in the coefficients from the file */
	getco_(&lun, lam, coty, comax, conum, &xco[1], &yco[1], istat);
	if (*istat != 0) {
	    umsput_("! Unable to read coefficients", &c__1, &c__0, istat, (
		    ftnlen)29);
	    ufclos_(&lun, istat);
	    *istat = 1;
	    return 0;
	}
/* Close the coefficients file */
	ufclos_(&lun, istat);
    }
/* Set a good status return */
    *istat = 0;
    return 0;
} /* getgeo_ */

