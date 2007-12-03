      REAL FUNCTION TWDRIZ(DATA,WEI,NDAT,NCOU,YSTART,
     : NX,NY,DNY,ONX,ONY,WCS,WCSOUT,PXG,PYG,XGDIM,YGDIM,
     : PFRACT,KERNEL,COEFFS,FILSTR,VFLAG,CLEN,NMISS,NSKIP,VERS)
C
C Call Wdrizzle without use of IRAF interfaces.
C
C Supplied:
C
C   DATA - a 2d REAL array with dimensions (NX,NY): the input
C          data.
C
C   WEI - a 2d REAL array with dimensions (NX,NY): the input
C          weight array.
C
C   NDAT - a 2d REAL array with dimensions (ONX,ONY): the output
C          data array which is updated. It should be initialised
C          the first time around.
C
C   NCOU - a 2d REAL array with dimensions (ONX,ONY): the output
C          weight array which is updated. It should be initialised
C          the first time around.
C
C   NX,NY,ONX,ONY - INTEGERs: the dimensions of the arrays described
C                   above.
C
C   WCS,WCSOUT - double precision arrays, 8 elements in each.
C                Input and output WCS.
C
C   PFRACT - REAL: the drizzle "pixfrac" parameter.
C
C   KERNEL - CHARACTER*8 f77 character string: the type of kernel
C            used to distribute weight onto the output. Currently
C            supports "square","gaussian","tophat","point","turbo",
C            "lanczos2" and "lanczos3".
C
C   COEFFS - CHARACTER*80 f77 character string: the name of the
C            text file containing distortion correction coefficients.
C
C Returned: 
C
C    a REAL status is returned: the status return of the DOBOX
C    routine
C
C This was developed as a simple interface to the lower-level
C drizzle utilities which could be invoked without the need for
C an IRAF environment.
C
C If it is intended to be called from IDL the following applies:
C
C All parameters are passed from IDL using the "call external"
C mechanism. The intermediate file is idldriz.f. The linking looks
C something like this on Solaris 8:
C
C  f77 -c idlwdriz.f tdriz.f drutil.f drcall.f -pic -G
C  f77 -o idlwdriz.so drcall.o drutil.o idldriz.o tdriz.o -G -lF77 -lM77 -lsunmath -lm -lc
C
C If just being called directly from some f77 application it is
C only necessary to link in the following:
C
C  f77 -o app app.f twdriz.f drutil.f drcall.f
C
C where "app" is the top-level application.
C
C Notes:
C
C  - Some parameters, not included in the calling sequence, are hardcoded
C below in the section marked below.
C
C  - Secondary parameters are not supported at present
C
C  - All the weighting information must be supplied in the WEI array.
C    Ie, wt_scl=1.0.
C
C  - Context images are not currently supported
C
C  Test version, Richard Hook, ST-ECF, STScI, September 2002
C  Improved documentation, Richard Hook, ST-ECF, STScI, November 2002
C
C Modified to run under Linux, Richard Hook, ST-ECF, STScI, April 2003
C
C Wdrizzle version, ST-ECF, STScI, September 2003
C Added VFLAG for verbosity, STScI/ECF, October 2003
C
      IMPLICIT NONE

      INTEGER NX,NY,ONX,ONY,DNY
      INTEGER IDD,ISTAT,XMIN,YMIN,XMAX,YMAX,YSTART
      INTEGER VFLAG
      INTEGER NMISS,NSKIP

C Distortion coefficients and related
      INTEGER COTY,COMAX,CONUM
      PARAMETER (COMAX=100)
      REAL LAM,XCO(COMAX),YCO(COMAX)
      CHARACTER*80 COEFFS, FILSTR
      INTEGER CLEN

C Distortion image arrays
      REAL PXG(XGDIM,YGDIM), PYG(XGDIM,YGDIM)
      INTEGER XGDIM,YGDIM
      LOGICAL DISIM

C Verbose flag
      LOGICAL VERBOSE
      COMMON /VERBOSE/VERBOSE

C Maximum size (in X) of the input image
      INTEGER MAXNX
      PARAMETER (MAXNX=100000) 

      REAL DATA(NX,DNY),WEI(NX,DNY)
      REAL NDAT(ONX,ONY),NCOU(ONX,ONY)

      DOUBLE PRECISION WCS(8)
      DOUBLE PRECISION WCSOUT(8)

      REAL XI(MAXNX,4),YI(MAXNX,4),XO(MAXNX,4),YO(MAXNX,4)
      REAL XIB(MAXNX),YIB(MAXNX),XOB(MAXNX),YOB(MAXNX)
      REAL XSH,YSH,ROT,DROT,SCALE
      REAL WTSCL,EXPIN,PFRACT,FILVAL
      INTEGER NEN,UNIQID

C Context related things
      LOGICAL CON
      LOGICAL BITCON
      LOGICAL FILL

C We give these minimal dimensions
      INTEGER NCON(1,1)  
      INTEGER DONE(1,1)  
      INTEGER MAXEN,MAXIM
      PARAMETER (MAXIM=100,MAXEN=100)
      INTEGER INTAB(MAXIM,MAXEN)

      CHARACTER*8 ALIGN,KERNEL
      LOGICAL ROTFIR,UPDATE,USEWEI,INCPS,USEWCS,NOOVER

C Secondary geometrical parameters, added in V1.5
      LOGICAL SECPAR
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE,ALPHA,BETA
      CHARACTER*8 SHFR2
      LOGICAL ROTF2
      CHARACTER*50 VERS

C Check for verbose
      IF(VFLAG.EQ.1) THEN
         VERBOSE=.TRUE.
      ELSE
         VERBOSE=.FALSE.
      ENDIF

C Define Version ID
      VERS = 'Callable WDRIZZLE Version 0.5 (25rd June 2004)'
C Announce
      CALL UMSPUT(VERS,1,0,ISTAT)

C Get geometric distortion coefficients
      IDD = 0
      CALL GETGEO(COEFFS,IDD,LAM,COTY,COMAX,CONUM,
     :                  XCO,YCO,CLEN,ISTAT)
      IF(ISTAT.NE.0) RETURN

C Set DISIM logical based on whether distortion images have
C  been passed in for use or not.
      IF (XGDIM .EQ. 2 .AND. YGDIM .EQ. 2) THEN
         DISIM = .FALSE.
      ELSE
         DISIM = .TRUE.
      ENDIF
      
C Setup reasonable defaults for the drizzling
      XMIN=1
      XMAX=ONX
      YMIN=1
      YMAX=ONY
      NOOVER=.FALSE.

C All weighting is done by the weight array
      WTSCL=1.0
      INCPS=.TRUE.

C Input exposure assumed to be 1.0s
      EXPIN=1.0
      ROTFIR=.FALSE.

C Convert the rotation to radians
      ROT=DROT*3.141592653/180.0

C Secondary parameters are not currently supported
      SECPAR=.FALSE.
      UPDATE=.TRUE.
      USEWEI=.TRUE.

C Wdrizzle, hence...
      USEWCS=.TRUE.
      CON=.FALSE.
      UNIQID = 1

C Do the drizzling
      CALL DOBOX(DATA,WEI,NDAT,NCOU,NCON,DONE,NX,NY,DNY,
     :     YSTART,XMIN,XMAX,YMIN,YMAX,NOOVER,
     :     KERNEL,XI,XO,YI,YO,XIB,XOB,YIB,YOB,
     :     ONX,ONY,COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :     WTSCL,ALIGN,INCPS,EXPIN,
     :     PFRACT,SCALE,ROT,XSH,YSH,WCS,WCSOUT,ROTFIR,
     :     SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :     CON,BITCON,INTAB,MAXIM,MAXEN,NEN,UNIQID,
     :     UPDATE,USEWEI,USEWCS,ISTAT,NMISS,NSKIP,ALPHA,BETA)

C The arrays NDAT and NCOU will have been updated
C Check for meaningful values
      IF(FILSTR(1:5).NE.'INDEF'.AND.FILSTR(1:5).NE.'indef') THEN
         READ(FILSTR,*,IOSTAT=ISTAT) FILVAL
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Invalid filling value specified',
     :                  1,0,ISTAT)
            FILL=.FALSE.
         ELSE
            FILL=.TRUE.
         ENDIF
      ELSE
         FILL=.FALSE.
      ENDIF
      
C Put in the fill values (if defined)
      IF (FILL) CALL PUTFIL(NDAT,NCOU,XMAX-XMIN+1,
     :    YMAX-YMIN+1,FILVAL)

C Return the value of status
      TWDRIZ=FLOAT(ISTAT)

      END
