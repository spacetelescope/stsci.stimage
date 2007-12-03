      REAL FUNCTION TDRIZ(DATA,WEI,NDAT,NCOU,NCON,UNIQID,YSTART,XMIN,
     : YMIN,NX,NY,DNY,ONX,ONY,XSH,YSH,SHFTFR,SHFTUN,DROT,SCALE,
     : XSH2,YSH2,XSCALE,YSCALE,ROT2, SHFR2,
     : PXG,PYG,XGDIM,YGDIM,ALIGN,
     : PFRACT,KERNEL,COEFFS,INUN,EXPIN,WTSCL,FILSTR,
     : WCS,VFLAG,CLEN,NMISS,NSKIP,VERS,ALPHA,BETA)
C
C Call Drizzle without use of IRAF interfaces.
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
C   NCON - a 2d INTEGER array with dimensions (ONX,ONY): the output
C          context array which is updated. It should be initialised
C          the first time around.
C
C   UNIQID - INTEGER: the unique processing ID for the input data
C
C   YSTART - INTEGER: index of the starting line for input data array
C                    relative to the entire image. 
C
C   NX,NY  - INTEGER: dimensions of the full input image 
C
C   DNY     - INTEGER: *Actual* number of rows in the input data array
C
C   ONX,ONY - INTEGERs: the dimensions of the output array
C
C   PXG,PYG - 2d REAL arrays with dimensions (XGDIM,YGDIM): the 
C           distortion image arrays. If no images are used, they
C           should default to blank 2x2 arrays.  
C
C   XSH,YSH,DROT,SCALE - REALs: the standard linear drizzle user
C                        parameters. DROT is in degrees.
C
C   SHFTFR - CHARACTER*8 f77 character string: WCS frame for shifts:
C            "input" (default) or "output"
C
C   SHFTUN - CHARACTER*8 f77 character string: specifies whether the
C            shift units are in terms of 
C            "input" (default) or "output" pixels
C
C   ALIGN  - CHARACTER*8 f77 character string: takes the values
C            'corner' or 'center' to determine the position of the
C            reference point for rotation.
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
C   INUN   - CHARACTER*8 f77 character string: the units for the input
C            image: "cps" or "counts" (default)
C
C   EXPIN  - REAL: exposure time for input image, to allow scaling
C            of data to counts/sec.
C
C   WTSCL  - REAL: scaling factor applied to input weighting image
C
C   FILSTR - CHARACTER*80: pixel value to use for filling empty output pixels
C            'INDEF' (default) or str(float value).  If 'INDEF', do not fill.
C
C   WCS    - DOUBLE PRECISION: input WCS (corrected in-place by UPWCS)
C
C   VFLAG  - INTEGER: 1=verbose, anything else not.
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
C  f77 -c idldriz.f tdriz.f drutil.f drcall.f -pic -G
C  f77 -o idldriz.so drcall.o drutil.o idldriz.o tdriz.o -G -lF77 -lM77 -lsunmath -lm -lc
C
C If just being called directly from some f77 application it is
C only necessary to link in the following:
C
C  f77 -o app app.f tdriz.f drutil.f drcall.f
C
C where "app" is the top-level application.
C
C Notes:
C
C  - Some parameters, not included in the calling sequence, are hardcoded
C below in the section marked below.
C
C  - Secondary paramFILSTReters are not supported at present
C
C  - All the weighting information must be supplied in the WEI array.
C    Ie, wt_scl=1.0.
C
C  - Context images FILSTRare not currently supported
C
C  Test version, Richard Hook, ST-ECF, STScI, September 2002
C  Improved documentation, Richard Hook, ST-ECF, STScI, November 2002
C
C Modified to run under Linux, Richard Hook, ST-ECF, STScI, April 2003
C
C F2PY control codes added by WJH, 18 Dec 2003
C
C
Cf2py integer intent(hide),depend(data) :: nx = shape(data,0)
Cf2py integer intent(hide),depend(data) :: ny = shape(data,1)
Cf2py integer intent(hide),depend(ndat) :: onx = shape(ndat,0)
Cf2py integer intent(hide),depend(ndat) :: ony = shape(ndat,1)
Cf2py real intent(in,c) :: data, wei
Cf2py real intent(in,out):: ndat,ncou
Cf2py real intent(in) :: xsh, ysh, drot, scale, pfract
Cf2py real intent(in) :: wtscl, expin
Cf2py character*8 intent(in) :: align
Cf2py character*8 intent(in) :: kernel
Cf2py character*80 :: coeffs
Cf2py integer intent(in) :: vflag
C
      IMPLICIT NONE

      INTEGER VFLAG 

      LOGICAL VERBOSE
      COMMON /VERBOSE/VERBOSE

      INTEGER NX,NY,ONX,ONY,DNY
      INTEGER IDD,ISTAT,XMIN,YMIN,XMAX,YMAX,YSTART
      INTEGER NMISS,NSKIP

C Distortion coefficients and related
      INTEGER COTY,COMAX,CONUM
      PARAMETER (COMAX=100)
      DOUBLE PRECISION LAM,XCO(COMAX),YCO(COMAX)
      CHARACTER*80 COEFFS, FILSTR
      INTEGER CLEN

C Maximum size (in X) of the input image
      INTEGER MAXNX
      PARAMETER (MAXNX=100000) 

      REAL DATA(NX,DNY),WEI(NX,DNY)
      REAL NDAT(ONX,ONY),NCOU(ONX,ONY)
      INTEGER NCON(ONX,ONY)
    
C Distortion image arrays
      REAL PXG(XGDIM,YGDIM), PYG(XGDIM,YGDIM)
      INTEGER XGDIM,YGDIM
      LOGICAL DISIM

      DOUBLE PRECISION WCS(8)
      DOUBLE PRECISION WCSOUT(8)

      DOUBLE PRECISION  XI(MAXNX,4),YI(MAXNX,4),XO(MAXNX,4),YO(MAXNX,4)
      DOUBLE PRECISION  XIB(MAXNX),YIB(MAXNX),XOB(MAXNX),YOB(MAXNX)
      DOUBLE PRECISION  XSH,YSH,ROT,DROT,SCALE,PFRACT
      REAL WTSCL,EXPIN,FILVAL
      INTEGER NEN,UNIQID

C Context related things
      LOGICAL CON
      LOGICAL BITCON
      LOGICAL FILL

C We give these minimal dimensions  
      INTEGER DONE(1,1)  
      INTEGER MAXEN,MAXIM
      PARAMETER (MAXIM=100,MAXEN=100)
      INTEGER INTAB(MAXIM,MAXEN)

      CHARACTER*8 ALIGN,KERNEL,INUN,SHFTFR,SHFTUN
      LOGICAL ROTFIR,UPDATE,USEWEI,INCPS,USEWCS,NOOVER

C Secondary geometrical parameters
      LOGICAL SECPAR
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE,ALPHA,BETA
      CHARACTER*8 SHFR2
      LOGICAL ROTF2
      CHARACTER*50 VERS

C Keep quiet
      IF(VFLAG.EQ.1) THEN
         VERBOSE=.TRUE.
      ELSE
         VERBOSE=.FALSE.
      ENDIF

C Define Version ID
      VERS = 'Callable DRIZZLE Version 0.7 (4th Apr 2005)'
C Announce
      CALL UMSPUT(VERS,1,0,ISTAT)

C Get geometric distortion coefficients
C
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
C      XMIN=1
      XMAX=ONX
C      YMIN=1
      YMAX=ONY
      NOOVER=.FALSE.
      
C All weighting is done by the weight array
C      WTSCL=1.0
C      INCPS=.FALSE.
C Check the switch value for whether to read in CPS or
C counts (latter is the default)
      IF(INUN(1:3).EQ.'cps') THEN
         INCPS=.TRUE.
      ELSE
         INCPS=.FALSE.
      ENDIF

C Input exposure assumed to be 1.0s
C      EXPIN=1.0
C      ROTFIR=.TRUE.
C Check the parameter specifying which frames shifts are applied
      IF(SHFTFR(1:5).EQ.'input') THEN
         ROTFIR=.FALSE.
      ELSE
         ROTFIR=.TRUE.
      ENDIF
      
C Convert the shift units if necessary
      IF(SHFTUN(1:6).EQ.'output') THEN
         XSH=XSH*SCALE
         YSH=YSH*SCALE
      ENDIF

C Convert the rotation to radians
      ROT=DROT*3.141592653/180.0

C Convert the secondary parameters into suitable units
      ROT2=ROT2*3.14159265358979D0/180.0D0
      IF(SHFR2(1:5).eq.'input') THEN
         ROTF2=.FALSE.
      ELSE
         ROTF2=.TRUE.
      ENDIF

C Give a warning message if secondary parameters will have an effect
      IF(XSCALE.NE.1.0D0 .OR. YSCALE.NE.1.0D0 .OR.
     :   XSH2.NE.0.0D0 .OR. YSH2.NE.0.0D0 .OR. ROT2.NE.0.0D0) THEN
       CALL UMSPUT(
     : '! Warning, secondary geometric transform is being used',
     :               1,0,ISTAT)
        SECPAR=.TRUE.
      ELSE
        SECPAR=.FALSE.
      ENDIF


      UPDATE=.TRUE.
      USEWEI=.TRUE.
      USEWCS=.FALSE.
      CON=.TRUE.
      BITCON = .TRUE.

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

C The arrays NDAT and NCOU will have been updated
C Update the WCS, if it needs to be updated.
C  Only need to do once per image, not once per section.
      IF (YSTART .EQ. 0) THEN
      CALL UPWCS(WCS,WCS,NX,NY,ONX,ONY,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,
     :            YGDIM,ALPHA,BETA)
      ENDIF

C Return the value of status
      TDRIZ=FLOAT(ISTAT)
          
      END
