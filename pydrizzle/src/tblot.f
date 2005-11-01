      SUBROUTINE TBLOT(DATA,NDAT,XMIN,XMAX,YMIN,YMAX,
     :                  DNX,DNY,ONX,ONY,XSH,YSH,DROT, SCALE,KSCALE,
     :                 XSH2,YSH2,XSCALE,YSCALE,ROT2, SHFR2,  
     :                 PXG,PYG,XGDIM,YGDIM,
     :                 ALIGN,INTERP,COEFFS,EF,MISVAL,SINSCL,CLEN,VFLAG)
C++
C
C TBLOT.F "Reverse drizzling" 
C IDL-callable version.
C
C   Richard Hook, ST-ECF/STScI, 23rd September 2002
C
C Added MISVAL parameter - September 2003
C--
C F2PY control codes added by WJH, 22 Nov 2002
C
Cf2py intent(in,c) data
Cf2py intent(in,out) ndat
Cf2py real intent(in) :: xsh, ysh, drot, scale
Cf2py character*8 intent(in) :: align
Cf2py character*7 intent(in) :: interp
Cf2py character*80 :: coeffs
Cf2py real intent(in) :: misval
Cf2py integer intent(in) :: vflag
Cf2py integer intent(hide),depend(data) :: dnx = shape(data,0)
Cf2py integer intent(hide),depend(data) :: dny = shape(data,1)
Cf2py integer intent(hide),depend(ndat) :: onx = shape(ndat,0)
Cf2py integer intent(hide),depend(ndat) :: ony = shape(ndat,1)
C
      IMPLICIT NONE

C Local variables
      INTEGER ISTAT,VFLAG
      INTEGER DNX,DNY,ONX,ONY
      INTEGER XMIN,XMAX,YMIN,YMAX
      INTEGER IDD
      INTEGER INTYPE
      REAL SINSCL, KSCALE
      REAL EF

C The main data and output arrays
      REAL DATA(XMAX-XMIN+1,YMAX-YMIN+1),NDAT(ONX,ONY)

C The distortion image arrays, and their size
C These arrays should be 2x2 blank arrays if no images are used
      REAL PXG(XGDIM,YGDIM), PYG(XGDIM,YGDIM)
      INTEGER XGDIM,YGDIM
      LOGICAL DISIM
      
C Buffers
      INTEGER MAXNX
      PARAMETER (MAXNX=10000)

      DOUBLE PRECISION XIN(MAXNX),XOUT(MAXNX),YIN(MAXNX),YOUT(MAXNX)

      DOUBLE PRECISION WCSIN(8),WCSOUT(8)
      CHARACTER*45 VERS
      CHARACTER*7 INTERP
      INTEGER COTY,COMAX,CONUM
      PARAMETER (COMAX=100)
      DOUBLE PRECISION  XCO(COMAX),YCO(COMAX)
      REAL MISVAL

C Geometrical parameters, the standard set
      DOUBLE PRECISION  SCALE,DROT,ROT,XSH,YSH,LAM
      DOUBLE PRECISION  XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*80 COEFFS
      CHARACTER*8 SHFR2,ALIGN,GEOMOD
      INTEGER CLEN

C Constants
      REAL PI
      PARAMETER (PI=3.1415926536)

C Logical flags
      LOGICAL VERBOSE
      LOGICAL ROTFIR
      LOGICAL INCPS
      LOGICAL USEWCS
      LOGICAL ROTF2
      LOGICAL SECPAR

C Verbose common
      COMMON /VERBOSE/VERBOSE

C-- Start of executable code

C Keep quiet
      IF(VFLAG.EQ.1) THEN
         VERBOSE=.TRUE.
      ELSE
         VERBOSE=.FALSE.
      ENDIF

C First announce the version
      VERS='Callable BLOT Version 0.6 (1st Nov 2005)'

      CALL UMSPUT('+ '//VERS,1,0,ISTAT)

      USEWCS=.FALSE.
C      ROTFIR=.FALSE.
      ROTFIR=.TRUE.
      SECPAR=.FALSE.
      INCPS=.TRUE.
C      EF=1.0

C Convert the rotation to radians
      ROT=DROT*PI/180.0
    
C Convert the secondary parameters into suitable units
      ROT2=ROT2*PI/180.0D0
      IF(SHFR2.EQ.'input') THEN
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


C Check for invalid scale
      IF(SCALE.EQ.0.0) THEN
         CALL UMSPUT('! Invalid scale',1,0,ISTAT)
         GO TO 99
      ENDIF

C Get the geometric distortion coefficient information
C 
      CALL GETGEO(COEFFS,IDD,LAM,COTY,COMAX,CONUM,
     :                  XCO,YCO,CLEN,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :  '! Error, failed to get geometric distortion coefficients',
     :  1,0,ISTAT)
          GO TO 99
      ENDIF

C Set DISIM logical based on whether distortion images have
C  been passed in for use or not.
      IF (XGDIM .EQ. 2 .AND. YGDIM .EQ. 2) THEN
         DISIM = .FALSE.
      ELSE
         DISIM = .TRUE.
      ENDIF
      
C Before we start also get the interpolation type and convert
C to a numerical code
      IF(INTERP(1:7).EQ.'nearest') THEN
         INTYPE=1
      ELSE IF(INTERP(1:6).EQ.'linear') THEN
         INTYPE=2
      ELSE IF(INTERP(1:5).EQ.'poly3') THEN
         INTYPE=3
      ELSE IF(INTERP(1:5).EQ.'poly5') THEN
         INTYPE=4
C      ELSE IF(INTERP(1:7).EQ.'spline3') THEN
C         INTYPE=5
      ELSE IF(INTERP(1:4).EQ.'sinc') THEN
         INTYPE=6
      ELSE IF(INTERP(1:5).EQ.'lsinc') THEN
         INTYPE=7
      ELSE IF(INTERP(1:4).EQ.'lan3') THEN
         INTYPE=100
      ELSE IF(INTERP(1:4).EQ.'lan5') THEN
         INTYPE=105
      ELSE 
         CALL UMSPUT('! Invalid interpolant specified',
     :               1,0,ISTAT)
          CALL UMSPUT(INTERP,1,0,ISTAT)
         GO TO 99
      ENDIF  

C Now do the actual combination using interpolation
      CALL DOBLOT(DATA,NDAT,INTYPE,SINSCL,KSCALE,MISVAL,
     :      XMIN,XMAX,YMIN,YMAX,
     :      DNX,DNY,ROTFIR,EF,ALIGN,
     :      ONX,ONY,COTY,CONUM,XCO,YCO,
     :      DISIM, PXG,PYG,XGDIM,YGDIM,
     :      XIN,YIN,XOUT,YOUT,
     :      SCALE,ROT,XSH,YSH,USEWCS,WCSIN,WCSOUT,GEOMOD,
     :      SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2)

 99   CONTINUE
      END
