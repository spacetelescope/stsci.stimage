      SUBROUTINE WMETRC 
C++
C
C WMETRIC.F V1.1 Convert X,Y pixel position to sky coordinates
C
C This task uses coefficients held in a text file to define
C geometrical distortions.
C
C This code uses the F77/VOS for IRAF i/o.
C It may be compiled and linked using a command like:
C
C xc -p stsdas wmetric.f drutil.f x_wmetric.x <dir>libiraf77.a
C
C History:
C
C First quick version from WTRAXY, 5th May 2004    
C
C Modified to use common callable code, 25th June 2004, WJH
C
C--
      IMPLICIT NONE

C Local variables
      INTEGER ISTAT,NXIN,NYIN,NXOUT,NYOUT
      INTEGER COTY,CONUM,COMAX
      PARAMETER (COMAX=100)

      DOUBLE PRECISION XCO(COMAX),YCO(COMAX)
      DOUBLE PRECISION XIN,YIN,XOUT,YOUT,X(4),Y(4),YO(4),XO(4)
      DOUBLE PRECISION RA,DEC
      INTEGER IID,DATTYP,NDIMS,DDIMS(7)
      CHARACTER*80 INIM
      CHARACTER*132 CHARS
      CHARACTER*45 VERS
      CHARACTER*8 CTYPE1,CTYPE2,ALIGN,SHFR2

C Geometrical parameters, the standard set
      DOUBLE PRECISION SCALE,ROT,XSH,YSH,LAM
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      DOUBLE PRECISION WCSIN(8),WCSOUT(8)
      DOUBLE PRECISION PAM
      CHARACTER*80 COEFFS,XGEOIM,YGEOIM,XYLIST,LINE
      INTEGER PXG,PYG,LUN,XGDIM,YGDIM
      LOGICAL DISIM

      REAL MEMR(1)
      COMMON /MEM/MEMR

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793D0)

      LOGICAL ROTFIR
      LOGICAL USEWCS
      LOGICAL ROTF2
      LOGICAL SECPAR
      LOGICAL SILENT

C-- Start of executable code
      VERS='WMETRIC Version 1.1 (25th June 2004)'

C Get the noise flag
      CALL UCLGSB('silent',SILENT,ISTAT)

C First announce the version
      IF(.NOT.SILENT) CALL UMSPUT('+ '//VERS,1,0,ISTAT)

C Get all the geometrical parameters
      CALL UCLGST('coeffs',COEFFS,ISTAT)
      CALL UCLGSD('lambda',LAM,ISTAT)
      CALL UCLGST('xgeoim',XGEOIM,ISTAT)
      CALL UCLGST('ygeoim',YGEOIM,ISTAT)
      IF(XGEOIM.NE.' ' .OR. YGEOIM.NE.' ') THEN
         DISIM=.TRUE.
      ELSE 
         DISIM=.FALSE.
      ENDIF

C If so, check for image names (output is optional)
      CALL UCLGST('image',INIM,ISTAT)

C Initialisation
      USEWCS=.TRUE.
      XSH=0.0D0
      YSH=0.0D0
      ROT=0.0D0
      SECPAR=.FALSE.
      SCALE=1.0
      PXG=0
      PYG=0

C Open the files and read the WCS from the headers
      IF(INIM.EQ.' ') THEN
         CALL UMSPUT('! No input image specified',
     :                  1,0,ISTAT)
         GO TO 99
      ELSE
         CALL UIMOPN(INIM,1,IID,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Unable to open image',
     :                     1,0,ISTAT)
            GO TO 99
         ELSE
            CALL UMSPUT('-Reading WCS from '//INIM,1,0,ISTAT)
            CALL GETWCS(IID,WCSIN,CTYPE1,CTYPE2,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT(
     : '! Unable to get WCS from image header',
     :                     1,0,ISTAT)
               GO TO 99
            ENDIF

C Get the dimensions and use them
            CALL UIMGID(IID,DATTYP,NDIMS,DDIMS,ISTAT)
            IF(NDIMS.NE.2) THEN
               CALL UMSPUT(
     :   '! Input image is not two dimensional',
     :                          1,0,ISTAT)
               GO TO 99
            ELSE
               NXIN=DDIMS(1)
               NYIN=DDIMS(2)
            ENDIF
         ENDIF
      ENDIF

C Get the geometric distortion coefficient information
      CALL GETGEO(COEFFS,IDD,LAM,
     :            COTY,COMAX,CONUM,XCO,YCO,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :  '! WTRAXY: Error, failed to get distortion coefficients',
     :  1,0,ISTAT)
          GO TO 99
      ENDIF

C  Read in any distortion image corrections, if specified
C  PXG,PYG will default to 2x2 arrays when DISIM returns as FALSE.
C 
      CALL GGEOIM(XGEOIM,YGEOIM,PXG,PYG,XGDIM,YGDIM,DISIM)

C Either we have a list of X,Y positions or a single one
      CALL UCLGST('xylist',XYLIST,ISTAT)
      IF(XYLIST.NE.' ') THEN
         CALL UFOPEN(XYLIST,1,LUN,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Unable to open xylist file',1,0,
     :                  ISTAT)
            ISTAT=1
            RETURN
         ELSE
            CALL UMSPUT('-Opening xylist file: '//XYLIST,
     :                      1,0,ISTAT)
         ENDIF
      ENDIF

C Get the X,Y pixel position to be transformed 
      IF(XYLIST.EQ.' ') THEN
         CALL UCLGSD('xin',XIN,ISTAT)
         CALL UCLGSD('yin',YIN,ISTAT)
      ENDIF

C Setup an output WCS. This is fairly arbitrary and doesn't
C affect the result. We use 50mas pixels and North at the
C top
      WCSOUT(1)=WCSIN(1)
      WCSOUT(2)=WCSIN(2)
      WCSOUT(3)=WCSIN(3)
      WCSOUT(4)=WCSIN(4)
      WCSOUT(5)=-0.05D0/3600.0D0
      WCSOUT(6)=0.0D0
      WCSOUT(7)=0.0D0
      WCSOUT(8)=0.05D0/3600.0D0
      NXOUT=4096
      NYOUT=4096

C Write column headings for the xylist mode
      IF(XYLIST.NE.' ') THEN
         CALL UMSPUT(
     : '        X   (pixels)   Y          RA   (degrees)   Dec',
     :     1,0,ISTAT)
      ENDIF

C Transform the position
C Note that this is exactly the same as that in Drizzle
C Double precision version
      DO WHILE(.TRUE.)

         IF(XYLIST.NE.' ') THEN
            CALL UFGLIN(LUN,LINE,ISTAT)
            IF(ISTAT.NE.0) GO TO 88
            READ(LINE,*) XIN,YIN
         ENDIF
         
         CALL DRIVAL(XIN,YIN,1,NXIN,NYIN,NXOUT,NYOUT,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,
     :            DISIM,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,XOUT,YOUT)

C Transform a unit square to calculate the pixel area
         X(1)=XIN-0.5D0
         X(2)=XIN+0.5D0
         X(3)=XIN+0.5D0
         X(4)=XIN-0.5D0
         Y(1)=YIN-0.5D0
         Y(2)=YIN-0.5D0
         Y(3)=YIN+0.5D0
         Y(4)=YIN+0.5D0

         CALL DRIVAL(X,Y,4,NXIN,NYIN,NXOUT,NYOUT,.FALSE.,
     :            0.0D0,0.0D0,0.0D0,1.0D0,ALIGN,ROTFIR,
     :            .FALSE.,0.0D0,0.0D0,0.0D0,1.0D0,1.0D0,SHFR2,ROTF2,
     :            .FALSE.,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,
     :            .FALSE.,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,XO,YO)

         PAM=DABS(0.5*((XO(2)-XO(4))*(YO(1)-YO(3)) -
     :              (XO(1)-XO(3))*(YO(2)-YO(4))))

C Convert to RA,Dec using the output (tangent plane) WCS
         CALL XY2RD(XOUT,YOUT,RA,DEC,WCSOUT)

C Write out the result
         IF(XYLIST.NE.' ') THEN
            WRITE(CHARS,'(4F14.7)') XIN,YIN,RA,DEC
            CALL UMSPUT(CHARS,1,0,ISTAT)
         ELSE
          IF(.NOT.SILENT) THEN
            WRITE(CHARS,
     :  '('' X,Y: '',2F10.3'' (pix), RA: '',F12.7,'' Dec: '',F12.7,
     :    '' (deg).'')')
     : XIN,YIN,RA,DEC
            CALL UMSPUT(CHARS,1,0,ISTAT)
          ENDIF
         ENDIF

         IF(XYLIST.EQ.' ') GO TO 88
      ENDDO

 88   CONTINUE

C Also write the result to the CL
      IF(XYLIST.EQ.' ') THEN
         CALL UCLPSD('ra',RA,ISTAT)
         CALL UCLPSD('dec',DEC,ISTAT)
      ELSE
         CALL UFCLOS(LUN,ISTAT)
      ENDIF

C End of main WMETRIC module
 99   CONTINUE

      RETURN
      END
