      SUBROUTINE DR2GEO
C
C Convert a drizzle format distortion coefficients file
C to an image distortion file.
C
C This is a utility for tests.
C
C    Richard Hook, ST-ECF/STScI, September 2003
C
C Converted to double precision,
C    Richard Hook, ST-ECF/STScI, October 2003
C
      IMPLICIT NONE

      INTEGER IDD,ISTAT,NX,NY
      INTEGER COTY,COMAX,CONUM,PXG,PYG,DIMS(7)
      PARAMETER (COMAX=100)
      DOUBLE PRECISION LAM,XCO(COMAX),YCO(COMAX)
      CHARACTER*80 COEFFS
      DOUBLE PRECISION WCSIN(8),WCSOUT(8)

C Geometrical parameters, the standard set
      DOUBLE PRECISION SCALE,ROT,XSH,YSH,LAM
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*80 XGEOIM,YGEOIM
      INTEGER XGDIM,YGDIM
      CHARACTER*8 SHFR2,ALIGN

      INTEGER MAXLEN
      PARAMETER (MAXLEN=100000)

      INTEGER I,J,IXG,IYG
      DOUBLE PRECISION XIN(MAXLEN),YIN(MAXLEN),XOUT(MAXLEN),YOUT(MAXLEN)

      REAL MEMR(1)
      COMMON /MEM/MEMR

      LOGICAL DISIM,SECPAR,USEWCS,ROTFIR,ROTF2

      XGEOIM=' '
      YGEOIM=' '
      PXG=0
      PYG=0
      XSH=0.0
      YSH=0.0
      ROT=0.0
      SCALE=1.0
      DISIM=.FALSE.
      SECPAR=.FALSE.
      USEWCS=.FALSE.
      ROTFIR=.FALSE.
      IDD=0

C Get the geometric flags
      CALL UCLGST('align',ALIGN,ISTAT)

C Get the wavelength (Trauger only)
      CALL UCLGSD('lambda',LAM,ISTAT)

C Get the geometric distortion file parameters:
      CALL UCLGST('coeffs',COEFFS,ISTAT)

      CALL GETGEO(COEFFS,IDD,LAM,
     :            COTY,COMAX,CONUM,XCO,YCO,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :  '! Error, failed to get geometric distortion coefficients',
     :  1,0,ISTAT)
          GO TO 99
      ENDIF

C  Read in any distortion image corrections, if specified
C  PXG,PYG will default to 2x2 arrays when DISIM returns as FALSE.
C 
      CALL GGEOIM(XGEOIM,YGEOIM,PXG,PYG,XGDIM,YGDIM,DISIM)

C Get the name of the output images and create them
      CALL UCLGST('xgeoim',XGEOIM,ISTAT)
      CALL UCLGST('ygeoim',YGEOIM,ISTAT)
      CALL UCLGSI('nxgeo',NX,ISTAT)
      CALL UCLGSI('nygeo',NY,ISTAT)

      DIMS(1)=NX
      DIMS(2)=NY
      CALL UIMCRE(XGEOIM,6,2,DIMS,IXG,ISTAT)
      CALL UIMCRE(YGEOIM,6,2,DIMS,IYG,ISTAT)

      DO I=1,NX
         XIN(I)=DBLE(I)
      ENDDO 
     
C Loop through the output image
      DO J=1,NY

         DO I=1,NX
            YIN(I)=DBLE(J)
         ENDDO

C Convert the positions
         CALL DRIVAL(XIN,YIN,NX,NX,NY,NX,NY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,
     :            MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,XOUT,YOUT)

         DO I=1,NX
            XOUT(I)=XOUT(I)-XIN(I)
            YOUT(I)=YOUT(I)-YIN(I)
         ENDDO

C Write out the lines
         CALL UIPL2R(IXG,J,XOUT,ISTAT)
         CALL UIPL2R(IYG,J,YOUT,ISTAT)
      ENDDO

      CALL UIMCLO(IXG,ISTAT)
      CALL UIMCLO(IYG,ISTAT)

 99   CONTINUE

C If allocated, free the distortion image memory
      IF(PXG.NE.0) CALL UDMFRE(PXG,6,ISTAT)
      IF(PYG.NE.0) CALL UDMFRE(PYG,6,ISTAT)

      END
