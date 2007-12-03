      SUBROUTINE WTRANK 
C++
C
C WTRANBACK.F V2.22 Convert X,Y positions in drizzle output frames
C                 back to input positions in the initial frames
C
C This is a version of tranback which also supports WCS control.
C
C This task uses coefficients held in a text file to define
C geometrical distortions.
C
C This code uses the F77/VOS for IRAF i/o.
C It may be compiled and linked using a command like:
C
C xc -p stsdas wtranback.f drutil.f x_wtranback.x <dir>libiraf77.a
C
C Where <dir> is the location of the STSDAS f77/vos library.
C
C History:
C
C First quick version (V0.1), 7th May 1999 (Richard Hook, STScI)
C Bug fixes and tests for release of V0.2 in the IRAF stecf package
C 17th December 1999, Richard Hook
C
C Bugfix for different calling sequences, Richard Hook, November 2001
C
C V1.0, December 2001, using the full geometrical options and a general
C       simple inversion scheme based on successively improving guesses.
C
C V1.1, January 2002, bug fix for shft_un=output case and change
C       accuracy back to 3 decimal places.
C
C V1.3, March 2002, use packaged coefficient file access
C
C V1.4, August 2003, added distortion image support
C
C V1.4 of WTRANBACK created from TRANBACK. October 2003.
C
C V1.8, double precision version, brought number into line with WTRAXY.
C         Richard Hook, ST-ECF @ STScI, October 2003
C
C V1.9, included refpix options in coefficients files.
C         Richard Hook, ST-ECF @ STScI, 5th December 2003
C
C V2.2, added xylist support.
C         Richard Hook, ST-ECF @ STScI, 29th March 2004.
C
C V2.3, added ALPHA, BETA to DRIVAL call
C         Chris Hanley, STScI, 9 Sept. 2007
C--
      IMPLICIT NONE

C Local variables
      INTEGER ISTAT,NXIN,NYIN,NXOUT,NYOUT
      INTEGER COTY,CONUM,COMAX,I
      PARAMETER (COMAX=100)

      INTEGER IID,IRD,DATTYP,NDIMS,DDIMS(7)
      CHARACTER*80 INIM,OUTIM
      CHARACTER*8 CTYPE1,CTYPE2
      DOUBLE PRECISION XCO(COMAX),YCO(COMAX)
      DOUBLE PRECISION XIN,YIN,XOUT,YOUT,X(3),Y(3),XO(3),YO(3)
      DOUBLE PRECISION ERR
      DOUBLE PRECISION DX1,DY1,DX2,DY2
      DOUBLE PRECISION DELX,DELY,XOLD,YOLD,EV
      CHARACTER*132 CHARS
      CHARACTER*45 VERS

C Geometrical parameters, the standard set
      DOUBLE PRECISION SCALE,ROT,XSH,YSH,LAM
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE,ALPHA,BETA
      DOUBLE PRECISION RACEN,DECCEN,XREFP,YREFP,WCSIN(8),WCSOUT(8)
      DOUBLE PRECISION ORIENT,OUTSCL
      CHARACTER*80 COEFFS,SHFTUN,XGEOIM,YGEOIM,XYLIST,LINE
      CHARACTER*8 SHFR2,SHFTFR,ALIGN,GEOMOD
      INTEGER PXG,PYG,LUN,XGDIM,YGDIM
      LOGICAL DISIM

      REAL MEMR(1)
      COMMON /MEM/MEMR

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793D0)

      LOGICAL VERBOSE
      LOGICAL ROTFIR
      LOGICAL USEWCS
      LOGICAL ROTF2
      LOGICAL SECPAR

C-- Start of executable code
      VERS='WTRANBACK Version 2.22 (25th February 2005)'

C First announce the version
      CALL UMSPUT('+ '//VERS,1,0,ISTAT)

C Acceptable position error for iterative search
      ERR=0.000001
      VERBOSE=.TRUE.

C This version will accept WCS, so
      CALL UCLGST('geomode',GEOMOD,ISTAT)

C Get all the geometrical parameters
      CALL GTGEPA(GEOMOD,COEFFS,LAM,XGEOIM,YGEOIM,
     :             SCALE,ROT,XSH,YSH,SHFTFR,SHFTUN,ALIGN,
     :             XSCALE,YSCALE,XSH2,YSH2,ROT2,SHFR2,
     :             OUTSCL,RACEN,DECCEN,XREFP,YREFP,ORIENT)

C If so, check for image names (output is optional)
      CALL UCLGST('inimage',INIM,ISTAT)
      CALL UCLGST('refim',OUTIM,ISTAT)

      IF(GEOMOD.EQ.'wcs') THEN
         USEWCS=.TRUE.
      ELSE
         USEWCS=.FALSE.
      ENDIF

C Open the files and read the WCS from the headers
      IF(USEWCS .OR. COEFFS.EQ.'wcs') THEN
         IF(INIM.EQ.' ') THEN
            CALL UMSPUT('! No input image specified',
     :                  1,0,ISTAT)
            GO TO 99
         ELSE
            CALL UIMOPN(INIM,1,IID,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT('! Unable to open input image',
     :                     1,0,ISTAT)
               GO TO 99
            ELSE
               CALL UMSPUT('-Reading WCS from '//INIM,1,0,ISTAT)
            ENDIF

            CALL GETWCS(IID,WCSIN,CTYPE1,CTYPE2,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT(
     : '! Unable to get WCS from reference image',
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

      IF(USEWCS) THEN

C If we have a reference image we need to try to get its WCS for
C use as the output
         IF(OUTIM.NE.' ') THEN
            CALL UIMOPN(OUTIM,1,IRD,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT('! Unable to open output image',
     :                     1,0,ISTAT)
               GO TO 99
            ELSE
               CALL UMSPUT('-Reading WCS from '//OUTIM,1,0,ISTAT)
            ENDIF

            CALL GETWCS(IRD,WCSOUT,CTYPE1,CTYPE2,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT(
     : '! Unable to get WCS from reference image',
     :                     1,0,ISTAT)
               GO TO 99
            ENDIF

C Get the dimensions and use them
            CALL UIMGID(IRD,DATTYP,NDIMS,DDIMS,ISTAT)
            IF(NDIMS.NE.2) THEN
               CALL UMSPUT(
     :   '! Reference image is not two dimensional',
     :                          1,0,ISTAT)
               GO TO 99
            ELSE
               NXOUT=DDIMS(1)
               NYOUT=DDIMS(2)
            ENDIF
         ELSE

C If there is no output image then we read the output WCS from the
C user supplied values and convert to a WCS representation
            CALL SETWCS(OUTSCL,ORIENT,XREFP,RACEN,
     :                  YREFP,DECCEN,WCSOUT)
            CALL UMSPUT(
     : '-Defining output WCS from user-specified parameters',
     :      1,0,ISTAT)

         ENDIF
      ENDIF

C Convert the secondary parameters into suitable units
      ROT2=ROT2*PI/180.0D0

      IF(SHFR2.eq.'input') THEN
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

      IF(SHFTFR.EQ.'input') THEN
         ROTFIR=.FALSE.
      ELSE
         ROTFIR=.TRUE.
      ENDIF

      IF(SHFTUN.EQ.'output') THEN
         XSH=XSH*SCALE
         YSH=YSH*SCALE
      ENDIF

C Get the geometric distortion coefficient information
C Note that if no input image is available coeff="header" 
C or coeffs='wcs' will fail
      CALL GETGEO(COEFFS,IRD,LAM,
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

C Get the X,Y pixel position to be transformed
      CALL UCLGSD('xout',XOUT,ISTAT)
      CALL UCLGSD('yout',YOUT,ISTAT)

C Also get the chip size - in and out
      IF(.NOT.USEWCS .AND. OUTIM.EQ.' ') THEN
         CALL UCLGSI('nxout',NXOUT,ISTAT)
         CALL UCLGSI('nyout',NYOUT,ISTAT)
      ENDIF

      IF(.NOT.USEWCS) THEN
         CALL UCLGSI('nxin',NXIN,ISTAT)
         CALL UCLGSI('nyin',NYIN,ISTAT)
      ENDIF

C Convert to radians
      ROT=ROT*PI/180.0D0

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
         CALL UCLGSD('xout',XIN,ISTAT)
         CALL UCLGSD('yout',YIN,ISTAT)
      ENDIF

C Loop around (only once if there is just a position, not a list)
      DO WHILE(.TRUE.)

         IF(XYLIST.NE.' ') THEN
            CALL UFGLIN(LUN,LINE,ISTAT)
            IF(ISTAT.NE.0) GO TO 89
            READ(LINE,*) XOUT,YOUT
         ENDIF

C We now need to find the position we want by iterative
C improvement of an initial guess - the centre of the chip
C
C The method is to derive an "effective CD matrix" and use that
C to apply a correction until we are close enough (as defined by
C the ERR variable)

C First guess - centre
         X(1)=DBLE(NXIN/2)
         Y(1)=DBLE(NYIN/2)

C Loop around until we are close enough
         DO I=1,20       

            X(2)=X(1)+1.0D0
            Y(2)=Y(1)
            X(3)=X(1)
            Y(3)=Y(1)+1.0D0

C Transform the position
C Note that this is exactly the same as that in Drizzle
C First check that the point is within the bounds
            IF(NINT(X(1)).GT.1 .AND. NINT(X(1)).LT.NXIN-1 .AND.
     :         NINT(Y(1)).GT.1 .AND. NINT(Y(1)).LT.NYIN-1) THEN

              CALL DRIVAL(X,Y,3,NXIN,NYIN,NXOUT,NYOUT,.FALSE.,
     :         XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :         SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :         USEWCS,WCSIN,WCSOUT,
     :         COTY,CONUM,XCO,YCO,DISIM,MEMR(PXG),MEMR(PYG),
     :         XGDIM,YGDIM,XO,YO,ALPHA,BETA)

            ELSE
              CALL DRIVAL(X,Y,3,NXIN,NYIN,NXOUT,NYOUT,.FALSE.,
     :         XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :         SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :         USEWCS,WCSIN,WCSOUT,
     :         COTY,CONUM,XCO,YCO,.FALSE.,MEMR(PXG),MEMR(PYG),
     :         XGDIM,YGDIM,XO,YO,ALPHA,BETA)
            ENDIF

            DX1=XO(2)-XO(1)
            DY1=YO(2)-YO(1)
            DX2=XO(3)-XO(1)
            DY2=YO(3)-YO(1)

C Invert the matrix
            CALL INMAT(DX1,DX2,DY1,DY2)

C The error in out position
            DELX=XOUT-XO(1)
            DELY=YOUT-YO(1)

C Record the old position
            XOLD=X(1)
            YOLD=Y(1)

C Update the position
            X(1)=XOLD+DELX*DX1+DELY*DX2
            Y(1)=YOLD+DELX*DY1+DELY*DY2

C Work out the error vector length
            EV=SQRT((X(1)-XOLD)**2+(Y(1)-YOLD)**2)
            IF(EV.LT.ERR) GO TO 88
         ENDDO
      
 88      CONTINUE

         XIN=X(1)
         YIN=Y(1)

C Write out the result
         WRITE(CHARS,
     :  '('' Xin,Yin: '',2F12.5,'' Xout,Yout: '',2F12.5)')
     : XIN,YIN,XOUT,YOUT
         CALL UMSPUT(CHARS,1,0,ISTAT)

         IF(XYLIST.EQ.' ') GO TO 89
      ENDDO

 89   CONTINUE

C Also write the result to the CL
      IF(XYLIST.EQ.' ') THEN
         CALL UCLPSD('xin',XIN,ISTAT)
         CALL UCLPSD('yin',YIN,ISTAT)
      ELSE
         CALL UFCLOS(LUN,ISTAT)
      ENDIF

C End of main TRANBACK module
 99   CONTINUE

C If allocated, free the distortion image memory
      IF(PXG.NE.0) CALL UDMFRE(PXG,6,ISTAT)
      IF(PYG.NE.0) CALL UDMFRE(PYG,6,ISTAT)

      RETURN
      END
