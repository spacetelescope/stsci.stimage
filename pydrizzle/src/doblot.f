      SUBROUTINE DOBLOT(DATA,NDAT,INTYPE,SINSCL,KSCALE,MISVAL,
     :     XMIN,XMAX,YMIN,YMAX,DNX,DNY,
     :     ROTFIR,EF,ALIGN,ONX,ONY,COTY,CONUM,XCO,YCO,
     :     DISIM,PXG,PYG,PXDIM,PYDIM,
     :     XIN,YIN,XOUT,YOUT,
     :     SCALE,ROT,XSH,YSH,USEWCS,WCSIN,WCSOUT,
     :     GEOMOD,SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2, 
     :     ALPHA,BETA)
C
C This routine does the interpolation of the input array
C In version 1.0 and later this is done using the standard DRIVAL
C routine in drutil.f.
C
C Note - when running in WBLOT mode this routine uses the WCS
C        values but has to supply them the other way around to DRIVAL.
C
      IMPLICIT NONE

      INTEGER DNX,DNY,ONX,ONY
      INTEGER I,J,NMISS,ISTAT
      INTEGER LX,LY,INTYPE
      INTEGER XMAX,XMIN,YMAX,YMIN
      REAL DATA(XMAX-XMIN+1,YMAX-YMIN+1),V
      REAL NDAT(ONX,ONY)
      DOUBLE PRECISION XCEN,YCEN,DX,DY
      INTEGER COTY,CONUM
      DOUBLE PRECISION XCO(CONUM),YCO(CONUM)
      DOUBLE PRECISION WCSIN(8),WCSOUT(8)
      DOUBLE PRECISION XOUT(ONX),YOUT(ONX),XIN(ONX),YIN(ONX),YV
      DOUBLE PRECISION ALPHA, BETA
      REAL XO,YO
      REAL MISVAL,KSCALE,SINSCL
      REAL NRIEVL,EF,S2
      CHARACTER*80 CHARS
      LOGICAL ROTFIR
      LOGICAL SECPAR
      LOGICAL ROTF2 
      LOGICAL USEWCS

C Standard geometrical parameters set
      DOUBLE PRECISION SCALE,ROT,XSH,YSH,SCALL,SCDIS
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*8 SHFR2,ALIGN,GEOMOD
      INTEGER PXDIM,PYDIM
      REAL PXG(PXDIM,PYDIM),PYG(PXDIM,PYDIM)
      LOGICAL DISIM

C Space for Lanczos-style look-up-tables
      INTEGER NLUT,NBOX
      PARAMETER (NLUT=2048)
      REAL LANLUT(NLUT),GINTER
      REAL SPACE,SPK,KS2
      PARAMETER (SPACE=0.01)
C--

C Some initial settings - note that the reference pixel
C is offset from the "centre":
C
      NMISS=0

C If we need a LUT calculate one here first
      IF(INTYPE.EQ.100) THEN
         CALL FILALU(3,NLUT,SPACE,LANLUT)
      ELSE IF(INTYPE.EQ.105) THEN
         CALL FILALU(5,NLUT,SPACE,LANLUT)
      ENDIF

C In the WCS case we can't use the scale to calculate the
C Jacobian so we need to do it
C
C Note that we use the centre of the image rather than the
C reference pixel as the reference here
C
C This is taken from DOBOX except for the inversion of the image
C order

C This section applies in WBLOT mode and now contains the 
C addition correction to separate the distortion induced scale change
      IF(USEWCS) THEN
         CALL UMSPUT("USEWCS true...\n",1,0,ISTAT)

         IF(ALIGN.EQ.'corner') THEN
            XCEN=DBLE(ONX/2.0)+0.5
            YCEN=DBLE(ONY/2.0)+0.5
         ELSE
            XCEN=DBLE(ONX/2.0)+1.0
            YCEN=DBLE(ONY/2.0)+1.0
         ENDIF

         XIN(1)=XCEN
         XIN(2)=XCEN
         XIN(3)=XCEN+1.0D0
         XIN(4)=XCEN+1.0D0
         YIN(1)=YCEN
         YIN(2)=YCEN+1.0D0
         YIN(3)=YCEN+1.0D0
         YIN(4)=YCEN

         CALL DRIVAL(XIN,YIN,4,ONX,ONY,DNX,DNY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSOUT,WCSIN,
     :            COTY,CONUM,XCO,YCO,DISIM,
     :            PXG,PYG,PXDIM,PYDIM,XOUT,YOUT,
     :            ALPHA, BETA)

         SCALL=SQRT(1.0D0/ABS(0.5D0*
     :              ((XOUT(2)-XOUT(4))*(YOUT(1)-YOUT(3)) -
     :              (XOUT(1)-XOUT(3))*(YOUT(2)-YOUT(4)))))

C Now calculate how much of this is from the geometric distortion
         XSH=0.D00
         YSH=0.D00
         ROT=0.D00
         SCALE=1.D00
         SECPAR=.FALSE.
         USEWCS=.FALSE.

         CALL DRIVAL(XIN,YIN,4,ONX,ONY,DNX,DNY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSOUT,WCSIN,
     :            COTY,CONUM,XCO,YCO,DISIM,
     :            PXG,PYG,PXDIM,PYDIM,
     :            XOUT,YOUT, ALPHA, BETA)

         SCDIS=SQRT(1.0D0/
     :            ABS(0.5D0*((XOUT(2)-XOUT(4))*(YOUT(1)-YOUT(3)) -
     :              (XOUT(1)-XOUT(3))*(YOUT(2)-YOUT(4)))))

         USEWCS=.TRUE.
         SCALE=SCALL/SCDIS
      ENDIF

C Image subset size
      LX=XMAX-XMIN+1
      LY=YMAX-YMIN+1

C Offsets
      DX=DBLE(XMIN-1)
      DY=DBLE(YMIN-1)

C Recalculate the area scaling factor
      S2=FLOAT(SCALE*SCALE)

C Some useful numbers
      SPK=SPACE*KSCALE
      IF(INTYPE.EQ.100) THEN
         NBOX=NINT(3.0/KSCALE)
      ELSE IF(INTYPE.EQ.105) THEN
         NBOX=NINT(5.0/KSCALE)
      ENDIF

      IF(INTYPE.GE.100) KS2=1.0/(KSCALE*KSCALE)

C Outer loop over output image pixels (X,Y)
      DO J=1,ONY
         YV=DBLE(J)
       
C Set the X and Y start positions
         XIN(1)=1.0D0
         YIN(1)=YV
         XIN(2)=0.0D0
         YIN(2)=0.0D0

C Transform this vector
         CALL DRIVAL(XIN,YIN,ONX,ONX,ONY,DNX,DNY,.TRUE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSOUT,WCSIN,
     :            COTY,CONUM,XCO,YCO,DISIM,
     :            PXG,PYG,PXDIM,PYDIM,XOUT,YOUT, ALPHA, BETA)

C Loop through the output positions and do the interpolation
         DO I=1,ONX

            XO=SNGL(XOUT(I)-DX)
            YO=SNGL(YOUT(I)-DY)

C Check it is on the input image
            IF(XO.GE.1.0 .AND. XO.LE.REAL(LX) .AND.
     :         YO.GE.1.0 .AND. YO.LE.REAL(LY)) THEN

C Check for look-up-table interpolation
               IF(INTYPE.GE.100) THEN
                  V=GINTER(XO,YO,DATA,LX,LY,LANLUT,NLUT,
     :                     SPK,NBOX,MISVAL)*KS2
               ELSE

C Do the interpolation
C and allow for stretching because of scale change
C Also, use callable version of from INTER2D.F.
C
                  V=NRIEVL(XO,YO,DATA,LX,LY,LX,INTYPE,SINSCL)
               ENDIF

               NDAT(I,J)=V*EF/S2
            ELSE

C If there is nothing for us then set the output to missing
C value flag
               NDAT(I,J)=MISVAL

C Count cases where the pixel is off the output image
               NMISS=NMISS+1
            ENDIF
         ENDDO
      ENDDO

C Report number of points which were outside the input frame
      IF(NMISS.GT.0) THEN
       WRITE(CHARS,'(''! Warning, '',I7,'' points were outside '','//
     :                '''the output image.'')') NMISS
         CALL UMSPUT(CHARS,1,0,ISTAT)
      ENDIF

C Finally we need to update the WCS using the standard routine
C modified to work in the "blot direction"
C
C Not needed for WBLOT
      IF(.NOT.USEWCS) THEN
         CALL BUPWCS(WCSIN,WCSOUT,DNX,DNY,ONX,ONY,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,PXDIM,PYDIM,
     :            ALPHA,BETA)
      ENDIF

      RETURN
      END

