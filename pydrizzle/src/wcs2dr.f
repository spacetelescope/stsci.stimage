      SUBROUTINE WCS2DR
C
C WCS2DR - use the WCS in the header of two images to deduce the
C          drizzle parameters (rot,scale,xsh,ysh) needed to map one
C          onto the other.
C
C Description:
C
C The 8 linear WCS values (CRVAL1/2, CRPIX1/2, CD1/2_1/2) are read from
C the headers of two images. One is regarded as the reference. The drizzle
C parameters which are needed to transform the image supplied so that it
C is registered with the reference image is written to the screen in a
C form suitable for feeding directly to the IRAF cl.
C
C Some checks that the scales are the same in X and Y and that there isn't
C excessive skew are made.
C
C It is assumed that the input images are both TAN projection (although not
C necessarily the same TAN projection).
C
C History:
C
C First try, 10th January 2000, Richard Hook
C Added support for multiple input images, Richard Hook, STScI, 13th January 2000
C Small changes for release with Drizzle 2.6, December 2001
C
C Fixed bug for larger angles and also added comment (#) character
C in version announcement string. V1.1, January 2002
C
C Modified to apply distortions to input and output WCS where needed.
C   Richard Hook & Andy Fruchter, STScI, May 2002
C
C Modifications for distortion images etc,
C   Richard Hook, ST-ECF/STScI, September 2003
C
C Double precision version,
C   Richard Hook, ST-ECF/STScI, October 2003
C
C Version 1.9, with refpix support,
C   Richard Hook, ST-ECF/STScI, 5th December 2003
C
      IMPLICIT NONE

      CHARACTER*80 COEFFS,IMAGES,IMAGE,REFIM
      CHARACTER*80 XGEOIM,YGEOIM
      CHARACTER*8 CTYPE1,CTYPE2
      CHARACTER*12 CHARS
      CHARACTER*132 LINE
      CHARACTER*4 STYLE
      INTEGER I,ID,IDR,ISTAT,NDIMS,DIMS(7),DATTYP,IMLD,I1,I2
      INTEGER PXG,PYG
      DOUBLE PRECISION XOUT,YOUT,XCEN,YCEN,XCENR,YCENR
      DOUBLE PRECISION XROT,YROT,ROT,XSCALE,YSCALE,SCALE,XSH,YSH
      DOUBLE PRECISION WCS(8),WCSR(8),X(4),Y(4),R,D
      CHARACTER*40 VERS

      INTEGER USEOF
      PARAMETER (USEOF=-2)

      DOUBLE PRECISION PIBY
      PARAMETER (PIBY=3.141592653/180.0D0)

      VERS='WCS2DR Version 1.9 (5th December 2003)'

C Announce the version
      CALL UMSPUT('#+ '//VERS,1,0,ISTAT)

C Get the names of the input and reference images
      CALL UCLGST('images',IMAGES,ISTAT)
      CALL UCLGST('refim',REFIM,ISTAT)

C Get the style of the output list
      CALL UCLGST('style',STYLE,ISTAT)

C Open image template processing
      CALL TIMOTP(IMAGES,IMLD,ISTAT)

C Before starting open the single reference image
      CALL UIMOPN(REFIM,1,IDR,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Error: unable to open reference image',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

C and get its size
      CALL UIMGID(IDR,DATTYP,NDIMS,DIMS,ISTAT)
      IF(NDIMS.NE.2) THEN
         CALL UMSPUT(
     : '! Error: reference image is not two dimensional',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

      XCENR=DBLE(DIMS(1)/2)+1.0D0
      YCENR=DBLE(DIMS(2)/2)+1.0D0

C Finally we can read the WCS from the reference as this only has to be 
C done once
      CALL GETWCS(IDR,WCSR,CTYPE1,CTYPE2,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     : '! Error: unable to read WCS from reference image header',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF
      IF(CTYPE1.NE.'RA---TAN' .OR. CTYPE2.NE.'DEC--TAN') THEN
         CALL UMSPUT('! Projection of reference image is not TAN',
     :               1,0,ISTAT)
         CALL UMSPUT('! (assuming it is anyway)',1,0,ISTAT)
      ENDIF

C Update the WCS to allow for the coefficients file
      CALL UCLGST('refcoefs',COEFFS,ISTAT)
      CALL UCLGST('refxgeo',XGEOIM,ISTAT)
      CALL UCLGST('refygeo',YGEOIM,ISTAT)

      CALL DUPWCS(WCSR,WCSR,DIMS(1),DIMS(2),COEFFS,XGEOIM,YGEOIM,
     :            PXG,PYG,IDR,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Failed to update WCS of reference image',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

C We can close the reference image at this point
      CALL UIMCLO(IDR,ISTAT)

C If style=line we can write the header
      IF(STYLE.EQ.'line') THEN
         CALL UMSPUT(
     : '# Image             xsh     ysh        scale        rot',
     :               1,0,ISTAT)
         CALL UMSPUT(
     : '# Note: shft_fr="output" shft_un="output" align="center"',
     :   1,0,ISTAT)
      ENDIF

C Loop around until the list of images is exhausted
      DO WHILE(.TRUE.)

C Get image name from list
         CALL TIMXTP(IMLD,IMAGE,ISTAT)
         IF(ISTAT.EQ.USEOF) GO TO 99

C Try to open it (readonly)
      CALL UIMOPN(IMAGE,1,ID,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Error: unable to open image: '//IMAGE(1:20),
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

C Get the dimensions of the input image and deduce its centre
C We assume align=center here
      CALL UIMGID(ID,DATTYP,NDIMS,DIMS,ISTAT)
      IF(NDIMS.NE.2) THEN
         CALL UMSPUT('! Error: image is not two dimensional',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

      XCEN=DBLE(DIMS(1)/2)+1.0D0
      YCEN=DBLE(DIMS(2)/2)+1.0D0

C Get the WCS from the headers. We just use the 8 standard values:
C CRVAL1/2, CRPIX1/2, CD_1/2_1/2
C
C If the projection isn't TAN we assume it is anyway, but tell the user
      CALL GETWCS(ID,WCS,CTYPE1,CTYPE2,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Error: unable to read WCS from image header',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF
      IF(CTYPE1.NE.'RA---TAN' .OR. CTYPE2.NE.'DEC--TAN') THEN
         CALL UMSPUT('! Projection of image is not TAN',1,0,ISTAT)
         CALL UMSPUT('! (assuming it is anyway)',1,0,ISTAT)
      ENDIF

C We have what we need from the image header so we can close it
      CALL UIMCLO(ID,ISTAT)

C Again update the WCS to allow for the coefficients
      CALL UCLGST('coeffs',COEFFS,ISTAT)
      CALL UCLGST('xgeoim',XGEOIM,ISTAT)
      CALL UCLGST('ygeoim',YGEOIM,ISTAT)

      CALL DUPWCS(WCS,WCS,DIMS(1),DIMS(2),COEFFS,XGEOIM,YGEOIM,
     :            PXG,PYG,ID,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Failed to update WCS of reference image',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

C Transform the corners of a unit square at the centre of the input
C image onto the reference
      X(1)=XCEN-0.5D0
      X(2)=XCEN+0.5D0
      X(3)=XCEN+0.5D0
      X(4)=XCEN-0.5D0
      Y(1)=YCEN-0.5D0
      Y(2)=YCEN-0.5D0
      Y(3)=YCEN+0.5D0
      Y(4)=YCEN+0.5D0

      DO I=1,4
         CALL XY2RD(X(I),Y(I),R,D,WCS)
         CALL RD2XY(R,D,X(I),Y(I),WCSR,ISTAT)
      ENDDO

C Calculate the shifts and X and Y rotations
      XOUT=0.0D0
      YOUT=0.0D0

      DO I=1,4
         XOUT=XOUT+X(I)
         YOUT=YOUT+Y(I)
      ENDDO

      XOUT=XOUT/4.0D0
      YOUT=YOUT/4.0D0

      XSH=XOUT-XCENR
      YSH=YOUT-YCENR

      XSCALE=SQRT((X(2)-X(1))**2 + (Y(2)-Y(1))**2)
      YSCALE=SQRT((X(3)-X(2))**2 + (Y(3)-Y(2))**2)

      IF(ABS(XSCALE/YSCALE-1.0) .GT. 0.001D0) THEN
         CALL UMSPUT(
     : '! Warning, X and Y scales differ by more than 0.1%',
     :   1,0,ISTAT)
      ENDIF

C Use the geometric mean scale
      SCALE=1.0D0/SQRT(XSCALE*YSCALE)

      XROT=ATAN2(Y(2)-Y(1),X(2)-X(1))/PIBY
      YROT=ATAN2(Y(3)-Y(2),X(3)-X(2))/PIBY-90.0D0
      
C Check the angle is in the right range
      IF(YROT.LT.-180.0D0) YROT=YROT+360.0D0

      IF(ABS(XROT-YROT) .GT. 0.1D0) THEN
         CALL UMSPUT(
     : '! Warning, X and Y rotation angles differ by more than 0.1deg',
     :   1,0,ISTAT)
      ENDIF

C Use the mean rotation angle      
      ROT=(XROT+YROT)/2.0D0

C Write out the results in a format suitable for drizzle
      IF(STYLE.EQ.'para') THEN
         CALL UMSPUT('#',1,0,ISTAT)
         CALL LENSTR(IMAGE,I1,I2)
         CALL UMSPUT('drizzle.data="'//IMAGE(I1:I2)//'"',1,0,ISTAT)
         CALL UMSPUT('drizzle.shft_fr="output"',1,0,ISTAT)
         CALL UMSPUT('drizzle.shft_un="output"',1,0,ISTAT)
         CALL UMSPUT('drizzle.align="center"',1,0,ISTAT)
         WRITE(CHARS,'(F12.5)') XSH
         CALL UMSPUT('drizzle.xsh='//CHARS,1,0,ISTAT)
         WRITE(CHARS,'(F12.5)') YSH
         CALL UMSPUT('drizzle.ysh='//CHARS,1,0,ISTAT)
         WRITE(CHARS,'(F12.5)') SCALE
         CALL UMSPUT('drizzle.scale='//CHARS,1,0,ISTAT)
         WRITE(CHARS,'(F12.5)') ROT
         CALL UMSPUT('drizzle.rot='//CHARS,1,0,ISTAT)
      ELSE IF(STYLE.EQ.'line') THEN
         WRITE(LINE,'(A15,4F12.5)') IMAGE(1:15),XSH,YSH,SCALE,ROT
         CALL UMSPUT(LINE,1,0,ISTAT)
      ELSE
         WRITE(LINE,'(A25,'' xsh='',F10.5,'' ysh='',F10.5,
     :    '' scale='',F10.5,'' rot='',F10.5,
     :    '' shft_fr="output" shft_un="output" align="center"'')')
     :    IMAGE(1:25),XSH,YSH,SCALE,ROT
         CALL UMSPUT(LINE,1,0,ISTAT)
        ENDIF
      ENDDO

 99   CONTINUE
      CALL TIMCTP(IMLD,ISTAT)

C If allocated, free the distortion image memory
      IF(PXG.NE.0) CALL UDMFRE(PXG,6,ISTAT)
      IF(PYG.NE.0) CALL UDMFRE(PYG,6,ISTAT)

      RETURN
      END
