C DRUTIL.F
C
C Utility routines for Drizzle and Blot
C
C These routines should call only themselves, without reliance on
C IRAF/STSDAS f77/vos.  Those routines will be found in 'ioutil.f'.
C   Exceptions: UIGL2R (in DOBOX)
C
C We need to revise GETGEO to read xgeoim/ygeoim images using 
C   non-IRAF f77/vos routines as needed for callable version.
C   The 'wcs' option for coeffs (reading Spitzer-style coeffs from
C   header) also needs to be implemented without using f77/vos 
C   routines.  This will probably require writing replacements for
C   the f77/vos routines to be put in 'drcall.f' that do not use f77/vos.
C
C Original version for DRIZZLE 3.3 and BLOT 3.3, December 2003
C Richard Hook, ST-ECF/ESO
C
C This version modified for callable and STSDAS DRIZZLE simultaneously 
C Warren Hack, STScI, 17-June-2004 
C
C History:
C
C The module GOLDH was transferred from drizzle as it is also
C now needed by blot.
C
C WCS routines added (XY2RD & RD2XY), September 1998
C
C Modified GTCOEF routine to also handle STIS and NICMOS,
C    January 1999
C
C Added additional code for reversing the cubic transformation added
C  June 1999
C
C Major extensions and some routines transferred from the EIS utilities
C collection (for context handling), October 2000
C
C Extensive modifications for a more flexible geometric coefficients
C scheme, changes in calling sequences, December 2000
C
C April 2003 - added code for checking overlaps with output images
C and minimising unnecessary drizzling (CHOVER).
C
C Added double precision version of several routines for more precision.
C  Richard Hook, ST-ECF/ESO/STScI, June 2003
C
C Full conversion to double precision (for geometry, not data values)
C   Richard Hook, ST-ECF/ESO/STScI, October 2003
C
C Modifications to support "refpix" contruct in distortion files
C   Richard Hook, ST-ECF/ESO/STScI, December 2003
C WCSLIN was modified to use all coefficients in the evaluation.
C Image center is calculated with floating point precision.
C   Nadia Dencheva, July 2006
C
      SUBROUTINE SETIM(A,NX,NY,V)
C
C Set a 2D array to a specified value 
C
C SINGLE PRECISION routine.
C
      IMPLICIT NONE

      INTEGER NX,NY
      REAL A(NX,NY),V

      INTEGER I,J

      DO J=1,NY
         DO I=1,NX
            A(I,J)=V
         ENDDO
      ENDDO

      RETURN
      END

      SUBROUTINE MULC(A,NX,NY,V)
C
C Multiply a 2D REAL array by a specified value
C
C SINGLE PRECISION routine.
C
      IMPLICIT NONE

      INTEGER NX,NY
      REAL A(NX,NY),V

      INTEGER I,J

      DO J=1,NY
         DO I=1,NX
            A(I,J)=A(I,J)*V
         ENDDO
      ENDDO

      RETURN
      END

      SUBROUTINE COPYIM(IN,OUT,NX,NY)
C
C Simply copy a REAL array into another
C
C SINGLE PRECISION routine.
C
      IMPLICIT NONE

      INTEGER NX,NY
      REAL IN(NX,NY),OUT(NX,NY)

      INTEGER I,J

      DO J=1,NY
         DO I=1,NX
            OUT(I,J)=IN(I,J)
         ENDDO
      ENDDO

      RETURN
      END

      DOUBLE PRECISION FUNCTION EVAL3(X,Y,CO)
C
C Evaluate a cubic geometric distortion in 2d
C
      IMPLICIT NONE

      DOUBLE PRECISION X,Y,CO(*)

      EVAL3=CO(1)+
     :      CO(2)*X+
     :      CO(3)*Y+
     :      CO(4)*X*X+
     :      CO(5)*X*Y+
     :      CO(6)*Y*Y+
     :      CO(7)*X*X*X+
     :      CO(8)*X*X*Y+
     :      CO(9)*X*Y*Y+
     :      CO(10)*Y*Y*Y 

      RETURN
      END

      DOUBLE PRECISION FUNCTION EVAL4(X,Y,CO)
C
C Evaluate a 4th order (quartic) geometric distortion in 2d
C
      IMPLICIT NONE

      DOUBLE PRECISION X,Y,CO(*)

      EVAL4=CO(1)+
     :      CO(2)*X+
     :      CO(3)*Y+
     :      CO(4)*X*X+
     :      CO(5)*X*Y+
     :      CO(6)*Y*Y+
     :      CO(7)*X*X*X+
     :      CO(8)*X*X*Y+
     :      CO(9)*X*Y*Y+
     :      CO(10)*Y*Y*Y+
     :      CO(11)*X*X*X*X+
     :      CO(12)*X*X*X*Y+
     :      CO(13)*X*X*Y*Y+
     :      CO(14)*X*Y*Y*Y+
     :      CO(15)*Y*Y*Y*Y

      RETURN
      END

      DOUBLE PRECISION FUNCTION EVAL5(X,Y,CO)
C
C Evaluate a 5th order geometric distortion in 2d
C
      IMPLICIT NONE

      DOUBLE PRECISION X,Y,CO(*)

      EVAL5=CO(1)+
     :      CO(2)*X+
     :      CO(3)*Y+
     :      CO(4)*X*X+
     :      CO(5)*X*Y+
     :      CO(6)*Y*Y+
     :      CO(7)*X*X*X+
     :      CO(8)*X*X*Y+
     :      CO(9)*X*Y*Y+
     :      CO(10)*Y*Y*Y+
     :      CO(11)*X*X*X*X+
     :      CO(12)*X*X*X*Y+
     :      CO(13)*X*X*Y*Y+
     :      CO(14)*X*Y*Y*Y+
     :      CO(15)*Y*Y*Y*Y+
     :      CO(16)*X*X*X*X*X+
     :      CO(17)*X*X*X*X*Y+
     :      CO(18)*X*X*X*Y*Y+
     :      CO(19)*X*X*Y*Y*Y+
     :      CO(20)*X*Y*Y*Y*Y+
     :      CO(21)*Y*Y*Y*Y*Y

      RETURN
      END

      DOUBLE PRECISION FUNCTION EVALN(X,Y,CO,ORDER)
C
C Evaluate the value of a general 2d polynomial in X and Y
C complete with "half" cross-terms.
C
C For orders lower than 7 it is slightly more efficient to
C use the specific (poly4 etc) routines instead.
C
C Richard Hook, ST-ECF, February 2002
C
      IMPLICIT NONE

      INTEGER ORDER
      DOUBLE PRECISION X,Y,CO(*)

      INTEGER N,M,NC
      DOUBLE PRECISION T

      T=0.0
      NC=1

      DO N=1,ORDER+1
         DO M=1,N
            T=T+CO(NC)*X**(N-M)*Y**(M-1)
            NC=NC+1
         ENDDO
      ENDDO

      EVALN=T
      RETURN
      END

      SUBROUTINE RAD3(X,Y,CO,XO,YO)
C
C Evaluate a 3rd order radial geometric distortion in 2d
C X version. Note that there is no zero order coefficient
C as this is physically meaningless.
C
      IMPLICIT NONE

      DOUBLE PRECISION X,Y,XO,YO,R,F,CO(*)

      R=DSQRT(X*X+Y*Y)

      F=1.0D0+CO(1)+CO(2)*R+CO(3)*R*R
      XO=F*X
      YO=F*Y

      RETURN
      END
      

      SUBROUTINE GETCO(LUN,LAM,COTY,COMAX,CONUM,XCO,YCO,ISTAT)
C
C Read geometric distortion coefficients in free format
C from a text file which has already been opened and return them.
C
C If there is a problem reading the numbers then the status flag will be
C returned as 1. If all is well it will be returned
C as 0.
C
C This routine caused problems with SUN SPARC SC4 compilers and was
C modified to avoid them. March 1997
C
C Modified to handle higher-order terms in addition.
C   December 2000
C
C Modified to also support the "refpix" specification at the start of the
C coefficients files to allow the reference point for the distortion to
C be decoupled from the center of the image.
C   December 2003
C
      IMPLICIT NONE

      INTEGER LUN,ISTAT,I,J
      CHARACTER*80 LINE,CHARS
      CHARACTER*1024 BUFFER
      DOUBLE PRECISION LAM,MGF2,N
      DOUBLE PRECISION A,B,C
      INTEGER COTY,COMAX,CONUM
      DOUBLE PRECISION XCO(COMAX),YCO(COMAX)

C Different reference pixel support
      LOGICAL NEWREF
      INTEGER COOFF
      PARAMETER (COOFF=100)
      DOUBLE PRECISION XDREF,YDREF

C By default no different reference point
      NEWREF=.FALSE.

C Skip blank lines and those beginning with # until we find a label
      DO WHILE(.TRUE.)
         CALL UFGLIN(LUN,LINE,ISTAT)
         IF(ISTAT.NE.0) GO TO 99

         IF(LINE.NE.' ' .AND. LINE(1:1).NE.'#') THEN

C First "Trauger" style coefficients - 60 of them
            IF(LINE(1:7).EQ.'trauger') THEN

C Now we need to calculate the coefficients which are a
C function of wavelength
               N=MGF2(LAM)

C Now we loop through extracting the coefficients, 3 per line,
C and calculating the wavelength dependence
               J=1
               DO WHILE(J.LE.20)
                  CALL UFGLIN(LUN,LINE,ISTAT)
                  IF(ISTAT.NE.0) GO TO 99

                  IF(LINE.NE.' ' .AND. LINE(1:1).NE.'#') THEN
                     READ(LINE,*,IOSTAT=ISTAT) A,B,C          
                     IF(ISTAT.NE.0) GO TO 99

                     IF(J.LE.10) THEN
                        XCO(J)=A+B*(N-1.5)+C*(N-1.5)**2
                     ELSE
                        YCO(J-10)=A+B*(N-1.5)+C*(N-1.5)**2
                     ENDIF

                     J=J+1
                  ENDIF
               ENDDO

               COTY=3
               CONUM=10
               ISTAT=0
               GO TO 99

C Next "cubic" style - 20 coefficients
            ELSE IF(LINE(1:5).EQ.'cubic' .OR. 
     :              LINE(1:5).EQ.'poly3') THEN

C Copy the rest of the file into a buffer ready for a free-format
C read operation
               CALL BFILL(LUN,BUFFER,ISTAT)

               CONUM=10
               IF(ISTAT.EQ.0) THEN
                 READ(BUFFER,*,IOSTAT=ISTAT) 
     :           (XCO(I),I=1,CONUM),(YCO(I),I=1,CONUM)
               ENDIF
               IF(ISTAT.NE.0) GO TO 99

               COTY=3
               ISTAT=0
               GO TO 99

C Now 4th order - 30 coefficients
            ELSE IF(LINE(1:5).EQ.'poly4' .OR. 
     :              LINE(1:7).EQ.'quartic') THEN

               CALL BFILL(LUN,BUFFER,ISTAT)

               CONUM=15
               IF(ISTAT.EQ.0) THEN
                 READ(BUFFER,*,IOSTAT=ISTAT) 
     :           (XCO(I),I=1,CONUM),(YCO(I),I=1,CONUM)
               ENDIF
               IF(ISTAT.NE.0) GO TO 99

               COTY=4
               ISTAT=0
               GO TO 99

C Now 5th order - 42 coefficients
            ELSE IF(LINE(1:5).EQ.'poly5' .OR. 
     :              LINE(1:7).EQ.'quintic') THEN

               CALL BFILL(LUN,BUFFER,ISTAT)

               CONUM=21
               IF(ISTAT.EQ.0) THEN
                 READ(BUFFER,*,IOSTAT=ISTAT) 
     :           (XCO(I),I=1,CONUM),(YCO(I),I=1,CONUM)
               ENDIF
               IF(ISTAT.NE.0) GO TO 99

               COTY=5
               ISTAT=0
               GO TO 99

C Higher orders (>5)
            ELSE IF(LINE(1:4).EQ.'poly') THEN

C First find the order by reading the rest of the line
               READ(LINE(5:80),*) COTY
               CONUM=(COTY+1)*(COTY+2)/2

               CALL BFILL(LUN,BUFFER,ISTAT)

               IF(ISTAT.EQ.0) THEN
                 READ(BUFFER,*,IOSTAT=ISTAT)
     :           (XCO(I),I=1,CONUM),(YCO(I),I=1,CONUM)
               ENDIF
               IF(ISTAT.NE.0) GO TO 99

               ISTAT=0
               GO TO 99

C Next third-order radial polynomial   
            ELSE IF(LINE(1:6).EQ.'radial' .OR.
     :              LINE(1:4).EQ.'rad3') THEN

C Copy the rest of the file into a buffer ready for a free-format
C read operation
               CALL BFILL(LUN,BUFFER,ISTAT)

               CONUM=3 
               IF(ISTAT.EQ.0) THEN
                 READ(BUFFER,*,IOSTAT=ISTAT)
     :           (XCO(I),I=1,CONUM)
               ENDIF
               IF(ISTAT.NE.0) GO TO 99

               COTY=-3
               ISTAT=0
               GO TO 99

C Optionally there may be a line with a different centre position
            ELSE IF(LINE(1:6).EQ.'refpix') THEN

               READ(LINE(7:80),*,IOSTAT=ISTAT) XDREF,YDREF
               IF(ISTAT.NE.0) THEN
                  CALL UMSPUT(
     : '! Invalid reference pixel specification in coefficients file',
     :                     1,0,ISTAT)
                  ISTAT=1
                  GO TO 99
               ELSE
                  NEWREF=.TRUE.
          WRITE(CHARS,
     : '(''-Using distortion reference point: ['',
     : F10.4,'','',F10.4,'']'')')
     :    XDREF,YDREF
                  CALL UMSPUT(CHARS,1,0,ISTAT)
               ENDIF

            ELSE
               CALL UMSPUT('! Unknown coefficient set type',
     :                     1,0,ISTAT)
               ISTAT=1
               GO TO 99
            ENDIF
         ENDIF
      ENDDO

99    CONTINUE

C If we have a modified reference pixel we encode this in the standard
C coefficients by adding an offset and incrementing the number of coefficients
C by one
      IF(NEWREF) THEN
         COTY=COTY+COOFF
         CONUM=CONUM+1
         XCO(CONUM)=XDREF
         YCO(CONUM)=YDREF
      ENDIF

      RETURN
      END

      DOUBLE PRECISION FUNCTION MGF2(LAM)
C
C Calculate the refractive index of MgF2 for a given
C wavelength (in nm) using the formula given by Trauger (1995)
C
      IMPLICIT NONE

      DOUBLE PRECISION LAM,SIG

      SIG=1.0D7/LAM

      MGF2=SQRT(1.0 + 2.590355D10/(5.312993D10-SIG*SIG) 
     :              + 4.4543708D9/(11.17083D9-SIG*SIG) 
     :              + 4.0838897D5/(1.766361D5-SIG*SIG))

      RETURN
      END

      SUBROUTINE PUTFIL(DAT,COU,ONX,ONY,FILVAL)
C
C Put in the missing filling value where the weight is
C zero
C
C SINGLE PRECISION routine.
C
      IMPLICIT NONE

      INTEGER ONX,ONY
      REAL DAT(ONX,ONY),COU(ONX,ONY),FILVAL

      INTEGER I,J

      DO J=1,ONY
         DO I=1,ONX
            IF(COU(I,J).EQ.0.0) DAT(I,J)=FILVAL
         ENDDO
      ENDDO
   
      RETURN
      END

      SUBROUTINE INMAT(P,Q,R,S)
C
C Invert a 2 by 2 double precision matrix
C in place. It assumes that the matrix is not
C singular.
C
      IMPLICIT NONE

      DOUBLE PRECISION P,Q,R,S
      DOUBLE PRECISION A,B,C,D,DET

      DET=P*S-Q*R

      A=S/DET
      B=-Q/DET
      C=-R/DET
      D=P/DET

      P=A
      Q=B
      R=C
      S=D

      RETURN
      END

      SUBROUTINE XY2RD(X,Y,R,D,WCS)
C
C XY2RD - convert a pixel position to an equatorial (RA,DEC)
C         position assuming a TAN projection and WCS with the
C         standard 8 elements.
C
C It is based on the xy2rd task in STSDAS
C
C Output angles are in degrees
C
C Richard Hook, STScI, 11th August 1998
C
      IMPLICIT NONE

      DOUBLE PRECISION X,Y,R,D,WCS(8)
      DOUBLE PRECISION XI,ETA,RA0,DEC0

      DOUBLE PRECISION PIBY
      PARAMETER (PIBY=3.141592653589793/180.0)

C First convert pixel coordinates to tangent plane in radians
      XI=(WCS(5)*(X-WCS(1))+WCS(7)*(Y-WCS(3)))*PIBY
      ETA=(WCS(6)*(X-WCS(1))+WCS(8)*(Y-WCS(3)))*PIBY

C And convert the reference point on the sky to radians
      RA0=WCS(2)*PIBY
      DEC0=WCS(4)*PIBY

C Now go to equatorial from tangent plane
      R=ATAN2(XI, COS(DEC0)-ETA*SIN(DEC0)) + RA0
      D=ATAN2(ETA*COS(DEC0)+SIN(DEC0),
     :        SQRT((COS(DEC0)-ETA*SIN(DEC0))**2 + XI**2))

C Convert back to degrees and check the range
      R=R/PIBY
      D=D/PIBY
      IF(R.LT.0.0) R=R+360.0

      RETURN
      END

      SUBROUTINE RD2XY(R,D,X,Y,WCS,ISTAT)
C
C RD2XY - convert an equatorial (RA,DEC) position to a pixel position
C         assuming a TAN projection and WCS with the
C         standard 8 elements.
C
C It is based on the rd2xy task in STSDAS
C
C Input angles are in degrees
C
C Richard Hook, STScI, 13th August 1998
C
      IMPLICIT NONE

      INTEGER ISTAT
      DOUBLE PRECISION X,Y,R,D,WCS(8),RA,DEC
      DOUBLE PRECISION XI,ETA,RA0,DEC0
      DOUBLE PRECISION DET,CDINV(2,2),BOTTOM
      DOUBLE PRECISION PIBY
      PARAMETER (PIBY=3.141592653589793/180.0)

C First invert the CD matrix
      DET=WCS(5)*WCS(8)-WCS(7)*WCS(6)

      IF(DET.EQ.0.0) THEN
         ISTAT=1
         RETURN
      ENDIF

      CDINV(1,1)=WCS(8)/DET
      CDINV(1,2)=-WCS(7)/DET
      CDINV(2,1)=-WCS(6)/DET
      CDINV(2,2)=WCS(5)/DET

C Translate from RA,Dec to X,Y
      RA0=WCS(2)*PIBY
      DEC0=WCS(4)*PIBY

      RA=R*PIBY
      DEC=D*PIBY

      BOTTOM=SIN(DEC)*SIN(DEC0)+COS(DEC)*COS(DEC0)*COS(RA-RA0)
      IF(BOTTOM.EQ.0) THEN
         ISTAT=1
         RETURN
      ENDIF

C Calculate tangent plane position and convert to degrees
      XI=COS(DEC)*SIN(RA-RA0)/BOTTOM/PIBY
      ETA=(SIN(DEC)*COS(DEC0)-COS(DEC)*SIN(DEC0)*COS(RA-RA0))/
     :     BOTTOM/PIBY

C Convert back to pixels using the inverse of the CD matrix
      X=CDINV(1,1)*XI+CDINV(1,2)*ETA+WCS(1)
      Y=CDINV(2,1)*XI+CDINV(2,2)*ETA+WCS(3)

      ISTAT=0
      RETURN
      END

      SUBROUTINE WCSLIN(WCSIN,WCSOUT,
     :            XCEN,YCEN,COTY,CONUM,
     :        XCO,YCO,DISIM,XG,YG,NX,NY,XC,YC,XS,YS,XT,YT)
C
C WCSLIN - derive best linear transformation coefficients to map
C          pixel positions from one world coordinate system to another.
C
C Supplied:
C
C  WCSIN - double precision 8 element array. The WCS of the input image.
C          This includes CRPIX1/2, CRVAL1/2 and a CD matrix.
C
C  WCSOUT - double precision 8 element array. The WCS of the output image.
C          This includes CRPIX1/2, CRVAL1/2 and a CD matrix.
C
C  XCEN,YCEN - double precision, the centre of the chip, for reference
C
C  COTY,CONUM - coefficients type code and number of coefficients
C
C  XCO,YCO - double arrays, the non-linear distortion coefficients
C
C DISIM,XG,YG,NX,NY - distortion image support
C
C Returned:
C
C XC,YC,XS,YS,XT,YT - doubles, linear transformation coefficients
C
C   Richard Hook, 18th September 1998
C   Modified for more general coefficients and changed calling
C   sequence, December 2000
C
C  Further modification to try to handle ACS images with big distortions
C  and offsets. January 2001.
C
C Modified for better accuracy by greater use of double precision
C values. November 2002.
C
C Modified to ignore distortion images - ie, to assume that they do
C not have a global effect.
C    Richard Hook, STScI/ECF, September 2003
C
C Added support for "refpix" offsets,
C    Richard Hook, STScI/ECF, December 2003
C
C Used a smaller range of offset for double precision version
C to minimise effects of nonlinear terms.
C   Richard Hook, STScI/ECF, January 2004
C
      IMPLICIT NONE

C Input parameters
      DOUBLE PRECISION WCSIN(8),WCSOUT(8)
      INTEGER COTY,CONUM,COOFF
      PARAMETER (COOFF=100)
      DOUBLE PRECISION XCO(CONUM),YCO(CONUM),XCEN,YCEN
      DOUBLE PRECISION XDREF,YDREF,XDOFF,YDOFF
      DOUBLE PRECISION EVAL3,EVAL4,EVAL5,EVALN

      INTEGER NX,NY
      LOGICAL DISIM,NEWREF
      REAL XG(NX,NY),YG(NX,NY)

C Output parameters
      DOUBLE PRECISION XC,YC,XS,YS,XT,YT

C Local variables
      DOUBLE PRECISION RA,DEC
      DOUBLE PRECISION XIN(4),YIN(4),XOUT(4),YOUT(4)
      DOUBLE PRECISION X0,Y0,A,B,C,D
      DOUBLE PRECISION PIBY
      DOUBLE PRECISION X(4),Y(4)
      INTEGER I,ISTAT,SCOTY

      PARAMETER (PIBY=180.0/3.14159265359)

C First check for the presence of "refpix" additional information
C in the coefficients
C If it is set a flag and offset again
      IF(COTY.GT.COOFF/2) THEN
         NEWREF=.TRUE.
         COTY=COTY-COOFF
         XDREF=XCO(CONUM)
         YDREF=YCO(CONUM)
         CONUM=CONUM-1
      ELSE
         NEWREF=.FALSE.
      ENDIF

C Set up a square at the reference pixel of the input
C image (WCSIN(1)=CRPIX1 and WCSIN(3)=CRPIX2)
      XIN(1)=WCSIN(1)    
      XIN(2)=WCSIN(1)    
      XIN(3)=WCSIN(1)+1.0D0
      XIN(4)=WCSIN(1)+1.0D0 
      YIN(1)=WCSIN(3)     
      YIN(2)=WCSIN(3)+1.0D0
      YIN(3)=WCSIN(3)+1.0D0
      YIN(4)=WCSIN(3)  
         
C Transform these points onto the sky and then back out again
C using the target WCS - all double precision
      DO I=1,4
         CALL XY2RD(XIN(I),YIN(I),RA,DEC,WCSIN)
         CALL RD2XY(RA,DEC,XOUT(I),YOUT(I),WCSOUT,ISTAT)
      ENDDO

C Check for different reference pixel
      IF(NEWREF) THEN
         XDOFF=XCEN-XDREF
         YDOFF=YCEN-YDREF
      ELSE
         XDOFF=0.0D0
         YDOFF=0.0D0
      ENDIF

C Now we apply the geometric distortion to the input points so that
C the linear transformation which we derive is appropriate after
C the distortion is corrected
C
C Just use LINEAR terms

      SCOTY=COTY

C
C  Why limit the evaluation to only linear terms???
C  This fit is only a linear fit, therefore, only linear terms
C  need to be considered.  When all terms are used, it introduces
C  slight non-orthogonality of the CD matrix after correction, as
C  well as additional offsets between the chips.  AK, WJH  1-Aug-2006
C
      IF(COTY.GT.1) COTY=1

      DO I=1,4
       IF(COTY.EQ.3) THEN
        X(I)=EVAL3(XIN(I)-XCEN+XDOFF,YIN(I)-YCEN+YDOFF,XCO)-XDOFF
        Y(I)=EVAL3(XIN(I)-XCEN+XDOFF,YIN(I)-YCEN+YDOFF,YCO)-YDOFF
       ELSE IF(COTY.EQ.4) THEN
        X(I)=EVAL4(XIN(I)-XCEN+XDOFF,YIN(I)-YCEN+YDOFF,XCO)-XDOFF
        Y(I)=EVAL4(XIN(I)-XCEN+XDOFF,YIN(I)-YCEN+YDOFF,YCO)-YDOFF
       ELSE IF(COTY.EQ.5) THEN
        X(I)=EVAL5(XIN(I)-XCEN+XDOFF,YIN(I)-YCEN+YDOFF,XCO)-XDOFF
        Y(I)=EVAL5(XIN(I)-XCEN+XDOFF,YIN(I)-YCEN+YDOFF,YCO)-YDOFF
       ELSE IF(COTY.GE.6 .OR. COTY.EQ.1 .OR. COTY.EQ.2) THEN
        X(I)=EVALN(XIN(I)-XCEN+XDOFF,YIN(I)-YCEN+YDOFF,XCO,COTY)-XDOFF
        Y(I)=EVALN(XIN(I)-XCEN+XDOFF,YIN(I)-YCEN+YDOFF,YCO,COTY)-YDOFF
       ELSE IF(COTY.EQ.-3) THEN
        CALL RAD3(XIN(I)-XCEN+XDOFF,YIN(I)-YCEN+YDOFF,XCO,X(I),Y(I))
        X(I)=X(I)-XDOFF
        Y(I)=Y(I)-YDOFF
       ELSE
        X(I)=XIN(I)-XCEN
        Y(I)=YIN(I)-YCEN
       ENDIF
      ENDDO

C Restore order
      COTY=SCOTY

C Now we have the inputs and outputs and can derive the linear
C transform between them
C This is now done in a general way using least squares
C Double precision version

      CALL FITLIN(XOUT,YOUT,X,Y,4,X0,Y0,A,B,C,D,ISTAT)

      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Failed to determine mapping from WCS',
     :               1,0,ISTAT)
         ISTAT=1
         RETURN
      ENDIF

C We change a sign here to fit in with convention later
      B=-B 

C And now the linear offset
      XT=XOUT(1)-A*X(1)+B*Y(1)
      YT=YOUT(1)-C*X(1)-D*Y(1)

      XC=A
      YS=B
      XS=C
      YC=D

C Before returning reset the offsets, if there are any
      IF(NEWREF) THEN
         COTY=COTY+COOFF
         CONUM=CONUM+1
      ENDIF

      RETURN
      END

      SUBROUTINE SETWCS(OUTSCL,ORIENT,CRPIX1,CRVAL1,
     :                  CRPIX2,CRVAL2,WCS)
C
C SETWCS - convert a scale and orientation, along with
C          reference pixel and position on sky, to a WCS
C          including a CD matrix.
C
C Note: OUTSCL is in arcsecs/pix, ORIENT and CRVAL1/2
C       are in degrees.
C
C Richard Hook, 22nd September 1998
C
C Full double precision version - Richard Hook,ST-ECF/STScI June 2003
C
      IMPLICIT NONE

C Supplied:
      DOUBLE PRECISION OUTSCL,ORIENT
      DOUBLE PRECISION CRPIX1,CRPIX2,CRVAL1,CRVAL2

C Returned:
      DOUBLE PRECISION WCS(8)

C Local variables
      DOUBLE PRECISION PIBY,ROR,DSC
      PARAMETER (PIBY=3.14159265358979/180.0)

C Convert to radians for orientation
      ROR=ORIENT*PIBY

C and to degrees for scale
      DSC=OUTSCL/3600.0D0

C Set reference points
      WCS(1)=CRPIX1
      WCS(3)=CRPIX2
      WCS(2)=CRVAL1
      WCS(4)=CRVAL2

C and CD matrix (no skew, equal X,Y scales)
      WCS(5)=-DSC*DCOS(ROR)
      WCS(6)=DSC*DSIN(ROR)
      WCS(7)=WCS(6) 
      WCS(8)=-WCS(5)       

      RETURN
      END

      SUBROUTINE COPY1D(IN,OUT,N)
C
C Copy a 1d real array from one place to another.
C
C SINGLE PRECISION routine.
C
      IMPLICIT NONE

      INTEGER N
      REAL IN(*),OUT(*)

      INTEGER I

      DO I=1,N
         OUT(I)=IN(I)
      ENDDO
 
      RETURN
      END

      SUBROUTINE LCORN(DNX,DNY,ONX,ONY,
     :           XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :           SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :           USEWCS,WCSIN,WCSOUT,
     :           COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :           XMIN,XMAX,YMIN,YMAX,ISTAT)
 
C
C LCORN - transform the range of input pixel coordinates covered
C         by an image onto the output and find the extreme values.
C
C Modified to also pass the secondary geometric parameters,
C  Richard Hook, STScI, July 2000
C
C Added SECPAR parameter, Sept 2000
C
C Added check of the border and just 5 pixel margin, October 2000
C
C Added more general coefficient support, December 2000
C
      IMPLICIT NONE

      INTEGER DNX,DNY,ONX,ONY
      DOUBLE PRECISION XSH,YSH,ROT,SCALE
      LOGICAL ROTFIR,USEWCS,SECPAR
      CHARACTER*8 ALIGN
      DOUBLE PRECISION WCSIN(8),WCSOUT(8)
      INTEGER COTY,CONUM
      INTEGER XGDIM,YGDIM
      REAL PXG(XGDIM,YGDIM),PYG(XGDIM,YGDIM)
      DOUBLE PRECISION XCO(CONUM),YCO(CONUM)
      INTEGER XMIN,XMAX,YMIN,YMAX,ISTAT,I
      DOUBLE PRECISION XIN(16),YIN(16)
      DOUBLE PRECISION XOUT(16),YOUT(16),XMA,YMA,XMI,YMI

C Secondary geometrical parameters, added in V1.5
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*8 SHFR2
      LOGICAL ROTF2,DISIM

      REAL MEMR(1)
      COMMON /MEM/MEMR

      XIN(1)=1.0
      XIN(2)=1.0
      XIN(3)=DBLE(DNX)
      XIN(4)=DBLE(DNX)
      YIN(1)=1.0
      YIN(2)=DBLE(DNY)
      YIN(3)=DBLE(DNY)
      YIN(4)=1.0
      
C Transform onto output coordinate system
      CALL DRIVAL(XIN,YIN,4,DNX,DNY,ONX,ONY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,
     :            PXG,PYG,XGDIM,YGDIM,XOUT,YOUT)

C Calculate the extreme values
      XMA=MAX(XOUT(1),XOUT(2),XOUT(3),XOUT(4))
      YMA=MAX(YOUT(1),YOUT(2),YOUT(3),YOUT(4))
      XMI=MIN(XOUT(1),XOUT(2),XOUT(3),XOUT(4))
      YMI=MIN(YOUT(1),YOUT(2),YOUT(3),YOUT(4))

C Now check a few more points on the edges
      DO I=1,4
         XIN(I)=DBLE(I)*DBLE(DNX)/5.0D0
         YIN(I)=1.0D0
         XIN(I+4)=XIN(I)
         YIN(I+4)=DBLE(DNY)
         XIN(I+8)=1.0
         YIN(I+8)=DBLE(I)*DBLE(DNY)/5.0D0
         XIN(I+12)=DBLE(DNX)
         YIN(I+12)=YIN(I+8)
      ENDDO

      CALL DRIVAL(XIN,YIN,16,DNX,DNY,ONX,ONY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,
     :            PXG,PYG,XGDIM,YGDIM,XOUT,YOUT)

      DO I=1,16
         IF(XOUT(I).GT.XMA) XMA=XOUT(I)
         IF(YOUT(I).GT.YMA) YMA=YOUT(I)
         IF(XOUT(I).LT.XMI) XMI=XOUT(I)
         IF(YOUT(I).LT.YMI) YMI=YOUT(I)
      ENDDO
 
C Calculate limits allowing for a 5 pixel margin of error
      XMAX=NINT(XMA+5.0D0)
      XMIN=NINT(XMI-5.0D0)
      YMAX=NINT(YMA+5.0D0)
      YMIN=NINT(YMI-5.0D0)

      ISTAT=0
      RETURN
      END

      SUBROUTINE DRIVAL(XIN,YIN,N,DNX,DNY,ONX,ONY,REG,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,XG,YG,XGDIM,YGDIM,
     :            XOUT,YOUT)
C
C DRIVAL - apply the standard Drizzle transformation
C          from input to output pixel coordinates.
C          This may optional include a polynomial
C          distortion solution or be specified using
C          input and output WCS. It replaces the
C          logic originally in the DOBOX routine within Drizzle.
C
C  Richard Hook, ST-ECF, 5th October 1998
C   
C Modified to include the secondary geometric transformation,
C  Richard Hook, STScI, July 2000
C
C Added SECPAR logical flag, Sept 2000
C 
C Added more general coefficients, December 2000
C Used COTY to define the transformation and remove NONLIN
C Added EVAL4/5 for higher order options
C
C Added the new "REG" parameter (logical) which decides whether
C the input X and Y are simply at steps of 1.0 in X (starting at
C the value supplied) and with Y constant at the value supplied.
C This is a new feature for Drizzle V2.7 intended to avoid some
C numerical precision problems and increase efficiency.
C   Richard Hook, ST-ECF, February 2002
C
C Added distortion image support (XG,YG), 
C    Richard Hook, ST-ECF/STScI, August 2003
C
C Added "refpix" support using higher values of COTY,
C    Richard Hook, ST-ECF/STScI, December 2003
C
      IMPLICIT NONE
      INTEGER N,DNX,DNY,ONX,ONY,I
      LOGICAL REG
      DOUBLE PRECISION XIN(N),YIN(N),XOUT(N),YOUT(N)
      DOUBLE PRECISION XCEN,YCEN,XF,YF,XCORN,YCORN
      DOUBLE PRECISION XSH,YSH,ROT,SCALE
      INTEGER COTY,CONUM,COOFF,IX,IY
      PARAMETER (COOFF=100)
      LOGICAL NEWREF
      DOUBLE PRECISION XCO(CONUM),YCO(CONUM)
      INTEGER XGDIM,YGDIM
      REAL XG(XGDIM,YGDIM),YG(XGDIM,YGDIM)
      DOUBLE PRECISION XS,XC,YS,YC,X,Y,XD,YD,XDOFF,YDOFF
      DOUBLE PRECISION EVAL3,EVAL4,EVAL5,EVALN     
      DOUBLE PRECISION SINTH,COSTH,XOFF,YOFF,XT,YT,XP,YP
      LOGICAL ROTFIR,USEWCS,SECPAR,DISIM
      CHARACTER*8 ALIGN
      DOUBLE PRECISION WCSIN(8),WCSOUT(8),XDREF,YDREF

      INTEGER COMAX
      PARAMETER (COMAX=100)

C Secondary geometrical parameters, added in V1.5
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*8 SHFR2
      LOGICAL ROTF2
      DOUBLE PRECISION XUT,YUT,XC2,XS2,YC2,YS2,XT2,YT2,XP2,YP2
      DOUBLE PRECISION SINTH2,COSTH2
      
C--

      IF(ALIGN.EQ.'corner') THEN
         XCEN=DBLE(DNX/2.0)+0.5
         YCEN=DBLE(DNY/2.0)+0.5
      ELSE
         XCEN=DBLE(DNX/2.0)+1.0
         YCEN=DBLE(DNY/2.0)+1.0
      ENDIF

C Calculate some numbers to simplify things later
      SINTH=DSIN(ROT)
      COSTH=DCOS(ROT)
 
C The shifts are in units of INPUT pixels.
      XOFF=XSH/SCALE
      YOFF=YSH/SCALE
      XF=1.0D0/SCALE
      YF=1.0D0/SCALE
 
C Some more economies
      XS=XF*SINTH
      XC=XF*COSTH
      YS=YF*SINTH
      YC=YF*COSTH

C Secondary ones
      IF(SECPAR) THEN
         SINTH2=DSIN(ROT2)
         COSTH2=DCOS(ROT2)
         XS2=SINTH2
         XC2=COSTH2
         YS2=SINTH2
         YC2=COSTH2
      ENDIF
 
      IF(ALIGN.EQ.'corner') THEN
         XP=DBLE(ONX/2.0)+0.5D0
         YP=DBLE(ONY/2.0)+0.5D0
      ELSE
         XP=DBLE(ONX/2.0)+1.0D0
         YP=DBLE(ONY/2.0)+1.0D0
      ENDIF
 
      XT=XOFF+XP
      YT=YOFF+YP

C Set the secondary ones
      IF(SECPAR) THEN
         XP2=XP
         YP2=YP
         XT2=XP2+XSH2
         YT2=YP2+YSH2
      ENDIF


C Here is some new code to support the WCS option
C first we work out a linear transformation from input to
C out
      IF(USEWCS) THEN            
         CALL WCSLIN(WCSIN,WCSOUT,
     :         XCEN,YCEN,COTY,CONUM,
     :         XCO,YCO,DISIM,XG,YG,XGDIM,YGDIM,
     :         XC,YC,XS,YS,XT,YT)

         ROTFIR=.TRUE.

      ENDIF

C Check for the presence of "refpix" additional information
C in the coefficients
C If it is, set a flag and offset again
      IF(COTY.GT.COOFF/2) THEN
         NEWREF=.TRUE.
         COTY=COTY-COOFF
         XDREF=XCO(CONUM)
         YDREF=YCO(CONUM)
         XDOFF=XCEN-XDREF
         YDOFF=YCEN-YDREF
         CONUM=CONUM-1
      ELSE
         NEWREF=.FALSE.
         XDOFF=0.0
         YDOFF=0.0
      ENDIF

C We consider the case of "regular" and not-regular separately.
C Regular means that the X positions are spaced at intervals of
C one pixel (starting at XIN(1)) and the Y's are all equal to Y(1)
C
C XIN(2) and YIN(2) are used for offsets
      IF(REG) THEN

C The case of secondary parameters is also separated
C to avoid excessive calculations

C First the case with secondary parameters
      IF(SECPAR) THEN

C Note that there is an extra -1 here because it is added again
C for the first point
       X=XSCALE*(XIN(1)-1.0D0-XCEN)
       Y=YSCALE*(YIN(1)+YIN(2)-YCEN)



       DO I=1,N
         X=X+XSCALE

         IF(COTY.EQ.3) THEN
            XCORN=EVAL3(X+XDOFF,Y+YDOFF,XCO)-XDOFF
            YCORN=EVAL3(X+XDOFF,Y+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.EQ.4) THEN
            XCORN=EVAL4(X+XDOFF,Y+YDOFF,XCO)-XDOFF
            YCORN=EVAL4(X+XDOFF,Y+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.EQ.5) THEN
            XCORN=EVAL5(X+XDOFF,Y+YDOFF,XCO)-XDOFF
            YCORN=EVAL5(X+XDOFF,Y+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.GE.6 .OR. COTY.EQ.1 .OR. COTY.EQ.2) THEN
            XCORN=EVALN(X+XDOFF,Y+YDOFF,XCO,COTY)-XDOFF
            YCORN=EVALN(X+XDOFF,Y+YDOFF,YCO,COTY)-YDOFF
         ELSE IF(COTY.EQ.-3) THEN
            CALL RAD3(X+XDOFF,Y+YDOFF,XCO,XCORN,YCORN)
            XCORN=XCORN-XDOFF
            YCORN=YCORN-YDOFF
         ELSE
            XCORN=X
            YCORN=Y
         ENDIF

C If we have a distortion image we add here
         IF(DISIM) THEN
            IX=INT(X+XCEN)
            IY=INT(Y+YCEN)
            XCORN=XCORN+DBLE(XG(IX,IY))
            YCORN=YCORN+DBLE(YG(IX,IY))
         ENDIF

         
C Apply the linear transform
C There are two ways this can be done - shift then
C rotate or rotate then shift
         IF(ROTFIR) THEN
            XUT=XC*XCORN-YS*YCORN+XT-XP
            YUT=XS*XCORN+YC*YCORN+YT-YP
         ELSE
            XUT=XC*(XCORN+XSH)-YS*(YCORN+YSH)
            YUT=XS*(XCORN+XSH)+YC*(YCORN+YSH)
         ENDIF

C Apply the secondary transform
         IF(ROTF2) THEN
            XOUT(I)=XC2*XUT-YS2*YUT+XT2
            YOUT(I)=XS2*XUT+YC2*YUT+YT2
         ELSE
            XOUT(I)=XC2*(XUT+XSH2)-YS2*(YUT+YSH2)+XP2
            YOUT(I)=XS2*(XUT+XSH2)+YC2*(YUT+YSH2)+YP2
         ENDIF
        ENDDO

C and now without secondary parameters
      ELSE

C Note again the extra 1.0 here!
C In this case there are some tricks to force the result to be
C the same as for V1.41 - XIN(1) and YIN(2) are used for the
C offsets
C
C Note that XIN and YIN are used differently!
       X=-XCEN
       XD=XIN(1)-1.0D0
       Y=YIN(1)-YCEN
       YD=YIN(2)

       DO I=1,N
         X=X+1.0D0

         IF(COTY.EQ.3) THEN
            XCORN=EVAL3(X+XD+XDOFF,Y+YD+YDOFF,XCO)-XDOFF
            YCORN=EVAL3(X+XD+XDOFF,Y+YD+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.EQ.4) THEN
            XCORN=EVAL4(X+XD+XDOFF,Y+YD+YDOFF,XCO)-XDOFF
            YCORN=EVAL4(X+XD+XDOFF,Y+YD+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.EQ.5) THEN
            XCORN=EVAL5(X+XD+XDOFF,Y+YD+YDOFF,XCO)-XDOFF
            YCORN=EVAL5(X+XD+XDOFF,Y+YD+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.GE.6 .OR. COTY.EQ.1 .OR. COTY.EQ.2) THEN
            XCORN=EVALN(X+XD+XDOFF,Y+YD+YDOFF,XCO,COTY)-XDOFF
            YCORN=EVALN(X+XD+XDOFF,Y+YD+YDOFF,YCO,COTY)-YDOFF
         ELSE IF(COTY.EQ.-3) THEN
            CALL RAD3(X+XD+XDOFF,Y+YD+YDOFF,XCO,XCORN,YCORN)
            XCORN=XCORN-XDOFF
            YCORN=YCORN-YDOFF
         ELSE
            XCORN=X+XD
            YCORN=Y+YD
         ENDIF

C If we have a distortion image we add here
C (missing XD,YD offsets corrected, January 2006)
C (Removed the offsets because they cause a segfault, March, 2006)
         IF(DISIM) THEN
            IX=INT(X+XCEN)
            IY=INT(Y+YCEN)
            XCORN=XCORN+DBLE(XG(IX,IY))
            YCORN=YCORN+DBLE(YG(IX,IY))
         ENDIF

C Apply the linear transform
C There are two ways this can be done - shift then
C rotate or rotate then shift
         IF(ROTFIR) THEN
            XOUT(I)=XC*XCORN-YS*YCORN+XT
            YOUT(I)=XS*XCORN+YC*YCORN+YT
         ELSE
            XOUT(I)=XC*(XCORN+XSH)-YS*(YCORN+YSH)+XP
            YOUT(I)=XS*(XCORN+XSH)+YC*(YCORN+YSH)+YP
         ENDIF
        ENDDO
       ENDIF
      ELSE

C The case of secondary parameters is also separated
C to avoid excessive calculations

C First the case with secondary parameters
       IF(SECPAR) THEN
        DO I=1,N
         X=XSCALE*(XIN(I)-XCEN)
         Y=YSCALE*(YIN(I)-YCEN)

         IF(COTY.EQ.3) THEN
            XCORN=EVAL3(X+XDOFF,Y+YDOFF,XCO)-XDOFF
            YCORN=EVAL3(X+XDOFF,Y+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.EQ.4) THEN
            XCORN=EVAL4(X+XDOFF,Y+YDOFF,XCO)-XDOFF
            YCORN=EVAL4(X+XDOFF,Y+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.EQ.5) THEN
            XCORN=EVAL5(X+XDOFF,Y+YDOFF,XCO)-XDOFF
            YCORN=EVAL5(X+XDOFF,Y+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.GE.6 .OR. COTY.EQ.1 .OR. COTY.EQ.2) THEN
            XCORN=EVALN(X+XDOFF,Y+YDOFF,XCO,COTY)-XDOFF
            YCORN=EVALN(X+XDOFF,Y+YDOFF,YCO,COTY)-YDOFF
         ELSE IF(COTY.EQ.-3) THEN
            CALL RAD3(X+XDOFF,Y+YDOFF,XCO,XCORN,YCORN)
            XCORN=XCORN-XDOFF
            YCORN=YCORN-YDOFF
         ELSE
            XCORN=X
            YCORN=Y
         ENDIF

C If we have a distortion image we add here
         IF(DISIM) THEN
            IX=INT(X+XCEN)
            IY=INT(Y+YCEN)
            XCORN=XCORN+DBLE(XG(IX,IY))
            YCORN=YCORN+DBLE(YG(IX,IY))
         ENDIF

C Apply the linear transform
C There are two ways this can be done - shift then
C rotate or rotate then shift
         IF(ROTFIR) THEN
            XUT=XC*XCORN-YS*YCORN+XT-XP
            YUT=XS*XCORN+YC*YCORN+YT-YP
         ELSE
            XUT=XC*(XCORN+XSH)-YS*(YCORN+YSH)
            YUT=XS*(XCORN+XSH)+YC*(YCORN+YSH)
         ENDIF

C Apply the secondary transform
         IF(ROTF2) THEN
            XOUT(I)=XC2*XUT-YS2*YUT+XT2
            YOUT(I)=XS2*XUT+YC2*YUT+YT2
         ELSE
            XOUT(I)=XC2*(XUT+XSH2)-YS2*(YUT+YSH2)+XP2
            YOUT(I)=XS2*(XUT+XSH2)+YC2*(YUT+YSH2)+YP2
         ENDIF
        ENDDO

C and now without secondary parameters
      ELSE

       DO I=1,N
         X=XIN(I)-XCEN
         Y=YIN(I)-YCEN

         IF(COTY.EQ.3) THEN
            XCORN=EVAL3(X+XDOFF,Y+YDOFF,XCO)-XDOFF
            YCORN=EVAL3(X+XDOFF,Y+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.EQ.4) THEN
            XCORN=EVAL4(X+XDOFF,Y+YDOFF,XCO)-XDOFF
            YCORN=EVAL4(X+XDOFF,Y+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.EQ.5) THEN
            XCORN=EVAL5(X+XDOFF,Y+YDOFF,XCO)-XDOFF
            YCORN=EVAL5(X+XDOFF,Y+YDOFF,YCO)-YDOFF
         ELSE IF(COTY.GE.6 .OR. COTY.EQ.1 .OR. COTY.EQ.2) THEN
            XCORN=EVALN(X+XDOFF,Y+YDOFF,XCO,COTY)-XDOFF
            YCORN=EVALN(X+XDOFF,Y+YDOFF,YCO,COTY)-YDOFF
         ELSE IF(COTY.EQ.-3) THEN
            CALL RAD3(X+XDOFF,Y+YDOFF,XCO,XCORN,YCORN)
            XCORN=XCORN-XDOFF
            YCORN=YCORN-YDOFF
         ELSE
            XCORN=X
            YCORN=Y
         ENDIF


C If we have a distortion image we add here
         IF(DISIM) THEN
            IX=INT(X+XCEN)
            IY=INT(Y+YCEN)
            XCORN=XCORN+DBLE(XG(IX,IY))
            YCORN=YCORN+DBLE(YG(IX,IY))
         ENDIF

C Apply the linear transform
C There are two ways this can be done - shift then
C rotate or rotate then shift
         IF(ROTFIR) THEN
            XOUT(I)=XC*XCORN-YS*YCORN+XT
            YOUT(I)=XS*XCORN+YC*YCORN+YT
         ELSE
            XOUT(I)=XC*(XCORN+XSH)-YS*(YCORN+YSH)+XP
            YOUT(I)=XS*(XCORN+XSH)+YC*(YCORN+YSH)+YP
         ENDIF
        ENDDO
       ENDIF
      ENDIF

C Before returning reset the offsets, if there are any
      IF(NEWREF) THEN
         COTY=COTY+COOFF
         CONUM=CONUM+1
      ENDIF

      RETURN
      END

      SUBROUTINE DERICU(X,Y,CO,DX,DY)
C
C Evaluate the derivatives of a cubic polynomial distortion with
C respect to X and Y. This is mainly for inverting the distortion
C using Newton-Raphson.
C
C Richard Hook, STScI, May 1999
C
      IMPLICIT NONE

      DOUBLE PRECISION X,Y,CO(*),DX,DY

      DX=CO(2)+
     :   2.0D0*CO(4)*X+
     :   CO(5)*Y+
     :   3.0D0*CO(7)*X*X+
     :   2.0D0*CO(8)*X*Y+
     :   CO(9)*Y*Y
 
      DY=CO(3)+
     :   CO(5)*X+
     :   2.0D0*CO(6)*Y+
     :   CO(8)*X*X+
     :   2.0D0*CO(9)*X*Y+
     :   3.0D0*CO(10)*Y*Y

      RETURN
      END

      SUBROUTINE INVECU(XOUT,YOUT,XCO,YCO,ERR,XIN,YIN)
C
C Invert a cubic distortion in 2d using Newton-Raphson
C
C Supplied:
C
C Xout,Yout - double - the output position from the geometric
C                     distortion.
C
C Xco,Yco - double arrays - the cubic distortion coefficients
C
C Err - double - the accuracy required of the inversion (in X and Y)
C
C Returned:
C
C Xin, Yin - double - the position which, when distorted, ends up
C                   at Xout,Yout to accuracy Err.
C
C This method is interative and relatively slow.
C
C Richard Hook, STScI, May 1999
C
      IMPLICIT NONE

      DOUBLE PRECISION XOUT,YOUT,XCO(*),YCO(*),XIN,YIN,ERR
      DOUBLE PRECISION DXX,DXY,DYX,DYY,XO,YO,X,Y,D
      DOUBLE PRECISION EVAL3
 
C First guess
      X=2.0D0*XOUT-EVAL3(XOUT,YOUT,XCO)
      Y=2.0D0*YOUT-EVAL3(XOUT,YOUT,YCO)

      DO WHILE(.TRUE.)

C Distort current guesses
         XO=EVAL3(X,Y,XCO)
         YO=EVAL3(X,Y,YCO)

C Check for error criterion
         IF(ABS(XOUT-XO).LT.ERR .AND.
     :   ABS(YOUT-YO).LT.ERR) GO TO 999

C Calculate derivatives - there are four of these
         CALL DERICU(X,Y,XCO,DXX,DXY)
         CALL DERICU(X,Y,YCO,DYX,DYY)

C Work out the determinant
         D=DXX*DYY-DYX*DXY

C Improve guess with Newton-Raphson
         X=X+((XOUT-XO)*DYY-(YOUT-YO)*DXY)/D
         Y=Y+((YOUT-YO)*DXX-(XOUT-XO)*DYX)/D
      ENDDO

 999  CONTINUE
      XIN=X
      YIN=Y

      RETURN
      END

      DOUBLE PRECISION FUNCTION OVER(I,J,XMIN,XMAX,YMIN,YMAX)
C
C OVER - calculate overlap between an arbitrary rectangle, aligned
C        with the axes, and a pixel.
C
C This is a simplified version of the BOXER code.
C
C Richard Hook, 6th May 1998
C
      IMPLICIT NONE

      INTEGER I,J
      DOUBLE PRECISION XMIN,XMAX,YMIN,YMAX

      DOUBLE PRECISION DX,DY

      DX=MIN(XMAX,DBLE(I)+0.5D0)-MAX(XMIN,DBLE(I)-0.5D0)
      DY=MIN(YMAX,DBLE(J)+0.5D0)-MAX(YMIN,DBLE(J)-0.5D0)

      IF(DX.GT.0 .AND. DY.GT.0) THEN
         OVER=DX*DY
      ELSE
         OVER=0.0D0
      ENDIF

      RETURN
      END

      SUBROUTINE COPY1I(IN,OUT,N)
C
C Copy a 1d integer array from one place to another.
C
      IMPLICIT NONE

      INTEGER N
      INTEGER IN(*),OUT(*)

      INTEGER I

      DO I=1,N
         OUT(I)=IN(I)
      ENDDO

      RETURN
      END

      SUBROUTINE UPCON(NCON,II,JJ,OLDCON,NEWCON,DONE,LX,LY,
     :                 INTAB,NEN,MAXIM,MAXEN,UNIQID,ISTAT)
C
C Update the context image
C
C This routine is called in the heart of Drizzle when the
C context image needs to be updated.
C
C October 2000 (from the EIS version)
C Added "bitmask" style context (16 bit), January 2001
C Converted to 32bit integers
C
      IMPLICIT NONE

      INTEGER ISTAT,K
      INTEGER MAXEN,MAXIM,NEN,ICON,NN,LX,LY
      INTEGER UNIQID,INTAB(MAXIM,MAXEN)
      INTEGER OLDCON,NEWCON
      INTEGER NEWMA(100),II,JJ
      INTEGER NCON(LX,LY)
      INTEGER DONE(LX,LY)
      LOGICAL MATCH
      CHARACTER*80 CHARS

C Look up the current context value
      ICON=NCON(II,JJ)

C If it is the same as the last one we don't need to
C go further
      IF(ICON.EQ.OLDCON) THEN
         NCON(II,JJ)=NEWCON
         GO TO 88
      ENDIF

C Combine with the new one
      IF(ICON.EQ.0) THEN
         NN=0
         NEWMA(1)=1
         NEWMA(2)=UNIQID
      ELSE
         NN=INTAB(1,ICON)

C Check for whether this image is already on this pixel
         DO K=3,NN+2
            IF(UNIQID.EQ.INTAB(K,ICON)) GO TO 88
         ENDDO

C If not create the new context by adding the current image
         NEWMA(1)=NN+1
         DO K=2,NN+1
            NEWMA(K)=INTAB(K+1,ICON)
         ENDDO

C Check for too many images at a given context
         IF(NN.GT.MAXIM-3) THEN
            CALL UMSPUT(
     :  '! Too many images - context table overloaded',
     :                  1,0,ISTAT)
            ISTAT=1
            RETURN
         ENDIF

         NEWMA(NN+2)=UNIQID
      ENDIF

C Before matching sort the context array
      IF(NN.GT.0) CALL CSORT(NN+1,NEWMA(2))

C See whether we have had this context before
      DO K=NEN,1,-1
         IF(MATCH(NEWMA,INTAB(1,K),MAXIM)) THEN
            NCON(II,JJ)=K
            GO TO 88
         ENDIF
      ENDDO

C No context match found - make a new one
      NEN=NEN+1

C Check for full table
      IF(NEN.EQ.MAXEN) THEN
         CALL UMSPUT('! Context table full',1,0,ISTAT)
         ISTAT=1
         RETURN
      ENDIF

      NCON(II,JJ)=NEN
      INTAB(1,NEN)=NEWMA(1)
      INTAB(2,NEN)=0
      DO K=3,NN+3
         INTAB(K,NEN)=NEWMA(K-1)
      ENDDO

C Tell the user about this new context
      IF(NN.LE.4) THEN
         WRITE(CHARS,'(''--New context #'',I5,
     :            '' | '',I4,'' images: '',
     :                   5I7)') NEN,(NEWMA(K),K=1,NN+2)
                  ELSE
         WRITE(CHARS,'(''--New context #'',I5,
     :            '' | '',I4,'' images: '',
     :                   5I7,''...'')') NEN,(NEWMA(K),K=1,6)

      ENDIF

      IF(NEWMA(1).EQ.1) CHARS(34:34)=' '

      CALL UMSPUT(CHARS,1,0,ISTAT)

 88   CONTINUE

C Save the old values for quick comparison
      OLDCON=ICON
      NEWCON=NCON(II,JJ)

C Lastly we update the counter
      IF(OLDCON.NE.NEWCON) THEN
         IF(OLDCON.GT.0) INTAB(2,OLDCON)=INTAB(2,OLDCON)-1
         INTAB(2,NEWCON)=INTAB(2,NEWCON)+1
      ENDIF

C Note that we have been here
      DONE(II,JJ)=1

      ISTAT=0
      RETURN
      END

      SUBROUTINE LENSTR(STRING,I1,I2)
C
C Find the start and end of a string
C
      IMPLICIT NONE

      CHARACTER*(*) STRING
      INTEGER I,I1,I2
      LOGICAL IN

      IN=.FALSE.

      DO I=1,LEN(STRING)
         IF(STRING(I:I).NE.' ' .AND.
     :      .NOT.IN) THEN
           I1=I
           IN=.TRUE.
         ENDIF

         IF(STRING(I:I).EQ.' ' .AND.
     :      IN) THEN
            I2=I-1
            GO TO 99
         ENDIF
      ENDDO

99    CONTINUE

      RETURN
      END

      SUBROUTINE SETIMI(A,NX,NY,V)
C
C Set a 2d integer image to a constant value
C
      IMPLICIT NONE

      INTEGER NX,NY
      INTEGER A(NX,NY),V
      INTEGER I,J

      DO J=1,NY
         DO I=1,NX
            A(I,J)=V
         ENDDO
      ENDDO

      RETURN
      END

      LOGICAL FUNCTION MATCH(A,B,N)
C
C Match up a context against a context table
C Note that after drizzle V0.40 (EIS) the context
C table itself has an extra second column for the
C counter.
C
      IMPLICIT NONE
      INTEGER I,N
      INTEGER A(N),B(N),AN,BN

      AN=A(1)
      BN=B(1)

      IF(AN.NE.BN) THEN
         MATCH=.FALSE.
         RETURN
      ELSE
         MATCH=.TRUE.
         DO I=2,AN+1
            IF(A(I).NE.B(I+1)) THEN
               MATCH=.FALSE.
               RETURN
            ENDIF
         ENDDO
      ENDIF

      RETURN
      END

      SUBROUTINE CSORT(N,ARR)
C
C This routine is modified from the Numerical Recipes one
C to work on INTEGER arrays.
C
C It sorts an array of integers in place.
C
      INTEGER N,M,NSTACK
      INTEGER ARR(N)
      PARAMETER (M=7,NSTACK=50)
      INTEGER I,IR,J,JSTACK,K,L,ISTACK(NSTACK)
      INTEGER A,TEMP
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 12 j=l+1,ir
          a=arr(j)
          do 11 i=j-1,1,-1
            if(arr(i).le.a)goto 2
            arr(i+1)=arr(i)
11        continue
          i=0
2         arr(i+1)=a
12      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        temp=arr(k)
        arr(k)=arr(l+1)
        arr(l+1)=temp
        if(arr(l+1).gt.arr(ir))then
          temp=arr(l+1)
          arr(l+1)=arr(ir)
          arr(ir)=temp
        endif
        if(arr(l).gt.arr(ir))then
          temp=arr(l)
          arr(l)=arr(ir)
          arr(ir)=temp
        endif
        if(arr(l+1).gt.arr(l))then
          temp=arr(l+1)
          arr(l+1)=arr(l)
          arr(l)=temp
        endif
        i=l+1
        j=ir
        a=arr(l)
3       continue
          i=i+1
        if(arr(i).lt.a)goto 3
4       continue
          j=j-1
        if(arr(j).gt.a)goto 4
        if(j.lt.i)goto 5
        temp=arr(i)
        arr(i)=arr(j)
        arr(j)=temp
        goto 3
5       arr(l)=arr(j)
        arr(j)=a
        jstack=jstack+2
C        if(jstack.gt.NSTACK)pause 'NSTACK too small in sort'
        if(jstack.gt.NSTACK) stop
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1
      END

      SUBROUTINE GTGLCO(CONTAB,INTAB,MAXIM,MAXEN,NEN,ISTAT)
C
C Read a context index table from global context file
C
C Richard Hook, November 1997
C Modified to handle missing context values, March 1999
C
      IMPLICIT NONE

      INTEGER MAXIM,MAXEN
      INTEGER NEN,CN
      INTEGER INTAB(MAXIM,MAXEN)
      INTEGER I,J,K,NVAL,ISTAT
      CHARACTER*80 CONTAB
      CHARACTER*1024 BUFFER

C Try to open the context file
      OPEN(11,FILE=CONTAB,STATUS='OLD',IOSTAT=ISTAT)
      IF(ISTAT.NE.0) RETURN

C Initialise the context table
      DO J=1,MAXEN
         INTAB(1,J)=1
         DO I=2,MAXIM
            INTAB(I,J)=0
         ENDDO
      ENDDO

C Work through the lines skipping comments and blank ones
      DO WHILE(.TRUE.)
         READ(11,'(A)',END=99) BUFFER
         IF(BUFFER(1:1).NE.'#' .AND. BUFFER.NE.' ') THEN
            READ(BUFFER,*,IOSTAT=ISTAT) CN,NVAL 
            IF(ISTAT.NE.0 .OR. NVAL.GT.99 .OR. CN.GT.MAXEN) GO TO 100
            
C Check for empty contexts
            IF(NVAL.GT.0) THEN
             READ(BUFFER,*,IOSTAT=ISTAT) CN,(INTAB(K,CN),K=1,NVAL+2)
             IF(ISTAT.NE.0) GO TO 100
            ENDIF
         ENDIF
      ENDDO

 99   CONTINUE
      ISTAT=0

100   CONTINUE
      CLOSE(11)

      NEN=CN

      RETURN
      END

      SUBROUTINE PTGLCO(CONTAB,INTAB,MAXIM,MAXEN,NEN,ISTAT)
C
C Write a context index table to global context file
C
C Richard Hook, November 1997
C Modified not to write empty contexts, March 1999
C
      IMPLICIT NONE

      INTEGER MAXIM,MAXEN
      INTEGER NEN
      INTEGER INTAB(MAXIM,MAXEN)
      INTEGER I,K,ISTAT
      CHARACTER*80 CONTAB

C Open the global context file with write access - this will work
C whether or not it exists
      OPEN(12,FILE=CONTAB,IOSTAT=ISTAT)
      IF(ISTAT.NE.0) RETURN

C Write a header of sorts
      WRITE(12,'(''# GLOBAL context table'')')
      WRITE(12,'(''#'')')
      WRITE(12,'(''# Context | Nima | Npix | Unique image ids...'')')

      DO K=1,NEN
        IF(INTAB(2,K).NE.0) THEN
         WRITE(12,'(I7,I6,I10,3X,200I7)') 
     :     K,(INTAB(I,K),I=1,INTAB(1,K)+2)
        ENDIF
      ENDDO

      CLOSE(12)
      RETURN
      END


      SUBROUTINE BFILL(LUN,BUFFER,ISTAT)
C
C Read free-format numbers from an open text file and
C concatenate them into a character string buffer.
C
C This is mostly to facilitate free-format numerical
C reads from text files.
C
C Richard Hook, ST-ECF, December 2000
C
      IMPLICIT NONE

      INTEGER LUN
      CHARACTER*(*) BUFFER
      CHARACTER*256 LINE
      INTEGER ISTAT

      INTEGER IS,I1,I2

      IS=1
      BUFFER=' '

      DO WHILE(.TRUE.)
         CALL UFGLIN(LUN,LINE,ISTAT)
         IF(ISTAT.NE.0) GO TO 88

         IF(LINE.NE.' ' .AND. LINE(1:1).NE.'#') THEN
            CALL ENDS(LINE,I1,I2)
            IF(IS+I2-I1.GT.LEN(BUFFER)) THEN
               ISTAT=1
               RETURN
            ELSE
               BUFFER(IS:IS+I2-I1)=LINE(I1:I2)
               IS=IS+I2-I1+2
            ENDIF
         ENDIF
      ENDDO

88    CONTINUE
      ISTAT=0
      RETURN
      END

      SUBROUTINE ENDS(STRING,I1,I2)
C
C Find the start and end of a string
C This differs from LENSTR in that internal whitespace
C is retained rather than just the first non-white section
C being extracted.
C
      IMPLICIT NONE

      CHARACTER*(*) STRING
      INTEGER I,I1,I2

      I1=1
      I2=1

      DO I=1,LEN(STRING)
         IF(STRING(I:I).NE.' ') THEN
           I1=I
           GO TO 88
         ENDIF
      ENDDO

88    CONTINUE
      DO I=LEN(STRING),1,-1
         IF(STRING(I:I).NE.' ') THEN
           I2=I
           GO TO 99
         ENDIF
      ENDDO

99    CONTINUE

      RETURN
      END

      SUBROUTINE SET1I(A,N,V)
C
C Set a 1d array to a value
C
      IMPLICIT NONE

      INTEGER N,V,I
      INTEGER A(N)

      DO I=1,N
         A(I)=V
      ENDDO

      RETURN
      END

      SUBROUTINE FITLIN(XO,YO,X,Y,N,X0,Y0,A,B,C,D,ISTAT)
C
C Fit a linear transformation (with six parameters, equivalent
C to the linear WCS) to two sets of points.
C
C This uses the standard least-squares linear equations method.
C
C Richard Hook, ST-ECF, January 31st 2001
C
C Fully double precision version, November 2002
C
      IMPLICIT NONE

      INTEGER ISTAT,N
      DOUBLE PRECISION XO(N),YO(N),X(N),Y(N)
      DOUBLE PRECISION X0,Y0,A,B,C,D
      DOUBLE PRECISION MAT(10,10),XORG,YORG,XOORG,YOORG
      DOUBLE PRECISION DET
      DOUBLE PRECISION SIGXOX,SIGXOY,SIGXO,SIGYOX,SIGYOY,SIGYO

      INTEGER I,J

C Initialize the matrix (3x3)
      DO J=1,3
         DO I=1,3
            MAT(I,J)=0.0D0
         ENDDO
      ENDDO

C Also initialise the vectors
      SIGXOX=0.0D0
      SIGXOY=0.0D0
      SIGXO=0.0D0
      SIGYOX=0.0D0
      SIGYOY=0.0D0
      SIGYO=0.0D0

C Take off an offset
      XORG=X(1)
      YORG=Y(1)
      XOORG=XO(1)
      YOORG=YO(1)

C Setup the normal equations
      DO I=1,N
         MAT(1,1)=MAT(1,1)+(X(I)-XORG)**2
         MAT(1,2)=MAT(1,2)+(X(I)-XORG)*(Y(I)-YORG)
         MAT(1,3)=MAT(1,3)+(X(I)-XORG)
         MAT(2,2)=MAT(2,2)+(Y(I)-YORG)**2
         MAT(2,3)=MAT(2,3)+(Y(I)-YORG)
         SIGXOX=SIGXOX+(XO(I)-XOORG)*(X(I)-XORG)
         SIGXOY=SIGXOY+(XO(I)-XOORG)*(Y(I)-YORG)
         SIGXO=SIGXO+(XO(I)-XOORG)
         SIGYOX=SIGYOX+(YO(I)-YOORG)*(X(I)-XORG)
         SIGYOY=SIGYOY+(YO(I)-YOORG)*(Y(I)-YORG)
         SIGYO=SIGYO+(YO(I)-YOORG)
      ENDDO

C Use symmetry (the matrix is diagonal)
      MAT(3,3)=DBLE(N)
      MAT(2,1)=MAT(1,2)
      MAT(3,1)=MAT(1,3)
      MAT(3,2)=MAT(2,3)

C Invert the matrix (we check it isn't singular)
      CALL MATINV(MAT,3,DET)

      IF(DET.EQ.0.0D0) THEN
         CALL UMSPUT('! Linear transformation matrix is singular',
     :               1,0,ISTAT)
         ISTAT=1
         RETURN
      ENDIF

C Multiply the inverse by the vector
      A=SIGXOX*MAT(1,1)+SIGXOY*MAT(1,2)+SIGXO*MAT(1,3)
      B=SIGXOX*MAT(2,1)+SIGXOY*MAT(2,2)+SIGXO*MAT(2,3)
      X0=SIGXOX*MAT(3,1)+SIGXOY*MAT(3,2)+SIGXO*MAT(3,3)

      C=SIGYOX*MAT(1,1)+SIGYOY*MAT(1,2)+SIGYO*MAT(1,3)
      D=SIGYOX*MAT(2,1)+SIGYOY*MAT(2,2)+SIGYO*MAT(2,3)
      Y0=SIGYOX*MAT(3,1)+SIGYOY*MAT(3,2)+SIGYO*MAT(3,3)

C Note that X0 and Y0 haven't been corrected for the offsets
C Normally they are not used

      ISTAT=0
      RETURN
      END

C SUBROUTINE MATINV.F 
C
C SOURCE								       
C   BEVINGTON, PAGES 302-303.
C
C PURPOSE
C   INVERT A SYMMETRIC MATRIX AND CALCULATE ITS DETERMINANT
C
C USAGE 
C   CALL MATINV (ARRAY, NORDER, DET)
C
C DESCRIPTION OF PARAMETERS
C   ARRAY  - INPUT MATRIX WHICH IS REPLACED BY ITS INVERSE
C   NORDER - DEGREE OF MATRIX (ORDER OF DETERMINANT)
C   DET    - DETERMINANT OF INPUT MATRIX
C
C SUBROUTINES AND FUNCTION SUBPROGRAMS REQUIRED 
C   NONE
C
C COMMENT
C   DIMENSION STATEMENT VALID FOR NORDER UP TO 10
C
C This version all double precision.
C
	SUBROUTINE MATINV (ARRAY,NORDER,DET)
C
        IMPLICIT NONE

	DOUBLE PRECISION ARRAY,AMAX,SAVE,DET

        INTEGER IK,JK,I,J,K,L,NORDER
	DIMENSION ARRAY(10,10),IK(10),JK(10)
C
10	DET=1.0D0
11	DO 100 K=1,NORDER
C
C FIND LARGEST ELEMENT ARRAY(I,J) IN REST OF MATRIX
C
	AMAX=0.0D0 
21	DO 30 I=K,NORDER
	DO 30 J=K,NORDER
23	IF (DABS(AMAX)-DABS(ARRAY(I,J))) 24,24,30
24	AMAX=ARRAY(I,J) 
	IK(K)=I 
	JK(K)=J 
30	CONTINUE
C
C INTERCHANGE ROWS AND COLUMNS TO PUT AMAX IN ARRAY(K,K)
C
31	IF (AMAX) 41,32,41
32	DET=0.0D0
	GOTO 140
41	I=IK(K) 
	IF (I-K) 21,51,43
43	DO 50 J=1,NORDER
	SAVE=ARRAY(K,J) 
	ARRAY(K,J)=ARRAY(I,J)
50	ARRAY(I,J)=-SAVE
51	J=JK(K) 
	IF (J-K) 21,61,53
53	DO 60 I=1,NORDER
	SAVE=ARRAY(I,K) 
	ARRAY(I,K)=ARRAY(I,J)
60	ARRAY(I,J)=-SAVE
C
C ACCUMULATE ELEMENTS OF INVERSE MATRIX 
C
61	DO 70 I=1,NORDER
	IF (I-K) 63,70,63
63	ARRAY(I,K)=-ARRAY(I,K)/AMAX
70	CONTINUE
71	DO 80 I=1,NORDER
	DO 80 J=1,NORDER
	IF (I-K) 74,80,74
74	IF (J-K) 75,80,75
75	ARRAY(I,J)=ARRAY(I,J)+ARRAY(I,K)*ARRAY(K,J)
80	CONTINUE
81	DO 90 J=1,NORDER
	IF (J-K) 83,90,83
83	ARRAY(K,J)=ARRAY(K,J)/AMAX
90	CONTINUE
	ARRAY(K,K)=1./AMAX
100	DET=DET*AMAX
C
C RESTORE ORDERING OF MATRIX
C
101	DO 130 L=1,NORDER
	K=NORDER-L+1
	J=IK(K) 
	IF (J-K) 111,111,105
105	DO 110 I=1,NORDER
	SAVE=ARRAY(I,K) 
	ARRAY(I,K)=-ARRAY(I,J)
110	ARRAY(I,J)=SAVE 
111	I=JK(K) 
	IF (I-K) 130,130,113
113	DO 120 J=1,NORDER
	SAVE=ARRAY(K,J) 
	ARRAY(K,J)=-ARRAY(I,J)
120	ARRAY(I,J)=SAVE 
130	CONTINUE
140	RETURN
	END
    
    
      SUBROUTINE DOBOX(DATA,WEI,NDAT,NCOU,NCON,DONE,DNX,DNY,NY,
     :     YSTART,XMIN,XMAX,YMIN,YMAX,NOOVER,
     :     KERNEL,XI,XO,YI,YO,XIB,XOB,YIB,YOB,
     :     ONX,ONY,COTY,CONUM,XCO,YCO,
     :     DISIM,PXG,PYG,XGDIM,YGDIM,
     :     WTSCL,ALIGN,INCPS,EXPIN,
     :     PFRACT,SCALE,ROT,XSH,YSH,WCS,WCSOUT,ROTFIR,
     :     SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :     CON,BITCON,INTAB,MAXIM,MAXEN,NEN,UNIQID,
     :     UPDATE,USEWEI,USEWCS,ISTAT,NMISS,NSKIP)
C
C This module does the actual mapping of input flux to output images using
C "boxer", a code written by Bill Sparks for FOC geometric
C distortion correction, rather than the "drizzling" approximation.
C
C This works by calculating the positions of the four corners of
C a quadrilateral on the output grid corresponding to the corners
C of the input pixel and then working out exactly how much of each
C pixel in the output is covered, or not.
C
C In V1.6 this was simplified to use the DRIVAL routine and also
C to include some limited multi-kernel support.
C
      IMPLICIT NONE

      INTEGER DNX,DNY,ONX,ONY,NP,NY
      INTEGER I,J,II,JJ,NMISS,ISTAT,NHIT,NXI,NXA,NYI,NYA,X1,X2
      INTEGER XMIN,XMAX,YMIN,YMAX,LX,LY,YSTART
      INTEGER XGDIM,YGDIM
      REAL DATA(DNX,NY),WEI(DNX,NY),W,EXPIN
      REAL NDAT(XMAX-XMIN+1,YMAX-YMIN+1)
      REAL NCOU(XMAX-XMIN+1,YMAX-YMIN+1)
      REAL PXG(XGDIM,YGDIM),PYG(XGDIM,YGDIM)
      INTEGER COTY,CONUM,NSKIP
      DOUBLE PRECISION XCO(CONUM),YCO(CONUM)
      DOUBLE PRECISION WCS(8)
      DOUBLE PRECISION WCSOUT(8)
      DOUBLE PRECISION Y,XOUT(4),YOUT(4)
      DOUBLE PRECISION XSH,YSH,DOVER,DH,XX,YY
      DOUBLE PRECISION XIN(4),YIN(4)
      DOUBLE PRECISION ROT,XF,YF,SCALE,XXI,XXA,YYI,YYA,SCALL,SCDIS
      DOUBLE PRECISION PFRACT,OFRAC
      DOUBLE PRECISION AC,JACO,DX,DY,PFO,PFO2,S2
      DOUBLE PRECISION XI(DNX,4),YI(DNX,4),XO(DNX,4),YO(DNX,4)
      DOUBLE PRECISION XIB(DNX),YIB(DNX),XOB(DNX),YOB(DNX)
      DOUBLE PRECISION R2,ES,EFAC,NSIG,XCEN,YCEN
      DOUBLE PRECISION OVER,TEM

C Some things are still single
      REAL VC,WTSCL,D,DOW,DD

      CHARACTER*8 ALIGN,KERNEL
      LOGICAL ROTFIR,UPDATE,USEWEI,INCPS,USEWCS,NOOVER,DISIM

C Context related things
      LOGICAL CON
      LOGICAL BITCON
      INTEGER NCON(XMAX-XMIN+1,YMAX-YMIN+1)
      INTEGER DONE(XMAX-XMIN+1,YMAX-YMIN+1)
      INTEGER MAXEN,MAXIM,NEN,UNIQID
      INTEGER OLDCON,NEWCON,BV
      INTEGER INTAB(MAXIM,MAXEN)

C Space for Lanczos-style look-up-tables
      INTEGER NLUT
      PARAMETER (NLUT=512)
      REAL LANLUT(NLUT)
      REAL DEL
      PARAMETER (DEL=0.01)
      INTEGER LANORD
      INTEGER IX,IY
      DOUBLE PRECISION SDP
      
C Number of sigma to be included for gaussians
      PARAMETER (NSIG=2.5)

C Secondary geometrical parameters, added in V1.5
      LOGICAL SECPAR
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*8 SHFR2
      LOGICAL ROTF2
C--

C Some initial settings - note that the reference pixel
C position is determined by the value of ALIGN
      OLDCON=-1

C The bitmask - trimmed to the appropriate range
      NP=(UNIQID-1)/32+1
      BV=2**(UNIQID-1-(32*(NP-1)))

C In the WCS case we can't use the scale to calculate the
C Jacobian so we need to do it
C
C Note that we use the centre of the image rather than the
C reference pixel as the reference here
      IF(USEWCS) THEN         
         IF(ALIGN.EQ.'corner') THEN
            XCEN=DBLE(DNX/2.0)+0.5
            YCEN=DBLE(DNY/2.0)+0.5
         ELSE
            XCEN=DBLE(DNX/2.0)+1.0
            YCEN=DBLE(DNY/2.0)+1.0
         ENDIF

         XIN(1)=XCEN
         XIN(2)=XCEN
         XIN(3)=XCEN+1.0D0
         XIN(4)=XCEN+1.0D0
         YIN(1)=YCEN
         YIN(2)=YCEN+1.0D0
         YIN(3)=YCEN+1.0D0
         YIN(4)=YCEN

         CALL DRIVAL(XIN,YIN,4,DNX,DNY,ONX,ONY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCS,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XOUT,YOUT)

         SCALL=SQRT(1.0/ABS(0.5*((XOUT(2)-XOUT(4))*(YOUT(1)-YOUT(3)) -
     :              (XOUT(1)-XOUT(3))*(YOUT(2)-YOUT(4)))))

C Now calculate how much of this is from the geometric distortion
         XSH=0.0D0
         YSH=0.0D0
         ROT=0.0D0
         SCALE=1.0D0
         SECPAR=.FALSE.
         USEWCS=.FALSE.

         CALL DRIVAL(XIN,YIN,4,DNX,DNY,ONX,ONY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCS,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XOUT,YOUT)

         SCDIS=SQRT(1.0/ABS(0.5*((XOUT(2)-XOUT(4))*(YOUT(1)-YOUT(3)) -
     :              (XOUT(1)-XOUT(3))*(YOUT(2)-YOUT(4)))))

         USEWCS=.TRUE.
         SCALE=SCALL/SCDIS

      ENDIF

C Image subset size
      LX=XMAX-XMIN+1
      LY=YMAX-YMIN+1
      XF=1.0/SCALE
      YF=1.0/SCALE

      DH=0.5D0*PFRACT
      AC=1.0D0/(PFRACT*PFRACT)

C Recalculate the area scaling factor 
      S2=SCALE*SCALE

C Offsets
      DX=DBLE(XMIN-1)
      DY=DBLE(YMIN-1)

C Half pixfrac on output
      PFO=PFRACT/SCALE/2.0D0
      PFO2=PFO**2

C Some Gaussian related numbers
      IF(KERNEL(1:8).EQ.'gaussian') THEN
         EFAC=2.3548**2*S2*AC/2.0D0
         ES=EFAC/3.1415926
         PFO=NSIG*PFRACT/2.3548/SCALE

C Added in V2.9 - make sure this doesn't get less than 1.2
C divided by the scale so that there are never holes in the output
         IF(PFO.LT.1.2/SCALE) PFO=1.2/SCALE
      ENDIF

C Set up a look-up-table for Lanczos-style interpolation kernels
C It is assumed that the last character is an order (single digit)
      IF(KERNEL(1:7).EQ.'lanczos') THEN
         READ(KERNEL(8:8),'(I1)') LANORD
         CALL FILALU(LANORD,NLUT,DEL,LANLUT)

C Note - removed the +1 here (RNH, 21/11/2002)
         PFO=DBLE(LANORD)*PFRACT/SCALE
         SDP=SCALE/DEL/PFRACT
      ENDIF

C We skip all this if there is no overlap
      IF(.NOT.NOOVER) THEN

C This is the outer loop over all the lines in the input image

C Before we start we can fill the X arrays as they don't change
C with Y
       IF(KERNEL.EQ.'square') THEN
          XI(1,1)=1.0D0-DH
          XI(1,2)=1.0D0+DH
          XI(1,3)=1.0D0+DH
          XI(1,4)=1.0D0-DH
       ELSE
          XIB(1)=1.0D0
       ENDIF
         
C If the input image is not in CPS we need to divide by
C the exposure
       IF(.NOT.INCPS) CALL MULC(DATA,DNX,NY,1.0/EXPIN)

C Loop over input lines
       Y=DBLE(YSTART)

       DO J=1,NY
        Y=Y+1.0D0

C Check the overlap with the output
        CALL CHOVER(Y,5,DNX,DNY,ONX,ONY,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCS,WCSOUT,COTY,CONUM,XCO,YCO,
     :            DISIM,PXG,PYG,XGDIM,YGDIM,OFRAC,X1,X2)

C If the line falls completely off the output then skip it
       IF(OFRAC.NE.0.0D0) THEN

C We know there are some misses
          NMISS=NMISS+DNX-(X2-X1+1)

C Fill the X arrays as they don't change with Y
          IF(KERNEL(1:6).EQ.'square') THEN
             XI(X1,1)=DBLE(X1)-DH
             XI(X1,2)=DBLE(X1)+DH
             XI(X1,3)=DBLE(X1)+DH
             XI(X1,4)=DBLE(X1)-DH
          ELSE
             XIB(X1)=DBLE(X1)
          ENDIF
       
C At this point we can handle the different kernels separately
C First the cases where we just transform a single point rather
C than four - every case except the "classic" square-pixel kernel
       IF(KERNEL(1:6).NE.'square') THEN

         YIB(X1)=Y
         YIB(X1+1)=0.0D0

C Transform onto the output grid
         CALL DRIVAL(XIB(X1),YIB(X1),X2-X1+1,DNX,DNY,ONX,ONY,
     :            .TRUE.,XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCS,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XOB(X1),YOB(X1))

C Now we consider the different cases, first the "point" option
C where a single pixel in the output image is affected
       IF(KERNEL(1:5).EQ.'point') THEN

C Offset within the subset
         DO I=X1,X2
            II=NINT(XOB(I)-DX)
            JJ=NINT(YOB(I)-DY)

C Check it is on the output image
            IF(II.GE.1 .AND. II.LE.LX .AND.
     :        JJ.GE.1 .AND. JJ.LE.LY) THEN

               VC=NCOU(II,JJ)

C Allow for stretching because of scale change
               D=DATA(I,J)*FLOAT(S2)

C Scale the weighting mask by the scale factor
C Note that we DON'T scale by the Jacobian as it hasn't been
C calculated
               DOW=WEI(I,J)*WTSCL

C If we are creating or modifying the context image we
C do so here
               IF(CON .AND. DOW.GT.0.0) THEN
                  IF(BITCON) THEN
                     NCON(II,JJ)=IOR(NCON(II,JJ),BV)
                  ELSE
                     IF(DONE(II,JJ).EQ.0) THEN
                        CALL UPCON(
     :                 NCON,II,JJ,OLDCON,NEWCON,DONE,LX,LY,
     :                 INTAB,NEN,MAXIM,MAXEN,UNIQID,ISTAT)
                     ENDIF
                  ENDIF
               ENDIF

C Just a simple calculation without logical tests
               IF(VC.EQ.0.0) THEN
                  NDAT(II,JJ)=D
               ELSE
                  NDAT(II,JJ)=
     :             (NDAT(II,JJ)*VC+DOW*D)/
     :             (VC+DOW)
               ENDIF
               NCOU(II,JJ)=VC+DOW
            ELSE
               NMISS=NMISS+1
            ENDIF
          ENDDO

C Next "tophat" - a circular kernel giving equal weight to
C all points within a certain radius
        ELSE IF(KERNEL(1:6).EQ.'tophat') THEN

         DO I=X1,X2

C Offset within the subset
          XX=XOB(I)-DX
          YY=YOB(I)-DY

          XXI=XOB(I)-DX-PFO
          XXA=XOB(I)-DX+PFO
          YYI=YOB(I)-DY-PFO
          YYA=YOB(I)-DY+PFO

          NXI=NINT(XXI)
          NXA=NINT(XXA)
          NYI=NINT(YYI)
          NYA=NINT(YYA)

          NHIT=0

C Allow for stretching because of scale change
          D=DATA(I,J)*FLOAT(S2)

C Scale the weighting mask by the scale factor
C and inversely by the Jacobian to ensure conservation
C of weight in the output
          DOW=WEI(I,J)*WTSCL
          DD=DOW*D

C Loop over output pixels which could be affected
          DO JJ=NYI,NYA
           DO II=NXI,NXA

C Check it is on the output image
            IF(II.GE.1 .AND. II.LE.LX .AND.
     :        JJ.GE.1 .AND. JJ.LE.LY) THEN

C Radial distance
               R2=(XX-DBLE(II))**2+(YY-DBLE(JJ))**2

C Weight is one within the specified radius and zero outside
C Note: weight isn't conserved in this case
               IF(R2.LE.PFO2) THEN

C Count the hits
                NHIT=NHIT+1

                VC=NCOU(II,JJ)

C If we are creating or modifying the context image we
C do so here
               IF(CON .AND. DOW.GT.0.0) THEN
                  IF(BITCON) THEN
                     NCON(II,JJ)=IOR(NCON(II,JJ),BV)
                  ELSE
                     IF(DONE(II,JJ).EQ.0) THEN
                        CALL UPCON(
     :                 NCON,II,JJ,OLDCON,NEWCON,DONE,LX,LY,
     :                 INTAB,NEN,MAXIM,MAXEN,UNIQID,ISTAT)
                     ENDIF
                  ENDIF
               ENDIF

C Just a simple calculation without logical tests
                IF(VC.EQ.0.0) THEN
                   NDAT(II,JJ)=D
                ELSE
                   NDAT(II,JJ)=
     :              (NDAT(II,JJ)*VC+DD)/
     :              (VC+DOW)
                ENDIF
                NCOU(II,JJ)=VC+DOW
              ENDIF
             ENDIF
           ENDDO
          ENDDO

C Count cases where the pixel is off the output image
         IF(NHIT.EQ.0) NMISS=NMISS+1

         ENDDO

C Next a gaussian weighting kernel with FWHM = pixfrac/scale
        ELSE IF(KERNEL(1:8).EQ.'gaussian') THEN
         DO I=X1,X2

C Offset within the subset
          XX=XOB(I)-DX
          YY=YOB(I)-DY

          XXI=XOB(I)-DX-PFO
          XXA=XOB(I)-DX+PFO
          YYI=YOB(I)-DY-PFO
          YYA=YOB(I)-DY+PFO

          NXI=NINT(XXI)
          NXA=NINT(XXA)
          NYI=NINT(YYI)
          NYA=NINT(YYA)

          NHIT=0

C Allow for stretching because of scale change
          D=DATA(I,J)*FLOAT(S2)

C Scale the weighting mask by the scale factor
C and inversely by the Jacobian to ensure conservation
C of weight in the output
          W=WEI(I,J)*WTSCL

C Loop over output pixels which could be affected
          DO JJ=NYI,NYA
           DO II=NXI,NXA

C Check it is on the output image
            IF(II.GE.1 .AND. II.LE.LX .AND.
     :        JJ.GE.1 .AND. JJ.LE.LY) THEN

C Radial distance
               R2=(XX-DBLE(II))**2+(YY-DBLE(JJ))**2

C Weight is a scaled gaussian function of radial distance
               DOVER=ES*EXP(-R2*EFAC)

C Count the hits
               NHIT=NHIT+1

               VC=NCOU(II,JJ)
               DOW=FLOAT(DOVER*W)

C If we are creating or modifying the context image we
C do so here
               IF(CON .AND. DOW.GT.0.0) THEN
                  IF(BITCON) THEN
                     NCON(II,JJ)=IOR(NCON(II,JJ),BV)
                  ELSE
                     IF(DONE(II,JJ).EQ.0) THEN
                        CALL UPCON(
     :                 NCON,II,JJ,OLDCON,NEWCON,DONE,LX,LY,
     :                 INTAB,NEN,MAXIM,MAXEN,UNIQID,ISTAT)
                     ENDIF
                  ENDIF
               ENDIF

C Just a simple calculation without logical tests
               IF(VC.EQ.0.0) THEN
                  NDAT(II,JJ)=D
               ELSE
                  NDAT(II,JJ)=
     :             (NDAT(II,JJ)*VC+DOW*D)/
     :             (VC+DOW)
               ENDIF
               NCOU(II,JJ)=VC+DOW
             ENDIF
           ENDDO
          ENDDO

C Count cases where the pixel is off the output image
         IF(NHIT.EQ.0) NMISS=NMISS+1

         ENDDO

C Next a Lanczos weighting kernel 
        ELSE IF(KERNEL(1:7).EQ.'lanczos') THEN
         DO I=X1,X2

C Offset within the subset
          XX=XOB(I)-DX
          YY=YOB(I)-DY

          XXI=XOB(I)-DX-PFO
          XXA=XOB(I)-DX+PFO
          YYI=YOB(I)-DY-PFO
          YYA=YOB(I)-DY+PFO

          NXI=NINT(XXI)
          NXA=NINT(XXA)
          NYI=NINT(YYI)
          NYA=NINT(YYA)

          NHIT=0

C Allow for stretching because of scale change
          D=DATA(I,J)*FLOAT(S2)

C Scale the weighting mask by the scale factor
C and inversely by the Jacobian to ensure conservation
C of weight in the output
          W=WEI(I,J)*WTSCL

C Loop over output pixels which could be affected
          DO JJ=NYI,NYA
           DO II=NXI,NXA

C Check it is on the output image
            IF(II.GE.1 .AND. II.LE.LX .AND.
     :        JJ.GE.1 .AND. JJ.LE.LY) THEN

C X and Y offsets
               IX=NINT(ABS(XX-DBLE(II))*SDP)+1
               IY=NINT(ABS(YY-DBLE(JJ))*SDP)+1

C Weight is product of Lanczos function values in X and Y
               DOVER=LANLUT(IX)*LANLUT(IY)

C Count the hits
               NHIT=NHIT+1

               VC=NCOU(II,JJ)
               DOW=FLOAT(DOVER*W)

C If we are creating or modifying the context image we
C do so here
               IF(CON .AND. DOW.GT.0.0) THEN
                  IF(BITCON) THEN
                     NCON(II,JJ)=IOR(NCON(II,JJ),BV)
                  ELSE
                     IF(DONE(II,JJ).EQ.0) THEN
                        CALL UPCON(
     :                 NCON,II,JJ,OLDCON,NEWCON,DONE,LX,LY,
     :                 INTAB,NEN,MAXIM,MAXEN,UNIQID,ISTAT)
                     ENDIF
                  ENDIF
               ENDIF

C Just a simple calculation without logical tests
               IF(VC.EQ.0.0) THEN
                  NDAT(II,JJ)=D
               ELSE

C There is an extra check here in the Lanczos case because the weight can
C be negative which means that, just occasionally, the new and existing
C weights at a given position can be have opposite signs and zero sum...
                IF(VC+DOW.EQ.0.0) THEN
                   NDAT(II,JJ)=D
                ELSE
                  NDAT(II,JJ)=
     :             (NDAT(II,JJ)*VC+DOW*D)/
     :             (VC+DOW)
                ENDIF
               ENDIF
               NCOU(II,JJ)=VC+DOW

             ENDIF
           ENDDO
          ENDDO

C Count cases where the pixel is off the output image
         IF(NHIT.EQ.0) NMISS=NMISS+1

         ENDDO

C The "turbo" option with a constant square pixfrac (as used in EIS Drizzle)
        ELSE IF(KERNEL(1:5).EQ.'turbo') THEN
         DO I=X1,X2

C Offset within the subset
          XXI=XOB(I)-DX-PFO
          XXA=XOB(I)-DX+PFO
          YYI=YOB(I)-DY-PFO
          YYA=YOB(I)-DY+PFO

          NXI=NINT(XXI)
          NXA=NINT(XXA)
          NYI=NINT(YYI)
          NYA=NINT(YYA)

          NHIT=0

C Allow for stretching because of scale change
          D=DATA(I,J)*FLOAT(S2)

C Scale the weighting mask by the scale factor
C and inversely by the Jacobian to ensure conservation
C of weight in the output
          W=WEI(I,J)*WTSCL

C Loop over output pixels which could be affected
          DO JJ=NYI,NYA
           DO II=NXI,NXA

C Check it is on the output image
            IF(II.GE.1 .AND. II.LE.LX .AND.
     :        JJ.GE.1 .AND. JJ.LE.LY) THEN

C Calculate the overlap using the simpler "aligned" box routine
              DOVER=OVER(II,JJ,XXI,XXA,YYI,YYA)

              IF(DOVER.GT.0.0) THEN

C Correct for pixfrac area factor
               DOVER=DOVER*S2*AC

C Count the hits
               NHIT=NHIT+1

               VC=NCOU(II,JJ)

               DOW=FLOAT(DOVER*W)

C If we are creating or modifying the context image we
C do so here
               IF(CON .AND. DOW.GT.0.0) THEN
                  IF(BITCON) THEN
                     NCON(II,JJ)=IOR(NCON(II,JJ),BV)
                  ELSE
                     IF(DONE(II,JJ).EQ.0) THEN
                        CALL UPCON(
     :                 NCON,II,JJ,OLDCON,NEWCON,DONE,LX,LY,
     :                 INTAB,NEN,MAXIM,MAXEN,UNIQID,ISTAT)
                     ENDIF
                  ENDIF
               ENDIF

C Just a simple calculation without logical tests
               IF(VC.EQ.0.0) THEN
                  NDAT(II,JJ)=D
               ELSE
                  NDAT(II,JJ)=
     :             (NDAT(II,JJ)*VC+DOW*D)/
     :             (VC+DOW)
               ENDIF
               NCOU(II,JJ)=VC+DOW
             ENDIF
            ENDIF
           ENDDO
          ENDDO

C Count cases where the pixel is off the output image
         IF(NHIT.EQ.0) NMISS=NMISS+1

         ENDDO

C End of the "single point" transform case statement
        ENDIF

C Next the "classic" drizzle square kernel...
C this is different because we have to transform all four corners
C of the shrunken pixel
       ELSE 

C Set the start corner positions - only in Y,
C X is already done
        YI(X1,1)=Y
        YI(X1,2)=Y
        YI(X1,3)=Y
        YI(X1,4)=Y
        YI(X1+1,1)=+DH
        YI(X1+1,2)=+DH
        YI(X1+1,3)=-DH
        YI(X1+1,4)=-DH

C Transform onto the output grid
      CALL DRIVAL(XI(X1,1),YI(X1,1),X2-X1+1,DNX,DNY,ONX,ONY,.TRUE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCS,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XO(X1,1),YO(X1,1))

      CALL DRIVAL(XI(X1,2),YI(X1,2),X2-X1+1,DNX,DNY,ONX,ONY,.TRUE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCS,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XO(X1,2),YO(X1,2))

      CALL DRIVAL(XI(X1,3),YI(X1,3),X2-X1+1,DNX,DNY,ONX,ONY,.TRUE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCS,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XO(X1,3),YO(X1,3))

      CALL DRIVAL(XI(X1,4),YI(X1,4),X2-X1+1,DNX,DNY,ONX,ONY,.TRUE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCS,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XO(X1,4),YO(X1,4))

      DO I=X1,X2

C Offset within the subset
         XOUT(1)=XO(I,1)-DX
         XOUT(2)=XO(I,2)-DX
         XOUT(3)=XO(I,3)-DX
         XOUT(4)=XO(I,4)-DX
         YOUT(1)=YO(I,1)-DY
         YOUT(2)=YO(I,2)-DY
         YOUT(3)=YO(I,3)-DY
         YOUT(4)=YO(I,4)-DY

C Work out the area of the quadrilateral on the output grid
C Note that this expression expects the points to be in
C clockwise order
          JACO=0.5*((XOUT(2)-XOUT(4))*(YOUT(1)-YOUT(3)) -
     :              (XOUT(1)-XOUT(3))*(YOUT(2)-YOUT(4)))
          IF (JACO.LT.0.0D0) THEN
             JACO = JACO*-1.0D0
             TEM=XOUT(2)
             XOUT(2)=XOUT(4)
             XOUT(4)=TEM
             TEM=YOUT(2)
             YOUT(2)=YOUT(4)
             YOUT(4)=TEM
          ENDIF

          NHIT=0

C Allow for stretching because of scale change
          D=DATA(I,J)*FLOAT(S2)

C Scale the weighting mask by the scale factor
C and inversely by the Jacobian to ensure conservation
C of weight in the output
          W=WEI(I,J)*WTSCL

C Loop over output pixels which could be affected
          DO JJ=NINT(MIN(YOUT(1),YOUT(2),YOUT(3),YOUT(4))),
     :          NINT(MAX(YOUT(1),YOUT(2),YOUT(3),YOUT(4)))

           DO II=NINT(MIN(XOUT(1),XOUT(2),XOUT(3),XOUT(4))),
     :          NINT(MAX(XOUT(1),XOUT(2),XOUT(3),XOUT(4)))
            
C Check it is on the output image
            IF(II.GE.1 .AND. II.LE.LX .AND.
     :        JJ.GE.1 .AND. JJ.LE.LY) THEN

C Call boxer to calculate overlap
              CALL BOXER(II,JJ,XOUT,YOUT,DOVER)

              IF(DOVER.GT.0.0) THEN

C Re-normalise the area overlap using the Jacobian
               DOVER=DOVER/JACO

C Count the hits
               NHIT=NHIT+1

               VC=NCOU(II,JJ)
               DOW=FLOAT(DOVER*W)

C If we are creating or modifying the context image we
C do so here
               IF(CON .AND. DOW.GT.0.0) THEN
                  IF(BITCON) THEN
                     NCON(II,JJ)=IOR(NCON(II,JJ),BV)
                  ELSE
                     IF(DONE(II,JJ).EQ.0) THEN
                        CALL UPCON(
     :                 NCON,II,JJ,OLDCON,NEWCON,DONE,LX,LY,
     :                 INTAB,NEN,MAXIM,MAXEN,UNIQID,ISTAT)
                     ENDIF
                  ENDIF
               ENDIF

C Just a simple calculation without logical tests
               IF(VC.EQ.0.0) THEN
                  NDAT(II,JJ)=D
               ELSE
                  NDAT(II,JJ)=
     :             (NDAT(II,JJ)*VC+DOW*D)/
     :             (VC+DOW)
               ENDIF
               NCOU(II,JJ)=VC+DOW
             ENDIF
            ENDIF
           ENDDO
         ENDDO

C Count cases where the pixel is off the output image
         IF(NHIT.EQ.0) NMISS=NMISS+1

         ENDDO

C End of the kernel "case" blocks
        ENDIF
       ELSE

C If we are skipping a line count it
        NSKIP=NSKIP+1
        NMISS=NMISS+DNX
       ENDIF
       ENDDO
      ELSE

C If there is no overlap at all set appropriate values
       NSKIP=DNY
       NMISS=DNX*DNY
      ENDIF

C Set good status if we get this far
      ISTAT=0

      RETURN
      END

      SUBROUTINE UPWCS(WCSIN,WCSOUT,DNX,DNY,ONX,ONY,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM)

C 
C Update the WCS to include the drizzling transformations
C
C This is done by applying the transform to a unit square at
C the centre pixel in the input whilst retaining the same
C reference point on the sky.
C
C    Richard Hook, ST-ECF, September 2000
C
C Modified to use double precision, Richard Hook, STScI/ST-ECF, June 2003
C
C Linear terms only used - Richard Hook, STScI/ST-ECF, January 2004
C
      IMPLICIT NONE

      INTEGER DNX,DNY,ONX,ONY,ISTAT
      DOUBLE PRECISION XSH,YSH,ROT,SCALE
      CHARACTER*8 ALIGN,SHFR2
      LOGICAL ROTFIR,SECPAR,ROTF2,USEWCS
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      INTEGER COTY,CONUM
      INTEGER XGDIM,YGDIM
      REAL PXG(XGDIM,YGDIM),PYG(XGDIM,YGDIM)
      DOUBLE PRECISION XCO(CONUM),YCO(CONUM)
      INTEGER COMAX,SCOTY
      PARAMETER (COMAX=100)

      DOUBLE PRECISION AM,BM,CM,DM,WCSIN(8),WCSOUT(8)
      DOUBLE PRECISION W5,W6,W7,W8
      DOUBLE PRECISION XIN(3),YIN(3),XOUT(3),YOUT(3)
      LOGICAL DISIM, OLDDIS

      REAL MEMR(1)
      COMMON /MEM/MEMR

C If we have the WCS already just return
      IF(USEWCS) RETURN

C Set up a single point at the reference pixel to map the reference point
      XIN(1)=WCSIN(1)
      YIN(1)=WCSIN(3)

C Verify that Reference pixel falls within distortion correction image
C If not, turn off use of distortion correction image
      OLDDIS = DISIM
      IF (XIN(1).LT.1 .OR. 
     :    XIN(1).GT.DNX .OR.
     :    YIN(1).LT.1 .OR.
     :    YIN(1).GT.DNY) THEN
         DISIM = .FALSE.
      ENDIF
C Transform
      CALL DRIVAL(XIN,YIN,1,DNX,DNY,ONX,ONY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XOUT,YOUT)
      DISIM = OLDDIS
C We can immediately set the reference point on the sky
      WCSOUT(1)=XOUT(1)
      WCSOUT(3)=YOUT(1)
      WCSOUT(2)=WCSIN(2)
      WCSOUT(4)=WCSIN(4)

C Set up a 1x1 box at the centre pixel (three sides only)
C to allow us to update the WCS
      XIN(2)=XIN(1)+1.0D0
      YIN(2)=YIN(1)    
      XIN(3)=XIN(1)
      YIN(3)=YIN(1)+1.0D0

C Transform
C Only use LINEAR terms and ignore distortion images

      SCOTY=COTY
      COTY=1
      OLDDIS = DISIM
      
      DISIM = .FALSE.
      CALL DRIVAL(XIN,YIN,3,DNX,DNY,ONX,ONY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XOUT,YOUT)

C Restore order
      COTY=SCOTY
      DISIM = OLDDIS

C Now work out the effective CD matrix of the transformation
      AM=(XOUT(2)-XOUT(1))
      BM=(XOUT(3)-XOUT(1))
      CM=(YOUT(2)-YOUT(1))
      DM=(YOUT(3)-YOUT(1))

C Check the determinant for singularity
      IF(AM*DM-BM*CM.EQ.0.0D0) THEN
         CALL UMSPUT('! Matrix is singular, cannot update WCS',
     :               1,0,ISTAT)
      ELSE

C Invert the matrix
         CALL INMAT(AM,BM,CM,DM)

C Correct the CD matrix
         W5=AM*WCSIN(5)+CM*WCSIN(7)
         W6=AM*WCSIN(6)+CM*WCSIN(8)
         W7=BM*WCSIN(5)+DM*WCSIN(7)
         W8=BM*WCSIN(6)+DM*WCSIN(8)

         WCSOUT(5)=W5
         WCSOUT(6)=W6
         WCSOUT(7)=W7
         WCSOUT(8)=W8
      ENDIF

      RETURN
      END

      SUBROUTINE BOXER (IS, JS, X, Y, DAREA)
C
C BOXER -- compute area of box overlap
C
C Calculate the area common to input clockwise polygon x(n), y(n) with
C square (is, js) to (is+1, js+1).
C This version is for a quadrilateral.
C
C W.B. Sparks STScI 2-June-1990.
C Phil Hodge        20-Nov-1990  Change calling sequence; single precision.
C Richard Hook ECF  24-Apr-1996  Change coordinate origin
C                                so that centre of pixel has integer position
C                   03-Jan-2001  Removed accuracy check

      IMPLICIT NONE

      INTEGER IS, JS
      DOUBLE PRECISION X(*), Y(*)
      DOUBLE PRECISION DAREA
C--
      DOUBLE PRECISION PX(4), PY(4), SUM
      DOUBLE PRECISION SGAREA
      INTEGER I

C Set up coords relative to unit square at origin
C Note that the +0.5s were added when this code was
C included in DRIZZLE
      DO I = 1, 4
         PX(I) = X(I) - IS +0.5D0
         PY(I) = Y(I) - JS +0.5D0
      ENDDO
 
C For each line in the polygon (or at this stage, input quadrilateral)
C calculate the area common to the unit square (allow negative area for
C subsequent `vector' addition of subareas).
      SUM = 0.0D0
      DO I = 1, 3
         SUM = SUM + SGAREA (PX(I), PY(I), PX(I+1), PY(I+1), IS, JS)
      ENDDO   

      SUM = SUM + SGAREA (PX(4), PY(4), PX(1), PY(1), IS, JS)
      DAREA = SUM

      RETURN
      END

      DOUBLE PRECISION FUNCTION SGAREA (X1, Y1, X2, Y2, IS, JS)
C
C To calculate area under a line segment within unit square at origin.
C This is used by BOXER
C
      IMPLICIT NONE
      DOUBLE PRECISION X1, Y1, X2, Y2

      INTEGER IS,JS
      DOUBLE PRECISION M, C, DX
      DOUBLE PRECISION XLO, XHI, YLO, YHI, XTOP
      LOGICAL NEGDX

      DX = X2 - X1

C Trap vertical line
      IF (DX .EQ. 0.0D0) THEN
         SGAREA = 0.0D0
         GO TO 80
      ENDIF

C Order the two input points in x
      IF (X1 .LT. X2) THEN
         XLO = X1
         XHI = X2
      ELSE
         XLO = X2
         XHI = X1
      ENDIF

C And determine the bounds ignoring y for now
      IF (XLO .GE. 1.0D0) THEN
         SGAREA = 0.0D0
         GO TO 80
      ENDIF

      IF (XHI .LE. 0.0D0) THEN
         SGAREA = 0.0D0
         GO TO 80
      ENDIF

      XLO = MAX (XLO, 0.0D0)
      XHI = MIN (XHI, 1.0D0)

C Now look at y
C basic info about the line y = mx + c
      NEGDX = (DX .LT. 0.0D0)
      M     = (Y2 - Y1) / DX
      C     = Y1 - M * X1
      YLO = M * XLO + C
      YHI = M * XHI + C

C Trap segment entirely below axis
      IF (YLO .LE. 0.0D0 .AND. YHI .LE. 0.0D0) THEN
         SGAREA = 0.0D0
         GO TO 80
      ENDIF

C Adjust bounds if segment crosses axis (to exclude anything below axis)
      IF (YLO .LT. 0.0D0) THEN
         YLO = 0.0D0
         XLO = -C/M
      ENDIF
      IF (YHI .LT. 0.0D0) THEN
         YHI = 0.0D0
         XHI = -C/M
      ENDIF

C There are four possibilities: both y below 1, both y above 1
C and one of each.

      IF (YLO .GE. 1.0D0 .AND. YHI .GE. 1.0D0) THEN

C Line segment is entirely above square
         IF (NEGDX) THEN
            SGAREA = XLO - XHI
         ELSE
            SGAREA = XHI - XLO
         ENDIF
         GO TO 80
      ENDIF

      IF (YLO .LE. 1.0D0 .AND. YHI .LE. 1.0D0) THEN

C Segment is entirely within square
         IF (NEGDX) THEN
            SGAREA = 0.5D0 * (XLO-XHI) * (YHI+YLO)
         ELSE
            SGAREA = 0.5D0 * (XHI-XLO) * (YHI+YLO)
         END IF
         GO TO 80
      ENDIF

C otherwise it must cross the top of the square
      XTOP = (1.0D0 - C) / M

      IF (YLO .LT. 1.0D0) THEN
         IF (NEGDX) THEN
            SGAREA = -(0.5D0 * (XTOP-XLO) * (1.0D0+YLO) + XHI - XTOP)
         ELSE
            SGAREA = 0.5D0 * (XTOP-XLO) * (1.0D0+YLO) + XHI - XTOP
         END IF
         GO TO 80
      ENDIF

      IF (NEGDX) THEN
         SGAREA = -(0.5D0 * (XHI-XTOP) * (1.0D0+YHI) + XTOP-XLO)
      ELSE
         SGAREA = 0.5D0 * (XHI-XTOP) * (1.0D0+YHI) + XTOP-XLO
      END IF

   80 CONTINUE

      RETURN
      END


      SUBROUTINE BUPWCS(WCSIN,WCSOUT,DNX,DNY,ONX,ONY,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,PXDIM,PYDIM)

C
C Update the WCS to include the drizzling transformations
C
C This version is for BLOT and works the other way around to
C UPWCS.
C
C This is done by applying the transform to a unit square at
C the reference pixel in the input whilst retaining the same
C reference point on the sky.
C
C    Richard Hook, ST-ECF, December 2001
C
      IMPLICIT NONE

      INTEGER DNX,DNY,ONX,ONY
      DOUBLE PRECISION XSH,YSH,ROT,SCALE
      CHARACTER*8 ALIGN,SHFR2
      LOGICAL ROTFIR,SECPAR,ROTF2,USEWCS
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      INTEGER COTY,CONUM
      DOUBLE PRECISION XCO(CONUM),YCO(CONUM)
      LOGICAL DISIM
      INTEGER PXDIM,PYDIM
      REAL PXG(PXDIM,PYDIM),PYG(PXDIM,PYDIM)

      DOUBLE PRECISION WCSIN(8),WCSOUT(8)
      DOUBLE PRECISION OFF
      DOUBLE PRECISION XIN(3),YIN(3),XOUT(3),YOUT(3)
      DOUBLE PRECISION R(3),D(3)
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793)

      PARAMETER (OFF=0.1D1)

C If we have the WCS already just return
      IF(USEWCS) RETURN

C Set up a 1x1 box at the centre of the output image (three sides only)
      XIN(1)=DBLE(ONX/2.0)
      YIN(1)=DBLE(ONY/2.0)
      XIN(2)=DBLE(ONX/2.0)+OFF
      YIN(2)=DBLE(ONY/2.0)
      XIN(3)=DBLE(ONX/2.0)
      YIN(3)=DBLE(ONY/2.0)+OFF

C Transform
      CALL DRIVAL(XIN,YIN,3,ONX,ONY,DNX,DNY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,DISIM,PXG,PYG,PXDIM,PYDIM,
     :            XOUT,YOUT)

C Convert the output pixel position to a sky position using
C the WCS
      CALL XY2RD(XOUT(1),YOUT(1),R(1),D(1),WCSIN)
      CALL XY2RD(XOUT(2),YOUT(2),R(2),D(2),WCSIN)
      CALL XY2RD(XOUT(3),YOUT(3),R(3),D(3),WCSIN)

C We can immediately set the reference point on the sky
      WCSOUT(1)=XIN(1)
      WCSOUT(3)=YIN(1)
      WCSOUT(2)=R(1)
      WCSOUT(4)=D(1)

C Now work out the effective CD matrix of the transformation
C Note - 6,7 were swapped (I think) originally????
      WCSOUT(5)=DCOS(D(1)*PI/180.0D0)*(R(2)-R(1))/OFF
      WCSOUT(6)=(D(2)-D(1))/OFF
      WCSOUT(7)=DCOS(D(1)*PI/180.0D0)*(R(3)-R(1))/OFF
      WCSOUT(8)=(D(3)-D(1))/OFF

      RETURN
      END


      SUBROUTINE FILALU(ORDER,NPIX,DEL,LANLUT)
C
C Fill up a look-up-table of Lanczos interpolation kernel values
C for rapid weighting determination for KERNEL='lanczos3' etc.
C
C Supplied:
C
C  ORDER - integer, the order of the kernel.
C
C  NPIX - integer, the size of the look-up-table. 
C
C  DEL - real, the spacings of the sampling of the function.
C
C Returned:
C
C  LANLUT - real, 1d array of look-up values. This is a single-sided
C           Lanczos function with LANLUT(1) being the central value
C
C Note that no checking is done to see whether the values are sensible.
C
C    Richard Hook, ST-ECF/STScI, August 2002
C
C Changed to REAL rather than DOUBLE, Richard Hook, ST-ECF/STScI, Feb 2004
C
      IMPLICIT NONE

      INTEGER NPIX,ORDER
      REAL LANLUT(NPIX),DEL
      INTEGER I

      REAL FOR,POFF,PI
      PARAMETER (PI=3.141592653)

      FOR=REAL(ORDER)

C Set the first value to avoid arithmetic problems
      LANLUT(1)=1.0

      DO I=2,NPIX
         POFF=PI*(REAL(I)-1.0)*DEL

         IF(POFF.LT.PI*FOR) THEN
            LANLUT(I)=SIN(POFF)/(POFF)*SIN(POFF/FOR)/(POFF/FOR)
         ELSE
            LANLUT(I)=0.0
         ENDIF
      ENDDO

      RETURN
      END

      SUBROUTINE CHOVER(Y,MARGIN,DNX,DNY,ONX,ONY,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,COTY,CONUM,XCO,YCO,
     :            DISIM,PXG,PYG,XGDIM,YGDIM,OFRAC,X1,X2)
C
C Check how much of a line will overlap an output image, if any,
C after applying the standard drizzle transformation.
C
C This is intended to allow the number of points which are needlessly
C drizzled outside the output image to be minimized.
C
C Richard Hook, ST-ECF at STScI, April 2003
C
      IMPLICIT NONE

C Supplied and returned
      DOUBLE PRECISION XSH,YSH,ROT,SCALE
      LOGICAL ROTFIR,USEWCS,SECPAR
      CHARACTER*8 ALIGN
      DOUBLE PRECISION WCSIN(8),WCSOUT(8)
      INTEGER COTY,CONUM
      INTEGER XGDIM,YGDIM
      REAL PXG(XGDIM,YGDIM),PYG(XGDIM,YGDIM)
      DOUBLE PRECISION XCO(CONUM),YCO(CONUM)
      INTEGER DNX,DNY,ONX,ONY
      LOGICAL DISIM

C Secondary geometrical parameters, added in V1.5
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*8 SHFR2
      LOGICAL ROTF2

      INTEGER NPOINT,STEP,FIRST,LAST,NHIT,NMISS
      PARAMETER (NPOINT=21)
      DOUBLE PRECISION Y,OFRAC
      DOUBLE PRECISION XVAL(NPOINT),YVAL(NPOINT)
      DOUBLE PRECISION XOUT(NPOINT),YOUT(NPOINT)
      LOGICAL LOGO(NPOINT)
      INTEGER MARGIN
      INTEGER X1,X2

      REAL MEMR(1)
      COMMON /MEM/MEMR

C Local variables
      INTEGER I,NP

C Loop along line
      IF(DNX.LT.NPOINT) THEN
         STEP=1
      ELSE
         STEP=DNX/(NPOINT/2)
      ENDIF

      FIRST=0
      LAST=0

      NP=0
      DO I=1,DNX,STEP
         NP=NP+1
         XVAL(NP)=DBLE(I)
         YVAL(NP)=Y
      ENDDO 

C Check end point
      IF(XVAL(NP).LT.DBLE(DNX)) THEN
         XVAL(NP+1)=DBLE(DNX)
         YVAL(NP+1)=Y
         NP=NP+1
      ENDIF

C Transform
      CALL DRIVAL(XVAL,YVAL,NP,DNX,DNY,ONX,ONY,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,COTY,CONUM,
     :            XCO,YCO,DISIM,PXG,PYG,XGDIM,YGDIM,
     :            XOUT,YOUT)

C Check where the overlap starts and ends
      DO I=1,NP
         LOGO(I)=.FALSE.
      ENDDO

      DO I=1,NP-1
         IF(MAX(XOUT(I),XOUT(I+1)).GE.1.0-MARGIN .AND.
     :      MIN(XOUT(I),XOUT(I+1)).LE.ONX+MARGIN .AND.
     :      MAX(YOUT(I),YOUT(I+1)).GE.1.0-MARGIN .AND.
     :      MIN(YOUT(I),YOUT(I+1)).LE.ONY+MARGIN) THEN
            LOGO(I)=.TRUE.
            LOGO(I+1)=.TRUE.
         ENDIF
      ENDDO

      NHIT=0
      DO I=1,NP
         IF(LOGO(I)) NHIT=NHIT+1
      ENDDO

      NMISS=NP-NHIT

      IF(NHIT.EQ.0) THEN
         OFRAC=0.0D0
         X1=0
         X2=0
         RETURN
      ELSE
         DO I=1,NP
            IF(LOGO(I)) THEN
               FIRST=I
               GO TO 77
            ENDIF
         ENDDO
  77     CONTINUE

         DO I=NP,1,-1
            IF(LOGO(I)) THEN
               LAST=I
               GO TO 88
            ENDIF
         ENDDO
  88     CONTINUE

         OFRAC=DBLE(NHIT)/DBLE(NHIT+NMISS)
      ENDIF

      IF(FIRST.GT.1) THEN
         X1=XVAL(FIRST-1)
      ELSE
         X1=XVAL(1)
      ENDIF

      IF(LAST.LT.NP) THEN
         X2=XVAL(LAST+1)
      ELSE
         X2=XVAL(LAST)
      ENDIF

      RETURN
      END


      REAL*4 FUNCTION GINTER(X,Y,DATA,NX,NY,LUT,NLUT,SPACE,NBOX,MISVAL)
C
C General interpolation using a lookup table
C
C The function returns the interpolated value at position X,Y
C in the array DATA.
C
C The array LUT is a one-dimensional, over-sampled, look up
C table of weights to be used for interpolating.
C
C Experimental version, Richard Hook, February 2004
C
      IMPLICIT NONE

C Supplied and returned values
      INTEGER NX,NY,NLUT,NBOX
      REAL X,Y,DATA(NX,NY)
      REAL LUT(NLUT),SPACE,MISVAL

C Local variables
      INTEGER I,J,IXS,IYS,IXE,IYE,XOFF,YOFF
      REAL LUTY,SUM

C First check for being close to the edge and if so return the missing
C value
      IXS=NINT(X)-NBOX
      IXE=NINT(X)+NBOX
      IYS=NINT(Y)-NBOX
      IYE=NINT(Y)+NBOX

      IF(IXS.LT.1 .OR.
     :   IXE.GT.NX .OR.
     :   IYS.LT.1 .OR.
     :   IYE.GT.NY) THEN
         GINTER=MISVAL
         RETURN
      ENDIF

C Loop over the box, which is assumed to be scaled appropriately
      SUM=0.0
      DO J=IYS,IYE
         YOFF=1+NINT(ABS((Y-REAL(J))/SPACE))
         LUTY=LUT(YOFF)
         DO I=IXS,IXE
            XOFF=1+NINT(ABS((X-REAL(I))/SPACE))
            SUM=SUM+DATA(I,J)*LUT(XOFF)*LUTY
         ENDDO
      ENDDO

      GINTER=SUM
      RETURN
    
      END 

      
      SUBROUTINE SET1R(A, ARR, NPIX)
C  This routine replaces the SPP routine 'amovkr', a routine which 
C  replaces the values of the 1-D array ARR with the value given 
C  as VAL, such that ARR(I) = A.  
C   NPIX is the number of elements in the array ARR
C
C  Warren Hack, 17 June 2004
C
      IMPLICIT NONE

C Supplied:
      INTEGER NPIX
      REAL A, ARR(NPIX)
   
C Local variables:
      INTEGER I
      
      DO I = 1, NPIX
         ARR(I) = A
      ENDDO

      RETURN
      END      

      SUBROUTINE WSUMR (A,B,C,NPIX,W1,W2)
C
C  WSUM - Weighted sum of 2 real vectors.
C        Replaces the call to the SPP routine 'awsur', where 
C           awsur(a,b,c,npix,k1,k2) : c(i) = k1*a(i) + k2*b(i)
C
C  Warren Hack, 15 June 2004
C
      IMPLICIT NONE

C Supplied:
      INTEGER NPIX
      REAL A(NPIX), B(NPIX), c(NPIX)
      REAL W1, W2
   
C Local variables:
      INTEGER I
      
      DO I = 1, NPIX
         C(I) = A(I) * W1 + B(I) * W2
      ENDDO

      RETURN
      END
