      SUBROUTINE STEEP
C
C STEEP - evaluate the absolute maximum gradient around a pixel
C         by comparing with the four neighboring pixels.
C
C  This is a reimplementation of the "deriv" task in STSDAS/dither.
C
C   Richard Hook, ST-ECF/ESO, October 2002
C
      IMPLICIT NONE

      CHARACTER*80 INFILE,OUTFILE
      INTEGER ID,IDO,ISTAT,DATTYP,DIMS(7),NDIMS,NX,NY,LATEST

      INTEGER MAXLIN
      PARAMETER (MAXLIN=100000)
      REAL BUFF1(MAXLIN)
      REAL BUFF2(MAXLIN)
      REAL BUFF3(MAXLIN)
      REAL DER(MAXLIN)

      INTEGER I,J

C Get the names of the input and output
      CALL UCLGST('data',INFILE,ISTAT)
      CALL UCLGST('deriv',OUTFILE,ISTAT)
     
C Try to open them
      CALL UIMOPN(INFILE,2,ID,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Unable to open input image file',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

C Get the size and shape
      CALL UIMGID(ID,DATTYP,NDIMS,DIMS,ISTAT)
      IF(NDIMS.NE.2) THEN
         CALL UMSPUT('! Unable to open input image file',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

      NX=DIMS(1)
      NY=DIMS(2)

C Create the output
      CALL UIMCRE(OUTFILE,6,2,DIMS,IDO,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Unable to open input image file',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

C Initialise buffers
      CALL ZAPLIN(BUFF1,NX)
      CALL ZAPLIN(BUFF2,NX)
      CALL ZAPLIN(BUFF3,NX)
      CALL ZAPLIN(DER,NX)

C Write out the first line
      CALL UIPL2R(ID,1,DER,ISTAT)

C Read the first lines
      CALL UIGL2R(ID,1,BUFF1,ISTAT)
      CALL UIGL2R(ID,2,BUFF2,ISTAT)

      LATEST=3

C Loop through the image line by line
      DO J=3,NY-1

C Read a new line
         IF(LATEST.EQ.3) THEN
            CALL UIGL2R(ID,J,BUFF3,ISTAT)

            DO I=2,NX-1
               DER(I)=MAX(ABS(BUFF1(I)-BUFF2(I)),
     :                    ABS(BUFF3(I)-BUFF2(I)),
     :                    ABS(BUFF2(I-1)-BUFF2(I)),
     :                    ABS(BUFF2(I)-BUFF2(I+1)))
            ENDDO
            LATEST=1

         ELSE IF(LATEST.EQ.1) THEN
            CALL UIGL2R(ID,J,BUFF1,ISTAT)

            DO I=2,NX-1
               DER(I)=MAX(ABS(BUFF1(I)-BUFF3(I)),
     :                    ABS(BUFF3(I)-BUFF2(I)),
     :                    ABS(BUFF3(I-1)-BUFF3(I)),
     :                    ABS(BUFF3(I)-BUFF3(I+1)))
            ENDDO
            LATEST=2

         ELSE IF(LATEST.EQ.2) THEN
            CALL UIGL2R(ID,J,BUFF2,ISTAT)

            DO I=2,NX-1
               DER(I)=MAX(ABS(BUFF1(I)-BUFF2(I)),
     :                    ABS(BUFF3(I)-BUFF1(I)),
     :                    ABS(BUFF1(I-1)-BUFF1(I)),
     :                    ABS(BUFF1(I)-BUFF1(I+1)))
            ENDDO
            LATEST=3
         ENDIF

C Write out the buffer
         CALL UIPL2R(IDO,J-1,DER,ISTAT)
      ENDDO

C Write a final empty line
      CALL ZAPLIN(DER,NX)
      CALL UIPL2R(IDO,NY,DER,ISTAT)

C Close the images
      CALL UIMCLO(ID,ISTAT)
      CALL UIMCLO(IDO,ISTAT)

 99   CONTINUE
      RETURN
      END

      SUBROUTINE ZAPLIN(A,N)
C
C Set a one-dimensional array to 0.0
C
      IMPLICIT NONE
     
      INTEGER I,N 
      REAL A(N)

      DO I=1,N
         A(I)=0.0
      ENDDO

      END
