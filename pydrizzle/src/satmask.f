      SUBROUTINE SATMAK
C
C SATMASK - make a satellite mask from parameters of a line
C
C The output is an integer*2 image with weights (1 for good and 0 for bad).
C
C Richard Hook, for the GOODS project, ST-ECF, STScI, October 2002
C
      IMPLICIT NONE

      DOUBLE PRECISION X1,Y1,X2,Y2,M,C,WIDTH,HWID,F,X,Y,D
      INTEGER I,J,ISTAT,DIMS(7),ID,NX,NY
      INTEGER MAXNX
      PARAMETER (MAXNX=100000)
      INTEGER BUFFER(MAXNX)
      CHARACTER*80 MASK
      LOGICAL VERTICAL

C Get the name and dimensions of the output
      CALL UCLGST('mask',MASK,ISTAT)
      CALL UCLGSI('nx',NX,ISTAT)
      CALL UCLGSI('ny',NY,ISTAT)

C Get the line parameters - positions of two points
      CALL UCLGSD('x1',X1,ISTAT)
      CALL UCLGSD('y1',Y1,ISTAT)
      CALL UCLGSD('x2',X2,ISTAT)
      CALL UCLGSD('y2',Y2,ISTAT)

C and the width of the band
      CALL UCLGSD('width',WIDTH,ISTAT)
      HWID=WIDTH/2.0

C Check for special case of vertical line
      IF(X1.EQ.X2) THEN
         VERTICAL=.TRUE.

      ELSE

C Work out equation of line
         M=(Y2-Y1)/(X2-X1)
         C=Y1-M*X1
         F=1.0/SQRT(1.0D0+M*M)
         VERTICAL=.FALSE.
      ENDIF

C Check for meaningless case
      IF(X1.EQ.Y1 .AND. X2.EQ.Y2) THEN
         CALL UMSPUT('! Points are coincident',1,0,ISTAT)
         GO TO 99
      ENDIF

      DIMS(1)=NX
      DIMS(2)=NY

C Create the image
      CALL UIMCRE(MASK,4,2,DIMS,ID,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Unable to create mask image',1,0,ISTAT)
         GO TO 99
      ENDIF

C Write out the image, line by line
      DO J=1,NY
         Y=DBLE(J)

C File buffer with 1 or 0
         DO I=1,NX
            X=DBLE(I)

            IF(VERTICAL) THEN
               D=ABS(X-X1)
            ELSE
               D=ABS(F*(M*X+C-Y))
            ENDIF

            IF(D.LE.HWID) THEN
               BUFFER(I)=0
            ELSE
               BUFFER(I)=1
            ENDIF
         ENDDO

C Write line of buffer
         CALL UIPL2I(ID,J,BUFFER,ISTAT)
      ENDDO

C Close
      CALL UIMCLO(ID,ISTAT)

 99   CONTINUE

      RETURN
      END
