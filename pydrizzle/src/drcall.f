C DRCALL - emulation routines for F77/VOS routines found in DRUTIL.F
C
C These routines are needed to allow the callable version of drutil to 
C link without needing F77/VOS routines. 
C
C Richard Hook, ST-ECF/STScI, September 2002

      SUBROUTINE UMSPUT(LINE,D1,D2,ISTAT)
C
C Just write a line of text to the screen. The second and
C third parameters are ignored.
C
      IMPLICIT NONE

      LOGICAL VERBOSE
      COMMON /VERBOSE/VERBOSE

      CHARACTER*(*) LINE
      INTEGER D1,D2,ISTAT

C Always write an error message (starting !), except if
C it is a warning
      IF(LINE(1:1).EQ.'!' .AND. LINE(3:5).NE.'War') THEN
         WRITE(*,'(1X,A,/)',IOSTAT=ISTAT) LINE
      ELSE
         IF(VERBOSE) WRITE(*,'(1X,A,/)',IOSTAT=ISTAT) LINE
      ENDIF

      RETURN
      END

      SUBROUTINE UFGLIN(LUN,LINE,ISTAT)
C
C Read a line of text from an already opened text file.
C
      IMPLICIT NONE
      INTEGER LUN
      CHARACTER*(*) LINE
      INTEGER ISTAT

      READ(LUN,'(A)',IOSTAT=ISTAT) LINE

      RETURN
      END

      SUBROUTINE UFOPEN(FILE,FLAG,LUN,ISTAT)
C
C Open a text file. The FLAG parameter controls the iomode.
C Only 1=readonly and 2=read-write, 4=append and 5=new are
C supported.
C
      CHARACTER*(*) FILE
      INTEGER FLAG,LUN,ISTAT

      IF(FLAG.EQ.1) THEN
         OPEN(23,FILE=FILE,STATUS='OLD',IOSTAT=ISTAT)
      ELSE IF(FLAG.EQ.2) THEN
         OPEN(23,FILE=FILE,STATUS='OLD',IOSTAT=ISTAT)
CCC      ELSE IF(FLAG.EQ.4) THEN
CCC         OPEN(23,FILE=FILE,STATUS='OLD',ACCESS='APPEND',IOSTAT=ISTAT)
      ELSE IF(FLAG.EQ.5) THEN
         OPEN(23,FILE=FILE,STATUS='NEW',IOSTAT=ISTAT)
      ELSE
         CALL UMSPUT('! Invalid file iomode specified',1,0,ISTAT)
      ENDIF

      LUN=23

      RETURN
      END

      SUBROUTINE UFCLOS(LUN,ISTAT)
C
C Close an open file on logical unit LUN
C
      IMPLICIT NONE
    
      INTEGER LUN,ISTAT

      CLOSE(LUN)

      RETURN
      END

      SUBROUTINE GETGEO(COEFFS,IDD,LAM,
     :                  COTY,COMAX,CONUM,XCO,YCO,CLEN,ISTAT)
C
C Get the geometrical distortion information, either from
C a file or the header.
C
C This new routine, October 2000
C Modified in drizzle 2.1 for more flexible coefficients
C
      IMPLICIT NONE

      INTEGER IDD
      INTEGER COTY,COMAX,CONUM
      REAL LAM,XCO(COMAX),YCO(COMAX)
      CHARACTER*80 COEFFS
      INTEGER CLEN

      INTEGER LUN,ISTAT,I
      LOGICAL NOCO

C Interpret the coefficients file name and act appropriately
      NOCO=.FALSE.
      IF(COEFFS.EQ.' ') NOCO=.TRUE.

C Now check for the special string "header"
      IF(COEFFS.EQ.'header') THEN
         CALL UMSPUT(
     : '! This version does not support the header option',
     :   1,0,ISTAT)
         ISTAT=1
         RETURN
      ENDIF

C Set default values
      IF(NOCO) THEN
         DO I=1,COMAX
            XCO(I)=0.0
            YCO(I)=0.0
         ENDDO
         CONUM=1
         COTY=0
      ELSE

C Open the coefficients file
         CALL UFOPEN(COEFFS,1,LUN,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Unable to open coefficients file',1,0,
     :                  ISTAT)
            ISTAT=1
            RETURN
         ELSE
            CALL UMSPUT('-Opening coefficients file: '//COEFFS(1:CLEN),
     :                      1,0,ISTAT)
         ENDIF

C Read in the coefficients from the file
         CALL GETCO(LUN,LAM,COTY,COMAX,CONUM,XCO,YCO,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Unable to read coefficients',
     :                       1,0,ISTAT)
            CALL UFCLOS(LUN,ISTAT)
            ISTAT=1
            RETURN
         ENDIF

C Close the coefficients file
         CALL UFCLOS(LUN,ISTAT)
      ENDIF

C Set a good status return
      ISTAT=0
      RETURN
      END
