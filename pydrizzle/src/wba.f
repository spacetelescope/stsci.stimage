      SUBROUTINE WBA
C++
C
C WBA.F Version 0.1
C
C This code implements a simple "weighted block averaging"
C method. It is intended to be used on outputs from the DRIZZLE task.
C task.
C
C There are two inputs, a data array and a "weight" array.
C The latter records how many images contributed to each
C point in the data, ie, a weighting. When the input is block
C averaged this weighting is taken into account and written to
C an output count image.
C
C It is based on an idea of Andy Fruchter.
C
C This code uses the F77/VOS for IRAF i/o.
C
C History:
C
C First try, Richard Hook, ST-ECF, November 1995 
C
C Tidied up a little (V0.1) and changed parameter names, 
C    Richard Hook, ST-ECF/STScI, March 2004
C
C--
      IMPLICIT NONE

C Iraf global storage (real in this program)
      REAL MEMR(1) !I
      COMMON /MEM/MEMR         !I

C Local variables
      INTEGER ISTAT
      INTEGER NDIMS,DNX,DNY,ONX,ONY,DDIMS(7),ODIMS(7),DATTYP
      INTEGER IDD,IDC,IDND,IDNC
      INTEGER PDATA,PNDAT,PNCOU,PCOU
      INTEGER BX,BY
      CHARACTER*80 DATA,COU,NEWDAT,NEWCOU
      LOGICAL VERBOSE
      CHARACTER*40 VERS

C-- Start of executable code

      VERBOSE=.TRUE.
      VERS='WBA Version 0.1 (March 4th 2004)'

C Write out the version
      CALL UMSPUT('+ '//VERS,1,0,ISTAT)

C Get the name of the input data image
      CALL UCLGST('data',DATA,ISTAT)
      CALL UIMOPN(DATA,1,IDD,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Unable to open data image',
     :                    1,0,ISTAT)
         GO TO 99
      ELSE
         IF(VERBOSE) CALL UMSPUT('-Opening data file: '//DATA,
     :                   1,0,ISTAT)
      ENDIF

C Get the array size and shape
      CALL UIMGID(IDD,DATTYP,NDIMS,DDIMS,ISTAT)
      IF(NDIMS.NE.2) THEN
         CALL UMSPUT('! Input image is not two dimensional',
     :                    1,0,ISTAT)
         GO TO 99
      ELSE
         DNX=DDIMS(1)
         DNY=DDIMS(2)
      ENDIF

C We must also have a weight image
      CALL UCLGST('weight',COU,ISTAT)
      IF(COU.EQ.' ') THEN
         CALL UMSPUT('! There must be a weight image as well',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

      CALL UIMOPN(COU,1,IDC,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Unable to open old weight image',
     :                    1,0,ISTAT)
         GO TO 99
      ELSE
         IF(VERBOSE) CALL UMSPUT(
     :      '-Opening old weight image: '//COU,
     :                   1,0,ISTAT)
      ENDIF

C Get the array size and shape of counts image frame
      CALL UIMGID(IDC,DATTYP,NDIMS,DDIMS,ISTAT)
      IF(NDIMS.NE.2) THEN
       CALL UMSPUT('! Input weight image is not two dimensional',
     :                    1,0,ISTAT)
         GO TO 99
      ELSE IF(DDIMS(1).NE.DNX .OR. DDIMS(2).NE.DNY) THEN
       CALL UMSPUT('! Weight image is not same size as data',
     :                    1,0,ISTAT)
         GO TO 99
      ENDIF

C Get the block averaging factors for X and Y
      CALL UCLGSI('xblock',BX,ISTAT)
      CALL UCLGSI('yblock',BY,ISTAT)

C Warn if there may be edge effects
      IF(MOD(DNX,BX).NE.0 .OR. MOD(DNY,BY).NE.0) THEN
         CALL UMSPUT('! Warning input array size not a multiple '//
     :         'of blocking factor - there may be edge effects',
     :               1,0,ISTAT)
      ENDIF 

C and calculate the size of the outputs
      ONX=DNX/BX
      ODIMS(1)=ONX
      ONY=DNY/BY
      ODIMS(2)=ONY

C Now we need to get the names for the output images - there
C are always both data and "count" images. Note there is no "update"
C of the inputs
      CALL UCLGST('outdata',NEWDAT,ISTAT)
      CALL UIMCRE(NEWDAT,6,2,ODIMS,IDND,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :     '! Unable to create output data image',
     :                 1,0,ISTAT)
         GO TO 99
      ENDIF

      CALL UCLGST('outweig',NEWCOU,ISTAT)
      CALL UIMCRE(NEWCOU,6,2,ODIMS,IDNC,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :     '! Unable to create output weight image',
     :                 1,0,ISTAT)
         GO TO 99
      ENDIF

C Now we allocate space for all the arrays we will need, note that they
C are all real
      CALL UDMGET(DNX*DNY,6,PDATA,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :          '! Unable to allocate memory for data array',
     :                 1,0,ISTAT)
         GO TO 99
      ENDIF

      CALL UDMGET(ONX*ONY,6,PNCOU,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :  '! Unable to allocate memory for output weight array',
     :                 1,0,ISTAT)
         GO TO 99
      ENDIF

      CALL UDMGET(ONX*ONY,6,PNDAT,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     : '! Unable to allocate memory for output data array',
     :                 1,0,ISTAT)
         GO TO 99
      ENDIF

C Now some space for the input arrays, if there are any
      CALL UDMGET(DNX*DNY,6,PCOU,ISTAT)
      IF(ISTAT.NE.0) THEN
      CALL UMSPUT(
     :'! Unable to allocate memory for input weight array',
     :   1,0,ISTAT)
         GO TO 99
      ENDIF

C Now we can actually read the data into the arrays
      CALL UIGS2R(IDD,1,DNX,1,DNY,MEMR(PDATA),ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :      '! Failed to read input data image',
     :                   1,0,ISTAT)
         GO TO 99
      ENDIF
 
      CALL UIGS2R(IDC,1,DNX,1,DNY,MEMR(PCOU),ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :     '! Failed to read input weight image',
     :      1,0,ISTAT)
         GO TO 99
      ENDIF

C Before we start averaging we need to drain out the output arrays
      CALL ZARRAY(MEMR(PNDAT),ONX,ONY)
      CALL ZARRAY(MEMR(PNCOU),ONX,ONY)

C Now do the averaging 
      CALL DWBA(MEMR(PDATA),MEMR(PCOU),
     :              MEMR(PNDAT),MEMR(PNCOU),DNX,DNY,
     :              ONX,ONY,BX,BY)

C Write out the two new images
      IF(VERBOSE)
     :   CALL UMSPUT('-Writing output averaged image: '//
     :               NEWDAT,1,0,ISTAT)
         
      CALL UIPS2R(IDND,1,ONX,1,ONY,MEMR(PNDAT),ISTAT)
      CALL UIMCLO(IDND,ISTAT)

      IF(VERBOSE)
     :   CALL UMSPUT('-Writing output weight image: '//
     :               NEWCOU,1,0,ISTAT)
         
      CALL UIPS2R(IDNC,1,ONX,1,ONY,MEMR(PNCOU),ISTAT)
      CALL UIMCLO(IDNC,ISTAT)
 
C Close down everything and exit

 99   CONTINUE
      END

      SUBROUTINE ZARRAY(A,NX,NY)
C
C Zero a 2d array
C
      IMPLICIT NONE

      INTEGER NX,NY
      REAL A(NX,NY)

      INTEGER I,J

      DO J=1,NY
         DO I=1,NX
            A(I,J)=0.0
         ENDDO
      ENDDO

      RETURN
      END

      SUBROUTINE DWBA(DATA,COU,NDAT,NCOU,DNX,DNY,ONX,ONY,BX,BY)
C
C Block averaging with weighting
C
      IMPLICIT NONE

      INTEGER DNX,DNY,ONX,ONY,BX,BY
      INTEGER I,J,IB,JB
      REAL DATA(DNX,DNY),COU(DNX,DNY),NDAT(ONX,ONY),NCOU(ONX,ONY)
      REAL NC,ND,CC

      DO J=1,ONY
         DO I=1,ONX
            NC=0.0
            ND=0.0
            DO JB=(J-1)*BY+1,J*BY
               DO IB=(I-1)*BX+1,I*BX
                  CC=COU(IB,JB)
                  NC=NC+CC
                  ND=ND+CC*DATA(IB,JB)
               ENDDO
            ENDDO

            IF(NC.EQ.0) THEN
               NCOU(I,J)=0.0
               NDAT(I,J)=0.0
            ELSE
               NDAT(I,J)=ND/NC
               NCOU(I,J)=NC
            ENDIF
         ENDDO
      ENDDO

      RETURN
      END
