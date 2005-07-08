C IOUTIL.F
C
C I/O routines for building IRAF interfaces to Drizzle and Blot
C
C These routines should call only themselves and the 
C    STSDAS f77/vos library.
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
C

      SUBROUTINE GETWCS(ID,WCS,CTYPE1,CTYPE2,ISTAT)
C
C Get the WCS header items (8 values) and CTYPE strings
C
C The image is assumed to be available with descriptor ID
C and the status flag is returned as non-zero if a parameter
C is not found
C
      IMPLICIT NONE

      INTEGER ID,ISTAT,IS
      CHARACTER*8 CTYPE1,CTYPE2
      DOUBLE PRECISION WCS(8)

      ISTAT=0

      CALL UHDGST(ID,'CTYPE1',CTYPE1,ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CTYPE1 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGST(ID,'CTYPE2',CTYPE2,ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CTYPE2 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGSD(ID,'CRPIX1',WCS(1),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CRPIX1 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN  

      CALL UHDGSD(ID,'CRVAL1',WCS(2),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CRVAL1 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN  

      CALL UHDGSD(ID,'CRPIX2',WCS(3),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CRPIX2 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN  

      CALL UHDGSD(ID,'CRVAL2',WCS(4),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CRVAL2 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN  

      CALL UHDGSD(ID,'CD1_1',WCS(5),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CD1_1 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN  

      CALL UHDGSD(ID,'CD2_1',WCS(6),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CD2_1 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN  

      CALL UHDGSD(ID,'CD1_2',WCS(7),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CD1_2 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN  

      CALL UHDGSD(ID,'CD2_2',WCS(8),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CD2_2 from header',1,0,IS)

      RETURN
      END

      SUBROUTINE GTPWCS(ID,PREFIX,WCS,CTYPE1,CTYPE2,ISTAT)
C
C Get the WCS header items (8 values) and CTYPE strings
C
C This version allows a prefix letter to be included
C
C The image is assumed to be available with descriptor ID
C and the status flag is returned as non-zero if a parameter
C is not found
C
      IMPLICIT NONE

      INTEGER ID,ISTAT,IS
      CHARACTER*8 CTYPE1,CTYPE2
      CHARACTER*1 PREFIX
      DOUBLE PRECISION WCS(8)

      ISTAT=0

C Note that the first two - the CTYPEs don't have prefixes
      CALL UHDGST(ID,'CTYPE1',CTYPE1,ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read CTYPE1 from header',
     :  1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGST(ID,'CTYPE2',CTYPE2,ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read CTYPE2 from header',
     :  1,0,IS)
      IF(ISTAT.NE.0) RETURN

C Now the eight numbers which do...
      CALL UHDGSD(ID,PREFIX//'CRPIX1',WCS(1),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read '//PREFIX//'CRPIX1 from header',
     :  1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGSD(ID,PREFIX//'CRVAL1',WCS(2),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read '//PREFIX//'CRVAL1 from header',
     :  1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGSD(ID,PREFIX//'CRPIX2',WCS(3),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read '//PREFIX//'CRPIX2 from header',
     :  1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGSD(ID,PREFIX//'CRVAL2',WCS(4),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read '//PREFIX//'CRVAL2 from header',
     :  1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGSD(ID,PREFIX//'CD1_1',WCS(5),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read '//PREFIX//'CD1_1 from header',
     :  1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGSD(ID,PREFIX//'CD2_1',WCS(6),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read '//PREFIX//'CD2_1 from header',
     :  1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGSD(ID,PREFIX//'CD1_2',WCS(7),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read '//PREFIX//'CD1_2 from header',
     :  1,0,IS)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGSD(ID,PREFIX//'CD2_2',WCS(8),ISTAT)
      IF(ISTAT.NE.0) CALL UMSPUT(
     : '! Warning, unable to read '//PREFIX//'CD2_2 from header',
     :  1,0,IS)

      RETURN
      END

      SUBROUTINE UWCS(ID,WCS,CTYPE1,CTYPE2)
C
C Update the WCS header items using the 8 values
C supplied in WCS and the values of CTYPE1 and CTYPE2.
C
C The image is assumed to be available with descriptor ID
C and to already have all the parameters
C
C Modified to get rid of CDELT and CROTA keywords if present
C and also the PC matrix to avoid confusion. October 2000
C
      IMPLICIT NONE

      INTEGER ID,ISTAT
      CHARACTER*8 CTYPE1,CTYPE2
      DOUBLE PRECISION WCS(8)

      CALL UHDAST(ID,'CTYPE1',CTYPE1,' ',0,ISTAT)
      CALL UHDAST(ID,'CTYPE2',CTYPE2,' ',0,ISTAT)
      CALL UHDASD(ID,'CRPIX1',WCS(1),' ',0,ISTAT)
      CALL UHDASD(ID,'CRVAL1',WCS(2),' ',0,ISTAT)
      CALL UHDASD(ID,'CRPIX2',WCS(3),' ',0,ISTAT)
      CALL UHDASD(ID,'CRVAL2',WCS(4),' ',0,ISTAT)
      CALL UHDASD(ID,'CD1_1',WCS(5),' ',0,ISTAT)
      CALL UHDASD(ID,'CD2_1',WCS(6),' ',0,ISTAT)
      CALL UHDASD(ID,'CD1_2',WCS(7),' ',0,ISTAT)
      CALL UHDASD(ID,'CD2_2',WCS(8),' ',0,ISTAT)

C Delete CDELT and CROTA keywords
C and the PC matrix
      CALL UHDDSP(ID,'CDELT1',ISTAT)
      CALL UHDDSP(ID,'CDELT2',ISTAT)
      CALL UHDDSP(ID,'CROTA1',ISTAT)
      CALL UHDDSP(ID,'CROTA2',ISTAT)
      CALL UHDDSP(ID,'PC001001',ISTAT)
      CALL UHDDSP(ID,'PC002001',ISTAT)
      CALL UHDDSP(ID,'PC001002',ISTAT)
      CALL UHDDSP(ID,'PC002002',ISTAT)

      RETURN
      END

      SUBROUTINE GTCOEF(ID,COEFFS,LAMBDA,ISTAT)
C
C This routine is invoked when it has been requested that
C the coefficients file name for geometric distortion, and
C the associated wavelength is to be extracted from the
C header. 
C
C It is currently handles WFPC2, STIS and NICMOS and assumes
C coefficients in a directory drizzle$coeffs.
C
C Extended for STIS and NICMOS, January 1999
C
      IMPLICIT NONE

      INTEGER ID,ISTAT,IDET
      DOUBLE PRECISION LAMBDA,PLAM
      CHARACTER*(*) COEFFS
      CHARACTER*8 INST,DETECT
      CHARACTER*80 CHARS

C First try to get the instrument name
      CALL UHDGST(ID,'INSTRUME',INST,ISTAT)
      IF(ISTAT.NE.0) RETURN

C Check instrument can be handled
      IF(INST.EQ.'WFPC2') THEN

C Get the other keywords we need
         CALL UHDGSI(ID,'DETECTOR',IDET,ISTAT)
         IF(ISTAT.NE.0) RETURN

         IF(IDET.LT.1 .OR. IDET.GT.4) THEN
            CALL UMSPUT('! Invalid detector number in header',
     :               1,0,ISTAT)
            ISTAT=1
            RETURN
         ENDIF

         CALL UHDGSD(ID,'PHOTPLAM',PLAM,ISTAT)
         IF(ISTAT.NE.0) RETURN

C We have to convert the wavelength from Angstroms to nm
         LAMBDA=PLAM/10.0D0
         
C Now we build the full "Trauger style" name for the coefficients
C file
         IF(IDET.EQ.1) THEN
            COEFFS='drizzle$coeffs/pc1-trauger'
         ELSE IF(IDET.EQ.2) THEN
            COEFFS='drizzle$coeffs/wf2-trauger'
         ELSE IF(IDET.EQ.3) THEN
            COEFFS='drizzle$coeffs/wf3-trauger'
         ELSE IF(IDET.EQ.4) THEN
            COEFFS='drizzle$coeffs/wf4-trauger'
         ENDIF

         WRITE(CHARS,
     : '(''-Instrument: '',A8,'' Chip: '''//
     : ',I1,'' Wavelength: '',F7.2,'' (nm)'')')
     :   INST,IDET,LAMBDA
         CALL UMSPUT(CHARS,1,0,ISTAT)
      ELSE IF(INST.EQ.'NICMOS') THEN

C Get the other keywords we need
         CALL UHDGSI(ID,'CAMERA',IDET,ISTAT)
         IF(ISTAT.NE.0) RETURN

         IF(IDET.LT.1 .OR. IDET.GT.3) THEN
            CALL UMSPUT('! Invalid camera number in header',
     :               1,0,ISTAT)
            ISTAT=1
            RETURN
         ELSE
            WRITE(COEFFS,'(''drizzle$coeffs/nic-'',I1)') IDET
         ENDIF

         WRITE(CHARS,
     : '(''-Instrument: NICMOS  Camera: '',i1)') IDET
         CALL UMSPUT(CHARS,1,0,ISTAT)
      ELSE IF(INST.EQ.'STIS') THEN

C Get the other keywords we need
         CALL UHDGST(ID,'DETECTOR',DETECT,ISTAT)
         IF(DETECT.EQ.'CCD') THEN
            COEFFS='drizzle$coeffs/stis-ccd'
         ELSE IF(DETECT.EQ.'NUV-MAMA') THEN
            COEFFS='drizzle$coeffs/stis-nuv-mama'
         ELSE IF(DETECT.EQ.'FUV-MAMA') THEN
            COEFFS='drizzle$coeffs/stis-fuv-mama'
         ELSE
            CALL UMSPUT('! Invalid STIS detector name in header',
     :               1,0,ISTAT)
            ISTAT=1
            RETURN
         ENDIF

         WRITE(CHARS,
     : '(''-Instrument: STIS    Detector: '',A8)') DETECT
         CALL UMSPUT(CHARS,1,0,ISTAT)
      ELSE
         CALL UMSPUT(
     :  '! We can currently only handle WFPC2, NICMOS or STIS headers',
     :               1,0,ISTAT)
         ISTAT=1
         RETURN
      ENDIF

      RETURN
      END

      SUBROUTINE GOLDH(ID,USEWCS,IC,LASTPF,LASTSC,
     :                 LASTUN,LASTOX,LASTOY,ISTAT)
C
C This routine finds out details of the last image
C to have been drizzled onto the image with identifier
C ID and returns then for comparison with the new
C ones.
C
C This version was moved from the source code of V1.2 of
C drizzle.
C
C Small modifications to fit in with changed header scheme.
C March 2001
C
      IMPLICIT NONE

      INTEGER IC,ISTAT,ID
      LOGICAL USEWCS
      CHARACTER*8 LASTUN
      CHARACTER*4 STEM
      CHARACTER*80 BUFFER
      DOUBLE PRECISION LASTPF,LASTSC,LASTOX,LASTOY

C Before doing anything check for old-style headers and
C warn the user
      CALL UHDGST(ID,'D01VER',BUFFER,ISTAT)
      IF(ISTAT.EQ.0) THEN
         CALL UMSPUT(
     : '! Warning - there is an old-style Drizzle header present,',
     : 1,0,ISTAT)
         CALL UMSPUT(
     : '! please consult the header for full processing history',
     : 1,0,ISTAT)
      ENDIF

C First find out if there are old values
C This is very ugly
      STEM='D000'

      DO IC=1,999
         WRITE(STEM(2:4),'(I3)') IC
         IF(STEM(2:2).EQ.' ') STEM(2:2)='0'
         IF(STEM(3:3).EQ.' ') STEM(3:3)='0'
         CALL UHDGST(ID,STEM//'VER',BUFFER,ISTAT)

C Check for a status code indicating that this header
C item doesn't exist
         IF(ISTAT.EQ.40) GO TO 111
      ENDDO

111   CONTINUE
      IC=IC-1

      WRITE(STEM(2:4),'(I3)') IC
      IF(STEM(2:2).EQ.' ') STEM(2:2)='0'
      IF(STEM(3:3).EQ.' ') STEM(3:3)='0'

      CALL UHDGSD(ID,STEM//'PIXF',LASTPF,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UHDGST(ID,STEM//'OUUN',LASTUN,ISTAT)
      IF(ISTAT.NE.0) RETURN

      IF(.NOT.USEWCS) THEN
         CALL UHDGSD(ID,STEM//'SCAL',LASTSC,ISTAT)
         IF(ISTAT.NE.0) RETURN

         CALL UHDGSD(ID,STEM//'OUXC',LASTOX,ISTAT)
         IF(ISTAT.NE.0) RETURN

         CALL UHDGSD(ID,STEM//'OUYC',LASTOY,ISTAT)
         IF(ISTAT.NE.0) RETURN
      ENDIF

      ISTAT=0

      RETURN
      END


      SUBROUTINE GETIMR(ID,NX,NY,SUB,XMIN,XMAX,YMIN,YMAX,
     :                  BUFF,P,ISTAT)
C
C Read a subset of an already opened image into an already
C allocated memory space. This version for REAL images.
C
C A one dimensional buffer is used and must be big enough.
C
C October 2000
C
      IMPLICIT NONE

      INTEGER ID,NX,NY,XMIN,XMAX,YMIN,YMAX,ISTAT
      REAL P(XMAX-XMIN+1,YMAX-YMIN+1)
      REAL BUFF(1)
      LOGICAL SUB
      INTEGER I

C If we have a subset then read just the bit of interest
      IF(SUB) THEN
         DO I=YMIN,YMAX
            CALL UIGL2R(ID,I,BUFF,ISTAT)
            IF(ISTAT.NE.0) RETURN

C Copy the part of the line into the subset image
            CALL COPY1D(BUFF(XMIN),
     :                 P(1,I-YMIN+1),
     :                 XMAX-XMIN+1)
         ENDDO
      ELSE

C Read the whole image, line by line
         DO I=1,NY
            CALL UIGL2R(ID,I,P(1,I),ISTAT)
            IF(ISTAT.NE.0) RETURN
         ENDDO
      ENDIF

      ISTAT=0
      RETURN
      END

      SUBROUTINE GETIMI(ID,NX,NY,SUB,XMIN,XMAX,YMIN,YMAX,
     :                  BUFF,P,ISTAT)
C
C Read a subset of an already opened image into an already
C allocated memory space. This version for INTEGER images.
C
C A one dimensional buffer is used and must be big enough.
C
C October 2000
C
      IMPLICIT NONE

      INTEGER ID,NX,NY,XMIN,XMAX,YMIN,YMAX,ISTAT
      INTEGER P(XMAX-XMIN+1,YMAX-YMIN+1)
      INTEGER BUFF(1)
      LOGICAL SUB
      INTEGER I

C If we have a subset then read just the bit of interest
      IF(SUB) THEN
         DO I=YMIN,YMAX
            CALL UIGL2I(ID,I,BUFF,ISTAT)
            IF(ISTAT.NE.0) RETURN

C Copy the part of the line into the subset image
            CALL COPY1I(BUFF(XMIN),
     :                 P(1,I-YMIN+1),
     :                 XMAX-XMIN+1)
         ENDDO
      ELSE

C Read the whole image, line by line
         DO I=1,NY
            CALL UIGL2I(ID,I,P(1,I),ISTAT)
            IF(ISTAT.NE.0) RETURN
         ENDDO
      ENDIF

      ISTAT=0
      RETURN
      END

      SUBROUTINE PUTIMR(ID,NX,NY,SUB,XMIN,XMAX,YMIN,YMAX,
     :                  UPDATE,OUTCPS,FS,BUFF,P,ISTAT)
C
C Write out an output image, possibly a subset.
C
C REAL version - with image scaling if required.
C
C October 2000
C
      IMPLICIT NONE

      INTEGER ID,NX,NY,XMIN,XMAX,YMIN,YMAX,ISTAT
      REAL P(XMAX-XMIN+1,YMAX-YMIN+1)
      REAL BUFF(1),FS
      LOGICAL SUB,UPDATE,OUTCPS
      INTEGER I

C First the case of a subset image
      IF(SUB) THEN

C First we read the ranges of Y for which the drizzle
C has had no effect. This is not needed in the CPS
C case except the first time around.
       IF(.NOT.UPDATE .OR. .NOT.OUTCPS) THEN

        IF(YMIN.GT.1) THEN
         DO I=1,YMIN-1
          IF(UPDATE) THEN
           CALL UIGL2R(ID,I,BUFF,ISTAT)
           IF(FS.NE.1.0) THEN
            CALL MULC(BUFF,NX,1,FS)
           ENDIF
          ELSE
           CALL SETIM(BUFF,NX,1,0.0)
          ENDIF
          CALL UIPL2R(ID,I,BUFF,ISTAT)
         ENDDO
        ENDIF

        IF(YMAX.LT.NY) THEN
         DO I=YMAX+1,NY
          IF(UPDATE) THEN
           CALL UIGL2R(ID,I,BUFF,ISTAT)
           IF(FS.NE.1.0) THEN
            CALL MULC(BUFF,NX,1,FS)
           ENDIF
          ELSE
           CALL SETIM(BUFF,NX,1,0.0)
          ENDIF
          CALL UIPL2R(ID,I,BUFF,ISTAT)
         ENDDO
        ENDIF
       ENDIF

C Now we read in the range of lines which we have already
C processed, scale them and copy in the appropriate section
        DO I=YMIN,YMAX
         IF(UPDATE) THEN
          CALL UIGL2R(ID,I,BUFF,ISTAT)
          IF(FS.NE.1.0) THEN
             CALL MULC(BUFF,NX,1,FS)
          ENDIF
         ELSE
          CALL SETIM(BUFF,NX,1,0.0)
         ENDIF
         CALL COPY1D(P(1,I-YMIN+1),BUFF(XMIN),XMAX-XMIN+1)
         CALL UIPL2R(ID,I,BUFF,ISTAT)
        ENDDO
        IF(ISTAT.NE.0) RETURN

C and now the simple case of the whole image
      ELSE
        DO I=1,NY
         CALL UIPL2R(ID,I,P(1,I),ISTAT)
         IF(ISTAT.NE.0) RETURN
        ENDDO
      ENDIF

      ISTAT=0
      RETURN
      END

      SUBROUTINE PUTIMI(ID,NX,NY,SUB,XMIN,XMAX,YMIN,YMAX,
     :                  UPDATE,OUTCPS,FS,BUFF,P,ISTAT)
C
C Write out an output image, possibly a subset.
C
C INTEGER version.
C
C Values of FS other than 1.0 result in an error exit.
C
C October 2000
C
      IMPLICIT NONE

      INTEGER ID,NX,NY,XMIN,XMAX,YMIN,YMAX,ISTAT
      INTEGER P(XMAX-XMIN+1,YMAX-YMIN+1)
      INTEGER BUFF(1)
      REAL FS
      LOGICAL SUB,UPDATE,OUTCPS
      INTEGER I

C Check the FS value - if not 1.0 then take an error exit
C (this is only here for compatibility with the real version)
      IF(FS.NE.1.0) THEN
         ISTAT=2
         RETURN
      ENDIF

C Handle subsets first
      IF(SUB) THEN

C First we read the ranges of Y for which the drizzle
C has had no effect. This is not needed in the CPS
C case except the first time around.
       IF(.NOT.UPDATE .OR. .NOT.OUTCPS) THEN
        IF(YMIN.GT.1) THEN
          DO I=1,YMIN-1
            IF(UPDATE) THEN
             CALL UIGL2I(ID,I,BUFF,ISTAT)
            ELSE
             CALL SETIMI(BUFF,NX,1,0)
            ENDIF
            CALL UIPL2I(ID,I,BUFF,ISTAT)
          ENDDO
        ENDIF

        IF(YMAX.LT.NY) THEN
          DO I=YMAX+1,NY
            IF(UPDATE) THEN
             CALL UIGL2I(ID,I,BUFF,ISTAT)
            ELSE
             CALL SETIMI(BUFF,NX,1,0)
            ENDIF
            CALL UIPL2I(ID,I,BUFF,ISTAT)
          ENDDO
        ENDIF
       ENDIF

C Now we read in the range of lines which we have already
C processed, scale them and copy in the appropriate section
        DO I=YMIN,YMAX
         IF(UPDATE) THEN
          CALL UIGL2I(ID,I,BUFF,ISTAT)
         ELSE
          CALL SETIMI(BUFF,NX,1,0)
         ENDIF
         CALL COPY1I(P(1,I-YMIN+1),BUFF(XMIN),XMAX-XMIN+1)
         CALL UIPL2I(ID,I,BUFF,ISTAT)
        ENDDO
        IF(ISTAT.NE.0) RETURN

C and now the simple case of the whole image
      ELSE
        DO I=1,NY
         CALL UIPL2I(ID,I,P(1,I),ISTAT)
         IF(ISTAT.NE.0) RETURN
        ENDDO
      ENDIF

      ISTAT=0
      RETURN
      END

      SUBROUTINE PTCOIN(ID,CONTAB,MAXHCN,INTAB,MAXIM,MAXEN,NEN,ISTAT)
C
C Write a context index table back to an open image header
C
C Richard Hook, November 1997
C Modified, 24th Nov 97 to included a "used" logical
C           vector so that only the relevant contexts are saved.
C
C Revised for new context format and other things, Richard Hook
C                                                  March 1999
C
C Added option to write to a file if there are more than a certain
C number of entries. This added the CONTAB,MAXHCN parameters.
C  Richard Hook, July 1999
C
      IMPLICIT NONE

      INTEGER MAXIM,MAXEN,MAXHCN
      INTEGER ID,NEN
      INTEGER INTAB(MAXIM,MAXEN)
      INTEGER I,J,K,ISTAT,KWL
      CHARACTER*8 KEY
      CHARACTER*80 CHARS
      CHARACTER*80 CONTAB

C Table of letters
      CHARACTER*1 LET(26)
      LET(1)='A'
      LET(2)='B'
      LET(3)='C'
      LET(4)='D'
      LET(5)='E'
      LET(6)='F'
      LET(7)='G'
      LET(8)='H'
      LET(9)='I'
      LET(10)='J'
      LET(11)='K'
      LET(12)='L'
      LET(13)='M'
      LET(14)='N'
      LET(15)='O'
      LET(16)='P'
      LET(17)='Q'
      LET(18)='R'
      LET(19)='S'
      LET(20)='T'
      LET(21)='U'
      LET(22)='V'
      LET(23)='W'
      LET(24)='X'
      LET(25)='Y'
      LET(26)='Z'

C First we clear out all traces of an old header
C Initialise header keyword matching for CON...
      CALL UHDOKL(ID,'CON*',.TRUE.,KWL,ISTAT)

C Loop over entries until we get a bad status return
      DO WHILE(.TRUE.)

C Get a keyword
         CALL UHDGNK(KWL,KEY,ISTAT)
         IF(ISTAT.NE.0) GO TO 77
      
C Check it is meaningful - to avoid too many mistakes
         READ(KEY(4:7),'(I4)',IOSTAT=ISTAT)
         IF(ISTAT.EQ.0) CALL UHDDSP(ID,KEY,ISTAT)
      ENDDO
 
 77   CONTINUE

C Close the keyword loop
      CALL UHDCKL(KWL,ISTAT)

C Check whether we have lots of entries, if so write them
C to a file rather than to the header and record the fact in the header
C itself. Added 1/7/99
      IF(NEN.GT.MAXHCN) THEN

C Write the name of the file to the header
	 CALL UHDAST(ID,'CONFILE',CONTAB,
     :        'File containing context text table',0,ISTAT)

C Write the context information to that file
	 CALL PTGLCO(CONTAB,INTAB,MAXIM,MAXEN,NEN,ISTAT)
	 IF(ISTAT.NE.0) THEN
	    CALL UMSPUT('! Failed to write context table file',
     :                  1,0,ISTAT)
	    ISTAT=1
	    RETURN
         ELSE
            CALL UMSPUT('-Writing context table file: '//CONTAB,
     :                  1,0,ISTAT)
         ENDIF
      ELSE
	  
C Loop over the different entries
       DO I=1,NEN

C Check for empty contexts and ignore them
        IF(INTAB(2,I).GT.0) THEN

C Put together a name for the keyword
         WRITE(KEY,'(''CON'',I4)') I
         DO J=4,7
            IF(KEY(J:J).EQ.' ') KEY(J:J)='0'
         ENDDO

C Write the first line
         WRITE(CHARS,'(I3,I10,7I7)') 
     :    (INTAB(J,I),J=1,MIN(9,INTAB(1,I)+2))

C If this is the first item then label it
         IF(I.EQ.1) THEN
            CALL UHDAST(ID,KEY,CHARS,'Image context index',
     :               0,ISTAT)
         ELSE
            CALL UHDAST(ID,KEY,CHARS,' ',
     :            0,ISTAT)
         ENDIF

C Now loop over the additional lines which may be
C required
         IF(INTAB(1,I).GT.7) THEN
            DO J=2,INT((INTAB(1,I)+1)/9)+1
               WRITE(CHARS,'(9I7)') 
     :    (INTAB(K,I),K=(J-1)*9+1,MIN(J*9,INTAB(1,I)+2))

C Delete any old keyword of the same name
               CALL UHDDSP(ID,KEY//LET(J-1),ISTAT)

               CALL UHDAST(ID,KEY//LET(J-1),CHARS,' ',
     :               0,ISTAT)
               IF(ISTAT.NE.0) RETURN
            ENDDO
          ENDIF
        ENDIF
       ENDDO

       CALL UMSPUT('-Writing context table to context image header',
     :             1,0,ISTAT)

      ENDIF

      ISTAT=0
      RETURN
      END

      SUBROUTINE GTCOIN(ID,INTAB,MAXIM,MAXEN,NEN,ISTAT)
C
C Read a context index table from an open image header
C
C Richard Hook, November 1997
C Modified, Richard Hook, March 1999
C
C Modified to use the file option if the CONFILE keyword is
C present - Richard Hook, July 1999
C
      IMPLICIT NONE

      INTEGER MAXIM,MAXEN
      INTEGER ID,NEN
      INTEGER INTAB(MAXIM,MAXEN)
      INTEGER I,J,K,L,N,NR,NVAL,ISTAT,KWL
      CHARACTER*8 KEY
      CHARACTER*80 CHARS,CONTAB

C Table of letters
      CHARACTER*1 LET(26)
      LET(1)='A'
      LET(2)='B'
      LET(3)='C'
      LET(4)='D'
      LET(5)='E'
      LET(6)='F'
      LET(7)='G'
      LET(8)='H'
      LET(9)='I'
      LET(10)='J'
      LET(11)='K'
      LET(12)='L'
      LET(13)='M'
      LET(14)='N'
      LET(15)='O'
      LET(16)='P'
      LET(17)='Q'
      LET(18)='R'
      LET(19)='S'
      LET(20)='T'
      LET(21)='U'
      LET(22)='V'
      LET(23)='W'
      LET(24)='X'
      LET(25)='Y'
      LET(26)='Z'

C Initialise the array
      DO J=1,MAXEN
         INTAB(1,J)=1
         DO I=2,MAXIM
            INTAB(I,J)=0
         ENDDO
      ENDDO

C First check for a CONFILE keyword. If this is found then the
C context table is read from there and not the header
      CALL UHDGST(ID,'CONFILE',CONTAB,ISTAT)
      IF(ISTAT.EQ.0) THEN

	 CALL GTGLCO(CONTAB,INTAB,MAXIM,MAXEN,NEN,ISTAT)
	 IF(ISTAT.NE.0) THEN
	    CALL UMSPUT('! Failed to read context from table file',
     :                  1,0,ISTAT)
	    ISTAT=1
	    RETURN
         ELSE
            CALL UMSPUT('-Read context table file: '//CONTAB,
     :                  1,0,ISTAT)
         ENDIF
      ELSE

C Initialise the total in case of error
        I=0

C Initialise header keyword matching for CON...
        CALL UHDOKL(ID,'CON*',.TRUE.,KWL,ISTAT)
        IF(ISTAT.NE.0) GO TO 99

C Loop over entries until we get a bad status return
        DO WHILE(.TRUE.)

C Get a keyword
         CALL UHDGNK(KWL,KEY,ISTAT)
         IF(ISTAT.NE.0) GO TO 99

C Check keyword is valid
         READ(KEY(4:8),*,IOSTAT=ISTAT) I
         
         IF(ISTAT.EQ.0) THEN

C Read the keyword from the header
         CALL UHDGST(ID,KEY,CHARS,ISTAT)
         IF(ISTAT.NE.0) GO TO 99

C First just extract the first number which is the number of
C items
         READ(CHARS,*) NVAL
         IF(NVAL.LT.8) THEN

C Case of one row only (less than 11 values)
            NR=NVAL+2
            READ(CHARS,*) (INTAB(K,I),K=1,NR)
         ELSE

C Case for more than one row
C First read the first row
            READ(CHARS,*) (INTAB(K,I),K=1,9)

C Then work out how many remain
            NR=NVAL-7

C and loop over them in groups of 9 (if possible)
            L=0
            DO WHILE(NR.GT.0)
               N=MIN(9,NR)
               L=L+1
               CALL UHDGST(ID,KEY//LET(L),CHARS,ISTAT)
               IF(ISTAT.NE.0) GO TO 99
               READ(CHARS,*,IOSTAT=ISTAT) 
     :             (INTAB(K,I),K=L*9+1,L*9+N)
               NR=NR-N
             ENDDO
          ENDIF
         ENDIF
        ENDDO

C If we get here something must be wrong
        ISTAT=1
        RETURN

C This is the normal exit point
99     CONTINUE
  
C Close keyword list
       CALL UHDCKL(KWL,ISTAT)

       NEN=I
      ENDIF

C Check that we have successfully read some values
      IF(NEN.GT.0) THEN
         ISTAT=0
      ELSE
         ISTAT=1
      ENDIF

      RETURN
      END

      SUBROUTINE GTUNID(NAME,ID,UNIQID,ISTAT)
C
C Get a unique image id (an integer) from the header of 
C an open image. This version is for EMMI and looks for
C firstly a keyword EISID and, if that doesn't exist,
C it uses the name of the image and takes the value
C between the last _ and the first . in the filename.
C
      IMPLICIT NONE
      INTEGER I,ISTAT,UNIQID,ID,IDOT,IUND
      CHARACTER*80 NAME

C First try to get the keyword
      CALL UHDGSI(ID,'EISID',UNIQID,ISTAT)
      IF(ISTAT.EQ.0) RETURN

C Then try to find it from the image name
      IUND=0
      IDOT=0
      DO I=1,LEN(NAME)
         IF(NAME(I:I).EQ.'_') IUND=I
         IF(NAME(I:I).EQ.'.') THEN
            IDOT=I
            GO TO 88
         ENDIF
      ENDDO

 88   CONTINUE

      IF(IUND.LT.IDOT-1 .AND. IUND.GT.0) 
     :    READ(NAME(IUND+1:IDOT-1),'(I)',IOSTAT=ISTAT) UNIQID

      RETURN
      END 

      SUBROUTINE UCLGSD(PNAME,VAL,ISTAT)
C
C This is a routine to emulate the F77/VOS routine of the same
C name which seems to be missing. It gets the value as a character
C string from the CL (although the CL itself checks that it is a
C valid number) and converts it to a double precision value via
C a free-format read.
C
      IMPLICIT NONE

      CHARACTER*80 PNAME
      CHARACTER*80 CHARS
      DOUBLE PRECISION VAL
      INTEGER ISTAT

      CALL UCLGST(PNAME,CHARS,ISTAT)
      READ(CHARS,*,IOSTAT=ISTAT) VAL

      RETURN
      END

      SUBROUTINE OPCON(NAME,ONX,ONY,ID,ODCO,UNIQID,ISTAT)
C
C Open a context image in bitmap, 3d form, check
C whether there is room for the bit to be set and,
C if not, recreate a bigger one. Return the ID
C of the opened channel.
C
C It is expected that the file exists and isn't being
C created for the first time.
C
C Richard Hook, ST-ECF, January 2001
C
      IMPLICIT NONE

      CHARACTER*80 NAME
      INTEGER ID,UNIQID,ONX,ONY,IDOLD,ISTAT,J,N
      INTEGER IB
      LOGICAL ODCO

C Local variables
      INTEGER NDIMS,DIMS(7),DATTYP,I1,I2,NP
      CHARACTER*80 PNAME

C Buffer array
      INTEGER MEMI(1)
      COMMON /MEM/MEMI

C Assume we fail
      ODCO=.FALSE.

C First try to open what we have
      CALL UIMOPN(NAME,2,ID,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :    '! Unable to open current context image',
     :                 1,0,ISTAT)
         ISTAT=1
         RETURN
      ENDIF

C Look at its size and shape
      CALL UIMGID(ID,DATTYP,NDIMS,DIMS,ISTAT)

C Check it is the right size 
      IF(NDIMS.NE.2 .AND. NDIMS.NE.3) THEN
         CALL UMSPUT(
     :    '! Input image is not 2 or 3 dimensional',
     :                 1,0,ISTAT)
         ISTAT=1
         RETURN
      ENDIF

      IF(NDIMS.EQ.2) DIMS(3)=1

C Close the image
      CALL UIMCLO(ID,ISTAT)

C Calculate the number of planes required
      NP=(UNIQID-1)/32+1

C Check whether we need a new one
      IF(NP.GT.DIMS(3)) THEN

         CALL UMSPUT('! Creating new, more spacious, context file',
     :               1,0,ISTAT)

C Take a copy of the current one
         CALL UIMDEL('ConTemp',ISTAT)
         CALL UIMRNM(NAME,'ConTemp',ISTAT)
         IF(ISTAT.NE.0) RETURN

C Open it again
         CALL UIMOPN('ConTemp',1,IDOLD,ISTAT)
         IF(ISTAT.NE.0) THEN
           CALL UMSPUT('! Failed to re-open copy of context image',
     :                    1,0,ISTAT)
           ISTAT=1
           RETURN
         ENDIF

C Create the new one
         DIMS(3)=NP
         NDIMS=3
         CALL UIMCRE(NAME,4,3,DIMS,ID,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     :        '! Unable to create new output context image',
     :                    1,0,ISTAT)
            ODCO=.FALSE.
            ISTAT=1
            RETURN
         ELSE
            CALL UMSPUT(
     :        '-Created new output context image: '//NAME,
     :                    1,0,ISTAT)
            ODCO=.TRUE.

C Copy the header as well
            CALL UHDCPY(IDOLD,ID,ISTAT)
            IF(ISTAT.NE.0) RETURN
         ENDIF
         
C Allocate a buffer
         CALL UDMGET(DIMS(1),4,IB,ISTAT)

C Copy the old stuff over
         DO N=1,NP-1
            DO J=1,DIMS(2)
               IF(NDIMS.EQ.2) THEN
                  CALL UIGL2I(IDOLD,J,MEMI(IB),ISTAT)
                  IF(ISTAT.NE.0) RETURN
               ELSE
                  CALL UIGL3I(IDOLD,J,N,MEMI(IB),ISTAT)
                  IF(ISTAT.NE.0) RETURN
               ENDIF

               CALL UIPL3I(ID,J,N,MEMI(IB),ISTAT)
               IF(ISTAT.NE.0) RETURN
            ENDDO
         ENDDO

C If this is the first extension (ie, there are now two planes)
C there are a few more things to add
         IF(NP.EQ.2) THEN
            CALL UHDAST(ID,'CTYPE3','CHANNEL',' ',0,ISTAT)
            CALL UHDASD(ID,'CRVAL3',0.0D0,' ',0,ISTAT)
            CALL UHDASD(ID,'CRPIX3',0.0D0,' ',0,ISTAT)
            CALL UHDASD(ID,'CD1_3',0.0D0,' ',0,ISTAT)
            CALL UHDASD(ID,'CD2_3',0.0D0,' ',0,ISTAT)
            CALL UHDASD(ID,'CD3_3',1.0D0,' ',0,ISTAT)
            CALL UHDASD(ID,'CD3_2',0.0D0,' ',0,ISTAT)
            CALL UHDASD(ID,'CD3_1',0.0D0,' ',0,ISTAT)
         ENDIF

C Close the temporary copy
         CALL UIMCLO(IDOLD,ISTAT)
         IF(ISTAT.NE.0) RETURN

C Initialise the new plane
         CALL SET1I(MEMI(IB),DIMS(1),0)

         DO J=1,DIMS(2)
            CALL UIPL3I(ID,J,NP,MEMI(IB),ISTAT)
            IF(ISTAT.NE.0) RETURN
         ENDDO

         CALL UMSPUT('-Written out new context image',1,0,ISTAT)

C Free buffer
         CALL UDMFRE(IB,4,ISTAT)

C Close the new copy
         CALL UIMCLO(ID,ISTAT)
         IF(ISTAT.NE.0) RETURN

C If we have got here all must be well so we can delete the
C old copy
         CALL UIMDEL('ConTemp',ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     : '! Warning, failed to delete copy of old context image',
     :    1,0,ISTAT)
         ENDIF
      ENDIF

      IF(NDIMS.EQ.2) THEN
         PNAME=NAME
      ELSE
         CALL LENSTR(NAME,I1,I2)
         WRITE(PNAME,'(A,''[*,*,'',I2,'']'')') NAME(I1:I2),NP
         IF(PNAME(I2+5:I2+5).EQ.' ') PNAME(I2+5:I2+5)='0'
      ENDIF

      CALL UIMOPN(PNAME,2,ID,ISTAT)      
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     : '! Unable to open context image'//PNAME,
     :                    1,0,ISTAT)
         ODCO=.FALSE.
         ISTAT=1
         RETURN
      ELSE
         CALL UMSPUT('-Opening extant context image: '//PNAME,
     :   1,0,ISTAT)
         ODCO=.TRUE.
      ENDIF

      RETURN
      END

      SUBROUTINE UHEAD(ID,VERS,DATA,WEIGHT,EXPTIM,WTSCL,OUTDAT,
     :                 OUTCOU,CONTIM,COEFFS,XGEOIM,YGEOIM,
     :                 SHFTUN,PFRACT,
     :                 LAM,SCALE,ROT,SHFTFR,ALIGN,KERNEL,EXPKEY,
     :                 FILSTR,INUN,OUTUN,
     :                 SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,
     :                 USEWCS,XSH,YSH,DNX,DNY,ONX,ONY)
C
C Update the header of the output image with all the
C parameters of the current DRIZZLE run.
C
C Added Kernel value and context file name, October 2000
C
C Added exposure time, January 2001
C
C Corrected bug in CONTIM output, March 2001
C
C Modified to be a little more intelligent and not to write
C out irrelevant information, March 2001
C
C Modified to include the align parameter even for wdrizzle
C to allow correct matching of distortions, October 2003
C
      IMPLICIT NONE

      INTEGER ID,DNX,DNY,ONX,ONY,ISTAT,I
      CHARACTER*80 DATA,WEIGHT
      CHARACTER*80 OUTDAT,OUTCOU,CONTIM,COEFFS,SHFTUN
      CHARACTER*80 BUFFER,FILSTR
      CHARACTER*80 XGEOIM,YGEOIM 
      CHARACTER*40 VERS
      CHARACTER*8 EXPKEY,INUN,OUTUN,ALIGN
      CHARACTER*4 STEM
      CHARACTER*8 SHFTFR,KERNEL
      DOUBLE PRECISION PFRACT,LAM,SCALE,ROT
      REAL EXPTIM
      DOUBLE PRECISION DROT,XSH,YSH,XSHI,YSHI,OFF
      REAL WTSCL
      LOGICAL USEWCS

C Secondary geometrical parameters, added in V1.5
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*8 SHFR2
      LOGICAL SECPAR

      STEM='D000'

C First find out if there are old values
C This is very ugly
      DO I=1,999
         WRITE(STEM(2:4),'(I3)') I
         IF(STEM(2:2).EQ.' ') STEM(2:2)='0'
         IF(STEM(3:3).EQ.' ') STEM(3:3)='0'
         CALL UHDGST(ID,STEM//'VER',BUFFER,ISTAT)

C Check for a status code indicating that this header
C item doesn't exist
         IF(ISTAT.EQ.40) GO TO 111
      ENDDO

111   CONTINUE

C First update the number of images on this output
      IF(I.EQ.1) THEN
         CALL UHDASI(ID,'NDRIZIM',I,
     :  'Drizzle, number of images drizzled onto this output',
     :  0,ISTAT)
      ELSE
         CALL UHDPSI(ID,'NDRIZIM',I,ISTAT)
      ENDIF

C Then the version information
      CALL UHDAST(ID,STEM//'VER',VERS,
     :            'Drizzle, task version',0,ISTAT)

C Then the source of the geometric information
      IF(USEWCS) THEN
         CALL UHDAST(ID,STEM//'GEOM','Header WCS',
     :            'Drizzle, source of geometric information',0,ISTAT)
      ELSE
         CALL UHDAST(ID,STEM//'GEOM','User parameters',
     :            'Drizzle, source of geometric information',0,ISTAT)
      ENDIF
      
C Now we continue to add the other items using the same
C "stem"
      CALL UHDAST(ID,STEM//'DATA',DATA(1:64),
     :            'Drizzle, input data image',0,ISTAT)

      CALL UHDASR(ID,STEM//'DEXP',EXPTIM,
     :   'Drizzle, input image exposure time (s)',0,ISTAT)

      CALL UHDAST(ID,STEM//'OUDA',OUTDAT(1:64),
     :            'Drizzle, output data image',0,ISTAT)

      CALL UHDAST(ID,STEM//'OUWE',OUTCOU(1:64),
     :            'Drizzle, output weighting image',0,ISTAT)

      CALL UHDAST(ID,STEM//'OUCO',CONTIM(1:64),
     :            'Drizzle, output context image',0,ISTAT)

      CALL UHDAST(ID,STEM//'MASK',WEIGHT(1:64),
     :            'Drizzle, input weighting image',
     :            0,ISTAT)

      CALL UHDASR(ID,STEM//'WTSC',WTSCL,
     :  'Drizzle, weighting factor for input image',0,ISTAT)

      CALL UHDAST(ID,STEM//'KERN',KERNEL,
     :            'Drizzle, form of weight distribution kernel',0,ISTAT)

      CALL UHDASD(ID,STEM//'PIXF',PFRACT,
     :  'Drizzle, linear size of drop',0,ISTAT)

      CALL UHDAST(ID,STEM//'COEF',COEFFS(1:64),
     :            'Drizzle, coefficients file name ',0,ISTAT)

      CALL UHDAST(ID,STEM//'XGIM',XGEOIM(1:64),
     :            'Drizzle, X distortion image name ',0,ISTAT)

      CALL UHDAST(ID,STEM//'YGIM',YGEOIM(1:64),
     :            'Drizzle, Y distortion image name ',0,ISTAT)

      CALL UHDASD(ID,STEM//'LAM',LAM,
     : 'Drizzle, wavelength applied for transformation (nm)',
     :  0,ISTAT)

C Only put the next entries is we are NOT using WCS
      IF(.NOT.USEWCS) THEN
         CALL UHDASD(ID,STEM//'SCAL',SCALE,
     :    'Drizzle, scale (pixel size) of output image',0,ISTAT)

C Convert the rotation angle back to degrees
         DROT=ROT/3.1415926535897*180.0
         CALL UHDASD(ID,STEM//'ROT',DROT,
     :  'Drizzle, rotation angle, degrees anticlockwise',0,ISTAT)

C Check the SCALE and units
C The units are INPUT pixels on entry to this routine
         IF(SHFTUN.EQ.'output') THEN
            XSHI=XSH/SCALE
            YSHI=YSH/SCALE
         ELSE
            XSHI=XSH
            YSHI=YSH
         ENDIF

         CALL UHDASD(ID,STEM//'XSH',XSHI,
     :     'Drizzle, X shift applied',0,ISTAT)

         CALL UHDASD(ID,STEM//'YSH',YSHI,
     :     'Drizzle, Y shift applied',0,ISTAT)

         CALL UHDAST(ID,STEM//'SFTU',SHFTUN,
     :    'Drizzle, units used for shifts (output or input pixels)',
     :            0,ISTAT)

         CALL UHDAST(ID,STEM//'SFTF',SHFTFR,
     :    'Drizzle, frame in which shifts were applied',
     :            0,ISTAT)
      ENDIF

      CALL UHDAST(ID,STEM//'EXKY',EXPKEY,
     :  'Drizzle, exposure keyword name in input image',0,ISTAT)

      CALL UHDAST(ID,STEM//'INUN',INUN,
     :  'Drizzle, units of input image - counts or cps',0,ISTAT)

      CALL UHDAST(ID,STEM//'OUUN',OUTUN,
     :  'Drizzle, units of output image - counts or cps',0,ISTAT)

      CALL UHDAST(ID,STEM//'FVAL',FILSTR,
     :  'Drizzle, fill value for zero weight output pixels',
     :  0,ISTAT)

      IF(ALIGN.EQ.'corner') THEN
         OFF=0.5D0
      ELSE
         OFF=1.0D0
      ENDIF

      CALL UHDASD(ID,STEM//'INXC',DBLE(DNX/2)+OFF,
     :     'Drizzle, reference center of input image (X)',0,ISTAT)

      CALL UHDASD(ID,STEM//'INYC',DBLE(DNY/2)+OFF,
     :     'Drizzle, reference center of input image (Y)',0,ISTAT)

      CALL UHDASD(ID,STEM//'OUXC',DBLE(ONX/2)+OFF,
     :     'Drizzle, reference center of output image (X)',0,ISTAT)

      CALL UHDASD(ID,STEM//'OUYC',DBLE(ONY/2)+OFF,
     :     'Drizzle, reference center of output image (Y)',0,ISTAT)

C If there are secondary parameters add these too
      IF(SECPAR) THEN
         CALL UHDASB(ID,STEM//'SECP',.TRUE.,
     :  'Drizzle, there are secondary geometric parameters',0,ISTAT)

         CALL UHDASD(ID,STEM//'XSCL',XSCALE,
     :  'Drizzle, Secondary X scale applied',0,ISTAT)

         CALL UHDASD(ID,STEM//'YSCL',YSCALE,
     :  'Drizzle, Secondary Y scale applied',0,ISTAT)

         CALL UHDASD(ID,STEM//'XSH2',XSH2,
     :  'Drizzle, Secondary X shift applied',0,ISTAT)

         CALL UHDASD(ID,STEM//'YSH2',YSH2,
     :  'Drizzle, Secondary Y shift applied',0,ISTAT)

C Convert the rotation angle back to degrees
         DROT=ROT2/3.1415926535897*180.0
         CALL UHDASD(ID,STEM//'ROT2',DROT,
     :  'Drizzle, secondary rotation angle, degrees anticlockwise',
     :   0,ISTAT)
   
         CALL UHDAST(ID,STEM//'SFF2',SHFR2,
     : 'Drizzle, frame in which secondary shifts were applied',
     :            0,ISTAT)

      ELSE
         CALL UHDASB(ID,STEM//'SECP',.FALSE.,
     :  'Drizzle, there are no secondary geometric parameters',0,ISTAT)
      ENDIF

      RETURN
      END

      SUBROUTINE GETPAR(GEOMOD,KERNEL,PFRACT,COEFFS,LAM,XGEOIM,YGEOIM,
     :                  SCALE,ROT,XSH,YSH,SHFTFR,SHFTUN,ALIGN,
     :                  XSCALE,YSCALE,XSH2,YSH2,ROT2,SHFR2,
     :                  EXPKEY,WTSTR,FILSTR,INUN,OUTUN,
     :                  DATA,WEIGHT,OUTDAT,OUTCOU,CONTIM,ONX,ONY,
     :                  OUTSCL,RACEN,DECCEN,XREFP,YREFP,ORIENT)
C
C Read all the parameter values needed by Drizzle. 
C
C In this version they are obtained from the IRAF CL.
C There is no error checking as these routines should
C always succeed.
C
C This separate routine, October 2000
C
C Modified for WCS option  - STScI, November 2000
C
C Modified so that the geometrical parameters are obtained
C from another, general purpose, routine.
C  December 2001
C 
C Added reference pixel, Richard Hook, ST-ECF/STScI, June 2003
C
C Added XGEOIM,YGEOIM, Richard Hook, ST-ECF/STScI, August 2003
C
      IMPLICIT NONE

C The parameters
      CHARACTER*8 KERNEL,GEOMOD,EXPKEY,SHFTFR,INUN,OUTUN,ALIGN
      CHARACTER*80 COEFFS,SHFTUN,WTSTR,FILSTR,XGEOIM,YGEOIM
      CHARACTER*80 DATA,WEIGHT,OUTDAT,OUTCOU,CONTIM
      DOUBLE PRECISION SCALE,ROT,PFRACT,XSH,YSH,LAM
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      DOUBLE PRECISION RACEN,DECCEN,XREFP,YREFP
      DOUBLE PRECISION ORIENT,OUTSCL
      CHARACTER*8 SHFR2
      INTEGER ONX,ONY

C Local variables
      INTEGER ISTAT

C First the mode parameter - this is the WDRIZZLE change
      IF(GEOMOD.NE.'user') CALL UCLGST('geomode',GEOMOD,ISTAT)

C Now all the drizzle related parameters

C Kernel shape and size
      CALL UCLGST('kernel',KERNEL,ISTAT)
      CALL UCLGSD('pixfrac',PFRACT,ISTAT)

C Exposure time, weighting and data scaling related keywords
      CALL UCLGST('expkey',EXPKEY,ISTAT)
      CALL UCLGST('wt_scl',WTSTR,ISTAT)
      CALL UCLGST('fillval',FILSTR,ISTAT)
      CALL UCLGST('in_un',INUN,ISTAT)
      CALL UCLGST('out_un',OUTUN,ISTAT)

C Data array names
      CALL UCLGST('data',DATA,ISTAT)
      CALL UCLGST('in_mask',WEIGHT,ISTAT)
      CALL UCLGST('outdata',OUTDAT,ISTAT)
      CALL UCLGST('outweig',OUTCOU,ISTAT)
      CALL UCLGST('outcont',CONTIM,ISTAT)

C Output array sizes
      CALL UCLGSI('outnx',ONX,ISTAT)
      CALL UCLGSI('outny',ONY,ISTAT)

C Get all the geometrical parameters
      CALL GTGEPA(GEOMOD,COEFFS,LAM,XGEOIM,YGEOIM,
     :             SCALE,ROT,XSH,YSH,SHFTFR,SHFTUN,ALIGN,
     :             XSCALE,YSCALE,XSH2,YSH2,ROT2,SHFR2,
     :             OUTSCL,RACEN,DECCEN,XREFP,YREFP,ORIENT)

      RETURN
      END

      SUBROUTINE GTGEPA(GEOMOD,COEFFS,LAM,XGEOIM,YGEOIM,
     :                  SCALE,ROT,XSH,YSH,SHFTFR,SHFTUN,ALIGN,
     :                  XSCALE,YSCALE,XSH2,YSH2,ROT2,SHFR2,
     :                  OUTSCL,RACEN,DECCEN,XREFP,YREFP,ORIENT)
C
C Get all the Drizzle-related geometric parameters from the CL.
C
C This is a general routine for all the tasks related to drizzle.
C
C The primary and secondary linear transformations are read as
C well as the name of the coefficients file and the wavelength.
C
C The secondary parameters are obtained from the pset dr2gpar.
C
C Note that the value of the logical GEOMOD which is supplied
C is used to decide whether or not the WCS parameters are also
C read.
C
C Richard Hook, ST-ECF, December 2001
C
C Added XGEOIM,YGEOIM for image distortion option.
C  Richard Hook, ST-ECF/STScI, August 2003
C
      IMPLICIT NONE

      DOUBLE PRECISION SCALE,ROT,XSH,YSH,LAM
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      DOUBLE PRECISION RACEN,DECCEN,XREFP,YREFP
      DOUBLE PRECISION ORIENT,OUTSCL
      CHARACTER*80 COEFFS,SHFTUN,XGEOIM,YGEOIM
      CHARACTER*8 SHFR2,SHFTFR,ALIGN,GEOMOD
      INTEGER ISTAT

C The name of the geometric distortion coefficients file
C and wavelength for the Trauger (WFPC2) case
      CALL UCLGST('coeffs',COEFFS,ISTAT)
      CALL UCLGSD('lambda',LAM,ISTAT)

C Get the image geometry file names
      CALL UCLGST('xgeoim',XGEOIM,ISTAT)
      CALL UCLGST('ygeoim',YGEOIM,ISTAT)

C Primary linear transformation parameters
      CALL UCLGSD('scale',SCALE,ISTAT)
      CALL UCLGSD('rot',ROT,ISTAT)
      CALL UCLGSD('xsh',XSH,ISTAT)
      CALL UCLGSD('ysh',YSH,ISTAT)
      CALL UCLGST('shft_fr',SHFTFR,ISTAT)
      CALL UCLGST('shft_un',SHFTUN,ISTAT)
      CALL UCLGST('align',ALIGN,ISTAT)

C Secondary transformation parameters - these come
C from a specific pset and are hence the same for
C all routines
      CALL UCLGSD('dr2gpar.xscale',XSCALE,ISTAT)
      CALL UCLGSD('dr2gpar.yscale',YSCALE,ISTAT)
      CALL UCLGSD('dr2gpar.xsh',XSH2,ISTAT)
      CALL UCLGSD('dr2gpar.ysh',YSH2,ISTAT)
      CALL UCLGSD('dr2gpar.rot',ROT2,ISTAT)
      CALL UCLGST('dr2gpar.shft_fr',SHFR2,ISTAT)

C WCS mode parameters of the output
C - only in the case where GEOMOD is set appropriately
      IF(GEOMOD.EQ.'wcs') THEN
         CALL UCLGSD('outscl',OUTSCL,ISTAT)
         CALL UCLGSD('raref',RACEN,ISTAT)
         CALL UCLGSD('decref',DECCEN,ISTAT)
         CALL UCLGSD('xrefpix',XREFP,ISTAT)
         CALL UCLGSD('yrefpix',YREFP,ISTAT)
         CALL UCLGSD('orient',ORIENT,ISTAT)
      ENDIF

      RETURN
      END

      SUBROUTINE ALLMEM(DNX,DNY,ONX,NSX,NSY,CON,BITCON,
     :   PDATA,PWEI,PNDAT,PNCOU,PNCON,PCDON,PBUFF,PIBUFF,
     :   PXI,PYI,PXO,PYO,PXIB,PYIB,PXOB,PYOB,ISTAT)
C
C Allocate the dynamic memory arrays needed for drizzle.
C
C This routine was added and the corresponding calls removed from
C the main module, October 2000
C
C Added BITCON flag for PCDON array - January 2001
C
      IMPLICIT NONE

      INTEGER DNX,DNY,ONX,NSX,NSY
      INTEGER PDATA,PWEI,PNDAT,PNCOU,PNCON,PCDON,PBUFF,PIBUFF
      INTEGER PXI,PYI,PXO,PYO
      INTEGER PXIB,PYIB,PXOB,PYOB,ISTAT
      LOGICAL CON
      LOGICAL BITCON

C Set all the pointers to zero to start with - if they stay that
C way we can assume the allocation has failed
      PDATA=0
      PWEI=0
      PNDAT=0
      PNCOU=0
      PNCON=0
      PCDON=0
      PBUFF=0
      PIBUFF=0
      PXI=0
      PYI=0
      PXO=0
      PYO=0
      PXIB=0
      PYIB=0
      PXOB=0
      PYOB=0

C Allocate the memory arrays and return the pointers, check for
C error status

C Single line buffers for input data and weight images

      CALL UDMGET(DNX,6,PWEI,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX,6,PDATA,ISTAT)
      IF(ISTAT.NE.0) RETURN

C Note that the next three, as they are subsets of the output
C are of a different size
      CALL UDMGET(NSX*NSY,6,PNDAT,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(NSX*NSY,6,PNCOU,ISTAT)
      IF(ISTAT.NE.0) RETURN

C This one, the context image, is optional
      IF(CON) THEN
         CALL UDMGET(NSX*NSY,4,PNCON,ISTAT)
         IF(ISTAT.NE.0) RETURN
         IF(.NOT.BITCON) THEN
            CALL UDMGET(NSX*NSY,4,PCDON,ISTAT)
            IF(ISTAT.NE.0) RETURN
         ENDIF
      ENDIF

C Scratch buffer space - the larger of the input and output 
      IF(DNX.GT.ONX) THEN
         CALL UDMGET(DNX,6,PBUFF,ISTAT) 
         CALL UDMGET(DNX,4,PIBUFF,ISTAT) 
      ELSE
         CALL UDMGET(ONX,6,PBUFF,ISTAT)
         CALL UDMGET(ONX,4,PIBUFF,ISTAT)
      ENDIF
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX*4,7,PXI,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX*4,7,PYI,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX*4,7,PXO,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX*4,7,PYO,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX,7,PXIB,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX,7,PYIB,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX,7,PXOB,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX,7,PYOB,ISTAT)      
      IF(ISTAT.NE.0) RETURN
      

C If we get here all is well
      ISTAT=0
      RETURN
      END

      SUBROUTINE FREMEM(PDATA,PWEI,PNDAT,PNCOU,PNCON,PCDON,PBUFF,PIBUFF,
     :   PXI,PYI,PXO,PYO,PXIB,PYIB,PXOB,PYOB)
C
C Free all allocated memory arrays. 
C
C Successful allocations will have non-zero pointers.
C
C Added October 2000.
C
      IMPLICIT NONE

      INTEGER PDATA,PWEI,PNDAT,PNCOU,PNCON,PCDON,PBUFF,PIBUFF
      INTEGER PXI,PYI,PXO,PYO
      INTEGER PXIB,PYIB,PXOB,PYOB,ISTAT

      IF(PDATA.NE.0) CALL UDMFRE(PDATA,6,ISTAT)
      IF(PWEI.NE.0) CALL UDMFRE(PWEI,6,ISTAT)
      IF(PNDAT.NE.0) CALL UDMFRE(PNDAT,6,ISTAT)
      IF(PNCOU.NE.0) CALL UDMFRE(PNCOU,6,ISTAT)
      IF(PNCON.NE.0) CALL UDMFRE(PNCON,4,ISTAT)
      IF(PCDON.NE.0) CALL UDMFRE(PCDON,4,ISTAT)
      IF(PIBUFF.NE.0) CALL UDMFRE(PIBUFF,4,ISTAT)
      IF(PXI.NE.0) CALL UDMFRE(PXI,7,ISTAT)
      IF(PYI.NE.0) CALL UDMFRE(PYI,7,ISTAT)
      IF(PXO.NE.0) CALL UDMFRE(PXO,7,ISTAT)
      IF(PYO.NE.0) CALL UDMFRE(PYO,7,ISTAT)
      IF(PXIB.NE.0) CALL UDMFRE(PXIB,7,ISTAT)
      IF(PYIB.NE.0) CALL UDMFRE(PYIB,7,ISTAT)
      IF(PXOB.NE.0) CALL UDMFRE(PXOB,7,ISTAT)
      IF(PYOB.NE.0) CALL UDMFRE(PYOB,7,ISTAT)

      RETURN
      END
      
      
      SUBROUTINE GETGEO(COEFFS,IDD,LAM,
     :                  COTY,COMAX,CONUM,XCO,YCO,ISTAT)
C
C Get the geometrical distortion information, either from
C a file or the header.
C
C This new routine, October 2000
C Modified in drizzle 2.1 for more flexible coefficients
C
C Added distortion image support, Richard Hook, ST-ECF/STScI, August 2003
C Moved distortion image support to GGEOIM, W. Hack, June 2004
C
C
      IMPLICIT NONE

      INTEGER IDD
      INTEGER COTY,COMAX,CONUM
      DOUBLE PRECISION LAM,XCO(COMAX),YCO(COMAX)
      CHARACTER*80 COEFFS
      INTEGER LUN,ISTAT,I
      LOGICAL NOCO

C Interpret the coefficients file name and act appropriately
      NOCO=.FALSE.
      IF(COEFFS.EQ.' ') NOCO=.TRUE.

C Now check for the special string "header"
      IF(COEFFS.EQ.'header') THEN

C and try to read these from the input header
         CALL GTCOEF(IDD,COEFFS,LAM,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     : '! Cannot deduce image distortion information from header',
     :      1,0,ISTAT)
            ISTAT=1
            RETURN
         ENDIF
      ENDIF

C Set default values
      IF(NOCO) THEN
         DO I=1,COMAX
            XCO(I)=0.0D0
            YCO(I)=0.0D0
         ENDDO
C To avoid problems later set the linear terms to the identity, not
C just zero (added, Richard Hook, July 2005)
         XCO(2)=1.0D0 
         YCO(3)=1.0D0
         CONUM=1
         COTY=0
      ELSE

C Check for WCS coefficients - Spitzer format in image
C header
         IF(COEFFS.EQ.'wcs') THEN
           CALL GTSPCO(IDD,COTY,CONUM,XCO,YCO,ISTAT)
           CALL UMSPUT('-Reading non-linear WCS coefficients',
     :                 1,0,ISTAT)
           RETURN

         ELSE
C Open the coefficients file
           CALL UFOPEN(COEFFS,1,LUN,ISTAT)
           IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Unable to open coefficients file',1,0,
     :                  ISTAT)
            ISTAT=1
            RETURN
           ELSE
            CALL UMSPUT('-Opening coefficients file: '//COEFFS,
     :                      1,0,ISTAT)
           ENDIF
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


      SUBROUTINE GGEOIM(XGEOIM,YGEOIM,PXG,PYG,XGDIM,YGDIM,DISIM)
C
C Read the distortion image arrays from distortion images
C   Dynamically allocates space for arrays PXG, PYG based on size 
C   of arrays as read from image header using 'UIMGID'.
C
C If no images are read in, XGDIM and YGDIM default to values of 1.
C
C
C Code was moved from 'GETGEO'.  W. Hack, June 2004
C

      CHARACTER*80 XGEOIM,YGEOIM,CHARS
      INTEGER DATTYP,DDIMS(7),NDIMS,IXG,IYG
      INTEGER ISTAT,PXG,PYG
      LOGICAL DISIM
      INTEGER XGDIM,YGDIM

      REAL MEMR(1)
      COMMON /MEM/MEMR

      DISIM=.FALSE.
      XGDIM = 2
      YGDIM = 2
      
C Check we have both distortion images
      IF(XGEOIM.NE.' ' .AND. YGEOIM.EQ.' ' .OR.
     :   XGEOIM.EQ.' ' .AND. YGEOIM.NE.' ') THEN
         CALL UMSPUT('! Must have both X and Y distortion images',
     :               1,0,ISTAT)
         ISTAT=1
                          
         RETURN
      ENDIF
         
C Try to open the distortion images
C allocate space and read them in...
      PXG=0
      PYG=0
C X first
      IF(XGEOIM.NE.' ') THEN
         CALL UIMOPN(XGEOIM,1,IXG,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Failed to open X-shift distortion image',
     :                  1,0,ISTAT)
            ISTAT=1
         ELSE
            CHARS='-Opening X-shift distortion image: '//XGEOIM
            CALL UMSPUT(CHARS,1,0,ISTAT)
         ENDIF

C Get the array size and shape
         IF(ISTAT.EQ.0) THEN
            CALL UIMGID(IXG,DATTYP,NDIMS,DDIMS,ISTAT)
            XGDIM = DDIMS(1)
            YGDIM = DDIMS(2)

C Allocate space
            IF (ISTAT .EQ. 0) THEN
               CALL UDMGET(DDIMS(1)*DDIMS(2),6,PXG,ISTAT)
               IF(ISTAT.NE.0) THEN
                  CALL UMSPUT('! Failed to allocate X-shift space',
     :                     1,0,ISTAT)
                  ISTAT=1
               ENDIF

C Read the image
               IF(ISTAT.EQ.0) THEN
                  CALL RIMR(IXG,DDIMS(1),DDIMS(2),MEMR(PXG),ISTAT)
                  IF(ISTAT.NE.0) THEN
                     CALL UMSPUT('! Failed to read X-shift image',
     :                          1,0,ISTAT)
                     ISTAT=1
                  ENDIF
               ENDIF
            ENDIF
         ENDIF

C Close the image
         CALL UIMCLO(IXG,ISTAT)
      ENDIF
      
      IF(ISTAT.NE.0) THEN
C Initialize default arrays upon error.
         DISIM=.FALSE.
         XGDIM = 2
         YGDIM = 2
         CALL UDMGET(XGDIM*YGDIM,6,PXG,ISTAT)
         CALL UDMGET(XGDIM*YGDIM,6,PYG,ISTAT)
         RETURN
      ENDIF
C Y next 
      IF(YGEOIM.NE.' ') THEN
         CALL UIMOPN(YGEOIM,1,IYG,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Failed to open Y-shift distortion image',
     :                  1,0,ISTAT)
            ISTAT=1
         ELSE
            CHARS='-Opening Y-shift distortion image: '//YGEOIM
            CALL UMSPUT(CHARS,1,0,ISTAT)
         ENDIF

C Get the array size and shape
         IF (ISTAT .EQ. 0) THEN
            CALL UIMGID(IYG,DATTYP,NDIMS,DDIMS,ISTAT)

            IF (XGDIM .LT. 0 .AND. YGDIM .LT. 0) THEN
               XGDIM = DDIMS(1)
               YGDIM = DDIMS(2)
            ENDIF
C Allocate space
            IF (ISTAT .EQ. 0) THEN
               CALL UDMGET(DDIMS(1)*DDIMS(2),6,PYG,ISTAT)

               IF(ISTAT.NE.0) THEN
                  CALL UMSPUT('! Failed to allocate Y-shift space',
     :                    1,0,ISTAT)
                  ISTAT=1
               ENDIF

C Read the image
               IF (ISTAT .EQ. 0) THEN
                  CALL RIMR(IYG,DDIMS(1),DDIMS(2),MEMR(PYG),ISTAT)

                  IF(ISTAT.NE.0) THEN
                     CALL UMSPUT('! Failed to read Y-shift image',
     :                        1,0,ISTAT)
                     ISTAT=1
                  ENDIF
               ENDIF
            ENDIF
         ENDIF

C Close the image
         CALL UIMCLO(IYG,ISTAT)

         DISIM=.TRUE.
      ENDIF
      IF(ISTAT.NE.0) THEN
C Initialize default arrays upon error.
         DISIM=.FALSE.
         XGDIM = 2
         YGDIM = 2
         CALL UDMGET(XGDIM*YGDIM,6,PXG,ISTAT)
         CALL UDMGET(XGDIM*YGDIM,6,PYG,ISTAT)
         RETURN
      ENDIF

C Set a good status return
      ISTAT=0
      RETURN
      END

      SUBROUTINE COPHED(IDI,IDO,COPALL,ISTAT)
C
C Copy over some important header items from the first
C drizzled input image to the newly created output images.
C
C Richard Hook, March 2001
C
      IMPLICIT NONE

      INTEGER IDI,IDO,ISTAT
      LOGICAL COPALL

      CHARACTER*80 BUFFER

C If we are to copy everything just do it directly
      IF(COPALL) THEN
         CALL UHDCPY(IDI,IDO,ISTAT)
      ELSE

C Just copy a few things

C First equinox and radesys - as strings
         CALL UHDGST(IDI,'EQUINOX',BUFFER,ISTAT)
         IF(ISTAT.EQ.0) 
     : CALL UHDAST(IDO,'EQUINOX',BUFFER,' ',0,ISTAT)
         CALL UHDGST(IDI,'RADESYS',BUFFER,ISTAT)
         IF(ISTAT.EQ.0) 
     : CALL UHDAST(IDO,'RADESYS',BUFFER,' ',0,ISTAT)

C Now instrument related things
         CALL UHDGST(IDI,'INSTRUME',BUFFER,ISTAT)
         IF(ISTAT.EQ.0) 
     : CALL UHDAST(IDO,'INSTRUME',BUFFER,' ',0,ISTAT)
         CALL UHDGST(IDI,'DETECTOR',BUFFER,ISTAT)
         IF(ISTAT.EQ.0) 
     : CALL UHDAST(IDO,'DETECTOR',BUFFER,' ',0,ISTAT)
         CALL UHDGST(IDI,'CAMERA',BUFFER,ISTAT)
         IF(ISTAT.EQ.0) 
     : CALL UHDAST(IDO,'CAMERA',BUFFER,' ',0,ISTAT)
         CALL UHDGST(IDI,'PHOTPLAM',BUFFER,ISTAT)
         IF(ISTAT.EQ.0) 
     : CALL UHDAST(IDO,'PHOTPLAM',BUFFER,' ',0,ISTAT)
      ENDIF

      RETURN
      END

      SUBROUTINE DUPWCS(WCSIN,WCSOUT,DNX,DNY,COEFFS,
     :                  XGEOIM,YGEOIM,PXG,PYG,ID,ISTAT)
C
C Update the WCS of an image to allow for the distortion
C of a coefficients file.
C
C This is a wrapper for UPWCS when all the geometric parameters
C are null.
C
C Written for WCS2DR.
C
C  Richard Hook, May 2002, STScI
C
C Updated for the distortion images, August 2003
C
      IMPLICIT NONE

      DOUBLE PRECISION WCSIN(8),WCSOUT(8)
      CHARACTER*80 XGEOIM,YGEOIM
      CHARACTER*(*) COEFFS
      INTEGER DNX,DNY,ID,ISTAT,I

      INTEGER COTY,CONUM,COMAX
      PARAMETER (COMAX=100)

      DOUBLE PRECISION XCO(COMAX),YCO(COMAX)

      REAL MEMR(1)
      COMMON /MEM/MEMR

C Geometrical parameters, the standard set
      DOUBLE PRECISION SCALE,ROT,XSH,YSH,LAM
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*80 SHFTUN
      CHARACTER*8 SHFR2,SHFTFR,ALIGN,GEOMOD
      INTEGER PXG,PYG,XGDIM,YGDIM
      LOGICAL DISIM

      LOGICAL SECPAR,ROTFIR,USEWCS,ROTF2

C Set default values
      XSH=0.0D0
      YSH=0.0D0
      SCALE=1.0D0
      ROT=0.0D0
      SHFTFR="output"
      SHFTUN="output"
      GEOMOD="user"
      SECPAR=.FALSE.
      ROTFIR=.TRUE.
      ALIGN="center"

C If no coefficients then just return after copying
      IF(COEFFS.EQ.' ') THEN
         DO I=1,8
            WCSOUT(I)=WCSIN(I)
         ENDDO
         CONUM=1
         COTY=0
      ENDIF

C Get the lambda parameter
      CALL UCLGSD('lambda',LAM,ISTAT)

C Get the geometric distortion coefficient information
      CALL GETGEO(COEFFS,ID,LAM,
     :      COTY,COMAX,CONUM,XCO,YCO,ISTAT)

      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :     '! Error, failed to get geometric distortion coefficients',
     :     1,0,ISTAT)
           ISTAT=1
         RETURN
      ENDIF

C  Read in any distortion image corrections, if specified
C  PXG,PYG will default to 2x2 arrays when DISIM returns as FALSE.
C 
      CALL GGEOIM(XGEOIM,YGEOIM,PXG,PYG,XGDIM,YGDIM,DISIM)

C Update the WCS
      CALL UPWCS(WCSIN,WCSOUT,DNX,DNY,DNX,DNY,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,COTY,CONUM,XCO,YCO,
     :            DISIM,PXG,PYG)

      ISTAT=0
      RETURN
      END

      SUBROUTINE RIMR(ID,NX,NY,D,ISTAT)
C
C Read an open, real image into a pre-allocated space
C
      IMPLICIT NONE

      INTEGER ID,NX,NY,ISTAT
      REAL D(NX,NY)

      INTEGER J

      DO J=1,NY
         CALL UIGL2R(ID,J,D(1,J),ISTAT)
      ENDDO

      END

      SUBROUTINE GTSPCO(ID,COTY,CONUM,XCO,YCO,ISTAT)
C
C Read Spitzer-format WCS distortion coefficients from an
C image header and populate the distortion arrays.
C
C Experimental first implementation, Richard Hook, March 2004
C Relies heavily on STSDAS f77/vos routines...
C
      IMPLICIT NONE

      INTEGER COTY,ID,CONUM,ISTAT,IS,XORD,YORD,ORDER,N,M
      INTEGER COOFF,NC
      PARAMETER (COOFF=100)

      CHARACTER*12 CTYPE1,CTYPE2
      CHARACTER*5 KEY
      DOUBLE PRECISION CRPIX1,CRPIX2,XCO(*),YCO(*)

C First try to get the CTYPE keywords and confirm they are as
C expected
      CALL UHDGST(ID,'CTYPE1',CTYPE1,ISTAT)

      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CTYPE1 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN

      IF(CTYPE1.NE.'RA---TAN-SIP') THEN
         CALL UMSPUT('! SIP-style CTYPE not found',1,0,IS)
         ISTAT=1
         RETURN
      ENDIF

      CALL UHDGST(ID,'CTYPE2',CTYPE2,ISTAT)

      IF(ISTAT.NE.0) CALL UMSPUT(
     :   '! Warning, unable to read CTYPE2 from header',1,0,IS)
      IF(ISTAT.NE.0) RETURN

      IF(CTYPE2.NE.'DEC--TAN-SIP') THEN
         CALL UMSPUT('! SIP-style CTYPE not found',1,0,IS)
         ISTAT=1
         RETURN
      ENDIF

C Get the reference pixel
      CALL UHDGSD(ID,'CRPIX1',CRPIX1,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :   '! Unable to read CRPIX1 from header',1,0,IS)
         
         RETURN
      ENDIF

      CALL UHDGSD(ID,'CRPIX2',CRPIX2,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :   '! Unable to read CRPIX2 from header',1,0,IS)
         ISTAT=1
         RETURN
      ENDIF

C Now look for the order in X and Y
      CALL UHDGSI(ID,'A_ORDER',XORD,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :   '! Unable to find X polynomial order in header',1,0,IS)
         ISTAT=1
         RETURN
      ENDIF

      CALL UHDGSI(ID,'B_ORDER',YORD,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :   '! Unable to find Y polynomial order in header',1,0,IS)
         ISTAT=1
         RETURN
      ENDIF

C Set the order to the higher
      ORDER=MAX(XORD,YORD)

C Offset to show we have an explicit refpix
      COTY=COOFF+ORDER
      CONUM=(ORDER+1)*(ORDER+2)/2+1

C and set the refpix to the CRPIXs
      XCO(CONUM)=CRPIX1
      YCO(CONUM)=CRPIX2

C Read all the coefficients and set any omitted ones to zero

C X first
      NC=1
      DO N=1,ORDER+1
         DO M=1,N
            WRITE(KEY,'(''A_'',I1,''_'',I1)') N-M,M-1
            CALL UHDGSD(ID,KEY,XCO(NC),ISTAT)
            IF(ISTAT.NE.0) XCO(NC)=0.0D0
            NC=NC+1
         ENDDO
      ENDDO

C and Y
      NC=1
      DO N=1,ORDER+1
         DO M=1,N
            WRITE(KEY,'(''B_'',I1,''_'',I1)') N-M,M-1
            CALL UHDGSD(ID,KEY,YCO(NC),ISTAT)
            IF(ISTAT.NE.0) YCO(NC)=0.0D0
            NC=NC+1
         ENDDO
      ENDDO

C The Spitzer coefficients are offsets so we need to add
C an extra X and Y
      XCO(2)=XCO(2)+1.0D0
      YCO(3)=YCO(3)+1.0D0

      ISTAT=0
      RETURN
      END
