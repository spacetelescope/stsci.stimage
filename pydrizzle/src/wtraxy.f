      SUBROUTINE WTRAXY  
C++
C
C WTRAXY.F V2.41 Convert X,Y pixel position to drizzle output position
C
C This task uses coefficients held in a text file to define
C geometrical distortions.
C
C This code uses the F77/VOS for IRAF i/o.
C It may be compiled and linked using a command like:
C
C xc -p stsdas tranback.f drutil.f x_tranback.x <dir>libiraf77.a
C
C History:
C
C First quick version, 17th April 1997
C
C V0.2 modified to have the centre position as parameters, 4th Sept 97
C
C V0.3 added a rotation around the centre, 19th August 1998, STScI
C      also specified "align" and chip size.
C
C V0.31 possibility to have coeffs="" for simple linear case, 25th August 98
C
C V0.4 with full set of drizzle parameter for compatibility with
C      TRANBACK and release of STECF IRAF package. 17th December 1999
C
C V0.5 modified to use new calling sequences for some DRUTIL routines
C      and hence fix bugs, November 2001
C
C V1.0 using the full geometrical possibilities available in Drizzle V2.6.
C      December 2001
C
C V1.1 return to 3 decimal places and small code tinyup
C      January 2002
C
C V1.2 using new DRIVAL call
C      Richard Hook, ST-ECF, 27th February 2002
C
C V1.3 using packaged coefficient file access and hence fixing bug
C      Richard Hook, ST-ECF, 11th March 2002
C
C V1.4 write out results to IRAF parameters and added silent flag
C      for more convenient scripting application
C      Richard Hook, ST-ECF/STScI, 4th September 2002
C
C V1.5 WTRAXY version for WCS tests,
C      Richard Hook, ST-ECF/STScI, 19th March 2003
C
C V1.6 include distortion image support,,
C      Richard Hook, ST-ECF/STScI, 26th August 2003
C
C V1.7 using double precision for almost everything
C      Richard Hook, ST-ECF/STScI, 21st October 2003
C
C V1.8 added PREFIX parameter
C       Richard Hook, ST-ECF/STScI, 23rd October 2003
C
C V1.9 added refpix support
C       Richard Hook, ST-ECF/STScI, 5th December 2003
C
C V2.0 added pixel-area output
C       Richard Hook, ST-ECF/STScI, 12th March 2004
C
C V2.1 allowed the input image header to be passed to
C      GETCOE to read coefficients, if appropriate. This is
C      part of Spitzer support.
C       Richard Hook, ST-ECF/STScI, 26th March 2004
C
C V2.2 add xylist input option.
C       Richard Hook, ST-ECF/STScI, April 2004
C
C V2.3 add output X,Y image support.
C       Richard Hook, ST-ECF/STScI, April 12th 2004.
C
C V2.4 modified to work with common callable code
C       Warren Hack, STScI, June 25th 2004.
C
C--
      IMPLICIT NONE

C Local variables
      INTEGER ISTAT,NXIN,NYIN,NXOUT,NYOUT
      INTEGER COTY,CONUM,COMAX
      PARAMETER (COMAX=100)

      DOUBLE PRECISION XCO(COMAX),YCO(COMAX)
      DOUBLE PRECISION XIN,YIN,XOUT,YOUT,X(4),Y(4),YO(4),XO(4)
      DOUBLE PRECISION RA,DEC
      INTEGER IID,IRD,DATTYP,NDIMS,DDIMS(7)
      CHARACTER*80 INIM,OUTIM
      CHARACTER*1 PREFIX
      CHARACTER*132 CHARS
      CHARACTER*45 VERS
      CHARACTER*8 CTYPE1,CTYPE2

      CHARACTER*80 OUTXIM,OUTYIM
      INTEGER MAXBUF,I,J,IDX,IDY
      PARAMETER (MAXBUF=100000)
      DOUBLE PRECISION XIA(2),YIA(2),XOA(MAXBUF),YOA(MAXBUF)
      REAL XOAS(MAXBUF),YOAS(MAXBUF)

C Geometrical parameters, the standard set
      DOUBLE PRECISION SCALE,ROT,XSH,YSH,LAM
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      DOUBLE PRECISION RACEN,DECCEN,XREFP,YREFP,WCSIN(8),WCSOUT(8)
      DOUBLE PRECISION ORIENT,OUTSCL,PAM
      CHARACTER*80 COEFFS,SHFTUN,XGEOIM,YGEOIM,XYLIST,LINE
      CHARACTER*8 SHFR2,SHFTFR,ALIGN,GEOMOD
      INTEGER PXG,PYG,LUN,XGDIM,YGDIM
      LOGICAL DISIM

      REAL MEMR(1)
      COMMON /MEM/MEMR

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793D0)

      LOGICAL ROTFIR
      LOGICAL USEWCS
      LOGICAL ROTF2
      LOGICAL SECPAR
      LOGICAL SILENT

C-- Start of executable code
      VERS='WTRAXY Version 2.41 (10th February 2005)'

C Get the noise flag
      CALL UCLGSB('silent',SILENT,ISTAT)

C First announce the version
      IF(.NOT.SILENT) CALL UMSPUT('+ '//VERS,1,0,ISTAT)

C Allow WCS mode
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
C
C Note that this can have a prefix (CCCD1_1 etc)
         IF(OUTIM.NE.' ') THEN

C Get the prefix character
            CALL UCLGST('refpref',PREFIX,ISTAT)

            CALL UIMOPN(OUTIM,1,IRD,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT('! Unable to open output image',
     :                     1,0,ISTAT)
               GO TO 99
            ELSE
               CALL UMSPUT('-Reading WCS from '//OUTIM,1,0,ISTAT)
               IF(PREFIX.NE.' ') THEN
                  CALL UMSPUT('-Using prefix: '//PREFIX,1,0,ISTAT)
               ENDIF
            ENDIF

            IF(PREFIX.NE.' ') THEN
               CALL GTPWCS(IRD,PREFIX,WCSOUT,CTYPE1,CTYPE2,ISTAT)
            ELSE
               CALL GETWCS(IRD,WCSOUT,CTYPE1,CTYPE2,ISTAT)
            ENDIF

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
      ROT2=ROT2*PI/180.0

      IF(SHFR2.eq.'input') THEN
         ROTF2=.FALSE.
      ELSE
         ROTF2=.TRUE.
      ENDIF

C Give a warning message if secondary parameters will have an effect
      IF(XSCALE.NE.1.0D0 .OR. YSCALE.NE.1.0D0 .OR.
     :   XSH2.NE.0.0D0 .OR. YSH2.NE.0.0D0 .OR. ROT2.NE.0.0D0) THEN
       IF(.NOT.SILENT) CALL UMSPUT(
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
C Note that an input image is not always available so 
C coeffs="header"  or coeffs="wcs" may fail
      CALL GETGEO(COEFFS,IRD,LAM,
     :            COTY,COMAX,CONUM,XCO,YCO,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :  '! WTRAXY: Error, failed to get distortion coefficients',
     :  1,0,ISTAT)
          GO TO 99
      ENDIF

C  Read in any distortion image corrections, if specified
C  PXG,PYG will default to 2x2 arrays when DISIM returns as FALSE.
C 
      CALL GGEOIM(XGEOIM,YGEOIM,PXG,PYG,XGDIM,YGDIM,DISIM)

C Also get the chip size - in and out
C Not needed for GEOMOD='wcs'
      IF(.NOT.USEWCS) THEN
         CALL UCLGSI('nxin',NXIN,ISTAT)
         CALL UCLGSI('nyin',NYIN,ISTAT)
      ENDIF

      IF(.NOT.USEWCS .AND. OUTIM.EQ.' ') THEN
         CALL UCLGSI('nxout',NXOUT,ISTAT)
         CALL UCLGSI('nyout',NYOUT,ISTAT)
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
         CALL UCLGSD('xin',XIN,ISTAT)
         CALL UCLGSD('yin',YIN,ISTAT)
      ENDIF

C Transform the position
C Note that this is exactly the same as that in Drizzle
C Double precision version
      DO WHILE(.TRUE.)

         IF(XYLIST.NE.' ') THEN
            CALL UFGLIN(LUN,LINE,ISTAT)
            IF(ISTAT.NE.0) GO TO 88
            READ(LINE,*) XIN,YIN
         ENDIF
         
         IF(NINT(XIN).GT.1 .AND. NINT(XIN).LT.NXIN-1 .AND.
     :         NINT(YIN).GT.1 .AND. NINT(YIN).LT.NYIN-1) THEN

            CALL DRIVAL(XIN,YIN,1,NXIN,NYIN,NXOUT,NYOUT,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,
     :            DISIM,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,XOUT,YOUT)
         ELSE 
            CALL DRIVAL(XIN,YIN,1,NXIN,NYIN,NXOUT,NYOUT,.FALSE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,
     :            .FALSE.,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,XOUT,YOUT)
         ENDIF

C Transform a unit square to calculate the pixel area
         X(1)=XIN-0.5D0
         X(2)=XIN+0.5D0
         X(3)=XIN+0.5D0
         X(4)=XIN-0.5D0
         Y(1)=YIN-0.5D0
         Y(2)=YIN-0.5D0
         Y(3)=YIN+0.5D0
         Y(4)=YIN+0.5D0

         CALL DRIVAL(X,Y,4,NXIN,NYIN,NXOUT,NYOUT,.FALSE.,
     :            0.0D0,0.0D0,0.0D0,1.0D0,ALIGN,ROTFIR,
     :            .FALSE.,0.0D0,0.0D0,0.0D0,1.0D0,1.0D0,SHFR2,ROTF2,
     :            .FALSE.,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,
     :            .FALSE.,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,XO,YO)

         PAM=DABS(0.5*((XO(2)-XO(4))*(YO(1)-YO(3)) -
     :              (XO(1)-XO(3))*(YO(2)-YO(4))))

C Write out the result
         IF(.NOT.SILENT) THEN
            WRITE(CHARS,
     :  '('' Xin,Yin: '',2F12.5'' Xout,Yout: '',2F12.5,
     :    '' PAM: '',F12.5)')
     : XIN,YIN,XOUT,YOUT,PAM
            CALL UMSPUT(CHARS,1,0,ISTAT)

C If in WCS mode also report the sky position
            IF(USEWCS) THEN
               CALL XY2RD(XOUT,YOUT,RA,DEC,WCSOUT)
               WRITE(CHARS,
     : '('' RA: '',F12.7,'' Dec: '',F12.7,'' (deg)'')')
     :    RA,DEC
               CALL UMSPUT(CHARS,1,0,ISTAT)
            ENDIF
         ENDIF

         IF(XYLIST.EQ.' ') GO TO 88
      ENDDO

 88   CONTINUE

C Also write the result to the CL
      IF(XYLIST.EQ.' ') THEN
         CALL UCLPSD('xout',XOUT,ISTAT)
         CALL UCLPSD('yout',YOUT,ISTAT)
      ELSE
         CALL UFCLOS(LUN,ISTAT)
      ENDIF

C Image creation support
      CALL UCLGST('outxim',OUTXIM,ISTAT)
      CALL UCLGST('outyim',OUTYIM,ISTAT)

C Skip this if not image option
      IF(OUTXIM.EQ.' ' .AND. OUTYIM.EQ.' ') GO TO 99

      DDIMS(1)=NXIN
      DDIMS(2)=NYIN

C Check for huge images
      IF(NXIN.GT.MAXBUF .OR. NYIN.GT.MAXBUF) THEN
         CALL UMSPUT('! Image too large for output image option',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF

C Create images if not null
      IF(OUTXIM.NE.' ') THEN
         CALL UIMCRE(OUTXIM,6,2,DDIMS,IDX,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Failed to created output X image',
     :                  1,0,ISTAT)
            GO TO 99
         ELSE
            CALL UMSPUT('-Created output X position image: '//
     :                  OUTXIM,1,0,ISTAT)
         ENDIF
      ENDIF

      IF(OUTYIM.NE.' ') THEN
         CALL UIMCRE(OUTYIM,6,2,DDIMS,IDY,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Failed to created output Y image',
     :                  1,0,ISTAT)
            GO TO 99
         ELSE
            CALL UMSPUT('-Created output Y position image: '//
     :                  OUTYIM,1,0,ISTAT)
         ENDIF
      ENDIF

C Loop over the input image grid
C We are going to use the "regular" feature so need to
C initialise some things
      XIA(1)=1.0D0
      XIA(2)=0.0D0
      YIA(2)=0.0D0

      DO J=1,NYIN
         YIA(1)=DBLE(J)

         CALL DRIVAL(XIA,YIA,NXIN,NXIN,NYIN,NXOUT,NYOUT,.TRUE.,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,WCSIN,WCSOUT,
     :            COTY,CONUM,XCO,YCO,
     :            DISIM,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,XOA,YOA)

C Copy to single precision
         DO I=1,NXIN
            XOAS(I)=SNGL(XOA(I))
            YOAS(I)=SNGL(YOA(I))
         ENDDO

C Write out the lines
         IF(OUTXIM.NE.' ') CALL UIPL2R(IDX,J,XOAS,ISTAT)
         IF(OUTYIM.NE.' ') CALL UIPL2R(IDY,J,YOAS,ISTAT)
      ENDDO

C Close the files
      IF(OUTXIM.NE.' ') CALL UIMCLO(IDX,ISTAT)
      IF(OUTYIM.NE.' ') CALL UIMCLO(IDY,ISTAT)

C End of main TRAXY module
 99   CONTINUE

C If allocated, free the distortion image memory
      IF(PXG.NE.0) CALL UDMFRE(PXG,6,ISTAT)
      IF(PYG.NE.0) CALL UDMFRE(PYG,6,ISTAT)

      RETURN
      END
