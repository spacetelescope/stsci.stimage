      SUBROUTINE TRAXY  
C++
C
C TRAXY.F V1.92 Convert X,Y pixel position to drizzle output position
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
C V1.91 Modified to use common callable code
C       Warren Hack, STScI, 25th June 2004
C
C--
      IMPLICIT NONE

C Local variables
      INTEGER ISTAT,NXIN,NYIN,NXOUT,NYOUT
      INTEGER COTY,CONUM,COMAX,ID
      PARAMETER (COMAX=100)

      DOUBLE PRECISION XCO(COMAX),YCO(COMAX)
      DOUBLE PRECISION XIN,YIN,YOUT,XOUT
      DOUBLE PRECISION RA,DEC
      INTEGER IID,IRD,DATTYP,NDIMS,DDIMS(7)
      CHARACTER*80 INIM,OUTIM
      CHARACTER*1 PREFIX
      CHARACTER*132 CHARS
      CHARACTER*45 VERS
      CHARACTER*8 CTYPE1,CTYPE2

C Geometrical parameters, the standard set
      DOUBLE PRECISION SCALE,ROT,XSH,YSH,LAM
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      DOUBLE PRECISION RACEN,DECCEN,XREFP,YREFP,WCSIN(8),WCSOUT(8)
      DOUBLE PRECISION ORIENT,OUTSCL
      CHARACTER*80 COEFFS,SHFTUN,XGEOIM,YGEOIM
      CHARACTER*8 SHFR2,SHFTFR,ALIGN,GEOMOD
      INTEGER PXG,PYG,XGDIM,YGDIM
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
      VERS='TRAXY Version 1.92 (10th February 2005)'

C Get the noise flag
      CALL UCLGSB('silent',SILENT,ISTAT)

C First announce the version
      IF(.NOT.SILENT) CALL UMSPUT('+ '//VERS,1,0,ISTAT)

C Allow WCS mode (WTRAXY/TRAXY switch)
CCC      CALL UCLGST('geomode',GEOMOD,ISTAT)
      GEOMOD='user'

C Get all the geometrical parameters
      CALL GTGEPA(GEOMOD,COEFFS,LAM,XGEOIM,YGEOIM,
     :             SCALE,ROT,XSH,YSH,SHFTFR,SHFTUN,ALIGN,
     :             XSCALE,YSCALE,XSH2,YSH2,ROT2,SHFR2,
     :             OUTSCL,RACEN,DECCEN,XREFP,YREFP,ORIENT)

C If so, check for image names (output is optional)
      IF(GEOMOD.EQ.'wcs') THEN
         CALL UCLGST('inimage',INIM,ISTAT)
         CALL UCLGST('refim',OUTIM,ISTAT)
         USEWCS=.TRUE.
      ELSE
         USEWCS=.FALSE.
      ENDIF   

C Open the files and read the WCS from the headers
      IF(USEWCS) THEN
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
C Note that no image is available so coeffs="header" will fail
      CALL GETGEO(COEFFS,ID,LAM,
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

C Get the X,Y pixel position to be transformed 
      CALL UCLGSD('xin',XIN,ISTAT)
      CALL UCLGSD('yin',YIN,ISTAT)

C Also get the chip size - in and out
C Not needed for GEOMOD='wcs'
      IF(.NOT.USEWCS) THEN
         CALL UCLGSI('nxin',NXIN,ISTAT)
         CALL UCLGSI('nyin',NYIN,ISTAT)
      ENDIF

      IF(.NOT.USEWCS .AND. OUTIM.NE.' ') THEN
         CALL UCLGSI('nxout',NXOUT,ISTAT)
         CALL UCLGSI('nyout',NYOUT,ISTAT)
      ENDIF

C Convert to radians
      ROT=ROT*PI/180.0D0

C Transform the position
C Note that this is exactly the same as that in Drizzle
C Double precision version
      IF(NINT(XIN) .GT. 1 .AND. NINT(XIN) .LT. NXIN-1 .AND.
     :   NINT(YIN) .GT. 1 .AND. NINT(YIN) .LT. NYIN-1) THEN

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

C Write out the result
      IF(.NOT.SILENT) THEN
         WRITE(CHARS,
     :  '('' Xin,Yin: '',2F12.5'' Xout,Yout: '',2F12.5)')
     : XIN,YIN,XOUT,YOUT
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

C Also write the result to the CL
      CALL UCLPSD('xout',XOUT,ISTAT)
      CALL UCLPSD('yout',YOUT,ISTAT)

C End of main TRAXY module
 99   CONTINUE

C If allocated, free the distortion image memory
      IF(PXG.NE.0) CALL UDMFRE(PXG,6,ISTAT)
      IF(PYG.NE.0) CALL UDMFRE(PYG,6,ISTAT)

      RETURN
      END
