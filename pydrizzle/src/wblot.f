      SUBROUTINE WBLOT
C++
C
C WBLOT.F Version 3.4   "Reverse drizzling" 
C
C It is based on ideas of Andy Fruchter and was coded by Richard Hook. 
C
C This task uses coefficients held in a text file to define
C geometrical distortions.
C
C This code uses the F77/VOS for IRAF i/o.
C It may be compiled and linked using a command like:
C
C xc -p stsdas (-z) blot.f drutil.f x_blot.x <dir>libiraf77.a <dir>libiminterp.a
C
C Note that the optional -z switch (which tells IRAF to create a static
C executable which doesn't use the IRAF sharable library) may be
C necessary to avoid a virtual memory use limitations. 
C
C History:
C
C Created first try (V0.1) from DRIZZLE.F V0.97, 9th July 1996
C Modified for new origin of coordinates etc (V0.2), 17th July 1996
C Modified to match up with DRIZZLE V1.05, 14th November 1996
C (rotfir parameter added, few bug fixes, change coordinate origin)
C Match up to Drizzle 1.08 - V0.4 of BLOT, 6th December 1996
C Correct shift units bug, V0.41, 16th December 1996
C Add ALIGN option, V0.42, 16th January 1997
C Use "drutil.f" utilities, no -z, 4th February 1997
C Prepared V0.5 for release, matches V1.097 of Drizzle, 17th March 1997
C
C First public release of Blot, 12th May 1997
C
C Added in_un/out_un switches for compatibility with V1.2 of Drizzle.
C This was for the first release of drizzle/blot in the dither package
C within STSDAS. 25th February 1998.
C
C Added automatic output image size selection, 27th February 1998
C
C Modified calling sequences to match the drutil.f with Drizzle 2.6,
C  December 2001
C
C V1.0 with full geometric support (via the routines in drutil.f)
C  December 2001.
C
C V1.1 with use of new DRIVAL call
C   Richard Hook, ST-ECF, 27th February 2002
C
C V1.3 with cleaner access to coefficients,
C   Richard Hook, ST-ECF, 11th March 2002
C
C V2.0: image subsetting to allow working with very large mosaics,
C       also wblot form for working with WCS. Also additional sinc
C       interpolation kernels.
C        Richard Hook, ST-ECF at STScI, December 2002
C
C V2.01: added test for zero exposure time input.
C        Richard Hook, ST-ECF at STScI, January 2003
C
C V3.0: compatibility with Drizzle 3.0, June 2003.
C
C V3.01: small modification to correctly get output dimensions
C        correctly in all cases,  ST-ECF at STScI, July 2003
C
C V3.1: updated for image distortion arrays.
C         Richard Hook, ST-ECF at STScI, August 2003
C
C V3.2: double precision version.
C         Richard Hook, ST-ECF at STScI, October 2003
C
C V3.32: added support for more kernels, through new interface
C        and scaling for kernel to support experiments.
C            Richard Hook, ST-ECF at STScI, January 2004
C
C V3.3.3 support for Spitzer style header polynomials.
C            Richard Hook, ST-ECF at STScI, March 2004
C
C V3.4:  modified to use common callable code.
C            Warren Hack, STScI, 25th June 2004
C
C
C--
      IMPLICIT NONE

C Iraf global storage (real in this program)
      REAL MEMR(1) !I
      DOUBLE PRECISION MEMD(1) !I
      COMMON /MEM/MEMR         !I
      EQUIVALENCE (MEMR,MEMD)  !I

C Local variables
      INTEGER ISTAT,I
      INTEGER NDIMS,DNX,DNY,ONX,ONY,DDIMS(7),ODIMS(7),DATTYP
      INTEGER IDD,IRD,IDND
      INTEGER PDATA,PNDAT,PBUFF
      INTEGER PXIN,PYIN,PXOUT,PYOUT
      INTEGER INTYPE,IC
      INTEGER XMIN,XMAX,YMIN,YMAX
      REAL EF,EXPIN,EXPOUT,EXPREF
      DOUBLE PRECISION LASTPF,LASTSC,LASTOX,LASTOY
      DOUBLE PRECISION WCSIN(8),WCSOUT(8),XREFP,YREFP
      CHARACTER*80 DATA,OUTDAT,REFIMA,CHARS
      CHARACTER*8 CTYPE1,CTYPE2
      CHARACTER*8 EXPKEY,INUN,OUTUN,LASTUN
      CHARACTER*80 EXPSTR
      CHARACTER*45 VERS
      CHARACTER*20 INTERP
      REAL KSCALE,SINSCL
      INTEGER COTY,COMAX,CONUM
      PARAMETER (COMAX=100)
      DOUBLE PRECISION XCO(COMAX),YCO(COMAX)
      REAL MISVAL

      INTEGER MAXPIX
      PARAMETER (MAXPIX=2E7)

C Geometrical parameters, the standard set
      DOUBLE PRECISION SCALE,ROT,XSH,YSH,LAM
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      DOUBLE PRECISION RACEN,DECCEN
      DOUBLE PRECISION ORIENT,OUTSCL
      CHARACTER*80 COEFFS,SHFTUN,XGEOIM,YGEOIM
      INTEGER PXG,PYG,XGDIM,YGDIM
      LOGICAL DISIM
      CHARACTER*8 SHFR2,SHFTFR,ALIGN,GEOMOD

C Constants
      DOUBLE PRECISION PI
      PARAMETER (PI=3.1415926536D0)

C Logical flags
      LOGICAL VERBOSE
      LOGICAL GOTWCS
      LOGICAL ROTFIR
      LOGICAL INCPS
      LOGICAL OUTCPS
      LOGICAL USEWCS
      LOGICAL ROTF2
      LOGICAL SECPAR
      LOGICAL SUB
      LOGICAL NOOVER

C-- Start of executable code

C First announce the version
      VERS='WBLOT Version 3.4 (25th June 2004)'

      CALL UMSPUT('+ '//VERS,1,0,ISTAT)

      VERBOSE=.TRUE.

C This line controls whether or not we can have WCS defined geometry
C This line needs to be changed to convert WBLOT to BLOT
CCC      GEOMOD='user'
      GEOMOD=' '

C Get the geomode
      IF(GEOMOD.EQ.' ') CALL UCLGST('geomode',GEOMOD,ISTAT)

C Get all the geometrical parameters
      CALL GTGEPA(GEOMOD,COEFFS,LAM,XGEOIM,YGEOIM,
     :             SCALE,ROT,XSH,YSH,SHFTFR,SHFTUN,ALIGN,
     :             XSCALE,YSCALE,XSH2,YSH2,ROT2,SHFR2,
     :             OUTSCL,RACEN,DECCEN,XREFP,YREFP,ORIENT)

C Get the output dimensions, although these may be overwritten
C by a reference image size if present (3.01 modification)
      CALL UCLGSI('outnx',ONX,ISTAT)
      CALL UCLGSI('outny',ONY,ISTAT)

C Find out whether we are using WCS to define the geometry
      IF(GEOMOD.EQ.'wcs') THEN
         USEWCS=.TRUE.
      ELSE
         USEWCS=.FALSE.
      ENDIF

C Get the exposure related parameters
      CALL UCLGST('expkey',EXPKEY,ISTAT)
      CALL UCLGST('expout',EXPSTR,ISTAT)

C Convert the parameters supplied to the output WCS if needed
C If using WCS...
      IF(USEWCS .OR. COEFFS.EQ.'wcs') THEN

C If we have a reference image we need to try to get its WCS for
C use as the output
         CALL UCLGST('refimage',REFIMA,ISTAT)
         IF(REFIMA.NE.' ') THEN
            CALL UIMOPN(REFIMA,1,IRD,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT('! Unable to open reference image',
     :                     1,0,ISTAT)
               GO TO 99
            ELSE
               CALL UMSPUT('-Reading WCS from '//REFIMA,1,0,ISTAT)
            ENDIF

            CALL GETWCS(IRD,WCSOUT,CTYPE1,CTYPE2,ISTAT)
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
               ONX=DDIMS(1)
               ONY=DDIMS(2)
               WRITE(CHARS,
     :  '(''-Blotted image will be '',I6,'' by '',I6)') ONX,ONY
               CALL UMSPUT(CHARS,1,0,ISTAT)
            ENDIF

C Get the reference image exposure time
            CALL UHDGSR(IRD,EXPKEY,EXPREF,ISTAT)
            IF(ISTAT.NE.0) EXPREF=0.0

C Close the image
            CALL UIMCLO(IRD,ISTAT)
         ELSE

C Now convert these into WCS style representation
            CALL SETWCS(OUTSCL,ORIENT,XREFP,RACEN,
     :               YREFP,DECCEN,WCSOUT)
         ENDIF
      ENDIF

C Get reference frame for shifts - specifying whether shifts are done before
C or after rotation
      IF(SHFTFR.EQ.'output') THEN
         ROTFIR=.FALSE.
      ELSE
         ROTFIR=.TRUE.
      ENDIF

C Convert the secondary parameters into suitable units
      ROT2=ROT2*PI/180.0D0
      IF(SHFR2.EQ.'input') THEN
         ROTF2=.FALSE.
      ELSE
         ROTF2=.TRUE.
      ENDIF

C Give a warning message if secondary parameters will have an effect
      IF(XSCALE.NE.1.0D0 .OR. YSCALE.NE.1.0D0 .OR.
     :   XSH2.NE.0.0D0 .OR. YSH2.NE.0.0D0 .OR. ROT2.NE.0.0D0) THEN
       CALL UMSPUT(
     : '! Warning, secondary geometric transform is being used',
     :               1,0,ISTAT)
        SECPAR=.TRUE.
      ELSE
        SECPAR=.FALSE.
      ENDIF

C Get the switch value for whether to read in CPS or
C counts (latter is the default)
      CALL UCLGST('in_un',INUN,ISTAT)
      IF(INUN.EQ.'cps') THEN
         INCPS=.TRUE.
      ELSE
         INCPS=.FALSE.
      ENDIF

C Get the switch value for whether to write out in CPS or
C counts (latter is the default)
      CALL UCLGST('out_un',OUTUN,ISTAT)
      IF(OUTUN.EQ.'cps') THEN
         OUTCPS=.TRUE.
      ELSE
         OUTCPS=.FALSE.
      ENDIF

C Check for invalid scale
      IF(SCALE.EQ.0.0D0) THEN
         CALL UMSPUT('! Invalid scale',1,0,ISTAT)
         GO TO 99
      ENDIF

C Convert the rot to radians
      ROT=ROT*PI/180.0D0

      IF(SHFTUN.EQ.'input') THEN
         XSH=XSH*SCALE
         YSH=YSH*SCALE
      ENDIF

C Note that the shifts are in INPUT pixel units

C Get the missing value flag
      CALL UCLGSR('fillval',MISVAL,ISTAT)

C Get the name of the input data image
      CALL UCLGST('data',DATA,ISTAT)

C Try to access the input data image
      CALL UIMOPN(DATA,1,IDD,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Unable to open data image',
     :                    1,0,ISTAT)
         GO TO 99
      ELSE
         IF(VERBOSE) CALL UMSPUT(
     :  '-Opening input data image: '//DATA,
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

C We now get some of the previous drizzle information from
C the header of the current image and check for funny
C combinations
      CALL GOLDH(IDD,USEWCS,IC,LASTPF,LASTSC,
     :           LASTUN,LASTOX,LASTOY,ISTAT)

C Check for incompatibilities
      IF(INCPS) THEN
         IF(LASTUN.EQ.'counts' .OR.
     :                LASTUN.EQ.'COUNTS') THEN
          CALL UMSPUT('! Warning input image header gives units',
     :               1,0,ISTAT)
          CALL UMSPUT('! as counts NOT cps! Assuming cps',
     :               1,0,ISTAT)
         ENDIF
      ELSE
         IF(LASTUN.EQ.'cps' .OR.
     :                LASTUN.EQ.'CPS') THEN
          CALL UMSPUT('! Warning input image header gives units',
     :               1,0,ISTAT)
          CALL UMSPUT('! as cps NOT counts! Assuming counts',
     :               1,0,ISTAT)
         ENDIF
      ENDIF

C Get the geometric distortion coefficient information
C If we have a reference image we use that
      IF(COEFFS.EQ.'wcs') THEN
         CALL GETGEO(COEFFS,IRD,LAM,
     :            COTY,COMAX,CONUM,XCO,YCO,ISTAT)
      ELSE
         CALL GETGEO(COEFFS,IDD,LAM,
     :            COTY,COMAX,CONUM,XCO,YCO,ISTAT)
      ENDIF

      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :  '! Error, failed to get geometric distortion coefficients',
     :  1,0,ISTAT)
          GO TO 99
      ENDIF

C  Read in any distortion image corrections, if specified
C  PXG,PYG will default to 2x2 arrays when DISIM returns as FALSE.
C 
      CALL GGEOIM(XGEOIM,YGEOIM,PXG,PYG,XGDIM,YGDIM,DISIM)
        
C We get the name of the output image here
      CALL UCLGST('outdata',OUTDAT,ISTAT)

C Check that a name is specified
      IF(OUTDAT.EQ.' ') THEN
         CALL UMSPUT(
     :       '! No output data array name specified',
     :                   1,0,ISTAT)
           GO TO 99
      ENDIF

C Set the output dimensions, these may be the values from
C the CL or from the reference image
      ODIMS(1)=ONX
      ODIMS(2)=ONY

      IDND=0

C Create the output image
      CALL UIMCRE(OUTDAT,6,2,ODIMS,IDND,ISTAT)

C This call sometimes returns a good status when it should not
C - hence the second test
      IF(ISTAT.NE.0 .OR. IDND.EQ.0) THEN
         CALL UMSPUT(
     :     '! Unable to create output data image',
     :                 1,0,ISTAT)
         GO TO 99
      ELSE
         CALL UMSPUT(
     :     '-Created new output data image: '//OUTDAT,
     :              1,0,ISTAT)
      ENDIF

C Copy the header to the output, this also transfers the exposure 
C time information
      CALL UHDCPY(IDD,IDND,ISTAT)

C Also try to get the header items for the WCS, note that
C this is not the full WCS but just 8 header items commonly used
C for HST images plus CTYPE1 & CTYPE2 
      CALL GETWCS(IDD,WCSIN,CTYPE1,CTYPE2,ISTAT)
      IF(ISTAT.EQ.0) THEN
         GOTWCS=.TRUE.
      ELSE
         CALL UMSPUT(
     :    '! Warning, unable to read WCS information from header',
     :    1,0,ISTAT)
      ENDIF

C Here we set the WCS matrix to something sensible anyway
      IF(.NOT.GOTWCS) THEN
         CTYPE1='PIXELS'
         CTYPE2='PIXELS'

         DO I=1,8
            WCSIN(I)=0.0D0
         ENDDO

         WCSIN(5)=-1.0D0
         WCSIN(8)=1.0D0
      ENDIF

C Calculate the range of pixels in the input which are accessed
C when blotting 
      CALL LCORN(ONX,ONY,DNX,DNY,
     :           XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :           SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :           USEWCS,WCSOUT,WCSIN,
     :           COTY,CONUM,XCO,YCO,
     :           DISIM,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,
     :           XMIN,XMAX,YMIN,YMAX,ISTAT)

C Assume we are using the whole frame unless set otherwise
      SUB=.FALSE.
      NOOVER=.FALSE.

C Check for no-overlap
      IF(XMAX.LT.1 .OR. YMAX.LT.1 .OR.
     :      XMIN.GT.DNX .OR. YMIN.GT.DNY) THEN
         CALL UMSPUT(
     : '! Warning, there is no overlap of the input image',
     :  1,0,ISTAT)
            WRITE(CHARS,
     : '(''! Input pixel range would be: ['',I7,'':'',I7,
     :      '','',I7,'':'',I7,'']'')') XMIN,XMAX,YMIN,YMAX
         CALL UMSPUT(CHARS,1,0,ISTAT)

         NOOVER=.TRUE.
         XMIN=1
         XMAX=1
         YMIN=1
         YMAX=1
         SUB=.TRUE.
      ELSE
         XMIN=MAX(1,XMIN)
         XMAX=MIN(DNX,XMAX)
         YMIN=MAX(1,YMIN)
         YMAX=MIN(DNY,YMAX)

C Check to see whether we really should use the subset or
C whether it makes more sense to work on the full frame.
C These criteria are a little arbitrary at this stage
         IF(DBLE(YMAX-YMIN+1)/DBLE(DNY).GT.0.5D0 .AND.
     :      DNX*DNY.LT.MAXPIX .OR.
     :      (XMIN.EQ.1 .AND. XMAX.EQ.DNX .AND.
     :      YMIN.EQ.1 .AND. YMAX.EQ.DNY)) THEN

            SUB=.FALSE.
            WRITE(CHARS,'(''-Blotting from full input image.'')')

            XMIN=1
            YMIN=1
            XMAX=DNX
            YMAX=DNY
            SUB=.FALSE.
         ELSE
            WRITE(CHARS,
     : '(''-Blotting from subset: ['',I5,'':'',I5,
     :      '','',I5,'':'',I5,''] ('',F6.2,''%).'')')
     :       XMIN,XMAX,YMIN,YMAX,
     :       100.0*DBLE((XMAX-XMIN+1)*(YMAX-YMIN+1))/
     :       DBLE(DNX*DNY)
            SUB=.TRUE.
         ENDIF

         CALL UMSPUT(CHARS,1,0,ISTAT)
      ENDIF

C Now we have the subarray sorted out we can allocate memory
C arrays and read in the data
      CALL BALLME(DNX,DNY,ONX,ONY,XMAX-XMIN+1,YMAX-YMIN+1,
     :        PBUFF,PDATA,PNDAT,PXIN,PYIN,PXOUT,PYOUT,ISTAT)

C Read in the data array subset
      CALL GETIMR(IDD,DNX,DNY,SUB,XMIN,XMAX,YMIN,YMAX,
     :                  MEMR(PBUFF),MEMR(PDATA),ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :      '! Failed to read input data image',
     :                   1,0,ISTAT)
         GO TO 99
      ENDIF

C Get the exposure time from the header, if possible
      CALL UHDGSR(IDD,EXPKEY,EXPIN,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     : '! Warning, failed to get exposure time keyword value',
     :      1,0,ISTAT)
            GO TO 99
      ENDIF

C Check for zero exposure time (yes, it does happen)
      IF(EXPIN.LE.0.0) THEN
         CALL UMSPUT(
     : '! Warning, exposure time <= 0 - setting to 1.0',1,0,ISTAT)
         EXPIN=1.0
      ENDIF

C Check for the "input" value for the output exposure,
C this is the standard behaviour
      IF(EXPSTR.EQ.'INPUT' .OR. EXPSTR.EQ.'input') THEN
         EXPOUT=EXPIN
      ELSE IF(
     :   EXPSTR.EQ.'REFERENCE' .OR. EXPSTR.EQ.'reference') THEN
         IF(EXPREF.EQ.0.0) THEN
            CALL UMSPUT(
     : '! Error - no reference image exposure time found',1,0,ISTAT)
            GO TO 99
         ELSE
            CALL UMSPUT(
     :  '-Using exposure time of reference image for output',
     :                  1,0,ISTAT)
            EXPOUT=EXPREF
         ENDIF
      ELSE
         READ(EXPSTR,*,IOSTAT=ISTAT) EXPOUT
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Invalid output exposure value',
     :                  1,0,ISTAT)
            GO TO 99
         ENDIF
      ENDIF

C Now we re-scale depends on the units, there are four cases
      IF(INCPS) THEN
         IF(OUTCPS) THEN
            EF=1.0
         ELSE
            EF=EXPOUT
         ENDIF
      ELSE
         IF(OUTCPS) THEN
            EF=1.0/EXPIN
         ELSE
            EF=EXPOUT/EXPIN
         ENDIF
      ENDIF

      CALL UHDPSR(IDND,EXPKEY,EXPOUT,ISTAT)
      WRITE(CHARS,
     :    '(''-Input exposure: '',F10.3,'' Output exposure: '''//
     :    'F10.3)') EXPIN,EXPOUT
      CALL UMSPUT(CHARS,1,0,ISTAT)

C Close the data array now we have it
      CALL UIMCLO(IDD,ISTAT)
 
C The final case is where the output is empty.
C Before we start blotting we need to drain out the output arrays
C if there are no old arrays 
      CALL SETIM(MEMR(PNDAT),ONX,ONY,0.0)

C Before we start also get the interpolation type and convert
C to a numerical code
      CALL UCLGST('interpol',INTERP,ISTAT)
      IF(INTERP.EQ.'nearest') THEN
         INTYPE=1
      ELSE IF(INTERP.EQ.'linear') THEN
         INTYPE=2
      ELSE IF(INTERP.EQ.'poly3') THEN
         INTYPE=3
      ELSE IF(INTERP.EQ.'poly5') THEN
         INTYPE=4
      ELSE IF(INTERP.EQ.'spline3') THEN
         INTYPE=5
      ELSE IF(INTERP.EQ.'sinc') THEN
         INTYPE=6
      ELSE IF(INTERP.EQ.'lsinc') THEN
         INTYPE=7
      ELSE IF(INTERP.EQ.'drizzle') THEN
         INTYPE=8
      ELSE IF(INTERP(1:4).EQ.'lan3') THEN
         INTYPE=100
      ELSE IF(INTERP(1:4).EQ.'lan5') THEN
         INTYPE=105
      ELSE
         CALL UMSPUT('! Invalid interpolant specified',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF 

C Try to read a kernel scaling factor
      IF(INTYPE.GE.100) THEN
         READ(INTERP(5:),*,IOSTAT=ISTAT) KSCALE
         IF(ISTAT.NE.0) THEN
            KSCALE=1.0
         ELSE
            WRITE(CHARS,'(''-Using kernel scale of: '',F10.4)')
     :            KSCALE
            CALL UMSPUT(CHARS,1,0,ISTAT)
         ENDIF
      ENDIF  

C Get the sinc stretch
      CALL UCLGSR('sinscl',SINSCL,ISTAT)

C Now do the actual combination using interpolation
      CALL DOBLOT(MEMR(PDATA),MEMR(PNDAT),INTYPE,SINSCL,KSCALE,MISVAL,
     :      XMIN,XMAX,YMIN,YMAX,DNX,DNY,
     :      ROTFIR,EF,ALIGN,ONX,ONY,COTY,CONUM,XCO,YCO,
     :      DISIM,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,
     :      MEMD(PXIN),MEMD(PYIN),MEMD(PXOUT),MEMD(PYOUT),
     :      SCALE,ROT,XSH,YSH,USEWCS,WCSIN,WCSOUT,
     :      GEOMOD,SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2)

C Write out the new image
      IF(VERBOSE)
     :   CALL UMSPUT('-Writing output blotted image: '//
     :               OUTDAT,1,0,ISTAT)
         
      CALL UIPS2R(IDND,1,ONX,1,ONY,MEMR(PNDAT),ISTAT)

C Update the header
      CALL BUHEAD(IDND,VERS,DATA,OUTDAT,COEFFS,XGEOIM,YGEOIM,
     :     SHFTUN,MISVAL,
     :     EXPKEY,EXPSTR,ALIGN,INUN,OUTUN,
     :     SHFTFR,INTERP,LAM,SCALE,ROT,XSH,YSH,
     :     SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ONX,ONY)

      IF(GOTWCS) THEN
         CALL UWCS(IDND,WCSOUT,CTYPE1,CTYPE2)
      ENDIF

      CALL UIMCLO(IDND,ISTAT)
      CALL UDMFRE(PNDAT,6,ISTAT)

C Close down everything and exit
      CALL UDMFRE(PDATA,6,ISTAT)

C If the distortion images have been used free that too
      IF(PXG.NE.0) CALL UDMFRE(PXG,6,ISTAT)
      IF(PYG.NE.0) CALL UDMFRE(PYG,6,ISTAT)

 99   CONTINUE
      END

      SUBROUTINE BUHEAD(ID,VERS,DATA,OUTDAT,COEFFS,XGEOIM,YGEOIM,
     :      SHFTUN,MISVAL,
     :      EXPKEY,EXPSTR,ALIGN,INUN,OUTUN,
     :      SHFTFR,INTERP,LAM,SCALE,ROT,XSH,YSH,
     :      SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,OUTNX,OUTNY)
C
C Update the header of the output image with all the
C parameters of the current BLOT run.
C
C Updated in December 2001 to include the secondary geometric parameters
C
      IMPLICIT NONE

      INTEGER ID,OUTNX,OUTNY,ISTAT
      CHARACTER*80 DATA
      CHARACTER*80 OUTDAT,COEFFS,XGEOIM,YGEOIM
      CHARACTER*7 INTERP
      CHARACTER*8 SHFTFR,SHFTUN,EXPKEY,ALIGN,INUN,OUTUN
      CHARACTER*80 EXPSTR
      CHARACTER*(*) VERS
      DOUBLE PRECISION LAM,SCALE,ROT
      DOUBLE PRECISION DROT,XSH,YSH,XSHI,YSHI
      REAL MISVAL

C Secondary geometrical parameters, added in V1.5
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*8 SHFR2
      LOGICAL SECPAR

C Constants
      DOUBLE PRECISION PI
      PARAMETER (PI=3.1415926536)

      CALL UHDAST(ID,'BLOTVER',VERS,
     :            'Blot, task version',0,ISTAT)
      
      CALL UHDAST(ID,'BLOTDATA',DATA(1:64),
     :            'Blot, input data image',0,ISTAT)

      CALL UHDAST(ID,'BLOTOUTDA',OUTDAT(1:64),
     :            'Blot, output data image',0,ISTAT)

      CALL UHDAST(ID,'BLOTCOEFF',COEFFS(1:64),
     :            'Blot, coefficients file name ',0,ISTAT)

      CALL UHDAST(ID,'BLOTXGIM',XGEOIM(1:64),
     :            'Blot, X distortion image name ',0,ISTAT)

      CALL UHDAST(ID,'BLOTYGIM',YGEOIM(1:64),
     :            'Blot, Y distortion image name ',0,ISTAT)

      CALL UHDASD(ID,'BLOTLAM',LAM,
     : 'Blot, wavelength applied for transformation (nm)',
     :  0,ISTAT)

      CALL UHDASD(ID,'BLOTSCALE',SCALE,
     : 'Blot, scale (pixel size) of input image',0,ISTAT)

C Convert the rotation angle back to degrees
      DROT=ROT/PI*180.0D0

      CALL UHDASD(ID,'BLOTROT',DROT,
     :  'Blot, rotation angle, degrees clockwise',0,ISTAT)

C Check the SCALE and units
      IF(SHFTUN.EQ.'input') THEN
         XSHI=XSH/SCALE
         YSHI=YSH/SCALE
      ELSE
         XSHI=XSH
         YSHI=YSH
      ENDIF

      CALL UHDASD(ID,'BLOTXSH',XSHI,
     :  'Blot, X shift applied',0,ISTAT)

      CALL UHDASD(ID,'BLOTYSH',YSHI,
     :  'Blot, Y shift applied',0,ISTAT)

      CALL UHDAST(ID,'BLOTSHFU',SHFTUN,
     : 'Blot, units used for shifts (output or input pixels)',
     :            0,ISTAT)

      CALL UHDAST(ID,'BLOTINUN',INUN,
     : 'Blot, units of input image (counts or cps)',
     :            0,ISTAT)

      CALL UHDAST(ID,'BLOTOUUN',OUTUN,
     : 'Blot, units for output image (counts or cps)',
     :            0,ISTAT)

      CALL UHDAST(ID,'BLOTSHFF',SHFTFR,
     : 'Blot, frame in which shifts were applied',0,ISTAT)

      CALL UHDASI(ID,'BLOTONX',OUTNX,
     :  'Blot, X size of output image',0,ISTAT)

      CALL UHDASI(ID,'BLOTONY',OUTNY,
     :  'Blot, Y size of output image',0,ISTAT)

      CALL UHDAST(ID,'BLOTINT',INTERP,
     :  'Blot, interpolation method used',0,ISTAT)

      CALL UHDASR(ID,'BLOTFILVA',MISVAL,
     :  'Blot, value assigned to invalid pixels',0,ISTAT)

      CALL UHDAST(ID,'BLOTEXKY',EXPKEY,
     :  'Blot, exposure time keyword in input header',0,ISTAT)

      CALL UHDAST(ID,'BLOTEXOU',EXPSTR,
     :  'Blot, exposure time for output image',0,ISTAT)

C If there are secondary parameters add these too
      IF(SECPAR) THEN
         CALL UHDASB(ID,'BLOTSECP',.TRUE.,
     :  'Blot, there are secondary geometric parameters',0,ISTAT)

         CALL UHDASD(ID,'BLOTXSCL',XSCALE,
     :  'Blot, Secondary X scale applied',0,ISTAT)

         CALL UHDASD(ID,'BLOTYSCL',YSCALE,
     :  'Blot, Secondary Y scale applied',0,ISTAT)

         CALL UHDASD(ID,'BLOTXSH2',XSH2,
     :  'Blot, Secondary X shift applied',0,ISTAT)

         CALL UHDASD(ID,'BLOTYSH2',YSH2,
     :  'Blot, Secondary Y shift applied',0,ISTAT)

C Convert the rotation angle back to degrees
         DROT=ROT2/PI*180.0D0

         CALL UHDASD(ID,'BLOTROT2',DROT,
     :  'Blot, secondary rotation angle, degrees anticlockwise',
     :   0,ISTAT)
  
         CALL UHDAST(ID,'BLOTSFF2',SHFR2,
     : 'Blot, frame in which secondary shifts were applied',
     :            0,ISTAT)

      ELSE
         CALL UHDASB(ID,'BLOTSECP',.FALSE.,
     :  'Blot, there are no secondary geometric parameters',0,ISTAT)
      ENDIF

      RETURN
      END

      SUBROUTINE BALLME(DNX,DNY,ONX,ONY,NSX,NSY,PBUFF,
     :            PDATA,PNDAT,PXIN,PYIN,PXOUT,PYOUT,ISTAT)
C
C Allocate the dynamic memory arrays needed for Blot.
C
C This routine was added and the corresponding calls removed from
C the main module, December 2002
C
      IMPLICIT NONE

      INTEGER DNX,DNY,ONX,ONY,NSX,NSY
      INTEGER PDATA,PNDAT,PBUFF
      INTEGER PXIN,PYIN,PXOUT,PYOUT
      INTEGER ISTAT

C Set all the pointers to zero to start with - if they stay that
C way we can assume the allocation has failed
      PDATA=0
      PNDAT=0
      PBUFF=0
      PXIN=0
      PYIN=0
      PXOUT=0
      PYOUT=0

C Allocate the memory arrays and return the pointers, check for
C error status
      CALL UDMGET(NSX*NSY,6,PDATA,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(ONX*ONY,6,PNDAT,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(DNX,6,PBUFF,ISTAT)

      CALL UDMGET(ONX*4,7,PXIN,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(ONX*4,7,PYIN,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(ONX*4,7,PXOUT,ISTAT)
      IF(ISTAT.NE.0) RETURN

      CALL UDMGET(ONX*4,7,PYOUT,ISTAT)
      IF(ISTAT.NE.0) RETURN
      
      ISTAT=0
      RETURN
      END
