      SUBROUTINE WDRIZE
C++
C
C WDRIZZLE.F Version 3.4 - Variable Pixel Linear Reconstruction
C
C Also widely known as "drizzling"
C
C This task implements a simple method for transforming one image
C onto the pixel grid of another. The method calculates where the
C input pixels map on the output grid and performs an optimal weighted
C combination of the input and output. Another image records the weighting
C at a given pixel.
C
C It is based largely on ideas of Andy Fruchter and was coded by Richard Hook. 
C (Contacts: rhook@eso.org & fruchter@stsci.edu)
C
C This task uses coefficients held in a text file to define
C geometrical distortions.
C
C This code uses the F77/VOS for IRAF i/o.
C It may be compiled and linked using a command like:
C
C xc -p stsdas (-z) drizzle.f drutil.f x_drizzle.x <dir>libiraf77.a
C
C Note that the optional -z switch (which tells IRAF to create a static 
C executable which doesn't use the IRAF sharable library) may be
C necessary to avoid a virtual memory use limitation of around 167MB in
C IRAF versions up to 2.10.4.
C
C History:
C
C First try, Richard Hook, ST-ECF, October 1995 
C Generalised coefficients added, RNH, Nov 95
C Working on V0.2, RNH, Nov 1995 (including "small|large" pixels)
C Modified again for continuously variable input pixel area, 29/11/95
C Modified again for X,Y position processing, 1/12/95
C Started work on V0.3 including "update" option, more header
C information and WCS, 12/12/95
C V0.3 released, 15th December 1995
C V0.31 modifications following comments from STScI, 27/12/95
C V0.32 modifications, 29/12/95
C V0.33 modifications, STScI, 5th Jan 1996
C V0.34 modifications, try to minimise memory use, 7th Jan 1996
C V0.35 tidyup and a few enhancements, 9th January 1996
C V0.36 correct rotation angle in output and also case
C       where outdat=olddat, 8th February 1996
C V0.37 correct silly rotation angle multiplication error
C       29th February 1996
C V0.40 first pre-release version, tidied up and corrected WCS
C       bug with CTYPE header items. 25th March 1996
C V0.5  started to incorporate the "boxer" code for linear
C       combination by precise area weighting rather than by the
C       drizzling approximation. The "boxer" code was originally
C       written by Bill Sparks, STScI, 19th April 1996
C V0.9  many simpifications and improvements for a pre-release
C       version. This removes drizzling and the XY option as well
C       as many options which weren't used. 16th May 1996
C V0.91 corrected typo when shunits=output. 21st May 1996     
C V0.92 changed parameter names and removed "olddata" option. 22nd May 1996
C V0.93 changed to separate coeffients files for each coefficients
C       set. 22nd May 1996
C V0.94 changed access to coefficients file to use IRAF filenames via
C       (hidden) F77 VOS calls and also put in a scale for the weights
C       image and changed the names. ST-ECF, 30th May 1996
C V0.95 bug fix in error message within SGAREA, format statement syntax
C       error. ST-ECF, 20th June 1996
C V0.96 name change to DRIZZLE. ST-ECF, 5th July 1996
C V0.97 added exposure time keyword handling. ST-ECF, 9th July 1996
C V0.98 added ability NOT to store output counts image, 16th July 1996
C V0.99 added flag for scaling by exposure time, 16th July 1996
C V0.991 added "fillval" parameter, option for wt_scl=exptime, 29th July 1996
C V0.992 slight change to fillval interpretation, 31st July 1996
C V0.993 small bugfix when reading coefficients, 28th August 1996
C
C V1.0 release. 17th September 1996.
C
C V1.01 - some bug fixes after first release, 19th September 1996
C V1.02 - same as V1.01 except permissions of tar file altered, 3rd October 1996
C V1.03 - small change to WCS logic to avoid crashes, 25th October 1996
C V1.08 - collection of several small changes, 5th December 1996
C         (including a change in the definition of the centres and the
C         addition of the out_typ parameter))
C V1.09 - correct scaling of offsets bug, 16th December 1996
C V1.091 - introduce Jacobian for weights, 10th January 1997
C V1.092 - correct header update bug, 21st January 1997
C V1.093 - put utility routines in separate file (drutil.f), 4th February 1997
C V1.094 - change disk access to minimise memory use, 10th February 1997
C V1.095 - introduced automatic coeffs/lambda from header, 7th March 1997
C V1.096 - added tidier memory management in error cases, 10th March 1997
C          also better file closure on error trapping.
C V1.097 - bugfix to V1.096, 12th March 1997
C V1.098 - pre-release version incorporating EXPSQ weighting, modified
C          error messages and a maximum of 999 images, 22nd April 1997
C
C V1.1 - second public release. 12th May 1997, STScI
C
C V1.2 - first release within STSDAS dither package, this version uses
C        the in_un and out_un system for defining whether input/outputs
C        are in count or counts/s. The old parameter exp_sc has been
C        removed. 20th February 1998.
C
C V1.3 - added support for WCS-driven geometrical mapping. 18th September 1998
C
C V1.4 - added image and memory sub-setting to reduce i/o and memory usage.
C        5th October 1998.
C
C V1.41 - version of V1.4 without WCS option and returning to the parameter
C         file of V1.2 for compatibility. 13th January 1999.
C
C V1.5D - development version, July 2000
C         secondary linear transforms added as a test, 24th July 2000
C
C V1.6D - some restructuring of the code and the addition of multiple
C         kernel support at a low level, September 2000
C
C V1.7D - continuing code restructuring and additions, October 2000
C
C V1.8D - adding context image code, October 2000
C
C V2.0beta - test version of V2.0 release - October 16th 2000
C
C V2.1beta - starting further modifications for ACS including
C            other possible distortion representations.
C            December 19th 2000
C
C V2.2beta - including "bitmask" style contexts, January 2001
C            convert all integer arrays to *4
C
C V2.3test - version for ACS pipeline. Includes better WCS handling
C            and some bug fixes for problems related to ACS.
C                    Richard Hook, STScI, 15th February 2001
C
C V2.4test - correct bugs, in particular related to secondary
C            geometric parameters. 
C            21st March 2001
C
C V2.5test - make copying all of the header of the first image to the
C            output optional (currently it IS copied). May 2001
C
C V2.5beta - transfer all auxiliary routines to drutil.f, change to
C            V2.5beta for more extensive testing.
C               Richard Hook, ST-ECF, 24th May 2001
C
C V2.6 - renamed to Drizzle again for release at STScI.
C               Richard Hook, ST-ECF, 16th October 2001
C
C V2.62 - modified the UPWCS logic in drutil.f to correct
C         a long-standing problem with the WCS update
C               Richard Hook, ST-ECF, 29th January 2002
C
C V2.7 - new option and calling sequence for DRIVAL to allow "regular" mode
C        for better precision and speed.
C        Also reverted to previous logic for UPWCS.
C               Richard Hook, ST-ECF, 27th February 2002
C
C V2.8 - new "lanczos" optimal interpolation kernel support added
C          Richard Hook, ST-ECF/STScI, 6th August 2002
C
C V2.81 - modified order of memory freeing and file closing to
C         allow working with very large mosaics. Also corrected bug
C         when updating subsets of large weight and context maps.
C          Richard Hook, ST-ECF/STScI, 5th September 2002
C
C V2.9 - more accurate WCS use in wdrizzle mode.
C          Richard Hook, ST-ECF/STScI, 5th November 2002
C
C V2.91 - change to the "gaussian" kernel box size to avoid
C         holes and smaller "lanczos" working box for improved
C         performance.
C
C V2.92 - added check for zero input exposure time.
C          Richard Hook, ST-ECF/STScI, 30th January 2003
C
C V2.93 - added checks for lines of input image falling outside
C         the output image and for partial overlaps
C          Richard Hook, ST-ECF/STScI, 2nd April 2003
C
C V2.94 - modified the logic for calculating the output WCS
C         to try to correct some inaccuracy found in earlier
C         versions. If COTY>0 (ie, polynomial distortion) then
C         the linear part is now combined with the input CD to
C         create the output.
C
C V3.0 - went back to old way of doing WCS calculation for drizzle
C        but using double precision. Added reference pixel specification
C        and changed the parameter names.
C          Richard Hook, ST-ECF/STScI, 13th June 2003
C
C V3.03 - corrected logic for wdrizzle photometric correction.
C
C V3.1 - new geometric distortion image support
C
C V3.2 - extra output header information
C
C V3.2D - double precision version for tests and UDF,
C               Richard Hook, ST-ECF/STScI, 21st October 2003
C
C V3.3 - added support for "refpix" in coefficients files so that the
C        the origin of the distortion is decoupled from the center of the
C        chip.
C               Richard Hook, ST-ECF/STScI, 4th December 2003
C
C V3.31 - used LINEAR part of distortion for determining output
C  WCS in Drizzle and also for WDRIZZLE transformation finding.
C               Richard Hook, ST-ECF/STScI, 9th December 2003
C
C V3.32 - made the delta in WCSLIN smaller to test whether this is
C         the source of a small skew error we see.
C               Richard Hook, ST-ECF/STScI, 19th April 2004
C
C V3.33 - added reference pixel area correction for SIP mode.
C               Richard Hook, ST-ECF/STScI, 3rd June 2004
C
C V3.4 - modified distortion images to be passed as arrays instead of 
C        as pointers, and modified the code to use the same routines for
C        both the callable interface and STSDAS interface. Based on 
C        Version 3.32 from Richard.
C              Warren Hack, STScI, 25th June 2004
C V3.4.1- added correction for 'wdrizzle' bug and Jacobian bug reported
C        by M. Kuemmel. Warren Hack, STScI, 24 Feb 2006
C V3.4.2 - The center of the image is a floating point calculation now, 
C        not int - this change was made in an attempt to achieve agreement
C        between pydizzle and wdrizzle
C         Nadia Dencheva 3 July 2006
C--
      IMPLICIT NONE

C Iraf global storage (real in this program)
      REAL MEMR(1) !I
      DOUBLE PRECISION MEMD(1) !I
      INTEGER MEMI(1) !I
      COMMON /MEM/MEMR         !I
      EQUIVALENCE (MEMR,MEMI)  !I
      EQUIVALENCE (MEMR,MEMD)  !I

C Local variables
      INTEGER ISTAT,I,J,JNY,NMISS,NSKIP
      INTEGER XMIN,XMAX,YMIN,YMAX,YSTART
      INTEGER NDIMS,DNX,DNY,ONX,ONY,DDIMS(7),ODIMS(7),DATTYP
      INTEGER IDD,IDW,IDND,IDNC,IDCO,IC
      INTEGER PDATA,PWEI,PNDAT,PNCOU,PNCON,PCDON,PBUFF,PIBUFF
      INTEGER PXI,PXO,PYI,PYO
      INTEGER PXIB,PXOB,PYIB,PYOB
      INTEGER COTY,CONUM,COMAX,PXG,PYG,XGDIM,YGDIM
      PARAMETER (COMAX=100)
      DOUBLE PRECISION XCO(COMAX),YCO(COMAX),XSH,YSH,LAM
      DOUBLE PRECISION SCALE,ROT,PFRACT
      REAL EXPIN,EXPOUT,EXPCUR,FILVAL,FS
      DOUBLE PRECISION LASTPF,LASTSC,LASTOX,LASTOY,OX,OY
      DOUBLE PRECISION OUTSCL,ORIENT
      DOUBLE PRECISION WCS(8)
      DOUBLE PRECISION WCSOUT(8),RACEN,DECCEN,XREFP,YREFP
      REAL WTSCL
      CHARACTER*80 DATA,WEIGHT,OUTDAT,OUTCOU,CHARS,XGEOIM,YGEOIM
      CHARACTER*80 COEFFS,SHFTUN,WTSTR,FILSTR
      CHARACTER*45 VERS
      CHARACTER*8 CTYPE1,CTYPE2,LASTUN,CT1,CT2
      CHARACTER*8 EXPKEY,SHFTFR,INUN,OUTUN,ALIGN
      CHARACTER*8 GEOMOD
      CHARACTER*8 KERNEL

C Context related variables and parameters
      CHARACTER*80 CONTIM,CONTAB
      INTEGER NEN,I1,I2
      INTEGER UNIQID

C This parameter controls the maximum number of entries in
C the context table. This value is rather arbitrary
      INTEGER MAXEN
      PARAMETER (MAXEN=1000)

C The following parameter controls the number of images which
C may be combined when contexts are in use. 
      INTEGER MAXIM
      PARAMETER (MAXIM=128)

C This parameter controls the context style
      LOGICAL BITCON
      PARAMETER (BITCON=.TRUE.)

C This parameter is the maximum number of contexts which are stored
C in the header of the context image. If there are more then they are
C written to a separate file
      INTEGER MAXHCN
      PARAMETER (MAXHCN=20)

      INTEGER INTAB(MAXIM,MAXEN)
      LOGICAL CON

C The maximum desirable array size (for about 64Mb memory use)
      INTEGER MAXPIX
      PARAMETER (MAXPIX=2e7)

C Logical flags
      LOGICAL VERBOSE
      LOGICAL USEWEI
      LOGICAL USEWCS
      LOGICAL UPDATE
      LOGICAL GOTWCS
      LOGICAL WRIWEI
      LOGICAL FILL
      LOGICAL ROTFIR
      LOGICAL INCPS
      LOGICAL OUTCPS
      LOGICAL CURCPS
      LOGICAL SUB
      LOGICAL NOOVER
      LOGICAL ODD
      LOGICAL ODW
      LOGICAL ODND
      LOGICAL ODNC
      LOGICAL ODCO
      LOGICAL DISIM

C Secondary geometrical parameters, added in V1.5
      DOUBLE PRECISION XSH2,YSH2,ROT2,XSCALE,YSCALE
      CHARACTER*8 SHFR2
      LOGICAL SECPAR
      LOGICAL ROTF2
      LOGICAL COPALL

C-- Start of executable code

C First initialise the logical flags
      VERBOSE=.TRUE.
      ODCO=.FALSE.
      ODNC=.FALSE.
      ODND=.FALSE.
      COPALL=.TRUE.

      VERS='WDRIZZLE Version 3.4.2 (Jul 3rd 2006)'

C Announce the version
      CALL UMSPUT('+ '//VERS,1,0,ISTAT)

C Do force geomode to be user (this is the only difference between
C WDRIZZLE and DRIZZLE, apart from the name)
CCC      GEOMOD='user'
      GEOMOD=' '

C Get all the parameter values without verifying
      CALL GETPAR(GEOMOD,KERNEL,PFRACT,COEFFS,LAM,XGEOIM,YGEOIM,
     :                  SCALE,ROT,XSH,YSH,SHFTFR,SHFTUN,ALIGN,
     :                  XSCALE,YSCALE,XSH2,YSH2,ROT2,SHFR2,
     :                  EXPKEY,WTSTR,FILSTR,INUN,OUTUN,
     :                  DATA,WEIGHT,OUTDAT,OUTCOU,CONTIM,ONX,ONY,
     :                  OUTSCL,RACEN,DECCEN,XREFP,YREFP,ORIENT)

C Find out whether we are using WCS to define the geometry
      IF(GEOMOD.EQ.'wcs') THEN
         USEWCS=.TRUE.
      ELSE
         USEWCS=.FALSE.
      ENDIF

C Check for small values and avoid zero values
      IF(PFRACT.LT.0.001D0) KERNEL='point'
      IF(KERNEL.EQ.'point') PFRACT=0.001D0

C Check the switch value for whether to read in CPS or
C counts (latter is the default)
      IF(INUN.EQ.'cps') THEN
         INCPS=.TRUE.
      ELSE
         INCPS=.FALSE.
      ENDIF

C Check the switch value for whether to write out in CPS or
C counts (latter is the default)
      IF(OUTUN.EQ.'cps') THEN
         OUTCPS=.TRUE.
      ELSE
         OUTCPS=.FALSE.
      ENDIF

C Check the parameter specifying which frames shifts are applied
      IF(SHFTFR.EQ.'input') THEN
         ROTFIR=.FALSE.
      ELSE
         ROTFIR=.TRUE.
      ENDIF

C Check for meaningful values
      IF(FILSTR.NE.'INDEF'.AND.FILSTR.NE.'indef') THEN
         READ(FILSTR,*,IOSTAT=ISTAT) FILVAL
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Invalid filling value specified',
     :                  1,0,ISTAT)
            GO TO 99
         ENDIF
         FILL=.TRUE.
      ELSE
         FILL=.FALSE.
      ENDIF

C Check for invalid scale
      IF(SCALE.EQ.0.0D0) THEN
         CALL UMSPUT('! Invalid scale',1,0,ISTAT)
         GO TO 99
      ENDIF

C Convert the rot to radians
      ROT=ROT*3.14159265358979D0/180.0D0

C Convert the shift units if necessary
      IF(SHFTUN.EQ.'output') THEN
         XSH=XSH*SCALE
         YSH=YSH*SCALE
      ENDIF

C Note that the shifts we now have are INPUT ones

C Convert the secondary parameters into suitable units
      ROT2=ROT2*3.14159265358979D0/180.0D0
      IF(SHFR2.eq.'input') THEN
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

C Check for compatible requests
      IF(EXPKEY.EQ.' ') THEN
         CALL UMSPUT('! No exposure time keyword has been given',
     :  1,0,ISTAT)
         GO TO 99
      ENDIF

C Try to access the input data image
      CALL UIMOPN(DATA,1,IDD,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Unable to open data image',
     :                    1,0,ISTAT)
         ODD=.FALSE.
         GO TO 99
      ELSE
         IF(VERBOSE) CALL UMSPUT(
     :  '-Opening input data image: '//DATA,
     :                   1,0,ISTAT)
         ODD=.TRUE.
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

C Get the geometric distortion coefficient information
      CALL GETGEO(COEFFS,IDD,LAM,
     :            COTY,COMAX,CONUM,XCO,YCO,ISTAT)
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

C Get the input exposure time from the header
      CALL UHDGSR(IDD,EXPKEY,EXPIN,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT(
     :    '! Error, unable to read exposure time keyword value',
     :         1,0,ISTAT)
         CALL UMSPUT(
     :    '! Please add one to the image header and try again.',
     :         1,0,ISTAT)
         GO TO 99
      ENDIF

C Check for zero exposure time (yes, it does happen)
      IF(EXPIN.LE.0.0) THEN
         CALL UMSPUT(
     : '! Warning, exposure time <= 0 - setting to 1.0',1,0,ISTAT)
         EXPIN=1.0
      ENDIF

C The scale is read as a string as it might be "exptime"
C in which case we use the exposure time - if we have it
C We also can have weighting by exposure time squared (eg,
C for read-noise dominated images)
      IF(WTSTR.EQ.'exptime' .OR. WTSTR.EQ.'EXPTIME') THEN
         WTSCL=EXPIN
      ELSE IF(WTSTR.EQ.'expsq' .OR. WTSTR.EQ.'EXPSQ') THEN
         WTSCL=EXPIN**2
      ELSE
         READ(WTSTR,*,IOSTAT=ISTAT) WTSCL
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Weighting scale is not a number',
     :                  1,0,ISTAT)
            GO TO 99
         ENDIF
      ENDIF

      IF(WEIGHT.EQ.' ') THEN
         USEWEI=.FALSE.
         ODW=.FALSE.
      ELSE
         CALL UIMOPN(WEIGHT,1,IDW,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Unable to open mask image',
     :                       1,0,ISTAT)
            ODW=.FALSE.
            GO TO 99
         ELSE
            IF(VERBOSE) CALL UMSPUT(
     :   '-Opening mask image: '//WEIGHT,
     :                      1,0,ISTAT)
            ODW=.TRUE.

C Get the array size and shape of the mask image - must
C be the same as the data
            CALL UIMGID(IDW,DATTYP,NDIMS,DDIMS,ISTAT)
            IF(NDIMS.NE.2 .OR. DDIMS(1).NE.DNX .OR.
     :         DDIMS(2).NE.DNY) THEN
               CALL UMSPUT('! Mask image is not the same shape '//
     :          'and size as data',1,0,ISTAT)
               GO TO 99
            ENDIF

            USEWEI=.TRUE.
         ENDIF
      ENDIF

C If the output weighting image is null we just don't
C create anything and don't write anything out
      IF(OUTCOU.EQ.' ') THEN
         WRIWEI=.FALSE.
      ELSE
         WRIWEI=.TRUE.
      ENDIF

C Similarly for the context image
      IF(CONTIM.EQ.' ') THEN
         CON=.FALSE.
      ELSE

C Prepare a context table file name, just in case we need it later
         CALL LENSTR(CONTIM,I1,I2)
         CONTAB=CONTIM(I1:I2)//'.CON'
         CON=.TRUE.
      ENDIF

C It may be that the "new" image currently exists and we
C want to update it by drizzling the new data on top
      IF(OUTDAT.EQ.' ') THEN
         CALL UMSPUT(
     :       '! No output data array name specified',
     :                   1,0,ISTAT)
           GO TO 99
      ENDIF

C We try to open it with update access
      CALL UIMOPN(OUTDAT,2,IDND,ISTAT)
      IF(ISTAT.NE.0) THEN
         ODND=.FALSE.
         UPDATE=.FALSE.

         ODIMS(1)=ONX
         ODIMS(2)=ONY

C If not updating and using WCS... 
         IF(USEWCS) THEN

C Now convert these into WCS style representation
            CALL SETWCS(OUTSCL,ORIENT,XREFP,RACEN,
     :                  YREFP,DECCEN,WCSOUT)
         ENDIF
      ELSE

         CALL UMSPUT(
     :    '-Opening extant data image: '//OUTDAT,
     :                1,0,ISTAT)
         ODND=.TRUE.
 
C Now we try to open the corresponding weighting image
         CALL UIMOPN(OUTCOU,2,IDNC,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     :     '! Unable to open extant weighting image',
     :                 1,0,ISTAT)
            ODNC=.FALSE.
            GO TO 99
         ENDIF

         CALL UMSPUT(
     :       '-Opening extant weighting image: '//OUTCOU,
     :                   1,0,ISTAT)
         ODNC=.TRUE.

C We are doing well and probably can use the existing
C images however we need to do a little more...
C Get the array size and shape of old data frame
         CALL UIMGID(IDND,DATTYP,NDIMS,ODIMS,ISTAT)

         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     : '! A problem was encountered accessing existing image: '//
     : OUTDAT,1,0,ISTAT)
            CALL UMSPUT(
     : '! Please delete or recreate it and try again',
     :      1,0,ISTAT)
            GO TO 99
         ENDIF

         IF(NDIMS.NE.2) THEN
            CALL UMSPUT('! Extant image is not two dimensional',
     :                       1,0,ISTAT)
            GO TO 99
         ELSE
            ONX=ODIMS(1)
            ONY=ODIMS(2)
         ENDIF

C Also check that the extant weighting image is the same size
         CALL UIMGID(IDNC,DATTYP,NDIMS,ODIMS,ISTAT)

         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     : '! A problem was encountered accessing existing image: '//
     : OUTDAT,1,0,ISTAT)
            CALL UMSPUT(
     : '! Please delete or recreate it and try again',
     :      1,0,ISTAT)
            GO TO 99
         ENDIF

         IF(NDIMS.NE.2) THEN
            CALL UMSPUT(
     :   '! Current weighting image is not two dimensional',
     :                    1,0,ISTAT)
            GO TO 99
         ELSE IF(ODIMS(1).NE.ONX .OR. ODIMS(2).NE.ONY) THEN
            CALL UMSPUT('! Weighting image not the same size as data',
     :                    1,0,ISTAT)
            GO TO 99
         ENDIF

C We can now set the update flag
         UPDATE=.TRUE.
         WRIWEI=.TRUE.

         IF(USEWCS) THEN

C Try to get the WCS of the current image
            CALL GETWCS(IDND,WCSOUT,CT1,CT2,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT('! Unable to get WCS from extant image',
     :         1,0,ISTAT)
               GO TO 99
            ENDIF

            CALL UMSPUT(
     : '! Note: using WCS of extant image to define geometry',
     :                  1,0,ISTAT)
            SHFTUN='output'
            SHFTFR='output'
         ENDIF

C Get the current exposure time from the header
         CALL UHDGSR(IDND,EXPKEY,EXPCUR,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     : '! Warning, failed to get exposure time keyword value',
     :      1,0,ISTAT)
            GO TO 99
         ENDIF              

C Check for zero exposure time (yes, it does happen)
         IF(EXPCUR.LE.0.0) THEN
            CALL UMSPUT(
     : '! Warning, exposure time <= 0 - setting to 1.0',1,0,ISTAT)
            EXPCUR=1.0
         ENDIF

C We now get some of the previous drizzle information from
C the header of the current image and check for funny
C combinations
         CALL GOLDH(IDND,USEWCS,IC,LASTPF,LASTSC,
     :                 LASTUN,LASTOX,LASTOY,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     : '! Warning, unable to read old drizzle header keywords',
     :      1,0,ISTAT)
            CALL UMSPUT('! assuming current image is in counts',
     :      1,0,ISTAT)
            CURCPS=.FALSE.
         ELSE
            IF(LASTUN.EQ.'counts' .OR. LASTUN.EQ.'COUNTS') THEN
               CURCPS=.FALSE.
            ELSE
               CURCPS=.TRUE.
            ENDIF

            IF(LASTPF.NE.PFRACT) THEN
               CALL UMSPUT(
     : '! Warning, current pixfrac does not match previous one',
     :         1,0,ISTAT)
            ENDIF

            IF(LASTSC.NE.SCALE .AND. .NOT.USEWCS) THEN
               CALL UMSPUT(
     : '! Warning, current scale does not match previous one',
     :         1,0,ISTAT)
            ENDIF

            IF(ALIGN.EQ.'corner') THEN
               OX=DBLE(ONX/2.0)+0.5D0
               OY=DBLE(ONY/2.0)+0.5D0
            ELSE
               OX=DBLE(ONX/2.0)+1.0D0
               OY=DBLE(ONY/2.0)+1.0D0
            ENDIF

            IF(.NOT.USEWCS) THEN
               IF(LASTOX.NE.OX .OR. LASTOY.NE.OY) THEN
                  CALL UMSPUT(
     : '! Warning, current reference pixel alignment '//
     : 'does not match previous one',
     :            1,0,ISTAT)
               ENDIF
            ENDIF
         ENDIF 
      ENDIF

C Set a reasonable running image number
      IF(.NOT.UPDATE) IC=0

C If we are using the context we need to get the unique image
C identifier from the data file
      IF(CON) THEN

C The image ID is the same as the count of images drizzled
            UNIQID=IC+1

C Open the context image and create a bigger one if necessary
         IF(UPDATE) THEN
            CALL OPCON(CONTIM,ONX,ONY,IDCO,ODCO,UNIQID,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT('! Failed to open context image',
     :         1,0,ISTAT)
               GO TO 99
            ENDIF
         ENDIF

C If we have a 32 bit context image then check this is in
C range
         IF(.NOT.BITCON) THEN
            IF(UNIQID.LT.1 .OR. UNIQID.GT.MAXIM) THEN
               CALL UMSPUT(
     :  '! Image id number out of range for context',
     :         1,0,ISTAT)
               GO TO 99
            ENDIF
         ENDIF
      ENDIF
 
C Also try to get the header items for the WCS, note that
C this is not the full WCS but just 8 header items commonly used
C for HST images plus CTYPE1 & CTYPE2
      GOTWCS=.FALSE.
      CALL GETWCS(IDD,WCS,CTYPE1,CTYPE2,ISTAT)
      IF(ISTAT.EQ.0) THEN
         GOTWCS=.TRUE.
      ELSE
         IF(.NOT.UPDATE) THEN
            CALL UMSPUT(
     :    '! Warning, unable to read WCS information from header',
     :    1,0,ISTAT)
         ENDIF
      ENDIF
 
C Here we set the WCS matrix to something sensible anyway
      IF(.NOT.GOTWCS) THEN
         CTYPE1='PIXELS'
         CTYPE2='PIXELS'
 
         DO I=1,8
            WCS(I)=0.0D0
         ENDDO

C These were added in V1.5D so that a sensible matrix for inversion
C was available for later - although a fictional one
         WCS(5)=-1.0D0
         WCS(8)=1.0D0
      ENDIF

C Calculate the range of pixels in the output which are affected
C by the input image - this is the new subsetting code which was
C added in V1.4
      CALL LCORN(DNX,DNY,ONX,ONY,
     :           XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :           SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :           USEWCS,WCS,WCSOUT,
     :           COTY,CONUM,XCO,YCO,
     :           DISIM,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,
     :           XMIN,XMAX,YMIN,YMAX,ISTAT)
 
C Assume we are using the whole frame unless set otherwise
      SUB=.FALSE.
      NOOVER=.FALSE.

C Announce which image this is
      IF(IC.EQ.0) THEN
         CALL UMSPUT(
     : '-This is the first image to be drizzled onto this output',
     :   1,0,ISTAT)
      ELSE
         WRITE(CHARS,
     : '(''-This is image # '',I3,
     :   '' to be drizzled onto this output'''
     :    //')') IC+1
         CALL UMSPUT(CHARS,1,0,ISTAT)
      ENDIF
 
C Check for no-overlap
      IF(XMAX.LT.1 .OR. YMAX.LT.1 .OR.
     :      XMIN.GT.ONX .OR. YMIN.GT.ONY) THEN
         CALL UMSPUT(
     : '! Warning, there is no overlap with the output image',
     :  1,0,ISTAT)
            WRITE(CHARS,
     : '(''! Output pixel range would be: ['',I7,'':'',I7,
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
         XMAX=MIN(ONX,XMAX)
         YMIN=MAX(1,YMIN)
         YMAX=MIN(ONY,YMAX)

C Check to see whether we really should use the subset or
C whether it makes more sense to work on the full frame.
C These criteria are a little arbitrary at this stage
         IF(DBLE(YMAX-YMIN+1)/DBLE(ONY).GT.0.5 .AND.
     :      ONX*ONY.LT.MAXPIX .OR.
     :      (XMIN.EQ.1 .AND. XMAX.EQ.ONX .AND. 
     :      YMIN.EQ.1 .AND. YMAX.EQ.ONY)) THEN

            SUB=.FALSE.
            WRITE(CHARS,
     : '(''-Drizzling onto full output image. Kernel: '',A8)')
     :      KERNEL

            XMIN=1
            YMIN=1
            XMAX=ONX
            YMAX=ONY
            SUB=.FALSE.
         ELSE
            WRITE(CHARS,
     : '(''-Drizzling onto subset: ['',I5,'':'',I5,
     :      '','',I5,'':'',I5,''] ('',F6.2,''%). Kernel: '',A8)') 
     :       XMIN,XMAX,YMIN,YMAX,
     :       100.0*DBLE((XMAX-XMIN+1)*(YMAX-YMIN+1))/
     :       DBLE(ONX*ONY),KERNEL
            SUB=.TRUE.
         ENDIF

         CALL UMSPUT(CHARS,1,0,ISTAT)
      ENDIF

C If there is no extant output image
C we will have to create output data and weighting images anyway
C so we need to find out their dimensions
C If there is no overlap we don't do this
      IF(NOOVER) THEN
         CALL UMSPUT('! No overlap - no drizzling',1,0,ISTAT)
         GO TO 99
      ELSE
       IF(.NOT.UPDATE) THEN

C If we are not updating we need to create the output images
         CALL UIMCRE(OUTDAT,6,2,ODIMS,IDND,ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     :        '! Unable to create output data image',
     :                    1,0,ISTAT)
            ODND=.FALSE.
            GO TO 99
         ELSE
            CALL UMSPUT(
     :        '-Created new output data image: '//OUTDAT,
     :                    1,0,ISTAT)
            ODND=.TRUE.
         ENDIF

C If we are going to store the counts image we must create it
         IF(WRIWEI) THEN
            CALL UIMCRE(OUTCOU,6,2,ODIMS,IDNC,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT(
     :        '! Unable to create output weighting image',
     :                    1,0,ISTAT)
               ODNC=.FALSE.
               GO TO 99
            ELSE
               CALL UMSPUT(
     :        '-Created new output weighting image: '//OUTCOU,
     :                    1,0,ISTAT)

               ODNC=.TRUE.
            ENDIF
         ELSE
            CALL UMSPUT(
     : '! Warning, output weighting image will not be saved',
     :      1,0,ISTAT)
         ENDIF

C If we are going to store the context image we must create it
         IF(CON) THEN

C We create a file with initially just one plane
C We will expand if needed later. 
            CALL UIMCRE(CONTIM,4,2,ODIMS,IDCO,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT(
     :        '! Unable to create output context image',
     :                    1,0,ISTAT)
               ODCO=.FALSE.
               GO TO 99
            ELSE
               CALL UMSPUT(
     :        '-Created new output context image: '//CONTIM,
     :                    1,0,ISTAT)

               ODCO=.TRUE.
            ENDIF
         ELSE
            CALL UMSPUT(
     : '! Warning, output context image will not be saved',
     :      1,0,ISTAT)
         ENDIF
      ENDIF

C If this is a new output then we need to copy some of the
C old header over
      IF(.NOT.UPDATE) THEN
         CALL COPHED(IDD,IDND,COPALL,ISTAT)
         IF(WRIWEI) CALL COPHED(IDD,IDNC,COPALL,ISTAT)
         IF(CON) CALL COPHED(IDD,IDCO,COPALL,ISTAT)
      ENDIF

C Update the exposure times in the outputs
       IF(UPDATE) THEN
         EXPOUT=EXPIN+EXPCUR
       ELSE
         EXPOUT=EXPIN
       ENDIF
      ENDIF

C Now we allocate space for all the arrays we will need, note that they
C are a mixture of real and double precision except for the optional 
C context image which is INTEGER
      CALL ALLMEM(DNX,DNY,ONX,XMAX-XMIN+1,YMAX-YMIN+1,CON,BITCON,
     :            PDATA,PWEI,PNDAT,PNCOU,PNCON,PCDON,PBUFF,PIBUFF,
     :            PXI,PYI,PXO,PYO,PXIB,PYIB,PXOB,PYOB,ISTAT)
      IF(ISTAT.NE.0) THEN
         CALL UMSPUT('! Unable to allocate required memory arrays',
     :               1,0,ISTAT)
         GO TO 99
      ENDIF
      YSTART = 0

C Optional weights array initialize
      IF(.NOT.USEWEI) CALL SETIM(MEMR(PWEI),DNX,1,1.0)

C If we are updating we read in the extant "new" images
      IF(UPDATE) THEN

C First the data image
         CALL GETIMR(IDND,ONX,ONY,SUB,XMIN,XMAX,YMIN,YMAX,
     :                  MEMR(PBUFF),MEMR(PNDAT),ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     :         '! Failed to read extant data image',
     :                      1,0,ISTAT)
            GO TO 99
         ENDIF

C Scale by exposure time
         IF(.NOT.CURCPS) THEN
            CALL MULC(MEMR(PNDAT),XMAX-XMIN+1,YMAX-YMIN+1,
     :                1.0/EXPCUR)
         ENDIF

C Weighting image
        CALL GETIMR(IDNC,ONX,ONY,SUB,XMIN,XMAX,YMIN,YMAX,
     :                  MEMR(PBUFF),MEMR(PNCOU),ISTAT)

         IF(ISTAT.NE.0) THEN
            CALL UMSPUT(
     :         '! Failed to read current weighting image',
     :                      1,0,ISTAT)
            GO TO 99
         ENDIF

C Context image (integers)
         IF(CON) THEN

            CALL GETIMI(IDCO,ONX,ONY,SUB,XMIN,XMAX,YMIN,YMAX,
     :                  MEMI(PIBUFF),MEMI(PNCON),ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT(
     :         '! Failed to read extant context image',
     :                      1,0,ISTAT)
              GO TO 99
            ENDIF
         ENDIF
      ELSE

C The final case is where the output is empty.
C Before we start drizzling we need to drain out the output arrays
C if there are no old arrays 
         CALL SETIM(MEMR(PNDAT),XMAX-XMIN+1,YMAX-YMIN+1,0.0)
         CALL SETIM(MEMR(PNCOU),XMAX-XMIN+1,YMAX-YMIN+1,0.0)
         IF(CON) THEN
            CALL SETIMI(MEMI(PNCON),XMAX-XMIN+1,YMAX-YMIN+1,0)
         ENDIF
      ENDIF

C If we are using a context array we need to either read its
C current values from the global context file, or, if the
C context is new, initialise it with something sensible
C
C This is read from the header of the context
C image not from an external file
C
C This is only needed for the EIS style context
      IF(CON .AND. .NOT.BITCON) THEN

         CALL SETIMI(MEMI(PCDON),XMAX-XMIN+1,YMAX-YMIN+1,0)
         IF(UPDATE) THEN
            CALL GTCOIN(IDCO,INTAB,MAXIM,MAXEN,NEN,ISTAT)

            IF(ISTAT.NE.0) THEN
               CALL UMSPUT(
     : '! Note: Valid context table not found in context image header',
     :                     1,0,ISTAT)
               NEN=0
            ENDIF
         ELSE
            NEN=0
         ENDIF
      ENDIF
C
C
C
      NMISS = 0
      NSKIP = 0
      DO J=1,DNY
C
C Read in one row of data
          CALL UIGL2R(IDD,J,MEMR(PDATA),ISTAT)
          IF(ISTAT.NE.0) THEN
             CALL UMSPUT('! Failed to read data image',
     :                1,0,ISTAT)
             RETURN
          ENDIF
C If there is one, read in a line of the mask image
          IF(USEWEI) THEN
             CALL UIGL2R(IDW,J,MEMR(PWEI),ISTAT)
             IF(ISTAT.NE.0) THEN
               CALL UMSPUT('! Failed to read mask image',
     :                1,0,ISTAT)
               RETURN
             ENDIF
          ENDIF
         
C Set JNY to 1 since only 1 line at a time will ever get
C  passed into DOBOX
          JNY = 1
C Now do the actual combination using BOXER 
C We can skip this if there is no overlap
      IF(.NOT.NOOVER) THEN

       CALL DOBOX(MEMR(PDATA),MEMR(PWEI),
     :      MEMR(PNDAT),MEMR(PNCOU),MEMI(PNCON),MEMI(PCDON),DNX,DNY,JNY,
     :      J-1,XMIN,XMAX,YMIN,YMAX,NOOVER,
     :      KERNEL,MEMD(PXI),MEMD(PXO),MEMD(PYI),MEMD(PYO),
     :      MEMD(PXIB),MEMD(PXOB),MEMD(PYIB),MEMD(PYOB),
     :      ONX,ONY,COTY,CONUM,XCO,YCO,
     :      DISIM,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM,
     :      WTSCL,ALIGN,INCPS,EXPIN,
     :      PFRACT,SCALE,ROT,XSH,YSH,WCS,WCSOUT,ROTFIR,
     :      SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :      CON,BITCON,INTAB,MAXIM,MAXEN,NEN,UNIQID,
     :      UPDATE,USEWEI,USEWCS,ISTAT,NMISS,NSKIP)

C Check the exit status
        IF(ISTAT.NE.0) GO TO 99
      ENDIF
C End loop over lines
      ENDDO
C Report number of points which were outside the output frame
      IF(NMISS.GT.0) THEN
         WRITE(CHARS,'(''! Warning, '',I9,'' points were '','//
     :        '''outside the output image.'')') NMISS
         CALL UMSPUT(CHARS,1,0,ISTAT)

C Also report the number of skipped lines
         IF(NSKIP.GT.0) THEN
            WRITE(CHARS,
     : '(''! Note, '',I5,'' input lines were skipped completely.'')')
     :    NSKIP
            CALL UMSPUT(CHARS,1,0,ISTAT)
         ENDIF
      ENDIF

C Close the input images
      IF(ODD) THEN
         CALL UIMCLO(IDD,ISTAT)
         ODD=.FALSE.
      ENDIF

      IF(ODW) THEN
         CALL UIMCLO(IDW,ISTAT)
         ODW=.FALSE.
      ENDIF

      IF(NOOVER) THEN
         CALL UMSPUT('! No overlap, not writing anything out',
     :               1,0,ISTAT)
      ELSE

C Write out the exposure time
         IF(UPDATE) THEN
            CALL UHDPSR(IDND,EXPKEY,EXPOUT,ISTAT)
            IF(WRIWEI) CALL UHDPSR(IDNC,EXPKEY,EXPOUT,ISTAT)
            IF(CON) CALL UHDPSR(IDCO,EXPKEY,EXPOUT,ISTAT)
         ELSE
            CALL UHDASR(IDND,EXPKEY,EXPOUT,
     : 'Drizzle, effective output exposure time',0,ISTAT)
            IF(WRIWEI) CALL UHDASR(IDNC,EXPKEY,EXPOUT,
     : 'Drizzle, effective output exposure time',0,ISTAT)
            IF(CON) CALL UHDASR(IDCO,EXPKEY,EXPOUT,
     : 'Drizzle, effective output exposure time',0,ISTAT)
         ENDIF

C Update the WCS
         CALL UPWCS(WCS,WCS,DNX,DNY,ONX,ONY,
     :            XSH,YSH,ROT,SCALE,ALIGN,ROTFIR,
     :            SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,ROTF2,
     :            USEWCS,COTY,CONUM,XCO,YCO,
     :            DISIM,MEMR(PXG),MEMR(PYG),XGDIM,YGDIM)

C Scale the output if not CPS output
         FS=1.0
         IF(.NOT.OUTCPS) THEN
            CALL MULC(MEMR(PNDAT),XMAX-XMIN+1,YMAX-YMIN+1,EXPOUT)
            IF(UPDATE) FS=EXPOUT/EXPCUR
         ENDIF

C Put in the fill values (if defined)
         IF(FILL) CALL PUTFIL(MEMR(PNDAT),MEMR(PNCOU),
     :         XMAX-XMIN+1,YMAX-YMIN+1,FILVAL)
      
C Write out the two new images
         IF(VERBOSE)
     :      CALL UMSPUT('-Writing output drizzled image: '//
     :               OUTDAT,1,0,ISTAT)

C Update the header
         CALL UHEAD(IDND,VERS,DATA,WEIGHT,EXPIN,WTSCL,OUTDAT,
     :                 OUTCOU,CONTIM,COEFFS,XGEOIM,YGEOIM,
     :                 SHFTUN,PFRACT,
     :                 LAM,SCALE,ROT,SHFTFR,ALIGN,KERNEL,
     :                 EXPKEY,FILSTR,INUN,OUTUN,
     :                 SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,
     :                 USEWCS,XSH,YSH,DNX,DNY,ONX,ONY)

C If we are not updating (ie, this is the first drizzle
C onto this output) then we modify the WCS
         IF((.NOT.UPDATE).AND.GOTWCS) THEN
            IF(USEWCS) THEN
               CALL UWCS(IDND,WCSOUT,CTYPE1,CTYPE2)
            ELSE
               CALL UWCS(IDND,WCS,CTYPE1,CTYPE2)
            ENDIF
         ENDIF
      ENDIF

C Free as much memory as possible at this stage
       CALL UDMFRE(PDATA,6,ISTAT)
       PDATA=0
       CALL UDMFRE(PWEI,6,ISTAT)
       PWEI=0
       CALL UDMFRE(PXI,7,ISTAT)
       PXI=0
       CALL UDMFRE(PYI,7,ISTAT)
       PYI=0
       CALL UDMFRE(PXO,7,ISTAT)
       PXO=0
       CALL UDMFRE(PYO,7,ISTAT)
       PYO=0
       CALL UDMFRE(PXIB,7,ISTAT)
       PXIB=0
       CALL UDMFRE(PYIB,7,ISTAT)
       PYIB=0
       CALL UDMFRE(PXOB,7,ISTAT)
       PXOB=0
       CALL UDMFRE(PYOB,7,ISTAT)
       PYOB=0

C Now write out the appropriate part (or whole) data image
      IF(.NOT.NOOVER) THEN
         CALL PUTIMR(IDND,ONX,ONY,SUB,XMIN,XMAX,YMIN,YMAX,
     :            UPDATE,OUTCPS,FS,MEMR(PBUFF),MEMR(PNDAT),ISTAT)
         IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Failed to write output data image',
     :               1,0,ISTAT)
            GO TO 99
         ENDIF
      ENDIF

C Close this output image
      IF(ODND) THEN
         CALL UIMCLO(IDND,ISTAT)
         ODND=.FALSE.
      ENDIF

C Now write out the weights array, if there is one
      IF(.NOT.NOOVER) THEN

         IF(WRIWEI) THEN
            IF(VERBOSE)
     :         CALL UMSPUT('-Writing output weighting image: '//
     :               OUTCOU,1,0,ISTAT)
         
C First update the header
            CALL UHEAD(IDNC,VERS,DATA,WEIGHT,EXPIN,WTSCL,OUTDAT,
     :                 OUTCOU,CONTIM,COEFFS,XGEOIM,YGEOIM,
     :                 SHFTUN,PFRACT,
     :                 LAM,SCALE,ROT,SHFTFR,ALIGN,KERNEL,
     :                 EXPKEY,FILSTR,INUN,OUTUN,
     :                 SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,
     :                 USEWCS,XSH,YSH,DNX,DNY,ONX,ONY)

            IF((.NOT.UPDATE).AND.GOTWCS) THEN
               IF(USEWCS) THEN
                  CALL UWCS(IDNC,WCSOUT,CTYPE1,CTYPE2)
               ELSE
                  CALL UWCS(IDNC,WCS,CTYPE1,CTYPE2)
               ENDIF
            ENDIF
         ENDIF
      ENDIF

C Clear some more space
      CALL UDMFRE(PNDAT,6,ISTAT)
      PNDAT=0

C Now write out the appropriate part (or whole) weights image
C note that there is never any re-scaling here
      IF(WRIWEI) THEN
         IF(.NOT.NOOVER) THEN
            CALL PUTIMR(IDNC,ONX,ONY,SUB,XMIN,XMAX,YMIN,YMAX,
     :             UPDATE,.TRUE.,1.0,MEMR(PBUFF),MEMR(PNCOU),ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT('! Failed to write output weight image',
     :               1,0,ISTAT)
               GO TO 99
            ENDIF
         ENDIF
      ENDIF

C Close this
      IF(ODNC) THEN
        CALL UIMCLO(IDNC,ISTAT)
        ODNC=.FALSE.
      ENDIF

C Finally write the context array, which is INTEGER, if we have
C one
      IF(.NOT.NOOVER) THEN
         IF(CON) THEN
            IF(VERBOSE)
     :         CALL UMSPUT('-Writing output context image: '//
     :               CONTIM,1,0,ISTAT)

C First update the header
             CALL UHEAD(IDCO,VERS,DATA,WEIGHT,EXPIN,WTSCL,OUTDAT,
     :                 OUTCOU,CONTIM,COEFFS,XGEOIM,YGEOIM,
     :                 SHFTUN,PFRACT,
     :                 LAM,SCALE,ROT,SHFTFR,ALIGN,KERNEL,
     :                 EXPKEY,FILSTR,INUN,OUTUN,
     :                 SECPAR,XSH2,YSH2,ROT2,XSCALE,YSCALE,SHFR2,
     :                 USEWCS,XSH,YSH,DNX,DNY,ONX,ONY)

C If we are not updating (ie, this is the first drizzle
C onto this output) then we modify the WCS
            IF((.NOT.UPDATE).AND.GOTWCS) THEN
               IF(USEWCS) THEN
                  CALL UWCS(IDCO,WCSOUT,CTYPE1,CTYPE2)
               ELSE
                  CALL UWCS(IDCO,WCS,CTYPE1,CTYPE2)
               ENDIF
            ENDIF
         ENDIF
      ENDIF

C Clear some more space
      CALL UDMFRE(PNCOU,6,ISTAT)
      PNCOU=0

C Now write out the appropriate part (or whole) context image
C No scaling needed here either
      IF(CON) THEN
       IF(.NOT.NOOVER) THEN
          CALL PUTIMI(IDCO,ONX,ONY,SUB,XMIN,XMAX,YMIN,YMAX,
     :             UPDATE,.TRUE.,1.0,MEMI(PIBUFF),MEMI(PNCON),ISTAT)
          IF(ISTAT.NE.0) THEN
            CALL UMSPUT('! Failed to write output weight image',
     :               1,0,ISTAT)
            GO TO 99
          ENDIF

C Now update the global context table in header
          IF(.NOT.BITCON) THEN
            CALL PTCOIN(IDCO,CONTAB,MAXHCN,INTAB,MAXIM,MAXEN,NEN,ISTAT)
            IF(ISTAT.NE.0) THEN
               CALL UMSPUT('! Failed to update context in header',
     :                    1,0,ISTAT)
               GO TO 99
            ENDIF
          ENDIF
       ENDIF
      ENDIF

C General error abort point
 99   CONTINUE

C Free up all allocated memory arrays
      CALL FREMEM(PDATA,PWEI,PNDAT,PNCOU,PNCON,PCDON,PBUFF,PIBUFF,
     : PXI,PYI,PXO,PYO,PXIB,PYIB,PXOB,PYOB)

C If the distortion images have been used free that too
      IF(PXG.NE.0) CALL UDMFRE(PXG,6,ISTAT)
      IF(PYG.NE.0) CALL UDMFRE(PYG,6,ISTAT)

C Close all the files which are still open
      IF(ODD) CALL UIMCLO(IDD,ISTAT)
      IF(ODW) CALL UIMCLO(IDW,ISTAT)
      IF(ODND) CALL UIMCLO(IDND,ISTAT)
      IF(ODNC) CALL UIMCLO(IDNC,ISTAT)
      IF(ODCO) CALL UIMCLO(IDCO,ISTAT)

C End of main DRIZZLE module
      END
