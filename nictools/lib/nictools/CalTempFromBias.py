#! /usr/bin/env python
#
# Authors: Dave Grumm (based on work by Eddie Bergeron)
# Program: CalTempFromBias.py
# Purpose: class to process data for a given filename
# History:
#   10/31/07 - first version [DGrumm]
#   11/08/07 - added nonlinearity application
#   11/21/07 - minor interface changes, including reodering of input parameters
#   12/04/07 - minor interface changes: no defaults for spt_key and raw_key;
#              under pyraf these keys are settable in both __init__() and in update_headers()
#   02/25/08 - code version 1.3
#            - the pyraf command line parser is now used for all arguments. An example under pyraf
#              in which parameters are overwritten:
#              --> tfb = CalTempFromBias.CalTempFromBias( "n8tf30jnq_raw.fits", edit_type="SPT",
#                        force="QUIET", hdr_key="MYHKEY", err_key="MYEKEY", verbosity=2)
#              --> [temp, sigma, winner, in_flag, dry_run ]= tfb.calctemp()
#              --> stat = tfb.update_header( temp, sigma, winner)
#              The same example under linux:
#              hal> ./CalTempFromBias.py "n8tf30jnq_raw.fits" -e "SPT" -f "Q" -k "MYHKEY" -s "MYEKEY" -v
#            - only the filename parameter is required. The parameters spt_key and raw_key
#              are no longer used. Currently the other parameters are:
#                hdr_key: string for keyword name for temperature; default = "TFBT"
#                err_key: string for keyword name for associated error; default = "TFBE"
#                edit_type: string for file type in which keywords will be updated; default = "RAW"
#                force: character for name of algorithm whose value is to be returned, regardless
#                     of which algorithm had the lowest estimated sigma.  Valid values are None,
#                     S(tate)B(lind),Q(uietest); default = None
#                nref_par: string for name of the directory containing the nonlinearity file;
#                        default = "/grp/hst/cdbs/nref/"
#                verbosity: integer level for diagnostic print statements
#                noclean: string for flag to force use of UNCLEANed 0th read; default = False
#            - the routine now writes to either the RAW or the SPT file; the default is
#              the RAW file, and can be overwritten by using the 'edit_type' parameter.
#            - keywords currently written:
#                keyword for temp: string for name of temperature keyword; default = "TFBT"
#                keyword for associated error: string for name of associated error keyword;
#                  default = "TFBE"
#                METHUSED: string for algorithm type used
#                TFB_RUN: time that temp-from-bias was run
#                TFB_VERS: version of temp-from-bias that was run
#            - the default values for all parameters are now set in the file tfbutil.py.
#   04/25/08 - code version 1.4: deleted unused code for algorithm 1 and cameras 1 and 2;
#              the input file can now be a list (a text file having a single file listed on each line.)
#   05/29/08 - code version 1.5: added epar interface, and ability to process filelists
#   07/08/08 - code version 1.6: changing keyword names and types, so they are now:
#                TFBTEMP : float, comment='Temperature (K) derived from bias', default =  0.00E+00
#                TFBERR : float, comment='Associated error (K) for temperature derived from bias', default =  0.00E+00
#                TFBVER : string, comment = 'Version of CalTempFromBias trun, default = 'N/A'
#                TFBDATE: string, comment = 'Date that CalTempFromBias that was run, default = 'N/A'
#                TFBMETH: string, comment = ' CalTempFromBias algorithm type used', default = 'N/A'
#                TFBCALC: string, comment = 'Do CalTempFromBias calculation: PERFORM, OMIT, COMPLETE,
#                         SKIPPED', default = 'N/A'
#   07/29/08 - fixed bug so that now nonlinearity file is searched for in $nref if nref_par not specified.
#              The routine will successfully calculate the temperature only if TFBCALC='PERFORM' and camera=3
#   08/12/08 - fixed uninitialized variable threshold
#   08/22/08 - fixed bug in which argument of median() call in nsamp=2 branch was (incorrectly) not a vector
#   09/18/08 - code version 2.0: modifying logic so that a selection (by the program) of which algorithm
#              to apply is no longer the default but can now be forced by setting force to 'Auto'; the default of
#              'force' is now the Blind correction, and 'Quietest quad' will only be used if explicitly forced. The
#              'State' algorithm is no longer supported.
#            - flow of routine also modified to allow a user to process files not having the 'TFBCALC' keyword, and
#              files for which he/she does not have write access.
#            - support added for cameras 1 and 2.
#            - implementing the following hard-coded values for sigma in degrees K for the 3 cameras:
#                  blindcorr   quietest-quad
#              -------------------------------------------
#              C1    0.02        0.25
#              C2    0.05        0.15
#              C3    0.05        0.25
#
#            - adding option '-d' ('dry_run') to force the program to *not* write to the header, but output to
#              to the screen only if verbosity set. Default is to write the keys to the header
#   10/08/08 - code version 2.01: improved handling of case where non-existent file is specified, and fixed some print
#              statements
#   11/18/08 - code version 2.02: added file existence checking, and removed requirement of reference files to be in $nref
#   01/06/09 - code version 2.03: added check in update header to abort if calctemp() failed to calculate a valid temperature; this has
#              been added for the case of running under pyraf to suppress display of traceback when TFBCALC and force are not set;
#              embellished error message for this case
#
#   06/05/09 - code version 2.04; changed '/' to '//' in quadmean() and added 'from __future__ import division' for upcoming python version
from __future__ import division  # confidence high
import os.path
import sys, time
from optparse import OptionParser
import tfbutil
import opusutil
import numpy as N
import pyfits
import stsci.tools
from stsci.tools import parseinput


# Define some constants
c2_min = 3.6218723e-06; c2_max = -3.6678544e-07; c3_min = 9.0923490e-11; c3_max = -4.1401650e-11  # used in bad pixel clipping

__version__ = "2.04"

ERROR_RETURN = 2

class CalTempFromBias:
    """Calculate the temperature from the bias for a given filename.

    Notes
    ------
    Basic syntax for using this class is::

       tfb = CalTempFromBias( filename, edit_type=edit_type, hdr_key=hdr_key, err_key=err_key,
             nref_par=nref_par, force=force, noclean=noclean, dry_run=dry_run, verbosity=verbosity)
       [temp, sigma, winner, in_flag, dry_run ]= tfb.calctemp()
       tfb.print_pars()
       stat = tfb.update_header( temp, sigma, winner)

    The full set of parameters for the methods:
    """

    def __init__( self, input_file, edit_type=None, hdr_key=None, err_key=None, nref_par=None,
                  force=None, noclean=False, dry_run=1, verbosity=0):

        """constructor

        Parameters
        -----------
        input_file : string
            name of the file or filelist to be processed
        edit_type : string type of file to update
        hdr_key : string
            name of keyword to update in file
        err_key : string
            name of keyword for error estimate
        nref_par : string
            name of the directory containing the nonlinearity file
        force : string
            name of algorithm whose value is to be returned
        noclean : {'True', 'False'}
            flag to force use of UNCLEANed 0th read.
        dry_run : {0,1} [Default: 1]
            flag to force not writing to header
        verbosity : {0,1,2}
            verbosity level (0 for quiet, 1 verbose, 2 very verbose)

        """

        if (edit_type == None):  edit_type = tfbutil.edit_type
        if (hdr_key == None):  hdr_key = tfbutil.hdr_key
        if (err_key == None):  err_key = tfbutil.err_key
        if (force == None):  force = tfbutil.force

        self.input_file = input_file
        self.edit_type = edit_type
        self.hdr_key = hdr_key
        self.err_key = err_key
        self.nref_par = nref_par
        self.force = force
        self.noclean = noclean
        self.dry_run = dry_run
        self.verbosity = verbosity
        self.tfb_version = __version__
        self.tfb_run = time.asctime()

        outlist = parseinput.parseinput(input_file)
        self.num_files =  parseinput.countinputs(input_file)[0]
        self.outlist0 = outlist[0]

        if (( dry_run == 0) & (self.verbosity >0)):
            print ' The dry_run option has been selected so keys will not be written.'

        if (self.verbosity >1):
           print ' Temp_from_bias run on ',self.tfb_run, ', version: ', self.tfb_version


    def calctemp(self):
        """ Calculate the temperature from the bias for the given input file

        Returns
        -------
        temp : float
        sig : float
        winner : int
        in_flag : str
        """

        if self.num_files == 0:
           opusutil.PrintMsg("F","ERROR "+ str('Input file ') + str(self.input_file)+ str(' does not exist'))
           return None, None, None, None, None

        self.temp_list = []
        self.sigma_list = []
        self.winner_list = []
        self.nonlinfile_list = []
        self.in_flag_list = []
        self.dry_run_list = []

        for ii in range(self.num_files):

            temp = 0.0
            edit_type = self.edit_type
            hdr_key = self.hdr_key
            err_key = self.err_key
            nref_par = self.nref_par
            noclean = self.noclean
            dry_run = self.dry_run
            filename = self.outlist0[ii]
            force = self.force
            verbosity = self.verbosity

            if ( verbosity > 0):
              print '   ' ; print '   '
              print 'Calculating temp for file ',ii,' : ', filename

            try:  # ... opening input in update mode
               fh_raw = pyfits.open( filename, mode='update')
               in_flag = 'u'  # update mode
            except:
               try:  # ... opening input in read only mode since the user does not have write access
                  fh_raw = pyfits.open( filename)
                  in_flag = 'r'    # readonly
                  print  '  WARNING : the input file', filename,' could not be opened in update mode, so no keywords will be written.'
               except:
                  in_flag = 'f'    # fail to open
                  opusutil.PrintMsg("F","ERROR "+ str('Unable to open input file') + str(filename))
                  return None, None, None, None, None

            # get header and value for TFBCALC
            try:   # get header and value for TFBCALC (pipeline)
               raw_header = fh_raw[0].header
               tfbcalc_key = raw_header['TFBCALC']
               tfbcalc = tfbcalc_key.lstrip().rstrip() # strip leading and trailing whitespace
            except: # set blank value for TFBCALC (user, with 'old-style' file having no TFBCALC keyword)
               tfbcalc = ''

            obsmode_key = raw_header[ 'OBSMODE' ]
            obsmode = obsmode_key.lstrip().rstrip() # strip leading and trailing whitespace

            nsamp_key = raw_header[ 'NSAMP' ]
            nsamp = nsamp_key

            camera_key = raw_header[ 'CAMERA' ]
            camera = camera_key

            zoffdone_key = raw_header[ 'ZOFFDONE' ]
            zoffdone = zoffdone_key.lstrip().rstrip() # strip leading and trailing whitespace

            if ( obsmode <> 'MULTIACCUM'):
                opusutil.PrintMsg("F","ERROR "+ str('Image must be in MULTIACCUM mode'))

            if ( zoffdone == 'PERFORMED'):
                opusutil.PrintMsg("F","ERROR "+ str('ZOFFCORR has already been performed on this image. No temp information left'))

            # get nonlinearity file name from NLINFILE in header of input file and open it
            if nref_par is not None:
               nref = os.path.expandvars( nref_par)
            else:
               nref = os.path.expandvars( "$nref")

            nonlinfile_key = raw_header[ 'NLINFILE' ]

            try :
               nl_file = nonlinfile_key.split('nref$')[1]
               nonlin_file = os.path.join( nref, nl_file)
            except :
               nonlin_file = nonlinfile_key

            try:
               fh_nl = pyfits.open( nonlin_file )
               self.nonlin_file = nonlin_file
            except:
               opusutil.PrintMsg("F","ERROR "+ str('Unable to open nonlinearity file: ') + str(nonlin_file))

               if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                   raw_header.update("TFBDONE", "SKIPPED")
                   raw_header.update("TFBVER", self.tfb_version)
                   raw_header.update("TFBDATE", self.tfb_run)

               self.nonlin_file = None
               raise IOError,'Unable to open nonlinearity file'
               return None, None, None ,None, None

            # read data from nonlinearity file
            c1 = fh_nl[ 1 ].data; c2 = fh_nl[ 2 ].data; c3 = fh_nl[ 3 ].data

            # do some bad pixel clipping
            u = N.where((c2 > c2_min)  | (c2 < c2_max))
            uu = N.where((c2 < c2_min) & (c2 > c2_max))

            if len(u[0]) > 0 :
               c2[u] = N.median(c2[uu])

            u = N.where((c3 > c3_min) | (c3 < c3_max))
            uu = N.where((c3 < c3_min) & (c3 > c3_max))

            if len(u[0]) > 0:
               c3[u] = N.median(c3[uu])

            if ( nsamp <= 1 ):
               opusutil.PrintMsg("F","ERROR "+ str(' : nsamp <=1, so will not process'))

               if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                   raw_header.update("TFBDONE", "SKIPPED")
                   raw_header.update("TFBVER", self.tfb_version)
                   raw_header.update("TFBDATE", self.tfb_run)

               fh_raw.close()

               return None, None, None, None, None

            if ( nsamp >= 3 ):
                im0 = fh_raw[ ((nsamp-1)*5)+1 ].data
                im1 = fh_raw[ ((nsamp-2)*5)+1 ].data
                im2 = fh_raw[ ((nsamp-3)*5)+1 ].data
                dtime0 = 0.203    # this is by definition - headers have 0.0 so can't use that
                ext_hdr1 = fh_raw[((nsamp-2)*5)+1].header
                dtime1 = ext_hdr1[ 'DELTATIM' ]
                ext_hdr2 = fh_raw[((nsamp-3)*5)+1].header
                dtime2 = ext_hdr2[ 'DELTATIM' ]

                signal_raw = (im2-im1)+(((im2-im1)/dtime2)*(dtime0+dtime1)) # total counts accumulated (1st estimate)

                signal0 = (signal_raw *(signal_raw*c2 + signal_raw**2*c3+1.0))/(dtime0+dtime1+dtime2) # refined 0 ADU countrate

                for ii in range(9):
                  signal_raw = signal0*(dtime0+dtime1) # pure 1st read signal if no nonlinearity
                  signal1 = (signal_raw / (signal_raw*c2 + signal_raw**2*c3 +1.0)) / (dtime0+dtime1) #1st read signal with nl
                  signal_raw = (im2-im1) + (signal1*(dtime0+dtime1)) # total counts accumulated (second estimate)
                  signal0 = (signal_raw * (signal_raw*c2 + signal_raw**2*c3 +1.0))/(dtime0+dtime1+dtime2) # doubly-refined 0 ADU countrate

                signal_raw = signal0 * dtime0 # pure 0th read signal if no nonlinearity
                signal = signal_raw / (signal_raw*c2 + signal_raw**2*c3 +1.0) # 0th read total counts with nonlinearity - this is what should be subtracted from the 0th read

                clean = im0 - signal
                if  (self.noclean == 'True'):
                   clean = im0

    # If there are only 2 reads, can do a partial clean. Subtraction of the signal
    #   measured in this way will have a  negative 0.302s shading imprint in it. The amplitude
    #   of this will be temp-dependent. Best way to deal is to decide if there is enough
    #   signal to warrant a subtraction. if not, just use the 0th read without any correction.

            if ( nsamp == 2 ):
                im0 = fh_raw[ ((nsamp-1)*5)+1 ].data
                im1 = fh_raw[ ((nsamp-2)*5)+1 ].data
                clean = im0
                signal= ((im1-im0)/0.302328 )

                threshold = 10.0  # in DN/s. Every 5 DN/s here is 5*0.203 = 1 DN in the quad median.

                if (N.median(N.ravel(signal)*0.203) > threshold ):
                   clean = im0-(signal * 0.203)

                if  (self.noclean == 'True'):
                   clean = im0

    # Following Eddie's suggestion: I'll catch these rare cases and abort by searching for:
    #   nsamp=2 and filter is NOT blank, instead of original code which compares the median
    #   of the signal and the threshold

                filter_key = raw_header[ 'FILTER' ]
                filter = filter_key.lstrip().rstrip()
                filter = str('BLANK')
                if (filter <> 'BLANK'):
                    opusutil.PrintMsg("F","ERROR "+ str('can not determine the temperature from the bias'))

    # Calculate the quad medians to feed the temp algorithms; current state values are based on a border=5 mean.
            quads = quadmean(clean, border=5)

            if (self.verbosity >1):
                print ' '
                print 'Camera: ',camera
                print 'The results of the quadmean call:'
                print quads

            # do requested caluculation based on TFBCALC and force settings
            if (tfbcalc == 'PERFORM'):

                if ((force == '') | (force == None) ) : #   [ reg test 1a, 1b ]
                    force = str("B")
                    [temp, sig ] = do_blind( camera, quads, verbosity )
                    winner = 2

                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "PERFORMED")
                    if ((self.verbosity >1) & (dry_run == 0)) :
                       print ' The calculated temperature = ' , temp,' and sigma = ', sig
                elif (force.upper()[0] == "B") : #   [ reg test  1c ]
                    force = str("B")
                    [temp, sig ] = do_blind( camera, quads, verbosity )
                    winner = 2

                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "PERFORMED")
                    if ((self.verbosity >1) & (dry_run == 0)) :
                       print ' The calculated temperature = ' , temp,' and sigma = ', sig
                elif (force.upper()[0] == "Q") : #   [ reg test 2 ]
                    [temp, sig ] = do_quietest( camera, quads, verbosity )
                    winner = 3

                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "PERFORMED")
                    if ((self.verbosity >1) & (dry_run == 0)) :
                       print ' The calculated temperature = ' , temp,' and sigma = ', sig
                elif (force.upper()[0] == "A") : # auto   #  [   reg test 3 ]
                    [temp2, sig2 ] = do_blind( camera, quads, verbosity )
                    [temp3, sig3 ] = do_quietest( camera, quads, verbosity )

                    if ( verbosity > 0):
                        print ' Blind correction: temp = ' ,temp2,'  sigma = ' , sig2
                        print ' Quietest quad correction: temp = ' ,temp3,'  sigma = ' , sig3

                    # Compare the error estimates of algorithms 2 and 3 and return the best one; the
                    #   lowest sigma is the winner.
                    winner = 2
                    temp = temp2
                    sig = sig2

                    if (sig3 < sig2):  # quietest quad has least sigme
                       temp = temp3
                       sig = sig3
                       winner = 3
                       if ( verbosity > 0):
                          print ' The algorithm with the least sigma is quietest quad. '
                    else:   # blind correction has least sigme
                       if ( verbosity > 0):
                          print ' The algorithm with the least sigma is blind correction. '

                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "PERFORMED")
                    if ((self.verbosity >1) & (dry_run == 0)) :
                       print 'The calculated temperature = ' , temp,' and sigma = ', sig
                else : # invalid selection   #  [  reg test 4 ]
                    opusutil.PrintMsg("F","ERROR "+ str('Invalid force selected : ') + str(force))
                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "SKIPPED")
                       raw_header.update("TFBVER", self.tfb_version)
                       raw_header.update("TFBDATE", self.tfb_run)
                    fh_raw.close()
                    return None, None, None, None, None
            elif (tfbcalc == 'OMIT'):     #  [  reg test 5 ]
                opusutil.PrintMsg("F","ERROR "+ str('TFBCALC set to omit, so aborting.'))
                if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                    raw_header.update("TFBDONE", "OMITTED")
                    raw_header.update("TFBVER", self.tfb_version)
                    raw_header.update("TFBDATE", self.tfb_run)
                fh_raw.close()
                return None, None, None, None, None
            else: # tfbcalc not set (user running on old file)
                print ' WARNING : TFBCALC is not set in the input file.'

                if ((force == '') | (force == None)) :   #  [  reg test 6 ]
                    opusutil.PrintMsg("F","ERROR "+ str('Because the keyword TFBCALC is not set in the input file and the parameter Force is not set, this run is aborting. If you want to process this file, either set TFBCALC to PERFORM or set Force to an allowed option (not None)'))

                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "SKIPPED")
                       raw_header.update("TFBVER", self.tfb_version)
                       raw_header.update("TFBDATE", self.tfb_run)
                    fh_raw.close()
                    return None, None, None, None, None
                elif (force.upper()[0] == "B") :     #  [  reg test 7 ]
                    [temp, sig ] = do_blind( camera, quads, verbosity )

                    winner = 2
                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "PERFORMED")
                    if ((self.verbosity >1) & (dry_run == 0)) :
                       print ' The calculated temperature = ' , temp,' and sigma = ', sig

                elif (force.upper()[0] == "Q") :    #  [  reg test 8 ]
                    [temp, sig ] = do_quietest( camera, quads, verbosity )
                    winner = 3
                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "PERFORMED")
                    if ((self.verbosity >1) & (dry_run == 0)) :
                       print ' The calculated temperature = ' , temp,' and sigma = ', sig
                elif (force.upper()[0] == "A") : # auto   #  [  reg test 9 ]
                    [temp2, sig2 ] = do_blind( camera, quads, verbosity )
                    [temp3, sig3 ] = do_quietest( camera, quads, verbosity )

                    if ( verbosity > 0):
                        print ' Blind correction: temp = ' ,temp2,'  sigma = ' , sig2
                        print ' Quietest quad correction: temp = ' ,temp3,'  sigma = ' , sig3

                    # Compare the error estimates of algorithms 2 and 3 and return the best one; the
                    #   lowest sigma is the winner.
                    winner = 2
                    temp = temp2
                    sig = sig2

                    if (sig3 < sig2):
                       temp = temp3
                       sig = sig3
                       winner = 3

                       if ( verbosity > 0):
                          print ' The algorithm with the least sigma is quietest quad.'
                    else:
                       if ( verbosity > 0):
                          print ' The algorithm with the least sigma is blind correction.'

                    if ( verbosity > 0):
                       print ' The least sigma is ', sig, ' with corresponding temp = ',temp
                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "PERFORMED")
                    if ((self.verbosity >1) & (dry_run == 0)) :
                       print ' The calculated temperature = ' , temp,' and sigma = ', sig
                else : # invalid selection    #  [  reg test 10 ]
                    opusutil.PrintMsg("F","ERROR "+ str(' Invalid selection of force, so aborting.'))
                    if ((in_flag[0] != 'f') & ( dry_run != 0)) :
                       raw_header.update("TFBDONE", "SKIPPED")
                       raw_header.update("TFBVER", self.tfb_version)
                       raw_header.update("TFBDATE", self.tfb_run)
                    fh_raw.close()
                    return None, None, None, None, None

            self.temp_list.append( temp )
            self.sigma_list.append( sig )
            self.winner_list.append( winner )
            self.nonlinfile_list.append( nonlin_file )
            self.in_flag_list.append( in_flag )
            self.dry_run_list.append( dry_run )

            # close any open file handles
            if fh_raw:
              fh_raw.close()
            if fh_nl:
              fh_nl.close()


        return self.temp_list, self.sigma_list, self.winner_list, self.in_flag_list, self.dry_run_list


## end of def calctemp()


    def update_header(self, temp, sig, winner, edit_type=None, hdr_key=None,
                      err_key=None):
        """ Update header method

        Parameters
        ----------
        temp : float
            calculated temperature
        sig : float
            standard deviation of calculated temperature
        winner : int
            algorithm used
        edit_type : string
            type of file to be updated
        hdr_key : string
            name of keyword to update in file
        err_key : string
            name of keyword for error estimate

        Returns
        -------
        status : int (not None for failure due to key not being specified)

        """

        if ( temp == None):   # added to abort if calctemp() failed to calculate a valid temperature; to suppress traceback under pyraf
            return ERROR_RETURN

        for ii in range(self.num_files):
            this_file = self.outlist0[ii]

            if (hdr_key == None):
                if (self.hdr_key <> None):  # use value given in constructor
                    hdr_key = self.hdr_key
                else:
                    opusutil.PrintMsg("F","ERROR "+ str('No value has been specified for hdr_key'))
                    return ERROR_RETURN
            else:
                self.hdr_key = hdr_key # for later use by print_pars()
                if ( self.verbosity > 0):
                    print ' Using value of hdr_key = ' , hdr_key,' that has been passed to update_header()'

            if (err_key == None):
                if (self.err_key <> None):  # use value given in constructor
                    err_key = self.err_key
                else:
                    opusutil.PrintMsg("F","ERROR "+ str('No value has been specified for err_key'))
                    return ERROR_RETURN
            else:
                self.err_key = err_key # for later use by print_pars()
                if ( self.verbosity > 0):
                    print ' Using value of err_key = ' , err_key,' that has been passed to update_header()'

            if (edit_type == None):
                if (self.edit_type <> None):  # use value given in constructor
                    edit_type = self.edit_type
                else:
                    opusutil.PrintMsg("F","ERROR "+ str('No value has been specified for edit_type'))
                    return ERROR_RETURN
            else:
                self.edit_type = edit_type # for later use by print_pars()
                if ( self.verbosity > 0):
                    print ' Using value of edit_type = ' , edit_type,' that has been passed to update_header()'

            if (winner[0] == 2):# 'blind-correction'
                meth_used = "BLIND CORRECTION"
            else: # (winner == 3):# 'quietest-quad'
                meth_used = "QUIETEST-QUAD"

            if ( self.verbosity > 0):
                 print ' The algorithm used is ' , meth_used

            comm = str('Temp from bias, sigma=')+str(sig)+str(' (K)')

            filename =  this_file

            # update either the RAW or SPT file
            if (edit_type[0] =="S"): # SPT
                underbar = filename.find('_')
                filename =  filename[:underbar] +'_spt.fits'

            try:
                fh = pyfits.open( filename, mode='update' )
            except :
                opusutil.PrintMsg("F","ERROR "+ str('Unable to open raw or spt file') + str(filename))
                return None, None, None, None, None

            hdr = fh[0].header # NOTE - this the PRIMARY extension

            try :
                hdr.update(hdr_key, temp[ii])
                hdr.update(err_key, sig[ii])
                hdr.update("TFBMETH", meth_used)
                hdr.update("TFBDATE", self.tfb_run)
                hdr.update("TFBVER", self.tfb_version)
                hdr.update("TFBDONE", "PERFORMED")

                fh.close()
            except :
                hdr.update("TFBDONE", "SKIPPED")

                fh.close()
                pass

        if (( self.verbosity > 0) & (self.dry_run != 0 )):
            print ' The headers have been updated.'

        return None

    def print_pars(self):
        """ Print parameters used.
        """
        print ' The parameters used are :'
        print '  input_file list:  ' , self.input_file
        print '  edit_type:  ' , self.edit_type
        print '  hdr_key: ' ,  self.hdr_key
        print '  err_key: ' ,  self.err_key
        print '  nref_par:  ' , self.nref_par
        print '  force: ' ,  self.force
        print '  noclean: ' ,  self.noclean
        print '  dry_run: ' ,  self.dry_run
        print '  verbosity: ' ,  self.verbosity
        print ' For the files given by the input file list: '
        for ii in range(self.num_files):
           try :
                this_file = self.outlist0[ii]
                print '  input_file[',ii,']:  ' , this_file
                this_nl_file = self.nonlinfile_list[ii]
                print '  nonlinearity file[',ii,']: ' ,  this_nl_file
           except :
                pass


def do_blind( camera, quads, verbosity):
        """ Calculate temperature using the blind correction

        Parameters
        ----------
        camera : int
            number of camera
        quads : float
            value of quad from quadmean
        verbosity : int
            level of veborsity

        Returns
        -------
        temp, sig :  float, float

        """
        if ( camera == 1):  # NIC1 (average of four phase-difference temps)
           p0 = [3675.0754,1.0586028]
           p1 = [0.0,1.0]
           p2 = [701.78575,1.0094869]
           p3 = [-548.65582,0.99385020]
           pp = [146.94269,0.0035428258]

           ccc = ( (quads[1]-quads[2]) + (quads[1]-quads[3]) ) * 0.85
           temp0 = quads[0]-((quads[0]-quads[2])*1.45)
           temp1 = quads[1]-ccc
           temp2 = quads[2]-((quads[1]-quads[3])*0.65)
           temp3 = quads[3]-((quads[1]-quads[2])*0.67)

           bc1_0 = (poly(temp0,p0)*pp[1])+pp[0]
           bc1_1 = (poly(temp1,p1)*pp[1])+pp[0]
           bc1_2 = (poly(temp2,p2)*pp[1])+pp[0]
           bc1_3 = (poly(temp3,p3)*pp[1])+pp[0]

           temp2 = (bc1_0+bc1_1+bc1_2+bc1_3)/4.
           sigma2 = 0.02

           temps2 = ["%5.2f" % (bc1_0),"%5.2f" % (bc1_1),"%5.2f" % (bc1_2),"%5.2f" % (bc1_3)]
        elif ( camera == 2):   # NIC2 (average of two phase-difference temps)
           p0 = [0.0,1.0]
           p1 = [-166.09229,0.99812304]
           pp = [154.06588,0.0036602771]

           temp0 = quads[1]-((quads[2]-quads[3])*1.53)
           temp1 = quads[0]-((quads[2]-quads[3])*0.62)

           bc2_0 = (poly(temp0,p0)*pp[1])+pp[0]
           bc2_1 = (poly(temp1,p1)*pp[1])+pp[0]

           temp2 = (bc2_0+bc2_1)/2.
           sigma2 = 0.05

           temps2 = ["%5.2f" % (bc2_0),"%5.2f" % (bc2_1),"NA","NA"]
        elif ( camera == 3):   # NIC3 (a single phase-difference temp)
           p0 = [0.0,1.0]
           pp = [180.58356,0.0040206280]
           bc3 = quads[3]+((quads[0]-quads[3])*7.5)
           bc3_1 = (poly(bc3,p0)*pp[1])+pp[0]

           temp2 = bc3_1
           sigma2 = 0.05

           temps2 = ["NA","NA","NA","%5.2f" % (temp2)]
        else:
           print ' Camera ', camera,' not supported'
           return None, None

        if (verbosity >1):
            print ' Algorithm - Blind Correction: ',temp2,' (K) +/- ',sigma2,' (sigma)'

        return temp2, sigma2


def do_quietest( camera, quads, verbosity ):
        """ Calculate temperature using the quietest quad correction

        Parameters
        ----------
        camera :  int
            number of camera
        quads :  float
            value of quad from quadmean
        verbosity : int
            level of veborsity

        Returns
        -------
        temp, sig : float

        """
        if ( camera == 1):           # NIC1 (average of two equally quiet quads)
           pp2 = [156.57265,0.0036923387]
           pp3 = [143.59284,0.0035363450]

           t2 = (quads[2]*pp2[1])+pp2[0]
           t3 = (quads[3]*pp3[1])+pp3[0]

           temp3 = (t2+t3)/2.
           sigma3 = 0.25

           temps3 = ["NA","NA","%5.2f" % (t2),"%5.2f" % (t3)]
        elif ( camera == 2):          # NIC2 (just one quiet quad)
           pp3 = [142.49164,0.0035247897]

           t3 = (quads[3]*pp3[1])+pp3[0]

           temp3 = t3
           sigma3 = 0.15

           temps3 = ["NA","NA","NA","%5.2f" % (t3)]
        elif  ( camera == 3):          # NIC3 (average of two equally quiet quads)
           pp1=[153.74029,0.0037303356]
           pp2=[151.51609,0.0036943955]

           t1=(quads[1]*pp1[1])+pp1[0]
           t2=(quads[2]*pp2[1])+pp2[0]

           temp3=(t1+t2)/2.
           sigma3 = 0.25

           temps3 = ["NA","%5.2f" % (t1),"%5.2f" % (t2),"NA"]
        else:
           print ' Camera ', camera,' not supported'
           return None, None

        if (verbosity >1):
            print ' Algorithm - Quietest Quad Correction: ',temp3,' (K) +/- ',sigma3,' (sigma)'

        return temp3, sigma3


#******************************************************************
# This quadmean does not have the following parameters that are in the idl version: section,
#    avg, mask, and calculates the mean only ( not median ).
def quadmean( im, border):
    """  This function computes the mean in the 4 quadrants of an input NxM array

    Parameters
    -----------
    im :  ndarray
        input rectangular array
    border : int
        border size (in pixels) aroud the perimeter of each quad to be excluded from the mean

    Returns
    -------
    quads : float

    Notes
    ------
    Following Eddie's convention, the quads are numbered as follows::

     |------|-----|
     |  Q4  |  Q3 |
     |      |     |   as seen in the standard NICMOS/HST data format.
     |------|-----|
     |  Q1  |  Q2 |
     |      |     |
     |------|-----|

    optionally, you can specify a border, in pixels, around the perimeter of EACH QUAD
    to be excluded from the mean
    """

    xsize = im.shape[1]
    ysize = im.shape[0]

    quads = N.zeros((4), dtype=N.float64)

    if not border:
      border = 0

    x1 = 0 + border
    x2 = ((xsize//2)-1) - border
    x3 = (xsize//2) + border
    x4 = (xsize-1) - border
    y1 = 0 + border
    y2 = ((ysize//2)-1) - border
    y3 = (ysize//2) + border
    y4 = (ysize-1) - border

    quads[0] = N.mean(im[y1:y2+1,x1:x2+1])
    quads[1] = N.mean(im[y1:y2+1,x3:x4+1])
    quads[2] = N.mean(im[y3:y4+1,x3:x4+1])
    quads[3] = N.mean(im[y3:y4+1,x1:x2+1])

    return  quads


def poly( var_x, coeffs):
    """ Return linear polynomial with given coefficients.

    var_x :  scalar, vector, or array
        input argument
    coeffs :  float
        vector of polynomial coefficients

    """
    poly_out = coeffs[ 0 ] + coeffs[ 1 ]* var_x

    return poly_out

if __name__=="__main__":
    """Get input file and other arguments, and call CalTempFromBias.

    The command-line options are::

        -q (quiet)
        -v (very verbose)

    Parameters
    ----------
    cmdline: list of strings
        command-line arguments
    """

    usage = "usage:  %prog [options] inputfile"
    parser = OptionParser( usage)

    # add options and set defaults for parameters
    parser.set_defaults( verbosity = tfbutil.QUIET)
    parser.add_option( "-q", "--quiet", action = "store_const",
            const = tfbutil.QUIET, dest = "verbosity",default=None,
            help = "quiet, print nothing")
    parser.add_option( "-v", "--verbose", action="store_const",
            const = tfbutil.VERY_VERBOSE, dest="verbosity",default=None,
            help="very verbose, print lots of information")
    parser.add_option( "-e", "--edit_type", dest = "edit_type",default = tfbutil.edit_type,
            help = "Type of file to update")
    parser.add_option( "-k", "--hdr_key", dest = "hdr_key",default = tfbutil.hdr_key,
            help = "Name of header keyword to populate")
    parser.add_option( "-s", "--err_key", dest = "err_key",default = tfbutil.err_key,
            help = "Name of keyword for estimate of error")
    parser.add_option( "-n", "--nref_dir", dest = "nref_dir",default = tfbutil.nref_par,
            help = "Name of directory containing the non linearity file")
    parser.add_option( "-f", "--force", dest = "force",default = tfbutil.force,
            help = "Name of algorithm whose value is to be returned,regardless of which algorithm had the lowest estimated sigma. Valid values are None,Blind,Quietest")
    parser.add_option( "-c", "--noclean", dest = "noclean",default = tfbutil.noclean,
            help = "Flag to force use of UNCLEANed 0th read.")

    parser.set_defaults( dry_run = tfbutil.DO_WRITE_KEYS)
    parser.add_option( "-d", "--do_not_write_keys", action = "store_const",
            const = tfbutil.DO_NOT_WRITE_KEYS, dest = "dry_run",default=None,
            help = "Do not write keys to header")

    (options, args) = parser.parse_args()

    if len(args)==1:  # ensure that exactly 1 argument is given (for the filename)
        filename = args[0]
    else:
        parser.print_help()
        opusutil.PrintMsg("F","ERROR "+ "Invalid Arguments")
        sys.exit(1)

    tfbutil.setVerbosity( options.verbosity)
    verbosity = options.verbosity

    tfbutil.setHdr_key(options.hdr_key )
    if options.hdr_key!=None: hdr_key = options.hdr_key

    tfbutil.setErr_key(options.err_key )
    if options.err_key!=None: err_key = options.err_key

    tfbutil.setEdit_type_key(options.edit_type )
    if options.edit_type!=None: edit_type = options.edit_type

    tfbutil.setNoclean(options.noclean )
    if options.noclean!=None: noclean = options.noclean

    tfbutil.setDry_run(options.dry_run )
    dry_run = options.dry_run

    tfbutil.setNref(options.nref_dir )
    if options.nref_dir!=None: nref_dir = options.nref_dir

    tfbutil.setForce(options.force )
    force = options.force

    try :
         tfb = CalTempFromBias( filename, edit_type=edit_type, hdr_key=hdr_key, err_key=err_key,
                                nref_par=nref_dir, force=force, noclean=noclean, dry_run=dry_run, verbosity=verbosity)

         [temp, sigma, winner, in_flag, dry_run ]= tfb.calctemp()

         if ([temp, sigma, winner ] != [None, None, None]):

             stat = None
             if (( in_flag[0] == 'u' ) & ( dry_run != 0)):
                stat = tfb.update_header( temp, sigma, winner)

             if ( in_flag[0] == 'r' ): # display since file was not writablee
                print ' The calculated temperature and sigma are: ', temp[0],' ', sigma[0]

             if ( (stat == None )and (verbosity > 0) | ( in_flag[0] == 'r' )):
               tfb.print_pars()

    except Exception, errmess:
         opusutil.PrintMsg("F","ERROR "+ str(errmess))
