#! /usr/bin/env python

""" calacsdriz.py - Module to control operation of CALACS and subsequent
    run of PyDrizzle to fully calibrate and combine ACS images in the
    pipeline.

USAGE: calacsdriz.py inputFilename [no] [yes]
   where
        runcal defaults to 'no' (first optional parameter)
        rundriz defaults to 'yes' (second optional parameter)
        and if 'runcal' needs to be specified in order to specify rundriz.

Alternative USAGE: python
                   from calacsdriz import *
                   runCalacsDriz(inputFilename)
IRAF Usage:
    epar calacsdriz
    calacsdriz input runcal=yes rundriz=no mode=h

*** INITIAL VERSION

W.J. Hack  9 April 2001


"""

# Import standard Python modules
import os, sys, string,time

# Import necessary IRAF modules for CALACS
import pyraf
import iraf

yes = iraf.yes
no = iraf.no

# For testing purposes only...
#if os.environ.has_key('stsdas'):
#    iraf.set(stsdas = os.environ['stsdas'])

######################################
#
# INSTALLATION Instructions:
#
# These paths need to be edited to point to the installed location
# of the new Multidrizzle 'mdrizzle' directory.
#
# The file 'multidrizzle.cl' sets up the IRAF interface for the new
# version independent of the STSDAS package and is required for now to
# set up the new version of MultiDrizzle.
#
######################################
sys.path.insert(1,'/data/chulak1/dev/Multidrizzle')
iraf.task(multidrizzle = '/data/chulak1/dev/Multidrizzle/mdrizzle/multidrizzle.cl')
######################################
#
#  End of installation customization
#
######################################

import mdrizzle
import pydrizzle
from pydrizzle import fileutil,drutil

from pyraf.iraf import stsdas,hst_calib,acs,calacs

# Import local modules
import pyfits

# Local variables
__version__ = "2.1.2 (25-June-2004)"
__bitvalue__ = 8192 + 2 + 128 +256        # default value for PyDrizzle usage
#default marker for trailer files
__trlmarker__ = '*** CALACS MultiDrizzle Processing Version '+__version__+'***\n'

# History:
# Version 2.0  - Changed to run MultiDrizzle instead of PyDrizzle
# Version 1.7b - Fixed trailer name creation problems.
# Version 1.7  - Implemented OPUS-style timestamps and
#                   improved input image name handling (supports processing _raw files).
#
# Version 1.6b - Added extra ERROR output to trailer file upon PyDrizzle
#                   raising an Exception.
# Version 1.6a - Trapped 'IRAF error' messages from 'stderr' for CALACS,
#                  and appends those messages to trailer file, then returns
#                   an immediate Exception.
# Version 1.6 - Returns Exception on ERROR in CALACS, and from all other
#                   processing step errors.
#               Adds ability to control both CALACS and PyDrizzle processing.
#               Captures STDERR messages for both CALACS and PyDrizzle and
#               appends them to the trailer file.
#
# Version 1.5c - Updated to support both 'DRIZCORR' and 'DITHCORR'
# Version 1.5b - Updated default bitvalue to also ignore '128'.
#
# Version 1.5a - Corrected problems with namespaces when using 'import'.
#
# Version 1.5 - Updated trlmarker to reflect version of this program used,
#               and added explicit import of pyraf for usability. Also, removed
#               explicit dependence on Python2.0 in '/usr/bin/env' line.
#
# Version 1.4 - Updated to use new Version attribute for PyDrizzle,
#               and to only drizzle those datasets which are
#               explicitly set to 'PERFORM'. Also, relies on numarray and
#               PyFITS V0.6.2 or greater.
#
# Version 1.3  - Added IRAF task parameter interface and par file.
#
# Version 1.2  - Creates default trailer file message for cases where
#                   PyDrizzle processing has been turned off for input images.
# Version 1.1a - Corrected RAW file usage to pass calibrated product
#                   name to drizzle, once computed and verified.
# Version 1.1  - Modified to correctly use either ASN or RAW filenames
#                as inputs, and to correctly figure out the real trailer
#                filename that needs to be appended. WJH 18Oct01
# Version 1.0a - Moved from STLOCAL to STSDAS and removed explicit path
#                modification for adding STSDAS to IRAF path. WJH 11Oct01
# Version 1.0  - Added to CALACS (stlocal.testacs) for pipeline use.
# Version 0.3  - Searches for and removes '_dth.fits' file created by CALACS
# Version 0.2a - Opens product file to get keyword switch [DITH/DRIZ]CORR
#                to control processing, and will update keyword upon
#                completion drizzling.
# Version 0.2 - Revised to make running CALACS optional and to
#               automatically pick-up the STSDAS home directory for PYRAF

# Function to control execution from the shell.
def _main():

    _nargs = len(sys.argv)
    # Check the command line arguments
    if _nargs < 2:
        print "syntax: calacsdriz.py inputFilename [runcal] [rundriz]"
        print "        where, if given, [runcal] is 'yes' or 'no'(default)"
        print "        and, [rundriz] can be 'yes'(default) or 'no', but"
        print "        [rundriz] can only be specified if [runcal] is given."
        sys.exit()

    # default values
    runcal = no
    rundriz = yes
    # Parse parameter for running CALACS
    if _nargs > 2:
        _rc = string.lower(sys.argv[2])
        # Convert runcal argument to IRAF boolean yes or no
        # from input string
        if not string.find(_rc,'y'):
            runcal = yes

    # Find out whether PyDrizzle was turned off or on.
    if _nargs > 3:
        _rd = string.lower(sys.argv[3])
        if not string.find(_rd,'y'):
            rundriz = yes
        else:
            rundriz = no

    if runcal == no and rundriz == no:
        print "No processing requested for %s. Exiting..." % sys.argv[1]
        sys.exit()

    # Invoke the module function
    try:
        runCalacsDriz(sys.argv[1],runcal=runcal,rundriz=rundriz)
    except Exception, errorobj:
        print str(errorobj)
        print "ERROR: Cannot run CALACS/PyDrizzle on %s." % sys.argv[1]
        raise Exception, str(errorobj)

    sys.exit()


def runCalacsDriz(inFile = None, runcal = None, rundriz=None):
    """ Run CALACS, then (by default) run PyDrizzle on input file/ASN table
        using default values for PyDrizzle parameters.
    """
    # Open the input file
    try:
        # Make sure given filename is complete and exists...
        inFilename = fileutil.buildRootname(inFile,ext=['.fits'])
        if not os.path.exists(inFilename):
            print "ERROR: Input file - %s - does not exist." % inFilename
            return
    except TypeError:
        print "ERROR: Inappropriate input file."
        return


    # Initialize for later use...
    _mname = None
    # Check input file to see if [DRIZ/DITH]CORR is set to PERFORM
    _indx_asn = inFilename.find('_asn')
    if _indx_asn > 0:
        # We are working with an ASN table.
        # Use PyDrizzle's code to extract filename
        _asndict = fileutil.readAsnTable(inFilename,None)
        _fname = fileutil.buildRootname(string.lower(_asndict['output']),ext=['_drz.fits'])
        _cal_prodname = string.lower(_asndict['output'])

        # Retrieve the first member's rootname for possible use later
        _fimg = pyfits.open(inFilename)
        _mname = string.lower(string.split(_fimg[1].data.field('MEMNAME')[0],'\0',1)[0])
        _fimg.close()
        del _fimg

        # Remove product of ASN table...
        if fileutil.findFile(_fname): os.remove(_fname)

    else:
        # Check to see if input is a _RAW file
        # If it is, strip off the _raw.fits extension...
        _indx = string.find(inFilename,'_raw')
        if _indx < 0: _indx = len(inFilename)
        # ... and build the CALACS product rootname.
        _fname = fileutil.buildRootname(inFilename[:_indx])
        _cal_prodname = inFilename[:_indx]
        # Reset inFilename to correspond to appropriate input for
        # drizzle: calibrated product name.
        inFilename = _fname

        if _fname == None:
            errorMsg = 'Could not find calibrated product!'
            raise Exception,errorMsg

    # Create trailer filenames based on ASN output filename or
    # on input name for single exposures
    _indx_raw = inFile.find('_raw')

    if _indx_raw > 0:
        # Output trailer file to RAW file's trailer
        _trlroot = inFile[:_indx_raw]
    elif _indx_asn > 0:
        # Output trailer file to ASN file's trailer, not product's trailer
        _trlroot = inFile[:_indx_asn]
    else:
        # Default: trim off last suffix of input filename
        # and replacing with .tra
        _indx = inFile.rfind('_')
        if _indx > 0:
            _trlroot = inFile[:_indx]
        else:
            _trlroot = inFile

    _trlfile = _trlroot + '.tra'

    #
    # If we want to run CALACS, do it here...
    #
    if runcal:
        # Now run CALACS
        print _timestamp('CALACS started ')

        # Print out human readable time-stamp
        print 'Processing %s started at %s' % (inFilename, _getTime())

        # define temporary file to capture STDERR output
        _cal_err = inFilename+'_cal_tra.stderr'
        try:
            # Run CALACS with output going to log file
            iraf.calacs.quiet = "yes"
            iraf.calacs(inFilename,Stderr=_cal_err)
            """
            _ferr = open(_cal_err)
            _errlines = _ferr.readlines()

            for l in _errlines:
                if l.lower().find('error') > -1:
                    raise Exception, str(errorobj)
            _ferr.close()
            """
            # If we have no errors...
            os.remove(_cal_err)
            print _timestamp('CALACS completed ')

        except Exception, errorobj:
            _appendTrlFile(_trlfile,_cal_err)
            errorMsg= str(errorobj)+'\nERROR: CALACS could not finish processing '+inFilename+'\n'
            raise Exception, errorMsg


    # Open product and read keyword value
    # Check to see if product already exists...
    dkey = 'DRIZCORR'
    dkey_old = 'DITHCORR'
    if fileutil.findFile(_fname):
        # If product exists, try to read in value
        # of DITHCORR keyword to guide processing...
        _fimg = pyfits.open(_fname)
        _phdr = _fimg['PRIMARY'].header

        if _phdr.has_key(dkey) > 0:
            dcorr = _phdr[dkey]
        else:
            # Try the old 'DITHCORR' keyword name instead
            if _phdr.has_key(dkey_old) > 0:
                dcorr = _phdr[dkey_old]
                dkey = dkey_old
            else:
                dcorr = None

        # Done with image
        _fimg.close()
        del _fimg
    else:
        # ...if product does NOT exist, interrogate the first member of ASN
        # to find out whehter 'dcorr' has been set to PERFORM
        if _mname :
            _fimg = pyfits.open(fileutil.buildRootname(_mname,ext=['_raw.fits']))
            _phdr = _fimg['PRIMARY'].header
            if _phdr.has_key(dkey) > 0:
                dcorr = _phdr[dkey]
            else:
                # Try the old 'DITHCORR' keyword name instead
                if _phdr.has_key(dkey_old) > 0:
                    dcorr = _phdr[dkey_old]
                    dkey = dkey_old
                else:
                    dcorr = None
            _fimg.close()
            del _fimg
        else:
            dcorr = None

    time_str = _getTime()
    _tmptrl = _trlroot + '_tmp.tra'
    _drizfile = _trlroot + '_pydriz.tra'

    if dcorr == 'PERFORM' and rundriz == yes:
        # Run the new MultiDrizzle package definition
        iraf.multidrizzle()
        # Determine actual input needed for MultiDrizzle
        # If runcal==yes, then we need the product as input
        if runcal:
            _infile = fileutil.buildRootname(_cal_prodname)
            # Update _cal_prodname with full final filename
            _cal_prodname = _infile
        else:
            _infile = inFilename
            _cal_prodname = _infile

        # Run PyDrizzle and send its processing statements to _trlfile
        _pyver = mdrizzle.__version__
        # Create trailer marker message for start of PyDrizzle processing
        _trlmsg = _timestamp('MultiDrizzle started ')
        _trlmsg = _trlmsg+ __trlmarker__
        _trlmsg = _trlmsg + '%s: Processing %s with MultiDrizzle Version %s\n' % (time_str,_infile,_pyver)
        print _trlmsg

        # Write out trailer comments to trailer file...
        ftmp = open(_tmptrl,'w')
        ftmp.writelines(_trlmsg)
        ftmp.close()
        _appendTrlFile(_trlfile,_tmptrl)

        _pyd_err = _trlroot+'_pydriz.stderr'

        try:
            # Run mdrizzle through the IRAF interface now...
            iraf.unlearn('mdrizzle')
            iraf.mdrizzle.clean = yes
            iraf.mdrizzle.mdriztab = yes
            iraf.mdrizzle(output=_trlroot,input=_infile,Stdout=_drizfile,Stderr=_pyd_err)
            os.remove(_pyd_err)
        except Exception, errorobj:
            _appendTrlFile(_trlfile,_drizfile)
            _appendTrlFile(_trlfile,_pyd_err)
            _ftrl = open(_trlfile,'a')
            _ftrl.write('ERROR: Could not complete MultiDrizzle processing of %s.\n' % _infile)
            _ftrl.write(str(sys.exc_type)+': ')
            _ftrl.writelines(str(errorobj))
            _ftrl.write('\n')
            _ftrl.close()
            print 'ERROR: Could not complete MultiDrizzle processing of %s.' % _infile
            raise Exception, str(errorobj)

        # Now, append comments created by PyDrizzle to CALACS trailer file
        print 'Updating trailer file %s with MultiDrizzle comments.' % _trlfile
        _appendTrlFile(_trlfile,_drizfile)

        # Save this for when PyFITS can modify a file 'in-place'
        # Update calibration switch
        _fimg = pyfits.open(_cal_prodname,mode='update')
        _fimg['PRIMARY'].header.update(dkey,'COMPLETE')
        _fimg.close()
        del _fimg

    else:
        # Create default trailer file messages when PyDrizzle is not
        # run on a file.  This will typically apply only to BIAS,DARK
        # and other reference images.
        # Start by building up the message...
        _trlmsg = _timestamp('MultiDrizzle skipped ')
        _trlmsg = _trlmsg + __trlmarker__
        _trlmsg = _trlmsg + '%s: MultiDrizzle processing not requested for %s.\n' % (time_str,inFilename)
        _trlmsg = _trlmsg + '       MultiDrizzle will not be run at this time.\n'
        print _trlmsg

        # Write message out to temp file and append it to full trailer file
        ftmp = open(_tmptrl,'w')
        ftmp.writelines(_trlmsg)
        ftmp.close()
        _appendTrlFile(_trlfile,_tmptrl)

    _fmsg = None
    # Append final timestamp to trailer file...
    _final_msg = '%s: Finished processing %s \n' % (time_str,inFilename)
    _final_msg += _timestamp('MultiDrizzle completed ')
    _trlmsg += _final_msg
    ftmp = open(_tmptrl,'w')
    ftmp.writelines(_trlmsg)
    ftmp.close()
    _appendTrlFile(_trlfile,_tmptrl)

    # Provide feedback to user
    print _final_msg


def _appendTrlFile(trlfile,drizfile):
    """ Append drizfile to already existing trlfile from CALACS.
    """
    # Open already existing CALACS trailer file for appending
    ftrl = open(trlfile,'a')
    # Open PyDrizzle trailer file
    fdriz = open(drizfile)

    # Read in drizzle comments
    _dlines = fdriz.readlines()

    # Append them to CALACS trailer file
    ftrl.writelines(_dlines)

    # Close all files
    ftrl.close()
    fdriz.close()

    # Now, clean up PyDrizzle trailer file
    os.remove(drizfile)


def _timestamp(_process_name):
    """Create formatted time string recognizable by OPUS."""
    _prefix= time.strftime("%Y%j%H%M%S-I-----",time.localtime())
    _lenstr = 60 - len(_process_name)
    return _prefix+_process_name+(_lenstr*'-')+'\n'

def _getTime():
    # Format time values for keywords IRAF-TLM, and DATE
    _ltime = time.localtime(time.time())
    time_str = time.strftime('%H:%M:%S (%d-%b-%Y)',_ltime)

    return time_str

if __name__ == "__main__":
    _main()
