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
#              --> [temp, sigma, winner ]= tfb.calctemp()
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
#                     S(tate),B(lind),Q(uietest); default = None
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
#
#              The routine will successfully calculate the temperature only if TFBCALC='PERFORM' and camera=3
#
import os.path
import sys, time
from optparse import OptionParser
import tfbutil
import opusutil
import numpy as N
import pyfits
import pytools 
from pytools import parseinput 


# Coefficients that are in the file 'temp_from_bias.define' as of 10/12/07; these need to instead
# be in a (fits) reference file.
# Define some constants
c2_min = 3.6218723e-06; c2_max = -3.6678544e-07; c3_min = 9.0923490e-11; c3_max = -4.1401650e-11    # used in bad pixel clipping
p3_c2_0 = 4037.6680; p3_c2_1 = 1.1533126; p3_c2_sigma = 0.1                                         # camera 3, algorithm 2
p3_c2_slope = 37.0; p3_c2_offset = 75.15                                                            # camera 3, algorithm 2
pt3_2_0 = 153.25747; pt3_2_1 = 0.0037115404; pt3_3_0 = 151.03888; pt3_3_1 = 0.0036755942            # camera 3, algorithm 3
p3_c3_sigma = 0.25                                                                                  # camera 3, algorithm 3

__version__ = "1.6"   

ERROR_RETURN = 2 

class CalTempFromBias:
    """Calculate the temperature from the bias for a given filename.

    example:
       tfb = CalTempFromBias( filename, edit_type=edit_type, hdr_key=hdr_key, err_key=err_key,
             nref_par=nref_par, force=force, noclean=noclean, verbosity=verbosity)
       [temp, sigma, winner ]= tfb.calctemp()
       tfb.print_pars()
       stat = tfb.update_header( temp, sigma, winner)         
    """
    def __init__( self, input_file, edit_type=None, hdr_key=None, err_key=None, nref_par=None,
                  force=None, noclean=False, verbosity=0):
        """constructor

        @param input_file: name of the file or filelist to be processed 
        @type input_file: string
        @type edit_type: string
        @param edit_type: type of file to update
        @param hdr_key: name of keyword to update in file
        @type hdr_key: string
        @param err_key: name of keyword for error estimate
        @type err_key: string
        @param nref_par: name of the directory containing the nonlinearity file
        @type nref_par: string
        @param force: name of algorithm whose value is to be returned,
                      regardless of which algorithm had the lowest estimated sigma.
        @type force: string
        @param noclean: flag to force use of UNCLEANed 0th read.
        @type noclean: string that is either True or False
        @param verbosity: verbosity level (0 for quiet, 1 verbose, 2 very verbose)
        @type verbosity: string
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
        self.verbosity = verbosity
        self.tfb_version = __version__
        self.tfb_run = time.asctime()

        outlist = parseinput.parseinput(input_file) ### extract the files in the list
        self.num_files =  parseinput.countinputs(input_file)[0]
        self.outlist0 = outlist[0]

        if (self.verbosity >1):
           print ' Temp_from_bias run on ',self.tfb_run, ', version: ', self.tfb_version 


    def calctemp(self): 
        """ Calculate the temperature from the bias for the given input file
        @return: temp, sig, winner
        @rtype: float, float, int
        """

        self.temp_list = []
        self.sigma_list = []
        self.winner_list = []
        self.nonlinfile_list = []

        for ii in range(self.num_files):

            temp = 0.0
            edit_type = self.edit_type        
            hdr_key = self.hdr_key
            err_key = self.err_key
            nref_par = self.nref_par
            noclean = self.noclean
            filename = self.outlist0[ii]        
            force = self.force
            verbosity = self.verbosity

            if ( verbosity > 0):
              print '   ' ; print '   '
              print 'Calculating temp for file ',ii,' : ', filename

           # get header
            try:
               fh_raw = pyfits.open( filename, mode='update')
            except:
               opusutil.PrintMsg("F","ERROR "+ str('Unable to open input file') + str(filename))   

            raw_header = fh_raw[0].header
            tfbcalc_key = raw_header['TFBCALC'] 
            tfbcalc = tfbcalc_key.lstrip().rstrip() # strip leading and trailing whitespace 

            if ( tfbcalc <> 'PERFORM'):
               print ' The keyword TFBCALC is not set to PERFORM, so will not calculate the temperature '
               fh_raw.close() 
               continue 

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
            nl_file = nonlinfile_key.split('nref$')[1]
            nonlin_file = os.path.join( nref, nl_file)

            try:
               fh_nl = pyfits.open( nonlin_file )
               self.nonlin_file = nonlin_file            
            except:
               opusutil.PrintMsg("F","ERROR "+ str('Unable to open nonlinearity file') + str(nonlin_file))
               raw_header.update("TFBCALC", "SKIPPED")
               self.nonlin_file = None
               raise IOError,'Unable to open nonlinearity file' 
               return None, None, None 

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
               raw_header.update('TFBCALC', 'SKIPPED') 
               fh_raw.close()  
               return None, None, None

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

                if not threshold:
                   threshold = 10.0  # in DN/s. Every 5 DN/s here is 5*0.203 = 1 DN in the quad median.

                if (N.median(signal*0.203) > threshold ):
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
            rawquads = quadmed(clean, border=5)

            if (self.verbosity >1):
                print ' The results of the quadmed call:'
                print rawquads

    # This is the start of cascade to select the most optimal temp algorithm 

    #  ALGORITHM 2. Blind correction. 
            quads = rawquads

            if (( camera == 1) | (camera ==2)):      
               print ' Cameras 1 and 2 not currently supported.'
               raw_header.update('TFBCALC', 'SKIPPED')
               fh_raw.close()     
               return None, None, None

            if ( camera == 3): 
               w = [ p3_c2_0, p3_c2_1 ] 

               quads[3] = (-(quads[3] - poly(quads[0],w)))
               temp2 = (quads[3]/p3_c2_slope)+ p3_c2_offset 

               sigma2 = p3_c2_sigma   # this is an empirical quantity determined from EOL data   
                                      # it is used for comparison at the bottom of the cascade

    # ALGORITHM 3. Quietest-quad method. 
    #       (no attempt at state removal - just use the quad(s) with smallest LVPS amplitudes)
            quads = rawquads

            if ( camera == 3): 
                 #  For NIC3, avg of quads 2 and 3 is best. RMS after avg is 39 DN (0.14 K)
               qtemps_3 = [poly(quads[1],[ pt3_2_0, pt3_2_1]),poly(quads[2],[ pt3_3_0, pt3_3_1])]

               temp3 = N.mean(qtemps_3)
               sigma3 = p3_c3_sigma 

    # Compare the error estimates of algorithms 2 and 3 and return the best one; the 
    #   lowest sigma is the winner.
            winner = 2  
            temp = temp2
            sig = sigma2

            if (sigma3 < sigma2):  
               temp = temp3
               sig = sigma3
               winner = 3
    # This is the end of cascade to select the most optimal temp algorithm 

    #  verbose output, if requested
            if (self.verbosity >1):
                print '**************************************************************************************'
                print ' '            
                print '    Camera: ',camera 
                print '                                               Q1         Q2         Q3         Q4'
                print 'Temp from Algorithm 2 (blind-correction):  ',temp2

                if ( camera == 3):
                    print 'Temp(s) from Algorithm 3 (quietest-quad):             ',qtemps_3[0],'  ',qtemps_3[1]
                print '**************************************************************************************'
                print '   '
                print '     Algorithm 2: ',temp2,' (K) +/- ',sigma2,' (sigma)'
                print '     Algorithm 3: ',temp3,' (K) +/- ',sigma3,' (sigma)'
                print '     ----------------------------------------------------------'

                if (force == None):
                    print '     The algorithm selected (not forced)  by this routine is ',winner,'.'
                print '  '
                print '**************************************************************************************'

    #  If the force keyword is set, force the output to return the value
    #    from that particular algorithm, even if it wasn't the best.

            if (force <> None and force.upper()[0] == "S"):   
                print ' This algorithm is not supported.'
                raw_header.update('TFBCALC', 'SKIPPED') 
                fh_raw.close()  
                return None, None, None

            if (force <> None and force.upper()[0] == "B"):
                if (self.verbosity >1):
                   print 'Forcing Algorithm 2 (Blind correction) result to be returned, at your request...'
                winner = 2
                temp = temp2
                sig = sigma2

            if (force <> None and force.upper()[0] == "Q"):
                if (self.verbosity >1):
                   print 'Forcing Algorithm 3 (Quietest-quad) result to be returned, at your request...'
                winner = 3
                temp = temp3 
                sig = sigma3

            if (force <> None):
                if (self.verbosity >1):
                   print '... which gives temp = ' , temp,' and sigma = ' , sig

            self.temp_list.append( temp )
            self.sigma_list.append( sig )
            self.winner_list.append( winner ) 
            self.nonlinfile_list.append( nonlin_file )

            # close any open file handles
            if fh_raw:
              fh_raw.close()
            if fh_nl:
              fh_nl.close()


        return self.temp_list, self.sigma_list, self.winner_list

## end of def calctemp()


    def update_header(self, temp, sig, winner, edit_type=None, hdr_key=None,
                      err_key=None):  

        
        """ Update header
        @param temp: calculated temperature
        @type temp: float
        @param sig: standard deviation of calculated temperature
        @type sig: float
        @param winner: algorithm selected by script
        @type winner: int
        @param edit_type: type of file to be updated
        @type edit_type: string
        @param hdr_key: name of keyword to update in file
        @type hdr_key: string
        @param err_key: name of keyword for error estimate
        @type err_key: string
        @return: status (not None for failure due to key not being specified)
        @rtype: int
        """

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

            if (winner == 2):# 'blind-correction'  
               meth_used = "BLIND CORRECTION"
            else: # (winner == 3):# 'quietest-quad'
               meth_used = "QUIETEST-QUAD"

            comm = str('Temp from bias, sigma=')+str(sig)+str(' (K)')

            filename =  this_file

            # update either the RAW or SPT file
            if (edit_type[0] =="S"): # SPT
               underbar = filename.find('_')
               filename =  filename[:underbar] +'_spt.fits'
            fh = pyfits.open( filename, mode='update' )      
            hdr = fh[0].header # NOTE - this the PRIMARY extension

            try :
                hdr.update(hdr_key, temp[ii])
                hdr.update(err_key, sig[ii])
                hdr.update("TFBMETH", meth_used)
                hdr.update("TFBDATE", self.tfb_run)
                hdr.update("TFBVER", self.tfb_version)
                hdr.update("TFBCALC", "COMPLETE")
                fh.close()
            except :
                hdr.update("TFBCALC", "SKIPPED")
                fh.close()
                pass
        
        if ( self.verbosity > 0):
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
        print '  verbosity: ' ,  self.verbosity
        print ' For the files given by the input file list: '
        for ii in range(self.num_files):
           try :
                this_file = self.outlist0[ii]        
                print '    input_file[',ii,']:  ' , this_file
                this_nl_file = self.nonlinfile_list[ii]                    
                print '    nonlinearity file[',ii,']: ' ,  this_nl_file
           except :
                pass 


#******************************************************************
# This quadmed does not have the following parameters that are in the idl version: section,
#    avg, mask, and calculates mean only ( not median ).
def quadmed( im, border):
    """  This function computes the mean in the 4 quadrants of an input NxM array

    @param im: input rectangular array
    @type im: numpy array
    @param border: border size (in pixels) aroud the perimeter of each quad to be excluded from the mean
    @type border: int
    
    """ 
#
# Following Eddie's convention, the quads are numbered as follows:
#
#    |------|-----|
#    |  Q4  |  Q3 |
#    |      |     |   as seen in the standard NICMOS/HST data format.
#    |------|-----|
#    |  Q1  |  Q2 |
#    |      |     |
#    |------|-----|
#
# optionally, you can specify a border, in pixels, around the perimeter of EACH QUAD
# to be excluded from the mean

    xsize = im.shape[1]
    ysize = im.shape[0]

    quads = N.zeros((4), dtype=N.float64)

    if not border:
      border = 0

    x1 = 0 + border
    x2 = ((xsize/2)-1) - border
    x3 = (xsize/2) + border
    x4 = (xsize-1) - border
    y1 = 0 + border
    y2 = ((ysize/2)-1) - border
    y3 = (ysize/2) + border
    y4 = (ysize-1) - border

    quads[0] = N.mean(im[y1:y2+1,x1:x2+1])
    quads[1] = N.mean(im[y1:y2+1,x3:x4+1])
    quads[2] = N.mean(im[y3:y4+1,x3:x4+1])
    quads[3] = N.mean(im[y3:y4+1,x1:x2+1])

    return  quads


def poly( var_x, coeffs):
    """ Return linear polynomial with given coefficients.

    @param var_x: input argument
    @type var_x: scalar, vector, or array
    @param coeffs: vector of polynomial coefficients
    @type coeffs: float

    """
    poly_out = coeffs[ 0 ] + coeffs[ 1 ]* var_x
    
    return poly_out

if __name__=="__main__":
    """Get input file and other arguments, and call CalTempFromBias.

    The command-line options are:
        -q (quiet)
        -v (very verbose)

    @param cmdline: command-line arguments
    @type cmdline: list of strings
    """
 
    usage = "usage:  %prog [options] inputfile"
    parser = OptionParser( usage)

    if ( sys.argv[1] ): filename = sys.argv[1]

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

    (options, args) = parser.parse_args()

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

    tfbutil.setNref(options.nref_dir )
    if options.nref_dir!=None: nref_dir = options.nref_dir

    tfbutil.setForce(options.force )
    force = options.force 

    try :
         tfb = CalTempFromBias( filename, edit_type=edit_type, hdr_key=hdr_key, err_key=err_key,
                                nref_par=nref_dir, force=force, noclean=noclean, verbosity=verbosity)
         [temp, sigma, winner ]= tfb.calctemp()

         if ([temp, sigma, winner ] != [None, None, None]):             
             stat = tfb.update_header( temp, sigma, winner)
             if ( (stat == None )and (verbosity > 0)):
               tfb.print_pars() 

    except Exception, errmess:
         opusutil.PrintMsg("F","ERROR "+ str(errmess))





