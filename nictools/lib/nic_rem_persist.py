#! /usr/bin/env python
#
# Authors: Dave Grumm, based on IDL routine by Adam Riess
# Program: nic_rem_persist.py
# Purpose: routine to remove persistence from NICMOS data
# History: 11/19/07 - first version
#        : 11/26/07 - incorporates dq arrays

import pyfits
import numpy as N
import glob, sys
import opusutil
from optparse import OptionParser
import persutil

__version__ = "1.0 (2007 Nov 19)"

ERROR_RETURN = -2
DQ_ARRAY_SIZE = 256

class NicRemPersist:
    """ Remove persistence from NICMOS data (on which pedsub has been run) using a medianed persistence model. 

    pyraf example:
       nrp = NicRemPersist( persist_lo, used_lo, persist_model, persist_mask, verbosity):
       NicRemPersist.persist( nrp )

    command line example:
       hal> ./nic_rem_persist.py -q persist_lo used_lo 'persistring.fits' 'persist_mask.fits'

    """

    def __init__( self, verbosity=1 ,persist_lo=0.2, used_lo=0.5, persist_model= 'persistring.fits', persist_mask= 'persist_mask.fits'): 
        """constructor

        @param persist_lo: minimum allowed value of the persistence  
        @type persist_lo:  Float32
        @param used_lo: minimum allowed value of the fraction of pixels used  
        @type used_lo:  Float32
        @param persist_model: filename containing persistence frame (ring median of)
        @type persist_model:  string
        @param persist_mask: filename containing pixel mask 
        @type persist_mask:  string
    
        """
        self.persist_lo = persist_lo
        self.used_lo = used_lo
        self.persist_model = persist_model
        self.persist_mask = persist_mask
        self.verbosity = verbosity

    def persist( self ):
        """ remove persistence due to the full bright Earth.    

        """

        persist_lo = self.persist_lo 
        used_lo = self.used_lo 
        persist_model = self.persist_model 
        persist_mask = self.persist_mask 
        verbosity = self.verbosity

        clip = 4. # number of std to use in sigma clipping  

        file_list = glob.glob('*ped.fits') # to get all *ped.fits in current working directory 
        num_files = len(file_list)

        if (verbosity > 0):
            print 'The' , num_files, 'ped files are ' , file_list

        sky = N.zeros( num_files, dtype=N.float64 )
        noise = N.zeros( num_files, dtype=N.float64 )
        tot = N.zeros( num_files, dtype=N.float64 )
        persist = N.zeros( num_files, dtype=N.float64 )
        used = N.zeros( num_files, dtype=N.float64 )

        try:  # read the image for the ring median of the persistence frame
           fh_per_median = pyfits.open( self.persist_model )
        except:    
           opusutil.PrintMsg("F"," ERROR "+ str('Unable to open medianed persistence model file') + str(persist_model))
           sys.exit( ERROR_RETURN)
        ps = fh_per_median[0].data 

        try:  # read the image for the mask
           fh_per_mask = pyfits.open( self.persist_mask )
        except:    
           opusutil.PrintMsg("F"," ERROR "+ str('Unable to open the mask file') + str(persist_mask))
           sys.exit( ERROR_RETURN)
           
        per_mask = fh_per_mask[0].data  # persistence mask data (0 is good, non-0 is bad)

        for ii in range( num_files ):  # calculate and remove persistence from each file

          fh_pedsub = pyfits.open(file_list[ii])
          out_hdr = fh_pedsub[1].header

          im = fh_pedsub[1].data    #  load file which has had pedsub run on it      
          med_mask = im *0  # for size; needed for median call

          dq_data = fh_pedsub[3].data  # read the dq array data (0 is good, non-0 is bad)
          dq_plus_per_mask = dq_data + per_mask # combined mask

    #     load a ring median version of the this ped file
          f_prefix = file_list[ii].split('.')[0]
          rm_file = f_prefix + str( '.rm.fits' )
          fh_rm = pyfits.open( rm_file )
          imrm = fh_rm[0].data

          src = im - imrm  # this image is zero except for sources
          
          md, std = iterstatc( clip, src ) # returns median, standard deviation

          level = median(im, med_mask)

    #     sigma clipping sources and rejecting bright regions >1.5*median and masked part
    #     only used pixels with clip standard deviations or less than 150% of the median;
    #     using combined mask
          gd = ((src < md+clip*std) & (src > md-clip*std) & (im < 1.5*level) & (dq_plus_per_mask == 0)) 

          num_gd = N.add.reduce( N.add.reduce( gd )) 
          used[ ii ] =  float(num_gd)/float( src.shape[0]* src.shape[1]) 

    #     set up the matrices for a linear system of equations
          y = im[gd]   #this is the image
          ng = len(y)
          x = N.zeros( [ng,2], dtype=N.float64 ) 
          x[:,0] = 1.0     # this part of the model is a flat sky
          x[:,1] = ps[gd]  # this part of the model is an image, i.e., the  persistence image

    #     inverts matrix to solve for the parameters       
          delta = N.dot(N.dot(N.linalg.inv( N.dot( N.transpose(x),x)),N.transpose(x)),y) 

          sky[ ii ] = delta[0]        # parameter one, sky
          persist[ ii ] = delta[1]    # parameter two, level of persistence
          correct = im - delta[1]*ps  # remove a fraction of the persistence image

          if (verbosity > 1):
              print 'For file number: ' , ii
              print ' sky =', sky[ii], ' persist = ' , persist[ii],' used =',used[ii]

          outfile = file_list[ ii ].split('.')[0] + str('x.fits')

    #     write correction if fraction of pixels used and calculated persistence exceed thresholds
          if ( persist[ ii ] < self.persist_lo ):
              if (verbosity > 0):
                  print ' The calculated persistence for this file is ', persist[ii],', which is below the minimum allowed value of the persistence (', persist_lo,').  Therefore no corrected image will be written.'
          elif ( used[ ii ] < self.used_lo ):
              if (verbosity > 0):
                  print ' The fraction of pixels for this file is ', used[ii],', which is below the minimum allowed value of the fraction (', used_lo,').  Therefore no corrected image will be written.'
          else:
              out_hdr.update( key = 'PERSIST', value = persist[ii], comment = "amount of persistence removed" ) 
              write_to_file( correct, outfile, out_hdr, verbosity )

        if (verbosity > 0):
          print 'Done!'
    # end of def persist

    def print_pars(self ):
        """ Print parameters used.
        """
        print 'The parameters used are :'
        print '  Lower limit on persistence (persist_lo):  ' , self.persist_lo
        print '  Lower limit on fraction of pixels used (used_lo):  ' , self.used_lo
        print '  The (medianed) persistence model from the file (persist_model):  ' , self.persist_model
        print '  The mask from the file (persist_mask): ' ,self.persist_mask

      
def write_to_file(data, filename, hdr, verbosity ):
    """ Write the specified data to the specified file with the specified header

    @param data:  array of values [2d]
    @type data:  Float64
    @param filename: output filename  
    @type filename:  string
    @param hdr: header to be written filename  
    @type hdr: string
    @param verbosity: level of verbosity
    @type hdr: int   
    """

    fimg = pyfits.HDUList()
    if hdr <> None:
       fimghdu = pyfits.PrimaryHDU( header = hdr)
    else:
       fimghdu = pyfits.PrimaryHDU()
    fimghdu.data = data
    fimg.append(fimghdu)
    fimg.writeto(filename)
    if (verbosity > 0):
        print ' persistence-subtracted data written to: ',filename

def iterstatc( clip, d ):
    """  version of nicmos iterstat
    
    @param clip: number of std to use in sigma clipping  
    @type clip:  Float32
    @param d: array of values to sigma clip
    @type clip:  Float32
    @return:  clipped mean, clipped std
    @rtype:  float, float    
    """

    img = d
    md = img.mean()
    std = img.std()
    for ii in range(6): 
      gd = N.where((img < md+clip*std) & (img > md-clip*std))
      md = img[gd].mean()
      std = img[gd].std()
   
    return md, std


# Return the median of the array y, ignoring masked elements.
#
def median( y, mask):
    """Return the median of the array y, ignoring masked elements.

    @param y:  array of values [2d]
    @type y:  Float32
    @param mask:  array of ones or zeros (0 indicates a good value) [2d]
    @type mask:  Int32
    @return:  median of y, ignoring masked elements
    @rtype:  float
    """

   # make the arrays 1d :
    yshape =  y.shape[0]* y.shape[1]
    y_1d = N.resize(y, yshape)
    labels = N.where( mask == 0, 1, 0)      # 1 indicates a good value
    labels_1d = N.resize(labels, yshape)

    y_ok = N.compress( labels_1d, y_1d)

    leny = len( y_ok)
    if leny < 1:
        return None
    elif leny == 1:
        return y_ok[0]
    elif leny == 2:
        return (y_ok[0] + y_ok[1]) / 2.

    index = y_ok.argsort()
    half_leny = leny // 2
    if half_leny * 2 < leny:
        return y_ok[index[half_leny]]
    else:
        return (y_ok[index[half_leny]] + y_ok[index[half_leny+1]]) / 2.

# end of median()


def main( cmdline):
    """Read arguments, and call NicRemPersist

    The command-line options are:
        -q (quiet)
        -v (very verbose)

    @param cmdline: command-line arguments
    @type cmdline: list of strings
    """

    usage = "usage:  %prog [options] inputfile"
    parser = OptionParser( usage)

    parser.set_defaults( verbosity = persutil.VERBOSE)

    parser.add_option( "-q", "--quiet", action = "store_const",
            const = persutil.QUIET, dest = "verbosity",
            help = "quiet, print nothing")
    parser.add_option( "-v", "--verbose", action="store_const",
            const = persutil.VERY_VERBOSE, dest="verbosity",
            help="very verbose, print lots of information")

    (options, args) = parser.parse_args()

    persutil.setVerbosity( options.verbosity)  
    verbosity = options.verbosity

    if ( len(args) > 0 ):
       persist_lo = float(args[0])
    else:
       persist_lo = 0.2 # default which should probably be changed
    if ( len(args) > 1 ):
       used_lo = float(args[1])
    else:
       used_lo = 0.5    # default which should probably be changed
    if ( len(args) > 2 ):
       persist_model = args[2]
    else:
       persist_model = 'persistring.fits' # default which should probably be changed
    if ( len(args) > 3 ):
       persist_mask = args[3]
    else:
       persist_mask = 'persist_mask.fits' # default which should probably be changed

    try:
       nrp = NicRemPersist( verbosity, persist_lo, used_lo, persist_model, persist_mask) 
       if (verbosity >=1 ):
            nrp.print_pars()
       NicRemPersist.persist( nrp )

       del nrp

    except Exception, errmess:
       opusutil.PrintMsg("F","ERROR "+ str(errmess))
       sys.exit( ERROR_RETURN)

if __name__ == "__main__":

    # Process
    main( sys.argv[1:])


