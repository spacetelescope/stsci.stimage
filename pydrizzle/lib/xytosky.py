import string, copy, os

import pyfits
import numpy as N
from numpy import char as C
from math import *

import pydrizzle,wcsutil

# Convenience definitions...
yes = True
no = False

#################
#
#
#               Interface to IRAF version
#    This version requires 'wcsutil_iraf.py' 
#               __version__= '1.1 (12-Feb-2007)'
#
#
#################
def XYtoSky_pars(input,x=None,y=None,coords=None,colnames=None,linear=yes,
                idckey='IDCTAB',hms=no,output=None,verbose=yes):

    #Determine whether we are working with a single input pos or 
    # a file with multiple positions
    if not coords:             
        # Working with a single position
        xy = (x,y)
    else:
        xy = []
        # Read multiple positions from specified/default columns
        # First, find out what kind of file we are working with: FITS or ASCII.
        if coords.find('.fits') > 0:
            # Find out names of columns to read from the FITS table
            if not colnames:
                # Assume column names of 'X','Y'
                _clist = ['X','Y']
            else:
                _clist = colnames.split(',')
            # Open FITS table and read appropriate columns
            _ftab = pyfits.open(coords)
            #
            # NOTE: assumes columns are listed in order of X,Y!
            #
            x = _ftab[1].data.field(_clist[0]).tolist()
            y = _ftab[1].data.field(_clist[1]).tolist()
            for i in xrange(len(x)):
                xy.append([x[i],y[i]])                
            _ftab.close()
            del _ftab
        else:
            # Open the ASCII table
            _ifile = open(coords)
            _lines = _ifile.readlines()
            _ifile.close()
            
            # Get the column names, if specified
            if colnames:
                _clist = colnames.split(',')
            else:
                _clist = ['1','2']            
            # Convert strings to ints for use in parsing the lines of the file
            for n in xrange(len(_clist)): _clist[n] = string.atoi(_clist[n]) - 1
            # Read in lines and convert appropriate columns to X,Y lists
            for line in _lines:
                line = line.strip()
                # Convert from tab delimited to space delimited
                line.replace('\t',' ')
                if len(line) > 0 and line[0] != '#':
                    _xy = line.split()
                    # Append floating point value from specified column
                    x = string.atof(_xy[_clist[0]])
                    y = string.atof(_xy[_clist[1]])
                    xy.append([x,y])
                    
    #
    # Get ra/dec position or Python list of ra/dec positions
    #
    radec = XYtoSky(input, xy, idckey=idckey, linear=linear, verbose=no)

    # 
    # Now format the output for the user...
    #
    # Break out the values into separate lists...
    if isinstance(radec, type([]) ):
        radd,decdd = [],[]
        for pos in radec:
            radd.append(pos[0])
            decdd.append(pos[1])
        radd = N.array(radd)
        decdd = N.array(decdd)
    else:
        radd = N.array([radec[0]])
        decdd = N.array([radec[1]])
           
    if hms:
        # Convert value(s) to HMS format from decimal degrees
        # Returns as Python lists of string values
        ra,dec = wcsutil.ddtohms(radd,decdd,verbose=verbose)
    else:
        ra,dec = radd,decdd
        # Print out what the user wants to see
        if not output or verbose:
            # If they input a list, but don't specify an output file,
            # the only way they are going to get results is to print them
            # to the screen. 
            for i in xrange(len(ra)):
                print 'RA (deg.) = ',ra[i],', Dec (deg.)= ',dec[i]

    # If user wants to output to a file...
    if output:
        _olines = []
        if output == coords:
            if coords.find('.fits') > 0:
            # User is appending to a FITS table.
                _fout = pyfits.open(coords,'update')
                # We need to touch the data so that it can be in memory for
                # creating the col_defs object
                _fout[1].data
                _tcol = _fout[1].get_coldefs()
                raname = 'RA'
                decname = 'Dec'
                # create columns for RA and Dec arrays
                if hms:
                    # We are working with lists of strings
                    racol = pyfits.Column(name=raname,format='24a',array=C.array(ra))
                    deccol = pyfits.Column(name=decname,format='24a',array=C.array(dec))
                else:
                    # normal numarray objects
                    racol = pyfits.Column(name=raname,format='1d',array=ra)
                    deccol = pyfits.Column(name=decname,format='1d',array=dec)
                    
                # Add columns to table now.
                _tcol.add_col(racol)
                _tcol.add_col(deccol)
                
                # Update table by replacing old HDU with new one
                _thdu = pyfits.new_table(_tcol)
                del _fout[1]
                _fout.append(_thdu)
                _fout.close()
                del _fout
                del _thdu
                del _tcol
                                    
            else:
                # If user wants to append results to an ASCII coord file...
                _olines.insert(0,'# Image: '+input+'\n')
                _olines.insert(0,'# RA Dec positions computed by PyDrizzle\n')
                pos = 0
                for line in _lines:
                    line = line.strip()
                    if line[0] != '#' and line != '':
                        line = line + '    '+ str(ra[pos])+'    '+str(dec[pos])+'\n'
                        pos = pos + 1
                    _olines.append(line)
                _ofile = open(output,'w')
                _ofile.writelines(_olines)
                _ofile.close()

        else:
            # Otherwise, user wants to create new file...
            _olines.append('# RA Dec positions computed by PyDrizzle\n')
            _olines.append('# Image: '+input+'\n')
            if coords:
                for pos in xrange(len(ra)):
                    _str = str(ra[pos])+'    '+str(dec[pos])+'\n'
                    _olines.append(_str)
            else:
                _str = str(ra)+'    '+str(dec)+'\n'
                _olines.append(_str)

            _ofile = open(output,'w')
            _ofile.writelines(_olines)
            _ofile.close()

    return ra,dec


#################
#
#
#               Coordinate Transformation Functions
#
#
#################

def XYtoSky(input, pos, idckey='IDCTAB', linear=yes, verbose=no):
    """ Convert input pixel position(s) into RA/Dec position(s).
        Output will be either an (ra,dec) pair or a 'list' of (ra,dec)
        pairs, not a numarray, to correspond with the input position(s).

        Parameter:
            input - Filename with extension specification of image
            pos   - Either a single [x,y] pair or a list of [x,y] pairs.
            idckey - Keyword which points to the IDC table to be used.
            linear - If no, apply distortion correction for image.

    """

    # Start by making sure we have a valid extension specification.
    _insplit = string.split(input,'[')
    if len(_insplit) == 1:
        raise IOError, 'No extension specified for input image!'

    # Now we need to insure that the input is an array:
    if not isinstance(pos,N.ndarray):
        if N.array(pos).ndim > 1:
            pos = N.array(pos,dtype=N.float64)

    # Set up Exposure object
    _exposure = pydrizzle.Exposure(input,idckey=idckey)

    ra,dec = _exposure.geometry.XYtoSky(pos,linear=linear,verbose=verbose)

    if not isinstance(ra,N.ndarray):
        # We are working with a single input, return single values
        return ra,dec
    else:
        # We are working with arrays, so we need to convert them
        # from 2 arrays with RA in one and Dec in the other to 1 array
        # with pairs of RA/Dec values.
        _radec = N.zeros(shape=(len(ra),2),dtype=ra.dtype)
        _radec[:,0] = ra
        _radec[:,1] = dec

        return _radec.tolist()
