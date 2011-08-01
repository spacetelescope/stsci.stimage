"""
getphotpars includes utilities for calculating the photometry keywords
PHOTZPT, PHOTFLAM, PHOTPLAM, and PHOTBW for a given obsmode and IMPHTTAB.
The calculations are performed in the same way here as they are in hstcal
pipelines.

To calculate a single set of keywords use the function `get_phot_pars`.

If you are calculating for several obsmodes from a single IMPHTTAB file it's best
to use the `GetPhotPars` class. For example::

  get_phot = GetPhotPars(imphttab)
  for obs in obsmodes:
    photzpt, photflam, photplam, photbw = get_phot.get_phot_pars(obs)
    ...
  get_phot.close()

"""

import pyfits
import numpy as np

import _computephotpars

__version__ = '0.1.1'
__vdate__ = '01-Aug-2011'

class ImphttabError(StandardError):
  """
  Class for errors associated with the imphttab file.
  
  """
  pass

def get_phot_pars(obsmode, imphttab):
  """
  Return PHOTZPT, PHOTFLAM, PHOTPLAM, and PHOTBW for specified obsmode
  and imphttab.
  
  Parameters
  ----------
  obsmode : str
    Complete obsmode string including any parameterized values.
    
    Example obsmodes are:
      'acs,wfc1,f625w,f660n'
      'acs,wfc1,f625w,f814w,MJD#55000.0'
      'acs,wfc1,f625w,fr505n#5000.0,MJD#55000.0'
    
  imphttab : str
    Path and filename of IMPHTTAB reference file.
    
  Returns
  -------
  photzpt : float
    PHOTZPT from IMPHTTAB header.
    
  photflam : float
    Interpolated PHOTFLAM for `obsmode`.
    
  photplam : float
    Interpolated PHOTPLAM for `obsmode`.
    
  photbw : float
    Interpolated PHOTBW for `obsmode`.
  
  """
  get_phot = GetPhotPars(imphttab)
  
  photzpt, photflam, photplam, photbw = get_phot.get_phot_pars(obsmode)
  
  get_phot.close()
  
  return photzpt, photflam, photplam, photbw

class GetPhotPars(object):
  """
  This object can be used to get photometry parameters from a given
  IMPHTTAB reference file. Initialize with the name of an IMPHTTAB
  reference file and then call this class or the get_phot_pars
  method with a complete obsmode to get the photometry parameters.
  
  Example obsmodes are:
    'acs,wfc1,f625w,f660n'
    'acs,wfc1,f625w,f814w,MJD#55000.0'
    'acs,wfc1,f625w,fr505n#5000.0,MJD#55000.0'
  
  Parameters
  ----------
  imphttab : str
    Filename and path of IMPHTTAB reference file.
    
  Attributes
  ----------
  imphttab_name : str
    Filename and path of IMPHTTAB reference file. Same as input `imphttab`.
    
  imphttab_fits : `pyfits.HDUList`
    Open `pyfits.HDUList` object from `imphttab`.
  
  """
  
  def __init__(self,imphttab):
    
    self.imphttab_name = imphttab
    self.imphttab_fits = pyfits.open(imphttab,'readonly')
    
  def get_phot_pars(self,obsmode):
    """
    Return PHOTZPT, PHOTFLAM, PHOTPLAM, and PHOTBW for specified obsmode.
    
    Parameters
    ----------
    obsmode : str
      obsmode string
      
    Returns
    -------
    photzpt : float
      PHOTZPT from `imphttab_fits` header.
      
    photflam : float
      Interpolated PHOTFLAM for `obsmode`.
      
    photplam : float
      Interpolated PHOTPLAM for `obsmode`.
      
    photbw : float
      Interpolated PHOTBW for `obsmode`.
    
    """
    
    npars, strp_obsmode, par_dict = self._parse_obsmode(obsmode)
    
    par_struct = self._make_par_struct(npars, par_dict)
    
    result_dict = {}
    
    for par in ('photflam','photplam','photbw'):
      row = self._get_row(strp_obsmode, par)
      
      row_struct = self._make_row_struct(row,npars)
      
      result_dict[par] = self._compute_value(row_struct, par_struct)
      
    photzpt = self.imphttab_fits[0].header['photzpt']
    
    return (photzpt, result_dict['photflam'],
            result_dict['photplam'], result_dict['photbw'])
    
  def close(self):
    """
    Close `imphttab_fits` attribute.
    
    """
    self.imphttab_fits.close() 
    
  def _parse_obsmode(self,obsmode):
    """
    Return number of parameterized variables in `obsmode` and obsmode string
    with the values of the parameterized variables removed. Also returns 
    a dictionary in which the keys are the names of the parameterized variables
    and the dictionary values are are the values of the parameterized variables.
    
    Parameters
    ----------
    obsmode : str
      obsmode string
      
    Returns
    -------
    npars : int
      Number of parameterized variables in `obsmode`.
      
    strp_obsmode : str
      Obsmode with parameterized values removed and converted to lower case.
      Retains order of input.
      
    pars : dict
      Keys are parameterized variable names and values are the parameterized
      variable values. Keys are all lower case.
    
    """
    
    npars = obsmode.count('#')
    
    strp_obsmode = ''
    pars = {}
    
    for mode in obsmode.split(','):
      if '#' not in mode:
        strp_obsmode += mode + ','
      else:
        hashind = mode.index('#')
        
        strp_obsmode += mode[:hashind+1] + ','
        
        pars[mode[:hashind+1].lower()] = float(mode[hashind+1:])
        
    # remove trailing comma and convert to lower case
    strp_obsmode = strp_obsmode[:-1].lower()
        
    return npars, strp_obsmode, pars
    
  def _get_row(self,obsmode,ext):
    """
    Return the `pyfits.FITS_rec` object corresponding to the table row from extension
    ext that has matching `obsmode`.
    
    Parameters
    ----------
    obsmode : str
      obsmode string.
      
    ext : str or int
      Specifier of FITS table extension from which to return a row.
      
    Returns
    -------
    row : `pyfits.FITS_record`
      Row matching input `obsmode`.
      
    Raises
    ------
    ImphttabError
      If `obsmode` does not appear in `imphttab_fits` or it appears multiple
      times in `imphttab_fits`.
    
    """
    o = np.char.strip(np.char.lower(self.imphttab_fits[ext].data['obsmode']))
    w = np.where(o == obsmode.lower())
    
    if len(w[0]) == 0:
      raise ImphttabError('Obsmode %s does not appear in %s extension %s' % 
                          (obsmode, self.imphttab_name, ext))
    elif len(w[0]) > 1:
      raise ImphttabError('Obsmode %s appears multiple times in %s extension %s' %
                          (obsmode, self.imphttab_name, ext))
    else:
      return self.imphttab_fits[ext].data[w]
      
  def _make_row_struct(self,row,npars):
    """
    Construct a dictionary corresponding to the PhtRow C structure used
    in the _computephotpars extension.
    
    Parameters
    ----------
    row : `pyfits.FITS_record`
      A single row from an IMPHTTAB extension.
      
    npars : int
      Number of parameterized variables for this row.
      
    Returns
    -------
    row_struct : dict
      Dictionary corresponding to the PhtRow C structure used in the
      `_computephotpars` extension. Has the same keys as PhtRow structure
      members. All keys are lower case.
    
    """
    row_struct = {}
    
    row_struct['obsmode'] = row['obsmode'][0]
    row_struct['datacol'] = row['datacol'][0]
    row_struct['parnames'] = [row['par%inames' % (i)][0] for i in xrange(1,npars+1)]
    row_struct['parnum'] = npars
    if npars == 0:
      row_struct['results'] = row[row['datacol'][0]]
      row_struct['telem'] = 1
    else:
      row_struct['results'] = row[row['datacol'][0]][0].tolist()
      row_struct['telem'] = len(row_struct['results'])
    row_struct['nelem'] = [row['nelem%i' % (i)][0] for i in xrange(1,npars+1)]
    row_struct['parvals'] = [row['par%ivalues' % (i)][0].tolist() 
                            for i in xrange(1,npars+1)]
                            
    return row_struct
    
  def _make_par_struct(self,npars,par_dict):
    """
    Construct a dictionary corresponding to (part of) the PhotPar structure
    used in the `_computephotpars` extension. Not all members of the structure
    are used in the extension so we supply the necessary ones here.
    
    Parameters
    ----------
    npars : int
      Number of parameterized variables for this obsmode.
      
    par_dict : dict
      Keys are parameterized variable names and values are the parameterized
      variable values. Keys are all lower case. As returned by `_parse_obsmode`.
      
    Returns
    -------
    par_struct : dict
      Dictionary with keys corresponding to some of the structure members
      of the PhotPar structure used in the `_computephotpars` extension.
    
    """
    par_struct = {}
    
    par_struct['npar'] = npars
    par_struct['parnames'] = par_dict.keys()
    
    par_struct['parvals'] = []
    for k in par_struct['parnames']:
      par_struct['parvals'].append(par_dict[k])
      
    return par_struct
    
  def _compute_value(self,row_struct,par_struct):
    """
    Compute a photometry parameter based on an obsmode and a given row from
    the IMPHTTAB reference file.
    
    Parameters
    ----------
    row_struct : dict
      Dictionary corresponding to the PhtRow C structure used in the
      `_computephotpar` extension. Has the same keys as PhtRow structure
      members. All keys are lower case.
      Should be the same as returned by `_make_row_struct`.
      
    par_struct : dict
      Dictionary with keys corresponding to some of the structure members
      of the PhotPar structure used in the `_computephotpars` extension.
      Should be the same as returned by `_make_par_struct`.
      
    Returns
    -------
    result : float
      Result returned by `_computephotpars.compute_value`.
    
    """
    return _computephotpars.compute_value(row_struct, par_struct)
