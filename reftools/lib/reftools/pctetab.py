"""
Functions for ACS PCTETAB reference file.

:Authors: Pey Lian Lim, Matt Davis

:Organization: Space Telescope Science Institute

:History:
    * 2010-08-31 PLL created this module.
    * 2010-11-09 PLL added RN2_NIT keyword and
      updated documentation.
    * 2011-04-25 MRD updated for new CTE algorithm parameters
    * 2011-07-18 MRD updated to handle time dependence
    * 2011-11-29 MRD updated with column-by-column CTE scaling

Examples
--------
>>> from reftools import pctetab
>>> pctetab.MakePCTETab('pctetab_pcte.fits','pctetab_dtdel_110425.txt',
                        'pctetab_chgleak_110425.txt',
                        'pctetab_levels_110425.txt',
                        'pctetab_scaling_110718.txt',
                        'pctetab_column_scaling_111129.txt',
                        history_file='pctetab_history.txt')

"""

# External modules
import os, glob, numpy, pyfits

__version__ = '1.1.1'
__vdata__ = '25-Jan-2012'

# generic exception for errors in this module
class PCTEFileError(StandardError):
  pass


class _Text2Fits(object):
  """
  Helper class for making the CTE parameters file (PCTETAB) from a
  collection of data saved in text files. The resulting fits file will
  have information in the primary header and in at least four table extensions.
  
  """
  
  def __init__(self):
    self.header = None
    self.dtde = None
    self.charge_leak = []
    self.levels = None
    self.scale = None
    self.col_scale = None
    self.out_name = None
    
  def make_header(self, out_name, sim_nit, shft_nit, read_noise, noise_model,
                  oversub_thresh, nchg_leak, useafter, pedigree, creatorName,
                  history_file, detector):
    """
    Make the primary extension header for the pctetab.
    
    Parameters
    ----------
    out_name : str
      Name of pcte fits file being created. May include path.
      
    sim_nit : int
      Value for ``SIM_NIT`` keyword in PCTEFILE header.
      Number of iterations of readout simulation per column.
      
    shft_nit : int
      Value for ``SHFT_NIT`` keyword in PCTEFILE header.
      Number of shifts each readout simulation is broken up into.
      A large number means pixels are shifted a smaller number of rows
      before the CTE is evaluated again.
      
    read_noise : float
      Value for ``RN_CLIP`` keyword in PCTEFILE
      EXT 0. This is the maximum amplitude of read noise
      used in the read noise mitigation. Unit is in electrons.
    
    noise_model : {0, 1, 2}
      Select the method to be used for readnoise removal.
      
      0: no read noise smoothing
      1: standard smoothing
      2: strong smoothing
    
    oversub_thresh : float
      Value for ``SUBTHRSH`` keyword in PCTEFILE header. CTE corrected
      pixels taken below this value are re-corrected. Unit is in electrons.
      
    useafter : str
        Value for ``USEAFTER`` keyword.

    pedigree : str
        Value for ``PEDIGREE`` keyword.

    creatorName : str
        Name of the person generating this `fitsFile`.

    historyFile : str
        ASCII file containing ``HISTORY`` lines for
        EXT 0. Include path. Each row will produce one
        ``HISTORY`` line.

    detector : str
        Supported detector.
    
    """
    
    self.out_name = out_name
    
    self.header = pyfits.PrimaryHDU()
    
    self.header.header.update('ORIGIN', 'STScI-STSDAS/TABLES')
    self.header.header.update('FILENAME', os.path.basename(out_name))
    self.header.header.update('FILETYPE', 'PIXCTE')
    self.header.header.update('TELESCOP', 'HST')
    self.header.header.update('USEAFTER', useafter)
    self.header.header.update('PEDIGREE', pedigree)
    self.header.header.update('DESCRIP', 'Parameters needed for pixel-based CTE correction ------------------') # Must have 67 char
    self.header.header.add_comment('= \'Created or updated by %s\'' % creatorName, before='ORIGIN')
    self.header.header.update('NCHGLEAK', nchg_leak, comment='number of chg_leak extensions')

    # This is detector-specific
    if detector == 'WFC':
      self.header.header.update('INSTRUME', 'ACS')
      self.header.header.update('DETECTOR', 'WFC')
    else:
      raise PCTEFileError('Detector not supported: ' + unicode(detector))
    # End if
    
    # Optional HISTORY
    if os.path.isfile(history_file):
      fin = open(history_file,'r')
      for line in fin: 
        self.header.header.add_history(line[:-1])
      fin.close()
    # End if
    
    # the number of readout simulations done per column
    self.header.header.update('SIM_NIT', int(sim_nit),
                              'number of readout simulations done per column')
    
    # the number of shifts each column readout simulation is broken up into
    self.header.header.update('SHFT_NIT', int(shft_nit),
                              'the number of shifts each column readout simulation is broken up into')

    # read noise level
    self.header.header.update('RN_CLIP', float(rn_clip),
                              'Read noise level in electrons.')
    
    # read noise smoothing algorithm
    self.header.header.update('NSEMODEL', int(noise_model),
                              'Read noise smoothing algorithm.')
    
    # over-subtraction correction threshold
    self.header.header.update('SUBTHRSH', float(oversub_thresh),
                              'Over-subtraction correction threshold.')
                              
  def make_dtde(self,dtde_file):
    """
    Make fits extension containing the dtde data that describes the
    marginal loss to CTE at the charge levels given in the file.
    
    The input file should have two columns with headers DTDE and Q, in that order.
    The first column is the dtde data and the second is the corresponding
    charge level for that dtde value.
    
    The file should have format:
    
    DTDE  Q
    float int
    ...   ...
    
    Lines beginning with # are ignored.
    
    Parameters
    ----------
    dtde_file : str
      Path to text file containing dtde data.
    
    """
    
    if not os.path.isfile(dtde_file):
      raise IOError('Invalid dtde file: ' + unicode(dtde_file))
      
    lRange, colName, colData, colForm, colUnit = 0, {}, {}, {}, {}
    
    # read in dtde data from text file
    fin = open(dtde_file,'r')
    
    for line in fin:
      # skip comments
      if line[0] == '#':
        continue
        
      row = line.split()
      
      # column names
      if row[0] == 'DTDE':
        colRange = range(len(row))
        for i in colRange:
          colName[i] = row[i]
          colData[i] = []
        continue
        
      # data
      for i in colRange:
        colData[i].append(row[i])
        
    # done reading data
    fin.close()
    
    # convert data to numpy arrays
    colData[0] = numpy.array(colData[0], dtype=numpy.float32)
    colForm[0] = 'E'
    colUnit[0] = ''
    
    colData[1] = numpy.array(colData[1], dtype=numpy.int32)
    colForm[1] = 'J'
    colUnit[1] = 'DN/S'
    
    c0 = pyfits.Column(name=colName[0], format=colForm[0], array=colData[0])
    c1 = pyfits.Column(name=colName[1], format=colForm[1], unit=colUnit[1], array=colData[1])
    
    self.dtde = pyfits.new_table(pyfits.ColDefs([c0,c1]))
    self.dtde.header.update('EXTNAME','DTDE')
    self.dtde.header.update('DATAFILE',dtde_file,comment='data source file')
    
  def make_charge_leak(self,chg_leak_file,num):
    """
    Make fits extension containing parameterization of CTE losses along
    the CTE tail and across different charge levels.
    
    The input file should contain 5 columns with following format:
    
    NODE LOG_Q_1 LOG_Q_2 LOG_Q_3 LOG_Q_4
    int  float   float   float   float 
    ...  ...     ...     ...     ...
    
    Lines beginning with # are ignored.
    
    Parameters
    ----------
    chg_leak_file : str
      Path to text file containing charge leak data.
      
    num : int
      number to append to extension name since there may be more than
      one charge_leak extension.
    
    """
    
    if not os.path.isfile(chg_leak_file):
      raise IOError('Invalid charge leak file: ' + unicode(chg_leak_file))
      
    colRange, colName, colData, colForm, colUnit = 0, {}, {}, {}, {}
    
    mjd1 = None
    mjd2 = None
    
    # read in dtde data from text file
    fin = open(chg_leak_file,'r')
    
    for line in fin:
      # skip comments
      if line[0] == '#':
        continue
        
      row = line.split()
      
      # MJD parameters
      if row[0] == 'MJD1':
        mjd1 = float(row[1])
      elif row[0] == 'MJD2':
        mjd2 = float(row[1])
      
      # column names
      elif row[0] == 'NODE':
        colRange = range(len(row))
        for i in colRange:
          colName[i] = row[i]
          colData[i] = []
        
      # data
      else:
        for i in colRange:
          colData[i].append(row[i])
        
    # done reading data
    fin.close()
    
    # make sure we got our MJD values
    if not mjd1:
      raise PCTEFileError('MJD1 parameter not correctly specified in ' + chg_leak_file)
    elif not mjd2:
      raise PCTEFileError('MJD2 parameter not correctly specified in ' + chg_leak_file)
    
    # Convert data to Numpy arrays
    colData[0] = numpy.array(colData[0], dtype=numpy.int16)
    colForm[0] = 'I'
    colUnit[0] = 'PIXEL'
    
    for i in colRange[1:]:
      colData[i] = numpy.array(colData[i], dtype=numpy.float32)
      colForm[i] = 'E'
      colUnit[i] = 'FRACTION'
    # End of i loop

    # Write to FITS table extension
    tabData = []
    for i in colRange: 
      tabData.append( pyfits.Column(name=colName[i], format=colForm[i], unit=colUnit[i], array=colData[i]) )
    
    self.charge_leak.append(pyfits.new_table(pyfits.ColDefs(tabData)))
    self.charge_leak[-1].header.update('EXTNAME','CHG_LEAK'+str(num))
    self.charge_leak[-1].header.update('MJD1',mjd1,comment='start valid time range for data')
    self.charge_leak[-1].header.update('MJD2',mjd2,comment='end valid time range for data')
    self.charge_leak[-1].header.update('DATAFILE',chg_leak_file,comment='data source file')
    
  def make_levels(self,levels_file):
    """
    Make fits extension containing charge levels at which to evaluate CTE
    losses (as opposed to every level from 0 - 99999).
    
    The input file should have a single column with the following format:
    
    LEVEL
    int
    ...
    
    Columns beginning with # are ignored.
    
    Parameters
    ----------
    levels_file : str
      Text file containing charge levels at which to do CTE evaluation.
    
    """
    
    if not os.path.isfile(levels_file):
      raise IOError('Invalid levels file: ' + unicode(levels_file))
    
    colData = []
    
    # read in data from text file
    fin = open(levels_file,'r')
    
    for line in fin:
      # skip comments
      if line[0] == '#':
        continue
        
      row = line.split()
      
      # column heading
      if row[0] == 'LEVEL':
        colName = row[0]
        continue
        
      colData.append(row[0])
      
    # done reading file
    fin.close()
    
    colData = numpy.array(colData, dtype=numpy.int32)
    colForm = 'J'
    
    c1 = pyfits.Column(name=colName, format=colForm, array=colData)
    
    self.levels = pyfits.new_table(pyfits.ColDefs([c1]))
    self.levels.header.update('EXTNAME','LEVELS')
    self.levels.header.update('DATAFILE',levels_file,comment='data source file')
    
  def make_scale(self, scale_file):
    """
    Make fits extension containing time dependent CTE scaling.
    
    The input file should have two columns with the following format:
      
      MJD     SCALE
      float   float
      ...     ...
      
      Columns beginning with # are ignored.
    
    Parameters
    ----------
    scale_file : str
      Text file containing time dependent CTE scaling parameters.
    
    """
    
    if not os.path.isfile(scale_file):
      raise IOError('Invalid scale file: ' + unicode(scale_file))
    
    lRange, colName, colData, colForm, colUnit = 0, {}, {}, {}, {}
    
    # read in dtde data from text file
    fin = open(scale_file,'r')
    
    for line in fin:
      # skip comments
      if line[0] == '#':
        continue
        
      row = line.split()
      
      # column names
      if row[0] == 'MJD':
        colRange = range(len(row))
        for i in colRange:
          colName[i] = row[i]
          colData[i] = []
        continue
        
      # data
      for i in colRange:
        colData[i].append(row[i])
        
    # done reading data
    fin.close()
    
    # convert data to numpy arrays
    colData[0] = numpy.array(colData[0], dtype=numpy.float32)
    colForm[0] = 'E'
    colUnit[0] = 'DAYS'
    
    colData[1] = numpy.array(colData[1], dtype=numpy.float32)
    colForm[1] = 'E'
    colUnit[1] = 'FRACTION'
    
    c0 = pyfits.Column(name=colName[0], format=colForm[0], unit=colUnit[0], array=colData[0])
    c1 = pyfits.Column(name=colName[1], format=colForm[1], unit=colUnit[1], array=colData[1])
    
    self.scale = pyfits.new_table(pyfits.ColDefs([c0,c1]))
    self.scale.header.update('EXTNAME','CTE_SCALE')
    self.scale.header.update('DATAFILE',scale_file,comment='data source file')
    
  def make_column_scale(self, column_file):
    """
    Make fits extension containing column by column CTE scaling.
    
    The input file should have 5 columns with the following format:
    
    COLUMN  AMPA    AMPB    AMPC    AMPD
    int     float   float   float   float
    ...     ...     ...     ...     ...
    
    Lines beginning with # are ignored.
    
    Parameters
    ----------
    column_file : str
      Text file containing CTE column-by-column scaling.
    
    """
    if not os.path.isfile(column_file):
      raise IOError('Invalid column scale file: ' + unicode(column_file))
    
    lRange, colName, colData, colForm, colUnit = 0, {}, {}, {}, {}
    
    # read in dtde data from text file
    fin = open(column_file,'r')
    
    for line in fin:
      # skip comments
      if line[0] == '#':
        continue
        
      row = line.split()
      
      # column names
      if row[0] == 'COLUMN':
        colRange = range(len(row))
        for i in colRange:
          colName[i] = row[i]
          colData[i] = []
        continue
        
      # data
      for i in colRange:
        colData[i].append(row[i])
        
    # done reading data
    fin.close()
    
    # convert data to numpy arrays
    colData[0] = numpy.array(colData[0], dtype=numpy.int32)
    colForm[0] = 'J'
    colUnit[0] = 'COLUMN NUMBER'
    
    colData[1] = numpy.array(colData[1], dtype=numpy.float32)
    colForm[1] = 'E'
    colUnit[1] = 'FRACTION'
    
    colData[2] = numpy.array(colData[2], dtype=numpy.float32)
    colForm[2] = 'E'
    colUnit[2] = 'FRACTION'
    
    colData[3] = numpy.array(colData[3], dtype=numpy.float32)
    colForm[3] = 'E'
    colUnit[3] = 'FRACTION'
    
    colData[4] = numpy.array(colData[4], dtype=numpy.float32)
    colForm[4] = 'E'
    colUnit[4] = 'FRACTION'
    
    c0 = pyfits.Column(name=colName[0], format=colForm[0],
                       unit=colUnit[0], array=colData[0])
    c1 = pyfits.Column(name=colName[1], format=colForm[1],
                       unit=colUnit[1], array=colData[1])
    c2 = pyfits.Column(name=colName[2], format=colForm[2],
                       unit=colUnit[2], array=colData[2])
    c3 = pyfits.Column(name=colName[3], format=colForm[3],
                       unit=colUnit[3], array=colData[3])
    c4 = pyfits.Column(name=colName[4], format=colForm[4],
                       unit=colUnit[4], array=colData[4])
    
    self.col_scale = pyfits.new_table(pyfits.ColDefs([c0,c1,c2,c3,c4]))
    self.col_scale.header.update('EXTNAME','COL_SCALE')
    self.col_scale.header.update('DATAFILE',column_file,comment='data source file')
    
  def make_fits(self):
    """
    Combine primary and table extensions into an HDU List and save to fits file.
    
    The methods make_header, make_dtde, make_charge, and make_levels must have
    been succesfully run before calling this method.
    
    Raises
    ------
    PCTEFileError
      If any of the necessary extensions have not been made.
    
    """
    
    if not self.header:
      raise PCTEFileError('Fits header has not been prepared: '
                          'call make_header method first.')
    if not self.dtde:
      raise PCTEFileError('DTDE extension has not been prepared: '
                          'call make_dtde method first.')
    if not self.charge_leak:
      raise PCTEFileError('Charge leak extension has not been prepared: '
                          'call make_charge_leak method first.')
    if not self.levels:
      raise PCTEFileError('Levels extension has not been prepared: '
                          'call make_levels method first.')
    if not self.scale:
      raise PCTEFileError('Scale extension has not been prepared: '
                          'call make_scale method first.')
    if not self.col_scale:
      raise PCTEFileError('Column scaline extension has not been prepared: '
                          'call make_column_scale method first')
      
    hduList = pyfits.HDUList([self.header, self.dtde, self.levels, self.scale,
                              self.col_scale] + self.charge_leak)
    
    hduList.writeto(self.out_name, clobber=True)
    
def MakePCTETab(out_name, dtde_file, chg_leak_file, levels_file, scale_file,
                column_file, sim_nit=7, shft_nit=7, read_noise=4.25,
                noise_model=1, oversub_thresh=-15,
                useafter='Mar 01 2002 00:00:00',
                pedigree='INFLIGHT 01/03/2002 22/07/2010',
                creatorName='ACS Team', history_file='', detector='WFC'):
  """
  Make the CTE parameters reference file.
  
  Parameters
  ----------
  out_name : str
    Name of pcte fits file being created. May include path.
    
  dtde_file : str
    Path to text file containing dtde data.
    
    The file should have 2 columns with the following format:
  
    DTDE  Q
    float int
    ...   ...
    
    Lines beginning with # are ignored.
    
  chg_leak_file : str or list of str
    Path to text file(s) containing charge leak data. If passed as a string
    the string may contain wild cards so that multiple files are specified.
    
    The input file should contain 5 columns with following format:
  
    NODE LOG_Q_1 LOG_Q_2 LOG_Q_3 LOG_Q_4
    int  float   float   float   float 
    ...  ...     ...     ...     ...
    
    Lines beginning with # are ignored.
    
  levels_file : str
    Text file containing charge levels at which to do CTE evaluation.
    
    The input file should have a single column with the following format:
  
    LEVELS
    int
    ...
    
    Lines beginning with # are ignored.
    
  scale_file : str
    Text file containing CTE scaling parameters
    
    The input file should have two columns with the following format:
    
    MJD     SCALE
    float   float
    ...     ...
    
    Lines beginning with # are ignored.
    
  column_file : str
    Text file containing CTE column-by-column scaling.
    
    The input file should have 5 columns with the following format:
    
    COLUMN  AMPA    AMPB    AMPC    AMPD
    int     float   float   float   float
    ...     ...     ...     ...     ...
    
    Lines beginning with # are ignored.
    
  sim_nit : int, optional
    Number of iterations of readout simulation per column. 
    Defaults to 5.
    
  shft_nit : int, optional
    Number of shifts each readout simulation is broken up into.
    A large number means pixels are shifted a smaller number of rows
    before the CTE is evaluated again.
    Defaults to 5.
    
  read_noise : float
    Value for ``RN_CLIP`` keyword in PCTEFILE
    EXT 0. This is the maximum amplitude of read noise
    used in the read noise mitigation. Unit is in electrons.
    Defaults to 4.25.
  
  noise_model : {0, 1, 2}
    Select the method to be used for readnoise removal.
    
    0: no read noise smoothing
    1: standard smoothing (default)
    2: strong smoothing
  
  oversub_thresh : float
    Value for ``SUBTHRSH`` keyword in PCTEFILE header. CTE corrected
    pixels taken below this value are re-corrected. Unit is in electrons.
    Defaults to -15.
      
  useafter : str, optional
    Value for ``USEAFTER`` keyword.
    Defaults to 'Mar 01 2002 00:00:00'

  pedigree : str, optional
    Value for ``PEDIGREE`` keyword.
    Defaults to 'INFLIGHT 01/03/2002 22/07/2010'

  creatorName : str, optional
    Name of the person generating this `fitsFile`.
    Defaults to 'ACS Team'

  historyFile : str, optional
    ASCII file containing ``HISTORY`` lines for
    EXT 0. Include path. Each row will produce one
    ``HISTORY`` line.
    Defaults to ''

  detector : str, optional
    Supported detector. Defaults to 'WFC'
      
  Examples
  --------
  >>>MakePCTETab('pctetab_pcte.fits','pctetab_dtdel_110425.txt',
                 'pctetab_chgleak_110425.txt','pctetab_levels_110425.txt',
                 'pctetab_scaling_110718.txt',
                 'pctetab_column_scaling_111129.txt',
                 history_file='pctetab_history.txt')
  Saving file pctetab_pcte.fits
  
  """
  
  # give the output file it's official suffix
  if out_name.find('_pcte.fits') == -1:
    out_name = out_name + '_pcte.fits'
    
  # test for the presence of the input files
  if not os.path.isfile(dtde_file):
    raise IOError('Invalid dtde file: ' + unicode(dtde_file))
  
  if isinstance(chg_leak_file, str):
    chg_leak_file = glob.glob(chg_leak_file)
    
  for f in chg_leak_file:
    if not os.path.isfile(f):
      raise IOError('Invalid charge leak file: ' + unicode(chg_leak_file))
  
  nchg_leak = len(chg_leak_file)
  
  if not os.path.isfile(levels_file):
    raise IOError('Invalid levels file: ' + unicode(levels_file))
    
  if not os.path.isfile(scale_file):
    raise IOError('Invalid scale file: ' + unicode(scale_file))
    
  if not os.path.isfile(column_file):
    raise IOError('Invalid column scaling file: ' + unicode(column_file))
    
  # make Text2Fits object and run it's methods to construct fits extensions
  t2f = _Text2Fits()
  t2f.make_header(out_name, sim_nit, shft_nit, read_noise, noise_model,
                  oversub_thresh, nchg_leak, useafter, pedigree,
                  creatorName, history_file, detector)
  t2f.make_dtde(dtde_file)
  
  for i,f in enumerate(chg_leak_file):
    t2f.make_charge_leak(f,i+1)
  
  t2f.make_levels(levels_file)
  
  t2f.make_scale(scale_file)
  
  t2f.make_column_scale(column_file)
  
  # have t2f save the fits file
  print('Saving file ' + unicode(out_name))
  t2f.make_fits()
  