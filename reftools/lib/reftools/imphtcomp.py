"""
Tools for comparing  two IMPHTTAB tables from the same instrument and detector.

"""

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

import pyfits

import pysynphot as S
  
__version__ = '1.0.2'
__vdate__ = '25-Jul-2011'

# general error class for this module
class ImphtcompError(StandardError):
  pass


class _ImphttabData(object):
  def __init__(self,data):
    self.names = []
  
    for n in data.names:
      if n.lower() not in ('pedigree','descrip'):
        self.names.append(n.lower())
        self.__dict__[n.lower()] = data[n].copy()
        
        
class _ImphttabHolder(object):
  def __init__(self,fits):
    self.photflam = _ImphttabData(fits['PHOTFLAM'].data)
    self.photplam = _ImphttabData(fits['PHOTPLAM'].data)
    self.photbw = _ImphttabData(fits['PHOTBW'].data)

    
class ImphttabComp(object):
  """
  Class for comparing two IMPHTTAB tables from the same instrument and detector.
  
  Parameters
  ----------
  tab1 : str
    Filename of first IMPHTTAB for comparison.
  
  tab2 : str
    Filename of second IMPHTTAB for comparison.
    
  Attributes
  ----------
  tab1_name : str
    Filename of first IMPHTTAB.
  
  tab2_name : str
    Filename of second IMPHTTAB.
  
  modes : ndarray
    Obsmodes present in both input files.
  
  flams1 : ndarray
    PHOTFLAM values from `tab1` for obsmodes in `modes`.
    
  plams1 : ndarray
    PHOTPLAM values from `tab1` for obsmodes in `modes`.

  bws1 : ndarray
    PHOTBW values from `tab1` for obsmodes in `modes`.
  
  flams2 : ndarray
    PHOTFLAM values from `tab2` for obsmodes in `modes`.
    
  plams2 : ndarray
    PHOTPLAM values from `tab2` for obsmodes in `modes`.

  bws2 : ndarray
    PHOTBW values from `tab2` for obsmodes in `modes`.
  
  flamdiff : ndarray
    Percent differences between `flams1` and `flams2` calculated as 
    (`flams1` - `flams2`) / `flams1`.
  
  plamdiff : ndarray
    Percent differences between `plams1` and `plams2` calculated as 
    (`plams1` - `plams2`) / `plams1`.
  
  bwdiff : ndarray
    Percent differences between `bws1` and `bws2` calculated as 
    (`bws1` - `bws2`) / `bws1`.
  
  """
  def __init__(self,tab1,tab2):
    self.tab1_name = tab1
    self.tab2_name = tab2
    
    self._read_tables(tab1,tab2)
    
    self._compare_tables()
    
  def _read_tables(self,tab1,tab2):
    fits1 = pyfits.open(tab1,'readonly')
    fits2 = pyfits.open(tab2,'readonly')
    
    # check that some import header values match, especially instrume/detector
    self._check_headers(fits1[0].header,fits2[0].header)
    
    # check that the two tables have at least some obsmodes in common
    # and report any differences
    self._check_obsmodes(fits1[1].data['obsmode'],fits2[1].data['obsmode'])
    
    # we've made sure these tables can be compared, now read and store
    # the data.
    self.tab1 = _ImphttabHolder(fits1)
    self.tab2 = _ImphttabHolder(fits2)
    
    fits1.close()
    fits2.close()
    
  def _check_headers(self,head1,head2):
    """
    Check that some important header keywords match. An ImphtcompError is
    raised if they do not match.
    
    """
    # check that we've got an IMPHTTAB
    if 'DBTABLE' not in head1 or head1['dbtable'] != 'IMPHTTAB':
      raise ImphtcompError('{} is not a valid IMPHTTAB table.'.format(self.tab1_name))
    if 'DBTABLE' not in head2 or head2['dbtable'] != 'IMPHTTAB':
      raise ImphtcompError('{} is not a valid IMPHTTAB table.'.format(self.tab2_name))
      
    if head1['instrume'] != head2['instrume']:
      s = 'IMPHTTAB tables are not for the same instrument. File1: {}, File2: {}.'
      s = s.format(head1['instrume'],head2['instrume'])
      raise ImphtcompError(s)
      
    if head1['detector'] != head2['detector']:
      s = 'IMPHTTAB tables are not for the same detector. File1: {}, File2: {}.'
      s = s.format(head1['detector'],head2['detector'])
      raise ImphtcompError(s)
    
    
  def _check_obsmodes(self,obs1,obs2):
    """
    Check that two obsmode lists have at least some modes in common
    and report any differences. If there are no modes in common an
    ImphtcompError is raised.
    
    Prints nothing and returns None if the obsmode lists are identical.
    Prints differing obsmodes and returns None if the tables appear consistent
    but have some differences.
    
    """
    # check for identical lists
    if obs1.shape == obs2.shape and (np.char.equal(obs1,obs2)).all():
      return None
      
    # modes in file1 but not in file2
    only1 = [o for o in obs1 if o not in obs2]
    
    # modes in file2 but not in file1
    only2 = [o for o in obs2 if o not in obs1]
    
    if only1:
      print('Modes appearing only in {}:'.format(self.tab1_name))
      for o in only1:
        print('\t{}'.format(o))
      print('')
      
    if only2:
      print('Modes appearing only in {}:'.format(self.tab2_name))
      for o in only2:
        print('\t{}'.format(o))
      print('')
      
    return None
    
  def _compare_tables(self):
    """
    Compare things as (table1 - table2) / table1.
    
    """
    flam1 = self.tab1.photflam
    plam1 = self.tab1.photplam
    bw1 = self.tab1.photbw

    flam2 = self.tab2.photflam
    plam2 = self.tab2.photplam
    bw2 = self.tab2.photbw
    
    modes = []
    flams1 = []
    plams1 = []
    bws1 = []
    
    flams2 = []
    plams2 = []
    bws2 = []
    
    flamdiff = []
    plamdiff = []
    bwdiff = []
    
    # go through all the modes in first table and finding matching rows
    # in the second table
    for i,mode in enumerate(flam1.obsmode):
      if mode not in flam2.obsmode:
        continue
      
      # there's a good chance the matching obsmode is on the same row in both
      # files so check for that to save calling where
      if i < len(flam2.obsmode) and flam2.obsmode[i] == mode:
        w = i
      else:
        w = np.where(flam2.obsmode == mode)[0][0]
      
      # no parameterized variables
      if mode.count('#') == 0:
        modes.append(mode)
        flams1.append(flam1.photflam[i])
        flams2.append(flam2.photflam[w])
        flamdiff.append((flams1[-1] - flams2[-1]) / flams1[-1])
        
        plams1.append(plam1.photplam[i])
        plams2.append(plam2.photplam[w])
        plamdiff.append((plams1[-1] - plams2[-1]) / plams1[-1])
        
        bws1.append(bw1.photbw[i])
        bws2.append(bw2.photbw[w])
        bwdiff.append((bws1[-1] - bws2[-1]) / bws1[-1])
      
      # one parameterized variable
      elif mode.count('#') == 1:
        if flam1.nelem1[i] != flam2.nelem1[w]:
          print('''Mode {} does not have matching number of parameterized elements 
                in each file. File 1: {} File 2: {}'''.format(mode,flam1.nelem1[i],
                                                              flam2.nelem1[w]))
          continue
        
        # expand the obsmodes with parameterized variable values and save
        # the result for each expanded obsmode
        for j,f in enumerate(flam1.par1values[i]):
          temp_mode = mode.replace(flam1.par1names[i].lower(),
                                   flam1.par1names[i].lower() + str(f))
                                   
          modes.append(temp_mode)
          flams1.append(flam1.photflam1[i][j])
          flams2.append(flam2.photflam1[w][j])
          flamdiff.append((flams1[-1] - flams2[-1]) / flams1[-1])
          
          plams1.append(plam1.photplam1[i][j])
          plams2.append(plam2.photplam1[w][j])
          plamdiff.append((plams1[-1] - plams2[-1]) / plams1[-1])
          
          bws1.append(bw1.photbw1[i][j])
          bws2.append(bw2.photbw1[w][j])
          bwdiff.append((bws1[-1] - bws2[-1]) / bws1[-1])
      
      # two parameterized variables
      elif mode.count('#') == 2:
        if flam1.nelem1[i] != flam2.nelem1[w]:
          print('''Mode {} does not have matching number of first parameterized elements 
                in each file. File 1: {} File 2: {}'''.format(mode,flam1.nelem1[i],
                                                              flam2.nelem1[w]))
          continue
          
        if flam1.nelem2[i] != flam2.nelem2[w]:
          print('''Mode {} does not have matching number of second parameterized elements 
                in each file. File 1: {} File 2: {}'''.format(mode,flam1.nelem2[i],
                                                              flam2.nelem2[w]))
          continue
          
        # expand the obsmodes with parameterized variable values and save
        # the result for each expanded obsmode  
        for j in xrange(flam1.nelem2[i]):
          for k in xrange(flam1.nelem1[i]):          
            temp_mode = mode.replace(flam1.par1names[i].lower(),
                                     flam1.par1names[i].lower() + str(flam1.par1values[k]))
            temp_mode = temp_mode.replace(flam1.par2names[i].lower(),
                                          flam1.par2names[i].lower() + str(flam1.par2values[j]))
                                     
            modes.append(temp_mode)
            flams1.append(flam1.photflam2[i][flam1.nelem2[i]*j + k])
            flams2.append(flam2.photflam2[w][flam1.nelem2[i]*j + k])
            flamdiff.append((flams1[-1] - flams2[-1]) / flams1[-1])
            
            plams1.append(plam1.photplam2[i][flam1.nelem2[i]*j + k])
            plams2.append(plam2.photplam2[w][flam1.nelem2[i]*j + k])
            plamdiff.append((plams1[-1] - plams2[-1]) / plams1[-1])
            
            bws1.append(bw1.photbw2[i][flam1.nelem2[i]*j + k])
            bws2.append(bw2.photbw2[w][flam1.nelem2[i]*j + k])
            bwdiff.append((bws1[-1] - bws2[-1]) / bws1[-1])
            
    self.modes = np.array(modes,dtype=np.string_)
    self.flams1 = np.array(flams1,dtype=np.float)
    self.plams1 = np.array(plams1,dtype=np.float)
    self.bws1 = np.array(bws1,dtype=np.float)
    
    self.flams2 = np.array(flams2,dtype=np.float)
    self.plams2 = np.array(plams2,dtype=np.float)
    self.bws2 = np.array(bws2,dtype=np.float)
    
    self.flamdiff = np.array(flamdiff,dtype=np.float)
    self.plamdiff = np.array(plamdiff,dtype=np.float)
    self.bwdiff = np.array(bwdiff,dtype=np.float)
        
  def print_diffs(self,orderby='photflam',lines=25):
    """
    Print obsmodes and parameters ordered by orderby parameter, with the
    largest absolute differences in that parameter at the top. This is for
    seeing which obsmodes have the largest difference in the specified 
    parameter. Prints the number of modes given in the lines parameter.
    
    Differences shown are calculated as 100 * (table1 - table2)/table1.
    
    Parameters
    ----------
    orderby : str, optional
      The parameter by which to order the printed results, with modes having
      the largest absolute difference in this parameter printed at the top. 
      
      May be one of: 'photflam', 'photplam', or 'photbw'. An ImphtcompError
      is raised if the input does not match one of these.
      
      Defaults to 'photflam'.
      
    lines : int, optional
      The number of lines to print. Defaults to 25.
      
    Raises
    ------
    ImphtcompError
      If `orderby` does not match a valid option.
    
    """
    modes = self.modes
    flams1 = self.flams1
    plams1 = self.plams1
    bws1 = self.bws1
    
    flams2 = self.flams2
    plams2 = self.plams2
    bws2 = self.bws2
    
    flamdiff = self.flamdiff * 100
    plamdiff = self.plamdiff * 100
    bwdiff = self.bwdiff * 100
    
    if orderby.lower() == 'photflam':
      order = np.abs(flamdiff).argsort()[::-1]
    elif orderby.lower() == 'photplam':
      order = np.abs(plamdiff).argsort()[::-1]
    elif orderby.lower() == 'photbw':
      order = np.abs(bwdiff).argsort()[::-1]
    else:
      raise ImphtcompError("Unregnized orderby keyword. " +
                            "Must be one of 'photflam','photplam','photbw'.")
      
    s = '{:<45}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}'
    
    print(s.format('OBSMODE','PHOTFLAM1','PHOTFLAM2','PHOTFLAM DIFF',
                    'PHOTPLAM1','PHOTPLAM2','PHOTPLAM DIFF',
                    'PHOTBW1','PHOTBW2','PHOTBW DIFF'))
                    
    s = '{:<45}{:<15.5e}{:<15.5e}{:< 15.8f}{:<15.5f}{:<15.5f}{:< 15.8f}{:<15.5f}{:<15.5f}{:< 15.8f}'
                    
    for i in order[:lines]:
      print(s.format(modes[i],flams1[i],flams2[i],flamdiff[i],
                      plams1[i],plams2[i],plamdiff[i],
                      bws1[i],bws2[i],bwdiff[i]))
                      
  def make_plot(self,outname='imphttab_comp.pdf'):
    """
    Make a plot with histograms of the percent differences between
    PHOTFLAM, PHOTPLAM, and PHOTBW for the IMPHTTAB tables.
    
    Differences plotted are 100 * (table1 - table2) / table1.
    
    Parameters
    ----------
    outname : str, optional
      Filename of output plot, including extension.
      Defaults to 'imphttab_comp.pdf'.
    
    """
    spars = SubplotParams(left=0.05,bottom=0.15,right=0.95,top=0.85,
                          wspace=0.22,hspace=0.3)
    fig = plt.figure(figsize=(12,4),subplotpars=spars)
    
    tab1 = os.path.basename(self.tab1_name)
    tab2 = os.path.basename(self.tab2_name)
    fig.suptitle('{}, {}'.format(tab1,tab2))
    
    flamax = fig.add_subplot(131)
    plamax = fig.add_subplot(132)
    bwax = fig.add_subplot(133)
    
    flamax.set_title('PHOTFLAM',size='small')
    plamax.set_title('PHOTPLAM',size='small')
    bwax.set_title('PHOTBW',size='small')
    
    axlist = (flamax,plamax,bwax)
    
    for ax in axlist:
      ax.set_ylabel('# Obsmodes')
      ax.set_xlabel('% Diff.')
      ax.get_xaxis().set_major_locator(MaxNLocator(nbins=4))
      ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
      
    flamax.hist(self.flamdiff*100,bins=20)
    plamax.hist(self.plamdiff*100,bins=20)
    bwax.hist(self.bwdiff*100,bins=20)
    
    print('Saving file ' + outname)
    fig.savefig(outname)
                      
                      
def print_table_diffs(table1,table2,orderby='photflam',lines=25):
  """
  Compare two IMPHTTAB tables and print their differences to the terminal.
  Prints any obsmodes which are in either table but not in both.
  
  Also prints the obsmodes and parameters for the modes which most differ
  in the parameter given in `orderby`. This is for seeing which obsmodes have the 
  largest percent difference in the specified  parameter. Prints the number of 
  modes given in the `lines` parameter.
    
  Differences shown are calculated as 100 * (table1 - table2)/table1.
  
  Parameters
  ----------
  table1 : str
    Filename of first IMPHTTAB for comparison.
    
  table2 : str
    Filename of the second IMPHTTAB for comparison.
    
  orderby : str, optional
    This specifies one of 'photflam', 'photplam', 'photbw', or 'all'.
    The printed results are ordered according to the absolute difference in the
    specified parameter, with the mode with the largest absolute difference at
    the top of the list. 
    
    Specifying 'all' will print 3 tables, one ordered by each of the parameters.
    
    Defaults to 'photflam'.
    
  lines : int, optional
    Number of lines of differences to print. Defaults to 25.
    
  Raises
  ------
  ImphtcompError
    If `orderby` does not match a valid option.
  
  """
  if orderby.lower() not in ('photflam','photplam','photbw','all'):
    raise ImphtcompError("Unrecognized orderby keyword. " +
                          "Must be one of 'photflam','photplam','photbw','all'.")
  
  comp = ImphttabComp(table1,table2)
  
  if orderby.lower() != 'all':
    comp.print_diffs(orderby,lines)
  else:
    for par in ('photflam','photplam','photbw'):
      comp.print_diffs(par,lines)
      print('')
      
def plot_table_diffs(table1,table2,outname='imphttab_comp.pdf'):
  """
  Make a plot with histograms of the percent differences between
  PHOTFLAM, PHOTPLAM, and PHOTBW for the IMPHTTAB tables.
  
  Differences plotted are 100 * (table1 - table2) / table1.
  
  Parameters
  ----------
  table1 : str
    Filename of first IMPHTTAB for comparison.
    
  table2 : str
    Filename of the second IMPHTTAB for comparison.
    
  outname : str, optional
    Filename of output plot, including extension.
    Defaults to 'imphttab_comp.pdf'.
  
  """
  comp = ImphttabComp(table1,table2)
  
  comp.make_plot(outname)
  