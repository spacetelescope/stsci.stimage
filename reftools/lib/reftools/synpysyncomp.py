"""
Tools for comparing pysynphot and synphot photometry calculations.

"""

import os
import csv
import tempfile

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

import pyfits

import pysynphot as S

from pyraf import iraf
from iraf import stsdas, hst_calib, synphot

__version__ = '1.0.0'
__vdate__ = '25-Jul-2011'


# general error class for this module
class SynPysynCompError(StandardError):
  pass
  

class SynPysynComp(object):
  """
  Compare synphot and pysynphot values for obsmodes listed in the input
  IMPHTTAB. Only obsmodes with three or fewer listed components are compared.
  For example, acs,wfc1,f555w would be processed but not acs,wfc1,f555w,MJD#.
  
  This can be a long process for large numbers of obsmodes so it is recommended
  that users use the CSV functionality of this class to save the results
  to a .csv file and then use the supporting functions print_synpysyn_diffs
  and plot_synpysyn_diffs in this module to investigate them.
  
  Parameters
  ----------
  imphttab : str
    Filename of IMPHTTAB from which to take obsmodes for comparison.
  
  """
  def __init__(self,imphttab):
    self._read_obsmodes(imphttab)
    
    self.spec = S.FlatSpectrum(1,fluxunits='flam')
    self.comp_dict = {}
    
    self.tempdir = tempfile.gettempdir()
  
  def _read_obsmodes(self,imphttab):
    """
    Read obsmodes from IMPHTTAB table. Grabs only obsmodes which are not
    more than 3 elements long.
    
    """
    self.obsmodes = []
    
    f = pyfits.open(imphttab)
    
    for row in f[1].data:
      # no compound obsmodes or too short obsmodes
      if row['obsmode'].count(',') > 2 or row['obsmode'].count(',') == 0:
        continue
    
      if row['obsmode'].count('#') == 0:
        self.obsmodes.append(row['obsmode'].lower())
      else:
        mode = row['obsmode'].lower()
        for i in xrange(row['nelem1']):
          temp_mode1 = mode.replace(row['par1names'].lower(),
                        row['par1names'].lower() + str(row['par1values'][i]))
          
          if 'NELEM2' not in f[1].data.names:
            self.obsmodes.append(temp_mode1)
            continue
          elif row['nelem2'] == 0:
            self.obsmodes.append(temp_mode1)
            continue
            
          for j in xrange(row['nelem2']):
            temp_mode2 = temp_mode1.replace(row['par2names'].lower(),
                          row['par2names'].lower() + str(row['par2values'][j]))
            self.obsmodes.append(temp_mode2)
    
    f.close()
    
    return self.obsmodes
    
  def _filter_modes(self,modes):
    """
    Remove parameterized obsmodes from modes list.
    
    """
    filtered = []
    
    for m in modes:
      if m.count('#') == 0:
        filtered.append(m)
        
    return filtered
    
  def get_pysyn_vals(self,mode):
    """
    Get comparison values from pysynphot.
    
    Parameters
    ----------
    mode : str
      Obsmode string.
      
    Returns
    -------
    ret : dict
      Dictionary containing calculated values.
    
    """
    bp = S.ObsBandpass(mode,component_dict=self.comp_dict)
    obs = S.Observation(self.spec,bp)
    
    photflam = obs.effstim('flam')/obs.effstim('counts')  # URESP
    del obs
    pivot = bp.pivot()          # PIVWV
    rmswidth = bp.photbw()      # BANDW
    avgwave = bp.avgwave()      # AVGWV
    recwidth = bp.rectwidth()   # RECTW
    equwidth = bp.equivwidth()  # EQUVW
    effic = bp.efficiency()     # QTLAM
    
    ret = {}
    ret['flam'] = photflam
    ret['pivot'] = pivot
    ret['rms'] = rmswidth
    ret['avg'] = avgwave
    ret['rect'] = recwidth
    ret['equiv'] = equwidth
    ret['effic'] = effic
    
    return ret
    
  def get_syn_vals(self,mode):
    """
    Get comparison values from synphot.
    
    Parameters
    ----------
    mode : str
      Obsmode string.
      
    Returns
    -------
    ret : dict
      Dictionary containing calculated values.
    
    """
    tmpfits = os.path.join(self.tempdir,'temp.fits')
    
    synphot.bandpar(mode,output=tmpfits,Stdout=1)
    
    fits = pyfits.open(tmpfits)
    d = fits[1].data
    
    photflam = d['uresp'][0]
    pivot = d['pivwv'][0]
    rmswidth = d['bandw'][0]
    avgwave = d['avgwv'][0]
    recwidth = d['rectw'][0]
    equwidth = d['equvw'][0]
    effic = d['qtlam'][0]
    
    fits.close()
    os.remove(tmpfits)
    
    ret = {}
    ret['flam'] = photflam
    ret['pivot'] = pivot
    ret['rms'] = rmswidth
    ret['avg'] = avgwave
    ret['rect'] = recwidth
    ret['equiv'] = equwidth
    ret['effic'] = effic
    
    return ret
    
  def comp_synpysyn(self,mode):
    """
    Returns a dictionary of pysynphot and synphot values and their percent
    differences calculated as (pysynphot - synphot)/synphot.
    
    Parameters
    ----------
    mode : str
      Obsmode string.
      
    Returns
    -------
    comp : dict
      Dictionary containing calculated values.
    
    """
    pysyn = self.get_pysyn_vals(mode)
    irsyn = self.get_syn_vals(mode)
    
    comp = {'obsmode': mode}
    
    for k in pysyn.iterkeys():
      comp['py' + k] = pysyn[k]
      comp['ir' + k] = irsyn[k]
      comp[k + '%'] = (pysyn[k] - irsyn[k])/irsyn[k]
      
    return comp
  
  def calculate_diffs(self,verbose=False):
    """
    Calculate diffs for all obsmodes and return.
    
    Parameters
    ----------
    verbose: bool, optional
      If True, print obsmodes as they are processed. Defaults to False.
    
    Returns
    -------
    res : dict
      Dictionary containing lists of all pysynphot and synphot calculated values
      and their differences calculated as (pysynphot - synphot)/synphot.
    
    """
    # set up return dictionary
    res = {}
    
    if verbose:
      print('Processing mode ' + self.obsmodes[0])
      
    comp = self.comp_synpysyn(self.obsmodes[0])
    
    for k in comp.iterkeys():
      res[k] = [comp[k]]
    
    # now fill result
    for mode in self.obsmodes[1:]:
      if verbose:
        print('Processing mode ' + mode)
      
      comp = self.comp_synpysyn(mode)
      
      for k in comp.iterkeys():
        res[k].append(comp[k])
        
    return res
  
  def write_csv(self,outfile,verbose=False):
    """
    Write a CSV file containing the pysynphot and synphot values for all
    obsmodes and their percent differences calculated as 
    (pysynphot - synphot)/synphot.
    
    Parameters
    ----------
    outfile : str
      Name of file to write to.
      
    verbose : bool, optional
      If True, print obsmodes as they are processed. Defaults to False.
    
    """
    f = open(outfile,'w')
    f.write('obsmode,pyflam,irflam,flam%,pypivot,irpivot,pivot%,')
    f.write('pyrms,irrms,rms%,pyavg,iravg,avg%,pyrect,irrect,rect%,')
    f.write('pyequiv,irequiv,equiv%,pyeffic,ireffic,effic%\n')
    
    wcsv = csv.DictWriter(f,('obsmode',
                             'pyflam','irflam','flam%',
                             'pypivot','irpivot','pivot%',
                             'pyrms','irrms','rms%',
                             'pyavg','iravg','avg%',
                             'pyrect','irrect','rect%',
                             'pyequiv','irequiv','equiv%',
                             'pyeffic','ireffic','effic%'))
                          
    for mode in self.obsmodes:
      if verbose:
        print 'Processing mode ' + mode
      
      comp = self.comp_synpysyn(mode)
      wcsv.writerow(comp)
      
    f.close()


class SynPysynPlot(object):
  """
  Make a plot from a CSV file created by `SynPsynComp.write_csv` illustrating 
  differences between synphot and pysynphot calculated products.
  
  The plots show differences between 7 parameters calculated by both synphot
  and pysynphot. The differences are shown as histograms. The difference plotted
  is calculated as 100 * (pysynphot - synphot) / synphot.
  
  Parameters
  ----------
  csvfile : str
    Filename of input CSV file containing comparison results. Should be a file
    created by `SynPysynComp.write_csv`.
    
  Attributes
  ----------
  fig : `matplotlib.figure.Figure`
    Useful for setting manipulating figure properties such as the title, which
    otherwise defaults to `csvfile`.
  
  """
  def __init__(self,csvfile):
    self._load_csv(csvfile)
    
    spars = SubplotParams(left=0.08,bottom=0.05,right=0.98,top=0.93,
                          wspace=0.22,hspace=0.3)
    self.fig = plt.figure(figsize=(10,10),subplotpars=spars)
    self.fig.suptitle(csvfile)
    
    self._add_axes()
    self._add_data()
    
  def _load_csv(self,csvfile):
    res = read_synpysyn(csvfile)
    
    flam = res['flam%'] * 100.
    pivot = res['pivot%'] * 100.
    rms = res['rms%'] * 100.
    avg = res['avg%'] * 100.
    rect = res['rect%'] * 100.
    equiv = res['equiv%'] * 100.
    effic = res['effic%'] * 100.
    
    del res
    
    self.datalist = (flam,pivot,rms,avg,rect,
                      equiv,effic)
    
  def _add_axes(self):
    fig = self.fig
    
    flamax = fig.add_subplot(331)
    pivotax = fig.add_subplot(332)
    rmsax = fig.add_subplot(333)
    avgax = fig.add_subplot(334)
    rectax = fig.add_subplot(335)
    equivax = fig.add_subplot(336)
    efficax = fig.add_subplot(337)
    
    flamax.set_title('PHOTFLAM',size='small')
    pivotax.set_title('PHOTPLAM',size='small')
    rmsax.set_title('PHOTBW',size='small')
    avgax.set_title('AVGWV',size='small')
    rectax.set_title('RECTW',size='small')
    equivax.set_title('EQUVW',size='small')
    efficax.set_title('QTLAM',size='small')
    
    axlist = (flamax,pivotax,rmsax,avgax,rectax,
              equivax,efficax)
    self.axlist = axlist
              
    for ax in axlist:
      ax.set_ylabel('# Obsmodes')
      ax.set_xlabel('% Diff.')
      ax.get_xaxis().set_major_locator(MaxNLocator(nbins=4))
      ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
      
  def _add_data(self):
    for i in xrange(len(self.datalist)):
      self.axlist[i].hist(self.datalist[i],bins=20)
      
  def save_plot(self,outname='synpysyn_comp.pdf'):
    """
    Save plots to a file.
    
    Parameters
    ----------
    outname : str, optional
      Name of file to save to, including extension.
      Defaults to 'synpysyn_comp.pdf'.
    
    """
    self.fig.savefig(outname)
    

def read_synpysyn(csvfile):
    """
    Read a CSV file created by `SynPysynComp.write_csv` into numpy arrays.
    
    Parameters
    ----------
    csvfile : str
      Filename of CSV to read. Should be a file created by 
      `SynPysynComp.write_csv`.
      
    Returns
    -------
    res : dict
      Contains one field for each column of the CSV file with a numpy array
      in that field.
    
    """
    fieldnames = ('obsmode',
                  'pyflam','irflam','flam%',
                  'pypivot','irpivot','pivot%',
                  'pyrms','irrms','rms%',
                  'pyavg','iravg','avg%',
                  'pyrect','irrect','rect%',
                  'pyequiv','irequiv','equiv%',
                  'pyeffic','ireffic','effic%')
    
    f = open(csvfile,'r')
    
    rcsv = csv.DictReader(f,fieldnames=fieldnames)
    
    res = {}
    
    for field in fieldnames:
      res[field] = []
    
    # skip the first row
    for line in rcsv:
      if line['pyavg'] == 'pyavg':
        continue
      for field in fieldnames:
        res[field].append(line[field])
      
    f.close()
    
    res['obsmode'] = np.array(res['obsmode'],dtype=np.string_)
    for field in fieldnames[1:]:
      res[field] = np.array(res[field],dtype=np.float)
      
    return res
 
       
def plot_synpysyn_diffs(csvfile,outname='synpysyn_comp.pdf'):
  """
  Make and save a plot illustrating differences between parameters calculated
  by both synphot and pysynphot. Data are taken from a CSV file made by
  `SynPysynComp.write_csv`.
  
  Differences are shown as histograms. The difference plotted is 
  calculated as 100 * (pysynphot - synphot) / synphot.
  
  Parameters
  ----------
  csvfile : str
    Filename of input CSV file containing comparison results. Should be a file
    created by `SynPysynComp.write_csv`.
    
  outname : str, optional
      Name of file to save to, including extension.
      Defaults to 'synpysyn_comp.pdf'.
  
  """
  p = SynPysynPlot(csvfile)
  
  print('Saving file ' + outname)
  p.save_plot(outname)
  

def print_synpysyn_diffs(csvfile,orderby='photflam',lines=25):
  """
  Print synphot/pysynphot comparison results from a CSV file produced by
  `SynPysynComp.write_csv` to  the terminal.
  
  Prints the obsmodes and parameters for the modes which most differ
  in the parameter given in `orderby`. This is for seeing which obsmodes have the 
  largest percent difference in the specified  parameter. Prints the number of 
  modes given in the `lines` parameter.
  
  Only prints data for the parameters PHOTFLAM, PHOTPLAM (PIVOT WV), 
  and PHOTBW (RMSWIDTH). Differences are given as 
  100 * (pysynphot - synphot) / synphot.
  
  Parameters
  ----------
  csvfile : str
    Filename of input CSV file containing comparison results. Should be a file
    created by `SynPysynComp.write_csv`.
  
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
  SynPysynCompError
    If `orderby` does not match a valid option.
    
  """
  if orderby.lower() not in ('photflam','photplam','photbw','all'):
    raise SynPysynCompError("Unrecognized orderby keyword. " +
                            "Must be one of 'photflam','photplam','photbw','all'.")
  
  res = read_synpysyn(csvfile)
  
  modes = res['obsmode'].copy()
  
  pyflam = res['pyflam'].copy()
  pyplam = res['pypivot'].copy()
  pybw = res['pyrms'].copy()
  
  irflam = res['irflam'].copy()
  irplam = res['irpivot'].copy()
  irbw = res['irrms'].copy()
  
  flamdiff = res['flam%'] * 100.
  plamdiff = res['pivot%'] * 100.
  bwdiff = res['rms%'] * 100.
  
  # we've got what we need, let the rest go away
  del res
  
  if orderby.lower() == 'photflam':
    order = np.abs(flamdiff).argsort()[::-1]
  elif orderby.lower() == 'photplam':
    order = np.abs(plamdiff).argsort()[::-1]
  elif orderby.lower() == 'photbw':
    order = np.abs(bwdiff).argsort()[::-1]
    
  s = '{:<45}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}'
  
  print(s.format('OBSMODE','PY PHOTFLAM','SYN PHOTFLAM','PHOTFLAM DIFF',
                  'PY PHOTPLAM','SYN PHOTPLAM','PHOTPLAM DIFF',
                  'PY PHOTBW','SYN PHOTBW','PHOTBW DIFF'))
                  
  s = '{:<45}{:<15.5e}{:<15.5e}{:< 15.8f}{:<15.5f}{:<15.5f}{:< 15.8f}{:<15.5f}{:<15.5f}{:< 15.8f}'
                    
  for i in order[:lines]:
    print(s.format(modes[i],pyflam[i],irflam[i],flamdiff[i],
                    pyplam[i],irplam[i],plamdiff[i],
                    pybw[i],irbw[i],bwdiff[i]))
