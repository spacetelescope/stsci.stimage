from __future__ import division # confidence high
import os,sys
import numpy as np
import pyfits
import pysynphotdev as S
from pysynphotdev import observationmode
from pyfits import Column
import time as _time
import re
from collections import OrderedDict

import graphfile as sgf

# some things for doing compute values with synphot
import tempfile
from pyraf import iraf
from iraf import stsdas, hst_calib, synphot

__version__ = '0.1.1'
__vdate__ = '28-Oct-2010'

def computeValues(obsmode,component_dict,spec=None):
    """ Compute the 3 photometric values needed for a given obsmode string
        using pysynphot
        This routine will return a dictionary with the photometry keywords and
        the computed value of the keyword
    """    
    if spec is None:
        # set up the flat spectrum used for the computing the photometry keywords
        spec = S.FlatSpectrum(1,fluxunits='flam')

    # Define the bandpass for this obsmode
    bp = S.ObsBandpass(obsmode,component_dict=component_dict)
    # create the observation using these elements
    obs = S.Observation(spec,bp)
    # compute the photometric values
    valdict = {}
    
    valdict['PHOTFLAM'] = obs.effstim('flam')/obs.effstim('counts')
    valdict['PHOTPLAM'] = bp.pivot()
    valdict['PHOTBW'] = bp.rmswidth()
    
    return valdict
    
def computeSynphotValues(obsmode):
    """
    Compute the 3 photometric values needed for a given obsmode string
    using synphot's bandpar function.
    This routine will return a dictionary with the photometry keywords and
    the computed value of the keyword.
    
    """
    tmpfits = os.path.join(tempfile.gettempdir(),'temp.fits')
    
    synphot.bandpar(obsmode,output=tmpfits,Stdout=1)
    
    fits = pyfits.open(tmpfits)
    d = fits[1].data
    
    photflam = d['uresp'][0]
    pivot = d['pivwv'][0]
    rmswidth = d['bandw'][0]
    
    fits.close()
    os.remove(tmpfits)
    
    valdict = {}
      
    valdict['PHOTFLAM'] = photflam
    valdict['PHOTPLAM'] = pivot
    valdict['PHOTBW'] = rmswidth
    
    return valdict
  
def expandObsmodes(basemode, pardict):
    """ Generate a set of obsmode strings spanning all the combinations of the 
        parameter as specified in the input dictionary
        
        The input will be a dictionary with the name of each parameterized variabble
        as the key and a list of values for that key; eg., 
            {'mjd':[52334,53919.99,53920,55516],'fr853n':[8158.0,8531.5,8905.0]}

        The basemode will be the root of the obsmode string including all 
        non-parameterized components; eg., acs,wfc1,f850lp
    """
    basemode = basemode.lower()
    obsmode_str = '%s,%s#%0.4f'
    olist = list()
    if len(pardict) > 0:
        for k in pardict.keys():
            basemode = basemode.replace('%s,'%k,'')
    if len(pardict) == 0:
        # we don't have any parameterized variables, so just return the basemode
        olist.append(basemode.rstrip(','))
    elif len(pardict) == 1:
        # build up list of obsmodes covering all combinations of parameterized
        # variable values
        key = pardict.keys()[0]
        for val in pardict[key]:
            ostr = basemode.replace(key.lower(),key.lower()+str(val))
            olist.append(ostr)
    else:        
        nkeys = len(pardict)
        for nkey in range(nkeys-1):
            key = pardict.keys()[nkey]
            for val in pardict[key]:
                pdict = {}
                for k in pardict.keys()[nkey+1:]:
                    pdict[k] = pardict[k]
                ostr = basemode.replace(key.lower(),key.lower()+str(val))
                olist.extend(expandObsmodes(ostr,pdict))

    return olist

def interpretObsmode(obsmode):
    ''' Convert a full obsmode string with parameterized values into a string
        which only lists the parameterized variable names without values for 
        comparison with the obsmode strings in the IMPHTTAB tables

        The return value corresponds to the OBSMODE value for each row in the
        table.
        
        For example, convert:
            acs,wfc1,mjd#52334.0000,fr853n#8158.0000
        into:
            acs,wfc1,mjd#,fr853n#
    '''
    ospl = obsmode.split(',')
    omode = ''
    for o in ospl:
        if '#' in o: o = o.split('#')[0]+'#'
        omode += o+','
    omode = omode.rstrip(',')
    return omode

        
def read_dict(fname):
    '''read a python dictionary from a file that was written
       with write_dict. [COPIED directly from pyetc/etc_engine/util.py]
    '''
    f=open(fname,'r')
    datastr = f.read()
    f.close()
    # convert DOS file to Unix - otherwise the eval will fail
    datastr = datastr.replace('\r','')
    try :
        datadict = eval(datastr)
    except Exception, e:
        print 'EXCEPTION:',e
        print 'cannot eval data in file ',fname
        raise
    return datadict
    
def parseFilters(filters):
    """ Parse the filters specification and return a string with any 
        non-parameterized filter names, and a list of parameterized filter names.
    """
    fspl = filters.split(',')
    fpars = list()
    fval = ''
    for f in fspl:
        if f.find('#') < 0:
            fval += f+','
        else:
            fpars.append(f)
    fval = fval[:-1]
    
    return fval,fpars

def getDate():
    """ Returns a formatted string with the current date.
        [This is simply a copy of 'getDate()' from pytools.fileutil]
    """
    _ltime = _time.localtime(_time.time())
    date_str = _time.strftime('%Y-%m-%dT%H:%M:%S',_ltime)

    return date_str
    
def makePrimaryHDU(filename,numpars,instrument):
    """ Create a Primary Header for the multi-extension FITS reference table
    """
    d = observationmode.getref()
    phdu = pyfits.PrimaryHDU()
    phdu.header.update('date',getDate(),comment="Date FITS file was generated")
    phdu.header.update('filename',filename,comment='name of file')
    phdu.header.update('nextend',3,comment='number of extensions in file')
    phdu.header.update('photzpt',-21.1,comment='Photometric zero-point for STMAG system')
    phdu.header.update('parnum',numpars,comment='Number of parameterized variables')
    phdu.header.update('dbtable','IMPHTTAB')
    phdu.header.update('instrume',instrument)
    phdu.header.update('synswver',S.__version__,comment='Version of synthetic photometry software')
    phdu.header.update('graphtab',d['graphtable'],comment='HST Graph table')
    phdu.header.update('comptab',d['comptable'],comment='HST Components table')
    phdu.header.update('useafter','')
    phdu.header.update('pedigree','Test data')
    phdu.header.update('descrip','photometry keywords reference file')

    return phdu

def saveSkippedObsmodes(output, obsmodes):
    ind = output.find('_imp.fits')
    
    if ind != -1:
      output = output[:ind] + '_skipped.txt'
    else:
      output = output + '_skipped.txt' 
    
    f = open(output,'w')
    
    for skipped in obsmodes:
      f.write(skipped + '\n')
      
    f.close()
    
def createTable(output,basemode,tmgtab,tmctab,tmttab, mode_list = [],
                nmodes=None,clobber=True,verbose=False):
    """ Create an IMPHTTAB file for a specified base configuration (basemode).
        
        Inputs:
        
        output: string
          Prefix for output reference file. ("_imp.fits" will be appended)
          
        basemode: string
          Base obsmode for which to generate a reference file. (e.g. acs,hrc)
          This is ignored if the mode_list keyword is a non-empty list.
          
        tmgtab: string
          File name of _tmg.fits reference file to be used.
          
        tmctab: string
          File name of _tmc.fits reference file to be used.
          
        tmttab: string
          File name of _tmt.fits reference file to be used.
        
        mode_list: list, optional
          A list of obsmodes which should be used to make an IMPHTTAB
          reference file. If this keyword is set to something other than 
          an empty list the basemode argument is ignored.
        
        nmodes: integer, optional
          Set to limit the number of modes to calculate, useful for testing.
          Defaults to None.
          
        clobber: boolean, optional
          True to overwrite an existing reference file, False to raise an error
          if file already exists. Defaults to True.
          
        verbose: boolean, optional
          True to print out extra information. Defaults to False.
    """
    if output.find('_imp.fits') < 0:
        output = output+'_imp.fits'
        
    # check status of output file
    if os.path.exists(output):
        if clobber: os.remove(output)
        else: raise IOError,'Output file already exists. Please delete/rename before restarting.'
    # interpret input data
    # The 'filtdata' dict contains the values for ALL the parameterized variables
    # used in the obsmodes supported by this table
    #if isinstance(filtdata,str):
    #    filtdata = read_dict(filtdata)    
    
    x = sgf.read_graphtable(tmgtab,tmctab,tmttab)
    
    if len(mode_list) == 0:
      # start by getting the full list of obsmodes before 
      # expanding the parameterized elements
      x.get_obsmodes(basemode,prefix=True)
      obsmodes = x.obsmodes
    else:
      obsmodes = mode_list
    
    # start building obsmodes for each row
    if nmodes is not None:
        nrows = nmodes
        obsmodes = obsmodes[:nmodes]
    else:
        nrows = len(obsmodes)
    
    fpars_vals = list() # list of parameterized variable names for each obsmode/row
    npar_vals = list() # number of parameterized variables for each obsmode/row
    flam_datacol_vals = list()
    plam_datacol_vals = list()
    bw_datacol_vals = list()
    fpars_sz = 1

    # Compute 'globally' required values: max number of parameterized variables,...
    for filt in obsmodes:
        # For each filter combination (row in the table)...
        basename,fpars = parseFilters(filt)
        # keep track of how many parameterized variables are used in this obsmode
        npars = len(fpars)
        npar_vals.append(npars)
        fpars_vals.append(fpars)
        fpars_len = [len(f) for f in fpars]
        
        if len(fpars_len) == 0: 
          fpars_max = 0
        else: 
          fpars_max = max(fpars_len)
        
        if fpars_max > fpars_sz: 
          fpars_sz = fpars_max
        
        if npars == 0: 
          nstr = ''
        else: 
          nstr = str(npars)
        
        flam_datacol_vals.append('PHOTFLAM'+nstr)
        plam_datacol_vals.append('PHOTPLAM'+nstr)
        bw_datacol_vals.append('PHOTBW'+nstr)
    #
    # At this point, all the interpretation for the following columns has been done:
    # OBSMODE, DATACOL (for all 3 tables), PEDIGREE, DESCRIP
    #
    # Start by determining the maximum number of parameters in any given obsmode
    max_npars = np.array(npar_vals,np.int32).max()
    print 'MAX_NPARS: ',max_npars,'   NROWS: ',nrows

    #
    # Now, define empty lists for NELEM* and PAR*VALUES columns
    #
    nelem_rows = np.zeros([nrows,max_npars],np.int16) # nelem_rows[i] for each column i
    parvals_rows = list()
    filtdata_set = dict()
    parnames_rows = np.chararray([nrows,max_npars],itemsize=fpars_sz) # create columns for PAR*NAMES
    parnames_rows[:] = ''*fpars_sz # initialize with blanks, just to be safe

    for nr in range(nrows):
        # create path through graphtab for this obsmode, reading in values for
        # all parameterized variables as well
        obspath = x.traverse(obsmodes[nr],verbose=False)
        filtdata = obspath._params
        
        # Create a master set of parameterized variables and their ranges of values
        for p in filtdata:
            if p.upper() not in filtdata_set.keys():
                filtdata_set[p] = filtdata[p]

        fpars = fpars_vals[nr]
        npars = npar_vals[nr]
        pvals = list()

        #extract entries from 'filtdata' for only the values given in 'fpars'
        for i in range(max_npars):
            if len(fpars) == 0:
                pvals.append(np.array([0]))
            else:
                if i < len(fpars):
                    f = fpars[i].upper()
                    nelem = len(filtdata[f])
                    nelem_rows[nr,i] = nelem
                    pvals.append(np.array(filtdata[f]))
                    parnames_rows[nr,i] = f
                else:
                    pvals.append(np.array([0]))
                            
        parvals_rows.append(pvals)

    #
    # All NELEM* and PAR*VALUES columns are correctly populated up to this point
    # in the code.
    #
    # Now, define the values for the actual results columns: PHOT*LAM, PHOTBW
    #
    flam_rows = list()
    plam_rows = list()
    bw_rows = list()
    
    nmode_vals = list()
    
    # set up the flat spectrum used for the computing the photometry keywords
    flatspec = S.FlatSpectrum(1,fluxunits='flam')

    # dictionary to hold optical components 
    # (pysynphot.observationmode._Component objects)
    component_dict = {}
    
    # list to hold skipped obsmodes
    skipped_obs = []

    if verbose:
        print "Computing photmetry values for each row's obsmode..."
        print 'Row: ',
        sys.stdout.flush()
    
    for nr in xrange(nrows):
        if verbose:
            print nr+1,' ',  # Provide some indication of which row is being worked
            sys.stdout.flush()
        
        obsmode = obsmodes[nr]
        fpars = fpars_vals[nr]
        npars = npar_vals[nr]
#        filtdict = dict()
        filtdict = OrderedDict()
        lenpars = list()
        for f in fpars:
            f = f.upper()
            filtdict[f] = filtdata_set[f]
            lenpars.append(len(filtdict[f]))
            
        # Now build up list of all obsmodes with all combinations of 
        # parameterized variables values
        olist = expandObsmodes(obsmode,filtdict)
        
        # Use these obsmodes to generate all the values needed for the row
        nmodes = len(olist)
        
        pflam = np.zeros(nmodes,np.float64)
        pplam = np.zeros(nmodes,np.float64)
        pbw = np.zeros(nmodes,np.float64)

        skip = False
        
        for n,fullmode in enumerate(olist):
            try:
                value = computeValues(fullmode,component_dict,spec=flatspec)
#                value = computeSynphotValues(fullmode)
            except ValueError,e:
                if e.message == 'Integrated flux is <= 0':
                    # integrated flux is zero, skip this obsmode
                    skip = True
                    skipped_obs.append(obsmode)
                    
                    flam_datacol_vals.pop(nr)
                    plam_datacol_vals.pop(nr)
                    bw_datacol_vals.pop(nr)
                    
                    if verbose:
                      print 'Skipping ' + obsmode + '\n'
                    
                    break
                elif e.message == 'math domain error':
                    skip = True
                    skipped_obs.append(obsmode)
                    
                    if verbose:
                      print 'Skipping ' + obsmode + '\n'
                    
                    flam_datacol_vals.pop(nr)
                    plam_datacol_vals.pop(nr)
                    bw_datacol_vals.pop(nr)
                          
                    break
                else:
                  raise
                
            if verbose:
                print 'PHOTFLAM(%s) = %g\n'%(fullmode,value['PHOTFLAM'])
            
            pflam[n] = value['PHOTFLAM']
            pplam[n] = value['PHOTPLAM']
            pbw[n] = value['PHOTBW']
        
        if skip is True:
          continue
          
        nmode_vals.append(nmodes)
    
        # Re-order results so that fastest varying variable is the last index
        # when accessed as a numpy array later by the C code
        photflam = ((pflam.reshape(lenpars)).transpose()).ravel()
        photplam = ((pplam.reshape(lenpars)).transpose()).ravel()
        photbw = ((pbw.reshape(lenpars)).transpose()).ravel()
        fvals = list()
        pvals = list()
        bvals = list()
        if npars == 0:
            fvals.append(photflam[0])
            pvals.append(photplam[0])
            bvals.append(photbw[0])
        else:
            fvals.append(0)
            pvals.append(0)
            bvals.append(0)
        for col in range(1,max_npars+1):
            if col == npars:
                fvals.append(np.array(photflam,np.float64))
                pvals.append(np.array(photplam,np.float64))
                bvals.append(np.array(photbw,np.float64))
            else:
                fvals.append(np.array([0]))
                pvals.append(np.array([0]))
                bvals.append(np.array([0]))
        flam_rows.append(fvals)
        plam_rows.append(pvals)
        bw_rows.append(bvals)
         
        del photflam,photplam,photbw,filtdict,lenpars
        
    del flatspec, component_dict
    
    # remove any skipped obsmodes from the obsmodes list
    for sk in skipped_obs:
      obsmodes.remove(sk)
      
    # save skipped obs to a file
    if len(skipped_obs) > 0:
      saveSkippedObsmodes(output, skipped_obs)
    
    del skipped_obs

    print "Creating table columns from photometry values..."
    
    # Convert nelem information from row-oriented to column oriented
    nelem_cols = nelem_rows.transpose()
    parnames_cols = parnames_rows.transpose()
    
    parvals_cols = list()
    flam_cols = list()
    plam_cols = list()
    bw_cols = list()
    for col in range(max_npars):
        pvals = list()
        for row in range(len(parvals_rows)):
            pvals.append(parvals_rows[row][col])
        parvals_cols.append(pvals)
 
    for col in range(max_npars+1):
        fvals = list()
        plvals = list()
        bvals = list()
        for row in range(len(flam_rows)):
            fvals.append(flam_rows[row][col])
            plvals.append(plam_rows[row][col])
            bvals.append(bw_rows[row][col])
        if col == 0:
            fvals = np.array(fvals)
            plvals = np.array(plvals)
            bvals = np.array(bvals)
        flam_cols.append(fvals)
        plam_cols.append(plvals)
        bw_cols.append(bvals)
    
    ped_vals =['Version %s data'%__version__]*len(nmode_vals)
    descrip_vals = ['Generated %s from %s'%(getDate(),tmgtab)]*len(nmode_vals)
    
    # Finally, create the structures needed to define this row in the FITS table
        
    # Define each column in the table based on max_npars which are not different
    # from one extension to the other
    obsmode_col = Column(name='obsmode',format='40A',array=np.array(obsmodes))
    pedigree_col = Column(name='pedigree',format='30A',array=np.array(ped_vals))
    descrip_col = Column(name='descrip',format='67A',array=np.array(descrip_vals))
    datacol_col = {}
    datacol_col['PHOTFLAM'] = Column(name='datacol',format='12A',array=np.array(flam_datacol_vals))
    datacol_col['PHOTPLAM'] = Column(name='datacol',format='12A',array=np.array(plam_datacol_vals))
    datacol_col['PHOTBW'] = Column(name='datacol',format='12A',array=np.array(bw_datacol_vals))
    
    parvals_tabcols = list()
    nelem_tabcols = list()
    parnames_tabcols = list()
    parnames_format = str(fpars_sz)+"A[]"
    # for each parameterized element, create a set of columns specifying the
    # range of values for that parameter and the number of elements covering that range
    # namely, the PAR<n>VALUES and NELEM<n> columns
    for p in range(max_npars):
        nelem_tabcols.append(Column(name="NELEM"+str(p+1),format="I",array=np.array(nelem_cols[p],np.int16)))
        parvals_tabcols.append(Column(name="PAR"+str(p+1)+"VALUES",format="PD[]",array=np.array((parvals_cols[p]),'O')))
        parnames_tabcols.append(Column(name="PAR"+str(p+1)+"NAMES",format=parnames_format,array=np.array((parnames_cols[p]),'O')))
        
    # create the set of results columns
    flam_tabcols = list()
    plam_tabcols = list()
    bw_tabcols = list()
    for p in range(max_npars+1):
        if p == 0:
            format_str = 'D'
            pstr = ''
            fcols = flam_cols[p]
            pcols = plam_cols[p]
            bcols = bw_cols[p]
        else:
            format_str = 'PD[]'
            pstr = str(p)
            fcols = np.array(flam_cols[p],'O')
            pcols = np.array(plam_cols[p],'O')
            bcols = np.array(bw_cols[p],'O')
          
        flam_tabcols.append(Column(name='PHOTFLAM'+pstr,format=format_str,array=fcols))
        plam_tabcols.append(Column(name='PHOTPLAM'+pstr,format=format_str,array=pcols))
        bw_tabcols.append(Column(name='PHOTBW'+pstr,format=format_str,array=bcols))
    print 'Creating full table: ',output
    # Now create the FITS file with the table in each extension
    
    phdu = makePrimaryHDU(output,max_npars,basemode.split(',')[0])
    flam_tab = pyfits.new_table([obsmode_col,datacol_col['PHOTFLAM']]+flam_tabcols+parnames_tabcols+parvals_tabcols+nelem_tabcols+[pedigree_col,descrip_col])
    flam_tab.header.update('extname','PHOTFLAM')
    flam_tab.header.update('extver',1)
    plam_tab = pyfits.new_table([obsmode_col,datacol_col['PHOTPLAM']]+plam_tabcols+parnames_tabcols+parvals_tabcols+nelem_tabcols+[pedigree_col,descrip_col])
    plam_tab.header.update('extname','PHOTPLAM')
    plam_tab.header.update('extver',1)
    bw_tab = pyfits.new_table([obsmode_col,datacol_col['PHOTBW']]+bw_tabcols+parnames_tabcols+parvals_tabcols+nelem_tabcols+[pedigree_col,descrip_col])    
    bw_tab.header.update('extname','PHOTBW')
    bw_tab.header.update('extver',1)
    
    ftab = pyfits.HDUList()
    ftab.append(phdu)
    ftab.append(flam_tab)
    ftab.append(plam_tab)
    ftab.append(bw_tab)
    ftab.writeto(output)

def createNicmosTable(output,pht_table,tmgtab,tmctab,tmttab,
                      clobber=True,verbose=False):
    """
    Use a NICMOS _pht.fits table to generate an IMPHTTAB table for obsmodes
    listed in the _pht table.
    
    Input
    -----
    output: string
      Prefix for output reference file. ("_imp.fits" will be appended)
      
    pht_table: string
      File name of _pht.fits table from which to take obsmodes.
      
    tmgtab: string
      File name of _tmg.fits reference file to be used.
      
    tmctab: string
      File name of _tmc.fits reference file to be used.
      
    tmttab: string
      File name of _tmt.fits reference file to be used.
      
    clobber: boolean, optional
      True to overwrite an existing reference file, False to raise an error
      if file already exists. Defaults to True.
      
    verbose: boolean, optional
      True to print out extra information. Defaults to False.
            
    """
    
    pht = pyfits.open(pht_table, 'readonly')
  
    modes = np.char.strip(pht[1].data['photmode']).tolist()
    
    pht.close()
    
    createTable(output,'nicmos',tmgtab,tmctab,tmttab,mode_list=modes,
                clobber=clobber,verbose=verbose)
                
def createTableFromTable(output, imphttab, tmgtab, tmctab, tmttab,
                          clobber=True, verbose=False):
    """
    Use a previously created IMPHTTAB reference file to generate a new
    IMPHTTAB reference file based on input graph and comp tables.
    
    Input
    -----
    output: string
      Prefix for output reference file. ("_imp.fits" will be appended)
      
    imphttab: string
      File name of _imp.fits IMPHTTAB table from which to take obsmodes.
      
    tmgtab: string
      File name of _tmg.fits reference file to be used.
      
    tmctab: string
      File name of _tmc.fits reference file to be used.
      
    tmttab: string
      File name of _tmt.fits reference file to be used.
      
    clobber: boolean, optional
      True to overwrite an existing reference file, False to raise an error
      if file already exists. Defaults to True.
      
    verbose: boolean, optional
      True to print out extra information. Defaults to False.
      
    """
    imp = pyfits.open(imphttab, 'readonly')
    
    inst = imp[0].header['instrume']
    
    modes = np.char.strip(imp[1].data['obsmode']).tolist()
    
    imp.close()
    
    createTable(output,inst,tmgtab,tmctab,tmttab,mode_list=modes,
                clobber=clobber,verbose=verbose)
    
  