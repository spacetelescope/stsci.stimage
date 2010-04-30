import os
import numpy as np
import pyfits
import pysynphot as S
from pysynphot import observationmode
from pyfits import Column
import time as _time

__version__ = '0.0.1'
__vdate__ = '29-Apr-2010'

def computeValues(obsmode):
    """ Compute the 3 photometric values needed for a given obsmode string
        using pysynphot
        This routine will return a dictionary with the photometry keywords and
        the computed value of the keyword
    """    
    # Define the bandpass for this obsmode
    bp = S.ObsBandpass(obsmode)
    # set up the flat spectrum used for the computing the photometry keywords
    sp = S.FlatSpectrum(1,fluxunits='flam')
    # create the observation using these elements
    obs = S.Observation(sp,bp)
    # compute the photometric values
    valdict = {}
    
    valdict['PHOTFLAM'] = obs.effstim('flam')/obs.effstim('counts')
    valdict['PHOTPLAM'] = obs.pivot()
    valdict['PHOTBW'] = bp.rmswidth()
    
    return valdict
    
def generateObsmodes(basemode, pardict):
    """ Generate a set of obsmode strings spanning all the combinations of the 
        parameter as specified in the input dictionary
        
        The input will be a dictionary with the name of each parameterized variabble
        as the key and a list of values for that key; eg., 
            {'mjd':[52334,53919.99,53920,55516],'fr853n':[8158.0,8531.5,8905.0]}

        The basemode will be the root of the obsmode string including all 
        non-parameterized components; eg., acs,wfc1,f850lp
    """
    obsmode_str = '%s,%s#%0.4f'
    olist = []
    if len(pardict) > 0:
        for k in pardict.keys():
            basemode = basemode.replace('%s,'%k,'')
    if len(pardict) == 0:
        # we don't have any parameterized variables, so just return the basemode
        olist.append(basemode.rstrip(','))
    elif len(pardict) == 1:
        # build up list of obsmodes covering all combinations of paramterized
        # variable values
        key = pardict.keys()[0]
        for val in pardict[key]:
            ostr = obsmode_str%(basemode,key,val)
            ostr = ostr.replace('%s,'%(key),'')
            olist.append(ostr)
    else:        
        nkeys = len(pardict)
        for nkey in range(nkeys-1):
            key = pardict.keys()[nkey]
            for val in pardict[key]:
                pdict = {}
                for k in pardict.keys()[nkey+1:]:
                    pdict[k] = pardict[k]
                olist.extend(generateObsmodes(obsmode_str%(basemode,key,val),pdict))

    return olist

def interpretObsmode(obsmode):
    ''' Convert a full obsmode string with parameterized values into a string
        which only lists the parameterized variable names without values

        The return value corresponds to the OBSMODE value for each row in the
        table.
        
        For example, convert:
            acs,wfc1,mjd#52334.0000,fr853n#8158.0000
        into:
            acs,wfc1,mjd,fr853n
    '''
    ospl = obsmode.split(',')
    omode = ''
    for o in ospl:
        if '#' in o: o = o.split('#')[0]
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
    fpars = []
    fval = ''
    for f in fspl:
        if f.find('#') > -1:
            f = f.replace('#','')
            fpars.append(f)
        fval += '%s,'%f
    fval.rstrip(',')
    
    return fval,fpars

def getDate():
    """ Returns a formatted string with the current date.
        [This is simply a copy of 'getDate()' from pytools.fileutil]
    """
    _ltime = _time.localtime(_time.time())
    date_str = _time.strftime('%Y-%m-%dT%H:%M:%S',_ltime)

    return date_str
    
def createPrimaryHDU(filename,numpars,parnames,instrument):
    """ Create a Primary Header for the multi-extension FITS reference table
    """
    d = observationmode.getref()
    phdu = pyfits.PrimaryHDU()
    phdu.header.update('date',getDate(),comment="Date FITS file was generated")
    phdu.header.update('filename',filename,comment='name of file')
    phdu.header.update('nextend',3,comment='number of extensions in file')
    phdu.header.update('photzpt',-21.1,comment='Photometric zero-point for STMAG system')
    phdu.header.update('parnum',numpars,comment='Number of parameterized variables')
    for p in range(numpars):
        phdu.header.update('par'+str(p+1)+'name',parnames[p],comment='Name of 1st parameterized variable,if any')
    phdu.header.update('dbtable','IMPHTTAB')
    phdu.header.update('instrume',instrument)
    phdu.header.update('synswver',S.__version__,comment='Version of synthetic photometry software')
    phdu.header.update('graphtab',d['graphtable'],comment='HST Graph table')
    phdu.header.update('comptab',d['comptable'],comment='HST Components table')
    phdu.header.update('useafter','')
    phdu.header.update('pedigree','Test data')
    phdu.header.update('descrip','photometry keywords reference file')

    return phdu
    
    
def run(output,basemode,filters,filtdata,order,clobber=True):
    """ Create a (sample) IMPHTTAB file for a specified base 
        configuration (basemode) and a set of filter combinations (filters).
        
        The input 'filters' will be a list of strings, with parameterized 
        filters ending in '#', for all the combinations of filters to be 
        computed for the table with one row per combination.  
        
        The parameter 'order' specifies the order of the parameterized
        variables specified in this set of filters so that keywords can 
        be assigned consistently. For example, MJD will always be variable 1,
        and FR will be variable 2 as used in NELEM[1,2] or PAR[1,2]VALUES.
        
        The range of values spanned by each of the parameterized filters 
        will be specified in the external file or dictionary 'filtdata'.
        If 'filtdata' is the name of a file, the data must be formatted as
        a Python list of dictionaries, one dictionary per filter.
    """
    # check status of output file
    if os.path.exists(output):
        if clobber: os.remove(output)
        else: raise IOError,'Output file already exists. Please delete/rename before restarting.'
    # interpret input data
    # The 'filtdata' dict contains the values for ALL the parameterized variables
    # used in the obsmodes supported by this table
    if isinstance(filtdata,str):
        filtdata = read_dict(filtdata)    
    
    # start building obsmodes for each row
    nfilt = len(filters)
    obsmode_vals = []
    nmode_vals = []
    ped_vals =['Test data']*nfilt
    descrip_vals = ['Generated Apr 2010 for testing only']*nfilt
    
    obsmode_vals = []
    fpars_vals = []
    npar_vals = []
    flam_datacol_vals = []
    plam_datacol_vals = []
    bw_datacol_vals = []
    for filt in filters:
        # For each filter combination (row in the table)...
        filtname,fpars = parseFilters(filt)
        # keep track of how many parameterized variables are used in this obsmode
        npars = len(fpars)
        npar_vals.append(npars)
        obsmode_vals.append((basemode+','+filtname).rstrip(','))
        fpars_vals.append(fpars)
        
        if npars == 0: nstr = ''
        else: nstr = str(npars)
        flam_datacol_vals.append('PHOTFLAM'+nstr)
        plam_datacol_vals.append('PHOTPLAM'+nstr)
        bw_datacol_vals.append('PHOTBW'+nstr)
    #
    # At this point, all the interpretation for the following columns has been done:
    # OBSMODE, DATACOL (for all 3 tables), PEDIGREE, DESCRIP
    #
    # Start by determining the maximum number of parameters in any given obsmode
    max_npars = np.array(npar_vals,np.int32).max()
    nrows = len(filters)
    print 'MAX_NPARS: ',max_npars,'   NROWS: ',nrows

    #
    # Now, define empty lists for NELEM* and PAR*VALUES columns
    #
    nelem_rows = np.zeros([nrows,max_npars],np.int16) # nelem_rows[i] for each column i
    parvals_rows = [] 

    for nr in range(nrows):
        fpars = fpars_vals[nr]
        npars = npar_vals[nr]
        pvals = []
        #extract entries from 'filtdata' for only the values given in 'fpars'
        for i in range(max_npars):
            if len(fpars) == 0:
                pvals.append(np.array([0]))
            else:
                f = order[i]
                found = False
                for fp in fpars:
                    if f.lower() == fp.lower()[:len(f)]:
                        found = True
                        f = fp
                        break            
                if found is True:
                    nelem = len(filtdata[f])
                    nelem_rows[nr][i] = nelem
                    pvals.append(np.array(filtdata[f]))
                else:
                    pvals.append(np.array([0]))
                    
        
        parvals_rows.append(pvals)

    #
    # All NELEM* and PAR*VALUES columns are correctly populated up to this point
    # in the code.
    #
    # Now, define the values for the actual results columns: PHOT*LAM, PHOTBW
    #
    flam_rows = []
    plam_rows = []
    bw_rows = []
    for nr in range(nrows):
        obsmode = obsmode_vals[nr]
        fpars = fpars_vals[nr]
        npars = npar_vals[nr]

        # define obsmode specific dictionary of parameterized variables
        # for this row alone
        fdict = {}
        for f in fpars:
            fdict[f] = filtdata[f]

        # Now build up list of all obsmodes with all combinations of 
        # parameterized variables values
        olist = generateObsmodes(obsmode,fdict)
        
        # Use these obsmodes to generate all the values needed for the row
        nmodes = len(olist)
        nmode_vals.append(nmodes)
        photflam = np.zeros(nmodes,np.float64)
        photplam = np.zeros(nmodes,np.float64)
        photbw = np.zeros(nmodes,np.float64)
        print 'nmodes: ',nmodes
        
        for obsmode,n in zip(olist,range(nmodes)):
            value = computeValues(obsmode)
            photflam[n] = value['PHOTFLAM']
            photplam[n] = value['PHOTPLAM']
            photbw[n] = value['PHOTBW']
        fvals = []
        pvals = []
        bvals = []
        for col in range(max_npars+1):
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
        
        del photflam,photplam,photbw,fdict

    # Convert nelem information from row-oriented to column oriented
    nelem_cols = nelem_rows.transpose()
    parvals_cols = []
    flam_cols = []
    plam_cols = []
    bw_cols = []
    for col in range(max_npars):
        pvals = []
        for row in range(len(parvals_rows)):
            pvals.append(parvals_rows[row][col])
        parvals_cols.append(pvals)
 
    for col in range(max_npars+1):
        fvals = []
        plvals = []
        bvals = []
        for row in range(len(flam_rows)):
            fvals.append(flam_rows[row][col])
            plvals.append(plam_rows[row][col])
            bvals.append(bw_rows[row][col])
        
        flam_cols.append(fvals)
        plam_cols.append(plvals)
        bw_cols.append(bvals)
    # Finally, create the structures needed to define this row in the FITS table
        
    # Define each column in the table based on max_npars which are not different
    # from one extension to the other
    obsmode_col = Column(name='obsmode',format='40A',array=np.array(obsmode_vals))
    pedigree_col = Column(name='pedigree',format='30A',array=np.array(ped_vals))
    descrip_col = Column(name='descrip',format='67A',array=np.array(descrip_vals))
    datacol_col = {}
    datacol_col['PHOTFLAM'] = Column(name='datacol',format='12A',array=np.array(flam_datacol_vals))
    datacol_col['PHOTPLAM'] = Column(name='datacol',format='12A',array=np.array(plam_datacol_vals))
    datacol_col['PHOTBW'] = Column(name='datacol',format='12A',array=np.array(bw_datacol_vals))
    
    parvals_tabcols = []
    nelem_tabcols = []
    # for each parameterized element, create a set of columns specifying the
    # range of values for that parameter and the number of elements covering that range
    # namely, the PAR<n>VALUES and NELEM<n> columns
    for p in range(max_npars):
        nelem_tabcols.append(Column(name="NELEM"+str(p+1),format="I",array=np.array(nelem_cols[p],np.int16)))
        parvals_tabcols.append(Column(name="PAR"+str(p+1)+"VALUES",format="PD[]",array=parvals_cols[p]))
        
    # create the set of results columns
    flam_tabcols = []
    plam_tabcols = []
    bw_tabcols = []
    for p in range(max_npars+1):
        if p == 0:
            format_str = 'D'
            pstr = ''
        else:
            format_str = 'PD[]'
            pstr = str(p)
        flam_tabcols.append(Column(name='PHOTFLAM'+pstr,format=format_str,array=flam_cols[p]))
        plam_tabcols.append(Column(name='PHOTPLAM'+pstr,format=format_str,array=plam_cols[p]))
        bw_tabcols.append(Column(name='PHOTBW'+pstr,format=format_str,array=bw_cols[p]))
    
    # Now create the FITS file with the table in each extension
    phdu = createPrimaryHDU(output,2,order,'acs')
    
    ftab = pyfits.HDUList()
    ftab.append(phdu)
    ftab.append(pyfits.new_table([obsmode_col,datacol_col['PHOTFLAM']]+nelem_tabcols+[pedigree_col,descrip_col]))
    #ftab.append(pyfits.new_table([obsmode_col,datacol_col['PHOTFLAM']]+flam_tabcols+parvals_tabcols+nelem_tabcols+[pedigree_col,descrip_col]))
    #ftab.append(pyfits.new_table([obsmode_col,datacol_col['PHOTPLAM']]+plam_tabcols+parvals_tabcols+nelem_tabcols+[pedigree_col,descrip_col]))
    #ftab.append(pyfits.new_table([obsmode_col,datacol_col['PHOTBW']]+bw_tabcols+parvals_tabcols+nelem_tabcols+[pedigree_col,descrip_col]))
    ftab.writeto(output)
    