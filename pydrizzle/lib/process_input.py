from pytools import parseinput, fileutil, readgeis, makewcs, asnutil
import pyfits
import os 

"""
Process input to pydrizzle.
Input can be one of 

- a python list of files
- a comma separated string of filenames (including wild card characters)
- an association table
- an @file (can have a second column with names of ivm files)

No mixture of instruments is allowed.
No mixture of association tables, @files and regular fits files is allowed.
Files can be in GEIS or MEF format (but not waiver fits).

Runs some sanity checks on the input files.
If necessary converts files to MEF format (this should not be left to makewcs 
because 'updatewcs' may be False).
Runs makewcs.
Returns an association table, ivmlist, output name

"""

def atfile_sci(f):
    return f.split()[0]
    
def atfile_ivm(f):
    return f.split()[1]    


def process_input(input, output=None, ivmlist=None, updatewcs=True, prodonly=False, shiftfile=None):
    
    ivmlist = None
    oldasndict = None
    
    if (isinstance(input, list) == False) and \
       ('_asn' in input or '_asc' in input) :
        # Input is an association table
        # Get the input files, and run makewcs on them
        oldasndict = asnutil.readASNTable(input, prodonly=prodonly)
        if not output:
            output = oldasndict['output']

        filelist = [fileutil.buildRootname(fname) for fname in oldasndict['order']]
        
        # There's no way to specify ivm files through an association table
        # so ivmlist is always None in this case
        #newfilelist, ivmlist = checkFiles(filelist, ivmlist)
        
    elif (isinstance(input, list) == False) and \
       (input[0] == '@') :
        # input is an @ file
        f = open(input)
        line = f.readline()
        f.close()
        if len(line.split()) == 2:
            filelist = irafglob.irafglob(input, atfile=atfile_sci)
            ivmlist = irafglob.irafglob(input, atfile=atfile_ivm)        
    else:
        #input is a string or a python list
        try:
            filelist, output = parseinput.parseinput(input, outputname=output)
        except IOError: raise
        

    newfilelist, ivmlist = checkFiles(filelist, ivmlist)
    if not newfilelist:
        return newfilelist, ivmlist, output
        #return None, None, None

    #make an asn table at the end
    if updatewcs:
        pydr_input = runmakewcs(newfilelist)  
    else:
        pydr_input = newfilelist

    # AsnTable will handle the case when output==None
    #asndict = fileutil.buildAsnDict(pydr_input, output=output,shiftfile=shiftfile)
    if oldasndict:
        if pydr_input[0].split('_')[1].split('.fits')[0] != 'flt':
            asndict = update_member_names(oldasndict, pydr_input)
        elif len(pydr_input) != len(filelist):
            for f in filelist:
                if f not in pydr_input:
                    fname = fileutil.buildNewRootname(f)
                    oldasndict['members'].pop(fname)
                    oldasndict['order'].remove(fname)
            asndict = oldasndict
        else:
            asndict = oldasndict
    else:
        #
        #remove shiftfile from here, only allow updates
        #!!!!!

        asndict = asnutil.ASNTable(pydr_input, output=output) #, shiftfile=shiftfile)
        asndict.create()
        #if pydr_input[0].split('_')[-1].split('.fits')[0] != 'flt':
        #asndict = update_member_names(asndict, pydr_input)
            
    if shiftfile:
        asndict.update(shiftfile=shiftfile)

    # Build output filename
    if output == None:
        output = fileutil.buildNewRootname(asndict['output'],extn='_drz.fits')
        print 'Setting up output name: ',output

    return asndict, ivmlist, output


def checkFiles(filelist, ivmlist = None):
    """
    1. Removes waiver fits files from input lists
    2. Converts all GEIS science and data quality files to MEF format
    3. Checks for stis association tables 
    4. Checks if kw idctab exists, if not tries to populate it 
        based on the spt file
    5. Removes files with EXPTIME=0 and the corresponding ivm files
    6. Removes files with NGOODPIX == 0 (to exclude saturated images)
    """
    #newfilelist = []
    removed_files = []
    translated_names = []
    newivmlist = []
    
    if ivmlist == None:
        ivmlist = [None for l in filelist]

    sci_ivm = zip(filelist, ivmlist)
    
    for file in sci_ivm:
        #find out what the input is
        # if science file is not found on disk, add it to removed_files for removal
        try:
            imgfits,imgtype = fileutil.isFits(file[0])
        except IOError:
            print "Warning:  File %s could not be found\n" %file[0]
            print "Removing file %s from input list" %file[0]
            removed_files.append(file)
            continue
        if file[1] != None:
            #If an ivm file is not found on disk
            # Remove the corresponding science file
            try:
                ivmfits,ivmtype = fileutil.isFits(file[1])
            except IOError:
                print "Warning:  File %s could not be found\n" %file[1]
                print "Removing file %s from input list" %file[0]
                removed_files.append(file)
        # Check for existence of waiver FITS input, and quit if found.
        # Or should we print a warning and continue but not use that file
        if imgfits and imgtype == 'waiver':
            print "Warning: PyDrizzle does not support waiver fits format.\n"
            print "Convert the input files to GEIS or multiextension FITS.\n"
            print "Removing file %s from input list" %file[0] 
            removed_files.append(file[0])
        # If a GEIS image is provided as input, create a new MEF file with 
        # a name generated using 'buildFITSName()'
        # Convert the corresponding data quality file if present    
        if not imgfits:
            newfilename = geis2mef(file[0], convert_dq=True)
            if newfilename == None:
                print "Removing file %s from input list" %file[0]
                removed_files.append(file[0])
            else:
                translated_names.append(newfilename)
        if file[1] != None:
            if ivmfits and ivmtype == 'waiver':
                print "Warning: PyDrizzle does not support waiver fits format.\n"
                print "Convert the input files to GEIS or multiextension FITS.\n"
                print "File %s appears to be in waiver fits format \n" %file[1]
                print "Removing file %s from input list" %file[0] 
                removed_files.append(file[0])
  
            if not ivmfits:
                newfilename = geis2mef(file[1], convert_dq=False)
                if newfilename == None:
                    print "Removing file %s from input list" %file[0]
                    removed_files.append(file[0])
                else:
                    newivmlist.append(newfilename)

    #check_exptime(filelist)
    
    newfilelist, ivmlist = update_input(filelist, ivmlist, removed_files)

    if newfilelist == []:
        errormsg = "\n No valid input was found. Quitting ...\n"
        raise IOError, errormsg
    
    if translated_names != []:
        # Since we don't allow input from different instruments
        # we can abandon the original input list and provide as 
        # input only the translated names
        removed_expt_files = check_exptime(translated_names)
        newfilelist, ivmlist = update_input(translated_names, newivmlist, removed_expt_files)
    else:
        # check for STIS association files. This must be done before 
        # the check for EXPTIME in order to handle correctly stis 
        # assoc files
        if pyfits.getval(newfilelist[0], 'INSTRUME') == 'STIS':
            newfilelist, ivmlist = checkStisFiles(newfilelist, ivmlist)
            #removed_files = check_exptime(newflist)
        
        removed_expt_files = check_exptime(newfilelist)
        newfilelist, ivmlist = update_input(newfilelist, ivmlist, removed_expt_files)
    if removed_expt_files:
        errorstr =  "#############################################\n"
        errorstr += "#                                           #\n"
        errorstr += "# ERROR:                                    #\n"
        errorstr += "#                                           #\n"
        errorstr += "#  The following files were excluded from   #\n"
        errorstr += "#  Multidrizzle processing because their    #\n"
        errorstr += "#  header keyword EXPTIME values were 0.0:  #\n"
        for name in removed_expt_files:
            errorstr += "         "+ str(name) + "\n" 
        errorstr += "#                                           #\n"
        errorstr += "#############################################\n\n"
        print errorstr
        
    removed_ngood_files = checkNGOODPIX(newfilelist)    
    newfilelist, ivmlist = update_input(newfilelist, ivmlist, removed_ngood_files)

    if removed_ngood_files:
        msgstr =  "####################################\n"
        msgstr += "#                                  #\n"
        msgstr += "# WARNING:                         #\n"
        msgstr += "#  NGOODPIX keyword value of 0 in  #\n"
        for name in removed_ngood_files:
            msgstr += "         "+ str(name) + "\n" 
        msgstr += "#  has been detected.  Images with #\n"
        msgstr += "#  no valid pixels will not be     #\n"
        msgstr += "#  used during processing.  If you #\n"
        msgstr += "#  wish this file to be used in    #\n"
        msgstr += "#  processing, please check its DQ #\n"
        msgstr += "#  array and reset driz_sep_bits   #\n"
        msgstr += "#  and final_bits parameters       #\n"
        msgstr += "#  to accept flagged pixels.       #\n"
        msgstr += "#                                  #\n"
        msgstr += "####################################\n"
        print msgstr   
                
    return newfilelist, ivmlist
    

def geis2mef(sciname, convert_dq=True):
    """
    Converts a GEIS science file and its corresponding 
    data quality file (if present) to MEF format
    Writes out both files to disk.
    Returns the new name of the science image.
    """
        
    def convert(file):
        newfilename = fileutil.buildFITSName(file)
        try:
            newimage = fileutil.openImage(file,writefits=True,
                fitsname=newfilename, clobber=True)            
            del newimage
            return newfilename
        except IOError:
            print 'Warning: File %s could not be found' % file     
            return None

    newsciname = convert(sciname)
    if convert_dq:
        dq_name = convert(sciname.split('.')[0] + '.c1h')
        
    return newsciname


def checkStisFiles(filelist, ivmlist=None):
    newflist = []
    newilist = []
    
    if len(filelist) != len(ivmlist):
        errormsg = "Input file list and ivm list have different lenghts\n"
        errormsg += "Quitting ...\n"
        raise ValueError, errormsg
        
    for t in zip(filelist, ivmlist):
        sci_count = stisObsCount(t[0])
        if sci_count >1:
            newfilenames = splitStis(t[0], sci_count)
            newflist.extend(newfilenames)
            if t[1] != None:
                newivmnames = splitStis(t[1], sci_count)
                newilist.extend(newivmnames)
            else:
                newilist.append(None)
        elif sci_count == 1:
            newflist.append(t[0])
            newilist.append(t[1])
        else:
            errormesg = "No valid 'SCI extension in STIS file\n"
            raise ValueError, errormsg

    return newflist, newilist


def runmakewcs(input):
    """
    Runs make wcs and recomputes the WCS keywords
    input: a list of files
    output: returns a list of names of the modified files
            (For GEIS files returns the translated names.)
    """
    newNames = makewcs.run(input)
    return newNames

def check_exptime(filelist):
    """
    Removes files with EXPTIME==0 from filelist.
    """
    removed_files = []
    
    for f in filelist:
        if fileutil.getKeyword(f, 'EXPTIME') <= 0: 
            removed_files.append(f)
            
    return removed_files

def checkNGOODPIX(filelist):
    """
    Only for ACS, and STIS, check NGOODPIX
    If all pixels are 'bad' on all chips, exclude this image
    from further processing. 
    Similar checks requiring comparing 'driz_sep_bits' against
    WFPC2 c1f.fits arrays and NICMOS DQ arrays will need to be
    done separately (and later).
    """
    removed_files = []
    for inputfile in filelist:
        if (fileutil.getKeyword(inputfile,'instrume') == 'ACS') \
           or fileutil.getKeyword(inputfile,'instrume') == 'STIS': 
            _file = fileutil.openImage(inputfile)
            _ngood = 0
            for extn in _file:
                if extn.header.has_key('EXTNAME') and extn.header['EXTNAME'] == 'SCI':
                    _ngood += extn.header['NGOODPIX']
            _file.close()
            
            if (_ngood == 0):
                removed_files.append(inputfile)
    return removed_files

def update_input(filelist, ivmlist=None, removed_files=None):
    """
    Removes files flagged to be removed from the input filelist.
    Removes the corresponding ivm files if present.
    """
    newfilelist = []

    if removed_files == []:
        return filelist, ivmlist
    else:
        """
        if ivmlist == None:
            newfilelist[:] = filelist[:] 
            for f in removed_files:
                newfilelist.remove(f)
        else:
        """
        sci_ivm = zip(filelist, ivmlist)
        for f in removed_files:
            result=[sci_ivm.remove(t) for t in sci_ivm if t[0] == f ]
        ivmlist = [el[1] for el in sci_ivm] 
        newfilelist = [el[0] for el in sci_ivm] 
        return newfilelist, ivmlist 
  

def stisObsCount(input):
    """
    Input: A stis multiextension file
    Output: Number of stis science extensions in input
    """
    count = 0
    f = pyfits.open(input)
    for ext in f:
        if ext.header.has_key('extname'):
            if (ext.header['extname'].upper() == 'SCI'):
                count += 1
    f.close()
    return count

def splitStis(stisfile, sci_count):
    """
    Purpose
    =======
    
    Split a STIS association file into multiple imset MEF files.
    Split the corresponding spt file if present into single spt files.
    If an spt file can't be split or is missing a Warning is printed.
    
    Output: a list with the names of the new flt files.
    """
    newfiles = []
    
    f = pyfits.open(stisfile)
    hdu0 = f[0].copy()


    for count in range(1,sci_count+1):
        #newfilename = rootname+str(count)+'.fits'
        fitsobj = pyfits.HDUList()            
        fitsobj.append(hdu0)
        hdu = f['sci',count].copy()
        fitsobj.append(hdu)
        rootname = hdu.header['EXPNAME']
        newfilename = fileutil.buildNewRootname(rootname, extn='_flt.fits')
        try:
            # Verify error array exists
            if f['err',count].data == None:
                raise ValueError
            # Verify dq array exists
            if f['dq',count].data == None:
                raise ValueError
            # Copy the err extension
            hdu = f['err',count].copy()
            fitsobj.append(hdu)
            # Copy the dq extension
            hdu = f['dq',count].copy()
            fitsobj.append(hdu)
        except:
            errorstr =  "\n###############################\n"
            errorstr += "#                             #\n"
            errorstr += "# ERROR:                      #\n"
            errorstr += "#  The input image:           #\n"
            errorstr += "      " + str(stisfile) +"\n"
            errorstr += "#  does not contain required  #\n"
            errorstr += "#  image extensions.  Each    #\n"
            errorstr += "#  must contain populated sci,#\n"
            errorstr += "#  dq, and err arrays.        #\n"
            errorstr += "#                             #\n"
            errorstr += "###############################\n"
            raise ValueError, errorstr
        
        
        # Update the 'EXTNER' keyword to indicate the new extnesion number
        # for the single exposure files.
        fitsobj[1].header['EXTVER'] = 1
        fitsobj[2].header['EXTVER'] = 1
        fitsobj[3].header['EXTVER'] = 1
        
        # Determine if the file you wish to create already exists on the disk.
        # If the file does exist, replace it.
        if (os.path.exists(newfilename)):
            os.remove(newfilename)
            print "       Replacing "+newfilename+"..."
            
            # Write out the new file
        fitsobj.writeto(newfilename)
        newfiles.append(newfilename)
    f.close()

    sptfilename = fileutil.buildNewRootname(stisfile, extn='_spt.fits')
    try:
        sptfile = pyfits.open(sptfilename)
    except IOError:
        print 'SPT file not found %s \n' % sptfilename

    if sptfile:
        hdu0 = sptfile[0].copy()
        try:
            for count in range(1,sci_count+1):
                fitsobj = pyfits.HDUList()            
                fitsobj.append(hdu0)
                hdu = sptfile[count].copy()
                fitsobj.append(hdu)
                rootname = hdu.header['EXPNAME']
                newfilename = fileutil.buildNewRootname(rootname, extn='_spt.fits')
                fitsobj[1].header['EXTVER'] = 1
                if (os.path.exists(newfilename)):
                    os.remove(newfilename)
                    print "       Replacing "+newfilename+"..."
            
                # Write out the new file
                fitsobj.writeto(newfilename)
        except:
            print "Warning: Unable to split spt file %s " % sptfilename
        sptfile.close()
    
    return newfiles 


def update_member_names(oldasndict, pydr_input):
    """
    Purpose
    =======
    Given an association dictionary with rootnames and a list of full 
    file names, it will update the names in the member dictionary to 
    contain '_*' extension. For example a rootname of 'u9600201m' will 
    be replaced by 'u9600201m_c0h' making sure that a MEf file is passed 
    as an input and not the corresponding GEIS file.
    """                 
    nmembers = oldasndict['members'].copy()
    translated_names = [f.split('.fits')[0] for f in pydr_input]
    nkeys = nmembers.keys()
    for file in pydr_input:
        okey = file.split('_')[0]
        if okey in nkeys:
            dmem = nmembers.pop(okey)
            nmembers[file.split('.fits')[0]]= dmem 
        else:
            dmem = nmembers.pop(okey)
    
    oldasndict.pop('members')
    # replace should be always True to cover the case when flt files were removed
    # and the case when names were translated 
    oldasndict.update(members=nmembers, replace=True)
    oldasndict['order'] = translated_names 
    return oldasndict

