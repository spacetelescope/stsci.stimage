import os
import iraf

no = iraf.no
yes = iraf.yes

import tran

# Point to default parameter file for task
_parfile = 'mypy$tran.par'
_taskname= 'tran'
######
# Set up Python IRAF interface here
######
def tran_iraf(origimage,drizimage,direction='f',x=None,y=None,List=None):
    # Transform IRAF empty parameters to Python None when
    # PyDrizzle expects it.
    if origimage == '': origimage = None
    if drizimage == '': drizimage = None
    if List == '': List = None

    print 'Running Tran Version ',tran.__version__

    if direction == 'forward':
        transtat = tran.f(origimage,drizimage,x,y,List)
    else:
        transtat = tran.b(drizimage,origimage,x,y,List)

# Setup this module as an IRAF task here
# by setting up an absolute path to the parfile...
#_ospath = os.path
# File that gets picked up here is: iraffunctions.py
#_abspath = _ospath.split(_ospath.abspath(__file__))[0]
#parfile = os.path.join(_abspath,'pydrizzle.par')

parfile = iraf.osfn(_parfile)
pyd = iraf.IrafTaskFactory(taskname=_taskname, value=parfile, pkgname=PkgName,
            pkgbinary=PkgBinary, function=tran_iraf)
