import shutil, os, os.path

PYTOOLS_MODULES = ['imageiter', 'nimageiter', 'numcombine', 'versioninfo', 'makewcs', 'irafglob', 'parseinput','iterfile', 'fitsdiff', 'readgeis', 'fileutil','testutil', 'wcsutil','linefit', 'nmpfit', 'gfit', 'xyinterp', 'numerixenv']
STIS_MODULES = ['sshift', 'stisnoise', 'gettable', 'r_util', 'wavelen', 'evaldisp', 'radialvel', 'mktrace', 'wx2d']


for f in PYTOOLS_MODULES:
    file = f+ '.py'
    cwd = os.getcwd()
    src = os.path.join('pytools', 'lib', file)
    dest = os.path.join(cwd, file)
    shutil.copy2(src, dest)

for f in STIS_MODULES:
    file = f+ '.py'
    cwd = os.getcwd()
    src = os.path.join('stistools', 'lib', file)
    dest = os.path.join(cwd, file)
    shutil.copy2(src, dest)
