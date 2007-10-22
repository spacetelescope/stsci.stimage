import shutil, os, os.path

STIS_MODULES = ['sshift', 'stisnoise', 'gettable', 'r_util', 'wavelen', 'evaldisp', 'radialvel', 'mktrace', 'wx2d']

for f in STIS_MODULES:
    file = f+ '.py'
    cwd = os.getcwd()
    src = os.path.join('stistools', 'lib', file)
    dest = os.path.join(cwd, file)
    shutil.copy2(src, dest)
