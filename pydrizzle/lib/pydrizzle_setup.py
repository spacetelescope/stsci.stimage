import os
import platform
import sys


def setup_hook(config):
    if platform.architecture()[0] == '64bit':
        # This should suffice, since for an extension module we still need to
        # use whatever architecture Python was built for
        f2c_macros = [('Skip_f2c_Undefs', 1), ('Allow_TYQUAD', 1),
                      ('INTEGER_STAR_8', 1)]
        f2c_files = ['ftell64_.c', 'pow_qq.c', 'qbitbits.c', 'qbitshft.c']
    else:
        f2c_macros = [('Skip_f2c_Undefs', 1)]
        f2c_files = ['ftell_.c']

    if sys.platform == 'win32':
        f2c_macros += [('USE_CLOCK', 1), ('MSDOS', 1), ('NO_ONEXIT', 1)]

    f2c_files = [os.path.join('libf2c', f) for f in f2c_files]
    ext = config['extension=pydrizzle.arrdriz']
    ext['define_macros'] += '\n' + '\n'.join('%s=%s' % m for m in f2c_macros)
    ext['sources'] += '\n' + '\n'.join(f2c_files)
