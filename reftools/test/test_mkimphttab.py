import os
import tempfile

from reftools import mkimphttab

# make sure that when making an imphttab there are no errors
def test_mkimphttab():
  tempdir = tempfile.gettempdir()
  
  output = os.path.join(tempdir, 'test_out_imp.fits')
  
  test_mode = 'acs,sbc'
  
  try:
    mkimphttab.createTable(output, test_mode)
  
  except:
    if os.path.exists(output):
      os.remove(output)
    
    raise
  
  else:
    os.remove(output)
