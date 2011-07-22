from unittest import TestCase

from reftools import imphtcomp

# test that all comparisons come up zero when comparing a file to itself
class TestSameFile(TestCase):
  def setUp(self):
    input_file = 'test_data/acs_sbc_dev_imp.fits'
    self.comp = imphtcomp.ImphttabComp(input_file,input_file)
    
  def test_diffs(self):
    self.assertTrue((self.comp.flamdiff == 0).all())
    self.assertTrue((self.comp.plamdiff == 0).all())
    self.assertTrue((self.comp.bwdiff == 0).all())
