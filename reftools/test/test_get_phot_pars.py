from unittest import TestCase
from nose import tools

from reftools import getphotpars

class TestGetPhotPars(TestCase):
  def setUp(self):
    self.get_pars = getphotpars.GetPhotPars('test_data/test_wfc1_dev_imp.fits')
    
  def tearDown(self):
    self.get_pars.close()
    
  def test_get_row_len(self):
    obsmode = 'acs,wfc1,f625w,f814w,MJD#'
    ext = 'photflam'
    
    row = self.get_pars._get_row(obsmode,ext)
    
    self.assertEqual(len(row),13)
    
  def test_get_row_obs(self):
    obsmode = 'acs,wfc1,f625w,f814w,MJD#'
    ext = 'photflam'
    
    row = self.get_pars._get_row(obsmode,ext)
    
    self.assertEqual(row['obsmode'],obsmode)
  
  def test_parse_obsmode0(self):
    obsmode = 'acs,wfc1,f625w,f660n'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    self.assertEqual(npars,0)
    self.assertEqual(strp_obsmode,'acs,wfc1,f625w,f660n')
    self.assertEqual(par_dict,{})
  
  def test_parse_obsmode1(self):
    obsmode = 'acs,wfc1,f625w,f814w,MJD#55000.0'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    self.assertEqual(npars,1)
    self.assertEqual(strp_obsmode,'acs,wfc1,f625w,f814w,MJD#'.lower())
    self.assertEqual(par_dict.keys()[0],'MJD#'.lower())
    self.assertEqual(par_dict['MJD#'.lower()],55000.0)
    
  def test_parse_obsmode2(self):
    obsmode = 'acs,wfc1,f625w,fr505n#5000.0,MJD#55000.0'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    self.assertEqual(npars,2)
    self.assertEqual(strp_obsmode,'acs,wfc1,f625w,fr505n#,MJD#'.lower())
    self.assertTrue('MJD#'.lower() in par_dict.keys())
    self.assertTrue('fr505n#'.lower() in par_dict.keys())
    self.assertEqual(par_dict['MJD#'.lower()],55000.0)
    self.assertEqual(par_dict['fr505n#'.lower()],5000.0)
    
  def test_make_row_struct0(self):
    obsmode = 'acs,wfc1,f625w,f660n'
    ext = 'photflam'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    row = self.get_pars._get_row(strp_obsmode,ext)
    
    rd = self.get_pars._make_row_struct(row,npars)
    
    self.assertEqual(rd['obsmode'].lower(),obsmode.lower())
    self.assertEqual(rd['datacol'].lower(),'photflam')
    self.assertEqual(rd['parnames'],[])
    self.assertEqual(rd['parnum'],0)
    self.assertEqual(rd['results'],5.8962401031019617e-18)
    self.assertEqual(rd['telem'],1)
    self.assertEqual(rd['nelem'],[])
    self.assertEqual(rd['parvals'],[])
    
  def test_make_row_struct1(self):
    obsmode = 'acs,wfc1,f625w,f814w,MJD#55000.0'
    ext = 'photflam'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    row = self.get_pars._get_row(strp_obsmode,ext)
    
    rd = self.get_pars._make_row_struct(row,npars)
    
    self.assertEqual(rd['obsmode'].lower(),strp_obsmode.lower())
    self.assertEqual(rd['datacol'].lower(),'photflam1')
    self.assertEqual(rd['parnames'],['MJD#'])
    self.assertEqual(rd['parnum'],1)
    self.assertEqual(rd['results'],[8.353948620387228e-18, 8.353948620387228e-18, 
                                    8.439405629259882e-18, 8.439405629259882e-18])
    self.assertEqual(rd['telem'],4)
    self.assertEqual(rd['nelem'],[4])
    self.assertEqual(rd['parvals'],[[52334.0, 53919.0, 53920.0, 55516.0]])
    
  def test_make_row_struct2(self):
    obsmode = 'acs,wfc1,f625w,fr505n#5000.0,MJD#55000.0'
    ext = 'photflam'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    row = self.get_pars._get_row(strp_obsmode,ext)
    
    rd = self.get_pars._make_row_struct(row,npars)
    
    self.assertEqual(rd['obsmode'].lower(),strp_obsmode.lower())
    self.assertEqual(rd['datacol'].lower(),'photflam2')
    self.assertEqual(rd['parnames'],['FR505N#','MJD#'])
    self.assertEqual(rd['parnum'],2)
    self.assertEqual(rd['results'],[7.003710657512407e-14, 6.944751784992699e-14, 
                                    5.933229935875258e-14, 6.903709791285467e-14, 
                                    6.679623972708115e-14, 5.70322329920528e-14, 
                                    6.871738863125632e-14, 6.430043639034794e-14, 
                                    5.2999039030883145e-14, 6.984240267831951e-14, 
                                    6.158040091027122e-14, 6.903709791285467e-14, 
                                    6.679623972708115e-14, 5.70322329920528e-14, 
                                    6.777606555875925e-14, 6.344417923365138e-14, 
                                    5.230630810980452e-14, 6.984240267831951e-14, 
                                    6.158040091027122e-14, 7.102142054088e-14, 
                                    7.038429771136416e-14, 6.012566486354809e-14, 
                                    6.777606555875925e-14, 6.344417923365138e-14, 
                                    5.230630810980452e-14, 6.889989612586935e-14, 
                                    6.076197306179641e-14, 7.102142054088e-14, 
                                    7.038429771136416e-14, 6.012566486354809e-14, 
                                    7.000118709415907e-14, 6.7696605688248e-14, 
                                    5.778989112976214e-14, 6.889989612586935e-14, 
                                    6.076197306179641e-14, 7.003710657512407e-14, 
                                    6.944751784992699e-14, 5.933229935875258e-14, 
                                    7.000118709415907e-14, 6.7696605688248e-14, 
                                    5.778989112976214e-14, 6.871738863125632e-14, 
                                    6.430043639034794e-14, 5.2999039030883145e-14])
    self.assertEqual(rd['telem'],44)
    self.assertEqual(rd['nelem'],[11,4])
    self.assertEqual(rd['parvals'],[[4824.0, 4868.2, 4912.4, 4956.6, 5000.8, 
                                     5045.0, 5089.2, 5133.4, 5177.6, 5221.8, 5266.0],
                                     [52334.0, 53919.0, 53920.0, 55516.0]])
                                     
  def test_make_par_struct0(self):
    obsmode = 'acs,wfc1,f625w,f660n'
    ext = 'photflam'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    ps = self.get_pars._make_par_struct(npars,par_dict)
    
    self.assertEqual(ps['npar'],0)
    self.assertEqual(ps['parnames'],[])
    self.assertEqual(ps['parvals'],[])
    
  def test_make_par_struct1(self):
    obsmode = 'acs,wfc1,f625w,f814w,MJD#55000.0'
    ext = 'photflam'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    ps = self.get_pars._make_par_struct(npars,par_dict)
    
    self.assertEqual(ps['npar'],1)
    self.assertEqual(ps['parnames'],['mjd#'])
    self.assertEqual(ps['parvals'],[55000.0])
    
  def test_make_par_struct2(self):
    obsmode = 'acs,wfc1,f625w,fr505n#5000.0,MJD#55000.0'
    ext = 'photflam'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    ps = self.get_pars._make_par_struct(npars,par_dict)
    
    self.assertEqual(ps['npar'],2)
    self.assertEqual(sorted(ps['parnames']),['fr505n#','mjd#'])
    self.assertEqual(sorted(ps['parvals']),[5000.0,55000.0])
    
  def test_compute_value0(self):
    obsmode = 'acs,wfc1,f625w,f660n'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    ps = self.get_pars._make_par_struct(npars,par_dict)
    
    ext = 'photflam'
    row = self.get_pars._get_row(strp_obsmode,ext)
    rd = self.get_pars._make_row_struct(row,npars)
    
    result = self.get_pars._compute_value(rd,ps)
    
    self.assertEqual(result,5.8962401031019617e-18)
    
    ext = 'photplam'
    row = self.get_pars._get_row(strp_obsmode,ext)
    rd = self.get_pars._make_row_struct(row,npars)
    
    result = self.get_pars._compute_value(rd,ps)
    
    self.assertEqual(result,6599.6045327828697)
    
    ext = 'photbw'
    row = self.get_pars._get_row(strp_obsmode,ext)
    rd = self.get_pars._make_row_struct(row,npars)
    
    result = self.get_pars._compute_value(rd,ps)
    
    self.assertEqual(result,13.622138313347964)
    
  def test_compute_value1(self):
    obsmode = 'acs,wfc1,f625w,f814w,MJD#55000.0'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    ps = self.get_pars._make_par_struct(npars,par_dict)
    
    ext = 'photflam'
    row = self.get_pars._get_row(strp_obsmode,ext)
    rd = self.get_pars._make_row_struct(row,npars)
    
    result = self.get_pars._compute_value(rd,ps)
    
    tools.assert_almost_equals(result,8.43940563e-18,25)
    
    ext = 'photplam'
    row = self.get_pars._get_row(strp_obsmode,ext)
    rd = self.get_pars._make_row_struct(row,npars)
    
    result = self.get_pars._compute_value(rd,ps)
    
    tools.assert_almost_equals(result,6992.37762323,8)
    
    ext = 'photbw'
    row = self.get_pars._get_row(strp_obsmode,ext)
    rd = self.get_pars._make_row_struct(row,npars)
    
    result = self.get_pars._compute_value(rd,ps)
    
    tools.assert_almost_equals(result,58.85223114,8)
    
  def test_compute_value2(self):
    obsmode = 'acs,wfc1,f625w,fr505n#5000.0,MJD#55000.0'
    
    npars, strp_obsmode, par_dict = self.get_pars._parse_obsmode(obsmode)
    
    ps = self.get_pars._make_par_struct(npars,par_dict)
    
    ext = 'photflam'
    row = self.get_pars._get_row(strp_obsmode,ext)
    rd = self.get_pars._make_row_struct(row,npars)
    
    result = self.get_pars._compute_value(rd,ps)
    
    tools.assert_almost_equals(result,5.99660350e-14,21)
    
    ext = 'photplam'
    row = self.get_pars._get_row(strp_obsmode,ext)
    rd = self.get_pars._make_row_struct(row,npars)
    
    result = self.get_pars._compute_value(rd,ps)
    
    tools.assert_almost_equals(result,5737.95131007,8)
    
    ext = 'photbw'
    row = self.get_pars._get_row(strp_obsmode,ext)
    rd = self.get_pars._make_row_struct(row,npars)
    
    result = self.get_pars._compute_value(rd,ps)
    
    tools.assert_almost_equals(result,643.42528984,8)
    
class TestGetPhotParsFunc(TestCase):
  def test_0_old(self):
    obsmode = 'acs,wfc1,f625w,f660n'
    imphttab = 'test_data/test_wfc1_dev_imp.fits'
    
    zpt, flam, plam, bw = getphotpars.get_phot_pars(obsmode, imphttab)
    
    self.assertEqual(zpt,-21.1)
    self.assertEqual(flam,5.8962401031019617e-18)
    self.assertEqual(plam,6599.6045327828697)
    self.assertEqual(bw,13.622138313347964)
    
  def test_1_old(self):
    obsmode = 'acs,wfc1,f625w,f814w,MJD#55000.0'
    imphttab = 'test_data/test_wfc1_dev_imp.fits'
    
    zpt, flam, plam, bw = getphotpars.get_phot_pars(obsmode, imphttab)
    
    self.assertEqual(zpt,-21.1)
    tools.assert_almost_equals(flam,8.43940563e-18,25)
    tools.assert_almost_equals(plam,6992.37762323,8)
    tools.assert_almost_equals(bw,58.85223114,8)
    
  def test_2_old(self):
    obsmode = 'acs,wfc1,f625w,fr505n#5000.0,MJD#55000.0'
    imphttab = 'test_data/test_wfc1_dev_imp.fits'
    
    zpt, flam, plam, bw = getphotpars.get_phot_pars(obsmode, imphttab)
    
    self.assertEqual(zpt,-21.1)
    tools.assert_almost_equals(flam,5.99660350e-14,21)
    tools.assert_almost_equals(plam,5737.95131007,8)
    tools.assert_almost_equals(bw,643.42528984,8)
    
  def test_0_new(self):
    obsmode = 'acs,wfc1,f625w,f660n'
    imphttab = 'test_data/test_acs_wfc1_dev_imp.fits'
    
    zpt, flam, plam, bw = getphotpars.get_phot_pars(obsmode, imphttab)
    
    self.assertEqual(zpt,-21.1)
    self.assertEqual(flam,5.8962401031019617e-18)
    self.assertEqual(plam,6599.6045327828697)
    self.assertEqual(bw,13.622138313347964)
    
  def test_1_new(self):
    obsmode = 'acs,wfc1,f625w,f814w,MJD#55000.0'
    imphttab = 'test_data/test_acs_wfc1_dev_imp.fits'
    
    zpt, flam, plam, bw = getphotpars.get_phot_pars(obsmode, imphttab)
    
    self.assertEqual(zpt,-21.1)
    tools.assert_almost_equals(flam,8.43940563e-18,25)
    tools.assert_almost_equals(plam,6992.37762323,8)
    tools.assert_almost_equals(bw,58.85223114,8)
    
  def test_2_new(self):
    obsmode = 'acs,wfc1,f625w,fr505n#5000.0,MJD#55000.0'
    imphttab = 'test_data/test_acs_wfc1_dev_imp.fits'
    
    zpt, flam, plam, bw = getphotpars.get_phot_pars(obsmode, imphttab)
    
    self.assertEqual(zpt,-21.1)
    tools.assert_almost_equals(flam,5.99660350e-14,21)
    tools.assert_almost_equals(plam,5737.95131007,8)
    tools.assert_almost_equals(bw,643.42528984,8)
