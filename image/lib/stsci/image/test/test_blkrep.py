from __future__ import division

import os
import gc
import stsci.image.blkrep as blkrep
import pyfits
import testutil

class TestBlkrepFLT(testutil.FPTestCase):

    def setUp(self):
        if not os.path.isfile('py_output_flt.fits'):
            data = os.path.join(os.path.dirname(__file__), 'data',
                                'j8ux08ceq_flt.fits')
            blkrep.blkrep(data, 'py_output_flt.fits', 1, 2, 2)

        self.flt_testfile = 'py_output_flt.fits'

        self.flt_hdulist = pyfits.open(self.flt_testfile, memmap=True)
        self.flt_data = self.flt_hdulist[0].data
        self.flt_hdulist.close()

        self.flt_prihdr = pyfits.getheader(self.flt_testfile)

    def test_fltDim(self):
        ref_x_dim = 8192
        ref_y_dim = 4096
        x_dim = self.flt_data.shape[1]
        y_dim = self.flt_data.shape[0]
        self.assertEqual(x_dim,ref_x_dim)
        self.assertEqual(y_dim,ref_y_dim)

    def test_fltNAXIS(self):
        ref_naxis1 = 8192
        ref_naxis2 = 4096
        NAXIS1 = self.flt_prihdr['NAXIS1']
        NAXIS2 = self.flt_prihdr['NAXIS2']
        self.assertEqual(NAXIS1,ref_naxis1)
        self.assertEqual(NAXIS2,ref_naxis2)

    def test_fltCRPIX(self):
        ref_crpix1 = 4095.619331550194
        ref_crpix2 = 2047.559665775097
        CRPIX1 = self.flt_prihdr['CRPIX1']
        CRPIX2 = self.flt_prihdr['CRPIX2']
        self.assertApproxFP(CRPIX1,ref_crpix1,accuracy=0.0025)
        self.assertApproxFP(CRPIX2,ref_crpix2,accuracy=0.0025)

    def test_fltCD(self):
        ref_cd1_1 = -6.2566166788090E-06
        ref_cd1_2 = -3.4199288768941E-06
        ref_cd2_1 = -2.9650647229861E-06
        ref_cd2_2 = 6.09333136841377E-06
        CD1_1 = self.flt_prihdr['CD1_1']
        CD1_2 = self.flt_prihdr['CD1_2']
        CD2_1 = self.flt_prihdr['CD2_1']
        CD2_2 = self.flt_prihdr['CD2_2']
        self.assertApproxFP(CD1_1,ref_cd1_1,accuracy=0.0025)
        self.assertApproxFP(CD1_2,ref_cd1_2,accuracy=0.0025)
        self.assertApproxFP(CD2_1,ref_cd2_1,accuracy=0.0025)
        self.assertApproxFP(CD2_2,ref_cd2_2,accuracy=0.0025)

    def test_fltLTV(self):
        ref_ltv1 = -0.5
        ref_ltv2 = -0.5
        LTV1 = self.flt_prihdr['LTV1']
        LTV2 = self.flt_prihdr['LTV2']
        self.assertApproxFP(LTV1,ref_ltv1,accuracy=0.0025)
        self.assertApproxFP(LTV2,ref_ltv2,accuracy=0.0025)

    def test_fltLTM(self):
        ref_ltm1_1 = 2.000058267358493
        ref_ltm2_2 = 2.000058267358493
        LTM1_1 = self.flt_prihdr['LTM1_1']
        LTM2_2 = self.flt_prihdr['LTM2_2']
        self.assertApproxFP(LTM1_1,ref_ltm1_1,accuracy=0.0025)
        self.assertApproxFP(LTM2_2,ref_ltm2_2,accuracy=0.0025)

    def test_fltnewKW(self):
        ref_wcsdim=2
        ref_wat0_001='system=image'
        ref_wat1_001='wtype=tan axtype=ra'
        ref_wat2_001='wtype=tan axtype=dec'
        wcsdim = self.flt_prihdr['WCSDIM']
        wat0_001 = self.flt_prihdr['WAT0_001']
        wat1_001 = self.flt_prihdr['WAT1_001']
        wat2_001 = self.flt_prihdr['WAT2_001']
        self.assertEqual(wcsdim,ref_wcsdim)
        self.assertEqual(wat0_001,ref_wat0_001)
        self.assertEqual(wat1_001,ref_wat1_001)
        self.assertEqual(wat2_001,ref_wat2_001)


class TestBlkrepDRZ(testutil.FPTestCase):

    def setUp(self):
        if not os.path.isfile('py_output_drz.fits'):
            data = os.path.join(os.path.dirname(__file__), 'data',
                                'final_drz_sci.fits')
            blkrep.blkrep(data, 'py_output_drz.fits', 0, 2, 2)

        self.drz_testfile = 'py_output_drz.fits'

        self.drz_hdulist = pyfits.open(self.drz_testfile, memmap=True)
        self.drz_data = self.drz_hdulist[0].data
        self.drz_hdulist.close()

        self.drz_prihdr = pyfits.getheader(self.drz_testfile)

    def test_drzDim(self):
        ref_x_dim = 8422
        ref_y_dim = 8482
        x_dim = self.drz_data.shape[1]
        y_dim = self.drz_data.shape[0]
        self.assertEqual(x_dim,ref_x_dim)
        self.assertEqual(y_dim,ref_y_dim)

    def test_drzNAXIS(self):
        ref_naxis1 = 8422
        ref_naxis2 = 8482
        NAXIS1 = self.drz_prihdr['NAXIS1']
        NAXIS2 = self.drz_prihdr['NAXIS2']
        self.assertEqual(NAXIS1,ref_naxis1)
        self.assertEqual(NAXIS2,ref_naxis2)

    def test_drzCRPIX(self):
        ref_crpix1 = 4210.5
        ref_crpix2 = 4240.5
        CRPIX1 = self.drz_prihdr['CRPIX1']
        CRPIX2 = self.drz_prihdr['CRPIX2']
        self.assertApproxFP(CRPIX1,ref_crpix1,accuracy=0.0025)
        self.assertApproxFP(CRPIX2,ref_crpix2,accuracy=0.0025)

    def test_drzCD(self):
        ref_cd1_1 = -6.1835364124385E-06
        ref_cd1_2 = -3.1601245862719E-06
        ref_cd2_1 = -3.1601248166236E-06
        ref_cd2_2 =  6.1835348626909E-06
        CD1_1 = self.drz_prihdr['CD1_1']
        CD1_2 = self.drz_prihdr['CD1_2']
        CD2_1 = self.drz_prihdr['CD2_1']
        CD2_2 = self.drz_prihdr['CD2_2']
        self.assertApproxFP(CD1_1,ref_cd1_1,accuracy=0.0025)
        self.assertApproxFP(CD1_2,ref_cd1_2,accuracy=0.0025)
        self.assertApproxFP(CD2_1,ref_cd2_1,accuracy=0.0025)
        self.assertApproxFP(CD2_2,ref_cd2_2,accuracy=0.0025)

    def test_drzLTV(self):
        ref_ltv1 = -0.5
        ref_ltv2 = -0.5
        LTV1 = self.drz_prihdr['LTV1']
        LTV2 = self.drz_prihdr['LTV2']
        self.assertApproxFP(LTV1,ref_ltv1,accuracy=0.0025)
        self.assertApproxFP(LTV2,ref_ltv2,accuracy=0.0025)

    def test_drzLTM(self):
        ref_ltm1_1 = 2.0
        ref_ltm2_2 = 2.0
        LTM1_1 = self.drz_prihdr['LTM1_1']
        LTM2_2 = self.drz_prihdr['LTM2_2']
        self.assertApproxFP(LTM1_1,ref_ltm1_1,accuracy=0.0025)
        self.assertApproxFP(LTM2_2,ref_ltm2_2,accuracy=0.0025)

    def test_drznewKW(self):
        ref_wcsdim = 2
        ref_wat0_001 = 'system=image'
        ref_wat1_001 = 'wtype=tan axtype=ra'
        ref_wat2_001 = 'wtype=tan axtype=dec'
        wcsdim = self.drz_prihdr['WCSDIM']
        wat0_001 = self.drz_prihdr['WAT0_001']
        wat1_001 = self.drz_prihdr['WAT1_001']
        wat2_001 = self.drz_prihdr['WAT2_001']
        self.assertEqual(wcsdim,ref_wcsdim)
        self.assertEqual(wat0_001,ref_wat0_001)
        self.assertEqual(wat1_001,ref_wat1_001)
        self.assertEqual(wat2_001,ref_wat2_001)
        gc.collect()

if __name__ == '__main__':
    if 'debug' in sys.argv:
        testutil.debug(__name__)
    else:
        result=testutil.testall(__name__,2)
