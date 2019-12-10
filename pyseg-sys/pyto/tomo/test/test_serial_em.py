"""

Tests module serial_em

# Author: Vladan Lucic
# $Id: test_serial_em.py 1314 2016-06-15 09:56:53Z vladan $
"""

__version__ = "$Revision: 1314 $"

import os
from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.tomo.serial_em import SerialEM
#import common


class TestSerialEM(np_test.TestCase):
    """
    Tests class SerialEM
    """

    def setUp(self):
        """
        Sets absolute path to this file directory and saves it as self.dir
        """

        # set absolute path to current dir
        working_dir = os.getcwd()
        file_dir, name = os.path.split(__file__)
        self.dir = os.path.join(working_dir, file_dir)

    def test_parse_single_mdoc(self):
        """
        Tests parse_single_mdoc()
        """
        print os.getcwd()
        print os.path.split(__file__)

        mdoc = os.path.join(
            self.dir, 'stack_original/29.08.14_syn_ctrl01_13.08_tomo01.st.mdoc')
        single = SerialEM(mdoc=mdoc)
        single.parse_single_mdoc()
        np_test.assert_equal(single.apixel, 3.42)
        np_test.assert_equal(len(single.tilt_angles), 46)
        np_test.assert_almost_equal(single.tilt_angles[0], -30., decimal=2)
        np_test.assert_almost_equal(single.tilt_angles[4], -22., decimal=2)
        np_test.assert_almost_equal(single.tilt_angles[10], -10., decimal=2)
        np_test.assert_equal(len(single.exposure_times), 46)
        np_test.assert_almost_equal(single.exposure_times[0], 0.88, decimal=2)
        np_test.assert_almost_equal(single.exposure_times[4], 0.8, decimal=2)
        np_test.assert_almost_equal(single.exposure_times[10], 0.8, decimal=2)
        np_test.assert_equal(single.z_values[0], 0)
        np_test.assert_equal(single.z_values[4], 4)
        np_test.assert_equal(single.z_values[10], 10)
        np_test.assert_equal(single.orig_frame_names[0], 'Aug29_21.02.09.mrc')
        np_test.assert_equal(single.orig_frame_names[4], 'Aug29_21.04.47.mrc')
        np_test.assert_equal(single.orig_frame_names[10], 'Aug29_21.08.00.mrc')

    def test_parse_mdocs(self):
        """
        Tests parse_mdocs()
        """

        sem = SerialEM(dir_=os.path.join(self.dir, 'stack_original'))
        sem.parse_mdocs()
        np_test.assert_equal(sem.apixel, 3.42)
        np_test.assert_equal(len(sem.tilt_angles), 61)
        np_test.assert_almost_equal(sem.tilt_angles[0], -60., decimal=2)
        np_test.assert_almost_equal(sem.tilt_angles[10], -40., decimal=2)
        np_test.assert_almost_equal(sem.tilt_angles[40], 20., decimal=1)
        np_test.assert_equal(len(sem.exposure_times), 61)
        np_test.assert_almost_equal(sem.exposure_times[0], 1.04, decimal=2)
        np_test.assert_almost_equal(sem.exposure_times[10], 0.88, decimal=2)
        np_test.assert_almost_equal(sem.exposure_times[40], 0.8, decimal=1)
        np_test.assert_equal(sem.orig_frame_names[0], 'Aug29_21.47.07.mrc')
        np_test.assert_equal(sem.orig_frame_names[10], 'Aug29_21.40.26.mrc')
        np_test.assert_equal(sem.orig_frame_names[40], 'Aug29_21.17.03.mrc')
        np_test.assert_equal(
            sem.stack_names[0], '29.08.14_syn_ctrl01_13.08_tomo01_01.st')
        np_test.assert_equal(
            sem.stack_names[10], '29.08.14_syn_ctrl01_13.08_tomo01_01.st')
        np_test.assert_equal(
            sem.stack_names[40], '29.08.14_syn_ctrl01_13.08_tomo01.st')
        


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSerialEM)
    unittest.TextTestRunner(verbosity=2).run(suite)
