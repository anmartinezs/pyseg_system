"""

Tests module series

# Author: Vladan Lucic
# $Id: test_series.py 1367 2016-12-14 15:51:56Z vladan $
"""

__version__ = "$Revision: 1367 $"

import os
from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.tomo.series import Series
#import common


class TestSeries(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """

        # set absolute path to current dir
        working_dir = os.getcwd()
        file_dir, name = os.path.split(__file__)
        self.dir = os.path.join(working_dir, file_dir)

    def testSortPaths(self):
        """
        Tests sortPaths()
        """

        # mode 'num'
        series = Series()
        paths = ['some_dir/neu_14.em',
                 'some_dir/neu_2.em',
                 'some_dir/neu_104.em',
                 'some_dir/neu_25.em'
                 ]
        sorted_num = [
            'some_dir/neu_2.em',
            'some_dir/neu_14.em',
            'some_dir/neu_25.em',
            'some_dir/neu_104.em']
        sorted = series.sortPaths(paths=paths, mode='num')
        np_test.assert_equal(sorted, sorted_num)

        # mode 'sequence'
        seq = [3, 4, 1, 2]
        sorted_seq = [
            'some_dir/neu_104.em',
            'some_dir/neu_25.em',
            'some_dir/neu_14.em',
            'some_dir/neu_2.em']
        sorted = series.sortPaths(paths=paths, mode='sequence', sequence=seq)
        np_test.assert_equal(sorted, sorted_seq)

        # need to test mode 'tilt_angles'

    def testReadTiltAngles(self):
        """
        Tests readTiltAngles()
        """

        path = os.path.join(self.dir, 'tomo_int16.tlt')
        angles = Series.readTiltAngles(file_=path)
        np_test.assert_equal(angles, [-2.8, -5.1, 1.1, -0.9])

    def testGetDose(self):
        """
        Tests getDose()
        """

        # paths
        tomo_path = os.path.join(self.dir, 'tomo_int16.mrc')
        tilt_path = os.path.join(self.dir, 'tomo_int16.tlt')

        # stack, conversion 1
        series = Series(
            path=tomo_path, stack=True, tilt_file=tilt_path)
        total, mean = series.getDose()
        np_test.assert_equal(total, 0.96)
        np_test.assert_equal(mean, 1.5)

        # stack, conversion 5
        series = Series(
            path=tomo_path, stack=True, tilt_file=tilt_path)
        total_dose, mean_counts = series.getDose(conversion=5)
        np_test.assert_almost_equal(total_dose, 0.96/5)
        np_test.assert_equal(mean_counts, 1.5)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSeries)
    unittest.TextTestRunner(verbosity=2).run(suite)
