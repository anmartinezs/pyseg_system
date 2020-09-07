"""

Tests module statistics

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
#from past.utils import old_div

__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.statistics import Statistics
#from pyto.segmentation.test import common


class TestStatistics(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        Define arrays and check scipy version
        """

        # data
        self.data = numpy.array(
            [[9, 1, 9, 9, 4, 1, 1, 9],
             [9, 2, 9, 9, 4, 9, 2, 9],
             [3, 3, 4, 5, 4, 6, 1, 4],
             [2, 9, 3, 9, 3, 9, 2, 5]])
        self.labels = numpy.array(
            [[1, 2, 3, 4, 5, 6, 7, 8],
             [1, 2, 3, 4, 5, 6, 7, 8],
             [1, 2, 3, 4, 5, 6, 7, 8],
             [1, 2, 3, 4, 5, 6, 7, 8]])

        # scipy version
        ver = [int(x) for x in scipy.__version__.split('.')]
        if (ver[0] >= 1) or (ver[1] >= 8):
            self.above_equal_08 = True
            self.ddof = 0
        else:
            self.above_equal_08 = False
            self.ddof = 1

    def testCalculateSingleId(self):
        """
        Tests calculate() method for a single id
        """

        # 1D data, ids single int
        st = Statistics()
        st.calculate(data=numpy.arange(6), 
                     labels=numpy.array([2, 2, 2, 5, 5, 5]), ids=2)
        np_test.assert_almost_equal(st.mean, 1.)
        if self.above_equal_08:
            desired_std = numpy.sqrt(2./3)
        else:
            desired_std = 1.
        np_test.assert_almost_equal(st.std, desired_std)
        np_test.assert_almost_equal(st.min, 0)
        np_test.assert_almost_equal(st.max, 2)
        np_test.assert_almost_equal(st.minPos, 0)
        np_test.assert_almost_equal(st.maxPos, 2)

        # >1D data, ids single int
        st = Statistics()
        st.calculate(data=self.data, labels=self.labels, ids=3)
        np_test.assert_almost_equal(st.mean, 25/4.)
        np_test.assert_almost_equal(st.std, self.data[:,2].std(ddof=self.ddof))
        np_test.assert_almost_equal(st.min, 3)
        np_test.assert_almost_equal(st.max, 9)
        if self.above_equal_08:
            np_test.assert_almost_equal(st.minPos, [3, 2])
            try:
                np_test.assert_almost_equal(st.maxPos, [0, 2])
            except AssertionError:
                np_test.assert_almost_equal(st.maxPos, [1, 2])

    def testCalculateArrayIds(self):
        """
        Tests calculate() method for multiple ids
        """

        # >1D data, ids list with one int
        st = Statistics(data=self.data, labels=self.labels, ids=[2])
        st.calculate()
        np_test.assert_almost_equal(st.mean[[2]], self.data[:,1].mean())
        np_test.assert_almost_equal(st.std[[2]], 
                                    self.data[:,1].std(ddof=self.ddof))
        np_test.assert_almost_equal(st.min[[2]], [1])
        np_test.assert_almost_equal(st.max[[2]], [9])
        np_test.assert_almost_equal(st.minPos[[2]], [[0, 1]])
        np_test.assert_almost_equal(st.maxPos[[2]], [[3, 1]])

        st = Statistics(data=self.data, labels=self.labels, ids=[4])
        st.calculate()

        # >1D data, >1 ids
        st = Statistics()
        st.calculate(data=self.data, labels=self.labels, ids=[2,3])
        np_test.assert_almost_equal(st.mean[[0, 2, 3]], [5., 15/4., 25/4.])
        np_test.assert_almost_equal(
            st.std[[0, 2, 3]], 
            [self.data[:,1:3].std(ddof=self.ddof), 
             self.data[:,1].std(ddof=self.ddof), 
             self.data[:,2].std(ddof=self.ddof)])
        np_test.assert_almost_equal(st.min[[0, 2, 3]], [1, 1, 3])
        np_test.assert_almost_equal(st.max[[0, 2, 3]], [9, 9, 9])
        np_test.assert_almost_equal(st.minPos[[0, 2, 3]], 
                                    [[0, 1], [0, 1], [3, 2]])
        np_test.assert_almost_equal(st.maxPos[[2]], [[3, 1]])

 
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStatistics)
    unittest.TextTestRunner(verbosity=2).run(suite)
