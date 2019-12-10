"""

Tests module distance

# Author: Vladan Lucic
# $Id: test_distance.py 914 2012-10-25 16:15:17Z vladan $
"""

__version__ = "$Revision: 914 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

import common
from pyto.segmentation.distance import Distance
from pyto.segmentation.segment import Segment

class TestDistance(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """

        ar_1 = numpy.array(
            [[1, 1, 1, 1, 1, 0, 0, 2, 2],
             [0, 0, 0, 0, 0, 0, 0, 2, 2],
             [0, 0, 3, 0, 0, 0, 0, 0, 0],
             [0, 3, 3, 0, 0, 0, 0, 0, 4]])
        self.segments_1 = Segment(data=ar_1) 

    def testCalculate(self):
        """
        Tests calculate() and implicitly getDistance() and setDistance().
        """

        distance = Distance()

        # simple test
        distance.calculate(segments=self.segments_1, ids=(1,2))
        np_test.assert_almost_equal(distance.getDistance(ids=(1,2)), 3)
        np_test.assert_almost_equal(distance.getDistance(ids=(2,1)), 3)
        np_test.assert_almost_equal(distance.getDistance(ids=(3,1)) is None, 
                                    True)
        
        # another distance
        distance.calculate(segments=self.segments_1, ids=(3,1))
        np_test.assert_almost_equal(distance.getDistance(ids=(1,3)), 2)

        # check arg force
        self.segments_1.data[0,5] = 1
        distance.calculate(segments=self.segments_1, ids=(1,2))
        np_test.assert_almost_equal(distance.getDistance(ids=(2,1)), 3)
        np_test.assert_almost_equal(
            distance.calculate(segments=self.segments_1, 
                               ids=(1,2), force=False), 3)
        np_test.assert_almost_equal(
            distance.calculate(segments=self.segments_1, ids=(1,2), force=True),
            2)
        np_test.assert_almost_equal(distance.getDistance(ids=(2,1)), 2)
        self.segments_1.data[0,5] = 0


if __name__ == '__main__':
    suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestDistance)
    unittest.TextTestRunner(verbosity=2).run(suite)
