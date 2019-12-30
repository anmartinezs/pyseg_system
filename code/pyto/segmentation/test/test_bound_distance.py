"""

Tests module bound_distance

# Author: Vladan Lucic
# $Id: test_bound_distance.py 916 2012-10-26 16:22:13Z vladan $
"""

__version__ = "$Revision: 916 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

import common
from pyto.segmentation.segment import Segment
from pyto.segmentation.contact import Contact
from pyto.segmentation.bound_distance import BoundDistance

class TestBoundDistance(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """

        bound_ar = numpy.array(
            [[1, 1, 1, 1, 1, 0, 2, 2, 2],
             [0, 0, 0, 0, 0, 0, 0, 2, 2],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 3, 0, 0, 0, 0, 0, 0],
             [0, 3, 3, 0, 0, 0, 4, 4, 4],
             [0, 3, 3, 0, 0, 0, 4, 4, 4]])
        self.bound = Segment(data=bound_ar) 
        segment_ar = numpy.array(
            [[0, 0, 0, 0, 0, 5, 0, 0, 0],
             [1, 0, 6, 0, 0, 0, 8, 0, 0],
             [1, 0, 6, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 7, 7, 0, 9, 0, 0],
             [1, 0, 0, 0, 7, 0, 0, 0, 0],
             [0, 0, 0, 0, 7, 7, 0, 0, 0]])
        self.segments = Segment(data=segment_ar)
        self.contacts = Contact()
        self.contacts.findContacts(
            segment=self.segments, boundary=self.bound, count=False)

    def testMerge(self):
        """
        Tests merge() 
        """

        dist1 = BoundDistance(ids=[1,4,5])
        dist1.distance = numpy.array([-1, 2, -1, -1, 5, 6])
        dist2 = BoundDistance(ids=[2])
        dist2.distance = numpy.array([-1, 1, 3])
        dist3 = BoundDistance(ids=[4,7])
        dist3.distance = numpy.array([-1, -1, -1, -1, 6, -1, -1, 8])

        dist1.merge(new=dist2)
        np_test.assert_equal(dist1.ids, numpy.array([1, 2, 4, 5]))
        np_test.assert_equal(dist1.distance, numpy.array([-1, 2, 3, -1, 5, 6]))
        dist1.merge(new=dist3)
        np_test.assert_equal(dist1.ids, numpy.array([1, 2, 4, 5, 7]))
        np_test.assert_equal(dist1.distance, 
                             numpy.array([-1, 2, 3, -1, 6, 6, -1, 8]))
        
    def testCalculate(self):
        """
        Tests calculate() and implicitly extend()
        """
        
        # segments one by one
        dist = BoundDistance()
        dist.calculate(contacts=self.contacts, boundaries=self.bound, ids=[1])
        np_test.assert_equal(dist.ids, numpy.array([1]))
        np_test.assert_almost_equal(dist.distance, numpy.array([-1, 3]))
        dist.calculate(contacts=self.contacts, boundaries=self.bound, 
                        ids=[7, 8])
        np_test.assert_equal(dist.ids, numpy.array([1, 7]))
        np_test.assert_almost_equal(
                            dist.distance,
                            numpy.array([-1, 3, -1, -1, -1, -1, -1, 4]))
                             
        # segments all at once, no extend
        dist = BoundDistance()
        dist.calculate(contacts=self.contacts, boundaries=self.bound)
        np_test.assert_equal(dist.ids, numpy.array([1, 5, 6, 7]))
        np_test.assert_almost_equal(
                            dist.distance,
                            numpy.array([-1, 3, -1, -1, -1, 2, 3, 4]))
        
        # segments all at once, extend without arg ids
        dist = BoundDistance()
        dist.calculate(contacts=self.contacts, boundaries=self.bound, 
                       extend=True)
        np_test.assert_equal(dist.ids, numpy.array([1, 5, 6, 7]))
        np_test.assert_almost_equal(
                            dist.distance,
                            numpy.array([-1, 3, -1, -1, -1, 2, 3, 4]))
 
        # segments all at once, extend with arg ids
        dist = BoundDistance()
        dist.calculate(contacts=self.contacts, boundaries=self.bound, 
                       ids = [1, 5, 6, 7, 8, 9], extend=True)
        np_test.assert_equal(dist.ids, numpy.array([1, 5, 6, 7, 8, 9]))
        np_test.assert_almost_equal(
                            dist.distance,
                            numpy.array([-1, 3, -1, -1, -1, 2, 3, 4, -1, -1]))
        
                             
if __name__ == '__main__':
    suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestBoundDistance)
    unittest.TextTestRunner(verbosity=2).run(suite)
