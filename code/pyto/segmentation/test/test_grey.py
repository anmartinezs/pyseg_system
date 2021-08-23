"""

Tests module grey

More tests needed

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import

__version__ = "$Revision$"

import importlib

import numpy
import scipy

import unittest
import numpy.testing as np_test

import pyto
from pyto.segmentation.grey import Grey
from pyto.segmentation.segment import Segment
from pyto.segmentation.morphology import Morphology
from pyto.segmentation.contact import Contact
from pyto.segmentation.test import common

class TestGrey(np_test.TestCase):
    """
    """

    def setUp(self):
        importlib.reload(common) # to avoid problems when running multiple tests
        self.shapes = common.make_shapes()
        self.grey = common.make_grey()

    def testSegmentStats(self):
        
        actual = self.grey.getSegmentStats(segment=self.shapes)

        np_test.assert_almost_equal(actual.mean[self.shapes.ids],
                                    [ 22.,  27.,  84.92307692,  79.57142857])
        np_test.assert_almost_equal(actual.std[self.shapes.ids],
                                    [8.206, 12.79, 8.22, 11.22], decimal=2)
        np_test.assert_almost_equal(actual.min[self.shapes.ids],
                                    [ 11.,   6.,  72.,  65.])
        np_test.assert_almost_equal(actual.max[self.shapes.ids],
                                    [ 33.,  48.,  96.,  98.])

    def testSegmentDensitySimple(self):
        
        actual = self.grey.getSegmentDensitySimple(segments=self.shapes)

        np_test.assert_almost_equal(actual.mean[self.shapes.ids],
                                    [ 22.,  27.,  84.92307692,  79.57142857])
        np_test.assert_almost_equal(actual.std[self.shapes.ids],
                                    [8.206, 12.79, 8.22, 11.22], decimal=2)
        np_test.assert_almost_equal(actual.min[self.shapes.ids],
                                    [ 11.,   6.,  72.,  65.])
        np_test.assert_almost_equal(actual.max[self.shapes.ids],
                                    [ 33.,  48.,  96.,  98.])
        np_test.assert_equal(actual.volume[self.shapes.ids],
                             [ 9, 21, 13,  7])
        
    def testLabelByBins(self):
        """
        Tests labelByBins()
        """
        
        # 1-parameter segmentation
        values = numpy.array([[-1, 1, 2, 2, 3, 3],
                              [1, 1, 2, -1, 3, -1],
                              [1, 1, 2, 2, 3, -1],
                              [-1, 1, 2, 2, 3, 3]])

        # simple
        labels, bins = Grey.labelByBins(values=values, 
                                        bins=[1,2,3,4])
        desired = numpy.array([[0, 1, 2, 2, 3, 3],
                                [1, 1, 2, 0, 3, 0],
                                [1, 1, 2, 2, 3, 0],
                                [0, 1, 2, 2, 3, 3]])
        np_test.assert_equal(labels, desired)
        desired_bins = {1:[1,2], 2:[2,3], 3:[3,4]}
        np_test.assert_equal(bins, desired_bins)       

        # masked array
        values = numpy.ma.array(values, mask=(values==2))
        labels, bins = Grey.labelByBins(values=values, 
                                        bins=[1,2,3,4])
        desired = numpy.array([[0, 1, 0, 0, 3, 3],
                                [1, 1, 0, 0, 3, 0],
                                [1, 1, 0, 0, 3, 0],
                                [0, 1, 0, 0, 3, 3]])
        np_test.assert_equal(labels, desired)
        desired_bins = {1:[1,2], 2:[2,3], 3:[3,4]}
        np_test.assert_equal(bins, desired_bins)       

        # multi-parameter segmentation
        values = numpy.zeros(shape=(2, 4, 6), dtype=int)
        values[0] = numpy.array([[-1, 1, 2, 2, 3, 3],
                                [1, 1, 2, -1, 3, -1],
                                [1, 1, 2, 2, 3, -1],
                                [-1, 1, 2, 2, 3, 3]])
        values[1] = numpy.array([[-1, 1, 1, 1, 1, 1],
                                [2, 2, 2, -1, 2, -1],
                                [2, 3, 3, 3, 3, -1],
                                [-1, 4, 4, 4, 3, 4]])

        # simple
        labels, bins = Grey.labelByBins(values=values, 
                                        bins=[[1,2,3,4], [1,2,3,4,5]])
        desired = numpy.array([[0, 1, 5, 5, 9, 9],
                                [2, 2, 6, 0, 10, 0],
                                [2, 3, 7, 7, 11, 0],
                                [0, 4, 8, 8, 11, 12]])
        np_test.assert_equal(labels, desired)
        desired = {
            1:[[1,2],[1,2]], 2:[[1,2],[2,3]], 3:[[1,2],[3,4]], 4:[[1,2],[4,5]],
            5:[[2,3],[1,2]], 6:[[2,3],[2,3]], 7:[[2,3],[3,4]], 8:[[2,3],[4,5]],
            9:[[3,4],[1,2]], 10:[[3,4],[2,3]], 11:[[3,4],[3,4]], 
            12:[[3,4],[4,5]]}
        np_test.assert_equal(bins, desired) 

        # check upper limit
        labels, bins = Grey.labelByBins(values=values, 
                                        bins=[[1,2,3], [1,2,3]])
        desired = numpy.array([[0, 1, 3, 3, 3, 3],
                                [2, 2, 4, 0, 4, 0],
                                [2, 2, 4, 4, 4, 0],
                                [0, 0, 0, 0, 4, 0]])
        np_test.assert_equal(labels, desired)
        desired_bins = {
            1:[[1,2],[1,2]], 2:[[1,2],[2,3]], 
            3:[[2,3],[1,2]], 4:[[2,3],[2,3]]}
        np_test.assert_equal(bins, desired_bins) 

        # check masked arrays, use the above desired
        mask = numpy.array(2 * [values[0] == 2])
        ma_values = numpy.ma.array(values, mask=mask)
        labels, bins = Grey.labelByBins(values=ma_values, 
                                        bins=[[1,2,3], [1,2,3]])
        desired = numpy.array([[0, 1, 0, 0, 3, 3],
                                [2, 2, 0, 0, 4, 0],
                                [2, 2, 0, 0, 4, 0],
                                [0, 0, 0, 0, 4, 0]])
        np_test.assert_equal(labels, desired)
        np_test.assert_equal(bins, desired_bins) 

        # check implicit bins
        labels, bins = Grey.labelByBins(values=values, 
                                        bins=[[1,2,3]])
        desired = numpy.array([[0, 1, 2, 2, 2, 2],
                                [1, 1, 2, 0, 2, 0],
                                [1, 1, 2, 2, 2, 0],
                                [0, 1, 2, 2, 2, 2]])
        np_test.assert_equal(labels, desired)
        desired_bins = {
            1:[[1,2]], 2:[[2,3]]}
        np_test.assert_equal(bins, desired_bins) 


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGrey)
    unittest.TextTestRunner(verbosity=2).run(suite)

