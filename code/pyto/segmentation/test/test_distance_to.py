"""

Tests module distance_to

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
#from past.utils import old_div

__version__ = "$Revision$"

from copy import copy, deepcopy
import importlib
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.test import common
from pyto.segmentation.distance_to import DistanceTo
from pyto.segmentation.segment import Segment
#from pyto.segmentation.hierarchy import Hierarchy


class TestDistanceTo(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        importlib.reload(common) # to avoid problems when running multiple tests

        # regions
        bound_data = numpy.zeros((10,10), dtype=int)
        bound_data[0:5, 0] = 3
        bound_data[9, 0:5] = 4
        self.bound = Segment(bound_data)

        # expected distances to region
        self.dist_mean = numpy.array([[2, 7, 5.46184388, 7.49631542],
                                      [7, 7.72255285, 1.10878566, 3.49817009]])
        self.dist_min = numpy.array([[1, 5, numpy.sqrt(13), numpy.sqrt(29)],
                                     [6, numpy.sqrt(29), 0, numpy.sqrt(8)]])
        self.dist_cl_min = numpy.array([1, 5, 0, numpy.sqrt(8)])
        self.cl_reg_min = numpy.array([3, 3, 4, 4])
        self.dist_max = numpy.array([[3, 9, numpy.sqrt(61), numpy.sqrt(89)],
                                     [8, numpy.sqrt(97), 2, numpy.sqrt(17)]])
        self.dist_median = numpy.array([[2, 7, numpy.sqrt(29), numpy.sqrt(58)],
                                        [7, numpy.sqrt(61), 1, numpy.sqrt(13)]])
        self.dist_center = numpy.array([[2, 7, numpy.sqrt(25), numpy.sqrt(58)],
                                        [7, numpy.sqrt(58), 1, numpy.sqrt(13)]])
        self.dist_median_s1 = numpy.array(
            [[2, 6.5, numpy.sqrt(25), numpy.sqrt(58)],
             [7, (numpy.sqrt(61) + numpy.sqrt(50)) / 2, 2, numpy.sqrt(13)]])
        self.dist_cl_median_s1 = numpy.array([2, 6.5, 2, numpy.sqrt(13)])
        self.cl_median_s1 = numpy.array([3, 3, 4, 4])


        # expected distances among segments
        self.seg_3_min = numpy.array([2, numpy.sqrt(13), 2])
        self.seg_3_4_min = numpy.array([2, numpy.sqrt(2)])
        self.seg_3_4_closest = numpy.array([3, 4])

    def testGetDistanceToRegion(self):
        """
        Tests getDistanceToRegion
        """
        
        # min 
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes)
        all_dist = dist.getDistanceToRegions(
            segments=shapes.data, segmentIds=shapes.ids, regionIds=[3,4], 
            regions=self.bound.data, mode='min')
        np_test.assert_almost_equal(all_dist, self.dist_min)

        # max 
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes)
        all_dist = dist.getDistanceToRegions(
            segments=shapes.data, segmentIds=shapes.ids, regionIds=[3,4], 
            regions=self.bound.data, mode='max')
        np_test.assert_almost_equal(all_dist, self.dist_max)

        # mean 
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes)
        all_dist = dist.getDistanceToRegions(
            segments=shapes.data, segmentIds=shapes.ids, regionIds=[3,4], 
            regions=self.bound.data, mode='mean')
        np_test.assert_almost_equal(all_dist, self.dist_mean)

        # median 
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes)
        all_dist = dist.getDistanceToRegions(
            segments=shapes.data, segmentIds=shapes.ids, regionIds=[3,4], 
            regions=self.bound.data, mode='median')
        np_test.assert_almost_equal(all_dist, self.dist_median)

        # median, surface 1
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes)
        all_dist = dist.getDistanceToRegions(
            segments=shapes.data, segmentIds=shapes.ids, regionIds=[3,4], 
            regions=self.bound.data, mode='median', surface=1)
        np_test.assert_almost_equal(all_dist, self.dist_median_s1)

        # center
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes)
        all_dist = dist.getDistanceToRegions(
            segments=shapes.data, segmentIds=shapes.ids, regionIds=[3,4], 
            regions=self.bound.data, mode='center')
        np_test.assert_almost_equal(all_dist, self.dist_center)

    def testCalculate(self):
        """
        Tests calculate
        """
        
        # one region, mean
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes)
        dist.calculate(regionIds=[3], regions=self.bound, mode='mean')
        np_test.assert_almost_equal(dist._distance, self.dist_mean[0,:])
        np_test.assert_almost_equal(dist.distance[dist.ids], 
                                    self.dist_mean[0,:])
        np_test.assert_almost_equal(dist._closestRegion, [3] * 4)
        np_test.assert_almost_equal(dist.closestRegion[dist.ids], [3] * 4)

        # two regions, min
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes)
        dist.calculate(regionIds=[3,4], regions=self.bound, mode='min')
        np_test.assert_almost_equal(dist._distance, self.dist_cl_min)
        np_test.assert_almost_equal(dist.distance[dist.ids], 
                                    self.dist_cl_min)
        np_test.assert_almost_equal(dist._closestRegion, self.cl_reg_min)
        np_test.assert_almost_equal(dist.closestRegion[dist.ids], 
                                    self.cl_reg_min)

        # segments and regions from the same object
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes, ids=[1,4,6])
        dist.calculate(regionIds=[3], mode='min')
        np_test.assert_almost_equal(dist.distance[dist.ids], self.seg_3_min)

        # segments and regions from the same object
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes, ids=[1,6])
        dist.calculate(regionIds=[3,4], mode='min')
        np_test.assert_almost_equal(dist.distance[dist.ids], self.seg_3_4_min)
        np_test.assert_almost_equal(dist.closestRegion[dist.ids], 
                                    self.seg_3_4_closest)

        # insets
        shapes = common.make_shapes()
        shapes.makeInset(ids=[1,3])
        bound = Segment(self.bound.data)
        bound.makeInset(ids=[4])
        dist = DistanceTo(segments=shapes, ids=[1,3])
        dist.calculate(regionIds=[4], regions=bound, mode='min')
        np_test.assert_almost_equal(dist.distance[dist.ids], 
                                    [6, numpy.sqrt(29)])
        
        # median, surface 1 
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes)
        all_dist = dist.calculate(segments=shapes, regionIds=[3,4], 
            regions=self.bound, mode='median', surface=1)
        np_test.assert_almost_equal(dist._distance, self.dist_cl_median_s1)
        np_test.assert_equal(dist._closestRegion, self.cl_median_s1)
        
    def testGetSetDistance(self):
        """
        Tests getDistance and setDistance
        """

        # without ids
        dist = DistanceTo()
        dist.setIds([1,3,4,6])
        dist.setDistance(distance=numpy.arange(7))
        np_test.assert_almost_equal(dist.distance, [-1, 1, -1, 3, 4, -1, 6])
        np_test.assert_almost_equal(dist.getDistance(), 
                                    [-1, 1, -1, 3, 4, -1, 6])
        np_test.assert_almost_equal(dist._distance, [1, 3, 4, 6])
        
        # with ids
        dist = DistanceTo()
        dist.setIds([1,3,6])
        dist.setDistance(distance=numpy.arange(6), ids=[2,4,5])
        np_test.assert_almost_equal(dist._distance, [2, 4, 5])
        np_test.assert_almost_equal(dist.distance, [-1, 2, -1, 4, -1, -1, 5])
        np_test.assert_almost_equal(dist.getDistance(ids=[2,4,5]), 
                                   [-1, -1, 2, -1, 4, 5, -1])
                                   
    def testGetSetClosestRegion(self):
        """
        Tests getClosestRegion and setClosestRegion
        """

        # without ids
        dist = DistanceTo()
        dist.setIds([1,3,4,6])
        dist.setClosestRegion(closestRegion=numpy.arange(7))
        np_test.assert_almost_equal(dist._closestRegion, [1, 3, 4, 6])
        np_test.assert_almost_equal(dist.closestRegion, 
                                    [-1, 1, -1, 3, 4, -1, 6])
        np_test.assert_almost_equal(dist.getClosestRegion(), 
                                    [-1, 1, -1, 3, 4, -1, 6])
        
        # with ids
        dist = DistanceTo()
        dist.setIds([1,3,6])
        dist.setClosestRegion(closestRegion=numpy.arange(6), ids=[2,4,5])
        np_test.assert_almost_equal(dist._closestRegion, [2, 4, 5])
        np_test.assert_almost_equal(dist.closestRegion, 
                                    [-1, 2, -1, 4, -1, -1, 5])
        np_test.assert_almost_equal(dist.getClosestRegion(ids=[2,4,5]), 
                                   [-1, -1, 2, -1, 4, 5])
                                   
    def testMerge(self):
        """
        Test merge
        """

        # one region, mean
        shapes = common.make_shapes()
        dist = DistanceTo(segments=shapes, ids=[1, 4])
        dist.calculate(regionIds=[3], regions=self.bound, mode='mean')
        np_test.assert_almost_equal(dist.distance[dist.ids], 
                                    self.dist_mean[[0,0], [0,2]])
        np_test.assert_almost_equal(dist._closestRegion, [3] * 2)
        np_test.assert_almost_equal(dist.closestRegion[dist.ids], [3] * 2)

        # other region
        dist_2 = DistanceTo(segments=shapes, ids=[3, 6])
        dist_2.calculate(regionIds=[3], regions=self.bound, mode='mean')
        np_test.assert_almost_equal(dist_2.distance[dist_2.ids], 
                                    self.dist_mean[[0,0],[1,3]])
        np_test.assert_almost_equal(dist._closestRegion, [3] * 2)
        np_test.assert_almost_equal(dist.closestRegion[dist.ids], [3] * 2)

        # merge replace
        dist.merge(new=dist_2, mode='replace')
        np_test.assert_equal(dist.ids, [1, 3, 4, 6])
        np_test.assert_almost_equal(dist.distance[dist.ids], 
                                    self.dist_mean[0,:])
        np_test.assert_almost_equal(dist._closestRegion, [3] * 4)
        np_test.assert_almost_equal(dist.closestRegion[dist.ids], [3] * 4)


if __name__ == '__main__':
    suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestDistanceTo)
    unittest.TextTestRunner(verbosity=2).run(suite)
