"""

Tests module cleft

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
#from past.utils import old_div

__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest
#import sys

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.cleft import Cleft
from pyto.core.image import Image
from pyto.segmentation.segment import Segment

class TestCleft(np_test.TestCase):
    """
    """

    def setUp(self):
        
        # make image
        self.image_1 = Image(numpy.arange(100).reshape(10,10)) 

        # make cleft
        cleft_data_1 = numpy.zeros((10,10), dtype=int)
        cleft_data_1[slice(1,7), slice(3,9)] = numpy.array(\
            [[0, 5, 5, 5, 0, 0],
             [2, 5, 5, 5, 4, 4],
             [2, 5, 5, 5, 4, 4],
             [2, 2, 5, 5, 5, 4],
             [2, 5, 5, 5, 4, 4],
             [2, 5, 5, 5, 4, 4]])
        self.bound_1 = Segment(cleft_data_1)

        # make more complicated cleft array
        self.cleft_data_2 = numpy.zeros((15, 15), dtype=int)
        self.cleft_data_2[slice(2, 12), slice(3, 12)] = numpy.array(\
            [[7, 7, 7, 7, 7, 7, 7, 7, 7],
             [0, 1, 1, 1, 1, 2, 2, 2, 0],
             [0, 1, 1, 1, 1, 2, 2, 2, 0],
             [6, 6, 5, 5, 5, 5, 5, 5, 6],
             [6, 6, 5, 5, 5, 5, 5, 5, 6],
             [6, 6, 5, 5, 5, 5, 5, 5, 6],
             [6, 6, 5, 5, 5, 5, 5, 5, 6],
             [0, 3, 3, 3, 3, 3, 3, 3, 0],
             [3, 3, 3, 3, 3, 3, 3, 3, 3],
             [8, 8, 8, 8, 9, 9, 8, 8, 8]])

        # cleft useful for limiting maximum distance 
        self.cleft_data_3 = numpy.zeros((10,10), dtype=int)
        self.cleft_data_3[slice(1,7), slice(3,9)] = numpy.array(\
            [[1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2],
             [2, 2, 3, 2, 2, 2],
             [2, 3, 3, 3, 2, 2],
             [3, 3, 3, 3, 3, 2],
             [3, 3, 3, 3, 3, 3]])

    def testWidth(self):
        """
        Tests Cleft.getWidth()
        """

        cl_1 = Cleft(data=self.bound_1.data, bound1Id=2, bound2Id=4, cleftId=5,
                     copy=True, clean=False)

        # test mean mode 
        mean_0, mean_0_vec = cl_1.getWidth(mode='mean', toBoundary=0)
        mean_1, mean_1_vec = cl_1.getWidth(mode='mean', toBoundary=1)
        mean_2, mean_2_vec = cl_1.getWidth(mode='mean', toBoundary=2)
        desired = numpy.array(\
            [1.5 * numpy.sqrt(10) + numpy.sqrt(13) + 2.5 * 4,
             2 * numpy.sqrt(10) + 2 * numpy.sqrt(13) + 1 * 4,
             1 * numpy.sqrt(10) + 0 * numpy.sqrt(13) + 4 * 4]) 
        desired = (desired / 5) - 1
        np_test.assert_almost_equal([mean_0, mean_1, mean_2], desired) 
        desired_vec = [0, (4 * 3 + 6 * 4) / 10.]
        np_test.assert_almost_equal(mean_0_vec.data, desired_vec, decimal=0) 
        desired_vec = [0, (4 * 3 + 4) / 5.]
        np_test.assert_almost_equal(mean_1_vec.data, desired_vec) 
        desired_vec = [0, 4]
        np_test.assert_almost_equal(mean_2_vec.data, desired_vec, decimal=0) 

        # test median mode
        med_0, med_0_vec = cl_1.getWidth(mode='median', toBoundary=0)
        med_1, med_1_vec = cl_1.getWidth(mode='median', toBoundary=1)
        med_2, med_2_vec = cl_1.getWidth(mode='median', toBoundary=2)
        desired = numpy.array(
            [(numpy.sqrt(13) + 4) / 2, numpy.sqrt(13), 4]) - 1
        np_test.assert_almost_equal([med_0, med_1, med_2], desired) 

        # test max distance
        cl_3 = Cleft(data=self.cleft_data_3, bound1Id=1, bound2Id=3, cleftId=2,
                     copy=True, clean=False)
        mean_1, mean_1_vec = cl_3.getWidth(mode='mean', toBoundary=1, 
                                           maxDistance=None)
        np_test.assert_almost_equal(mean_1, numpy.mean([3, 2, 1, 2, 3, 4]))
        mean_1, mean_1_vec = cl_3.getWidth(mode='mean', toBoundary=1, 
                                           maxDistance=4.1)
        np_test.assert_almost_equal(mean_1, numpy.mean([3, 2, 1, 2, 3]))
        mean_1, mean_1_vec = cl_3.getWidth(mode='mean', toBoundary=1, 
                                           maxDistance=3.1)
        np_test.assert_almost_equal(mean_1, numpy.mean([2, 1, 2]))
        np_test.assert_almost_equal(mean_1_vec.data, [(3 + 2 + 3)/3., 0])


    def testDistances(self):
        """
        Tests Cleft.getBoundaryDistances
        """

        cl = Cleft(data=self.bound_1.data, bound1Id=2, bound2Id=4, cleftId=5, 
                   copy=True, clean=False)

        # no inset
        dis_to_1, pos_to_1, pos_2, dis_to_2, pos_to_2, pos_1 = \
            cl.getBoundaryDistances()

        des_dis_1 = [numpy.sqrt(13), numpy.sqrt(10), 4, numpy.sqrt(10), 
                     numpy.sqrt(13)]
        np_test.assert_almost_equal(dis_to_1, des_dis_1) 
        np_test.assert_equal(pos_to_1, [[4]*5, [4]*5])

        des_dis_2 = [4, 4, numpy.sqrt(10), 4, 4] 
        np_test.assert_almost_equal(dis_to_2, des_dis_2) 
        np_test.assert_equal(pos_2, [[2,3,4,5,6], [7,7,8,7,7]])
        try:
            np_test.assert_equal(pos_to_2, [[2,3,3,5,6], [7,7,7,7,7]])
        except AssertionError:
            np_test.assert_equal(pos_to_2, [[2,3,5,5,6], [7,7,7,7,7]])
        np_test.assert_equal(pos_1, [[2,3,4,5,6], [3,3,4,3,3]])

        # with tight inset
        cl.useInset(inset=[slice(1,7), slice(3,9)])
        dis_to_1, pos_to_1, pos_2, dis_to_2, pos_to_2, pos_1 = \
            cl.getBoundaryDistances()

        des_dis_1 = [numpy.sqrt(13), numpy.sqrt(10), 4, numpy.sqrt(10), 
                     numpy.sqrt(13)]
        np_test.assert_almost_equal(dis_to_1, des_dis_1) 
        np_test.assert_equal(pos_to_1, [[4]*5, [4]*5])

        des_dis_2 = [4, 4, numpy.sqrt(10), 4, 4] 
        np_test.assert_almost_equal(dis_to_2, des_dis_2) 
        np_test.assert_equal(pos_2, [[2,3,4,5,6], [7,7,8,7,7]])
        try:
            np_test.assert_equal(pos_to_2, [[2,3,3,5,6], [7,7,7,7,7]])
        except AssertionError:
            np_test.assert_equal(pos_to_2, [[2,3,5,5,6], [7,7,7,7,7]])
        np_test.assert_equal(pos_1, [[2,3,4,5,6], [3,3,4,3,3]])

        # intermediate inset
        cl = Cleft(data=self.bound_1.data, bound1Id=2, bound2Id=4, cleftId=5, 
                   copy=True, clean=False)
        cl.useInset(inset=[slice(1,7), slice(2,9)])

        dis_to_1, pos_to_1, pos_2, dis_to_2, pos_to_2, pos_1 = \
            cl.getBoundaryDistances()

        des_dis_1 = [numpy.sqrt(13), numpy.sqrt(10), 4, numpy.sqrt(10), 
                     numpy.sqrt(13)]
        np_test.assert_almost_equal(dis_to_1, des_dis_1) 
        np_test.assert_equal(pos_to_1, [[4]*5, [4]*5])

        des_dis_2 = [4, 4, numpy.sqrt(10), 4, 4] 
        np_test.assert_almost_equal(dis_to_2, des_dis_2) 
        np_test.assert_equal(pos_2, [[2,3,4,5,6], [7,7,8,7,7]])
        try:
            np_test.assert_equal(pos_to_2, [[2,3,3,5,6], [7,7,7,7,7]])
        except AssertionError:
            np_test.assert_equal(pos_to_2, [[2,3,5,5,6], [7,7,7,7,7]])
        np_test.assert_equal(pos_1, [[2,3,4,5,6], [3,3,4,3,3]])

    def testMakeLayers(self):
        """
        Tests Cleft.makeLayers(). This method wraps 
        Segment.makeLayersBetweenMore, so extensive tests are in test_segment. 
        """

        # no extra layers
        cleft = Cleft(data=self.cleft_data_2, cleftId=5, 
                      bound1Id=numpy.array([1,2]), bound2Id=3)
        layers, width = cleft.makeLayers(nExtraLayers=3)
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 2, 2, 2, 2, 2, 2, 2, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 0],
             [0, 0, 4, 4, 4, 4, 4, 4, 0],
             [0, 0, 5, 5, 5, 5, 5, 5, 0],
             [0, 0, 6, 6, 6, 6, 6, 6, 0],
             [0, 0, 7, 7, 7, 7, 7, 7, 0],
             [0, 8, 8, 8, 8, 8, 8, 8, 0],
             [10, 9, 9, 9, 9, 9, 9, 9, 9],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(layers.data, desired[1:9, :])
        layers.useInset(inset=cleft.inset, mode='abs', expand=True)
        np_test.assert_almost_equal(width, 4)
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # multiple cleft ids, extra layers, max distance
        cleft = Cleft(data=self.cleft_data_2, cleftId=[5,6], 
                      bound1Id=numpy.array([1,2]), bound2Id=3)
        layers, width = cleft.makeLayers(nExtraLayers=3, maxDistance=5.5,
                                         extra_1=7, extra_2=[8,9])
        layers.useInset(inset=cleft.inset, mode='abs', expand=True)
        desired = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 2, 2, 2, 2, 2, 2, 2, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 0],
             [0, 4, 4, 4, 4, 4, 4, 4, 0],
             [5, 5, 5, 5, 5, 5, 5, 5, 5],
             [6, 6, 6, 6, 6, 6, 6, 6, 6],
             [0, 7, 7, 7, 7, 7, 7, 7, 0],
             [0, 8, 8, 8, 8, 8, 8, 8, 0],
             [9, 9, 9, 9, 9, 9, 9, 9, 9],
             [10, 10, 10, 10, 10, 10, 10, 10, 10]])
        np_test.assert_almost_equal(width, 4)
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # test layers with nLayers and maxDistance
        cl_3 = Cleft(data=self.cleft_data_3, bound1Id=1, bound2Id=3, cleftId=2,
                     copy=True, clean=False)
        layers, mean_1 = cl_3.makeLayers(nLayers=2, widthMode='mean', 
                                         maxDistance=3)
        desired = numpy.array([[0, 0, 0, 0, 0, 0],
                               [0, 1, 2, 1, 0, 0],
                               [0, 2, 0, 2, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(layers.data, desired)
        layers, mean_1 = cl_3.makeLayers(nLayers=2, widthMode='mean', 
                                         maxDistance=4)
        desired = numpy.array([[0, 0, 0, 0, 0, 0],
                               [1, 1, 2, 1, 1, 1],
                               [2, 2, 0, 2, 2, 0],
                               [2, 0, 0, 0, 2, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(layers.data, desired)
        layers, mean_1 = cl_3.makeLayers(nLayers=2, widthMode='mean', 
                                         maxDistance=None)
        desired = numpy.array([[0, 0, 0, 0, 0, 0],
                               [1, 1, 2, 1, 1, 1],
                               [2, 2, 0, 2, 2, 1],
                               [2, 0, 0, 0, 2, 2],
                               [0, 0, 0, 0, 0, 2],
                               [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(layers.data, desired)
        layers, mean_1 = cl_3.makeLayers(nLayers=3, widthMode='mean', 
                                         maxDistance=3)
        desired = numpy.array([[0, 0, 0, 0, 0, 0],
                               [0, 2, 2, 2, 0, 0],
                               [0, 3, 0, 3, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(layers.data, desired)
        layers, mean_1 = cl_3.makeLayers(nLayers=3, widthMode='mean', 
                                         maxDistance=4)
        desired = numpy.array([[0, 0, 0, 0, 0, 0],
                               [1, 2, 2, 2, 1, 1],
                               [2, 3, 0, 3, 2, 0],
                               [3, 0, 0, 0, 3, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(layers.data, desired)
        layers, mean_1 = cl_3.makeLayers(nLayers=3, widthMode='mean', 
                                         maxDistance=None)
        desired = numpy.array([[0, 0, 0, 0, 0, 0],
                               [1, 2, 2, 2, 1, 1],
                               [2, 3, 0, 3, 2, 2],
                               [3, 0, 0, 0, 3, 3],
                               [0, 0, 0, 0, 0, 3],
                               [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(layers.data, desired)

        # test max distance
        cl_3 = Cleft(data=self.cleft_data_3, bound1Id=1, bound2Id=3, cleftId=2,
                     copy=True, clean=False)
        layers, mean_1 = cl_3.makeLayers(widthMode='mean', maxDistance=3)
        desired = numpy.mean([3, 2, 3, 
                              2, numpy.sqrt(5), numpy.sqrt(5), 
                              numpy.sqrt(8), numpy.sqrt(8)]) - 1
        np_test.assert_almost_equal(mean_1, desired)
        desired = numpy.array([[0, 0, 0, 0, 0, 0],
                               [0, 1, 2, 1, 0, 0],
                               [0, 2, 0, 2, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(layers.data, desired)
        layers, mean_1 = cl_3.makeLayers(widthMode='mean', maxDistance=4)
        desired = numpy.mean([4, 3, 2, 3, 4, 
                              2, numpy.sqrt(5), numpy.sqrt(5), 
                              numpy.sqrt(8), numpy.sqrt(8), numpy.sqrt(13)]) - 1
        np_test.assert_almost_equal(mean_1, desired)
        desired = numpy.array([[0, 0, 0, 0, 0, 0],
                               [1, 1, 2, 1, 1, 1],
                               [2, 2, 0, 2, 2, 0],
                               [2, 0, 0, 0, 2, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(layers.data, desired)
        layers, mean_1, = cl_3.makeLayers(widthMode='mean', maxDistance=None)
        desired = numpy.mean([4, 3, 2, 3, 4, 5, 
                              2, numpy.sqrt(5), numpy.sqrt(5), 
                              numpy.sqrt(8), numpy.sqrt(8), 
                              numpy.sqrt(13)]) - 1
        np_test.assert_almost_equal(mean_1, desired)
        desired = numpy.array([[0, 0, 0, 0, 0, 0],
                               [1, 1, 2, 1, 1, 1],
                               [2, 2, 0, 2, 2, 1],
                               [2, 0, 0, 0, 2, 2],
                               [0, 0, 0, 0, 0, 2],
                               [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(layers.data, desired)

    def testAdjustToRegions(self):
        """
        Tests adjustToRegions()
        """

        # all ids
        cleft = Cleft(data=self.cleft_data_2, cleftId=[5,6], 
                      bound1Id=numpy.array([1,2]), bound2Id=3)
        regions_data = numpy.array(
            [[1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4],
             [5, 5, 5, 5, 5],
             [6, 6, 6, 6, 6]])
        regions = Segment(data=regions_data)
        regions.inset = [slice(4, 10), slice(5, 10)]
        cleft.adjustToRegions(regions=regions)
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 2, 2, 0, 0],
             [0, 0, 5, 5, 5, 5, 5, 0, 0],
             [0, 0, 5, 5, 5, 5, 5, 0, 0],
             [0, 0, 5, 5, 5, 5, 5, 0, 0],
             [0, 0, 5, 5, 5, 5, 5, 0, 0],
             [0, 0, 3, 3, 3, 3, 3, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(cleft.data[2:12, 3:12], desired)

        # specified ids, value
        cleft = Cleft(data=self.cleft_data_2, cleftId=[5,6], 
                      bound1Id=numpy.array([1,2]), bound2Id=3)
        regions_data = numpy.array(
            [[1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4],
             [5, 5, 5, 5, 5],
             [6, 6, 6, 6, 6]])
        regions = Segment(data=regions_data)
        regions.inset = [slice(4, 10), slice(5, 10)]
        cleft.adjustToRegions(regions=regions, ids=[1, 3, 5], value=9)
        desired = numpy.array(
            [[7, 7, 7, 7, 7, 7, 7, 7, 7],
             [0, 9, 9, 9, 9, 2, 2, 2, 0],
             [0, 9, 1, 1, 1, 2, 2, 2, 0],
             [6, 6, 5, 5, 5, 5, 5, 9, 6],
             [6, 6, 5, 5, 5, 5, 5, 9, 6],
             [6, 6, 5, 5, 5, 5, 5, 9, 6],
             [6, 6, 5, 5, 5, 5, 5, 9, 6],
             [0, 9, 3, 3, 3, 3, 3, 9, 0],
             [9, 9, 9, 9, 9, 9, 9, 9, 9],
             [8, 8, 8, 8, 9, 9, 8, 8, 8]])
        np_test.assert_equal(cleft.data[2:12, 3:12], desired)

    @classmethod
    def makeRectangle(cls):
        
        data = numpy.array(
            [[0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0], 
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0]])
        cleft = Cleft(data)
        return cleft

    @classmethod
    def makeHorizontal(cls):
        
        data = numpy.array(
            [[0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 0, 3, 3, 3, 0], 
             [0, 0, 4, 4, 4, 0],
             [0, 5, 5, 5, 5, 0]])
        cleft = Cleft(data)
        return cleft

    @classmethod
    def makeSlanted(cls):

        data = numpy.array(
            [[5, 5, 5, 1, 0, 0],
             [5, 5, 1, 1, 2, 0],
             [5, 5, 1, 2, 2, 3],
             [5, 1, 1, 2, 3, 3],
             [0, 1, 2, 2, 3, 6],
             [0, 0, 2, 3, 3, 6],
             [0, 0, 0, 3, 6, 6]])
        cleft = Cleft(data)
        return cleft

    @classmethod
    def make3DRectangle(cls):
        
        data = numpy.array(
            [[5, 5, 5, 5, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5], 
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 5, 5, 5, 5, 5]])
        ar0 = numpy.zeros(shape=data.shape)
        ar0 = numpy.expand_dims(ar0, axis=-1)
        ar23 = numpy.expand_dims(data, axis=-1)
        ar23 = numpy.concatenate([ar0, ar23, ar0], axis=-1)
        cleft = Cleft(ar23)
        return cleft

    @classmethod
    def make3DHorizontal(cls):

        data = numpy.array([[8, 8, 8, 8, 8, 8, 8],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 2, 2, 2, 2, 2, 0],
                            [0, 3, 3, 3, 3, 3, 0],
                            [8, 8, 8, 8, 8, 8, 8]])
        ar0 = numpy.zeros(shape=(5, 7, 1))
        ar23 = numpy.expand_dims(data, axis=-1)
        ar3 = numpy.concatenate([ar0, ar23, ar23, ar23, ar0], axis=-1)
        cleft = Cleft(ar3)
        return cleft

    @classmethod
    def make3DSlanted(cls):

        ar0 = numpy.zeros(shape=cls.makeSlanted().data.shape)
        ar0 = numpy.expand_dims(ar0, axis=-1)
        ar23 = numpy.expand_dims(cls.makeSlanted().data, axis=-1)
        ar3 = numpy.concatenate([ar0, ar23, ar23, ar23, ar0], axis=-1)
        cleft = Cleft(ar3)
        return cleft
    
    def testPickCenter(self):
        """
        Tests the basic functionality of pickCenter(). The rest is tested in
        testPickLayersCenters()
        """

        hor_cl = self.makeHorizontal()
        sla_cl = self.makeSlanted()

        # horizontal, one max distance
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        center = hor_cl.pickCenter(id_=2, distance=dist)
        np_test.assert_equal(center, [2, 2]) 
        
        # horizontal, one max distance, nonexisting layer
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        center = hor_cl.pickCenter(id_=5, distance=dist, fromPosition=[2,2])
        np_test.assert_equal(center is None, True) 
        
        # 1D, >1 max distances, cm discriminatory
        dist = numpy.array([-1, 2, 2, 2, 1, 1, -1])
        cl_data = numpy.array([0, 3, 3, 3, 3, 3, 0])
        cl = Cleft(cl_data)
        center = cl.pickCenter(id_=3, distance=dist)
        np_test.assert_equal(center, [3]) 

        # 2D, >1 max distances, cm discriminatory
        dist = numpy.array([[-1, 2, 2, 2, 1, 1, -1],
                            [-1, 2, 3, 4, 5, 6, 3]])
        cl_data = numpy.array([[0, 3, 3, 3, 3, 3, 0],
                               [2, 2, 2, 2, 2, 2, 2]])
        cl = Cleft(cl_data)
        center = cl.pickCenter(id_=3, distance=dist)
        np_test.assert_equal(center, [0,3]) 

        # 2D, >1 max distances, cm not discriminatory
        dist = numpy.array([[-1, 2, 2, 2, 1, -1, -1],
                            [-1, 2, 3, 4, 5, 6, 3]])
        cl_data = numpy.array([[0, 3, 3, 3, 3, 3, 0],
                               [2, 2, 2, 2, 2, 2, 2]])
        cl = Cleft(cl_data)
        center = cl.pickCenter(id_=3, distance=dist)
        try:
            np_test.assert_equal(center, [0,2]) 
        except AssertionError:
            # numerical ambiguity
            np_test.assert_equal(center, [0,3]) 

    def testPickLayerCenters(self):
        """
        Tests pickLayerCenters() and pickCenter()
        """

        hor_cl = self.makeHorizontal()
        sla_cl = self.makeSlanted()
        sla_3d_cl = self.make3DSlanted()

        # horizontal
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        centers = hor_cl.pickLayerCenters(
            ids=[2,3,4], distance=dist, mode='one', startId=2)
        desired = {2 : [2, 2], 3 : [3, 3], 4 : [4, 3]}
        np_test.assert_equal(centers, desired) 
       
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        centers = hor_cl.pickLayerCenters(
            ids=[2,3,4], distance=dist, mode='one', startId=3)
        desired = {2 : [2, 2], 3 : [3, 3], 4 : [4, 3]}
        np_test.assert_equal(centers, desired) 
       
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        centers = hor_cl.pickLayerCenters(
            ids=[2,3,4], distance=dist, mode='one', startId=4)
        desired = {2 : [2, 2], 3 : [3, 3], 4 : [4, 3]}
        np_test.assert_equal(centers, desired) 

        # notFound == 'previous' 
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        centers = hor_cl.pickLayerCenters(
            ids=[2,5,3,4], distance=dist, mode='one', startId=2)
        desired = {2 : [2, 2], 3 : [3, 3], 4 : [4, 3], 5:[2,2]}
        np_test.assert_equal(centers, desired) 
       
        # notFound is None 
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        centers = hor_cl.pickLayerCenters(
            ids=[2,5,3,4], distance=dist, mode='one', startId=2, notFound=None)
        desired = {2 : [2, 2], 3 : [3, 3], 4 : [4, 3], 5:None}
        np_test.assert_equal(centers, desired) 
       
        # slanted, euclidean
        dist = sla_cl.elementDistanceToRim(
            ids=[1,2,3], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        centers = sla_cl.pickLayerCenters(
            ids=[1,2,3], distance=dist, mode='one', startId=1)
        desired = {1 : [2, 2], 2 : [3, 3], 3 : [4, 4]}
        np_test.assert_equal(centers, desired) 
       
        # slanted, geodesic conn=1
        dist = sla_cl.elementDistanceToRim(
            ids=[1,2,3], metric='geodesic', connectivity=1, rimId=0, 
            rimLocation='out', rimConnectivity=1)
        centers = sla_cl.pickLayerCenters(
            ids=[1,2,3], distance=dist, mode='one', startId=1)
        desired = {1 : [2, 2], 2 : [3, 3], 3 : [4, 4]}
        np_test.assert_equal(centers, desired) 
       
        # slanted, geodesic conn=-1
        dist = sla_cl.elementDistanceToRim(
            ids=[1,2,3], metric='geodesic', connectivity=-1, rimId=0, 
            rimLocation='out', rimConnectivity=1)
        centers = sla_cl.pickLayerCenters(
            ids=[1,2,3], distance=dist, mode='one', startId=1)
        desired = {1 : [1, 2], 2 : [2, 3], 3 : [3, 4]}
        try:
            np_test.assert_equal(centers, desired)
        except AssertionError:
            # numerical ambiguity in choosing center 
            for id_ in [1,2,3]:
                np_test.assert_equal(centers[id_][1], desired[id_][1])
       
        # 3d, geodesic conn=-1
        dist = sla_3d_cl.elementDistanceToRim(
            ids=[1,2,3], metric='euclidean', connectivity=1, rimId=0, 
            rimLocation='out', rimConnectivity=1)
        centers = sla_3d_cl.pickLayerCenters(
            ids=[1,2,3], distance=dist, mode='one', startId=1)
        desired = {1 : [2, 2, 2], 2 : [3, 3, 2], 3 : [4, 4, 2]}
        #np_test.assert_equal(centers, desired) 
      
    def testParametrizeLayers(self):
        """
        Tests parametrizeLayers()
        """

        rec_cl = self.makeRectangle()
        hor_cl = self.makeHorizontal()
        sla_cl = self.makeSlanted()
        hor_3d_cl = self.make3DHorizontal()
        sla_3d_cl = self.make3DSlanted()
        def sq(x): return numpy.sqrt(x)

        # simple one rectangular layer
        coord = rec_cl.parametrizeLayers(
            ids=[1], system='radial', normalize=False, startId=None, 
            metric='euclidean', rimId=0, rimLocation='in')
        #sr2 = numpy.sqrt(2)
        #sr5 = numpy.sqrt(5)
        #sr10 = numpy.sqrt(10)
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0],
             [0, sq(10), 3, sq(10), 0, 0],
             [0, sq(5), 2, sq(5), 0, 0],
             [0, sq(2), 1, sq(2), 0, 0],
             [0, 1,   0,  1, 0, 0],
             [0, sq(2), 1, sq(2), 0, 0],
             [0, sq(5), 2, sq(5), 0, 0],
             [0, sq(10), 3, sq(10), 0, 0],
             [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(coord, desired)

        # simple one rectangular layer, normalized
        coord = rec_cl.parametrizeLayers(
            ids=[1], system='radial', normalize=True, startId=None, 
            metric='euclidean', rimId=0, rimLocation='in')
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 2/3., 1, 0, 0],
             [0, 1, 1/2., 1, 0, 0],
             [0, 1,   0,  1, 0, 0],
             [0, 1, 1/2., 1, 0, 0],
             [0, 1, 2/3., 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(coord, desired)

        # degenerate case, all elements have distance to rim 0
        coord = hor_cl.parametrizeLayers(
            ids=[1], system='radial', normalize=False, startId=None, 
            metric='euclidean', rimId=0, rimLocation='in')
        desired = numpy.zeros(shape=(6,6))
        desired[1, 1:5] = numpy.array([[0., 1, 2, 3]])
        try:
            np_test.assert_almost_equal(coord, desired) 
        except AssertionError:
            # if this passes probably ok because there's a numerical
            # ambiguity about chosing the origin whan all distances 0
            np_test.assert_equal((coord[1, 1:5] == 0).any(), True)

        # 2D horizontal, rim out
        coord = hor_cl.parametrizeLayers(
            ids=[2,3,4], system='radial', normalize=False, startId=2, 
            metric='euclidean', rimId=0, rimLocation='out')
        desired = numpy.zeros(shape=(6,6))
        desired[2:5, 1:5] = numpy.array([[1., 0, 1, -1],
                                         [-1, 1, 0, 1],
                                         [-1, 1, 0, 1]])
        np_test.assert_almost_equal(coord, desired) 

        # 2D horizontal, rim in
        coord = hor_cl.parametrizeLayers(
            ids=[2,3,4], system='radial', normalize=True, startId=2, 
            metric='euclidean', rimId=0, rimLocation='in')
        desired = numpy.zeros(shape=(6,6))
        desired[2:5, 1:5] = numpy.array([[1., 0, 1, -1],
                                         [-1, 1, 0, 1],
                                         [-1, 1, 0, 1]])
        np_test.assert_almost_equal(coord, desired) 

        # 2D horizontal, normalized
        coord = hor_cl.parametrizeLayers(
            ids=[2,3,4], system='radial', normalize=True, startId=2, 
            metric='euclidean', rimId=0, rimLocation='out')
        desired = numpy.zeros(shape=(6,6))
        desired[2:5, 1:5] = numpy.array([[0.5, 0, 0.5, -1],
                                         [-1, 0.5, 0, 0.5],
                                         [-1, 0.5, 0, 0.5]])
        np_test.assert_almost_equal(coord, desired) 

        # 2D slanted, geodesic conn=1
        coord = sla_cl.parametrizeLayers(
            ids=[1,2,3], system='radial', normalize=False, startId=2, 
            metric='geodesic', connectivity=1, rimId=0, rimLocation='out')
        desired = numpy.array([[-1, -1, -1, 3, -1, -1],
                               [-1, -1, 1, 2, 3, -1],
                               [-1, -1, 0, 1, 2, 3],
                               [-1, 2, 1, 0, 1, 2],
                               [-1, 3, 2, 1, 0, -1],
                               [-1, -1, 3, 2, 1, -1],
                               [-1, -1, -1, 3, -1, -1]])
        np_test.assert_almost_equal(coord, desired) 
        
        # 2D slanted, geodesic conn=2
        coord = sla_cl.parametrizeLayers(
            ids=[1,2,3], system='radial', normalize=False, startId=2, 
            metric='euclidean-geodesic', connectivity=-1, 
            rimId=0, rimLocation='out')
        desired = numpy.array([[-1, -1, -1, 1+sq(2), -1, -1],
                               [-1, -1, 1, sq(2), 1+sq(2), -1],
                               [-1, -1, 0, 1, sq(2), 1+sq(2)],
                               [-1, sq(2), 1, 0, 1, sq(2)],
                               [-1, 1+sq(2), sq(2), 1, 0, -1],
                               [-1, -1, 1+sq(2), sq(2), 1, -1],
                               [-1, -1, -1, 1+sq(2), -1, -1]])
        np_test.assert_almost_equal(coord, desired) 
        
        # 3D horizontal cartesian (origin: [2, 3, 2])
        coord = hor_3d_cl.parametrizeLayers(
            ids=[1,2,3], system='cartesian', normalize=False, startId=1, 
            metric='euclidean', connectivity=1, rimId=0, rimLocation='out')
        np_test.assert_almost_equal(coord[:,2,3,2], [0,0,0])
        np_test.assert_almost_equal(coord[0,1,1,1:4], [1,0,-1])
        np_test.assert_almost_equal(coord[0,2,1,1:4], [1,0,-1])
        np_test.assert_almost_equal(coord[0,2,3,1:4], [1,0,-1])
        for z_coord in [1,2,3]:
            np_test.assert_almost_equal(
                coord[0,:,:,z_coord], numpy.zeros(shape=(5,7)) + 2 - z_coord)
        for y_coord in [1,2,3,4,5]:
            np_test.assert_almost_equal(
                coord[1,:,y_coord,:], numpy.zeros(shape=(5,5)) - 3 + y_coord)
        for x_coord in [1,2,3]:
            np_test.assert_almost_equal(
                coord[2,x_coord,:,:], numpy.zeros(shape=(7,5)))

        # 3D horizontal polar
        coord = hor_3d_cl.parametrizeLayers(
            ids=[1,2,3], system='polar', normalize=False, startId=1, 
            metric='euclidean', connectivity=1, rimId=0, rimLocation='out')
        np_test.assert_almost_equal(coord[:,2,3,2], [0,0,0])
        for x_coord in [1,2,3]:
            np_test.assert_almost_equal(coord[1,x_coord,3,1:4], 
                                        [0, 0, numpy.pi])
            np_test.assert_almost_equal(coord[1,x_coord,1:6,2], 
                                        numpy.array([-1,-1,0,1,1]) * numpy.pi/2)
        
        # 3D slanted radial
        coord = sla_3d_cl.parametrizeLayers(
            ids=[1,2,3], system='radial', normalize=False, startId=2, 
            metric='geodesic', connectivity=1, rimId=0, rimLocation='out')
        desired = numpy.zeros(shape=(7,6,5)) - 1
        desired_1 = numpy.array([[-1, -1, -1, 3, -1, -1],
                                 [-1, -1,  1, 2,  3, -1],
                                 [-1, -1,  0, 1,  2, 3],
                                 [-1,  2,  1, 0,  1, 2],
                                 [-1,  3,  2, 1,  0, -1],
                                 [-1, -1,  3, 2,  1, -1],
                                 [-1, -1, -1, 3, -1, -1]])
        desired[:,:,2] = desired_1
        desired[:,:,1] = numpy.where(desired_1>=0, desired_1+1, -1)
        desired[:,:,3] = numpy.where(desired_1>=0, desired_1+1, -1)
        np_test.assert_almost_equal(coord, desired) 
      
        # 3D slanted cartesian (phi=45, theta=90, origin=[3,3,2]), euclidean
        coord = sla_3d_cl.parametrizeLayers(
            ids=[1,2,3], system='cartesian', normalize=False, startId=2, 
            metric='euclidean', connectivity=1, rimId=0, rimLocation='out')
        # id 1
        try:
            # python 2
            np_test.assert_almost_equal(coord[:,2,2,2], [0,0,0])
            np_test.assert_almost_equal(coord[:,1,2,2], [0,sq(2)/2, -sq(2)/2])
            np_test.assert_almost_equal(coord[:,1,3,2], [0,sq(2),0])
            np_test.assert_almost_equal(coord[:,3,2,2], [0,-sq(2)/2, sq(2)/2])
            np_test.assert_almost_equal(coord[:,3,1,2], [0,-sq(2),0])
            np_test.assert_almost_equal(coord[:,2,2,3], [-1,0,0])
            np_test.assert_almost_equal(coord[:,1,2,3], [-1,sq(2)/2, -sq(2)/2])
            np_test.assert_almost_equal(coord[:,1,3,3], [-1,sq(2),0])
            np_test.assert_almost_equal(coord[:,2,2,1], [1,0,0])
        except AssertionError:
            # global ambiguity in the direction of best fit line in
            # scp.linalg.svd() (flipped between python 2 and 3)
            np_test.assert_almost_equal(coord[:,2,2,2], [0,0,0])
            np_test.assert_almost_equal(coord[:,1,2,2], [0,-sq(2)/2, sq(2)/2])
            np_test.assert_almost_equal(coord[:,1,3,2], [0,-sq(2),0])
            np_test.assert_almost_equal(coord[:,3,2,2], [0,sq(2)/2, -sq(2)/2])
            np_test.assert_almost_equal(coord[:,3,1,2], [0,sq(2),0])
            np_test.assert_almost_equal(coord[:,2,2,3], [-1,0,0])
            np_test.assert_almost_equal(coord[:,1,2,3], [-1,-sq(2)/2, sq(2)/2])
            np_test.assert_almost_equal(coord[:,1,3,3], [-1,-sq(2),0])
            np_test.assert_almost_equal(coord[:,2,2,1], [1,0,0])            
        # id 2
        try:
            np_test.assert_almost_equal(coord[:,3,3,2], [0,0,0])
            np_test.assert_almost_equal(coord[:,2,3,2], [0,sq(2)/2,-sq(2)/2])
            np_test.assert_almost_equal(coord[:,2,4,2], [0,sq(2),0])
            np_test.assert_almost_equal(coord[:,4,3,2], [0,-sq(2)/2,sq(2)/2])
            np_test.assert_almost_equal(coord[:,4,2,2], [0,-sq(2),0])
            np_test.assert_almost_equal(coord[:,3,3,3], [-1,0,0])
            np_test.assert_almost_equal(coord[:,2,3,3], [-1,sq(2)/2,-sq(2)/2])
            np_test.assert_almost_equal(coord[:,2,4,3], [-1,sq(2),0])
            np_test.assert_almost_equal(coord[:,3,3,1], [1,0,0])
        except AssertionError:
            np_test.assert_almost_equal(coord[:,3,3,2], [0,0,0])
            np_test.assert_almost_equal(coord[:,2,3,2], [0,-sq(2)/2,sq(2)/2])
            np_test.assert_almost_equal(coord[:,2,4,2], [0,-sq(2),0])
            np_test.assert_almost_equal(coord[:,4,3,2], [0,sq(2)/2,-sq(2)/2])
            np_test.assert_almost_equal(coord[:,4,2,2], [0,sq(2),0])
            np_test.assert_almost_equal(coord[:,3,3,3], [-1,0,0])
            np_test.assert_almost_equal(coord[:,2,3,3], [-1,-sq(2)/2,sq(2)/2])
            np_test.assert_almost_equal(coord[:,2,4,3], [-1,-sq(2),0])
            np_test.assert_almost_equal(coord[:,3,3,1], [1,0,0])            
        # id 3
        try:
            np_test.assert_almost_equal(coord[:,4,4,2], [0,0,0])
            np_test.assert_almost_equal(coord[:,3,4,2], [0,sq(2)/2,-sq(2)/2])
            np_test.assert_almost_equal(coord[:,3,5,2], [0,sq(2),0])
            np_test.assert_almost_equal(coord[:,5,4,2], [0,-sq(2)/2,sq(2)/2])
            np_test.assert_almost_equal(coord[:,5,3,2], [0,-sq(2),0])
            np_test.assert_almost_equal(coord[:,4,4,3], [-1,0,0])
            np_test.assert_almost_equal(coord[:,3,4,3], [-1,sq(2)/2,-sq(2)/2])
            np_test.assert_almost_equal(coord[:,3,5,3], [-1,sq(2),0])
            np_test.assert_almost_equal(coord[:,4,4,1], [1,0,0])
        except AssertionError:
            np_test.assert_almost_equal(coord[:,4,4,2], [0,0,0])
            np_test.assert_almost_equal(coord[:,3,4,2], [0,-sq(2)/2,sq(2)/2])
            np_test.assert_almost_equal(coord[:,3,5,2], [0,-sq(2),0])
            np_test.assert_almost_equal(coord[:,5,4,2], [0,sq(2)/2,-sq(2)/2])
            np_test.assert_almost_equal(coord[:,5,3,2], [0,sq(2),0])
            np_test.assert_almost_equal(coord[:,4,4,3], [-1,0,0])
            np_test.assert_almost_equal(coord[:,3,4,3], [-1,-sq(2)/2,sq(2)/2])
            np_test.assert_almost_equal(coord[:,3,5,3], [-1,-sq(2),0])
            np_test.assert_almost_equal(coord[:,4,4,1], [1,0,0])
        # 3D slanted cartesian (phi=45, theta=90, origin=[3,3,2]), geodesic
        coord = sla_3d_cl.parametrizeLayers(
            ids=[1,2,3], system='cartesian', normalize=False, startId=2, 
            metric='geodesic', connectivity=1, rimId=0, rimLocation='out')
        try:
            self.block_3d_slanted_cartesian(coord)
        except AssertionError:
            # global ambiguity in the direction of best fit line in
            # scp.linalg.svd() (flipped between python 2 and 3)
            coord[1,:,:,:] = -coord[1,:,:,:]
            self.block_3d_slanted_cartesian(coord)
            
        # 3D slanted polar
        coord = sla_3d_cl.parametrizeLayers(
            ids=[1,2,3], system='polar', normalize=False, startId=2, 
            metric='geodesic', connectivity=1, rimId=0, rimLocation='out')
        # r
        desired_r = numpy.zeros(shape=(7,6,5)) - 1
        desired_r1 = numpy.array([[-1, -1, -1, 3, -1, -1],
                                  [-1, -1,  1, 2,  3, -1],
                                  [-1, -1,  0, 1,  2, 3],
                                  [-1,  2,  1, 0,  1, 2],
                                  [-1,  3,  2, 1,  0, -1],
                                  [-1, -1,  3, 2,  1, -1],
                                  [-1, -1, -1, 3, -1, -1]])
        desired_r[:,:,2] = desired_r1
        desired_r[:,:,1] = numpy.where(desired_r1>=0, desired_r1+1, -1)
        desired_r[:,:,3] = numpy.where(desired_r1>=0, desired_r1+1, -1)
        np_test.assert_almost_equal(coord[0,:,:,:], desired_r) 

        try:
            self.block_3d_slanted_polar(coord)
        except AssertionError:
            # global ambiguity in the direction of best fit line in
            # scp.linalg.svd() (flipped between python 2 and 3)
            coord[1,:,:,:] = -coord[1,:,:,:]
            self.block_3d_slanted_polar(coord)

    def block_3d_slanted_cartesian(self, coord):
        """
        Block of tests
        """
        def sq(x): return numpy.sqrt(x)
        
        # id 1
        np_test.assert_almost_equal(coord[:,2,2,2], [0,0])
        np_test.assert_almost_equal(coord[:,1,2,2], [0,1])
        np_test.assert_almost_equal(coord[:,1,3,2], [0,2])
        np_test.assert_almost_equal(coord[:,3,2,2], [0,-1])
        np_test.assert_almost_equal(coord[:,3,1,2], [0,-2])
        np_test.assert_almost_equal(coord[:,2,2,3], [-1.,0])
        np_test.assert_almost_equal(coord[:,1,2,3], [-2*sq(2/3.), 2/sq(3.)])
        np_test.assert_almost_equal(coord[:,1,3,3], [-3/sq(3), 3*sq(2/3.)])
        np_test.assert_almost_equal(coord[:,2,2,1], [1,0])
        # id 2
        np_test.assert_almost_equal(coord[:,3,3,2], [0,0])
        np_test.assert_almost_equal(coord[:,2,3,2], [0,1])
        np_test.assert_almost_equal(coord[:,2,4,2], [0,2])
        np_test.assert_almost_equal(coord[:,4,3,2], [0,-1])
        np_test.assert_almost_equal(coord[:,4,2,2], [0,-2])
        np_test.assert_almost_equal(coord[:,3,3,3], [-1,0])
        np_test.assert_almost_equal(coord[:,2,3,3], [-2*sq(2/3.), 2/sq(3)])
        np_test.assert_almost_equal(coord[:,2,4,3], [-3/sq(3), 3*sq(2/3.)])
        np_test.assert_almost_equal(coord[:,3,3,1], [1,0])
        # id 3
        np_test.assert_almost_equal(coord[:,4,4,2], [0,0])
        np_test.assert_almost_equal(coord[:,3,4,2], [0,1])
        np_test.assert_almost_equal(coord[:,3,5,2], [0,2])
        np_test.assert_almost_equal(coord[:,5,4,2], [0,-1])
        np_test.assert_almost_equal(coord[:,5,3,2], [0,-2])
        np_test.assert_almost_equal(coord[:,4,4,3], [-1,0])
        np_test.assert_almost_equal(coord[:,3,4,3], [-2*sq(2/3.), 2/sq(3)])
        np_test.assert_almost_equal(coord[:,3,5,3], [-3/sq(3), 3*sq(2/3.)])
        np_test.assert_almost_equal(coord[:,4,4,1], [1,0])

    def block_3d_slanted_polar(self, coord):
        """
        Block of tests
        """
        def sq(x): return numpy.sqrt(x)

        # phi id 1
        np_test.assert_almost_equal(coord[1,2,2,2], 0)
        np_test.assert_almost_equal(coord[1,1,2,2], numpy.pi / 2)
        np_test.assert_almost_equal(coord[1,1,3,2], numpy.pi / 2)
        np_test.assert_almost_equal(coord[1,3,2,2], -numpy.pi / 2)
        np_test.assert_almost_equal(coord[1,3,1,2], -numpy.pi / 2)
        np_test.assert_almost_equal(numpy.abs(coord[1,2,2,3]), numpy.pi)
        np_test.assert_almost_equal(coord[1,1,2,3], numpy.arctan2(sq(2)/2,-1))
        np_test.assert_almost_equal(coord[1,1,3,3], numpy.arctan2(sq(2),-1))
        np_test.assert_almost_equal(coord[1,2,2,1], 0)
        # phi id 2
        try:
            np_test.assert_almost_equal(coord[1,3,3,2], 0)
        except AssertionError:
            # can be anything
            pass
        np_test.assert_almost_equal(coord[1,2,3,2], numpy.pi / 2)
        np_test.assert_almost_equal(coord[1,2,4,2], numpy.pi / 2)
        np_test.assert_almost_equal(coord[1,4,3,2], -numpy.pi / 2)
        np_test.assert_almost_equal(coord[1,4,2,2], -numpy.pi / 2)
        np_test.assert_almost_equal(numpy.abs(coord[1,3,3,3]), numpy.pi)
        np_test.assert_almost_equal(coord[1,2,3,3], numpy.arctan2(sq(2)/2,-1))
        np_test.assert_almost_equal(coord[1,2,4,3], numpy.arctan2(sq(2),-1))
        np_test.assert_almost_equal(coord[1,3,3,1], 0)
        # phi id 3
        try:
            np_test.assert_almost_equal(coord[1,4,4,2], 0)
        except AssertionError:
            # can be anything
            pass
        np_test.assert_almost_equal(coord[1,3,4,2], numpy.pi / 2)
        np_test.assert_almost_equal(coord[1,3,5,2], numpy.pi / 2)
        np_test.assert_almost_equal(coord[1,5,4,2], -numpy.pi / 2)
        np_test.assert_almost_equal(coord[1,5,3,2], -numpy.pi / 2)
        np_test.assert_almost_equal(numpy.abs(coord[1,4,4,3]), numpy.pi)
        np_test.assert_almost_equal(coord[1,3,4,3], numpy.arctan2(sq(2)/2,-1))
        np_test.assert_almost_equal(coord[1,3,5,3], numpy.arctan2(sq(2),-1))
        np_test.assert_almost_equal(coord[1,4,4,1], 0)

    def testMakeColumns(self):
        """
        Tests makeColumns()
        """

        rec_3d_cl = self.make3DRectangle()
        sla_3d_cl = self.make3DSlanted()

        # simple rectangle, radial
        col = rec_3d_cl.makeColumns(
            ids=[1], system='radial', normalize=False, startId=1, 
            metric='euclidean', connectivity=1, rimId=5, 
            bins=[0, 1, 2, 3, 4])
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0],
             [0, 4, 4, 4, 0, 0],
             [0, 3, 3, 3, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 2, 1, 2, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 3, 3, 3, 0, 0],
             [0, 4, 4, 4, 0, 0],
             [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(col.data[:,:,1], desired)

        # simple rectangle, radial, bin [0,2,4]
        col = rec_3d_cl.makeColumns(
            ids=[1], system='radial', normalize=False, startId=1, 
            metric='euclidean', connectivity=1, rimId=5, 
            bins=[0, 2, 4])
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(col.data[:,:,1], desired)

        # simple rectangle, radial, normalized
        col = rec_3d_cl.makeColumns(
            ids=[1], system='radial', normalize=True, startId=1, 
            metric='euclidean', connectivity=1, rimId=5, 
            bins=[0, 0.5, 1])
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 2, 1, 2, 0, 0],
             [0, 2, 1,  2, 0, 0],
             [0, 2, 1, 2, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(col.data[:,:,1], desired)

        # r binned
        col = sla_3d_cl.makeColumns(
            ids=[1,2,3], system='polar', normalize=False, startId=2, 
            metric='geodesic', connectivity=1, rimId=0, 
            bins=[[0,1,2,3,4,5], [-numpy.pi, numpy.pi]])
        desired_r1 = numpy.array([[0, 0, 0, 4, 0, 0],
                                  [0, 0,  2, 3,  4, 0],
                                  [0, 0,  1, 2,  3, 4],
                                  [0,  3,  2, 1,  2, 3],
                                  [0,  4,  3, 2,  1, 0],
                                  [0, 0,  4, 3,  2, 0],
                                  [0, 0, 0, 4, 0, 0]])
        np_test.assert_almost_equal(col.data[:,:,2], desired_r1) 
        np_test.assert_almost_equal(col.data[:,:,1], 
                                    numpy.where(desired_r1>0, desired_r1+1, 0)) 
        np_test.assert_almost_equal(col.data[:,:,3], 
                                    numpy.where(desired_r1>0, desired_r1+1, 0)) 

        # r binned, phi bin implicit
        col = sla_3d_cl.makeColumns(
            ids=[1,2,3], system='polar', normalize=False, startId=2, 
            metric='geodesic', connectivity=1, rimId=0, 
            bins=[[0,1,2,3,4,5]])
        desired_r1 = numpy.array([[0, 0, 0, 4, 0, 0],
                                  [0, 0,  2, 3,  4, 0],
                                  [0, 0,  1, 2,  3, 4],
                                  [0,  3,  2, 1,  2, 3],
                                  [0,  4,  3, 2,  1, 0],
                                  [0, 0,  4, 3,  2, 0],
                                  [0, 0, 0, 4, 0, 0]])
        np_test.assert_almost_equal(col.data[:,:,2], desired_r1) 
        np_test.assert_almost_equal(col.data[:,:,1], 
                                    numpy.where(desired_r1>0, desired_r1+1, 0)) 
        np_test.assert_almost_equal(col.data[:,:,3], 
                                    numpy.where(desired_r1>0, desired_r1+1, 0)) 

        # x binned
        col = sla_3d_cl.makeColumns(
            ids=[1,2,3], system='cartesian', normalize=False, startId=2, 
            metric='euclidean', connectivity=1, rimId=0, 
            bins=[[-1.5,-0.5,0.5,1.5]])
        desired_xy_2 = numpy.array([[0, 0, 0, 2, 0, 0],
                                    [0, 0,  2, 2,  2, 0],
                                    [0, 0,  2, 2,  2, 2],
                                    [0,  2,  2, 2,  2, 2],
                                    [0,  2,  2, 2,  2, 0],
                                    [0, 0,  2, 2,  2, 0],
                                    [0, 0, 0, 2, 0, 0]])
        np_test.assert_almost_equal(col.data[:,:,2], desired_xy_2) 
        np_test.assert_almost_equal(col.data[:,:,1], 
                                    numpy.where(desired_xy_2>0, 3, 0)) 
        np_test.assert_almost_equal(col.data[:,:,3], 
                                    numpy.where(desired_xy_2>0, 1, 0)) 

        # x binned, cartesian, geodesic
        col = sla_3d_cl.makeColumns(
            ids=[1,2,3], system='cartesian', normalize=False, startId=2, 
            metric='geodesic', connectivity=1, rimId=0, 
            bins=[[-2.5,-1.5,-0.5,0.5,1.5,2.5]])
        desired_xy_2 = numpy.array([[0, 0, 0, 3, 0, 0],
                                    [0, 0,  3, 3,  3, 0],
                                    [0, 0,  3, 3,  3, 3],
                                    [0,  3,  3, 3,  3, 3],
                                    [0,  3,  3, 3,  3, 0],
                                    [0, 0,  3, 3,  3, 0],
                                    [0, 0, 0, 3, 0, 0]])
        np_test.assert_almost_equal(col.data[:,:,2], desired_xy_2) 
        desired_xy_1 = numpy.array([[0, 0, 0, 5, 0, 0],
                                    [0, 0, 5, 5, 5, 0],
                                    [0, 0, 4, 5, 5, 5],
                                    [0, 5, 5, 4, 5, 5],
                                    [0, 5, 5, 5, 4, 0],
                                    [0, 0, 5, 5, 5, 0],
                                    [0, 0, 0, 5, 0, 0]])
        np_test.assert_almost_equal(col.data[:,:,1], desired_xy_1) 
        desired_xy_3 = numpy.array([[0, 0, 0, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 0],
                                    [0, 0, 2, 1, 1, 1],
                                    [0, 1, 1, 2, 1, 1],
                                    [0, 1, 1, 1, 2, 0],
                                    [0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 1, 0, 0]])
        np_test.assert_almost_equal(col.data[:,:,3], desired_xy_3) 

        # xy binned
        col = sla_3d_cl.makeColumns(
            ids=[1,2,3], system='cartesian', normalize=False, startId=2, 
            metric='euclidean', connectivity=1, rimId=0, 
            bins=[[-1.5,-0.5,0.5,1.5], [-3,-1,1,3]])
        desired_xy_2 = numpy.array([[0, 0, 0, 6, 0, 0],
                                    [0, 0, 5, 6, 6, 0],
                                    [0, 0, 5, 5, 6, 6],
                                    [0, 4, 5, 5, 5, 6],
                                    [0, 4, 4, 5, 5, 0],
                                    [0, 0, 4, 4, 5, 0],
                                    [0, 0, 0, 4, 0, 0]])
        try:
            np_test.assert_almost_equal(col.data[:,:,2], desired_xy_2) 
            np_test.assert_almost_equal(
                col.data[:,:,1],
                numpy.where(desired_xy_2>0, desired_xy_2+3, 0)) 
            np_test.assert_almost_equal(
                col.data[:,:,3],
                numpy.where(desired_xy_2>0, desired_xy_2-3, 0)) 
        except AssertionError:
            # global ambiguity in the direction scipy.linalg.svd()
            # interchange 4 and 6
            desired_xy_2[desired_xy_2==4] = 10
            desired_xy_2[desired_xy_2==6] = 4
            desired_xy_2[desired_xy_2==10] = 6
            np_test.assert_almost_equal(col.data[:,:,2], desired_xy_2) 
            np_test.assert_almost_equal(
                col.data[:,:,1],
                numpy.where(desired_xy_2>0, desired_xy_2+3, 0)) 
            np_test.assert_almost_equal(
                col.data[:,:,3],
                numpy.where(desired_xy_2>0, desired_xy_2-3, 0)) 

        # phi binned
        col = sla_3d_cl.makeColumns(
            ids=[1,2,3], system='polar', normalize=False, startId=2, 
            metric='geodesic', connectivity=1, rimId=0, 
            bins=[[0,5], numpy.array([-3/4., -1/4., 1/4., 3/4.]) * numpy.pi])
        # phi id 1
        try:
            np_test.assert_almost_equal(col.data[2,2,2], 2)
            np_test.assert_almost_equal(col.data[1,2,2], 3)
            np_test.assert_almost_equal(col.data[1,3,2], 3)
            np_test.assert_almost_equal(col.data[3,2,2], 1)
            np_test.assert_almost_equal(col.data[3,1,2], 1)
            np_test.assert_almost_equal(col.data[2,2,3], 0)
            np_test.assert_almost_equal(col.data[1,2,3], 0)
            np_test.assert_almost_equal(col.data[1,3,3], 3)
            np_test.assert_almost_equal(col.data[2,2,1], 2)
        except AssertionError:
            # best fit line direction ambiguity (scipy.linalg.svd())
            np_test.assert_almost_equal(col.data[2,2,2], 2)
            np_test.assert_almost_equal(col.data[1,2,2], 1)
            np_test.assert_almost_equal(col.data[1,3,2], 1)
            np_test.assert_almost_equal(col.data[3,2,2], 3)
            np_test.assert_almost_equal(col.data[3,1,2], 3)
            np_test.assert_almost_equal(col.data[2,2,3], 0)
            np_test.assert_almost_equal(col.data[1,2,3], 0)
            np_test.assert_almost_equal(col.data[1,3,3], 1)
            np_test.assert_almost_equal(col.data[2,2,1], 2)
        # phi id 2
        try:
            np_test.assert_almost_equal(col.data[3,3,2], 2)
        except AssertionError:
            # can be anything
            pass
        try:
            np_test.assert_almost_equal(col.data[2,3,2], 3)
            np_test.assert_almost_equal(col.data[2,4,2], 3)
            np_test.assert_almost_equal(col.data[4,3,2], 1)
            np_test.assert_almost_equal(col.data[4,2,2], 1)
            np_test.assert_almost_equal(col.data[3,3,3], 0)
            np_test.assert_almost_equal(col.data[2,3,3], 0)
            np_test.assert_almost_equal(col.data[2,4,3], 3)
            np_test.assert_almost_equal(col.data[3,3,1], 2)
        except AssertionError:
            # best fit line direction ambiguity (scipy.linalg.svd())
            np_test.assert_almost_equal(col.data[2,3,2], 1)
            np_test.assert_almost_equal(col.data[2,4,2], 1)
            np_test.assert_almost_equal(col.data[4,3,2], 3)
            np_test.assert_almost_equal(col.data[4,2,2], 3)
            np_test.assert_almost_equal(col.data[3,3,3], 0)
            np_test.assert_almost_equal(col.data[2,3,3], 0)
            np_test.assert_almost_equal(col.data[2,4,3], 1)
            np_test.assert_almost_equal(col.data[3,3,1], 2)
        # phi id 3
        try:
            np_test.assert_almost_equal(col.data[4,4,2], 2)
        except AssertionError:
            # can be anything
            pass
        try:
            np_test.assert_almost_equal(col.data[3,4,2], 3)
            np_test.assert_almost_equal(col.data[3,5,2], 3)
            np_test.assert_almost_equal(col.data[5,4,2], 1)
            np_test.assert_almost_equal(col.data[5,3,2], 1)
            np_test.assert_almost_equal(col.data[4,4,3], 0)
            np_test.assert_almost_equal(col.data[3,4,3], 0)
            np_test.assert_almost_equal(col.data[3,5,3], 3)
            np_test.assert_almost_equal(col.data[4,4,1], 2)
        except AssertionError:
            # best fit line direction ambiguity (scipy.linalg.svd())
            np_test.assert_almost_equal(col.data[3,4,2], 1)
            np_test.assert_almost_equal(col.data[3,5,2], 1)
            np_test.assert_almost_equal(col.data[5,4,2], 3)
            np_test.assert_almost_equal(col.data[5,3,2], 3)
            np_test.assert_almost_equal(col.data[4,4,3], 0)
            np_test.assert_almost_equal(col.data[3,4,3], 0)
            np_test.assert_almost_equal(col.data[3,5,3], 1)
            np_test.assert_almost_equal(col.data[4,4,1], 2)
        
        # normalize
        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCleft)
    unittest.TextTestRunner(verbosity=2).run(suite)
