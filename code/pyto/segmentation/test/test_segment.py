"""
Tests class Segment.

Only the layers related methods tested in the moment.

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

from pyto.segmentation.segment import Segment
from pyto.segmentation.test import common

class TestSegment(np_test.TestCase):
    """
    """

    def setUp(self):
        
        # to avoid problems when running multiple tests
        try:
            importlib.reload(common) 
        except AttributeError:
            pass # Python2
        
        # make image
        bound_data = numpy.zeros((15,15), dtype=int)
        bound_data[slice(2, 12), slice(3, 12)] = numpy.array(\
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
        self.bound_1 = Segment(bound_data)

    def testFindNonUnique(self):
        """
        Tests findNonUnique()
        """

        data = numpy.array(
            [[0, 2, 3, 4],
             [2, 3, 3, 4],
             [4, 5, 5, 7]])

        #  unique
        seg = Segment(data=data, ids=[3,5,7])
        nonu = seg.findNonUnique()
        np_test.assert_equal(nonu['many'], [])
        np_test.assert_equal(nonu['empty'], [])

        # non unique
        seg = Segment(data=data)
        seg.setIds(ids=[2,3,4,5,6,7,8])
        nonu = seg.findNonUnique()
        np_test.assert_equal(nonu['many'], [2, 4])
        np_test.assert_equal(nonu['empty'], [6, 8])

        # non-unique, some ids excluded
        seg = Segment(data=data)
        seg.setIds(ids=[3,4,5,6,7])
        nonu = seg.findNonUnique()
        np_test.assert_equal(nonu['many'], [4])
        np_test.assert_equal(nonu['empty'], [6])

    def testRemove(self):
        """
        Tests remove() with mode 'remove'
        """
        
        removed = self.bound_1.remove(data=self.bound_1.data[2:12, 3:12],
                                      ids=[2,3,5,9], value=4)
        desired = numpy.array(\
            [[7, 7, 7, 7, 7, 7, 7, 7, 7],
             [0, 1, 1, 1, 1, 4, 4, 4, 0],
             [0, 1, 1, 1, 1, 4, 4, 4, 0],
             [6, 6, 4, 4, 4, 4, 4, 4, 6],
             [6, 6, 4, 4, 4, 4, 4, 4, 6],
             [6, 6, 4, 4, 4, 4, 4, 4, 6],
             [6, 6, 4, 4, 4, 4, 4, 4, 6],
             [0, 4, 4, 4, 4, 4, 4, 4, 0],
             [4, 4, 4, 4, 4, 4, 4, 4, 4],
             [8, 8, 8, 8, 4, 4, 8, 8, 8]])
        np_test.assert_equal(removed, desired)

    def testLayersFrom(self):
        """
        Tests makeLayersFrom().
        """

        # no extra layers
        layers = \
            self.bound_1.makeLayersFrom(bound=[1,2], thick=1, nLayers=3, mask=5) 
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # thick = 2, no extra layers
        layers = \
            self.bound_1.makeLayersFrom(bound=[1,2], thick=2, nLayers=2, mask=5)

        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # thick = 2, w extra layers
        layers = \
            self.bound_1.makeLayersFrom(bound=[1,2], thick=2, nLayers=2, mask=5,
                                        nExtraLayers=1) 
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # thick = 2, w extra layers
        layers = \
            self.bound_1.makeLayersFrom(bound=[1,2], thick=3, nLayers=1,
                                        mask=[5, 6], nExtraLayers=1, extra=7) 
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        desired = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

    def testLayersBetweenLabels(self):
        """
        Tests different mask, boundary and extra regions assignments in
        makeLayersBetween(). Also checks size of the data array and offset.
        """

        # no extra layers
        layers, width = \
            self.bound_1.makeLayersBetween(bound_1=[1,2], bound_2=3, mask=5) 
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0],
             [0, 0, 4, 4, 4, 4, 4, 4, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(layers.data, desired[1:9, :])
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        np_test.assert_almost_equal(width, 4)
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # w extra layers on boundaries
        layers, width = self.bound_1.makeLayersBetween(
            bound_1=[1,2], bound_2=3, mask=5, nExtraLayers=3)
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
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
        np_test.assert_almost_equal(width, 4)
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # w extra layers on boundaries and extra regions
        layers, width = self.bound_1.makeLayersBetween(
            bound_1=[1,2], bound_2=3, mask=5, 
            nExtraLayers=3, extra_1=7, extra_2=[8,9]) 
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        desired = numpy.array(
            [[0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 2, 2, 2, 2, 2, 2, 2, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 0],
             [0, 0, 4, 4, 4, 4, 4, 4, 0],
             [0, 0, 5, 5, 5, 5, 5, 5, 0],
             [0, 0, 6, 6, 6, 6, 6, 6, 0],
             [0, 0, 7, 7, 7, 7, 7, 7, 0],
             [0, 8, 8, 8, 8, 8, 8, 8, 0],
             [10, 9, 9, 9, 9, 9, 9, 9, 9],
             [0, 10, 10, 10, 10, 10, 10, 10, 10]])
        np_test.assert_almost_equal(width, 4)
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # w multi id mask and extra layers on boundaries and extra regions
        layers, width = self.bound_1.makeLayersBetween(
            bound_1=numpy.array([1,2]), bound_2=3, mask=[5,6], 
            nExtraLayers=3, extra_1=7, extra_2=[8,9], maxDistance=5.5)
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
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

    def testLayersBetween(self):
        """
        Tests different nLayers and width arguments in makeLayersBetween().
        """

        # w nLayers, no extra layers
        layers, width = self.bound_1.makeLayersBetween(
            nLayers=2, bound_1=[1,2], bound_2=3, mask=5) 
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(width, 4)
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # w multi id mask and extra layers on boundaries and extra regions
        layers, width = self.bound_1.makeLayersBetween(
            nLayers=2, bound_1=numpy.array([1,2]), bound_2=3, mask=[5,6], 
            nExtraLayers=1, extra_1=7, extra_2=[8,9], maxDistance=5.5)
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 2, 2, 2, 2, 2, 2, 2, 0],
             [2, 2, 2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3, 3, 3, 3],
             [0, 3, 3, 3, 3, 3, 3, 3, 0],
             [0, 4, 4, 4, 4, 4, 4, 4, 0],
             [4, 4, 4, 4, 4, 4, 4, 4, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(width, 4)
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # w multi id mask and extra layers on boundaries and extra regions
        # and layers thinner than 1
        layers, width = self.bound_1.makeLayersBetween(
            nLayers=8, bound_1=numpy.array([1,2]), bound_2=3, mask=[5,6], 
            nExtraLayers=2, extra_1=7, extra_2=[8,9], maxDistance=5.5)
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        # possibly ambiguous layer assignment
        np_test.assert_equal(
            ((layers.data[4, 4:11] >= 1) & (layers.data[4, 4:11] <= 2)).all(),
            True)
        np_test.assert_equal(
            ((layers.data[9, 4:11] >= 11) & (layers.data[9, 4:11] <= 12)).all(),
            True)

        # w multi id mask and extra layers on boundaries and extra regions
        layers, width = self.bound_1.makeLayersBetween(
            nLayers=2, bound_1=numpy.array([1,2]), bound_2=3, mask=[5,6], 
            nExtraLayers=2, extra_1=7, extra_2=[8,9], maxDistance=5.5)
        layers.useInset(inset=self.bound_1.inset, mode='abs', expand=True)
        desired = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 2, 2, 2, 2, 2, 2, 2, 0],
             [0, 2, 2, 2, 2, 2, 2, 2, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 0],
             [3, 3, 3, 3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4, 4, 4, 4],
             [0, 4, 4, 4, 4, 4, 4, 4, 0],
             [0, 5, 5, 5, 5, 5, 5, 5, 0],
             [5, 5, 5, 5, 5, 5, 5, 5, 5],
             [6, 6, 6, 6, 6, 6, 6, 6, 6]])
        np_test.assert_almost_equal(width, 4)
        np_test.assert_equal(layers.data[2:12, 3:12], desired)

        # w maxDistance and no fill
        data = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1],
             [1, 1, 5, 5, 5, 1, 1],
             [5, 5, 5, 5, 5, 5, 5],
             [5, 5, 5, 5, 5, 5, 5],
             [3, 3, 5, 5, 5, 3, 3],
             [3, 3, 7, 7, 7, 7, 7]])
        seg = Segment(data)
        layers, width = seg.makeLayersBetween(
            nLayers=2, bound_1=numpy.array([1]), bound_2=[3, 7], mask=[5], 
            nExtraLayers=1, maxDistance=4, fill=False)
        desired = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 1, 1],
             [2, 2, 2, 0, 2, 2, 2],
             [3, 3, 3, 0, 3, 3, 3],
             [4, 4, 0, 0, 0, 4, 4],
             [0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(layers.data, desired)
        np_test.assert_almost_equal(width, 2.)

        # w maxDistance, with fill, mode median
        layers, width = seg.makeLayersBetween(
            nLayers=2, bound_1=numpy.array([1]), bound_2=[3, 7], mask=[5], 
            nExtraLayers=1, between='median', maxDistance=3, fill=True)
        desired = numpy.array(
            [[0, 1, 1, 1, 1, 1, 0],
             [1, 1, 2, 2, 2, 1, 1],
             [2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3, 3],
             [4, 4, 3, 3, 3, 4, 4],
             [0, 4, 4, 4, 4, 4, 0]])
        np_test.assert_equal(layers.data, desired)
        np_test.assert_almost_equal(width, 2.)

        # w maxDistance, with fill, mode mean
        layers, width = seg.makeLayersBetween(
            nLayers=2, bound_1=numpy.array([1]), bound_2=[3, 7], mask=[5], 
            nExtraLayers=1, between='mean', maxDistance=3, fill=True)
        desired = numpy.array(
            [[0, 1, 1, 1, 1, 1, 0],
             [1, 1, 2, 2, 2, 1, 1],
             [2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3, 3],
             [4, 4, 3, 3, 3, 4, 4],
             [0, 4, 4, 4, 4, 4, 0]])
        np_test.assert_equal(layers.data, desired)
        np_test.assert_almost_equal(
            width, 
            ((4 * 3 + 2 * numpy.sqrt(17) + numpy.sqrt(20)) / 7. - 1))

        # funny shaped mask
        segments_data = numpy.array([[3, 3, 3, 3, 3, 0],
                                     [3, 3, 3, 3, 0, 0],
                                     [3, 3, 3, 0, 0, 0],
                                     [3, 3, 0, 0, 0, 0],
                                     [3, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [2, 2, 2, 2, 2, 2]])
        mask_data = numpy.array([[0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 5, 5, 5],
                                 [0, 0, 0, 0, 0, 5],
                                 [0, 0, 0, 0, 0, 5],
                                 [0, 0, 0, 5, 5, 5],
                                 [0, 0, 0, 0, 0, 0]])
        seg = Segment(data=segments_data)
        layers, width = seg.makeLayersBetween(
            bound_1=3, bound_2=2, mask=mask_data, between='min')
        desired_layers = numpy.array([[0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 1],
                                      [0, 0, 0, 0, 0, 2],
                                      [0, 0, 0, 0, 0, 3],
                                      [0, 0, 0, 3, 3, 3],
                                      [0, 0, 0, 0, 0, 0]])
        np_test.assert_almost_equal(width, numpy.sqrt(17) - 1)
        np_test.assert_equal(layers.data, desired_layers)

    def makeHorizontalLayers(self, item):

        if item == 1:

            data = numpy.array(
                [[0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0],
                 [0, 2, 2, 2, 2, 0],
                 [0, 3, 3, 3, 3, 0], 
                 [0, 4, 4, 4, 4, 0],
                 [0, 5, 5, 5, 5, 0]])
            segments = Segment(data)
            region = numpy.zeros(shape=data.shape)
            region[:,0] = 5
            return segments, region
    
        elif item == 2:

            data = numpy.array(
                [[0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0],
                 [0, 2, 2, 2, 0, 0],
                 [0, 0, 3, 3, 3, 0], 
                 [0, 0, 4, 4, 4, 0],
                 [0, 5, 5, 5, 5, 0]])
            layers = Segment(data)
            return layers

    def makeSlantedLayers(self, item):
        
        if item == 1:

            data = numpy.array(
                [[0, 0, 0, 0, 1, 2],
                 [0, 0, 0, 1, 2, 2],
                 [0, 0, 1, 2, 2, 3],
                 [0, 1, 2, 2, 3, 3],
                 [5, 2, 2, 3, 3, 0],
                 [5, 2, 3, 3, 0, 0],
                 [5, 5, 5, 3, 0, 0]])
            segments = Segment(data)
            return segments

        elif item == 2:

            data = numpy.array(
                [[5, 5, 5, 1, 0, 0],
                 [5, 5, 1, 1, 2, 0],
                 [5, 5, 1, 2, 2, 3],
                 [5, 1, 1, 2, 3, 3],
                 [0, 1, 2, 2, 3, 6],
                 [0, 0, 2, 3, 3, 6],
                 [0, 0, 0, 3, 6, 6]])
            layers = Segment(data)
            return layers

    def testElementDistanceToRegion(self):
        """
        Tests elementDistanceToRegion
        """

        # make layers
        hor_seg, hor_reg = self.makeHorizontalLayers(1)
        sl_seg = self.makeSlantedLayers(1)
        sl_reg = sl_seg.data == 5

        # horizontal, euclidean
        dist = hor_seg.elementDistanceToRegion(ids=[2,3,4,5], region=hor_reg)
        desired = numpy.array(
            [[-1, -1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1, -1],
             [-1,  1.,  2,  3,  4, -1],
             [-1,  1,  2,  3,  4, -1],
             [-1,  1,  2,  3,  4, -1],
             [-1,  1,  2,  3,  4, -1]])
        np_test.assert_almost_equal(dist, desired)

        # slanted, euclidean
        dist = sl_seg.elementDistanceToRegion(ids=[1,2,3], region=sl_reg)
        def sq(x): return numpy.sqrt(x)
        desired = numpy.array(
            [[-1,  -1,  -1,  -1,    sq(32), sq(41)],
             [-1,  -1,  -1, sq(18), sq(25), sq(34)],
             [-1,  -1, sq(8), sq(13), sq(20), sq(25)],
             [-1, sq(2), sq(5), sq(10), sq(13), sq(18)],
             [-1,   1,   2, sq(5), sq(8), -1],
             [-1,   1,   1, sq(2), -1,  -1],
             [-1,  -1,  -1,   1,   -1,  -1]])
        np_test.assert_almost_equal(dist, desired)

        # slanted, geodesic conn 1
        seg = self.makeSlantedLayers(1)
        dist = seg.elementDistanceToRegion(
            ids=[2,3], region=(seg.data==5), metric='geodesic', connectivity=1)
        desired = numpy.array(
            [[-1, -1,  -1,  -1, -1, 9],
             [-1, -1,  -1,  -1,  7, 8],
             [-1, -1,   -1,  5,  6, 7],
             [-1,  -1,   3,  4,  5, 6],
             [-1,  1,   2,   3,  4, -1],
             [-1,  1,   1,   2, -1, -1],
             [-1, -1,  -1,   1, -1, -1]])
        np_test.assert_equal(dist, desired)

        # slanted, geodesic conn 2
        seg = self.makeSlantedLayers(1)
        dist = seg.elementDistanceToRegion(
            ids=[1,2,3], region=(seg.data==5), metric='geodesic', 
            connectivity=2)
        desired = numpy.array(
            [[-1, -1,  -1,  -1, 4, 5],
             [-1, -1,  -1,  3,  4, 5],
             [-1, -1,   2,  3,  4, 4],
             [-1,  1,   2,  3,  3, 3],
             [-1,  1,   2,   2,  2, -1],
             [-1,  1,   1,   1, -1, -1],
             [-1, -1,  -1,   1, -1, -1]])
        np_test.assert_equal(dist, desired)

        # slanted, euclidean-geodesic
        seg = self.makeSlantedLayers(1)
        dist = seg.elementDistanceToRegion(
            ids=[1,2,3], region=(seg.data==5), metric='euclidean-geodesic')
        sq2 = numpy.sqrt(2)
        desired = numpy.array(
            [[-1, -1,  -1,    -1,     4*sq2, 1+4*sq2],
             [-1, -1,  -1,    3*sq2,  1+3*sq2, 2+3*sq2],
             [-1, -1,  2*sq2, 1+2*sq2, 2+2*sq2, 1+3*sq2],
             [-1, sq2, 1+sq2, 2+sq2, 1+2*sq2, 3*sq2],
             [-1,  1,   2,    1+sq2,     2*sq2, -1],
             [-1,  1,   1,     sq2,       -1,  -1],
             [-1, -1,  -1,       1,       -1,  -1]])
        np_test.assert_almost_equal(dist, desired)

    def testElementDistanceToRim(self):
        """
        Tests elementDistanceToRim()
        """

        hor_cl = self.makeHorizontalLayers(2)
        sla_cl = self.makeSlantedLayers(2)
        def sq(x): return numpy.sqrt(x)

        # horizontal, euclidean, out
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        desired = numpy.array(
            [[-1, -1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1, -1],
             [-1,  1, numpy.sqrt(2),  1,  -1, -1],
             [-1,  -1,  1, numpy.sqrt(2),   1, -1],
             [-1,  -1,  1,  2,   1, -1],
             [-1, -1, -1, -1, -1, -1]])
        np_test.assert_almost_equal(dist, desired)

        # horizontal, euclidean, in
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='in', rimConnectivity=1)
        desired = numpy.array(
            [[-1, -1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1, -1],
             [-1,  0,   1,  0,  -1, -1],
             [-1,  -1,  0,  1,   0, -1],
             [-1,  -1,  0,  1,   0, -1],
             [-1, -1, -1, -1, -1, -1]])
        np_test.assert_almost_equal(dist, desired)

        # horizontal, euclidean, in, conn -1
        dist = hor_cl.elementDistanceToRim(
            ids=[2,3,4], metric='euclidean', rimId=0, 
            rimLocation='in', rimConnectivity=-1)
        desired = numpy.array(
            [[-1, -1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1, -1],
             [-1,  0,   0,  0,  -1, -1],
             [-1,  -1,  0,  0,   0, -1],
             [-1,  -1,  0,  1,   0, -1],
             [-1, -1, -1, -1, -1, -1]])
        np_test.assert_almost_equal(dist, desired)

        # slanted, euclidean, out
        dist = sla_cl.elementDistanceToRim(
            ids=[1,2,3], metric='euclidean', rimId=0, 
            rimLocation='out', rimConnectivity=1)
        desired = numpy.array(
            [[-1, -1,   -1,   1, -1, -1],
             [-1, -1, sq(5), sq(2), 1, -1],
             [-1, -1, sq(8), sq(5), sq(2), 1],
             [-1, sq(2), sq(5), sq(8), sq(5), 2],
             [-1,  1, sq(2), sq(5), sq(8), -1],
             [-1, -1,    1,  sq(2), sq(5), -1],
             [-1, -1,   -1,    1,   -1,    -1]])
        np_test.assert_almost_equal(dist, desired)

        # slanted, geodesic, conn 1, out
        dist = sla_cl.elementDistanceToRim(
            ids=[1,2,3], metric='geodesic', connectivity=1, rimId=0, 
            rimLocation='out', rimConnectivity=1)
        desired = numpy.array(
            [[-1, -1,   -1,   1, -1, -1],
             [-1, -1,   3, 2, 1, -1],
             [-1, -1, 4,   3, 2, 1],
             [-1, 2,  3, 4, 3,  2],
             [-1,  1, 2, 3,  4, -1],
             [-1, -1,    1,  2, 3, -1],
             [-1, -1,   -1,    1,   -1,    -1]])
        np_test.assert_equal(dist, desired)

        # slanted, geodesic, conn 1, in
        dist = sla_cl.elementDistanceToRim(
            ids=[1,2,3], metric='geodesic', connectivity=1, rimId=0, 
            rimLocation='in', rimConnectivity=1)
        desired = numpy.array(
            [[-1, -1,   -1,   0, -1, -1],
             [-1, -1,   2,    1,  0, -1],
             [-1, -1,   3,    2,  1,  0],
             [-1,  1,    2,   3,  2,  1],
             [-1,  0,    1,   2,  3, -1],
             [-1, -1,    0,   1,  2, -1],
             [-1, -1,   -1,   0,   -1,    -1]])
        np_test.assert_equal(dist, desired)

        # slanted, geodesic, conn 2, in
        dist = sla_cl.elementDistanceToRim(
            ids=[1,2,3], metric='geodesic', connectivity=2, rimId=0, 
            rimLocation='in', rimConnectivity=1)
        desired = numpy.array(
            [[-1, -1,   -1,   0, -1, -1],
             [-1, -1,   1,    1,  0, -1],
             [-1, -1,   2,    1,  1,  0],
             [-1,  1,   1,    2,  1,  1],
             [-1,  0,    1,   1,  2, -1],
             [-1, -1,    0,   1,  1, -1],
             [-1, -1,   -1,   0,   -1,    -1]])
        np_test.assert_equal(dist, desired)

        # slanted, euclidean-geodesic, in
        dist = sla_cl.elementDistanceToRim(
            ids=[1,2,3], metric='euclidean-geodesic', rimId=0, 
            rimLocation='in', rimConnectivity=1)
        desired = numpy.array(
            [[-1, -1,   -1,     0,  -1, -1],
             [-1, -1, sq(2),    1,   0, -1],
             [-1, -1, 1+sq(2), sq(2),  1,  0],
             [-1,  1,  sq(2), 1+sq(2), sq(2),  1],
             [-1,  0,    1,  sq(2),  1+sq(2), -1],
             [-1, -1,    0,   1,     sq(2), -1],
             [-1, -1,   -1,   0,   -1,    -1]])
        np_test.assert_equal(dist, desired)

    def testDistanceFromOrigin(self):
        """
        Tests distanceFromOrigin()
        """

        hor_cl = self.makeHorizontalLayers(2)
        origins = {2 : [2, 2], 3 : [3, 3], 4 : [4, 3]}

        # horizontal, euclidean
        dist = hor_cl.distanceFromOrigin(origins=origins, metric='euclidean')
        desired = numpy.array(
            [[-1, -1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1, -1],
             [-1,  1,  0,  1,  -1, -1],
             [-1,  -1,  1, 0,   1, -1],
             [-1,  -1,  1,  0,   1, -1],
             [-1, -1, -1, -1, -1, -1]])    
        np_test.assert_almost_equal(dist, desired) 

        sla_cl = self.makeSlantedLayers(2)
        origins = {1 : [2, 2], 2 : [3, 3], 3 : [4, 4]}
        def sq(x): return numpy.sqrt(x)

        # slanted, euclidean
        dist = sla_cl.distanceFromOrigin(origins=origins, metric='euclidean')
        desired = numpy.array(
            [[-1, -1,   -1,   sq(5), -1, -1],
             [-1, -1,    1,   sq(2), sq(5), -1],
             [-1, -1,    0,    1, sq(2), sq(5)],
             [-1, sq(2), 1,    0,   1, sq(2)],
             [-1, sq(5), sq(2), 1,  0, -1],
             [-1, -1,    sq(5),  sq(2), 1, -1],
             [-1, -1, -1, sq(5), -1, -1]])
        np_test.assert_almost_equal(dist, desired) 

        # slanted, geodesic conn 1
        dist = sla_cl.distanceFromOrigin(origins=origins, metric='geodesic',
                                         connectivity=1)
        desired = numpy.array(
            [[-1, -1,   -1,   3, -1, -1],
             [-1, -1,    1,   2, 3, -1],
             [-1, -1,    0,   1, 2, 3],
             [-1,  2, 1,    0,   1, 2],
             [-1,  3, 2, 1,  0, -1],
             [-1, -1,    3,  2, 1, -1],
             [-1, -1, -1, 3, -1, -1]])
        np_test.assert_equal(dist, desired) 

        # slanted, geodesic conn 2
        dist = sla_cl.distanceFromOrigin(origins=origins, metric='geodesic',
                                         connectivity=2)
        desired = numpy.array(
            [[-1, -1, -1,  2, -1, -1],
             [-1, -1,  1,  1, 2, -1],
             [-1, -1,  0,  1, 1, 2],
             [-1,  1,  1,  0,  1, 1],
             [-1,  2,  1,  1,  0, -1],
             [-1, -1,  2,  1,  1, -1],
             [-1, -1, -1,  2, -1, -1]])
        np_test.assert_equal(dist, desired) 

        # slanted, euclidean-geodesic
        dist = sla_cl.distanceFromOrigin(origins=origins, 
                                         metric='euclidean-geodesic')
        desired = numpy.array(
            [[-1, -1,   -1,   1+sq(2), -1, -1],
             [-1, -1,    1,   sq(2), 1+sq(2), -1],
             [-1, -1,    0,    1, sq(2), 1+sq(2)],
             [-1, sq(2), 1,    0,   1, sq(2)],
             [-1, 1+sq(2), sq(2), 1,  0, -1],
             [-1, -1,    1+sq(2),  sq(2), 1, -1],
             [-1, -1, -1, 1+sq(2), -1, -1]])
        np_test.assert_almost_equal(dist, desired) 

    def testGetStructureFootprint(self):
        """
        Tests getStructureFootprint()
        """

        # 2d conn 1
        seg = Segment(numpy.zeros(shape=(3,4)))
        se, foot = seg.getStructureFootprint(connectivity=1)
        desired = scipy.ndimage.generate_binary_structure(2,1)
        np_test.assert_equal(foot, desired)
        desired = desired.astype(int)
        desired[1,1] = 0
        np_test.assert_equal(se, desired)

        # 3d conn 2
        seg = Segment(numpy.zeros(shape=(3,4,5)))
        se, foot = seg.getStructureFootprint(connectivity=2)
        desired = scipy.ndimage.generate_binary_structure(3,2)
        np_test.assert_equal(foot, desired)
        desired = desired.astype(int)
        desired[1,1,1] = 0
        np_test.assert_equal(se, desired)

        # 3d conn -1 (3)
        seg = Segment(numpy.zeros(shape=(3,4,5)))
        se, foot = seg.getStructureFootprint(connectivity=-1)
        desired = scipy.ndimage.generate_binary_structure(3,3)
        np_test.assert_equal(foot, desired)
        desired = desired.astype(int)
        desired[1,1,1] = 0
        np_test.assert_equal(se, desired)

        # 2d conn [2]
        seg = Segment(numpy.zeros(shape=(3,4)))
        se, foot = seg.getStructureFootprint(connectivity=[2])
        desired = scipy.ndimage.generate_binary_structure(2,2)
        np_test.assert_equal(foot, desired)
        desired = desired.astype(int) * numpy.sqrt(2)
        desired[1,1] = 0
        np_test.assert_almost_equal(se, desired)

        # 3d conn [1,2,3]
        seg = Segment(numpy.zeros(shape=(3,4,5)))
        se, foot = seg.getStructureFootprint(connectivity=[3,1,2])
        desired = scipy.ndimage.generate_binary_structure(3,3)
        np_test.assert_equal(foot, desired)
        desired = numpy.array(
            [[[3.,2,3], [2,1,2], [3,2,3]],
             [[2,1,2], [1,0,1], [2,1,2]],
             [[3,2,3], [2,1,2], [3,2,3]]])
        np_test.assert_almost_equal(se**2, desired)

    def testElementGeodesicDistanceToRegion(self):
        """
        Tests elementGeodesicDistanceToRegion()
        """

        # connectivity 1, horizontal
        seg, reg = self.makeHorizontalLayers(1)
        dist = seg.elementGeodesicDistanceToRegion(ids=[1,2,4,5], 
                                                   region=reg>0, connectivity=1)
        desired = numpy.array(
                [[-1., -1, -1, -1, -1, -1],
                 [-1, 1, 2, 3, 4, -1],
                 [-1, 1, 2, 3, 4, -1],
                 [-1, -1, -1, -1, -1, -1],
                 [-1, 1, 2, 3, 4, -1],
                 [-1, 1, 2, 3, 4, -1]])
        np_test.assert_equal(dist, desired)

        # connectivity 1, slanted
        seg = self.makeSlantedLayers(2)
        reg = seg.data==0
        reg[1,:] = True
        dist = seg.elementGeodesicDistanceToRegion(
            ids=[1,2,3], region=reg, connectivity=1)
        desired = numpy.array(
            [[-1, -1,   -1,   1, -1, -1],
             [-1, -1,    0,   0, 0, -1],
             [-1, -1,    1,   1, 1, 1],
             [-1,  2,    2,   2, 3, 2],
             [-1,  1,    2,   3, 4, -1],
             [-1, -1,    1,   2, 3, -1],
             [-1, -1,   -1,   1,   -1,    -1]])
        np_test.assert_equal(dist, desired)

        # connectivity 1, slanted
        seg = self.makeSlantedLayers(1)
        dist = seg.elementGeodesicDistanceToRegion(
            ids=[2,3], region=(seg.data==5), connectivity=1)
        desired = numpy.array(
            [[-1, -1,  -1,  -1, -1, 9],
             [-1, -1,  -1,  -1,  7, 8],
             [-1, -1,   -1,  5,  6, 7],
             [-1,  -1,   3,  4,  5, 6],
             [-1,  1,   2,   3,  4, -1],
             [-1,  1,   1,   2, -1, -1],
             [-1, -1,  -1,   1, -1, -1]])
        np_test.assert_equal(dist, desired)

        # connectivity 2, slanted
        seg = self.makeSlantedLayers(1)
        dist = seg.elementGeodesicDistanceToRegion(
            ids=[1,2,3], region=(seg.data==5), connectivity=2)
        desired = numpy.array(
            [[-1, -1,  -1,  -1, 4, 5],
             [-1, -1,  -1,  3,  4, 5],
             [-1, -1,   2,  3,  4, 4],
             [-1,  1,   2,  3,  3, 3],
             [-1,  1,   2,   2,  2, -1],
             [-1,  1,   1,   1, -1, -1],
             [-1, -1,  -1,   1, -1, -1]])
        np_test.assert_equal(dist, desired)

        # connectivity [1,2], slanted
        seg = self.makeSlantedLayers(1)
        dist = seg.elementGeodesicDistanceToRegion(
            ids=[1,2,3], region=(seg.data==5), connectivity=[1,2])
        sq2 = numpy.sqrt(2)
        desired = numpy.array(
            [[-1, -1,  -1,    -1,     4*sq2, 1+4*sq2],
             [-1, -1,  -1,    3*sq2,  1+3*sq2, 2+3*sq2],
             [-1, -1,  2*sq2, 1+2*sq2, 2+2*sq2, 1+3*sq2],
             [-1, sq2, 1+sq2, 2+sq2, 1+2*sq2, 3*sq2],
             [-1,  1,   2,    1+sq2,     2*sq2, -1],
             [-1,  1,   1,     sq2,       -1,  -1],
             [-1, -1,  -1,       1,       -1,  -1]])
        np_test.assert_almost_equal(dist, desired)

    def testMakeFree(self):
        """
        Tests makeFree()
        """
        
        # int mask
        free = common.bound_1.makeFree(ids=[3,4], size=0, mask=5, update=False)
        np_test.assert_equal(free.data.shape, (10,10))
        np_test.assert_equal(
            free.data, numpy.where(common.bound_1.data==5, 1, 0))

        # ndarray mask
        mask = numpy.where(common.bound_1.data==5, 2, 0)
        free = common.bound_1.makeFree(
            ids=[3,4], size=0, mask=mask, update=False)
        np_test.assert_equal(free.data.shape, (10,10))
        np_test.assert_equal(
            free.data, numpy.where(common.bound_1.data==5, 1, 0))

        # Segment mask
        mask = Segment(data=numpy.where(common.bound_1.data==5, 3, 0))
        free = common.bound_1.makeFree(
            ids=[3,4], size=0, mask=mask, update=False)
        np_test.assert_equal(free.data.shape, (10,10))
        np_test.assert_equal(
            free.data, numpy.where(common.bound_1.data==5, 1, 0))

        # boundary inset, Segment mask full size
        mask = Segment(data=numpy.where(common.bound_1.data==5, 3, 0))
        free = common.bound_1in.makeFree(
            ids=[3,4], size=0, mask=mask, update=False)
        np_test.assert_equal(free.data.shape, (6,8))
        np_test.assert_equal(
            free.data, numpy.where(common.bound_1in.data==5, 1, 0))
        np_test.assert_equal(free.inset, [slice(1,7), slice(1,9)])

        # boundary inset, mask Segment same inset
        mask = Segment(data=numpy.where(common.bound_1.data==5, 3, 0))
        mask.useInset(inset=[slice(1,7), slice(1,9)])
        free = common.bound_1in.makeFree(
            ids=[3,4], size=0, mask=mask, update=False)
        np_test.assert_equal(free.data.shape, (6,8))
        np_test.assert_equal(
            free.data, numpy.where(common.bound_1in.data==5, 1, 0))
        np_test.assert_equal(free.inset, [slice(1,7), slice(1,9)])

        # boundary full size, mask Segment inset that includes boundary
        mask = Segment(data=numpy.where(common.bound_1.data==5, 3, 0))
        mask.useInset(inset=[slice(1,7), slice(1,9)])
        free = common.bound_1.makeFree(
            ids=[3,4], size=0, mask=mask, update=False)
        np_test.assert_equal(free.data.shape, (10,10))
        np_test.assert_equal(
            free.data[1:7,1:9], numpy.where(common.bound_1in.data==5, 1, 0))
        np_test.assert_equal(free.inset, [slice(0,10), slice(0,10)])

        # boundary full size, mask Segment inset that doesn't include boundary
        mask = Segment(data=numpy.where(common.bound_1.data==5, 3, 0))
        mask.useInset(inset=[slice(2,6), slice(1,9)])
        free = common.bound_1.makeFree(
            ids=[3,4], size=0, mask=mask, update=False)
        np_test.assert_equal(free.data.shape, (10,10))
        np_test.assert_equal(
            free.data[1:7,1:9], numpy.where(common.bound_1in.data==5, 1, 0))
        np_test.assert_equal(free.inset, [slice(0,10), slice(0,10)])

        # boundary inset, mask smaller inset that doesn't include boundary
        mask = Segment(data=numpy.where(common.bound_1.data==5, 3, 0))
        mask.useInset(inset=[slice(2,6), slice(1,9)])
        free = common.bound_1in.makeFree(
            ids=[3,4], size=0, mask=mask, update=False)
        np_test.assert_equal(free.data.shape, (6,8))
        np_test.assert_equal(
            free.data, numpy.where(common.bound_1in.data==5, 1, 0))
        np_test.assert_equal(free.inset, [slice(1,7), slice(1,9)])

        # boundary smaller inset than mask
        mask = Segment(data=numpy.where(common.bound_1.data==5, 3, 0))
        mask.useInset(inset=[slice(2,6), slice(1,9)])
        common.bound_1in.useInset([slice(1,7), slice(2,7)], mode='absolute')
        free = common.bound_1in.makeFree(
            ids=[3,4], size=0, mask=mask, update=False)
        np_test.assert_equal(free.data.shape, (6,5))
        np_test.assert_equal(
            free.data, numpy.where(common.bound_1.data==5, 1, 0)[1:7,2:7])
        np_test.assert_equal(free.inset, [slice(1,7), slice(2,7)])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSegment)
    unittest.TextTestRunner(verbosity=2).run(suite)
