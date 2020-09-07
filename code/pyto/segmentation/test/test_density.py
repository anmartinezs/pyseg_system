"""

Tests module denisty

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

from copy import copy, deepcopy
import importlib
import unittest
#from numpy.testing import *

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.density import Density
from pyto.segmentation.segment import Segment
import pyto.segmentation.test.common as common

class TestDensity(np_test.TestCase):
    """
    """

    def setUp(self):
        importlib.reload(common) # to avoid problems when running multiple tests
        self.grey = common.make_grey()
        self.segments = common.make_shapes()
        self.density = Density()
        self.density.calculate(image=self.grey, segments=self.segments)

    def testCalculate(self):
        """
        Tests calculate()
        """

        ####################################################
        #
        # Case 1
        #

        density = Density()
        density.calculate(image=self.grey, segments=self.segments)
        ids = numpy.concatenate([[0], density.ids])

        desired = [48.52, 22., 27., 84.92, 79.57]
        np_test.assert_almost_equal(density.mean[ids], desired, decimal=2)

        desired = [30.28, 8.20, 12.79, 8.22, 11.22]
        np_test.assert_almost_equal(density.std[ids], desired, decimal=2)

        desired = [6, 11.,   6.,  72.,   65.]
        np_test.assert_equal(density.min[ids], desired)

        desired = [98, 33.,   48.,  96.,   98.]
        np_test.assert_equal(density.max[ids], desired)

        desired = [50, 9,  21, 13,  7]
        np_test.assert_equal(density.volume[ids], desired)

        ####################################################
        #
        # Case 2 without positioning
        #

        density = Density()
        density.calculate(image=common.image_1, segments=common.bound_1)
        ids = numpy.concatenate([[0], density.ids])

        desired = [6.5, 9., 9, 5.25]
        np_test.assert_almost_equal(density.mean[ids], desired)

        desired = [1, 9,   9.,  1]
        np_test.assert_equal(density.min[ids], desired)

        desired = [9, 9,   9.,  9]
        np_test.assert_equal(density.max[ids], desired)

        desired = [48, 8,  8, 32]
        np_test.assert_equal(density.volume[ids], desired)

        ####################################################
        #
        # Case 2 with positioning
        #

        density = Density()
        density.calculate(image=common.image_1, segments=common.bound_1in)
        ids = numpy.concatenate([[0], density.ids])

        desired = [6.5, 9., 9, 5.25]
        np_test.assert_almost_equal(density.mean[ids], desired)

        desired = [1, 9,   9.,  1]
        np_test.assert_equal(density.min[ids], desired)

        desired = [9, 9,   9.,  9]
        np_test.assert_equal(density.max[ids], desired)

        desired = [48, 8,  8, 32]
        np_test.assert_equal(density.volume[ids], desired)

    def testAggregate(self):
        """
        Tests aggregate()
        """

        # calculate
        density = Density()
        density.calculate(image=self.grey, segments=self.segments)
        new = density.aggregate(ids=[[1, 3, 6], [4]])

        # make aggregated segments
        desired_seg_data = self.segments.data.copy()
        desired_seg_data[desired_seg_data==3] = 1
        desired_seg_data[desired_seg_data==6] = 1
        desired_seg = Segment(desired_seg_data)
        desired = Density()
        desired.calculate(image=self.grey, segments=desired_seg)

        # test
        new_ids = numpy.concatenate([[0], new.ids])
        desired_ids = numpy.concatenate([[0], desired.ids])
        np_test.assert_almost_equal(
            new.mean[new_ids], desired.mean[desired_ids])
        np_test.assert_almost_equal(new.std[new_ids], desired.std[desired_ids])
        np_test.assert_equal(new.min[new_ids], desired.min[desired_ids])
        np_test.assert_equal(new.max[new_ids], desired.max[desired_ids])
        np_test.assert_equal(new.volume[new_ids], desired.volume[desired_ids])
        

    def testRestrict(self):
        """
        Tests restrict
        """

        # make and restrict density object
        density = Density()
        density.calculate(image=self.grey, segments=self.segments)
        ids = [1, 4]
        density.restrict(ids=ids)

        # make another object
        dens_2 = Density()
        dens_2.calculate(image=self.grey, segments=self.segments, ids=ids)

        # test
        ids0 = [0] + ids
        np_test.assert_almost_equal(density.mean[ids0], dens_2.mean[ids0])
        np_test.assert_almost_equal(density.std[ids0], dens_2.std[ids0])
        np_test.assert_almost_equal(density.min[ids0], dens_2.min[ids0])
        np_test.assert_almost_equal(density.max[ids0], dens_2.max[ids0])
        np_test.assert_almost_equal(density.volume[ids0], dens_2.volume[ids0])

    def testCalculateTotal(self):
        """
        Tests calculateTotal()
        """
        
        # define data
        this = Density()
        this.setIds(ids=[2, 4, 5])
        data2 = numpy.array([1,2,3])
        data4 = numpy.array([4, 5])
        data5 = numpy.array([6, 7, 8, 9])
        data = numpy.hstack((data2, data4, data5))

        # make data structures
        this.min = numpy.array(
            [-1, -1, data2.min(), -1, data4.min(), data5.min()])
        this.max = numpy.array(
            [-1, -1, data2.max(), -1, data4.max(), data5.max()])
        this.volume = numpy.array(
            [-1, -1, len(data2), -1, len(data4), len(data5)])
        this.mean = numpy.array(
            [-1, -1, data2.mean(), -1, data4.mean(), data5.mean()])
        this.std = numpy.array(
            [-1, -1, data2.std(), -1, data4.std(), data5.std()])

        # check
        this.calculateTotal()
        np_test.assert_equal(this.min[0], data.min())
        np_test.assert_equal(this.max[0], data.max())
        np_test.assert_equal(this.volume[0], len(data))
        np_test.assert_almost_equal(this.mean[0], data.mean())
        np_test.assert_almost_equal(this.std[0], data.std())

    def testMerge(self):
        """
        Tests merge
        """

        # simple, mode replace, mode0 consistent
        this = Density()
        this.setIds(ids=[2])
        data2 = numpy.array([1,2,3])
        this.mean = numpy.array([-1, -1, data2.mean()])
        this.std = numpy.array([-1, -1, data2.std()])
        this.min = numpy.array([-1, -1, data2.min()])
        this.max = numpy.array([-1, -1, data2.max()])
        this.volume = numpy.array([-1, -1, len(data2)])
        other = Density()
        other.setIds(ids=[3])
        data3 = numpy.array([5,6,7,8])
        other.mean = numpy.array([-1, -1, -1, data3.mean()])
        other.std = numpy.array([-1, -1, -1, data3.std()])
        other.min = numpy.array([-1, -1, -1, data3.min()])
        other.max = numpy.array([-1, -1, -1, data3.max()])
        other.volume = numpy.array([-1, -1, -1, len(data3)])
        data = numpy.hstack((data2, data3))

        this.merge(new=other, mode='replace', mode0='consistent')
        np_test.assert_equal(this.ids, [2, 3])
        np_test.assert_equal(
            this.min, [data.min(), -1, data2.min(), data3.min()])
        np_test.assert_equal(
            this.max, [data.max(), -1, data2.max(), data3.max()])
        np_test.assert_equal(
            this.volume, [len(data), -1, len(data2), len(data3)])
        np_test.assert_almost_equal(
            this.mean, [data.mean(), -1, data2.mean(), data3.mean()])
        np_test.assert_almost_equal(
            this.std, [data.std(), -1, data2.std(), data3.std()], decimal=5)

        # mode replace, mode0 consistent
        this = Density()
        this.setIds(ids=[1,4,5])
        this.mean = numpy.array([-1, 10, -1, -1, 40, 50])
        this.std = numpy.array([-1, 2, -1, -1, 3, 4])
        this.min = numpy.array([-1, 5, -1, -1, 20, 25])
        this.max = numpy.array([-1, 20, -1, -1, 80, 100])
        this.volume = numpy.array([-1, 20, -1, -1, 10, 10])
        other = Density()
        other.setIds(ids=[2,3,6])
        other.mean = numpy.array([-1, -1, 20, 30, -1, -1, 60])
        other.std = numpy.array([-1, -1, 2, 3, -1, -1, 6])
        other.min = numpy.array([-1, -1, 10, 15, -1, -1, 30])
        other.max = numpy.array([-1, -1, 40, 60, -1, -1, 120])
        other.volume = numpy.array([-1, -1, 20, 10, -1, -1, 10])
 
        this.merge(new=other, mode='replace', mode0='consistent')
        np_test.assert_equal(this.ids, [1, 2, 3, 4, 5, 6])
        np_test.assert_equal(this.min, [5, 5, 10, 15, 20, 25, 30])
        np_test.assert_equal(this.max, [120, 20, 40, 60, 80, 100, 120])
        np_test.assert_equal(this.volume, [80, 20, 20, 10, 10, 10, 10])
        np_test.assert_almost_equal(this.mean, [30, 10, 20, 30, 40, 50, 60])
        np_test.assert_almost_equal(this.std, [17., 2, 2, 3, 3, 4, 6])


    def testExtractOne(self):
        """
        Tests extractOne()
        """
        
        dens_1 = self.density.extractOne(id_=1, array_=False)
        np_test.assert_almost_equal(dens_1.mean, 22)

    def testNameIds(self):
        """
        Tests nameIds()
        """

        # calculate
        density = Density()
        density.calculate(image=self.grey, segments=self.segments)

        #
        names = {1:'one', 2:'two', 3:'three', 4:'four'}
        density.nameIds(names=names)

        np_test.assert_almost_equal(density.one.mean, density.mean[1])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDensity)
    unittest.TextTestRunner(verbosity=2).run(suite)
