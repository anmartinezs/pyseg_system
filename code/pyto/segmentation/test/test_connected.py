"""

Tests module connected. 

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import

__version__ = "$Revision$"

from copy import copy, deepcopy
import importlib
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.grey import Grey
from pyto.segmentation.segment import Segment
from pyto.segmentation.connected import Connected
from pyto.segmentation.test import common

class TestConnected(np_test.TestCase):
    """
    """

    def setUp(self):
        importlib.reload(common) # to avoid problems when running multiple tests

    def testMake(self):
        """
        Tests make()
        """

        conn, contacts = \
            Connected.make(image=common.image_1, boundary=common.bound_1, 
                           thresh=4, boundaryIds=[3, 4], mask=5,
                           nBoundary=1, boundCount='ge')
        np_test.assert_equal(conn.ids, [1,2])
        i1 = conn.data[2,2]
        i2 = conn.data[2,5]
        desired = numpy.zeros((10,10), dtype=int)
        desired[2:6, 1:9] = numpy.array(\
            [[0, 1, 0, 0, 2, 2, 2, 0],
             [0, 1, 0, 0, 2, 0, 2, 0],
             [1, 1, 1, 0, 2, 0, 2, 2],
             [1, 0, 1, 0, 2, 0, 2, 0]])
        self.id_correspondence(conn.data, desired)

        conn, contacts = \
            Connected.make(image=common.image_1, boundary=common.bound_1, 
                           thresh=4, boundaryIds=[3, 4], mask=5,
                           nBoundary=1, boundCount='exact')
        np_test.assert_equal(conn.ids, [])

        conn, contacts = \
            Connected.make(image=common.image_1, boundary=common.bound_1, 
                           thresh=2, boundaryIds=[3, 4], mask=5,
                           nBoundary=2, boundCount='eq')
        np_test.assert_equal(conn.ids, [1])

        conn, contacts = \
            Connected.make(image=common.image_1, boundary=common.bound_1, 
                           thresh=2, boundaryIds=[3, 4], mask=5,
                           nBoundary=1, boundCount='at_most')
        np_test.assert_equal(conn.ids, [1,2])

        # test ids and data
        conn, contacts = \
            Connected.make(image=common.image_1, boundary=common.bound_1, 
                           thresh=2, boundaryIds=[3, 4], mask=5,
                           nBoundary=1, boundCount='at_least')
        np_test.assert_equal(conn.ids, [1,2,3])
        desired = numpy.zeros((10,10), dtype=int)
        desired[2:6, 1:9] = numpy.array(\
            [[0, 1, 0, 0, 0, 3, 3, 0],
             [0, 1, 0, 0, 0, 0, 3, 0],
             [0, 0, 0, 0, 0, 0, 3, 0],
             [2, 0, 0, 0, 0, 0, 3, 0]])
        self.id_correspondence(conn.data, desired)

        # use insets
        conn, contacts = Connected.make(
            image=common.image_1in2, boundary=common.bound_1in, thresh=2, 
            boundaryIds=[3, 4], mask=5, nBoundary=1, boundCount='at_least')
        np_test.assert_equal(conn.ids, [1,2,3])
        self.id_correspondence(conn.data, desired[1:7, 1:9])

        # mask Segment
        mask = Segment(data=numpy.where(common.bound_1.data==5, 1, 0))
        image_inset = copy(common.image_1.inset)
        bound_inset = copy(common.bound_1.inset)
        image_data = common.image_1.data.copy()
        bound_data = common.bound_1.data.copy()
        conn, contacts = Connected.make(
            image=common.image_1, boundary=common.bound_1, thresh=2., 
            boundaryIds=[3, 4], mask=mask, nBoundary=1, boundCount='at_least')
        np_test.assert_equal(conn.ids, [1,2,3])
        desired = numpy.zeros((10,10), dtype=int)
        desired[2:6, 1:9] = numpy.array(
            [[0, 1, 0, 0, 0, 3, 3, 0],
             [0, 1, 0, 0, 0, 0, 3, 0],
             [0, 0, 0, 0, 0, 0, 3, 0],
             [2, 0, 0, 0, 0, 0, 3, 0]])
        np_test.assert_equal(conn.data>0, desired>0)
        np_test.assert_equal(image_inset, common.image_1.inset)
        np_test.assert_equal(bound_inset, common.bound_1.inset)
        np_test.assert_equal(image_data, common.image_1.data)
        np_test.assert_equal(bound_data, common.bound_1.data)

        # boundary inset, mask Segment same inset
        mask = Segment(data=numpy.where(common.bound_1in.data==5, 1, 0))
        mask.setInset(inset=[slice(1,7), slice(1,9)], mode='abs')
        conn, contacts = Connected.make(
            image=common.image_1, boundary=common.bound_1in, thresh=2, 
            boundaryIds=[3, 4], mask=mask, nBoundary=1, boundCount='at_least')
        np_test.assert_equal(conn.ids, [1,2,3])
        np_test.assert_equal(conn.data.shape, (6,8))
        np_test.assert_equal(conn.inset, [slice(1,7), slice(1,9)])
        desired = numpy.zeros((6,8), dtype=int)
        desired[1:5, 0:8] = numpy.array(
            [[0, 1, 0, 0, 0, 3, 3, 0],
             [0, 1, 0, 0, 0, 0, 3, 0],
             [0, 0, 0, 0, 0, 0, 3, 0],
             [2, 0, 0, 0, 0, 0, 3, 0]])
        np_test.assert_equal(conn.data>0, desired>0)

        # boundary inset, mask Segment smaller inset (inside boundaries)
        mask = Segment(data=numpy.where(common.bound_1in.data==5, 1, 0))
        mask.setInset(inset=[slice(1,7), slice(1,9)], mode='abs')
        mask.useInset([slice(2,6), slice(1,9)], mode='abs')
        conn, contacts = Connected.make(
            image=common.image_1, boundary=common.bound_1in, thresh=2, 
            boundaryIds=[3, 4], mask=mask, nBoundary=1, boundCount='at_least')
        np_test.assert_equal(conn.ids, [1,2,3])
        np_test.assert_equal(conn.data.shape, (6,8))
        np_test.assert_equal(conn.inset, [slice(1,7), slice(1,9)])
        desired = numpy.zeros((6,8), dtype=int)
        desired[1:5, 0:8] = numpy.array(
            [[0, 1, 0, 0, 0, 3, 3, 0],
             [0, 1, 0, 0, 0, 0, 3, 0],
             [0, 0, 0, 0, 0, 0, 3, 0],
             [2, 0, 0, 0, 0, 0, 3, 0]])
        np_test.assert_equal(conn.data>0, desired>0)

        # boundary inset, mask Segment even smaller inset (inside boundaries)
        mask = Segment(data=numpy.where(common.bound_1in.data==5, 1, 0))
        mask.setInset(inset=[slice(1,7), slice(1,9)], mode='abs')
        mask.useInset([slice(2,6), slice(2,9)], mode='abs')
        image_inset = copy(common.image_1.inset)
        bound_inset = copy(common.bound_1in.inset)
        image_data = common.image_1.data.copy()
        bound_data = common.bound_1in.data.copy()
        conn, contacts = Connected.make(
            image=common.image_1, boundary=common.bound_1in, thresh=2, 
            boundaryIds=[3, 4], mask=mask, nBoundary=1, boundCount='at_least')
        np_test.assert_equal(conn.ids, [1,2])
        np_test.assert_equal(conn.data.shape, (6,8))
        np_test.assert_equal(conn.inset, [slice(1,7), slice(1,9)])
        desired = numpy.zeros((6,8), dtype=int)
        desired[1:5, 1:8] = numpy.array(
            [[1, 0, 0, 0, 2, 2, 0],
             [1, 0, 0, 0, 0, 2, 0],
             [0, 0, 0, 0, 0, 2, 0],
             [0, 0, 0, 0, 0, 2, 0]])
        np_test.assert_equal(conn.data, desired)
        np_test.assert_equal(image_inset, common.image_1.inset)
        np_test.assert_equal(bound_inset, common.bound_1in.inset)
        np_test.assert_equal(image_data, common.image_1.data)
        np_test.assert_equal(bound_data, common.bound_1in.data)

        # image smaller than boundaries
        mask = Segment(data=numpy.where(common.bound_1in.data==5, 1, 0))
        mask.setInset(inset=[slice(1,7), slice(1,9)], mode='abs')
        mask.useInset([slice(2,6), slice(1,9)], mode='abs')
        image_inset = copy(common.image_1in.inset)
        image_data = common.image_1in.data.copy()
        bound_inset = copy(common.bound_1in.inset)
        bound_data = common.bound_1in.data.copy()
        conn, contacts = Connected.make(
            image=common.image_1in, boundary=common.bound_1in, thresh=2, 
            boundaryIds=[3, 4], mask=mask, nBoundary=1, boundCount='at_least')
        np_test.assert_equal(conn.ids, [1,2,3])
        np_test.assert_equal(conn.data.shape, (6,8))
        np_test.assert_equal(conn.inset, [slice(1,7), slice(1,9)])
        desired = numpy.zeros((6,8), dtype=int)
        desired[1:5, 0:8] = numpy.array(
            [[0, 1, 0, 0, 0, 3, 3, 0],
             [0, 1, 0, 0, 0, 0, 3, 0],
             [0, 0, 0, 0, 0, 0, 3, 0],
             [2, 0, 0, 0, 0, 0, 3, 0]])
        np_test.assert_equal(conn.data>0, desired>0)
        np_test.assert_equal(image_inset, common.image_1in.inset)
        np_test.assert_equal(image_data, common.image_1in.data)
        np_test.assert_equal(bound_inset, common.bound_1in.inset)
        np_test.assert_equal(bound_data, common.bound_1in.data)

        # image smaller than boundaries and intersects with free, boundaries
        # intersects with free
        image = Grey(data=common.image_1.data.copy())
        image.useInset(inset=[slice(2,6), slice(2,9)], mode='abs')
        image_inset = copy(image.inset)
        image_data = image.data.copy()
        common.bound_1in.useInset(inset=[slice(1, 7), slice(1, 8)], mode='abs')
        bound_1in_inset = copy(common.bound_1in.inset)
        bound_data = common.bound_1in.data.copy()
        mask = Segment(data=numpy.where(common.bound_1in.data==5, 1, 0))
        mask.setInset(inset=[slice(1,7), slice(1,9)], mode='abs')
        mask.useInset([slice(2,6), slice(1,9)], mode='abs')
        conn, contacts = Connected.make(
            image=image, boundary=common.bound_1in, thresh=3, 
            boundaryIds=[3, 4], mask=mask, nBoundary=1, boundCount='at_least')
        np_test.assert_equal(conn.ids, [1,2,3,4])
        np_test.assert_equal(conn.data.shape, (6,7))
        np_test.assert_equal(conn.inset, [slice(1,7), slice(1,8)])
        desired = numpy.zeros((6,7), dtype=int)
        desired[1:5, 0:8] = numpy.array(
            [[0, 1, 0, 0, 0, 3, 3],
             [0, 1, 0, 0, 0, 0, 3],
             [0, 1, 0, 0, 0, 0, 3],
             [0, 0, 4, 0, 2, 0, 3]])
        np_test.assert_equal(conn.data>0, desired>0)
        np_test.assert_equal(image_inset, image.inset)
        np_test.assert_equal(bound_1in_inset, common.bound_1in.inset)
        np_test.assert_equal(image_data, image.data)
        np_test.assert_equal(bound_data, common.bound_1in.data)
        common.bound_1in.useInset(
            inset=[slice(1, 7), slice(1, 9)], mode='abs', expand=True)


    def id_correspondence(self, actual, desired):
        """
        Check that data (given in actual and desired) agree and return
        dictionary with actual_id : desired_id pairs
        """

        # check overall agreement 
        np_test.assert_equal(actual>0, desired>0)

        # checl that individual segments agree
        desired_ids = numpy.unique(desired[desired>0])
        id_dict = {}
        for d_id in desired_ids:
            a_id = actual[desired==d_id][0]
            np_test.assert_equal(actual==a_id, desired==d_id)
            id_dict[d_id] = a_id

        return id_dict


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConnected)
    unittest.TextTestRunner(verbosity=2).run(suite)

