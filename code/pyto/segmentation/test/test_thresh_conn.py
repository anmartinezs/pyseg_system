"""

Tests module thresh_connd and implicitly Hierarchy.addLevel and related

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range

__version__ = "$Revision$"

from copy import copy, deepcopy
import importlib
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.grey import Grey
from pyto.segmentation.segment import Segment
from pyto.segmentation.thresh_conn import ThreshConn
from pyto.segmentation.test import common

class TestThreshConn(np_test.TestCase):
    """
    """

    def setUp(self):
        importlib.reload(common) # to avoid problems when running multiple tests

    def testMakeLevelsGen(self):
        """
        Tests makeLevelsGen and implicitly Hierarchy.addLevel
        """

        ################################################
        #
        # old (non generator way)
        #
        tc_a = ThreshConn()
        tc_a.setConnParam(boundary=common.bound_1, boundaryIds=[3, 4], 
                          nBoundary=1, boundCount='at_least', mask=5)
        thresh = [3, 1, 5, 7, 6, 2, 0, 4]
        tc_a.makeLevels(image=common.image_1, thresh=thresh)
        desired_levelIds = [[], [5,6], [10,11,12], [1,2,3,4], [13,14],
                            [7], [9], [8]]
        np_test.assert_equal(tc_a.levelIds, desired_levelIds)
        np_test.assert_equal(tc_a.threshold, list(range(8)))
        np_test.assert_equal(tc_a.thresh, [0,3,3,3,3,1,1,5,7,6,2,2,2,4,4])
        self.id_correspondence(actual=tc_a.data, desired=common.hi_data_1)

        # contacts
        np_test.assert_equal(
            tc_a.contacts.findSegments(boundaryIds=[3,4], nBoundary=1), 
            tc_a.ids)

        ################################################
        #
        # generator compare with old
        #
        tc_b = ThreshConn()
        tc_b.setConnParam(boundary=common.bound_1, boundaryIds=[3, 4], 
                          nBoundary=1, boundCount='at_least', mask=5)
        for vars in tc_b.makeLevelsGen(image=common.image_1, thresh=thresh, 
                                       order=None):
            pass
        np_test.assert_equal(tc_b.levelIds, tc_a.levelIds)
        self.id_correspondence(actual=tc_b.data, desired=common.hi_data_1)

        # contacts
        np_test.assert_equal(
            tc_b.contacts.findSegments(boundaryIds=[3,4], nBoundary=1), 
            tc_a.contacts.findSegments(boundaryIds=[3,4], nBoundary=1)) 


        ################################################
        #
        # generator ascend with count
        #
        tc_c = ThreshConn()
        tc_c.setConnParam(boundary=common.bound_1, boundaryIds=[3, 4], 
                          nBoundary=1, boundCount='at_least', mask=5)
        for vars in tc_c.makeLevelsGen(image=common.image_1, thresh=thresh, 
                                       order='ascend', count=True):
            seg, level, curr_thresh = vars

            # test segments at individual levels
            np_test.assert_equal(tc_c.threshold[level], 
                                 common.threshold_1[level])
            np_test.assert_equal(curr_thresh, common.threshold_1[level])
            desired_thresh = [curr_thresh] * len(seg.ids)
            np_test.assert_equal(seg.threshold[seg.ids], desired_thresh)
            try:
                np_test.assert_equal(seg.data[2:6, 1:9], common.data_1[level])
            except AssertionError:
                np_test.assert_equal(seg.data[2:6, 1:9]>0, 
                                     common.data_1[level]>0)
                np_test.assert_equal(seg.ids, common.levelIds_1[level])
                np_test.assert_equal(numpy.unique(seg.data), 
                                     [0] + common.levelIds_1[level])
                np_test.assert_equal(seg.ids, [6,7,8,9])
                if self.reorder_warning:
                    print(
                        "The exact id assignment is different from what it was "
                        + "when this test was written, but this really depends "
                        + "on internals of scipy.ndimage. Considering that the "
                        + "segments are correct, most likely everything is ok.")
                self.reorder_warning = False

            # contacts
            np_test.assert_equal(
                seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=1), 
                common.levelIds_1[level])
            if len(seg.ids) > 0:
                np_test.assert_equal(seg.contacts.getN(boundaryId=3)[seg.ids],
                                     common.n_contact_1[0, seg.ids])
                np_test.assert_equal(seg.contacts.getN(boundaryId=4)[seg.ids],
                                     common.n_contact_1[1, seg.ids])
            
        # test final hierarchy
        np_test.assert_equal(tc_c.threshold, common.threshold_1)
        desired_levelIds = [[], [1,2], [3,4,5], [6,7,8,9], [10,11],
                            [12], [13], [14]]
        np_test.assert_equal(tc_c.levelIds, desired_levelIds)
        self.id_correspondence(actual=tc_c.data, desired=common.hi_data_1)
        np_test.assert_equal(tc_c.threshold, list(range(8)))
        np_test.assert_equal(tc_c.thresh, [0,1,1,2,2,2,3,3,3,3,4,4,5,6,7])

        # contacts
        np_test.assert_equal(
            tc_c.contacts.findSegments(boundaryIds=[3,4], nBoundary=1), 
            tc_c.ids)

        ################################################
        #
        # generator descend
        #
        tc_c = ThreshConn()
        tc_c.setConnParam(boundary=common.bound_1, boundaryIds=[3, 4], 
                          nBoundary=1, boundCount='at_least', mask=5)
        desired_levelIds = [[], [13,14], [10,11,12], [6,7,8,9], [4,5],
                            [3], [2], [1]]
        for vars in tc_c.makeLevelsGen(image=common.image_1, thresh=thresh, 
                                       order='descend'):
            seg, level, curr_thresh = vars
            real_level = curr_thresh
            np_test.assert_equal(seg.data[2:6, 1:9]>0, 
                                 common.data_1[real_level]>0)
            np_test.assert_equal(
                seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=1), 
                desired_levelIds[real_level])

        np_test.assert_equal(tc_c.levelIds, desired_levelIds)
        self.id_correspondence(actual=tc_c.data, desired=common.hi_data_1)

        ################################################
        #
        # generator, inset
        #
        tc_d = ThreshConn()
        tc_d.setConnParam(boundary=common.bound_1in, boundaryIds=[3, 4], 
                          nBoundary=1, boundCount='at_least', mask=5)
        desired_levelIds = [[], [13,14], [10,11,12], [6,7,8,9], [4,5],
                            [3], [2], [1]]
        for vars in tc_d.makeLevelsGen(image=common.image_1in, thresh=thresh, 
                                       order='descend'):
            seg, level, curr_thresh = vars
            real_level = curr_thresh
            np_test.assert_equal(seg.data[1:5, :8]>0, 
                                 common.data_1[real_level]>0)
            np_test.assert_equal(
                seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=1), 
                desired_levelIds[real_level])

        np_test.assert_equal(tc_d.levelIds, desired_levelIds)
        self.id_correspondence(actual=tc_d.data, 
                               desired=common.hi_data_1[1:7, 1:9])
        tc_d.useInset(inset=[slice(2,6), slice(1,9)], mode='abs', useFull=True,
                    expand=True)
        np_test.assert_equal(tc_d.data>0, common.data_1[-1]>0)

        ################################################
        #
        # generator, boundaries 2
        #
        tc_e = ThreshConn()
        tc_e.setConnParam(boundary=common.bound_2in, boundaryIds=[3, 4, 6, 8], 
                          nBoundary=1, boundCount='at_least', mask=[5, 9])
        for vars in tc_e.makeLevelsGen(image=common.image_1in, thresh=thresh, 
                                       order='descend'):
             seg, level, curr_thresh = vars
             real_level = curr_thresh
             np_test.assert_equal(seg.data[1:5, 0:8]>0, 
                                  common.data_1[real_level]>0)

        desired_levelIds = [[], [13,14], [10,11,12], [6,7,8,9], [4,5],
                            [3], [2], [1]]
        np_test.assert_equal(tc_e.levelIds, desired_levelIds)
        self.id_correspondence(actual=tc_e.data, 
                               desired=common.hi_data_1[1:7, 1:9])

    def id_correspondence(self, actual, desired):
        """
        Check that data (given in actual and desired) agree and return
        dictionary with actual_id : desired_id pairs
        """

        # check overall agreement 
        np_test.assert_equal(actual>0, desired>0)

        # check that individual segments agree
        desired_ids = numpy.unique(desired[desired>0])
        id_dict = {}
        for d_id in desired_ids:
            a_id = actual[desired==d_id][0]
            np_test.assert_equal(actual==a_id, desired==d_id)
            id_dict[d_id] = a_id

        return id_dict

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThreshConn)
    unittest.TextTestRunner(verbosity=2).run(suite)

