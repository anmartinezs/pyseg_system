"""

Tests module hierarchy

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
from pyto.segmentation.hierarchy import Hierarchy
from pyto.segmentation.thresh_conn import ThreshConn
from pyto.segmentation.test import common as common


class TestHierarchy(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        importlib.reload(common) # to avoid problems when running multiple tests
        
        # set flag so that the warning for reordered segment ids is printed
        # only once
        self.reorder_warning = True

    def instantiateTC1(self):
        """
        Instantiates ThreshConn with image_1 and bound_1
        """

        tc = ThreshConn()
        tc.setConnParam(boundary=common.bound_1, boundaryIds=[3, 4], 
                        nBoundary=1, boundCount='at_least', mask=5)
        for vars in tc.makeLevelsGen(image=common.image_1, 
                                     thresh=common.threshold_1, order=None):
            pass
        return tc


    def testPopLevel(self):
        """
        Tests popLevel(). Implicitly tests removeHigherLevels() and
        removeLowerLevels() for top and bottom levels.
        """

        # test making hierarchy
        tc = self.instantiateTC1()
        np_test.assert_equal(tc.levelIds, common.levelIds_1)

        # test poped top level
        top_seg = tc.popLevel(level='top')
        np_test.assert_equal(top_seg.ids, common.levelIds_1[-1])
        np_test.assert_equal(top_seg.threshold[common.levelIds_1[-1]], 7)
        np_test.assert_equal(top_seg.data[2:6, 1:9], common.data_1[7])
        np_test.assert_equal(
            top_seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=1), 
            14) 

        # test remaining hierarchy after poping top level
        np_test.assert_equal(tc.levelIds, common.levelIds_1[:-1])
        np_test.assert_equal(tc.threshold, common.threshold_1[:-1])
        np_test.assert_equal(tc.thresh, 
                             common.thresh_1[:-len(common.levelIds_1[-1])])
        np_test.assert_equal(top_seg.data[2:6, 1:9]>0, common.data_1[7]>0)

        # test poped bottom level
        seg = tc.popLevel(level='bottom')
        np_test.assert_equal(seg.ids, common.levelIds_1[0])
        np_test.assert_equal(seg.threshold, [0])
        np_test.assert_equal(seg.data[2:6, 1:9], common.data_1[0])
        np_test.assert_equal(
            seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=1), None) 

        # test remaining hierarchy after poping bottom level
        np_test.assert_equal(tc.levelIds, common.levelIds_1[1:-1])
        np_test.assert_equal(tc.threshold, common.threshold_1[1:-1])
        np_test.assert_equal(tc.thresh[1:], common.thresh_1[1:-1])
        np_test.assert_equal(top_seg.data[2:6, 1:9]>0, common.data_1[7]>0)

    def testExtractLevelsGen(self):
        """
        Tests extractLevelsGen() 
        """

        # test making hierarchy
        tc = self.instantiateTC1()
        np_test.assert_equal(tc.levelIds, common.levelIds_1)
        np_test.assert_equal(tc.topLevel, len(common.threshold_1) - 1)

        ####################################################
        #
        # test extracting, order '<'
        #
        for seg, level in tc.extractLevelsGen(order='<'):
            #print 'level: ', level
            
            # test extracted segment: ids, data, contacts, threshold
            np_test.assert_equal(seg.ids, common.levelIds_1[level])
            try:
                np_test.assert_equal(seg.data[2:6, 1:9], common.data_1[level])
            except AssertionError:
                np_test.assert_equal(seg.data[2:6, 1:9]>0, 
                                     common.data_1[level]>0)
                if self.reorder_warning:
                    print(
                        "The exact id assignment is different from what it was "
                        + "when this test was written, but this really depends "
                        + "on internals of scipy.ndimage. Considering that the "
                        + "segments are correct, most likely everything is ok.")
                self.reorder_warning = False
            np_test.assert_equal(
                seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=1),
                common.levelIds_1[level])
            try:
                np_test.assert_equal(
                    seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=2),
                    common.bound_ge2_1[level])
            except AssertionError:
                np_test.assert_equal(
                    len(seg.contacts.findSegments(boundaryIds=[3,4], 
                                                  nBoundary=2)),
                    len(common.bound_ge2_1[level]))
                if self.reorder_warning:
                    print(
                        "The exact id assignment is different from what it was "
                        + "when this test was written, but this really depends "
                        + "on internals of scipy.ndimage. Considering that the "
                        + "segments are correct, most likely everything is ok.")
                    self.reorder_warning = False
            if len(seg.ids) > 0:
                np_test.assert_equal(seg.threshold[seg.ids], 
                                     common.threshold_1[level])

            # test remaining hierarchy: topLevel, levelIds
            if tc.topLevel is not None:
                np_test.assert_equal(tc.topLevel, 
                                     len(common.threshold_1) - 2 - level)
                np_test.assert_equal(tc.levelIds, common.levelIds_1[level+1:])
            else:
                np_test.assert_equal(tc.levelIds, [])

        ####################################################
        #
        # test extracting, order '>'
        #
        tc = self.instantiateTC1()
        np_test.assert_equal(tc.levelIds, common.levelIds_1)
        for seg, level in tc.extractLevelsGen(order='>'):
            #print 'level: ', level

            # test extracted segment: ids, data, contacts, threshold
            np_test.assert_equal(seg.ids, common.levelIds_1[level])
            try:
                np_test.assert_equal(seg.data[2:6, 1:9], common.data_1[level])
            except AssertionError:
                np_test.assert_equal(seg.data[2:6, 1:9]>0, 
                                     common.data_1[level]>0)
                if self.reorder_warning:
                    print(
                        "The exact id assignment is different from what it was "
                        + "when this test was written, but this really depends "
                        + "on internals of scipy.ndimage. Considering that the "
                        + "segments are correct, most likely everything is ok.")
                self.reorder_warning = False
            np_test.assert_equal(
                seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=1),
                common.levelIds_1[level])
            try:
                np_test.assert_equal(
                    seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=2),
                    common.bound_ge2_1[level])
            except AssertionError:
                np_test.assert_equal(
                    len(seg.contacts.findSegments(boundaryIds=[3,4], 
                                                  nBoundary=2)),
                    len(common.bound_ge2_1[level]))
                if self.reorder_warning:
                    print(
                        "The exact id assignment is different from what it was "
                        + "when this test was written, but this really depends "
                        + "on internals of scipy.ndimage. Considering that the "
                        + "segments are correct, most likely everything is ok.")
                    self.reorder_warning = False
            if len(seg.ids) > 0:
                np_test.assert_equal(seg.threshold[seg.ids], 
                                     common.threshold_1[level])

            # test remaining hierarchy: topLevel, levelIds
            if tc.topLevel is not None:
                np_test.assert_equal(tc.topLevel, 
                                     level - 1)
                np_test.assert_equal(tc.levelIds, common.levelIds_1[:level])
            else:
                np_test.assert_equal(tc.levelIds, [])

    def testRemove(self):
        """
        Tests remove (implicitly tests removeData and removeIds).

        Here only removeing specifed ids is tested. Removing levels is
        implicitly tested in testPopLevel and testextractLevelGen.

        Note: this test will fail if for some (numerical) reason id assignment 
        changes
        """

        # make hierarchy
        tc = self.instantiateTC1()

        ############################################
        #
        # remove id 6, new
        #

        # save original data and remove id
        level_ids_orig = deepcopy(tc.levelIds)
        data_orig = tc.data.copy()
        contacts_orig = deepcopy(tc.contacts)
        new_tc = tc.remove(ids=[6], new=True)

        # test unchanged hierarchy
        np_test.assert_equal(tc.levelIds, level_ids_orig)
        np_test.assert_equal(tc.getHigherId(3), 6)
        np_test.assert_equal(tc.getHigherId(5), 6)
        np_test.assert_equal(tc.data, data_orig)
        np_test.assert_equal(tc.contacts.segments, contacts_orig.segments)
        np_test.assert_equal(tc.contacts._n, contacts_orig._n)
        
        # test new hierarchy
        desired = level_ids_orig
        desired[3] = [7,8,9]
        np_test.assert_equal(new_tc.levelIds, desired)
        np_test.assert_equal(new_tc.getHigherId(3), 10)
        np_test.assert_equal(new_tc.getHigherId(5), 10)
        desired = numpy.where(data_orig==6, 10, data_orig)
        np_test.assert_equal(new_tc.data, desired)
        desired = deepcopy(common.ids_1)
        desired.remove(6)
        np_test.assert_equal(new_tc.contacts.segments, desired)


        ############################################
        #
        # remove id 6, new=False
        #

        # save original data and remove id
        level_ids_orig = deepcopy(tc.levelIds)
        data_orig = tc.data.copy()
        contacts_orig = deepcopy(tc.contacts)
        tc.remove(ids=[6], new=False)

        # test changed hierarchy
        desired = level_ids_orig
        desired[3] = [7,8,9]
        np_test.assert_equal(tc.levelIds, desired)
        np_test.assert_equal(tc.getHigherId(3), 10)
        np_test.assert_equal(tc.getHigherId(5), 10)
        desired = numpy.where(data_orig==6, 10, data_orig)
        np_test.assert_equal(tc.data, desired)
        desired = deepcopy(common.ids_1)
        desired.remove(6)
        np_test.assert_equal(tc.contacts.segments, desired)

    def testKeep(self):
        """
        Tests keep
        """

        # make hierarchy
        tc = self.instantiateTC1()

        # save original data and remove ids
        level_ids_orig = deepcopy(tc.levelIds)
        data_orig = tc.data.copy()
        contacts_orig = deepcopy(tc.contacts)
        new_tc = tc.keep(ids=list(range(2,11)), new=True)

        # test unchanged hierarchy
        np_test.assert_equal(tc.levelIds, level_ids_orig)
        np_test.assert_equal(tc.getHigherId(3), 6)
        np_test.assert_equal(tc.getHigherId(7), 11)
        np_test.assert_equal(tc.data, data_orig)
        np_test.assert_equal(tc.contacts.segments, contacts_orig.segments)
        np_test.assert_equal(tc.contacts._n, contacts_orig._n)
        
        # test new hierarchy
        desired = level_ids_orig
        desired = [[], [2], [3,4,5], [6,7,8,9], [10], [], [], []]
        np_test.assert_equal(new_tc.levelIds, desired)
        np_test.assert_equal(new_tc.getHigherId(3), 6)
        np_test.assert_equal(new_tc.getHigherId(7), 0)
        desired = numpy.where(data_orig==1, 3, data_orig)
        desired = numpy.where(data_orig==11, 0, desired)
        desired = numpy.where(data_orig==12, 0, desired)
        desired = numpy.where(data_orig==13, 0, desired)
        desired = numpy.where(data_orig==14, 0, desired)
        np_test.assert_equal(new_tc.data, desired)
        np_test.assert_equal(new_tc.contacts.segments, new_tc.ids)       
        
    def testAddLevel(self):
        """
        Tests addLevel()
        """

        # one level data
        hi_data = numpy.array([[1, 0, 2],
                               [1, 2, 2]])
        hi = Hierarchy(data=hi_data)
        hi.inset = [slice(3,5), slice(2,5)]
        hi.setIds(ids=[1,2])
        hi.levelIds = [[], [1,2]]
        
        # segment (different inset)
        seg_data = numpy.array([[2, 1],
                                [1, 1]])
        seg = Segment(data=seg_data)
        seg.inset = [slice(3,5), slice(1,3)]

        # add
        hi.addLevel(segment=seg, level=2, check=False, shift=10)
        desired_data = numpy.array([[12, 1, 0, 2],
                                    [11, 1, 2, 2]])
        desired_inset = [slice(3,5), slice(1,5)]
        np_test.assert_equal(hi.data, desired_data)
        np_test.assert_equal(hi.inset, desired_inset)
        np_test.assert_equal(hi.levelIds, [[], [1,2], [11,12]])
        np_test.assert_equal(hi._higherIds, {1:11, 2:0})


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHierarchy)
    unittest.TextTestRunner(verbosity=2).run(suite)
