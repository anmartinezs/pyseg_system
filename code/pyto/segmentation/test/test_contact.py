"""

Tests module contact. Need to add more tests.

# Author: Vladan Lucic
# $Id: test_contact.py 1082 2014-11-17 15:43:11Z vladan $
"""

__version__ = "$Revision: 1082 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.contact import Contact
import common as common


class TestContact(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass

    def testGetSetN(self):
        """
        Tests methods getN() and setN() and also getSegments() and 
        setSegments().
        """

        # test empty contact
        con = Contact()
        np_test.assert_equal(con._n.shape, [1,1])
        np_test.assert_equal(con.getN(boundaryId=2), [])
        np_test.assert_equal(con.getN(segmentId=2), [])
        np_test.assert_equal(con.getSegments(), [])
        np_test.assert_equal(con.getBoundaries(), [])

        # test setting and getting contacts 
        con.setN(boundaryId=2, segmentId=21, nContacts=1)
        np_test.assert_equal(con.getSegments(), [21])
        np_test.assert_equal(con.getBoundaries(), [2])
        con.setN(boundaryId=2, segmentId=23, nContacts=3)
        np_test.assert_equal(con.getSegments(), [21, 23])
        np_test.assert_equal(con.getBoundaries(), [2])
        con.setN(boundaryId=3, segmentId=32, nContacts=2)
        con.setN(boundaryId=3, segmentId=21, nContacts=1)
        np_test.assert_equal(con.getSegments(), [21, 23, 32])
        np_test.assert_equal(con.getBoundaries(), [2, 3])
        np_test.assert_equal(con.getN(boundaryId=2, segmentId=23), 3)
        np_test.assert_equal(con.getN(boundaryId=2)[[21, 23]], [1, 3])
        np_test.assert_equal(con.getN(boundaryId=2, segmentId=[21, 23]), [1, 3])
        np_test.assert_equal(con.getN(segmentId=21)[[2, 3]], [1, 1])

    def testSegmentsBoundaries(self):
        """
        Tests getSegments() and getBoundries().
        """

        contacts = Contact()
        contacts._n = numpy.ma.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0], 
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1]])
        contacts._n._mask = numpy.zeros(shape=(4,5), dtype=bool)
        contacts._n._mask[0:1,0:1] = True
        np_test.assert_equal(contacts.segments, [1, 2, 3, 4])
        np_test.assert_equal(contacts.boundaries, [1, 2, 3])
        # prehaps this should be the correct behavior
        #np_test.assert_equal(contacts.segments, [2, 3, 4])
        #np_test.assert_equal(contacts.boundaries, [1, 3])

    def testAddBoundary(self):
        """
        Tests addBoundary()
        """

        # start from empty contacts
        contacts = Contact()
        contacts.addBoundary(id=2, nContacts=[-1, 1, 0, 3])
        np_test.assert_equal(contacts.maxBoundary, 2)
        np_test.assert_equal(contacts.maxSegment, 3)
        np_test.assert_equal(contacts.getN(boundaryId=2), [-2, 1, 0, 3])
        np_test.assert_equal(contacts.getN(segmentId=3), [-99, -99, 3])
        
        # add to extend 
        contacts.addBoundary(id=4, nContacts=[-1, 11, 0, 33, 44])
        np_test.assert_equal(contacts.maxBoundary, 4)
        np_test.assert_equal(contacts.maxSegment, 4)
        np_test.assert_equal(contacts.getN(boundaryId=2), [-99, 1, 0, 3, -99])
        np_test.assert_equal(contacts.getN(boundaryId=4), [-99, 11, 0, 33, 44])
        np_test.assert_equal(contacts.getN(segmentId=3), [-99, -99, 3, 0, 33])

        # add wo extending 
        contacts.addBoundary(id=1, nContacts=[-1, 7])
        np_test.assert_equal(contacts.maxBoundary, 4)
        np_test.assert_equal(contacts.maxSegment, 4)
        np_test.assert_equal(contacts.getN(boundaryId=1), [-99, 7, 0, 0, 0])
        np_test.assert_equal(contacts.getN(boundaryId=2), [-99, 1, 0, 3, -99])
        np_test.assert_equal(contacts.getN(boundaryId=4), [-99, 11, 0, 33, 44])
        np_test.assert_equal(contacts.getN(segmentId=3), [-99, 0, 3, 0, 33])

    def testCountContacted(self):
        """
        Tests contContactedBoundaries() and countCountedSegments()
        """

        # empty contacts
        contacts = Contact()
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[5]).mask, [True])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[6]).mask, [True])

        # one boundary
        contacts.addBoundary(id=2, nContacts=[-1, 1, 0, 3])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[2]), [-99, 1, 0, 1])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[]).mask, 
            [True, True, True, True])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[1]).mask, 
            [True, True, True, True])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[77]).mask, 
            [True, True, True, True])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[2,77]), [-99, 1, 0, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[1]), [-99, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[2]), [-99, -99, 0])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[3]), [-99, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[1,3]), [-99, -99, 2])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[]).mask, [True, True, True])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[77]).mask, [True, True, True])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[77,3]), [-99, -99, 1])

        # another boundary added
        contacts.addBoundary(id=4, nContacts=[-1, 11, 0, 33, 44])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[2]), [-99, 1, 0, 1, 0])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[4]), [-99, 1, 0, 1, 1])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[2,4]), [-99, 2, 0, 2, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[1]), [-99, -99, 1, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[3]), [-99, -99, 1, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[4]), [-99, -99, 0, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[3,4]), [-99, -99, 1, -99, 2])

        # test nested
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[[2,4]]), [-99, 1, 0, 1, 1])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[[2,4], 2]), 
            [-99, 2, 0, 2, 1])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContact)
    unittest.TextTestRunner(verbosity=2).run(suite)
