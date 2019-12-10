"""

Tests module topology

# Author: Vladan Lucic
# $Id: test_topology.py 1304 2016-06-02 13:12:55Z vladan $
"""

__version__ = "$Revision: 1304 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.segment import Segment
from pyto.segmentation.topology import Topology
import pyto.segmentation.test.common as common


class TestTopology(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """

        # trivial flat
        seg_data = numpy.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 1, 0, 0, 0]])
        seg_data = numpy.expand_dims(seg_data, 0)
        self.trivial_flat = Segment(data=seg_data)

        # trivial 3d
        seg_data = numpy.array(
            [[[0, 0, 2, 0, 0],
              [0, 2, 2, 0, 0],
              [0, 2, 2, 2, 2],
              [0, 2, 0, 0, 0]],
             [[0, 2, 0, 2, 2],
              [0, 2, 2, 2, 2],
              [2, 2, 2, 2, 2],
              [0, 0, 0, 2, 2]]])
        self.trivial = Segment(data=seg_data)

        # loop
        seg_data = numpy.array(
            [[[0, 0, 3, 3, 3, 0],
              [3, 3, 3, 0, 3, 3],
              [3, 3, 3, 0, 0, 3],
              [3, 0, 0, 0, 0, 0]],
             [[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 3, 3, 3, 3, 3],
              [0, 0, 0, 0, 3, 0]]])
        self.loop = Segment(data=seg_data)

        # small sphere
        seg_data = numpy.array(
            [[[0, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0]],
             [[0, 1, 1, 1, 0, 0],
              [0, 1, 0, 1, 0, 0],
              [0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0]],
             [[0, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0]]])
        self.small_sphere = Segment(data=seg_data)

        # sphere
        seg_data = numpy.array(
            [[[0, 1, 1, 1, 1, 0],
              [0, 1, 1, 1, 1, 0],
              [0, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 0]],
             [[0, 1, 1, 1, 1, 0],
              [0, 1, 1, 0, 1, 0],
              [0, 1, 0, 0, 1, 1],
              [0, 1, 1, 1, 1, 1]],
             [[0, 0, 1, 1, 1, 0],
              [0, 1, 1, 1, 1, 0],
              [0, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 0]]])
        self.sphere = Segment(data=seg_data)

        # torus
        seg_data = numpy.array(
            [[[4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 0, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4]],
             [[4, 4, 4, 4, 4, 4, 4],
              [4, 0, 0, 0, 0, 0, 4],
              [4, 0, 4, 4, 4, 0, 4],
              [4, 0, 4, 0, 4, 0, 4],
              [4, 0, 4, 4, 4, 0, 4],
              [4, 0, 0, 0, 0, 0, 4],
              [4, 4, 4, 4, 4, 4, 4]],
             [[4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 0, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4],
              [4, 4, 4, 4, 4, 4, 4]]])
        self.torus  = Segment(data=seg_data)

    def testCalculate(self):
        """
        Tests calculate, getEuler, countFaces and getHomologyRank methods.
        """

        # calculate
        topo = Topology(segments=common.segment_3)
        ids = numpy.insert(topo.ids, 0, 0)
        topo.calculate()

        # test
        np_test.assert_equal(topo.euler[ids], common.euler_3)
        np_test.assert_equal(topo.nObjects[ids], common.objects_3)
        np_test.assert_equal(topo.nLoops[ids], common.loops_3)
        np_test.assert_equal(topo.nHoles[ids], common.holes_3)
        np_test.assert_equal(topo.homologyRank[ids,0], common.objects_3)
        np_test.assert_equal(topo.homologyRank[ids,1], common.loops_3)
        np_test.assert_equal(topo.homologyRank[ids,2], common.holes_3)
        np_test.assert_equal(topo.homologyRank[ids,3], numpy.zeros(8))
        np_test.assert_equal(topo.nFaces[ids], common.faces_3)

        # add a nonexisting index
        topo.calculate(ids=numpy.append(topo.ids, 23))
        np_test.assert_equal(topo.homologyRank[23,0], 0)
        np_test.assert_equal(topo.homologyRank[23,1], 0)
        np_test.assert_equal(topo.homologyRank[23,2], 0)
        np_test.assert_equal(topo.homologyRank[23,3], 0)

        # trivial flat
        topo = Topology(segments=self.trivial_flat)
        topo.calculate()
        np_test.assert_equal(topo.euler[1], 1)
        np_test.assert_equal(topo.nLoops[1], 0)
        np_test.assert_equal(topo.nHoles[1], 0)
 
        # trivial
        topo = Topology(segments=self.trivial)
        topo.calculate()
        np_test.assert_equal(topo.euler[2], 1)
        np_test.assert_equal(topo.nLoops[2], 0)
        np_test.assert_equal(topo.nHoles[2], 0)

        # loop
        topo = Topology(segments=self.loop)
        topo.calculate()
        np_test.assert_equal(topo.euler[3], 0)
        np_test.assert_equal(topo.nLoops[3], 1)
        np_test.assert_equal(topo.nHoles[3], 0)
 
        # small sphere
        topo = Topology(segments=self.small_sphere)
        topo.calculate()
        np_test.assert_equal(topo.euler[1], 2)
        np_test.assert_equal(topo.nLoops[1], 0)
        np_test.assert_equal(topo.nHoles[1], 1)
 
         # sphere
        topo = Topology(segments=self.sphere)
        topo.calculate()
        np_test.assert_equal(topo.euler[1], 2)
        np_test.assert_equal(topo.nLoops[1], 0)
        np_test.assert_equal(topo.nHoles[1], 1)
 
        # torus
        topo = Topology(segments=self.torus)
        topo.calculate()
        np_test.assert_equal(topo.euler[4], 0)
        np_test.assert_equal(topo.nLoops[4], 2)
        np_test.assert_equal(topo.nHoles[4], 1)

        # no segments
        data_0 = numpy.zeros(shape=(2,4,3), dtype=int)
        seg_0 = Segment(data=data_0)
        topo_0 = Topology(segments=seg_0)
        topo_0.calculate()
        np_test.assert_equal(topo_0.euler, [0])
        np_test.assert_equal(topo_0.nFaces, numpy.array([[0, 0, 0, 0]]))
 
        # segments exist but no ids
        data_1 = numpy.ones(shape=(2,4,3), dtype=int)
        seg_1 = Segment(data=data_1)
        topo_1 = Topology(segments=seg_1)
        topo_1.calculate(ids=[2,4])
        np_test.assert_equal(topo_1.euler, [0, 0, 0, 0, 0])
        np_test.assert_equal(topo_1.nFaces, numpy.zeros(shape=(5,4), dtype=int))
 
    def testRestrict(self):
        """
        Tests restrict and setTotal methods
        """

        # make object and restrict
        topo_1 = Topology(segments=common.segment_3)
        topo_1.calculate()
        res_ids = [2,3,11,22]
        topo_1.restrict(ids=res_ids)

        # make a restricted object
        topo_2 = Topology(segments=common.segment_3, ids=res_ids)
        topo_2.calculate()

        # test
        ids0 = [0] + res_ids
        np_test.assert_equal(topo_1.euler[ids0], topo_2.euler[ids0])
        np_test.assert_equal(topo_1.nFaces[ids0], topo_2.nFaces[ids0])
        np_test.assert_equal(topo_1.homologyRank[ids0], 
                             topo_2.homologyRank[ids0])


if __name__ == '__main__':
    suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestTopology)
    unittest.TextTestRunner(verbosity=2).run(suite)

