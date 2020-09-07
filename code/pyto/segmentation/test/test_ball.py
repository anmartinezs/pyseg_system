"""
Tests class Balls.



# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
#from past.utils import old_div

__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.ball import Ball

class TestBall(np_test.TestCase):
    """
    """

    def setUp(self):

        self.array_1 = numpy.zeros((10,10,10), dtype='int8')
        self.array_1[1:9,1:9,1:9] = 8
        self.array_1[:,:,4] = numpy.array(\
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 2, 2, 2, 2, 8, 8, 8, 8, 0],
             [0, 2, 2, 2, 2, 8, 8, 8, 8, 0],
             [0, 2, 2, 2, 2, 8, 8, 8, 8, 0],
             [0, 2, 2, 2, 2, 8, 8, 8, 8, 0],
             [0, 8, 8, 8, 8, 8, 8, 8, 8, 0],
             [0, 8, 8, 8, 8, 8, 8, 8, 8, 0],
             [0, 8, 8, 8, 8, 4, 4, 4, 4, 0],
             [0, 8, 8, 8, 8, 8, 8, 8, 8, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.disc_1 = Ball(data=self.array_1)

        self.array_2 = numpy.zeros((10,10,10), dtype='int8')
        self.array_2[:,:,4] = numpy.array(\
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
             [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
             [0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
             [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
             [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
             [0, 0, 0, 0, 0, 0, 4, 4, 4, 0],
             [0, 0, 7, 0, 0, 0, 0, 4, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.disc_2 = Ball(data=self.array_2)

    def testExtendDiscs(self):
        """
        Tests extendDiscs()
        """
        ball_1 = self.disc_1.extendDiscs(ids=[2,4], external=8)
        desired_4 = numpy.array(\
            [[0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
             [2, 2, 2, 2, 2, 8, 8, 8, 8, 0],
             [2, 2, 2, 2, 2, 8, 8, 8, 8, 0],
             [2, 2, 2, 2, 2, 8, 8, 8, 8, 0],
             [0, 2, 2, 2, 8, 8, 8, 8, 8, 0],
             [0, 8, 8, 8, 8, 8, 8, 8, 8, 0],
             [0, 8, 8, 8, 8, 4, 4, 4, 8, 0],
             [0, 8, 8, 8, 8, 4, 4, 4, 8, 0],
             [0, 8, 8, 8, 8, 4, 4, 4, 8, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(ball_1.data[:,:,4], desired_4)
        desired_5 = numpy.array(\
            [[0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
             [0, 2, 2, 2, 8, 8, 8, 8, 8, 0],
             [2, 2, 2, 2, 2, 8, 8, 8, 8, 0],
             [0, 2, 2, 2, 8, 8, 8, 8, 8, 0],
             [0, 8, 2, 8, 8, 8, 8, 8, 8, 0],
             [0, 8, 8, 8, 8, 8, 8, 8, 8, 0],
             [0, 8, 8, 8, 8, 8, 4, 8, 8, 0],
             [0, 8, 8, 8, 8, 4, 4, 4, 8, 0],
             [0, 8, 8, 8, 8, 8, 4, 8, 8, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np_test.assert_equal(ball_1.data[:,:,5], desired_5)
        np_test.assert_equal(ball_1.data.dtype, numpy.dtype('int8'))
        np_test.assert_equal(ball_1.ids, [2,4,8])
        
    def testExtractDiscs(self):
        """
        Tests extractDisks()
        """

        centers, radii = self.disc_1.extractDiscs(ids=[2,4])
        np_test.assert_equal(centers[0], [2.5, 2.5, 4])
        np_test.assert_equal(centers[1], [7, 6.5, 4])
        np_test.assert_almost_equal(
            radii[0], 
            (2 + 3*numpy.sqrt(2) + 2*2 + 4*numpy.sqrt(5) + numpy.sqrt(8)) / 12,
            decimal=5)
        np_test.assert_almost_equal(radii[1], (1.5+0.5)/2)

    def testThinToMaxDiscs(self):
        """
        Tests thinToMaxDisks()
        """

        # extend disk and thin, check against init dist
        ball_1 = self.disc_1.extendDiscs(ids=[2,4], external=8)
        ball_1.thinToMaxDiscs(ids=[2,4], external=8)
        np_test.assert_equal(
            ball_1.data[0:5,0:5,0], self.disc_1.data[0:5,0:5,0])

        # extend to ball, thin and extend, chack against init ball
        ball_2_desired = self.disc_2.extendDiscs(ids=[2,4,7])
        ball_2 = self.disc_2.extendDiscs(ids=[2,4,7])
        ball_2.thinToMaxDiscs(ids=[2,4,7])
        ball_2_res = ball_2.extendDiscs(ids=[2,4,7])
        np_test.assert_equal(ball_2_res.data, ball_2_desired.data)

    def testFindOverlaps(self):
        """
        Tests findOverlaps()
        """

        #
        ball = Ball()
        ids=[2, 4, 5, 7]
        radii=[6, 4, 10*numpy.sqrt(2)-4, 10*numpy.sqrt(3)-6]
        centers=[[10, 10, 10], [20, 10, 10], [30, 20, 10], [0, 0, 0]]

        # clearance 1
        overlaps = ball.findOverlaps(
            ids=ids, centers=centers, radii=radii, clearance=1)
        np_test.assert_equal(overlaps, [[2,4], [2,7], [4,5]])
        
        # clearance 0
        overlaps = ball.findOverlaps(
            ids=ids, centers=centers, radii=radii, clearance=0)
        np_test.assert_equal(overlaps, [])
        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBall)
    unittest.TextTestRunner(verbosity=2).run(suite)
