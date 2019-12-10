"""

Tests module vector

# Author: Vladan Lucic
# $Id: test_vector.py 1324 2016-07-12 15:03:34Z vladan $
"""

__version__ = "$Revision: 1324 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.geometry.vector import Vector


class TestVector(np_test.TestCase):

    def setUp(self):
        """
        """
        self.cartesian = numpy.array([[1./2, numpy.sqrt(3)/2, 2],
                                      [-1./2, numpy.sqrt(3)/2, 2],
                                      [1./2, -numpy.sqrt(3)/2, -2]])

        self.spherical = numpy.array(\
            [[numpy.sqrt(5), numpy.pi / 3, numpy.arctan2(1, 2)],
             [numpy.sqrt(5), 2 * numpy.pi / 3, numpy.arctan2(1, 2)],
             [numpy.sqrt(5), -numpy.pi / 3, numpy.pi - numpy.arctan2(1, 2)]])

        theta = 180 * numpy.arctan2(1, 2) / numpy.pi 
        self.degrees = numpy.array([[60, theta],
                                    [120, theta],
                                    [-60, 180 - theta]])

    def testCartesianToSpherical(self):
        """
        Tests conversion from cartesian to shperical coordinates.
        """

        # one 2d vec
        vec = Vector(data=[1./2, numpy.sqrt(3)/2])
        np_test.assert_almost_equal([vec.r, vec.phi], 
                                    numpy.array([1, numpy.pi / 3]))        

        # many 3d
        vec = Vector(data=self.cartesian)
        np_test.assert_almost_equal(vec.r, self.spherical[:, 0])
        np_test.assert_almost_equal(vec.phi, self.spherical[:, 1])
        np_test.assert_almost_equal(vec.theta, self.spherical[:, 2])

        # degrees
        np_test.assert_almost_equal(vec.getPhi(units='deg'), self.degrees[:, 0])
        np_test.assert_almost_equal(vec.getTheta(units='deg'), 
                                    self.degrees[:, 1])

    def testSphericalToCartesian(self):
        """
        Tests conversion from spherical to cartesian coordinates.
        """

        vec = Vector(data=self.spherical, coordinates='spherical')
        np_test.assert_almost_equal(vec.data, self.cartesian)

    def testBestFitLine(self):
        """
        Tests bestFitLine()
        """
        
        # 3D exact
        points = numpy.array([[1, 2, 3],
                              [2, 4, 6],
                              [3, 6, 9],
                              [4, 8, 12]])
        ve = Vector(points)
        cm, line = ve.bestFitLine()
        np_test.assert_almost_equal(cm.data, [2.5, 5, 7.5])
        np_test.assert_almost_equal(line.data, 
                                    numpy.array([1,2,3]) / numpy.sqrt(14))

        # 3D approximate
        points = numpy.array([[2, 3, 0],
                              [2.1, 3.2, 1],
                              [2.1, 2.9, 2],
                              [1.8, 2.9, 3]])
        ve = Vector(points)
        cm, line = ve.bestFitLine()
        np_test.assert_almost_equal(cm.data, [2., 3, 1.5])
        np_test.assert_almost_equal(line.data, numpy.array([0,0,1]), decimal=1)

        # 3D exact
        points = numpy.array([[2, 2, 2],
                              [3, 3, 2],
                              [4, 4, 2]])
        ve = Vector(points)
        cm, line = ve.bestFitLine()
        np_test.assert_almost_equal(cm.data, [3, 3, 2.])
        np_test.assert_almost_equal(line.phi, numpy.pi / 4)
        np_test.assert_almost_equal(line.theta, numpy.pi / 2)

        # 3D very slightly off
        points = numpy.array([[2, 2, 2],
                              [3, 3, 2],
                              [4, 4, 2.0001]])
        ve = Vector(points)
        cm, line = ve.bestFitLine()
        np_test.assert_almost_equal(cm.data, [3, 3, 2.], decimal=3)
        np_test.assert_almost_equal(line.phi, numpy.pi / 4, decimal=3)
        np_test.assert_almost_equal(line.theta, numpy.pi / 2, decimal=3)

    def testAngleBetween(self):
        """
        Tests angleBetween()
        """

        vec_1 = numpy.array([[1, 0, 0], [1,3,2], [0,1,1]])
        vec_2 = numpy.array([[0, 1, 0], [2,6,4], [0,0,3]])
        angles = Vector.angleBetween(vec_1, vec_2, units='deg')
        np_test.assert_almost_equal(angles, [90, 0, 45])

        vec_1 = [3,0]
        vec_2 = [0.5, numpy.sqrt(3)/2]
        angles = Vector.angleBetween(vec_1, vec_2, units='deg')
        np_test.assert_almost_equal(angles, [60])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVector)
    unittest.TextTestRunner(verbosity=2).run(suite)
