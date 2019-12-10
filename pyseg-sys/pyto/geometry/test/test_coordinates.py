"""

Tests module coordinates

# Author: Vladan Lucic
# $Id: test_coordinates.py 1311 2016-06-13 12:41:50Z vladan $
"""

__version__ = "$Revision: 1311 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.geometry.coordinates import Coordinates
from pyto.geometry.affine import Affine
from pyto.geometry.affine_2d import Affine2D
from pyto.geometry.affine_3d import Affine3D

class TestCoordinates(np_test.TestCase):
    """
    """
    
    def setUp(self):
        pass

    def testTransform(self):
        """
        Tests transform()
        """

        # 2d rotation pi/2
        a2d = Affine2D(phi=numpy.pi/2, d=[0,0], scale=[1,1])
        transf = Coordinates.transform(shape=(3,3), affine=a2d, origin=[0,0])
        desired = numpy.array(
            [[[0, -1, -2],
              [0, -1, -2],
              [0, -1, -2]],
             [[0, 0, 0],
              [1, 1, 1],
              [2, 2, 2]]])
        np_test.assert_almost_equal(transf, desired)

        # 2d rotation -pi/2
        a2d = Affine2D(phi=-numpy.pi/2, d=[0,0], scale=[1,1])
        transf = Coordinates.transform(shape=(2,4), affine=a2d, origin=[0,0])
        desired = numpy.array(
            [[[0, 1, 2, 3],
              [0, 1, 2, 3]],
             [[0, 0, 0, 0],
              [-1, -1, -1, -1]]])
        np_test.assert_almost_equal(transf, desired)

        # 2d rotation pi/2, origin
        a2d = Affine2D(phi=numpy.pi/2, d=[0,0], scale=[1,1])
        transf = Coordinates.transform(shape=(3,3), affine=a2d, origin=[1,1])
        desired = numpy.array(
            [[[2, 1, 0],
              [2, 1, 0],
              [2, 1, 0]],
             [[0, 0, 0],
              [1, 1, 1],
              [2, 2, 2]]])
        np_test.assert_almost_equal(transf, desired)

        # 2d scale, origin
        a2d = Affine2D(phi=0, d=[0,0], scale=[2,2])
        transf = Coordinates.transform(shape=(3,3), affine=a2d, origin=[1,1])
        desired = numpy.array(
            [[[-1, -1, -1],
              [1, 1, 1],
              [3, 3, 3]],
             [[-1, 1, 3],
              [-1, 1, 3],
              [-1, 1, 3]]])
        np_test.assert_almost_equal(transf, desired)

        # 2d scale, rotation, origin
        a2d = Affine2D(phi=numpy.pi/2, d=[0,0], scale=[2,2])
        transf = Coordinates.transform(shape=(3,5), affine=a2d, origin=[1,1])
        desired = numpy.array(
            [[[3, 1, -1, -3, -5],
              [3, 1, -1, -3, -5],
              [3, 1, -1, -3, -5]],
             [[-1, -1, -1, -1, -1],
              [1, 1, 1, 1, 1],
              [3, 3, 3, 3, 3]]])
        np_test.assert_almost_equal(transf, desired)

        # 2d scale, rotation, origin, center
        a2d = Affine2D(phi=numpy.pi/2, d=[0,0], scale=[2,2])
        transf = Coordinates.transform(shape=(3,5), affine=a2d, origin=[1,1],
                                       center=True)
        desired = numpy.array(
            [[[2, 0, -2, -4, -6],
              [2, 0, -2, -4, -6],
              [2, 0, -2, -4, -6]],
             [[-2, -2, -2, -2, -2],
              [0, 0, 0, 0, 0],
              [2, 2, 2, 2, 2]]])
        np_test.assert_almost_equal(transf, desired)

        # 3D rotation, origin
        q = Affine3D.getQ(numpy.pi/2, 'y')
        a3d = Affine3D(gl=q, d=[0,0,0])
        transf = Coordinates.transform(shape=(3,3,3), affine=a3d, 
                                       origin=[1,1,1])
        desired = numpy.array(
            [[[[0., 1, 2], [0, 1, 2], [0, 1, 2]],
              [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
              [[0, 1, 2], [0, 1, 2], [0, 1, 2]]],

             [[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
              [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
              [[0, 0, 0], [1, 1, 1], [2, 2, 2]]],

             [[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
        np_test.assert_almost_equal(transf, desired)

    def testTransformIndices(self):
        """
        Tests transformIndices()
        """

        gl = numpy.array([[1, 10], [2, 20]])
        transf = Coordinates.transformIndices(shape=(3,4), gl=gl, initial=[1,2])
        desired = numpy.array(
            [[[1, 11, 21, 31],
              [2, 12, 22, 32],
              [3, 13, 23, 33]],

             [[2, 22, 42, 62],
              [4, 24, 44, 64],
              [6, 26, 46, 66]]])
        np_test.assert_almost_equal(transf, desired)
 

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoordinates)
    unittest.TextTestRunner(verbosity=2).run(suite)
