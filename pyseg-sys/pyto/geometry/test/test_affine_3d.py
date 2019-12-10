"""

Tests module affine_3d

# Author: Vladan Lucic
# $Id: test_affine_3d.py 1072 2014-11-06 14:07:58Z vladan $
"""

__version__ = "$Revision: 1072 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.geometry.affine_3d import Affine3D


class TestAffine3D(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass        

    def testGetQ(self):
        """
        Tests getQ()
        """

        vector = numpy.array([1,2,3])

        # rotations around main axes
        aff = Affine3D(alpha=numpy.pi/2, axis='x', d=0)
        trans = aff.transform(vector)
        np_test.assert_almost_equal(trans, [1, -3, 2])
        aff = Affine3D(alpha=numpy.pi/2, axis='y')
        trans = aff.transform(vector)
        np_test.assert_almost_equal(trans, [3, 2, -1])
        aff = Affine3D(alpha=numpy.pi/2, axis='z')
        trans = aff.transform(vector)
        np_test.assert_almost_equal(trans, [-2, 1, 3])

        # rotations around main axes specified as vectors
        q = Affine3D.getQ(alpha=numpy.pi/2, axis=[1, 0, 0])
        aff = Affine3D(gl=q, d=0)
        trans = aff.transform(vector)
        np_test.assert_almost_equal(trans, [1, -3, 2])
        q = Affine3D.getQ(alpha=numpy.pi/2, axis=[0, 1, 0])
        aff = Affine3D(gl=q)
        trans = aff.transform(vector)
        np_test.assert_almost_equal(trans, [3, 2, -1])
        q = Affine3D.getQ(alpha=numpy.pi/2, axis=[0, 0, 1])
        aff = Affine3D(gl=q)
        trans = aff.transform(vector)
        np_test.assert_almost_equal(trans, [-2, 1, 3])

        # rotations around other axes
        vector = [0, numpy.sqrt(3) / 2, 1/2.]
        axis = [0, 1/2., numpy.sqrt(3) / 2]
        aff = Affine3D(alpha=numpy.pi, axis=axis, d=0)
        trans = aff.transform(vector)
        np_test.assert_almost_equal(trans, [0, 0, 1.])
        aff = Affine3D(alpha=numpy.pi/2, axis=axis, d=0)
        trans = aff.transform(vector)
        np_test.assert_almost_equal(trans, [-0.5, numpy.sqrt(3)/4, 3/4.])
       
    def testCompose(self):
        """
        Tests compose()
        """

        # make a transform
        rot_phi = Affine3D(alpha=numpy.pi/2, axis='z')
        rot_theta = Affine3D(alpha=numpy.pi/3, axis='y')
        rot = Affine3D.compose(rot_theta, rot_phi)

        desired = numpy.array(
            [[0, -0.5, numpy.sqrt(3)/2],
             [1, 0, 0],
             [0, numpy.sqrt(3)/2, 0.5]])
        np_test.assert_almost_equal(desired, rot.gl)
        np_test.assert_almost_equal(1, rot.scale)
        np_test.assert_almost_equal(1, rot.parity)
        #np_test.assert_almost_equal(numpy.pi/2, rot.phi)
        #np_test.assert_almost_equal(numpy.pi/3, rot.theta)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAffine3D)
    unittest.TextTestRunner(verbosity=2).run(suite)
