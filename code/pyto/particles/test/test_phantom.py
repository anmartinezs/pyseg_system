"""

Tests phantom.py

# Author: Vladan Lucic (Max Planck Institute of Biochemistry)
# $Id: test_phantom.py 1454 2017-05-11 16:56:32Z vladan $
"""

__version__ = "$Revision: 1454 $"


import os
import unittest

import numpy as np
import numpy.testing as np_test 

from pyto.geometry.rigid_3d import Rigid3D as Rigid3D
from pyto.particles.phantom import Phantom as Phantom

class TestPhantom(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass

    def test_make_2box(self):
        """
        Tests make_2box()
        """

        # simplest case
        base_size = [2,3,8]
        prot_size = [2,6,5]
        base_pos = [0,0,0]
        prot_pos = [0,3,3]
        desired_x_0 = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1]])
        ph = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[15,15,15], center=[0,0,0], 
            origin=[0,0,0], binary=True)
        np_test.assert_equal(ph.data[0, 0:9, 0:8], desired_x_0)
        np_test.assert_equal(ph.data[1, 0:9, 0:8], desired_x_0)
        np_test.assert_equal(ph.data.sum(), 2 * desired_x_0.sum())

        # origin None
        base_size = [2,3,8]
        prot_size = [2,6,5]
        base_pos = [0,0,0]
        prot_pos = [0,3,3]
        ph = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[20,20,20], center=[0,0,0], 
            origin=None, binary=True)
        np_test.assert_equal(ph.data[10, 10:19, 10:18], desired_x_0)
        np_test.assert_equal(ph.data[11, 10:19, 10:18], desired_x_0)
        np_test.assert_equal(ph.data.sum(), 2 * desired_x_0.sum())

        # origin and center
        base_size = [2,3,8]
        prot_size = [2,6,5]
        base_pos = [0,0,0]
        prot_pos = [0,3,3]
        ph = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[15,15,15], center=[1,3,2], 
            origin=[4,5,6], binary=True)
        np_test.assert_equal(ph.data[3, 2:11, 4:12], desired_x_0)
        np_test.assert_equal(ph.data[4, 2:11, 4:12], desired_x_0)
        np_test.assert_equal(ph.data.sum(), 2 * desired_x_0.sum())

        # everything
        base_size = [2,3,8]
        prot_size = [2,6,5]
        base_pos = [1,2,3]
        prot_pos = [1,5,6]
        ph = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[20,20,20], center=[1,3,2],
            origin=[4,5,6], binary=True)
        np_test.assert_equal(ph.data[4, 4:13, 7:15], desired_x_0)
        np_test.assert_equal(ph.data[5, 4:13, 7:15], desired_x_0)
        np_test.assert_equal(ph.data.sum(), 2 * desired_x_0.sum())

        # overlap binary=True
        base_size = [2,3,8]
        prot_size = [2,6,5]
        base_pos = [0,0,0]
        prot_pos = [0,1,2]
        desired = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]])
        ph = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[15,15,15], center=[0,0,0], 
            origin=[0,0,0], binary=True)
        np_test.assert_equal(ph.data[0, 0:9, 0:8], desired)
        np_test.assert_equal(ph.data[1, 0:9, 0:8], desired)
        np_test.assert_equal(ph.data.sum(), 2 * desired.sum())

        # overlap binary=False
        base_size = [2,3,8]
        prot_size = [2,6,5]
        base_pos = [0,0,0]
        prot_pos = [0,1,2]
        desired = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 2, 2, 2, 2, 2, 1],
             [1, 1, 2, 2, 2, 2, 2, 1],
             [0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]])
        ph = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[15,15,15], center=[0,0,0], 
            origin=[0,0,0], binary=False)
        np_test.assert_equal(ph.data[0, 0:9, 0:8], desired)
        np_test.assert_equal(ph.data[1, 0:9, 0:8], desired)
        np_test.assert_equal(ph.data.sum(), 2 * desired.sum())

    def test_make_homomer(self):
        """
        Tests test_make_homomer()
        """
        
        # binary wo overlap
        base_size = [2,3,8]
        prot_size = [2,6,5]
        base_pos = [0,0,0]
        prot_pos = [0,3,3]
        ph = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[20,20,20], center=[0,0,0], 
            origin=[10,10,10], binary=True)
        desired_x_0 = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1]])
        homo = ph.make_homomer(symmetry='c4', binary=True)
        np_test.assert_equal(homo.data[10, 10:19, 10:18], desired_x_0)
        np_test.assert_equal(homo.data[11, 10:19, 10:18], desired_x_0)
        np_test.assert_equal(homo.data[10:19, 10, 10:18], desired_x_0)
        np_test.assert_equal(homo.data[10, 10:1:-1, 10:18], desired_x_0)
        np_test.assert_equal(homo.data[10:1:-1, 10, 10:18], desired_x_0)
        np_test.assert_equal(homo.data[:, :, :10].sum(), 0)

        # not binary w overlap
        base_size = [2,3,8]
        prot_size = [2,6,5]
        base_pos = [0,0,0]
        prot_pos = [0,3,3]
        ph = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[20,20,20], center=[0,0,0], 
            origin=[10,10,10], binary=True)
        desired_x_0 = np.array(
            [[4, 4, 4, 4, 4, 4, 4, 4],
             [2, 2, 2, 2, 2, 2, 2, 2],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1]])
        homo = ph.make_homomer(symmetry='c4', binary=False)
        np_test.assert_almost_equal(
            homo.data[10, 10:19, 10:18], desired_x_0)
        np_test.assert_equal(
            homo.data[11, 12:19, 10:18], desired_x_0[2:,:])
        np_test.assert_equal(
            homo.data[11, 10:12, 10:18], desired_x_0[0:2,:] / 2.)
        np_test.assert_almost_equal(
            homo.data[10:19, 10, 10:18], desired_x_0)
        np_test.assert_almost_equal(
            homo.data[10, 10:1:-1, 10:18], desired_x_0)
        np_test.assert_almost_equal(
            homo.data[10:1:-1, 10, 10:18], desired_x_0)
        np_test.assert_equal(homo.data[:, :, :10].sum(), 0)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhantom)
    unittest.TextTestRunner(verbosity=2).run(suite)

