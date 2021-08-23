"""

Tests module rigid_3d

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"

#from copy import copy, deepcopy
import unittest

import numpy as np
import numpy.testing as np_test 
#import scipy as sp

from pyto.geometry.affine_2d import Affine2D
from pyto.geometry.rigid_3d import Rigid3D


class TestRigid3D(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        self.ninit = 10

    def testIdentity(self):
        """
        Tests identity()
        """
        ide = Rigid3D.identity()
        np_test.assert_equal(ide.q, np.identity(3))
        np_test.assert_equal(ide.s_scalar, 1)

        ide = Rigid3D.identity(ndim=3)
        np_test.assert_equal(ide.q, np.identity(3))
        np_test.assert_equal(ide.s_scalar, 1)

    def testS(self):
        """
        Tests getS and setS
        """
        
        r3d = Rigid3D()
        r3d.s_scalar = 2.3
        np_test.assert_equal(r3d.s, 2.3 * np.identity(3))

        r3d = Rigid3D(scale=2.4)
        np_test.assert_equal(r3d.s, 2.4 * np.identity(3))

        r3d = Rigid3D.identity()
        r3d.s = 3.4 * np.identity(3)
        np_test.assert_equal(r3d.s_scalar, 3.4)

    def testMakeScalar(self):
        """
        Tests makeScalar()
        """

        s = Rigid3D.makeScalar(s=np.identity(3))
        np_test.assert_equal(s, 1)

        q = Rigid3D.make_r_euler([1.,2.,3.])
        s_scalar = 2.5
        r3d = Rigid3D(q=q, scale=s_scalar)
        q, p, s, m = r3d.decomposeQR(order='qr')
        np_test.assert_almost_equal(Rigid3D.makeScalar(s=s, check=False), 2.5)

    def testGl(self):
        """
        Tests getGl and segGl
        """

        # test getGl()
        r3d = Rigid3D.identity()
        np_test.assert_equal(r3d.gl, np.identity(3))
        r3d.s_scalar = 1.7
        np_test.assert_equal(r3d.gl, 1.7 * np.identity(3))
        r3d.s_scalar = 2.7
        np_test.assert_equal(r3d.gl, 2.7 * np.identity(3))
        r3d.q = np.array([[0,0,1], [0,1,0], [-1,0,0]])
        np_test.assert_equal(r3d.gl, 2.7 * np.array(
            [[0,0,1], [0,1,0], [-1,0,0]]))
       
        # test setGl()
        r3d = Rigid3D()
        r3d.gl = 2.6 * np.identity(3)
        np_test.assert_equal(r3d.q, np.identity(3))
        np_test.assert_equal(r3d.s_scalar, 2.6)
        r3d.gl = 2.8 * np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np_test.assert_equal(r3d.q, np.array([[0,-1,0],[1,0,0],[0,0,1]]))
        np_test.assert_equal(r3d.s_scalar, 2.8)

    def testD(self):
        """
        Tests setting d and makeD()
        """
        
        r3d = Rigid3D(q=np.identity(3))
        np_test.assert_almost_equal(r3d.d, [0,0,0])
        r3d = Rigid3D()
        np_test.assert_almost_equal(r3d.d, [0,0,0])
        r3d = Rigid3D(q=np.identity(3), d=-1)
        np_test.assert_almost_equal(r3d.d, [-1,-1,-1])
        r3d = Rigid3D(q=np.identity(3), d=[1,2,3])
        np_test.assert_almost_equal(r3d.d, [1,2,3])

        r3d = Rigid3D(q=np.identity(3))
        r3d.d = [2,3,4]
        np_test.assert_almost_equal(r3d.d, [2,3,4])
        r3d.d = Rigid3D.makeD(2)
        np_test.assert_almost_equal(r3d.d, [2,2,2])
        r3d.d = Rigid3D.makeD(d=[5,4,3])
        np_test.assert_almost_equal(r3d.d, [5,4,3])

    def testCompose(self):
        """
        Tests compose()
        """

        t1 = Rigid3D(q=np.identity(3), d=[1,2,3])
        t2 = Rigid3D(q=np.identity(3))
        com = Rigid3D.compose(t2, t1)
        np_test.assert_equal(com.d, [1,2,3])

    def test_make_r_axis(self):
        """
        Tests make_r_axis()
        """
        
        # pi/2 z
        np_test.assert_almost_equal(
            Rigid3D.make_r_axis(angle=np.pi/2, axis='z'),
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np_test.assert_almost_equal(
            Rigid3D.make_r_axis(angle=np.pi/3, axis='z'),
            [[np.cos(np.pi/3), -np.sin(np.pi/3), 0], 
             [np.sin(np.pi/3), np.cos(np.pi/3), 0], 
             [0, 0, 1]])

        # -pi/2 y
        np_test.assert_almost_equal(
            Rigid3D.make_r_axis(angle=-np.pi/2, axis='y'),
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        np_test.assert_almost_equal(
            Rigid3D.make_r_axis(angle=-1., axis='y'),
            [[np.cos(-1.), 0, np.sin(-1.)], 
             [0, 1, 0], 
             [-np.sin(-1.), 0, np.cos(-1.)]])

        # pi x
        np_test.assert_almost_equal(
            Rigid3D.make_r_axis(angle=np.pi, axis='x'),
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        np_test.assert_almost_equal(
            Rigid3D.make_r_axis(angle=0.5, axis='x'),
            [[1, 0, 0],
             [0, np.cos(0.5), -np.sin(0.5)], 
             [0, np.sin(0.5), np.cos(0.5)]])

    def testInverse(self):
        """
        Tests inverse()
        """

        q = Rigid3D.make_r_euler(angles=[1., 2., 3])
        r3d = Rigid3D(q=q, d=[1,2,-4])
        r3di = r3d.inverse()
        com = Rigid3D.compose(r3d, r3di)
        np_test.assert_almost_equal(com.q, np.identity(3))
        np_test.assert_almost_equal(com.d, [0,0,0])

    def test_shift_angle_range(self):
        """
        Tests shift_angle_range()
        """

        # default low
        np_test.assert_almost_equal(Rigid3D.shift_angle_range(angle=1), 1)
        np_test.assert_almost_equal(
            Rigid3D.shift_angle_range(angle=2), 2)
        np_test.assert_almost_equal(
            Rigid3D.shift_angle_range(angle=4.), 4 - 2*np.pi)
        np_test.assert_almost_equal(
            Rigid3D.shift_angle_range(angle=6.), 6 - 2*np.pi)
        np_test.assert_almost_equal(
            Rigid3D.shift_angle_range(angle=12.), 12 - 4*np.pi)
        np_test.assert_almost_equal(
            Rigid3D.shift_angle_range(angle=-1), -1)
        np_test.assert_almost_equal(
            Rigid3D.shift_angle_range(angle=-5), -5 + 2*np.pi)

        # given low
        np_test.assert_almost_equal(
            Rigid3D.shift_angle_range(angle=4., low=0), 4)
        np_test.assert_almost_equal(
            Rigid3D.shift_angle_range(angle=4., low=-2*np.pi), 4 - 2*np.pi)
        
    def test_find_32_constr_ck_scale_1(self):
        """
        Tests find_32_constr_ck(scale=1)
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # identity
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.gl, np.identity(3))
        np_test.assert_almost_equal(res.y, x_cs)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around z axis
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        r_desired = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r_desired, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around z axis, 3 markers
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        r_desired = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs[:,1:4], y=y[:2,1:4], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r_desired, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/6 rotation around z axis
        r = Rigid3D.make_r_euler([0, np.pi/6, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/6 rotation around z axis, 3 markers
        r = Rigid3D.make_r_euler([0, np.pi/6, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs[:,1:4], y=y[:2,1:4], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,1:4], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 8 pi/7 rotation around z axis
        r = Rigid3D.make_r_euler([0, 8*np.pi/7, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around x axis
        y = np.array([[0., 1, 0, 0],
                      [0, 0, 0, -3],
                      [0, 0, 2, 0]])
        r_desired = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r_desired, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/3 rotation around x axis
        r = Rigid3D.make_r_euler([np.pi/3, 0, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/3 rotation around x axis, 3 markers
        r = Rigid3D.make_r_euler([np.pi/3, 0, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs[:,1:4], y=y[:2,1:4], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,1:4], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 8 pi/9 rotation around x axis
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around y axis
        y = np.array([[0., 0, 0, 3],
                      [0, 0, 2, 0],
                      [0, -1, 0, 0]])
        r_desired = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r_desired, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/5 rotation around y axis
        r = Rigid3D.make_r_euler([np.pi/2, np.pi/5, -np.pi/2])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 7 pi/5 rotation around y axis
        r = Rigid3D.make_r_euler([np.pi/2, 7 * np.pi/5, -np.pi/2])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # identity, non-optimal initial
        # doesn't find optimal
        # cm=True improves but doesn't find optimal
        # fine if optimizing scale
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=1, cm=False, use_jac=False, 
            init=[0.2, -0.4, 0.5, -0.1])
        #np_test.assert_almost_equal(res.y, x_cs, decimal=3)
        #np_test.assert_almost_equal(res.gl, np.identity(3), decimal=3)
        #np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        # cm=True improves but doesn't find optimal
        # fine if optimizing scale
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=1, cm=False, use_jac=True, 
            init=[0.2, -0.4, 0.5, np.sqrt(0.55)])
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        # fine when +np.sqrt(0.55)
        # cm=True improves but doesn't find optimal
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=False, 
            init=[0.2, -0.4, 0.5, -np.sqrt(0.55)])

        # fails for 4, 5, 6 * pi/5, ok for 3 and 7
        # small init changes don't help
        # reducing z helps, the closer theta to pi the larger reduction  
        # cm=True improves but doesn't find optimal
        # fine if optimizing scale
        r = Rigid3D.make_r_euler([np.pi/2, 6 * np.pi/5, -np.pi/2])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True) 

        # pi around z (fi)
        # fails after 1 iter when init=[1, 0, 0, 0] and cm=False
        r = Rigid3D.make_r_euler([np.pi, 0, 0])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            init=[0.2, -0.4, 0.5, -np.sqrt(0.55)])
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi around z (psi)
        # fails after 1 iter when init=[1, 0, 0, 0] and cm=False
        r = Rigid3D.make_r_euler([0, 0, np.pi])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            init=[0.2, -0.4, 0.5, -np.sqrt(0.55)])
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # low z example 1
        x = np.array([[2., 5.3, 7.2, 0.3, 4],
                      [4, 3.2, 6, 5.4, 1.2],
                      [0.5, 1.2, 0.3, 0.5, 0.1]])
        angles = np.array([50, 40, 24]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.gl, mode='x'), 
            angles, decimal=3)

        # low z example 2
        x = np.array([[2., 5.3, 7.2, 0.3, 4],
                      [4, 3.2, 6, 5.4, 1.2],
                      [0.5, 1.2, 0.3, 0.5, 0.1]])
        angles = np.array([-150, 45, 130]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.gl, mode='x'), 
            angles, decimal=3)

        # low z example 3
        # Note: fails with default init when fi and psi interchanged, added to 
        # test_find_32_constr_ck_multi()
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([168, 32, -123]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.gl, mode='x'), 
            angles, decimal=3)

        # low z example w noise
        x = np.array([[32., 78, 3, 41, 50, 47],
                      [13, 36, 54, 6, 38, 63],
                      [1.1, 3.5, 2.8, 4.2, 1.3, 3.2]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = np.dot(r, x_cm)
        y[:2,:] = y[:2,:] + np.array([[0.8, -0.3, 1.1, -0.9, 0.4, -0.5],
                                      [0.1, 0.9, 0.4, 0.1, -0.3, -0.4]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=y[:2,:], scale=1, cm=False, use_jac=True)
        #np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=2)
        #np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=2)
        #np_test.assert_almost_equal(res.gl, r, decimal=1)
        #np_test.assert_almost_equal(
        #    Rigid3D.extract_euler(res.gl, mode='x'), 
        #    angles, decimal=1)

    def test_find_32_constr_ck_scale_fixed(self):
        """
        Tests find_32_constr_ck() where scale is fixed but is not 1.
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # pi/2 rotation around z axis
        y = 3 * np.array([[0., 0, -2, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 3]])
        r_desired = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=3, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.q, r_desired, decimal=3)
        np_test.assert_almost_equal(res.gl, 3 * r_desired, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 7 pi/5 rotation around y axis
        r = Rigid3D.make_r_euler([np.pi/2, 7 * np.pi/5, -np.pi/2])
        y = 4.5 * np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=4.5, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 4.5)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # low z example
        # Note: fails with default init when fi and psi interchanged, added to 
        # test_find_32_constr_ck_multi()
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([168, 32, 123]) * np.pi / 180
        scale = 4.8
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=scale*np.dot(r, x_cm)[:2,:], scale=4.8, cm=False, 
            use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], 4.8 * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)
 
    def test_find_32_constr_ck(self):
        """
        Tests find_32_constr_ck(scale=None) 
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # identity
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=None, cm=False, use_jac=False)
        np_test.assert_almost_equal(res.gl, np.identity(3))
        np_test.assert_almost_equal(res.y, x_cs)
        np_test.assert_almost_equal(res.s_scalar, 1)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around z axis
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        r_desired = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=None, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.q, r_desired, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 8 pi/9 rotation around x axis
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        scale = 12.3
        y = scale * np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=None, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(scale * np.dot(res.q, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # identity, non-optimal initial
        # doesn't find optimal without scale optimization
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=None, cm=False, use_jac=True, 
            init=[0.2, -0.4, 0.5, -0.1, 1])
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), x_cs, decimal=3)
        np_test.assert_almost_equal(res.gl, np.identity(3), decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # identity, non-optimal initial
        # doesn't find optimal without scale optimization
        # still fails when -np.sqrt(0.55)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=None, cm=False, use_jac=True, 
            init=[0.2, -0.4, 0.5, np.sqrt(0.55), 1])
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        np_test.assert_almost_equal(res.y, x_cs, decimal=3)
        np_test.assert_almost_equal(res.q, np.identity(3), decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 6 pi / 5 around y axis
        # fails without scale optimization
        r = Rigid3D.make_r_euler([np.pi/2, 6 * np.pi/5, -np.pi/2])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=None, cm=False, use_jac=True) 
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi around z (fi)
        # fails with init=[1, 0, 0, 0], even if cm=True
        r = Rigid3D.make_r_euler([np.pi, 0, 0])
        y = 2.1 * np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=None, cm=False, use_jac=True,
            init=[0.9, 0.2, 0.2, 0.2, 1])
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 2.1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi around z (psi)
        # fails with init=[1, 0, 0, 0], even if cm=True
        r = Rigid3D.make_r_euler([0, 0, np.pi])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=None, cm=False, use_jac=True,
            init=[0.9, 0.2, 0.2, 0.2, 1])
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # low z example 1
        x = np.array([[2., 5.3, 7.2, 0.3, 4],
                      [4, 3.2, 6, 5.4, 1.2],
                      [0.5, 1.2, 0.3, 0.5, 0.1]])
        angles = np.array([50, 40, 24]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=None, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.gl, mode='x'), 
            angles, decimal=3)

        # low z example 2
        x = np.array([[2., 5.3, 7.2, 0.3, 4],
                      [4, 3.2, 6, 5.4, 1.2],
                      [0.5, 1.2, 0.3, 0.5, 0.1]])
        angles = np.array([-150, 45, 130]) * np.pi / 180
        scale = 3.4
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3
        # fails for default and some other init conditions
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([168, 32, 123]) * np.pi / 180
        scale = 1.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, init=[0.2, -0.4, 0.5, -np.sqrt(0.55), 1])
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3
        # fails for some init conditions, even if init scale close to correct
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([134, 32, -78]) * np.pi / 180
        scale = 56.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, init=[-0.4, -0.3, 0.8, np.sqrt(0.11), 10])
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

    def test_find_32_constr_ck_multi(self):
        """
        Tests find_32_constr_ck_multi()
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # theta = 6 pi/5, random init e
        r = Rigid3D.make_r_euler([0., 6 * np.pi/5, 0.])
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            ninit=self.ninit, randome=True, randoms=False) 
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # theta = 6 pi/5, phi = pi/2, theta = -pi/2, random init e
        r = Rigid3D.make_r_euler([np.pi/2, 6 * np.pi/5, -np.pi/2])
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            ninit=self.ninit, randome=True, randoms=False) 
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # theta = pi, phi = pi/2, theta = -pi/2 (identity), random init e
        r = Rigid3D.make_r_euler([np.pi/2, np.pi, -np.pi/2])
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            ninit=self.ninit, randome=True, randoms=False) 
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # low z example 3, random init e, optimize scale
        # fails for some init conditions, even if init scale close to correct
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        scale = 56.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, ninit=self.ninit, randome=True, randoms=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3, random init e, fixed scale
        # Note: fails with the default init e 
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=1, cm=False, 
            use_jac=True, ninit=self.ninit, randome=True, randoms=False)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3, random init e and scale, optimize scale
        # fails for some init conditions (including default), even if init 
        # scale close to correct
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        scale = 56.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, ninit=self.ninit, randome=True, randoms=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3
        # fails for some init conditions (eg [0.4, -0.3, 0.8, 0.2, 1]), 
        # even if init scale close to correct
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([134, 32, -78]) * np.pi / 180
        scale = 56.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, ninit=self.ninit, randome=True, randoms=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

    def test_find_32(self):
        """
        Tests find_32()
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # low z in cm frame
        x_low_z = np.array([[3.2, 7.8, 0.3, 4, 5],
                            [1.3, 3.6, 5.4, 6, 3.8],
                            [0.1, 0.5, 0.8, 0.2, 0.3]])
        x_low_z = x_low_z - x_low_z.mean(axis=-1).reshape((3,1))

        # transfom including translation, single default initial
        # pi/6 rotation around z axis
        r = Rigid3D.make_r_euler([0, np.pi/6, 0])
        d = np.array([5,6,7.])
        y = np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=1, use_jac=True, ninit=1, randome=False,
            randoms=False)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:] - res.y[2:].mean(), 
            y[2,:] - y[2:].mean(), decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # transfom including translation and scale, single default initial
        # pi/6 rotation around z axis
        r = Rigid3D.make_r_euler([0, np.pi/6, 0])
        d = np.array([5,6,7.])
        s = 5.6
        y = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=None, use_jac=True, ninit=1, randome=False,
            randoms=False)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:] - res.y[2:].mean(), 
            y[2,:] - y[2:].mean(), decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.gl, s * r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, s, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # single specified initial rotation, no scale nor translation
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, use_jac=True, ninit=1, 
            randome=False, einit=[0.2, -0.4, 0.5, np.sqrt(0.55)], randoms=False)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # w translation and scale, single specified initial rotation
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        s = 124.3
        d = np.array([-3, -45., 17])
        y = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, einit=[0.2, -0.4, 0.5, np.sqrt(0.55)], randoms=False)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, s, decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # w translation and scale, single specified initial scale
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        s = 124.3
        d = np.array([-3, -45., 17])
        y = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, randoms=False, sinit=0.5)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, s, decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # w translation and scale, single specified initial rotation and scale
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        s = 124.3
        d = np.array([-3, -45., 17])
        y = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, einit=[0.2, -0.4, 0.5, np.sqrt(0.55)], 
            randoms=False, sinit=0.5)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, s, decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # single specified initial rotation, w scale and translation
        # fails for default inits
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        #res = Rigid3D.find_32(
        #    x=x_low_z, y=y[:2,:], scale=None use_jac=True, ninit=1, 
        #    randome=False, randoms=False)
        #np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, einit=[0.6, -0.5, 0.3, np.sqrt(0.3)], randoms=False)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # single specified initial scale, w scale and translation
        # fails for default inits
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, randoms=False, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # single gl2 initial rotation, w scale and translation
        # fails for default inits
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, einit='gl2', randoms=False, sinit='gl2')
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation, single initial scale
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit=None, randoms=False, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial scale, single initial rotation
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=False, einit=None, randoms=True, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation and scale
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit=None, randoms=True, sinit=None)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation around specified, single scale
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit=[1,0,0,0], randoms=False, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial scale around specified, single rotation
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=False, einit=[1,0,0,0], randoms=True, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation and scale around specified
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit=[1,0,0,0], randoms=True, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation around gl2, single scale
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit='gl2', randoms=False, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial scale around gl2, single rotation
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=False, einit=[1,0,0,0], randoms=True, sinit='gl2')
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation and scale around gl2
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit='gl2', randoms=True, sinit='gl2')
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

    def test_approx_gl2_to_ck3(self):
        """
        Test approx_gl2_to_ck3()
        """
        
        # coord system-like points
        x = np.array([[0., 1, 0, -1],
                      [0, 0, 2, -1],
                      [0, 0, 0, 0]])

        # arbitrary rotation, euler
        euler = np.array([60, 40, -35.]) * np.pi / 180
        e_euler = Rigid3D.euler_to_ck(euler)
        r = Rigid3D.make_r_euler(euler)
        y = Rigid3D().transform(x=x, q=r, s=1., d=0)
        e_res, s_res = Rigid3D().approx_gl2_to_ck3(x=x, y=y[:2,:], ret='both')
        np_test.assert_almost_equal(s_res, 1.)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_euler))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == np.sign(e_euler)).all() or
             (np.sign(e_res[0]) == -np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == -np.sign(e_euler)).all()), True)

        # arbitrary rotation, euler
        euler = np.array([100, 80, -170.]) * np.pi / 180
        e_euler = Rigid3D.euler_to_ck(euler)
        r = Rigid3D.make_r_euler(euler)
        y = Rigid3D().transform(x=x, q=r, s=35., d=0)
        e_res, s_res = Rigid3D().approx_gl2_to_ck3(x=x, y=y[:2,:], ret='both')
        np_test.assert_almost_equal(s_res, 35.)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_euler))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == np.sign(e_euler)).all() or
             (np.sign(e_res[0]) == -np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == -np.sign(e_euler))).all(), True)

        # arbitrary rotation, euler, theta > pi/2
        euler = np.array([100, 100, -170.]) * np.pi / 180
        euler_flipped = np.array([100, 80, -170.]) * np.pi / 180
        e_euler = Rigid3D.euler_to_ck(euler_flipped)
        r = Rigid3D.make_r_euler(euler)
        y = Rigid3D().transform(x=x, q=r, s=1., d=0)
        e_res, s_res = Rigid3D().approx_gl2_to_ck3(x=x, y=y[:2,:], ret='both')
        np_test.assert_almost_equal(s_res, 1.)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_euler))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == np.sign(e_euler)).all() or
             (np.sign(e_res[0]) == -np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == -np.sign(e_euler)).all()), True)

        # arbitrary rotation, ck params, theta > pi/2
        e = np.array([-0.3, 0.7, 0.6, np.sqrt(0.06)])
        r = Rigid3D.make_r_ck(e=e)
        y = Rigid3D().transform(x=x, q=r, s=1, d=0)
        e_res, s_res = Rigid3D.approx_gl2_to_ck3(
            x=x, y=y[:2,:], xy_axes='dim_point')
        euler_flipped = Rigid3D.extract_euler(r, ret='one')
        euler_flipped[1] = np.pi - euler_flipped[1]
        e_flipped = Rigid3D.euler_to_ck(euler_flipped)
        np_test.assert_almost_equal(s_res, 1)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_flipped))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_flipped)).all() or
             (np.sign(e_res[1]) == np.sign(e_flipped)).all() or
             (np.sign(e_res[0]) == -np.sign(e_flipped)).all() or
             (np.sign(e_res[1]) == -np.sign(e_flipped)).all()), True)

        # arbitrary rotation, ck params + scale, theta > pi/2
        e = np.array([-0.3, 0.7, 0.6, np.sqrt(0.06)])
        r = Rigid3D.make_r_ck(e=e)
        y = Rigid3D().transform(x=x, q=r, s=4.6, d=0)
        e_res, s_res = Rigid3D.approx_gl2_to_ck3(
            x=x, y=y[:2,:], xy_axes='dim_point')
        euler_flipped = Rigid3D.extract_euler(r, ret='one')
        euler_flipped[1] = np.pi - euler_flipped[1]
        e_flipped = Rigid3D.euler_to_ck(euler_flipped)
        np_test.assert_almost_equal(s_res, 4.6)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_flipped))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_flipped)).all() or
             (np.sign(e_res[1]) == np.sign(e_flipped)).all() or
             (np.sign(e_res[0]) == -np.sign(e_flipped)).all() or
             (np.sign(e_res[1]) == -np.sign(e_flipped)).all()), True)

    def test_gl2_to_ck3(self):
        """
        Test gl2_to_ck3()
        """
        
        # make 3D r from known Gl and check angles
        euler = np.array([70, 50, -30.]) * np.pi / 180
        u = Affine2D.makeQ(euler[2])
        d = np.diag([1, np.cos(euler[1])])
        v = Affine2D.makeQ(euler[0])
        gl = 2.3 * np.dot(np.dot(u, d), v)
        res_e_param, res_s = Rigid3D.gl2_to_ck3(gl=gl, ret='one')
        res_r = Rigid3D.make_r_ck(res_e_param)
        res_euler = Rigid3D.extract_euler(res_r, mode='x', ret='one')
        np_test.assert_almost_equal(res_s, 2.3)
        np_test.assert_almost_equal(
            np.remainder(res_euler[0], np.pi), np.remainder(euler[0], np.pi))
        np_test.assert_almost_equal(np.abs(res_euler[1]), np.abs(euler[1]))
        np_test.assert_almost_equal(
            np.remainder(res_euler[2], np.pi), np.remainder(euler[2], np.pi))

    def test_make_r_euler(self):
        """
        Tests make_r_euler()
        """

        # arbitrary rotation and inverse
        res = np.dot(
            Rigid3D.make_r_euler([1., 2, 3], mode='zxz_ex_active'), 
            Rigid3D.make_r_euler([-3., -2, -1], mode='zxz_ex_active'))
        np_test.assert_almost_equal(res, np.identity(3))

        # arbitrary rotation and inverse
        res = np.dot(Rigid3D.make_r_euler([-0.5, 1.2, 2.8]), 
                     Rigid3D.make_r_euler([-2.8, -1.2, 0.5]))
        np_test.assert_almost_equal(res, np.identity(3))

        # arbitrary rotation and inverse, intrinsic
        res = np.dot(
            Rigid3D.make_r_euler([-0.5, 1.2, 2.8], mode='zxz_in_active'), 
            Rigid3D.make_r_euler([-2.8, -1.2, 0.5], mode='zxz_in_active'))
        np_test.assert_almost_equal(res, np.identity(3))

        # arbitrary rotation and inverse, intrinsic and extrinsic
        res = np.dot(
            Rigid3D.make_r_euler([-2.5, 0.6, 1.8], mode='zxz_in_active'), 
            Rigid3D.make_r_euler([2.5, -0.6, -1.8], mode='zxz_ex_active'))
        np_test.assert_almost_equal(res, np.identity(3))

        # check individual rotations fine
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0.6,0,0], mode='zxz_ex_active'),
            Rigid3D.make_r_axis(angle=0.6, axis='z'))
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0.6,0,0], mode='zyz_ex_active'),
            Rigid3D.make_r_axis(angle=0.6, axis='z'))
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0.6,0,0], mode='zyz_in_active'),
            Rigid3D.make_r_axis(angle=0.6, axis='z'))
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0,0,-2.3], mode='zxz_ex_active'),
            Rigid3D.make_r_axis(angle=-2.3, axis='z'))
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0,0,-2.3], mode='zyz_in_active'),
            Rigid3D.make_r_axis(angle=-2.3, axis='z'))
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0,0,-2.3], mode='zyz_in_active'),
            Rigid3D.make_r_axis(angle=-2.3, axis='z'))
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0,1.6,0], mode='zxz_ex_active'),
            Rigid3D.make_r_axis(angle=1.6, axis='x'))
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0,1.6,0], mode='zxz_in_active'),
            Rigid3D.make_r_axis(angle=1.6, axis='x'))
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0,1.6,0], mode='zyz_ex_active'),
            Rigid3D.make_r_axis(angle=1.6, axis='y'))
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(angles=[0,1.7,0], mode='zyz_in_active'),
            Rigid3D.make_r_axis(angle=1.7, axis='y'))

        # zxz extrinsic
        r = Rigid3D.make_r_euler(angles=[1., 2, 3], mode='zxz_ex_active')
        desired_1 = Rigid3D.make_r_axis(angle=1., axis='z')
        desired_2 = Rigid3D.make_r_axis(angle=2., axis='x')
        desired_3 = Rigid3D.make_r_axis(angle=3., axis='z')
        desired = np.dot(desired_3, np.dot(desired_2, desired_1))
        np_test.assert_almost_equal(r, desired)

        # zxz intrinsic
        r = Rigid3D.make_r_euler(angles=[0.5, -0.7, -1.2], mode='zxz_in_active')
        desired_1 = Rigid3D.make_r_axis(angle=0.5, axis='z')
        desired_2 = Rigid3D.make_r_axis(angle=-0.7, axis='x')
        desired_3 = Rigid3D.make_r_axis(angle=-1.2, axis='z')
        desired = np.dot(desired_1, np.dot(desired_2, desired_3))
        np_test.assert_almost_equal(r, desired)

        # zyz extrinsic
        r = Rigid3D.make_r_euler(angles=[-1.1, 0.5, 2.3], mode='zyz_ex_active')
        desired_1 = Rigid3D.make_r_axis(angle=-1.1, axis='z')
        desired_2 = Rigid3D.make_r_axis(angle=0.5, axis='y')
        desired_3 = Rigid3D.make_r_axis(angle=2.3, axis='z')
        desired = np.dot(desired_3, np.dot(desired_2, desired_1))
        np_test.assert_almost_equal(r, desired)

        # zyz intrinsic
        r = Rigid3D.make_r_euler(
            angles=[1.4, -2.5, -0.3], mode='zyz_in_active')
        desired_1 = Rigid3D.make_r_axis(angle=-0.3, axis='z')
        desired_2 = Rigid3D.make_r_axis(angle=-2.5, axis='y')
        desired_3 = Rigid3D.make_r_axis(angle=1.4, axis='z')
        desired = np.dot(desired_3, np.dot(desired_2, desired_1))
        np_test.assert_almost_equal(r, desired)

        # extrinsic and intrinsic
        np_test.assert_almost_equal(
            Rigid3D.make_r_euler(
                angles=[1.1, -0.5, -2.3], mode='zyz_ex_active'),
            Rigid3D.make_r_euler(
                angles=[-2.3, -0.5, 1.1], mode='zyz_in_active'))

    def test_extract_euler(self):
        """
        Tests extract_euler()
        """

        # non-degenerate extrinsic xzx
        angles = [1.5, 0.7, -0.4]
        r = Rigid3D.make_r_euler(angles, mode='zxz_ex_active')
        res = Rigid3D.extract_euler(r, ret='both', mode='zxz_ex_active')
        angles_mod = np.remainder(angles, 2*np.pi)
        res_mod = np.remainder(res, 2*np.pi)
        np_test.assert_almost_equal(res_mod[0], angles_mod)
        desired = [angles[0] + np.pi, -angles[1], angles[2] + np.pi]
        desired_mod = np.remainder(desired, 2*np.pi)
        np_test.assert_almost_equal(res_mod[1], desired_mod)

        # non-degenerate intrinsic xzx
        angles = [1.5, 0.7, -0.4]
        r = Rigid3D.make_r_euler(angles, mode='zxz_in_active')
        res = Rigid3D.extract_euler(r, ret='both', mode='zxz_in_active')
        np_test.assert_almost_equal(res[0], [1.5, 0.7, -0.4])
        np_test.assert_almost_equal(res[1], [1.5-np.pi, -0.7, -0.4+np.pi])
        angles_mod = np.remainder(angles, 2*np.pi)
        res_mod = np.remainder(res, 2*np.pi)
        np_test.assert_almost_equal(res_mod[0], angles_mod)
        desired = [angles[0] + np.pi, -angles[1], angles[2] + np.pi]
        desired_mod = np.remainder(desired, 2*np.pi)
        np_test.assert_almost_equal(res_mod[1], desired_mod)
        res = Rigid3D.extract_euler(r, ret='one', mode='zxz_in_active')
        np_test.assert_almost_equal(res, [1.5, 0.7, -0.4])

        # non-degenerate extrinsic, intrinsic xzx
        angles = [1.5, 0.7, -0.4]
        r = Rigid3D.make_r_euler(angles, mode='zxz_in_active')
        res = Rigid3D.extract_euler(r, ret='both', mode='zxz_in_active')
        np_test.assert_almost_equal(res[0], [1.5, 0.7, -0.4])
        np_test.assert_almost_equal(res[1], [1.5-np.pi, -0.7, -0.4+np.pi])
        res = Rigid3D.extract_euler(r, ret='one', mode='zxz_in_active')
        np_test.assert_almost_equal(res, [1.5, 0.7, -0.4])

        # degenerate extrinsic xzx
        angles = [1.5, 0., -0.4]
        r = Rigid3D.make_r_euler(angles)
        res = Rigid3D.extract_euler(r, ret='both')
        np_test.assert_almost_equal(res[0], [1.1, 0, 0])
        np_test.assert_almost_equal(res[1], [0, 0, 1.1])

        # non-degenerate extrinsic xyx
        angles = [1.5, 0.7, -0.4]
        r = Rigid3D.make_r_euler(angles, mode='zyz_ex_active')
        res = Rigid3D.extract_euler(r, ret='both', mode='zyz_ex_active')
        np_test.assert_almost_equal(res[0], [1.5, 0.7, -0.4])
        np_test.assert_almost_equal(res[1], [1.5-np.pi, -0.7, -0.4+np.pi])
        res = Rigid3D.extract_euler(r, ret='one', mode='zyz_ex_active')
        np_test.assert_almost_equal(res, [1.5, 0.7, -0.4])

        # non-degenerate intrinsic xyx
        angles = [1.5, 0.7, -0.4]
        r = Rigid3D.make_r_euler(angles, mode='zyz_in_active')
        res = Rigid3D.extract_euler(r, ret='both', mode='zyz_in_active')
        np_test.assert_almost_equal(res[0], [1.5, 0.7, -0.4])
        np_test.assert_almost_equal(res[1], [1.5-np.pi, -0.7, -0.4+np.pi])
        res = Rigid3D.extract_euler(r, ret='one', mode='zyz_in_active')
        np_test.assert_almost_equal(res, [1.5, 0.7, -0.4])

        # non-degenerate intrinsic xyx
        angles = [4, -0.8, 0.4]
        r = Rigid3D.make_r_euler(angles, mode='zyz_in_active')
        res = Rigid3D.extract_euler(r, ret='both', mode='zyz_in_active')
        np_test.assert_almost_equal(res[0], [4-np.pi, 0.8, 0.4-np.pi])
        np_test.assert_almost_equal(res[1], [4-2*np.pi, -0.8, 0.4])
        res = Rigid3D.extract_euler(r, ret='one', mode='zyz_in_active')
        np_test.assert_almost_equal(res, [4-np.pi, 0.8, 0.4-np.pi])

    def test_convert_euler(self):
        """
        Tests convert_euler()

        Some tests assume make_r_euler() is correct.
        """
        
        # from / to 'zxz_ex_active'
        angles = [1.2, 1.5, -0.7]
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_ex_active', final='zxz_ex_active')
        np_test.assert_almost_equal(res, [1.2, 1.5, -0.7])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_ex_active', final='zxz_in_active')
        np_test.assert_almost_equal(res, [-0.7, 1.5, 1.2])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_ex_active', final='zyz_ex_active')
        np_test.assert_almost_equal(res, [1.2+np.pi/2, 1.5, -0.7-np.pi/2])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_ex_active', final='zyz_in_active')
        np_test.assert_almost_equal(res, [-0.7+np.pi/2, 1.5, 1.2-np.pi/2])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_ex_active', final='zxz_ex_passive')
        np_test.assert_almost_equal(res, [0.7, -1.5, -1.2])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_ex_active', final='zxz_in_passive')
        np_test.assert_almost_equal(res, [-1.2, -1.5, 0.7])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_ex_active', final='zyz_ex_passive')
        np_test.assert_almost_equal(res, [0.7+np.pi/2, -1.5, -1.2-np.pi/2])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_ex_active', final='zyz_in_passive')
        np_test.assert_almost_equal(res, [-1.2+np.pi/2, -1.5, 0.7-np.pi/2])

        # same initial and final
        angles = [1.2, 1.5, -0.7]
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_ex_active', final='zxz_ex_active')
        np_test.assert_almost_equal(res, [1.2, 1.5, -0.7])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_in_active', final='zxz_in_active')
        np_test.assert_almost_equal(res, [1.2, 1.5, -0.7])
        res = Rigid3D.convert_euler(
            angles=angles, init='zyz_in_active', final='zyz_in_active')
        np_test.assert_almost_equal(res, [1.2+np.pi, 1.5, -0.7-np.pi])
        r_euler = Rigid3D.make_r_euler(angles, mode='zyz_in_active')
        res_angles = Rigid3D.extract_euler(r_euler, mode='zyz_in_active')
        np_test.assert_almost_equal(res_angles, angles)
        res = Rigid3D.convert_euler(
            angles=angles, init='zyz_ex_passive', final='zyz_ex_passive')
        np_test.assert_almost_equal(res, [1.2, 1.5, -0.7])

        # mixed
        angles = [1.2, 1.5, -0.7]
        res = Rigid3D.convert_euler(
            angles=angles, init='zyz_in_active', final='zxz_in_active')
        np_test.assert_almost_equal(res, [1.2+np.pi/2, 1.5, -0.7-np.pi/2])
        r_euler = Rigid3D.make_r_euler(angles, mode='zyz_in_active')
        res = Rigid3D.extract_euler(r_euler, mode='zxz_in_active')
        np_test.assert_almost_equal(res, [1.2+np.pi/2, 1.5, -0.7-np.pi/2])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_in_passive', final='zxz_ex_passive')
        np_test.assert_almost_equal(res, [-0.7, 1.5, 1.2])
        res = Rigid3D.convert_euler(
            angles=angles, init='zxz_in_passive', final='zxz_in_active')
        np_test.assert_almost_equal(res, [0.7, -1.5, -1.2])

    def test_make_r_ck(self):
        """
        Tests make_r_ck()

        Assumes extract_euler() is correct
        """

        # invert angle
        res = np.dot(Rigid3D.make_r_ck([0.2, 0.4, 0.3, np.sqrt(0.71)]), 
                     Rigid3D.make_r_ck([-0.2, 0.4, 0.3, np.sqrt(0.71)]))
        np_test.assert_almost_equal(res, np.identity(3))

        # invert axis
        res = np.dot(Rigid3D.make_r_ck([0.5, 0.4, 0.6, np.sqrt(0.23)]), 
                     Rigid3D.make_r_ck([0.5, -0.4, -0.6, -np.sqrt(0.23)]))
        np_test.assert_almost_equal(res, np.identity(3))

        # no rot
        e = [1., 0, 0, 0]
        r_ck = Rigid3D.make_r_ck(e)
        euler = Rigid3D.extract_euler(r_ck, mode='zxz_ex_active')
        np_test.assert_almost_equal(euler[1], 0)

        # rot around x axis
        e = np.array([1., 1, 0, 0]) / np.sqrt(2)
        r_ck = Rigid3D.make_r_ck(e)
        euler = Rigid3D.extract_euler(r_ck, mode='zxz_ex_active')
        np_test.assert_almost_equal(euler, [0, np.pi/2, 0])

        # rot around x axis
        e = np.array([0.8, 0.6, 0, 0])
        r_ck = Rigid3D.make_r_ck(e)
        euler = Rigid3D.extract_euler(r_ck, mode='zxz_ex_active')
        np_test.assert_almost_equal(euler, [0, 2* np.arccos(0.8), 0])

        # rot around y axis
        e = np.array([1., 0, 1, 0]) / np.sqrt(2)
        r_ck = Rigid3D.make_r_ck(e)
        euler = Rigid3D.extract_euler(r_ck, mode='zyz_ex_active')
        np_test.assert_almost_equal(euler, [0, np.pi/2, 0])

        # rot around z axis
        e = np.array([0.6, 0, 0, 0.8])
        r_ck = Rigid3D.make_r_ck(e)
        euler = Rigid3D.extract_euler(r_ck, mode='zxz_ex_active')
        np_test.assert_almost_equal(euler[0] + euler[2], 2* np.arccos(0.6))
        np_test.assert_almost_equal(euler[1], 0)

        # arbitrary rot 
        e = np.array([0.4, -0.5, -0.3, np.sqrt(0.52)])
        r_ck = Rigid3D.make_r_ck(e)
        np_test.assert_almost_equal(
            r_ck, 
            [[-0.2, -0.2768882, -0.96111026],
             [ 0.8768882 , -0.52      , -0.03266615],
             [-0.48111026, -0.83266615,  0.34      ]])

    def test_euler_to_ck(self):
        """
        Tests euler_to_ck()

        Some tests assume make_r_euler() and make_r_ck() are correct
        """

        # 'zxz_ex_active'
        angles = [-1.3, 0.9, 0.4]
        e = Rigid3D.euler_to_ck(angles, mode='zxz_ex_active')
        desired = [
            np.cos(-0.9/2) * np.cos(0.9/2),
            np.cos(1.7/2.) * np.sin(0.9/2),
            np.sin(1.7/2.) * np.sin(0.9/2),
            np.sin(-0.9/2) * np.cos(0.9/2)]
        np_test.assert_almost_equal(e, desired)

        # compare r, 'zxz_ex_active'
        angles = [-1.3, 0.9, 0.4]
        r_euler = Rigid3D.make_r_euler(angles, mode='zxz_ex_active')
        e = Rigid3D.euler_to_ck(angles, mode='zxz_ex_active')
        r_ck = Rigid3D.make_r_ck(e)
        np_test.assert_almost_equal(r_ck, r_euler)

        # 'zyz_ex_active'
        angles = [-1.3, 0.9, 0.4]
        e = Rigid3D.euler_to_ck(angles, mode='zyz_ex_active')
        desired = [
            np.cos(-0.9/2) * np.cos(0.9/2),
            np.cos((1.7 + np.pi)/2.) * np.sin(0.9/2),
            np.sin((1.7 + np.pi)/2.) * np.sin(0.9/2),
            np.sin(-0.9/2) * np.cos(0.9/2)]
        np_test.assert_almost_equal(e, desired)

        # compare r, 'zyz_ex_active'
        angles = [-1.3, 0.9, 0.4]
        angles = [np.pi/2, np.pi/6, 0]
        r_euler = Rigid3D.make_r_euler(angles, mode='zyz_ex_active')
        e = Rigid3D.euler_to_ck(angles, mode='zyz_ex_active')
        r_ck = Rigid3D.make_r_ck(e)
        np_test.assert_almost_equal(r_ck, r_euler)

        # compare r, 'zyz_in_active'
        angles = [-1.3, 0.9, 0.4]
        r_euler = Rigid3D.make_r_euler(angles, mode='zyz_in_active')
        e = Rigid3D.euler_to_ck(angles, mode='zyz_in_active')
        r_ck = Rigid3D.make_r_ck(e)
        np_test.assert_almost_equal(r_ck, r_euler)

        # compare r via euler and ck
        angles = [-0.5, 1.2, 2.3]
        r_euler = Rigid3D.make_r_euler(angles, mode='zxz_ex_active')
        e = Rigid3D.euler_to_ck(angles, mode='zxz_ex_active')
        r_ck = Rigid3D.make_r_ck(e)
        np_test.assert_almost_equal(r_ck, r_euler)

        # zxz_in_active
        angles = [-0.5, 1.2, 2.3]
        r_euler = Rigid3D.make_r_euler(angles, mode='zxz_in_active')
        e = Rigid3D.euler_to_ck(angles, mode='zxz_in_active')
        r_ck = Rigid3D.make_r_ck(e)
        np_test.assert_almost_equal(r_ck, r_euler)
        (e0, e1, e2, e3) = Rigid3D.euler_to_ck(angles, mode='zxz_ex_active')
        np_test.assert_almost_equal(e, (e0, e1, -e2, e3))

        # zyz_ex_active
        angles = [-0.5, 1.2, 2.3]
        r_euler = Rigid3D.make_r_euler(angles, mode='zyz_ex_active')
        e = Rigid3D.euler_to_ck(angles, mode='zyz_ex_active')
        r_ck = Rigid3D.make_r_ck(e)
        np_test.assert_almost_equal(r_ck, r_euler)
        (e0, e1, e2, e3) = Rigid3D.euler_to_ck(angles, mode='zxz_ex_active')
        np_test.assert_almost_equal(e, (e0, -e2, e1, e3))

        # zyz_in_active
        angles = [-0.5, 1.2, 2.3]
        r_euler = Rigid3D.make_r_euler(angles, mode='zyz_in_active')
        e = Rigid3D.euler_to_ck(angles, mode='zyz_in_active')
        r_ck = Rigid3D.make_r_ck(e)
        np_test.assert_almost_equal(r_ck, r_euler)
        (e0, e1, e2, e3) = Rigid3D.euler_to_ck(angles, mode='zxz_ex_active')
        np_test.assert_almost_equal(e, (e0, e2, e1, e3))

    def test_make_random_ck(self):
        """
        Tests make_random_ck()
        """

        # repeat few times because random
        for ind in range(10):

            # no center
            res = Rigid3D.make_random_ck(center=None, distance=None)
            np_test.assert_almost_equal(np.square(res).sum(), 1)

            # center e_0 = 1
            distance = 0.2
            res = Rigid3D.make_random_ck(
                center=[1., 0, 0, 0], distance=distance)
            np_test.assert_almost_equal(np.square(res).sum(), 1)
            np_test.assert_equal((res[1:] <= distance).all(), True)

            # arbitrary center
            distance = 0.1
            center = [0.5, -0.7, 0.2, -np.sqrt(0.22)] 
            res = Rigid3D.make_random_ck(center=center, distance=distance)
            np_test.assert_almost_equal(np.square(res).sum(), 1)
            r_center = Rigid3D.make_r_ck(center)
            r_res = Rigid3D.make_r_ck(res)
            r_rt_res = np.dot(r_center.transpose(), r_res)
            rt_res_euler = Rigid3D.extract_euler(
                r=r_rt_res, ret='one', mode='x')
            rt_res = Rigid3D.euler_to_ck(rt_res_euler)
            np_test.assert_almost_equal(np.square(rt_res).sum(), 1)
            np_test.assert_equal((rt_res[1:] <= distance).all(), True)

    def test_transform(self):
        """
        Tests transform()
        """

        # coord system-like points and params
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        r = Rigid3D.make_r_euler(angles, mode='x')
        s = 23.
        d = [1., 2, 3]

        # dim_point
        rigid3d = Rigid3D()
        rigid3d.q = r
        rigid3d.s_scalar = s
        y_desired = s * np.dot(r, x_cs)
        y = rigid3d.transform(x=x_cs)
        np_test.assert_almost_equal(y, y_desired)
        
        # xy_axes=dim_point
        rigid3d = Rigid3D()
        rigid3d.q = r
        rigid3d.s_scalar = s
        rigid3d.d = d
        y_desired = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        y = rigid3d.transform(x=x_cs)
        np_test.assert_almost_equal(y, y_desired)
        
        # xy_axes=dim_point
        rigid3d = Rigid3D()
        rigid3d.s_scalar = s
        y_desired = s * np.dot(r, x_cs)
        y = rigid3d.transform(x=x_cs, q=r)
        np_test.assert_almost_equal(y, y_desired)
        
        # xy_axes=dim_point
        rigid3d = Rigid3D()
        y_desired = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        y = rigid3d.transform(x=x_cs, q=r, s=s, d=d)
        np_test.assert_almost_equal(y, y_desired)
        
        # xy_axes=dim_point, xy_axes='point_dim'
        rigid3d = Rigid3D()
        rigid3d.q = r
        rigid3d.s_scalar = s
        rigid3d.d = d
        y_desired = s * np.inner(x_cs.transpose(), r) + d
        y = rigid3d.transform(x=x_cs.transpose(), xy_axes='point_dim')
        np_test.assert_almost_equal(y, y_desired)
        
        # d=None, xy_axes='point_dim'
        rigid3d = Rigid3D()
        y_desired = s * np.inner(x_cs.transpose(), r)
        y = rigid3d.transform(x=x_cs.transpose(), q=r, s=s, xy_axes='point_dim')
        np_test.assert_almost_equal(y, y_desired)
        
    def test_recalculate_translation(self):
        """
        Tests recalculate_translation()
        """

        # no initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 2.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        #r3d.d = np.array([1,2,0])
        center = np.array([0,1,0]).reshape(3,1)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-2, -2, 0]).reshape((3,1)))

        # with initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 2.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        center = np.array([0,1,0]).reshape(3,1)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-1, 1, 1]).reshape((3,1)))

        # center point_dim form, with initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 2.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([-1,3,1])
        center = np.array([0,1,0]).reshape(1,3)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-3, 1, 1]).reshape((1,3)))

        # center 1d form, with initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 2.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        center = np.array([0,1,0]).reshape(1,3)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-1, 1, 1]).reshape((1,3)))

        # another example with initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 3.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        center = np.array([2,0,0]).reshape(3,1)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-5, 9, 1]).reshape((3,1)))

        # mimick rotation around another center
        r3d = Rigid3D()
        r3d.s_scalar = 3.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        x=np.array([-1, 2, 1]).reshape((3,1))
        y_desired = r3d.transform(x=x)
        center = np.array([2,0,0]).reshape(3,1)
        y_actual = (
            r3d.s_scalar * (np.dot(r3d.q, x-center) + center) 
            + r3d.recalculate_translation(rotation_center=center))
        np_test.assert_almost_equal(y_actual, y_desired)
 
        # mimick rotation around another center, more complicated
        r3d = Rigid3D()
        r3d.s_scalar = 2.5
        angles = np.array([40, 67, -89]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        x=np.array([-1, 2, 1]).reshape((3,1))
        y_desired = r3d.transform(x=x)
        center = np.array([3,2,1]).reshape(3,1)
        y_actual = (
            r3d.s_scalar * (np.dot(r3d.q, x-center) + center) 
            + r3d.recalculate_translation(rotation_center=center))
        np_test.assert_almost_equal(y_actual, y_desired)

    def testTransformArrayRigid3D(self):
        """
        Tests transformArray(). This function is implemented in Affine,
        but the tests here pertain to 3D rigid transformations.
        """

        # array: up in +z, end in +y 
        gg = np.zeros((10,10,10))
        gg[4,2:7,2:7] = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,0],
             [1,2,3,4,5],
             [0,0,0,0,6],
             [0,0,0,0,0]])
        pi = np.pi

        # no rotation, center at 0
        r3d = Rigid3D(q=np.identity(3))
        transd = r3d.transformArray(array=gg, center=[0,0,0])
        np_test.assert_almost_equal(transd, gg)

        # no rotation, center at something
        r3d = Rigid3D(q=np.identity(3))
        transd = r3d.transformArray(array=gg, center=[2,3,4])
        np_test.assert_almost_equal(transd[4,2:7,2:7], gg[4,2:7,2:7])

        # phi = pi/2, zxz_ex_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[pi/2, 0, 0], mode='zxz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,6],
             [1,2,3,4,5],
             [0,0,0,0,0],
             [0,0,0,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,4,2:7], desired)

        # psi = pi/2, zxz_ex_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[0, 0, pi/2], mode='zxz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,6],
             [1,2,3,4,5],
             [0,0,0,0,0],
             [0,0,0,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,4,2:7], desired)

        # theta = pi/2, zxz_ex_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[0, pi/2, 0], mode='zxz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,5,6,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[4,2:7,2:7], desired)

        # phi=pi/2, theta = pi/2, zxz_ex_active
        # desired: -y and in -x
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[pi/2, pi/2, 0], mode='zxz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,0,0,0],
             [6,0,0,0,0],
             [5,4,3,2,1],
             [0,0,0,0,0],
             [0,0,0,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,2:7,4], desired)

        # phi=pi/2, theta = -pi/2, zxz_ex_active
        # desired: +y and in -x
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[pi/2, -pi/2, 0], mode='zxz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,6],
             [1,2,3,4,5],
             [0,0,0,0,0],
             [0,0,0,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,2:7,4], desired)

        # theta = pi/2, psi = pi/2 zxz_ex_active
        # desired: +x and in +z
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[0, pi/2, pi/2], mode='zxz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,1,0,0],
             [0,0,2,0,0],
             [0,0,3,0,0],
             [0,0,4,0,0],
             [0,0,5,6,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,4,2:7], desired)

        # theta = -pi/2, psi = pi/2 zxz_ex_active
        # desired: -x and in -z
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[0, -pi/2, pi/2], mode='zxz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,6,5,0,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,4,2:7], desired)

        # psi=pi/2, theta = -pi/2, zxz_in_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[0, -pi/2, pi/2], mode='zxz_in_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,6],
             [1,2,3,4,5],
             [0,0,0,0,0],
             [0,0,0,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,2:7,4], desired)

        # phi=pi/2, theta = -pi/2, zyz_ex_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[pi/2, -pi/2, 0], mode='zyz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,6,5,0,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,4,2:7], desired)

        # psi=pi/2, theta = -pi/2, zyz_in_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[0, -pi/2, pi/2], mode='zyz_in_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,6,5,0,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,4,2:7], desired)

        # psi=-pi/2, theta = pi/2, zxz_ex_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[0, pi/2, -pi/2], mode='zxz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,5,6,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,4,2:7], desired)

        # phi=-pi/2, theta = pi/2, zxz_in_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(angles=[-pi/2, pi/2, 0], mode='zxz_in_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,5,6,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,4,2:7], desired)

        # phi=-pi/2, theta = -pi/2, psi=pi/2, zxz_ex_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(
            angles=[-pi/2, -pi/2, pi/2], mode='zxz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,5,6,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,2:7,4], desired)

        # phi=pi/2, theta = -pi/2, psi=-pi/2, zxz_in_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(
            angles=[pi/2, -pi/2, -pi/2], mode='zxz_in_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,5,6,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[2:7,2:7,4], desired)

        # phi=-pi/2, theta = -pi/2, psi=pi/2, zyz_ex_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(
            angles=[-pi/2, -pi/2, pi/2], mode='zyz_ex_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,5,6,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[4,2:7,2:7], desired)

        # phi=pi/2, theta = -pi/2, psi=-pi/2, zyz_in_active
        r3d = Rigid3D()
        r3d.q = r3d.make_r_euler(
            angles=[pi/2, -pi/2, -pi/2], mode='zyz_in_active')
        r3d.s_scalar = 1
        desired = np.array(
            [[0,0,5,6,0],
             [0,0,4,0,0],
             [0,0,3,0,0],
             [0,0,2,0,0],
             [0,0,1,0,0]])
        transd = r3d.transformArray(array=gg, center=[4,4,4])
        np_test.assert_almost_equal(transd[4,4,4], 3)
        np_test.assert_almost_equal(transd[4,2:7,2:7], desired)

    def testShiftCenter(self):
        """
        Tests Affine.shiftCenter() and indirectly Affine.resetCenter()
        and transform()
        """

        # rotation and scale
        center = [1, 2, 0]
        q = Rigid3D.make_r_euler(angles=(np.pi/2, 0, 0))
        ct_1 = Rigid3D(q=q, scale=5, d=[3,-1,2])
        ct_or = ct_1.shiftCenter(center=center)
        np_test.assert_almost_equal(ct_or.center, center)
        np_test.assert_almost_equal(ct_or.gl, ct_1.gl)
        np_test.assert_almost_equal(ct_or.d, [-8, 2, 2])

        # check shift_center and reset_center are complementary
        ct_2 = ct_or.resetCenter(center=center)
        np_test.assert_almost_equal(ct_2.gl, ct_1.gl)
        np_test.assert_almost_equal(ct_2.d, ct_1.d)

        # transform points
        points = [[1,2,3], [-2.3,1, 4.9], [-3.4,-2,3]]
        p1 = ct_1.transform(points, xy_axes='point_dim')
        por = ct_or.transform(points, center=center, xy_axes='point_dim')
        p2 = ct_2.transform(points, xy_axes='point_dim')
        np_test.assert_almost_equal(p1, por)
        np_test.assert_almost_equal(p2, por)
        
        # genaral transformation
        q = Rigid3D.make_r_euler(angles=(-np.pi/2, np.pi/6, np.pi/3))
        center = [1.5, -3.7, 3.2]
        ct_1 = Rigid3D(q=q, scale=3.4, d=[-2.3, 5.1, 2.1])
        ct_or = ct_1.shift_center(center=center)
        ct_2 = ct_or.reset_center(center=center)
        np_test.assert_almost_equal(ct_or.center, center)
        np_test.assert_almost_equal(ct_or.gl, ct_1.gl)
        np_test.assert_almost_equal(ct_or.gl, ct_2.gl)

        # transform points
        points = [[0,2.1,0.3], [2.4,-4.1,4.9], [-3.2,-5,3.5]]
        p1 = ct_1.transform(points, xy_axes='point_dim')
        por = ct_or.transform(points, center=center, xy_axes='point_dim')
        p2 = ct_2.transform(points, xy_axes='point_dim')
        np_test.assert_almost_equal(p1, por)
        np_test.assert_almost_equal(p2, por)
        
    def testResetCenter(self):
        """
        Tests Affine.resetCenter() and indirectly Affine.shiftCenter()
        and transform()
        """

        # rotation and scale
        center = [1, 2, 0]
        q = Rigid3D.make_r_euler(angles=(np.pi/2, 0, 0))
        ct_or = Rigid3D(q=q, scale=5, d=[-8,2,2])
        ct = ct_or.resetCenter(center=center)
        np_test.assert_almost_equal(
            (ct.center==None) or (ct.center==0), True)
        np_test.assert_almost_equal(ct_or.gl, ct.gl)
        np_test.assert_almost_equal(ct.d, [3, -1, 2])

        # check shift_center and reset_center are complementary
        ct_or_2 = ct.shiftCenter(center=center)
        np_test.assert_almost_equal(ct_or_2.gl, ct_or.gl)
        np_test.assert_almost_equal(ct_or_2.d, ct_or.d)
        
        # transform points
        points = [[5,-2,3.4], [-4.3,1.2, -9], [-3.8,-2.3,-1]]
        por = ct_or.transform(points, center=center, xy_axes='point_dim')
        pp = ct.transform(points, xy_axes='point_dim')
        por2 = ct_or_2.transform(points, center=center, xy_axes='point_dim')
        np_test.assert_almost_equal(pp, por)
        np_test.assert_almost_equal(por2, por)
        
        # genaral transformation
        q = Rigid3D.make_r_euler(angles=(-np.pi/2, np.pi/6, np.pi/3))
        center = [1.5, -3.7, 3.2]
        ct_or = Rigid3D(q=q, scale=3.4, d=[-2.3, 5.1, 2.1])
        ct = ct_or.reset_center(center=center)
        ct_or_2 = ct.shift_center(center=center)
        np_test.assert_almost_equal(ct_or_2.center, center)
        np_test.assert_almost_equal(ct_or.gl, ct.gl)
        np_test.assert_almost_equal(ct_or.gl, ct_or_2.gl)

        # transform points
        points = [[0,2.1,0.3], [2.4,-4.1,4.9], [-3.2,-5,3.5]]
        por = ct_or.transform(points, center=center, xy_axes='point_dim')
        pp = ct.transform(points, xy_axes='point_dim')
        por2 = ct_or_2.transform(points, center=center, xy_axes='point_dim')
        np_test.assert_almost_equal(pp, por)
        np_test.assert_almost_equal(por2, por)
        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRigid3D)
    unittest.TextTestRunner(verbosity=2).run(suite)


