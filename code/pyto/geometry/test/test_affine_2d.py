"""

Tests module affine_2d

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
#from past.utils import old_div

__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest

import numpy as np
import numpy.testing as np_test 
import scipy

from pyto.geometry.affine_2d import Affine2D

class TestAffine2D(np_test.TestCase):
    """
    """

    def setUp(self):

        # basic
        self.x0 = np.array([[1, 0.], [0, 1], [-1, 0]])
        self.y0_0 = 2 * self.x0
        self.y0_90 = 2 * np.array([[0, 1], [-1, 0], [0, -1]])
        self.y0_180 = 2 * np.array([[-1, 0.], [0, -1], [1, 0]])
        self.y0_270 = 2 * np.array([[0, -1], [1, 0], [0, 1]])

        # parallelogram, rotation, scale, exact
        self.d1 = [-1, 2]
        self.x1 = np.array([[0., 0], [2, 0], [2, 1], [0, 1]])
        self.y1 = np.array([[0., 0], [4, 2], [3, 4], [-1, 2]]) + self.d1
        self.y1m = np.array([[0., 0], [-4, 2], [-3, 4], [1, 2]]) + self.d1

        # parallelogram, rotation, scale, not exact
        self.d2 = [-1, 2]
        self.x2 = np.array([[0.1, -0.2], [2.2, 0.1], [1.9, 0.8], [0.2, 1.1]])
        self.y2 = np.array([[0., 0], [4, 2], [3, 4], [-1, 2]]) + self.d2
        self.y2m = np.array([[0., 0], [-4, 2], [-3, 4], [1, 2]]) + self.d2

        # transformations
        self.af1 = Affine2D.find(x=self.x1, y=self.y1)
        self.af1_gl = np.array([[2,-1],[1,2]])
        self.af1_d = self.d1
        self.af1_phi = np.arctan(0.5)
        self.af1_scale = np.array([np.sqrt(5)] * 2)
        self.af1_parity = 1
        self.af1_shear = 0

        self.af1m = Affine2D.find(x=self.x1, y=self.y1m)
        self.af1m_gl = np.array([[-2,1],[1,2]])
        self.af1m_d = self.af1_d
        self.af1m_phi = np.pi - self.af1_phi
        self.af1m_q = self.af1m.makeQ(phi=self.af1m_phi)
        self.af1m_scale = np.array([np.sqrt(5)] * 2)
        self.af1m_parity = -1
        self.af1m_shear = 0

        self.af2 = Affine2D.find(x=self.x2, y=self.y2)
        self.af2_d = [-1.42584884,  2.05326245]
        self.af2_gl = np.array([[2.09463865, -0.84056372],
                                 [ 1.00406239,  1.87170871]])
        self.af2_phi = 0.446990530695
        self.af2_scale = np.array([2.32285435, 2.05115392])

        self.af2m = Affine2D.find(x=self.x2, y=self.y2m)

        # L-shaped u, scale, v_angle=0
        self.x3 = np.array([[3,0], [2,0], [1,0], [1, -1]])
        self.y3 = np.array([[-1,2], [-1,1.5], [-1,1], [1, 1]])
        self.af3 = Affine2D.find(x=self.x3, y=self.y3)
        self.af3.decompose(order='usv')
        self.af3_uAngleDeg = 0
        self.af3_vAngleDeg = 90
        self.af3_scale = [2, 0.5]
        self.af3_d = [-1, 0.5]

    def testFindGL(self):
        """
        Tests find (transform 'gl'), decompose individual parameters and
        transform.
        """

        ##################################################
        #
        # parallelogram, rotation, scale, exact
        #

        #aff2d = Affine2D.find(x=self.x1, y=self.y1)
        np_test.assert_almost_equal(self.af1.d, self.af1_d)
        np_test.assert_almost_equal(self.af1.gl, self.af1_gl)

        # xy_axis = 'dim_point'
        aff2d_xy = Affine2D.find(
            x=self.x1.transpose(), y=self.y1.transpose(), xy_axes='dim_point')
        np_test.assert_almost_equal(aff2d_xy.d, self.af1_d)
        np_test.assert_almost_equal(aff2d_xy.gl, self.af1_gl)

        # test decompose
        self.af1.decompose(order='qpsm') 
        np_test.assert_almost_equal(self.af1.phi, self.af1_phi)
        desired_q = np.array(\
            [[np.cos(self.af1_phi), -np.sin(self.af1_phi)],
             [np.sin(self.af1_phi), np.cos(self.af1_phi)]])
        np_test.assert_almost_equal(self.af1.q, desired_q)
        np_test.assert_almost_equal(self.af1.p, np.diag([1, 1]))
        np_test.assert_almost_equal(self.af1.s, 
                                    self.af1_scale * np.diag([1,1]))
        np_test.assert_almost_equal(self.af1.m, np.diag([1, 1]))

        # test parameters
        np_test.assert_almost_equal(self.af1.scale, self.af1_scale)
        np_test.assert_almost_equal(self.af1.phi, self.af1_phi)
        np_test.assert_almost_equal(self.af1.parity, self.af1_parity)
        np_test.assert_almost_equal(self.af1.shear, self.af1_shear)
        
        # test transformation and error
        y1_calc = self.af1.transform(self.x1)
        np_test.assert_almost_equal(y1_calc, self.y1)
        np_test.assert_almost_equal(self.af1.error, np.zeros_like(self.y1))
        np_test.assert_almost_equal(self.af1.rmsError, 0)

        #################################################
        #
        # parallelogram, scale, rotation, parity, exact
        #

        # test parameters
        #aff2d = Affine2D.find(x=self.x1, y=self.y1m)
        np_test.assert_almost_equal(self.af1m.d, self.af1m_d)
        np_test.assert_almost_equal(self.af1m.gl, self.af1m_gl)
        np_test.assert_almost_equal(self.af1m.scale, self.af1m_scale)
        np_test.assert_almost_equal(self.af1m.phi, self.af1m_phi)
        #np_test.assert_almost_equal(self.af1m.phiDeg, 
        #                            180 - desired_phi * 180 / np.pi)
        np_test.assert_almost_equal(self.af1m.parity, self.af1m_parity)
        np_test.assert_almost_equal(self.af1m.shear, self.af1m_shear)

        # test transformation and error
        y1_calc = self.af1m.transform(self.x1, gl=self.af1m.gl, d=self.af1m.d)
        np_test.assert_almost_equal(y1_calc, self.y1m)
        np_test.assert_almost_equal(self.af1m.error, np.zeros_like(self.y1))
        np_test.assert_almost_equal(self.af1m.rmsError, 0)

        # xy_axis = 'dim_point'
        af1m_xy = Affine2D.find(
            x=self.x1.transpose(), y=self.y1m.transpose(), xy_axes='dim_point')
        np_test.assert_almost_equal(af1m_xy.d, self.af1m_d)
        np_test.assert_almost_equal(af1m_xy.gl, self.af1m_gl)
        np_test.assert_almost_equal(af1m_xy.scale, self.af1m_scale)
        np_test.assert_almost_equal(af1m_xy.phi, self.af1m_phi)
        np_test.assert_almost_equal(af1m_xy.parity, self.af1m_parity)
        np_test.assert_almost_equal(af1m_xy.shear, self.af1m_shear)

        ##################################################
        #
        # same as above but rq order
        #

        # test parameters
        q, p, s, m = self.af1m.decompose(gl=self.af1m.gl, order='rq')
        q_new = np.dot(np.dot(p, self.af1m_q), p) 
        np_test.assert_almost_equal(q, q_new)
        np_test.assert_almost_equal(p, self.af1m.makeP(parity=self.af1m_parity))
        np_test.assert_almost_equal(s, self.af1m.makeS(self.af1m_scale))
        np_test.assert_almost_equal(m, self.af1m.makeM(self.af1m_shear))

        # test transformation 
        psmq = np.dot(np.dot(p, s), np.dot(m, q))
        y_new = np.inner(self.x1, psmq) + self.af1m.d
        np_test.assert_almost_equal(y_new, self.y1m)

        ##################################################
        #
        # parallelogram, rotation, scale, not exact
        #

        aff2d = Affine2D.find(x=self.x2, y=self.y2)

        # test transformation matrices and parameters
        desired_d = [-1.42584884,  2.05326245]
        desired_gl = np.array([[2.09463865, -0.84056372],
                                 [ 1.00406239,  1.87170871]])
        desired_phi = 0.446990530695
        desired_scale = [2.32285435, 2.05115392]
        np_test.assert_almost_equal(aff2d.d, desired_d)
        np_test.assert_almost_equal(aff2d.gl, desired_gl)
        np_test.assert_almost_equal(aff2d.phi, desired_phi)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, 1)
        np_test.assert_almost_equal(aff2d.m, [[1, 0.02198716], [0, 1]])

        # test transform method
        y2_calc_gl = aff2d.transform(self.x2, gl=aff2d.gl, d=aff2d.d)
        qpsm = np.dot(np.dot(aff2d.q, aff2d.p), 
                         np.dot(aff2d.s, aff2d.m))
        y2_calc_qpsm = np.inner(self.x2, qpsm) + aff2d.d
        np_test.assert_almost_equal(y2_calc_gl, y2_calc_qpsm)
        #np_test.assert_almost_equal(y2_calc_gl, self.y2)

        ##################################################
        #
        # parallelogram, rotation, scale, not exact, xy_axes=dim_point
        #

        aff2d_xy = Affine2D.find(
            x=self.x2.transpose(), y=self.y2.transpose(), xy_axes='dim_point')

        # test transformation matrices and parameters
        desired_d = [-1.42584884,  2.05326245]
        desired_gl = np.array([[2.09463865, -0.84056372],
                                 [ 1.00406239,  1.87170871]])
        desired_phi = 0.446990530695
        desired_scale = [2.32285435, 2.05115392]
        np_test.assert_almost_equal(aff2d_xy.d, desired_d)
        np_test.assert_almost_equal(aff2d_xy.gl, desired_gl)
        np_test.assert_almost_equal(aff2d_xy.phi, desired_phi)
        np_test.assert_almost_equal(aff2d_xy.scale, desired_scale)
        np_test.assert_almost_equal(aff2d_xy.parity, 1)
        np_test.assert_almost_equal(aff2d_xy.m, [[1, 0.02198716], [0, 1]])

        # test transform method
        y2_calc_gl = aff2d.transform(
            self.x2.transpose(), gl=aff2d_xy.gl, d=aff2d_xy.d, 
            xy_axes='dim_point')
        qpsm = np.dot(np.dot(aff2d_xy.q, aff2d_xy.p), 
                         np.dot(aff2d_xy.s, aff2d_xy.m))
        y2_calc_qpsm = np.dot(
            qpsm, self.x2.transpose()) + np.expand_dims(aff2d_xy.d, 1)
        np_test.assert_almost_equal(y2_calc_gl, y2_calc_qpsm)
        #np_test.assert_almost_equal(y2_calc_gl, self.y2)

        ##################################################
        #
        # parallelogram, rotation, scale, parity, not exact
        #

        aff2d = Affine2D.find(x=self.x2, y=self.y2m)

        # test transformation matrices and parameters
        desired_d = [-0.57415116,  2.05326245]
        desired_gl = np.array([[-2.09463865, 0.84056372],
                                 [ 1.00406239,  1.87170871]])
        desired_phi = 0.446990530695
        desired_scale = [2.32285435, 2.05115392]
        np_test.assert_almost_equal(aff2d.d, desired_d)
        np_test.assert_almost_equal(aff2d.gl, desired_gl)
        np_test.assert_almost_equal(aff2d.phi, np.pi - desired_phi)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, -1)
        np_test.assert_almost_equal(aff2d.m, [[1, 0.02198716], [0, 1]])

        # test transform method
        y2m_calc_gl = aff2d.transform(self.x2)
        qpsm = np.dot(np.dot(aff2d.q, aff2d.p), 
                         np.dot(aff2d.s, aff2d.m))
        np_test.assert_almost_equal(qpsm, aff2d.gl)
        y2m_calc_qpsm = np.inner(self.x2, qpsm) + aff2d.d
        np_test.assert_almost_equal(y2m_calc_gl, y2m_calc_qpsm)
        #np_test.assert_almost_equal(y2_calc_gl, self.y2m)

        ##################################################
        #
        # parallelogram, rotation, scale, parity, not exact, xy_axes='dim_point'
        #

        aff2d = Affine2D.find(
            x=self.x2.transpose(), y=self.y2m.transpose(), xy_axes='dim_point')

        # test transformation matrices and parameters
        desired_d = [-0.57415116,  2.05326245]
        desired_gl = np.array([[-2.09463865, 0.84056372],
                                 [ 1.00406239,  1.87170871]])
        desired_phi = 0.446990530695
        desired_scale = [2.32285435, 2.05115392]
        np_test.assert_almost_equal(aff2d.d, desired_d)
        np_test.assert_almost_equal(aff2d.gl, desired_gl)
        np_test.assert_almost_equal(aff2d.phi, np.pi - desired_phi)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, -1)
        np_test.assert_almost_equal(aff2d.m, [[1, 0.02198716], [0, 1]])

        # test transform method
        y2m_calc_gl = aff2d.transform(self.x2.transpose())
        qpsm = np.dot(np.dot(aff2d.q, aff2d.p), 
                         np.dot(aff2d.s, aff2d.m))
        np_test.assert_almost_equal(qpsm, aff2d.gl)
        y2m_calc_qpsm = (np.dot(qpsm, self.x2.transpose()) 
                         + np.expand_dims(aff2d.d, 1))
        np_test.assert_almost_equal(y2m_calc_gl, y2m_calc_qpsm)
        #np_test.assert_almost_equal(y2_calc_gl, self.y2m)

        ##################################################
        #
        # L-shape: rotation, scale; check usv 
        #
        af3 = Affine2D.find(x=self.x3, y=self.y3)
        af3.decompose(order='usv')
        np_test.assert_almost_equal(af3.vAngleDeg, 90)
        np_test.assert_almost_equal(af3.uAngleDeg, 0)
        np_test.assert_almost_equal(af3.scale, [2, 0.5])
        np_test.assert_almost_equal(af3.scaleAngle, np.arccos(0.25))
        np_test.assert_almost_equal(af3.d, self.af3_d)

    def testFindRS(self):
        """
        Tests find (transform 'rs'), decompose individual parameters and
        transform.
        """

        ###############################################
        #
        # parallelogram, rotation, scale, exact
        #

        aff2d = Affine2D.find(x=self.x1, y=self.y1, type_='rs')
        np_test.assert_almost_equal(aff2d.d, self.d1)

        # test finding transformation
        desired_phi = np.arctan(0.5)
        desired_scale = [np.sqrt(5)] * 2
        desired_q = np.array(\
            [[np.cos(desired_phi), -np.sin(desired_phi)],
             [np.sin(desired_phi), np.cos(desired_phi)]])
        np_test.assert_almost_equal(aff2d.parity, 1)
        #np_test.assert_almost_equal(aff2d.phi, desired_phi)
        #np_test.assert_almost_equal(aff2d.q, desired_q)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.error, np.zeros_like(self.y1))

        # test doing transformation
        y1_calc = aff2d.transform(self.x1)
        np_test.assert_almost_equal(y1_calc, self.y1)
        qpsm = np.dot(np.dot(aff2d.q, aff2d.p), 
                         np.dot(aff2d.s, aff2d.m))
        y_new = np.inner(self.x1, qpsm) + aff2d.d
        np_test.assert_almost_equal(y_new, self.y1)
        
        ###############################################
        #
        # parallelogram, rotation, scale, parity, exact
        #

        aff2d = Affine2D.find(x=self.x1, y=self.y1m, type_='rs')
        np_test.assert_almost_equal(aff2d.d, self.d1)

        # test finding transformation
        desired_phi = np.arctan(0.5)
        desired_scale = [np.sqrt(5)] * 2
        desired_q = np.array(\
            [[-np.cos(desired_phi), -np.sin(desired_phi)],
             [np.sin(desired_phi), -np.cos(desired_phi)]])
        np_test.assert_almost_equal(aff2d.phi, np.pi - desired_phi)
        np_test.assert_almost_equal(aff2d.q, desired_q)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, -1)
        np_test.assert_almost_equal(aff2d.error, np.zeros_like(self.y1))

        # test doing transformation
        y1_calc = aff2d.transform(self.x1)
        np_test.assert_almost_equal(y1_calc, self.y1m)
        qpsm = np.dot(np.dot(aff2d.q, aff2d.p), 
                         np.dot(aff2d.s, aff2d.m))
        y_new = np.inner(self.x1, qpsm) + aff2d.d
        np_test.assert_almost_equal(y_new, self.y1m)
        
        ###############################################
        #
        # parallelogram, rotation, scale, parity, exact, xy_axes='dim_point'
        #

        aff2d = Affine2D.find(
            x=self.x1.transpose(), y=self.y1m.transpose(), type_='rs', 
            xy_axes='dim_point')
        np_test.assert_almost_equal(aff2d.d, self.d1)

        # test finding transformation
        desired_phi = np.arctan(0.5)
        desired_scale = [np.sqrt(5)] * 2
        desired_q = np.array(\
            [[-np.cos(desired_phi), -np.sin(desired_phi)],
             [np.sin(desired_phi), -np.cos(desired_phi)]])
        np_test.assert_almost_equal(aff2d.phi, np.pi - desired_phi)
        np_test.assert_almost_equal(aff2d.q, desired_q)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, -1)
        np_test.assert_almost_equal(
            aff2d.error, np.zeros_like(self.y1.transpose()))

        # test doing transformation
        y1_calc = aff2d.transform(self.x1.transpose())
        np_test.assert_almost_equal(y1_calc, self.y1m.transpose())
        qpsm = np.dot(np.dot(aff2d.q, aff2d.p), 
                         np.dot(aff2d.s, aff2d.m))
        y_new = (np.dot(qpsm, self.x1.transpose()) 
                 + np.expand_dims(aff2d.d, 1))
        np_test.assert_almost_equal(y_new, self.y1m.transpose())
        
        ###############################################
        #
        # parallelogram, rotation, scale, parity not exact
        #

        af2m = Affine2D.find(x=self.x2, y=self.y2m)
        af2mrs = Affine2D.find(x=self.x2, y=self.y2m, type_='rs')
 
        # test finding transformation
        desired_d = [-0.57415116,  2.05326245]
        desired_phi = np.pi - 0.442817288965
        desired_scale = [2.18278075] * 2
        desired_q = np.array(\
            [[np.cos(desired_phi), -np.sin(desired_phi)],
             [np.sin(desired_phi), np.cos(desired_phi)]])
        #np_test.assert_almost_equal(af2mrs.d, desired_d)
        np_test.assert_almost_equal(af2mrs.phi, desired_phi)
        np_test.assert_almost_equal(af2mrs.scale, desired_scale)
        np_test.assert_almost_equal(af2mrs.parity, -1)

        # compare with gl
        #np_test.assert_almost_equal(af2mrs.d, af2m.d)
        np_test.assert_almost_equal(af2mrs.scale, af2m.scale, decimal=1)
        np_test.assert_almost_equal(af2mrs.phi, af2m.phi, decimal=2)
        np_test.assert_almost_equal(af2mrs.parity, af2m.parity)
        np_test.assert_almost_equal(af2mrs.error, af2m.error, decimal=0)
        np_test.assert_almost_equal(af2mrs.rmsError, af2m.rmsError, decimal=1)

        # test doing transformation
        y2_calc = af2mrs.transform(self.x2)
        qpsm = np.dot(np.dot(af2mrs.q, af2mrs.p), 
                         np.dot(af2mrs.s, af2mrs.m))
        np_test.assert_almost_equal(qpsm, af2mrs.gl)
        y2_calc_qpsm = np.inner(self.x2, qpsm) + af2mrs.d
        np_test.assert_almost_equal(y2_calc, y2_calc_qpsm)
        
        ###############################################
        #
        # parallelogram, rotation, scale, parity not exact, xy_axes='dim_point'
        #

        af2m = Affine2D.find(x=self.x2.transpose(), y=self.y2m.transpose())
        af2mrs = Affine2D.find(
            x=self.x2.transpose(), y=self.y2m.transpose(), type_='rs',
            xy_axes='dim_point')
 
        # test finding transformation
        desired_phi = np.pi - 0.442817288965
        desired_scale = [2.18278075] * 2
        desired_q = np.array(\
            [[np.cos(desired_phi), -np.sin(desired_phi)],
             [np.sin(desired_phi), np.cos(desired_phi)]])
        np_test.assert_almost_equal(af2mrs.phi, desired_phi)
        np_test.assert_almost_equal(af2mrs.scale, desired_scale)
        np_test.assert_almost_equal(af2mrs.parity, -1)

    def testInverse(self):
        """
        Tests inverse
        """
        ###############################################
        #
        # parallelogram, rotation, scale not exact
        #

        af2 = Affine2D.find(x=self.x2, y=self.y2)
        af2rs = Affine2D.find(x=self.x2, y=self.y2, type_='rs')
        af2rsi = Affine2D.find(y=self.x2, x=self.y2, type_='rs')
        af2rs_inv = af2rs.inverse()
        af2rs_inv.decompose(order='qpsm')

        # tests inverse method
        np_test.assert_almost_equal(af2rs_inv.phi, -af2rs.phi)
        np_test.assert_almost_equal(af2rs_inv.scale, 1/af2rs.scale)
        np_test.assert_almost_equal(af2rs_inv.parity, af2rs.parity)

        # tests inversed x and y
        np_test.assert_almost_equal(af2rsi.phi, -af2rs.phi)
        np_test.assert_almost_equal(af2rsi.scale, 1/af2rs.scale, decimal=1)
        np_test.assert_almost_equal(af2rsi.parity, af2rs.parity)

        ###############################################
        #
        # parallelogram, rotation, scale, parity not exact
        #

        af2m = Affine2D.find(x=self.x2, y=self.y2m)
        af2mrs = Affine2D.find(x=self.x2, y=self.y2m, type_='rs')
        af2mrsi = Affine2D.find(y=self.x2, x=self.y2m, type_='rs')
        af2mrs_inv = af2mrs.inverse()
        af2mrs_inv.decompose(order='qpsm')
        
        # tests inverse method
        np_test.assert_almost_equal(af2mrs_inv.phi, af2mrs.phi)
        np_test.assert_almost_equal(af2mrs_inv.scale, 1/af2mrs.scale)
        np_test.assert_almost_equal(af2mrs_inv.parity, af2mrs.parity)

        # tests inversed x and y
        np_test.assert_almost_equal(af2mrsi.phi, af2mrs.phi)
        np_test.assert_almost_equal(af2mrsi.scale, 1/af2mrs.scale, decimal=1)
        np_test.assert_almost_equal(af2mrsi.parity, af2mrs.parity)

    def testCompose(self):
        """
        Tests compose
        """

        af11 = Affine2D.compose(self.af1, self.af1)
        af11.decompose(order='qpsm')
        np_test.assert_almost_equal(af11.phi, 2 * self.af1_phi)
        np_test.assert_almost_equal(af11.scale, 
                                    self.af1_scale * self.af1_scale)
        np_test.assert_almost_equal(af11.parity, 1)
        np_test.assert_almost_equal(af11.rmsErrorEst, 
                                    np.sqrt(2) * self.af1.error)

        af11m = Affine2D.compose(self.af1, self.af1m)
        af11m.decompose(order='qpsm')
        np_test.assert_almost_equal(
            np.mod(af11m.phi, 2*np.pi), self.af1_phi + self.af1m_phi)
        np_test.assert_almost_equal(af11m.scale, 
                                    self.af1_scale * self.af1m_scale)
        np_test.assert_almost_equal(af11m.parity, 
                                    self.af1_parity * self.af1m_parity)
        np_test.assert_almost_equal(af11m.rmsErrorEst, 
                                    np.sqrt(2) * self.af1.error)

        # test rms error
        af12 = Affine2D.compose(self.af1, self.af2)
        self.af1.decompose(order='qpsm')
        np_test.assert_almost_equal(af12.rmsErrorEst, 
                                    self.af1.scale[0] * self.af2.rmsError)
        af21 = Affine2D.compose(self.af2, self.af1)
        np_test.assert_almost_equal(af21.rmsErrorEst, self.af2.rmsError)

    def testTransformArray(self):
        """
        Tests transformArray(). This function is implemented in Affine,
        but the tests here pertain to 3D rigid2D affine transformations.
        """

        ar = np.arange(6).reshape(3,2)

        # translation, shape=None
        aff = Affine2D(gl=np.identity(2), d=(1,0))
        actual = aff.transformArray(array=ar)
        desired = np.array([[0, 0], [0, 1], [2, 3]])
        np_test.assert_almost_equal(actual, desired)
        
        # translation, shape specified
        aff = Affine2D(gl=np.identity(2), d=(1,0))
        actual = aff.transformArray(array=ar, shape=(4,2))
        desired = np.array([[0, 0], [0, 1], [2, 3], [4, 5]])
        np_test.assert_almost_equal(actual, desired)
        
        # pi/2 + translation, shape, data not on edges
        ar_in = np.arange(6).reshape(3,2)+1
        ar = np.zeros((5,4))
        ar[1:4, 1:3] = ar_in
        aff = Affine2D(phi=np.pi/2, scale=1, d=(3,0))
        actual = aff.transformArray(array=ar, shape=(6,6))
        desired_in = np.array([[2,4,6], [1,3,5]])
        np_test.assert_almost_equal(actual[1:3,1:4], desired_in)
        
        # pi/2 + translation, shape, data not on edges
        ar_in = np.arange(6).reshape(3,2)+1
        ar = np.zeros((5,4))
        ar[1:4, 1:3] = ar_in
        aff = Affine2D(phi=np.pi/2, scale=1, d=(5,1))
        actual = aff.transformArray(array=ar, shape=(6,6))
        desired_in = np.array([[2,4,6], [1,3,5]])
        np_test.assert_almost_equal(actual[3:5,2:5], desired_in)
        
    def testShiftCenter(self):
        """
        Tests Affine.shiftCenter() and indirectly Affine.resetCenter()
        and transform()
        """

        # check if center attribute not set or different from 0
        aff2 = Affine2D(phi=np.pi/2, scale=5, d=[-2, 1])
        aff2.center = [2, 3]
        np_test.assert_raises(ValueError, aff2.shiftCenter, center=[3,4])
        #with np_test.assert_raises(AttributeError): aff2.ffff
        #np_test.assert_raises(Exception, aff2.fffff)
        #np_test.assert_raises(ValueError, aff2.shiftCenter, [2,1])
        
        # rotation and scale
        center = [1, 2]
        aff2 = Affine2D(phi=np.pi/2, scale=5, d=[-2, 1])
        aff2_or = aff2.shiftCenter(center=center)
        np_test.assert_almost_equal(aff2_or.center, center)
        np_test.assert_almost_equal(aff2.gl, aff2_or.gl)
        np_test.assert_almost_equal(aff2_or.gl, [[0, -5], [5, 0]])
        np_test.assert_almost_equal(aff2_or.d, [-13, 4])
                                    
        # check shift_center and reset_center are complementary
        aff2_2 = aff2_or.resetCenter(center=center)
        np_test.assert_almost_equal(aff2_2.gl, aff2.gl)
        np_test.assert_almost_equal(aff2_2.d, aff2.d)

        # transform points
        points = np.array([[1, 2], [0, 0], [-2, -3]])
        points_1 = aff2.transform(x=points, xy_axes='point_dim')
        points_2 = aff2_or.transform(
            x=points, center=center, xy_axes='point_dim')
        np_test.assert_almost_equal(
            points_1, [[-12.,   6.], [ -2.,   1.], [ 13.,  -9.]])
        np_test.assert_almost_equal(points_1, points_2)

        # genaral transformation
        gl_full = [[1.2, -3.4], [2.6, -0.3]]
        center = [1.5, -3.4]
        aff2_full = Affine2D(phi=np.pi/2, scale=5, d=[-2.4, 3])
        aff2_full_or = aff2_full.shiftCenter(center=center)
        aff2_full_2 = aff2_full_or.resetCenter(center=center)
        points_1 = aff2_full.transform(x=points, xy_axes='point_dim')
        points_2 = aff2_full_or.transform(
            x=points, center=center, xy_axes='point_dim')
        points_3 = aff2_full_2.transform(x=points, xy_axes='point_dim')
        np_test.assert_almost_equal(points_1, points_2)
        np_test.assert_almost_equal(points_3, points_2)
        np_test.assert_almost_equal(aff2_full.gl, aff2_full_or.gl)
        np_test.assert_almost_equal(aff2_full_2.gl, aff2_full_or.gl)
        np_test.assert_almost_equal(aff2_full.d, aff2_full_2.d)

        # zero center
        center = [0, 0]
        aff2 = Affine2D(phi=np.pi/2, scale=5, d=[-2, 1])
        aff2_or = aff2.shiftCenter(center=center)
        np_test.assert_almost_equal(aff2.gl, aff2_or.gl)
        np_test.assert_almost_equal(aff2.d, aff2_or.d)
        
    def testResetCenter(self):
        """
        Tests Affine.resetCenter() and indirectly Affine.shiftCenter()
        and transform()
        """

        # rotation and scale
        center = [1, 2]
        aff2_or = Affine2D(phi=-np.pi/2, scale=3., d=[3, -1])
        aff2 = aff2_or.resetCenter(center=center)
        np_test.assert_almost_equal(
            (aff2.center==None) or (aff2.center==0), True)
        np_test.assert_almost_equal(aff2.gl, aff2_or.gl)
        np_test.assert_almost_equal(aff2.center, 0)
        np_test.assert_almost_equal(aff2.gl, [[0, 3], [-3, 0]])
        np_test.assert_almost_equal(aff2.d, [-2, 4])

        # arg center = Null
        aff2_or = Affine2D(phi=-np.pi/2, scale=3., d=[3, -1])
        aff2_or.center = center
        aff2 = aff2_or.resetCenter()
        np_test.assert_almost_equal(aff2.center, 0)
        np_test.assert_almost_equal(aff2.gl, aff2_or.gl)
        np_test.assert_almost_equal(aff2_or.gl, [[0, 3], [-3, 0]])
        np_test.assert_almost_equal(aff2.d, [-2, 4])

        # transform points
        points = np.array([[1, 2], [0, 0], [-2, -3]])
        points_1 = aff2.transform(x=points, xy_axes='point_dim')
        points_2 = aff2_or.transform(
            x=points, center=center, xy_axes='point_dim')
        np_test.assert_almost_equal(points_1, points_2)
        np_test.assert_almost_equal(
            points_1, [[4., 1], [-2, 4], [-11,  10]])

        # genaral transformation
        gl_full = [[-4.2, -3.1], [5.2, 2.3]]
        center=[1.5, -3.4]
        aff2_full_or = Affine2D(phi=np.pi/2, scale=5, d=[-2.4, 3])
        aff2_full = aff2_full_or.resetCenter(center=center)
        aff2_full_or_2 = aff2_full.shiftCenter(center=center)
        points_1 = aff2_full_or.transform(
            x=points, center=center, xy_axes='point_dim')
        points_2 = aff2_full.transform(x=points, xy_axes='point_dim')
        points_3 = aff2_full_or_2.transform(
            x=points, center=center, xy_axes='point_dim')
        np_test.assert_almost_equal(points_1, points_2)
        np_test.assert_almost_equal(points_3, points_2)
        np_test.assert_almost_equal(aff2_full.gl, aff2_full_or.gl)
        np_test.assert_almost_equal(aff2_full_or_2.gl, aff2_full_or.gl)
        np_test.assert_almost_equal(aff2_full_or.d, aff2_full_or_2.d)
         
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAffine2D)
    unittest.TextTestRunner(verbosity=2).run(suite)
