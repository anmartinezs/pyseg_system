"""

Tests module affine

# Author: Vladan Lucic
# $Id: test_affine.py 1430 2017-03-24 13:18:43Z vladan $
"""

__version__ = "$Revision: 1430 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.geometry.affine import Affine
from pyto.geometry.affine_2d import Affine2D
from pyto.geometry.rigid_3d import Rigid3D

class TestAffine(np_test.TestCase):
    """
    """

    def setUp(self):

        # parallelogram, rotation, scale, exact
        self.d1 = [-1, 2]
        self.x1 = numpy.array([[0., 0], [2, 0], [2, 1], [0, 1]])
        self.y1 = numpy.array([[0., 0], [4, 2], [3, 4], [-1, 2]]) + self.d1
        self.y1m = numpy.array([[0., 0], [-4, 2], [-3, 4], [1, 2]]) + self.d1

        # parallelogram, rotation, scale, not exact
        self.d2 = [-1, 2]
        self.x2 = numpy.array([[0.1, -0.2], [2.2, 0.1], [1.9, 0.8], [0.2, 1.1]])
        self.y2 = numpy.array([[0., 0], [4, 2], [3, 4], [-1, 2]]) + self.d2
        self.y2m = numpy.array([[0., 0], [-4, 2], [-3, 4], [1, 2]]) + self.d2

    def testIdentity(self):
        """
        Tests identity()
        """

        ndim = 3
        ident = Affine.identity(ndim=ndim)
        ident.decompose(order='qpsm')
        np_test.assert_almost_equal(ident.scale, numpy.ones(ndim))
        np_test.assert_almost_equal(ident.parity, 1)
        np_test.assert_almost_equal(ident.translation, numpy.zeros(shape=ndim))
        np_test.assert_almost_equal(ident.gl, numpy.identity(ndim))

    def testScale(self):
        """
        Tests getScale and setScale
        """

        af1m_desired = Affine.find(x=self.x1, y=self.y1m)
        af1m_changed = Affine.find(x=self.x1, y=self.y1m)
        af1m_changed.scale = [1, 2]
        np_test.assert_almost_equal(af1m_changed.s, [[1, 0], [0, 2]])
        np_test.assert_almost_equal(af1m_changed.scale, [1,2])
        np_test.assert_almost_equal(af1m_changed.q, af1m_desired.q)
        np_test.assert_almost_equal(af1m_changed.p, af1m_desired.p)
        np_test.assert_almost_equal(af1m_changed.m, af1m_desired.m)
        np_test.assert_almost_equal(af1m_changed.d, af1m_desired.d)

    def testMakeD(self):
        """
        Tests makeD()
        """
        
        af = Affine()
        np_test.assert_almost_equal(af.makeD(d=None), 0)
        np_test.assert_almost_equal(af.makeD(d=None, ndim=4), [0,0,0,0])
        np_test.assert_almost_equal(af.makeD(d=3, ndim=2), [3,3])

    def testFind(self):
        """
        Tests find() method
        """

        af = Affine.find(x=self.x1, y=self.y1m)
        desired = numpy.inner(self.x1, af.gl) + af.d
        np_test.assert_almost_equal(self.y1m, desired)
        
    def testFindTranslation(self):
        """
        Tests findTranslation()
        """

        af = Affine.findTranslation(x=numpy.array([[1,2,3], [2,3,4]]),
                                    y=numpy.array([[2,4,6], [3,6,9]]))
        
        np_test.assert_almost_equal(af.translation, [1., 2.5, 4.])
        af.decompose(order='qpsm')
        np_test.assert_almost_equal(af.scale, numpy.ones(3))
        np_test.assert_almost_equal(af.parity, 1)
        np_test.assert_almost_equal(af.gl, numpy.identity(3))

    def testFindTwoStep(self):
        """
        Tests findTwoStep()
        """

        # parallelogram, rotation, scale, exact
        af = Affine.findTwoStep(x=self.x1[0:1], y=self.y1[0:1],
                                x_gl=self.x1, y_gl=self.y1+3)
        af_desired = Affine.find(x=self.x1, y=self.y1)
        np_test.assert_almost_equal(af.gl, af_desired.gl)
        np_test.assert_almost_equal(af.d, af_desired.d)
        np_test.assert_almost_equal(af.glError, numpy.zeros_like(self.x1))
        np_test.assert_almost_equal(af.dError, numpy.zeros_like(self.x1[0:1]))
        np_test.assert_almost_equal(af.rmsErrorEst, 0)

        # parallelogram, rotation, scale, parity, exact
        af = Affine.findTwoStep(x=self.x1[0:2], y=self.y1m[0:2],
                                x_gl=self.x1, y_gl=self.y1m+[2,-3])
        af_desired = Affine.find(x=self.x1, y=self.y1m)
        np_test.assert_almost_equal(af.gl, af_desired.gl)
        np_test.assert_almost_equal(af.d, af_desired.d)
        np_test.assert_almost_equal(af.glError, numpy.zeros_like(self.x1))
        np_test.assert_almost_equal(af.dError, numpy.zeros_like(self.x1[0:2]))
        np_test.assert_almost_equal(af.rmsErrorEst, 0)

        # parallelogram, rotation, scale, parity, not exact
        af = Affine.findTwoStep(x=self.x2, y=self.y2m,
                                x_gl=self.x2, y_gl=self.y2m+[2,-3])
        af_desired = Affine.find(x=self.x2, y=self.y2m)
        np_test.assert_almost_equal(af.gl, af_desired.gl)
        np_test.assert_almost_equal(af.d, af_desired.d)
        np_test.assert_almost_equal(af.rmsErrorEst, af_desired.rmsError, 
                                    decimal=0)

    def testDecompose(self):
        """
        Tests decompose (decomposeQR and decomposeSV) and composeGl
        """

        repeat = 10
        for i in range(repeat):
 
            # initialize 3x3 random array
            gl = numpy.random.random((3,3))

            # check qpsm 
            af = Affine(gl=gl)
            af.decompose(order='qpsm')
            self.checkQRDecompose(af)
            af.gl = None
            new_gl = af.composeGl(order='qpsm')
            np_test.assert_almost_equal(new_gl, gl)

            # check psmq 
            af = Affine(gl=gl)
            af_1 = Affine()
            q, p, s, m = af_1.decompose(order='psmq', gl=gl)
            af_1.q = q
            af_1.p = p
            af_1.s = s
            af_1.m = m
            self.checkQRDecompose(af_1)
            af_2 = Affine()
            gl_2 = af_2.composeGl(order='psmq', q=q, p=p, s=s, m=m)
            np_test.assert_almost_equal(gl_2, gl)

            # check usv 
            af = Affine(gl=gl)
            af.decompose(order='usv')
            self.checkSVDecompose(af)
            af_1 = Affine()
            af_1.u = af.u
            af_1.s = af.s
            af_1.p = af.p
            af_1.v = af.v
            new_gl = af_1.composeGl(order='usv')
            np_test.assert_almost_equal(new_gl, gl)

            # initialize 4x4 random array
            gl = numpy.random.random((4,4))

            # check qpsm 
            af = Affine(gl=gl)
            af_1 = Affine()
            q, p, s, m = af_1.decompose(order='qpsm', gl=gl)
            af_1.q = q
            af_1.p = p
            af_1.s = s
            af_1.m = m
            self.checkQRDecompose(af_1)
            af_2 = Affine()
            gl_2 = af_2.composeGl(order='qpsm', q=q, p=p, s=s, m=m)
            np_test.assert_almost_equal(gl_2, gl)

            # check psmq 
            af = Affine(gl=gl)
            af.decompose(order='psmq')
            self.checkQRDecompose(af)
            af.gl = None
            new_gl = af.composeGl(order='psmq')
            np_test.assert_almost_equal(new_gl, gl)

            # check psmq 
            af = Affine(gl=gl)
            af_1 = Affine()
            af_1.u, af_1.p, af_1.s, af_1.v = af_1.decompose(order='usv', gl=gl)
            self.checkSVDecompose(af_1)
            af_2 = Affine()
            af_2.u = af_1.u
            af_2.s = af_1.s
            af_2.p = af_1.p
            af_2.v = af_1.v
            new_gl = af_2.composeGl(order='usv')
            np_test.assert_almost_equal(new_gl, gl)

    def checkQRDecompose(self, af):
        """
        Check properties of QR decomposition
        """

        size = af.q.shape[0]

        # q
        np_test.assert_almost_equal(scipy.linalg.det(af.q), 1)
        ortho_0, ortho_1 = self.checkOrtho(af.q)
        np_test.assert_almost_equal(ortho_0, numpy.identity(size))
        np_test.assert_almost_equal(ortho_1, numpy.identity(size))

        # p
        np_test.assert_equal(numpy.abs(af.p), numpy.identity(size))
        p_diag = af.p.diagonal()
        if p_diag[af.parity_axis] == 1:
            np_test.assert_equal((p_diag==1).all(), True)
        else:
            np_test.assert_equal(numpy.count_nonzero(~(p_diag==1)), 1)

        # s
        np_test.assert_equal((af.s > 0)*1., numpy.identity(size))
        np_test.assert_equal((af.s.diagonal() >= 0).all(), True)

        # m
        np_test.assert_almost_equal(af.m.diagonal(), numpy.ones(size))
        for i in range(size):
            for j in range(i):
                np_test.assert_almost_equal(af.m[i,j], 0)

    def checkSVDecompose(self, af):
        """
        Check properties of singular value decomposition
        """

        size = af.u.shape[0]

        # u
        np_test.assert_almost_equal(scipy.linalg.det(af.u), 1)
        ortho_0, ortho_1 = self.checkOrtho(af.u)
        np_test.assert_almost_equal(ortho_0, numpy.identity(size))
        np_test.assert_almost_equal(ortho_1, numpy.identity(size))

        # v
        np_test.assert_almost_equal(scipy.linalg.det(af.v), 1)
        ortho_0, ortho_1 = self.checkOrtho(af.v)
        np_test.assert_almost_equal(ortho_0, numpy.identity(size))
        np_test.assert_almost_equal(ortho_1, numpy.identity(size))

        # p
        np_test.assert_equal(numpy.abs(af.p), numpy.identity(size))
        p_diag = af.p.diagonal()
        if p_diag[af.parity_axis] == 1:
            np_test.assert_equal((p_diag==1).all(), True)
        else:
            np_test.assert_equal(numpy.count_nonzero(~(p_diag==1)), 1)
            
        # s
        np_test.assert_equal((af.s > 0)*1., numpy.identity(size))
        np_test.assert_equal((af.s.diagonal() >= 0).all(), True)

    def testInverse(self):
        """
        Tests inverse method
        """

        #################################################
        #
        # parallelogram, scale, rotation, parity, exact
        #

        # 
        af = Affine.find(x=self.x1, y=self.y1m)

        # test inverse
        af_inverse = af.inverse()
        np_test.assert_almost_equal(numpy.dot(af.gl, af_inverse.gl),
                                    numpy.identity(2))
        afi = Affine.find(x=self.y1m, y=self.x1)
        np_test.assert_almost_equal(af_inverse.gl, afi.gl)
        np_test.assert_almost_equal(af_inverse.d, afi.d)
        np_test.assert_almost_equal(self.x1, af_inverse.transform(self.y1m))

        # error
        np_test.assert_almost_equal(af_inverse.error, afi.error)
        np_test.assert_almost_equal(af_inverse.rmsError, afi.rmsError)

        #################################################
        #
        # parallelogram, scale, rotation, parity, not exact
        #
        # Note: only approximate comparisons because inverse of an optimal
        # (least squares) x->y transformation is not the optimal y->x.
        
        af = Affine.find(x=self.x2, y=self.y2m)

        # test inverse
        af_inverse = af.inverse()
        np_test.assert_almost_equal(numpy.dot(af.gl, af_inverse.gl),
                                    numpy.identity(2))
        afi = Affine.find(x=self.y2m, y=self.x2)
        np_test.assert_almost_equal(af_inverse.gl, afi.gl, decimal=1)
        np_test.assert_almost_equal(af_inverse.d, afi.d, decimal=1)
        np_test.assert_almost_equal(self.x2, af_inverse.transform(self.y2m),
                                    decimal=0)

        # error
        np_test.assert_almost_equal(af_inverse.error, afi.error, decimal=1)
        np_test.assert_almost_equal(af_inverse.rmsError, afi.rmsError, 
                                    decimal=1)        

    def testTransform(self):
        """
        Tests transform() method
        """

        # simple
        af = Affine.find(x=self.x1, y=self.y1m)
        desired = numpy.inner(self.x1, af.gl) + af.d
        np_test.assert_almost_equal(af.transform(self.x1), desired)

        # 2D phi=90, 'point_dim'
        af = Affine2D(phi=numpy.pi/2, scale=1)
        desired = numpy.array([[0, 0], [0, 2], [-1, 2], [-1, 0]])
        np_test.assert_almost_equal(
            af.transform(self.x1, xy_axes='point_dim'), desired)
        
        # 2D phi=90, 'point_dim', origin = None
        af = Affine2D(phi=numpy.pi/2, scale=1)
        desired = numpy.array([[0, 0], [0, 2], [-1, 2], [-1, 0]])
        np_test.assert_almost_equal(
            af.transform(self.x1, xy_axes='point_dim', origin=None), desired)
        
        # 2D phi=90, 'point_dim', origin 
        af = Affine2D(phi=numpy.pi/2, scale=1)
        #desired = numpy.array([[0, 0], [0, 2], [-1, 2], [-1, 0]])
        desired = numpy.array([[3, -1], [3, 1], [2, 1], [2, -1]])
        np_test.assert_almost_equal(
            af.transform(self.x1, xy_axes='point_dim', origin=[2,1]), desired)
        
        # 2D phi=-90, 'dim_point'
        af = Affine2D(phi=-numpy.pi/2, scale=1)
        desired = numpy.array([[0, 0, 1, 1], [0, -2, -2, 0]])
        np_test.assert_almost_equal(
            af.transform(self.x1.transpose(), xy_axes='dim_point'), desired)

        # 2D phi=-90, 'dim_point'
        af = Affine2D(phi=-numpy.pi/2, scale=1)
        #desired = numpy.array([[0, 0, 1, 1], [0, -2, -2, 0]])
        desired = numpy.array([[-3, -3, -2, -2], [1, -1, -1, 1]])
        np_test.assert_almost_equal(
            af.transform(
                self.x1.transpose(), xy_axes='dim_point', origin=[-1,2]), 
            desired)

        # 2d phi 90, 'mgrid'
        af = Affine2D(phi=numpy.pi/2, scale=1)
        grid = numpy.mgrid[0:3, 0:2]
        desired = numpy.array(
            [[[0, -1], [0, -1], [0, -1]],
             [[0, 0], [1, 1], [2, 2]]])
        np_test.assert_almost_equal(
            af.transform(grid, xy_axes='mgrid'), desired)
        
        # 2d phi 90, 'mgrid', origin=0
        af = Affine2D(phi=numpy.pi/2, scale=1)
        grid = numpy.mgrid[0:3, 0:2]
        desired = numpy.array(
            [[[0, -1], [0, -1], [0, -1]],
             [[0, 0], [1, 1], [2, 2]]])
        np_test.assert_almost_equal(
            af.transform(grid, xy_axes='mgrid', origin=0), desired)
        
        # 2d phi 90, 'mgrid', origin
        af = Affine2D(phi=numpy.pi/2, scale=1)
        grid = numpy.mgrid[0:3, 0:2]
        desired = numpy.array(
            [[[1, 0], [1, 0], [1, 0]],
             [[-3, -3], [-2, -2], [-1, -1]]])
        np_test.assert_almost_equal(
            af.transform(grid, xy_axes='mgrid', origin=[2,-1]), desired)
        
       # 2d phi 90, scale 2, 'mgrid', origin
        af = Affine2D(phi=numpy.pi/2, scale=2)
        grid = numpy.mgrid[0:3, 0:2]
        desired = numpy.array(
            [[[0., -2], [0, -2], [0, -2]],
             [[-5, -5], [-3, -3], [-1, -1]]])
        np_test.assert_almost_equal(
            af.transform(grid, xy_axes='mgrid', origin=[2,-1]), desired)
        
        # 2d phi -90, 'mgrid' (meshgrid)
        af = Affine2D(phi=-numpy.pi/2, scale=1)
        grid = numpy.meshgrid([0,2,4], [1,3], indexing='ij')
        desired = numpy.array(
            [[[1, 3], [1, 3], [1, 3]],
             [[0, 0], [-2, -2], [-4, -4]]])
        np_test.assert_almost_equal(
            af.transform(grid, xy_axes='mgrid'), desired)
        
        # 2d phi -90, 'mgrid' (meshgrid), translation
        af = Affine2D(phi=-numpy.pi/2, scale=1)
        af.d = [1, -1]
        grid = numpy.meshgrid([0,2,4], [1,3], indexing='ij')
        desired = numpy.array(
            [[[2, 4], [2, 4], [2, 4]],
             [[-1, -1], [-3, -3], [-5, -5]]])
        np_test.assert_almost_equal(
            af.transform(grid, xy_axes='mgrid'), desired)
        
       # 2d phi -90, 'mgrid' (meshgrid), translation, origin
        af = Affine2D(phi=-numpy.pi/2, scale=1)
        af.d = [1, -1]
        grid = numpy.meshgrid([0,2,4], [1,3], indexing='ij')
        desired = numpy.array(
            [[[5., 7], [5, 7], [5, 7]],
             [[-2, -2], [-4, -4], [-6, -6]]])
        np_test.assert_almost_equal(
            af.transform(grid, xy_axes='mgrid', origin=[1,-2]), desired)
 
        # gl
        af = Affine2D(gl=numpy.array([[1., 2], [0, -1]]))
        np_test.assert_almost_equal(
            af.transform([[1,-1]], xy_axes='point_dim'), [[-1, 1]])
        
        # gl, translation
        af = Affine2D(gl=numpy.array([[1., 2], [0, -1]]), d=[2, -1])
        np_test.assert_almost_equal(
            af.transform([[1], [-1]], xy_axes='dim_point'), [[1], [0]])
        
        # gl, translation, origin
        af = Affine2D(gl=numpy.array([[1., 2], [0, -1]]), d=[2, -1])
        np_test.assert_almost_equal(
            af.transform([[1], [-1]], xy_axes='dim_point', origin=[1, -1]), 
            [[3], [-2]])
       
    def testTransformArray(self):
        """
        Tests transformArray() for 1D and 2D. Tests for 3D are in
        test_rigid_3d.
        """

        # 1D
        ar1 = numpy.arange(5, dtype=float)
        af = Affine(gl=[[1.]], d=[1.])
        trans = af.transformArray(array=ar1, origin=[0], cval=50)
        np_test.assert_almost_equal(trans, [50,0,1,2,3])

        # 1D fractional
        ar1 = numpy.arange(5, dtype=float)
        af = Affine(gl=[[1.]], d=[0.5])
        trans = af.transformArray(array=ar1, origin=[0], cval=50)
        np_test.assert_almost_equal(trans, [50,0.5,1.5,2.5,3.5])

       # 2D array
        ar2 = numpy.arange(20, dtype=float).reshape(4,5)

        # translation
        af = Affine2D(phi=0, scale=1, d=[0,1])
        trans = af.transformArray(array=ar2, origin=[0,0], cval=50)
        desired = numpy.array(
            [[50,0,1,2,3],
             [50,5,6,7,8],
             [50,10,11,12,13],
             [50,15,16,17,18]])
        np_test.assert_almost_equal(trans, desired)

        # translation
        af = Affine2D(phi=0, scale=1, d=[0,-1])
        trans = af.transformArray(array=ar2, origin=[0,0], cval=50)
        desired = numpy.array(
            [[1,2,3,4,50],
             [6,7,8,9,50],
             [11,12,13,14,50],
             [16,17,18,19,50]])
        np_test.assert_almost_equal(trans, desired)

        # translation
        af = Affine2D(phi=0, scale=1, d=[1,-2])
        trans = af.transformArray(array=ar2, origin=[0,0], cval=50)
        desired = numpy.array(
            [[50,50,50,50,50],
             [2,3,4,50,50],
             [7,8,9,50,50],
             [12,13,14,50,50]])
        np_test.assert_almost_equal(trans, desired)

        # translation float
        af = Affine2D(phi=0, scale=1, d=[0,0.5])
        trans = af.transformArray(array=ar2, origin=[0,0], cval=50)
        desired = numpy.array(
            [[50,0.5,1.5,2.5,3.5],
             [50,5.5,6.5,7.5,8.5],
             [50,10.5,11.5,12.5,13.5],
             [50,15.5,16.5,17.5,18.5]])
        np_test.assert_almost_equal(trans, desired)

        # translation float
        af = Affine2D(phi=0, scale=1, d=[-0.5,0.5])
        trans = af.transformArray(array=ar2, origin=[0,0], cval=50)
        desired = numpy.array(
            [[50,3,4,5,6],
             [50,8,9,10,11],
             [50,13,14,15,16],
             [50,50,50,50,50]])
        np_test.assert_almost_equal(trans, desired)

        # 2D rotations different origin 
        af = Affine2D(phi=numpy.pi/2, scale=1)
        trans = af.transformArray(array=ar2, origin=[0,0], cval=50)
        np_test.assert_almost_equal(trans[0,:], [0,5,10,15,50])
        np_test.assert_almost_equal(trans[1:4,:], numpy.zeros((3,5))+50)
        trans = af.transformArray(array=ar2, origin=[2,1], cval=50)
        desired = numpy.array(
            [[8,13,18,50,50],
             [7,12,17,50,50],
             [6,11,16,50,50],
             [5,10,15,50,50]])
        np_test.assert_almost_equal(trans, desired)

        # 2D rotation + translation
        af = Affine2D(phi=numpy.pi/2, scale=1, d=[0,1])
        trans = af.transformArray(array=ar2, origin=[2,1], cval=50)
        desired = numpy.array(
            [[3,8,13,18,50],
             [2,7,12,17,50],
             [1,6,11,16,50],
             [0,5,10,15,50]])
        # Note: because out or boundary (slightly) the following are cval
        #desired[0,0] = desired[3,0] = desired[3,1] = 50
        np_test.assert_almost_equal(trans[1:3, 1:4], desired[1:3, 1:4])
        af = Affine2D(phi=numpy.pi/2, scale=1, d=[-1,1])
        trans = af.transformArray(array=ar2, origin=[2,1], cval=50)
        desired = numpy.array(
            [[2,7,12,17,50],
             [1,6,11,16,50],
             [0,5,10,15,50],
             [50,4,9,14,19]])
        # Note: because out or boundary (slightly) the following are cval
        #desired[0,0] = desired[2,0] = 50
        np_test.assert_almost_equal(trans[1:3,1:4], desired[1:3,1:4])

    def testRemoveMasked(self):
        """
        Tests removeMasked()
        """

        x = numpy.array([[1,2], [3,4], [5,6]])
        x_mask = numpy.array([1, 0, 0])
        y = numpy.array([[2,4], [6,8], [10,12]])
        y_mask = numpy.array([0, 0, 1])
  
        data, total_mask = Affine.removeMasked(arrays=[x, y], 
                                               masks=(x_mask, y_mask))
        np_test.assert_equal(data[0], numpy.array([[3,4]]))
        np_test.assert_equal(data[1], numpy.array([[6,8]]))
        np_test.assert_equal(total_mask, numpy.array([1,0,1]))

        data, total_mask = Affine.removeMasked(arrays=[x, y])
        np_test.assert_equal(data[0], x)
        np_test.assert_equal(data[1], y)
        np_test.assert_equal(total_mask, numpy.array([0,0,0]))

    def checkOrtho(self, ar):
        """
        Calculates dot products between all rows and between all columns of a 
        matrix (arg ar). Used to check orthonormality of a matrix.

        Returns: the dot products if the form of two matrices
        """

        res_0 = numpy.zeros_like(ar) - 1.
        for i in range(ar.shape[0]):
            for j in range(ar.shape[0]):
                res_0[i, j] = numpy.dot(ar[i,:], ar[j,:])

        res_1 = numpy.zeros_like(ar) - 1.
        for i in range(ar.shape[0]):
            for j in range(ar.shape[0]):
                res_1[i, j] = numpy.dot(ar[:,i], ar[:,j])

        return res_0, res_1


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAffine)
    unittest.TextTestRunner(verbosity=2).run(suite)
