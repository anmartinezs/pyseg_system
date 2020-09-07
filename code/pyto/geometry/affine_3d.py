"""
Contains class Affine3D for preforming affine transformation (general linear
transformation followed by translation) on points (vectors) in 3D.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from past.builtins import basestring

__version__ = "$Revision$"


import logging
import numpy
import scipy
import scipy.linalg as linalg

from .vector import Vector
from .points import Points
from .affine import Affine

class Affine3D(Affine):
    """
    Finds and preforms affine transformation (general linear transformation
    followed by translation) on points (vectors) in 3D.

    Only implements method getQ(), everything else is inherited from class
    Affine.

    Note: not integrated well with Rigid3D.
    """

    def __init__(
            self, d=None, gl=None, alpha=None, axis=None, xy_axes='point_dim'):
        """
        If arg gl is not None sets self.gl to gl. Otherwise, and if args alpha
        and axis are not None calculates the corresponding rotation matrix
        (uses Affine3D.getQ()) and sets self.gl to that value.

        If arg d is None, or 0, and gl is not None, self.d is set to 
        numpy.array([0, 0, ...]) with the correct length. Otherwise self.d is 
        set to arg d

        If the arg xy_axes is 'point_dim' / 'dim_point', points used in this 
        instance should be specified as n_point x 3 / 3 x n_point 
        matrices.
        
        Arguments:
          - gl: (numpy.ndarray of shape (ndim, ndim)) general linear 
          transormation matrix
          - d: (numpy.ndarray of shape ndim) translation
          - alpha: rotation angle in rad
          - axis: rotation axis, can be 'x', 'y' or 'z', or specified by a 
          vector (numpy.ndarray)
          - xy_axes: order of axes in matrices representing points, can be
          'point_dim' (default) or 'dim_point'

        ToDo: extract phi and theta from gl
        """
        
        # if gl is None, try to set it from alpha and axis
        if gl is None:
            if (alpha is not None) and (axis is not None):
                gl = Affine3D.getQ(alpha=alpha, axis=axis)

        # set d
        d = Affine.makeD(d, ndim=3)

        # initialize
        super(self.__class__, self).__init__(gl, d)

    @classmethod
    def identity(cls, ndim=3):
        """
        Returnes an identity object of this class, that is a transformation 
        that leaves all vectors invariant.

        Argument ndim is ignored, it should be 3 here.
        """

        obj = cls.__base__.identity(ndim=2)        
        return obj
        
    @classmethod
    def getQ(cls, alpha, axis):
        """
        Returns rotation matrix corresponding to rotation around specified axis
        by angle.

        Arguments:
          - alpha: rotation angle in rad
          - axis: rotation axis, can be 'x', 'y' or 'z', or specified by a 
          vector (numpy.ndarray)

        Returns: (3x3 ndarray) rotation matrix
        """

        if isinstance(axis, basestring):

            # rotation about one of the main axes
            if axis == 'x':
                q = numpy.array(
                    [[1, 0, 0],
                     [0, numpy.cos(alpha), -numpy.sin(alpha)],
                     [0, numpy.sin(alpha), numpy.cos(alpha)]])
            elif axis == 'y':
                q = numpy.array(
                    [[numpy.cos(alpha), 0, numpy.sin(alpha)],
                     [0, 1, 0],
                     [-numpy.sin(alpha), 0, numpy.cos(alpha)]])
            elif axis == 'z':
                q = numpy.array(
                    [[numpy.cos(alpha), -numpy.sin(alpha), 0],
                     [numpy.sin(alpha), numpy.cos(alpha), 0],
                     [0, 0, 1]])

        elif isinstance(axis, (list, tuple, numpy.ndarray)):

            # get phi and theta for the axis vector
            axis_vector = Vector(axis)
            phi = axis_vector.phi
            theta = axis_vector.theta

            q_tilt = numpy.dot(cls.getQ(phi, 'z'), cls.getQ(theta, 'y'))
            q_back = numpy.dot(cls.getQ(-theta, 'y'), cls.getQ(-phi, 'z'))
            q = numpy.dot(cls.getQ(alpha, 'z'), q_back)
            q = numpy.dot(q_tilt, q)
            
        else:
            raise ValueError(
                "Axis can be one of the majot axes ('x', 'y', 'z') or a "
                + "vector.")

        return q
