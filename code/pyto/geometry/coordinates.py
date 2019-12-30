"""
Contains class Coordinates.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: coordinates.py 1311 2016-06-13 12:41:50Z vladan $
"""

__version__ = "$Revision: 1311 $"


import logging
from copy import copy, deepcopy
import numpy
import scipy

from affine import Affine


class Coordinates(object):
    """
    Currently implemented methods for affine transformation of cartesion
    coordinates. All coordinates of a given space (grid, or mash) are
    transformed.    
    """

    ###############################################################
    #
    # Initialization
    #
    
    def __init__(self):
        """
        """
        pass    

    ###############################################################
    #
    # 
    #
    
    @classmethod
    def transform(cls, shape, affine, origin, center=False):
        """
        Does affine transformation of coordinates. All coordinates (indices)
        of an array whose shape is specified (arg shape) are transformed.

        The affine transformation is a general linear (gl) transformation 
        followed by translation (see class Affine).

        If arg center is True, after the transformation the coordinate
        system is translated so that the transformation origin has 
        coordinates [0, 0, ... ].

        This method should be pretty fast. It first calculates the gl part of 
        the transformation of the array origin (index (0, 0, ...)). Then it adds
        constant increments (elements of gl) along all axis to finish the gl 
        transformation. Translation is done at the end.

        Transformed indices are returned as an array that has the same 
        shape is the initial one (arg shape) but it has an additional axis
        of length ndim (n dim of shape) prepended to the shape. In short, the
        transformation is:

          [x, y, z, ...] -> result[i, x, y, z, ...]

        where i denotes the axis number (0 for x axis, 1 for y and so on).

        Arguments:
          - shape: shape of the initial array that specifies indices 
          - affine: (Affine) affine transformation 
          - origin: coodrinates of the transformation origin
          - center: flag indicating if the transformation origin is set to 0

        Return: (ndarray) transformed indices
        """

        # find how the array origin transforms
        origin = numpy.asarray(origin)
        initial = affine.transform(x=-origin) + origin

        # transform by gl
        transformed = cls.transformIndices(shape=shape, gl=affine.gl, 
                                           initial=initial)

        # translation
        ndim = len(shape)
        translation = affine.d
        for axis in range(ndim):
            translation = numpy.expand_dims(translation, -1)
        transformed = transformed + translation

        # centering
        if center:
            origin_exp = origin
            for axis in range(ndim):
                origin_exp = numpy.expand_dims(origin_exp, -1)
            transformed = transformed - origin_exp

        return transformed
        
    @classmethod
    def transformIndices(cls, shape, gl, initial=0):
        """
        Transforms coordinates by (general linear) matrix gl in N dimensions. 
        The coordinates are specified as indices of an array specified by its
        shape.

        This method should be pretty fast. It assigns the value specified
        by arg initial to its origin (index (0, 0, ...)) and then adds
        constant increments (elements of gl) along all axis.

        Transformed indices are retrurned as an array that has the same 
        shape is the initial one (arg shape) but it has an additional axis
        of length ndim (n dim of shape) prepended to the shape. In short, the
        transformation is:

          [x, y, z, ...] -> result[i, x, y, z, ...]

        where i denotes the axis number (0 for x axis, 1 for y and so on).

        Arguments:
          - shape: shape of the initial array that specifies indices 
          - gl: (ndim x ndim ndarray) general linear transformation matrix
          - initial: (array of length ndim) origin values for all axes

        Return: (ndarray) transformed indices
        """

        # initialize array to hold transformed coordinates
        ndim = len(shape)
        new_shape = [ndim] + list(shape)
        new_ndim = len(new_shape)
        transformed = numpy.zeros(shape=new_shape, dtype=float)

        # set element at 0,0, ... position
        zero_coord = tuple([slice(None)] + ndim * [0])
        if ((not isinstance(initial, (list, tuple, numpy.ndarray))) 
            and (initial == 0)):
            initial = ndim * [0]
        transformed[zero_coord] = initial

        # set values on 1d subarrays emanating from the array origin
        for axis in range(1, new_ndim):
            coord = [slice(None)] + ndim * [0]
            coord[axis] = slice(1,None)
            gl_column = numpy.expand_dims(gl[:, axis-1], axis=1)
            transformed[tuple(coord)] = gl_column

        # make transformation 
        for axis in range(1, new_ndim):
            transformed = numpy.add.accumulate(transformed, axis=axis)
            
        return transformed
