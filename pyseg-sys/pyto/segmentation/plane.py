"""
Class Plane provides methods for crating and manipulating n-1 dimensional planes
in n dimentional space, in addition to methods defined in its base class Segment.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: plane.py 21 2007-12-23 11:38:03Z vladan $
"""

__version__ = "$Revision: 21 $"


import sys
import logging
import inspect
import numpy
import scipy
import scipy.ndimage as ndimage

from segment import Segment

class Plane(Segment):
    """
    """

    def __init__(self, data=None, copy=True, ids=None, clean=False):
        """
        Calls Segment.__init__ to initialize data, data-related and id-related
        attributes.
        """
        super(Plane, self).__init__(data=data, copy=copy, ids=ids, clean=clean)


    #################################################################
    #
    # Methods
    #
    ###################################################################

    @classmethod
    def generate(cls, vectors, points, slices, thick=1):
        """
        Generator yielding (ndarray) planes defined by normal vectors
        (vectors) and points that belong to the arrays (points), as well as the
        subspaces on the positive and negative sides of the planes.

        Each output array has the shape and position defined by the corresponding
        element of slices. Points are given in respect to the same coordinate
        system that is used to define slices.

        Output array elements that belong to a plane are set to 0, while those
        that do not are labeled by 1 (on the positive side on the plane according
        to the direction of the corresponding normal vector) or by -1 (on the
        negative side).
        
        An element of an output array is considered to belong to a plane if its
        Euclidean distance to the plane is <= thick/2, so that thick is the
        "thickness" of each plane. If thick=0, only the array elements whose
        distance to the plane is 0 exactly are labeled by 0.

        Points and slices have to have the same number of elements. If only one
        vector is given it is used for all points/slices.

        Arguments:
          - vectors: normal vectors (or just one vector) needed to define a
          plane (shape n_planes, ndim, or ndim)
          - points: (coordinates of) points that belong to planes (shape
          n_planes, ndim), given in respect to the same coordinate system
          used for slices
          - slices: slices that define the size and position of each output
          array (shape n_planes, ndim)
          - thick: thickness of planes

        Yields:
          - plane: array that labels the plane
        """

        # convert points and vectors to ndarray and normalize vectors  
        points = numpy.asarray(points)
        vectors = numpy.asarray(vectors, dtype='float_')
        vectors = vectors / numpy.sqrt(numpy.inner(vectors, vectors))
        if (vectors.ndim == 1) and (points.ndim) > 1:
            vectors = vectors[numpy.newaxis, ...].repeat(points.shape[0], axis=0)

        # genereate subarrays with planes
        for slice_, vector, point in zip(slices, vectors, points):

            # make plane array indices
            shape = [sl_1d.stop - sl_1d.start for sl_1d in slice_]
            indices = numpy.indices(shape)

            # mark points on, above and below the plane
            point_0 = [point_1d - sl_1d.start \
                       for point_1d, sl_1d in zip(point, slice_)]
            dist = numpy.tensordot(indices, vector, axes=(0,0)) \
                   - numpy.vdot(point_0, vector)
            plane = numpy.where(dist > thick/2, 1, 0)
            plane = numpy.where(dist > -thick/2, plane, -1)
            if thick == 0:
                plane = numpy.where(dist == 0, 0, plane) 

            yield plane
            
    @classmethod
    def generate_one(cls, vector, point, ndslice, thick=1):
        """
        Returns (ndarray) plane defined by normal vector (vector) and a point
        that belong to the array (point), as well as the
        subspaces on the positive and negative sides of the plane.

        The output array has the shape and position defined by ndslice. Point
        is given in respect to the same coordinate system that is used to define
        ndslice.

        Output array elements that belong to the plane are set to 0, while those
        that do not are labeled by 1 (on the positive side on the plane according
        to the direction of the normal vector) or by -1 (on the negative side).
        
        An element of an output array is considered to belong to a plane if its
        Euclidean distance to the plane is <= thick/2, so that thick is the
        "thickness" of the plane. If thick=0, only the array elements whose
        distance to the plane is 0 exactly are labeled by 0.
        
        Uses generate method.

        Arguments:
          - vector: normal vector needed to define a plane (shape ndim)
          - point: (coordinates of) a point that belong to the plane (shape ndim),
          given in respect to the same coordinate system used for slice
          - slice: (n-dim) slice that define the size and position of the output
          array (shape ndim)
          - thick: thickness of the planes

        Returns:
          - (ndarray) plane
        """
        for plane in cls.generate(vectors=vector, points=[point],
                                  slices=[ndslice], thick=thick):
            pass
        return plane
            
