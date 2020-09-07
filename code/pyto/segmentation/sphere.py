"""
Class Shapes provides methods for crating and manipulating spheres,
in addition to methods defined in its base class Segment.

Note: This class is depreciated

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range

__version__ = "$Revision$"


import sys
import logging
import inspect
import numpy
import scipy
import scipy.ndimage as ndimage

from .segment import Segment

class Sphere(Segment):
    """
    """

    def __init__(self, data=None, copy=True, ids=None, clean=False):
        """
        Calls Segment.__init__ to initialize data, data-related and id-related
        attributes.
        """
        super(Sphere, self).__init__(data=data, copy=copy, ids=ids, clean=clean)


    #################################################################
    #
    # Methods
    #
    ###################################################################

    def extendDiscs(self, data=None, ids=None, axis=2, angle=None):
        """
        Extends circles to make spheres.

        Center and radius (mean distance Each of 
        """

        # parse arguments
        data, ids, update = self.parseInput(data=data, ids=ids)
        
        # find cicle centers for ids
        from .morphology import Morphology
        mor = Morphology(segments=data, ids=ids)
        mor.getCenter()
        centers = mor.center[ids]

        # make (empty) circles
        circles = numpy.zeros_like(data)
        seg = Segment()
        for cent, id in zip(centers, ids):

            # find empty circles on central slices
            circle_slice = data.take(indices=[cent[axis]], axis=axis).squeeze()
            circle_slice = circ.makeSurfaces(data=circle_slice, size=1, ids=[id])

            # put them in empty circles array
            none_slice = [slice(None)] * surface.ndim
            none_slice[axis] = cent[axis]
            circles[tuple(none_slice)] = circle_slice

        # find radii of circles for ids
        mor.getRadius(surface=circles)
        radii = mor.sliceRadius.mean[ids]

        # make filled spheres (balls)
        sph = self.__class__.make(center=centers, radius=radii,
                           shape=self.data.shape, ids=ids, axis=axis, angle=angle)

        return sph

    @classmethod
    def make(cls, centers, radii, shape, ids=None, axis=2, angle=None):
        """
        Makes an instance of this class containing balls with
        radii at centers and labels them with ids.

        Ids are determined in the following
        order: argument ids, self.ids, ids extracted form data, 1, 2, ... .

        Arguments:
          - centers: array containing coordinates of centers for segments listed
          in ids
          - radii: array containing radii for segments listed in ids
          - shape: shape of the data array that holds the circles
          - ids: ids of segments (spheres)
 
        """

        # parse arguments
        if (ids is None) or (len(ids) == 0) or (max(ids) < 1):
            ids = list(range(1, len(centers)+1))

        # pur centers in a data array and make an instance
        data = numpy.zeros(shape=shape, dtype='int_')
        for cent, id in zip(centers, ids):
            data[tuple(cent)] = id
        print(data)
        inst = cls(data=data, copy=False, ids=ids, clean=False)
        print(inst.data)

        # find max_radius
        max_radius = max(radii)

        # make spheres
        index = 0
        for distance, slice_, id in inst.markDistance(size=max_radius):

            # make a sphere
            sphere = numpy.where(distance<=radii[index], id, 0) 
            inst.data[slice_] = sphere

            # cut a sphere
            
            index += 1

        return inst
    
