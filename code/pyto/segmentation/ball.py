"""
Class Ball provides methods for crating and manipulating n-dimensional balls
(filled spheres), in addition to methods defined in its base class Segment.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from builtins import next
from builtins import zip
from builtins import str
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"


import sys
import logging
import inspect
import itertools
import numpy
import scipy

import pyto.util.numpy_plus as numpy_plus
from .segment import Segment
from .plane import Plane

class Ball(Segment):
    """
    """

    def __init__(self, data=None, copy=True, ids=None, clean=False):
        """
        Calls Segment.__init__ to initialize data, data-related and id-related
        attributes.
        """
        super(Ball, self).__init__(data=data, copy=copy, ids=ids, clean=clean)


    #################################################################
    #
    # Methods
    #
    ###################################################################

    def extendDiscs(self, data=None, ids=None, axis=-1, cut=None, theta=None,
                    enlarge=0.5, external=0, check=True):
        """
        Extends disc-like segments to balls and cuts their tops and bottoms
        with planes.

        The discs and the balls have the same centers, while the radii of
        balls are increased by enlarge (in respect to the discs).

        If data is not given self.data is used. If ids is not given tries
        self.ids (if data was not given), or extracts ids from data (or
        self.data).

        First calculates centers and radii of the discs using extractDiscs
        method. Then uses make method to generate balls and cuts them if needed
        (see extractDiscs and make for details). Finally, the balls are put in
        the self.data array. Elements that belong to discs, but do not belong
        to any of the balls are set to external (expected to be the id of the
        region surrounding discs).

        Other segments are kept unchanged.

        Note: when discs are first extracted and then extended (using this 
        method) the best value for enlarge seems to be 0.5. The reason is 
        essentialy a rounding error. Namely, when a disc of a radius R is 
        created, that means that all array elements whose distance is at most
        R to the center are selected. Now when a mean radius is calculated it
        naturally smaller than R. Therefore, in order to reproduce an in initial
        dist etend > 0 should be used.
        
        Arguments:
          - data: (ndarray) labels containg disc-like segments and possibly
          other segments
          - ids: segment ids of discs
          - axis: the principal axis of disks, should be along one of the
          coordinate axes (in 3d 0 is x axis, 1 is y and 2 or -1 is z)
          - cut: normal vector of the plane that cuts each ball (shape ndim)
          - theta: angle that determines how much of each ball is cut (nothing 
          is cut for 0, while if theta is numpy.pi/2 each of the two planes cuts
          a half of a ball)
          - external: id surrounding discs / balls
          - check: if True, checks if balls overlap (uses findOverlaps()) 

        Returns instance of this class containing the balls, having attributes:
          - data:  (ndarray) labels containg ball-like segments and possibly
          other segments
          - ids: ids of all segments, balls and possibly other segments
        """

        # parse arguments
        data, ids, update = self.parseInput(data=data, ids=ids)

        # extract discs
        centers, radii = self.extractDiscs(data=data, ids=ids, axis=axis)

        # enlarge radii
        radii += enlarge

        # expand the discs to balls (filled spheres), other elements 0
        ball = self.__class__.make(
            centers=centers, radii=radii, dtype=data.dtype, shape=data.shape, 
            ids=ids, cut=cut, theta=theta, check=check)

        # make array that contain balls and all non-disc segments
        other_data = self.remove(ids=ids, data=self.data.copy(), value=external)
        all_data = numpy.where(ball.data>0, ball.data, other_data)

        ball.setData(data=all_data, ids=self.ids, copy=False, clean=False)

        return ball

    def extractDiscs(self, data=None, ids=None, axis=-1):
        """
        Calculates the center of each segment, and the mean outside (boundary)
        radius of central slices of each segment given by ids.

        The data has to contain contains 1 pixel thick disk-like segments
        and that all disks have their principal axis along one of the
        coordinate axes.
        
        A center is calculated from a whole disk. A disk radius is calculated as
        the mean distance between the center and points that lay on the
        boundary of the corresponding ndim-1 dimensional disk. A ndim-1 disk is
        obtained from a ndim disk by keeping only a slice along axis that
        contains the disk center.

        If data is not given self.data is used. If ids is not given tries
        self.ids (if data was not given), or extracts ids from data (or
        self.data).

        Arguments:
          - data: (ndarray) segments
          - ids: segment ids
          - axis: the principal axis of disks (in 3d 0 is x axis, 1 is y and 2
          or -1 is z)

        Return (centers, radii):
          - centers: array of disk centers (shape n_disks, ndim) for segments
          with ids
          - radii: array od mean disk radii (shape n_disks) for segments given
          by ids
        """

        # parse arguments
        data, ids, update = self.parseInput(data=data, ids=ids)
        
        # find disc centers
        from .morphology import Morphology
        mor = Morphology(segments=data, ids=ids)
        mor.getCenter(real=True)
        centers = mor.center[ids]

        # make (empty) circles from discs
        circles = numpy.zeros_like(data)
        seg = Segment()
        for cent, id in zip(centers, ids):

            # extract the disc plane
            cent_slice = [slice(None)] * data.ndim
            cent_slice[axis] = int(numpy.rint(cent[axis]))
            cent_plane = data[tuple(cent_slice)].squeeze()

            # find boundary of the current disc (circle) and put it in circles
            circle = seg.makeSurfaces(data=cent_plane.copy(), size=1, ids=[id])
            circles[tuple(cent_slice)] += circle

        # find radii of circles for ids
        mor.getRadius(surface=circles)
        radii = mor.radius.mean[ids]

        return centers, radii

    def thinToMaxDiscs(self, data=None, ids=None, axis=-1, external=0):
        """
        Thins segments to a largest (volume) slice. That is, for each
        segment, all slices except the maximal one are removed.

        If arg ids is specified, only those segments are thinned. 

        Arguments:
          - data: (ndarray) segments
          - ids: ids of segments that need to be thinned
          - axis: the principal axis of disks (in 3d 0 is x axis, 1 is y and 2
          or -1 is z)
          - external: segment elements that are removed are set to this value
        """

        # parse arguments
        data, ids, update = self.parseInput(data=data, ids=ids)
        
        # find insets for all balls
        all_insets = scipy.ndimage.find_objects(input=data)

        # make a disk out of each balls
        for ball_id in ids:

            # check if the current ball exists
            if all_insets[ball_id-1] is None:
                logging.warning(
                    "Ball with id %d does not exist. Continuing." % id)
                continue

            # find volume slices along disc_axis
            curr_slices = None
            max_slice_sum = 0
            curr_sl = all_insets[ball_id-1][axis]
            for sl_ind in range(curr_sl.start, curr_sl.stop):

                # count elements of the current ball in the current slice
                curr_slices = list(all_insets[ball_id-1])
                curr_slices[axis] = slice(sl_ind, sl_ind+1) 
                curr_slice_sum = (data[curr_slices]==ball_id).sum()

                # remove elements of the 'smaller' slice
                if curr_slice_sum > max_slice_sum:

                    # current slice is maximal
                    if max_slice_sum > 0:
                        max_inset = data[max_slices]
                        max_inset[max_inset==ball_id] = external
                    max_slice_sum = curr_slice_sum
                    max_slices = curr_slices

                else:

                    # current slice is not maximal
                    curr_inset = data[curr_slices]
                    curr_inset[curr_inset==ball_id] = external

    @classmethod
    def make(cls, centers, radii, shape, dtype='int', ids=None, cut=None, 
             theta=None, check=True):
        """
        Makes an instance of this class containing balls defined by centers and
        radii. The balls are optionally cut by planes defined by (normal vector)
        cut and theta.

        A (cut) ball is generated for each element of centers and its
        corresponding element of radii and it is labeled by (a corresponding)
        id. If ids are not given the segments are labeled in the order starting
        from 1.

        Array elements whose distance to the center is <= radius are defined to
        belong to a ball.

        If neither cut nor theta arguments are None, each ball is cut by two
        planes. One is defined by a normal vector cut and an angle theta, where
        theta is the angle between the normal vector and points on the edge of
        the ball - plane intersection in respect to the ball center (theta 0
        means nothing and >pi/2 means everything is cut out). This angle is
        used to determine a point that belongs to the plane:

          radius * cos(theta) * cut / abs(cut) + radius

        so that the plane can be properly defined (by the normal vector and the
        point).

        Array elements that belong (exactly) to the plane, or are on the
        negative (oposite direction from the normal vector) are retained, while
        those that are on the positive side are removed. The other plane is
        symetrical to the first one with respect to the center.
         
        Arguments:
          - centers: array containing centers of balls (segments) listed in ids,
          shape (n_balls, ndim)
          - radii: array containing radii for segments listed in ids (shape
          n_balls)
          - shape: shape of the data array that holds the circles
          - dtype: dtype of the data array that holds the circles 
          - ids: ids of segments (balls)
          - cut: normal vector of the plane that cuts each ball
          - theta: angle that determines how much of each ball is cut (nothing
          is cut for 0, while if theta is numpy.pi/2 each of the two planes cuts
          a half of a ball)
          - check: if True, checks if balls overlap (uses findOverlaps()) 

        Returns instance of this class containing the balls, with other
        elements set to 0.  
        """

        # parse arguments
        if (ids is None) or (len(ids) == 0) or (max(ids) < 1):
            ids = list(range(1, len(centers)+1))

        # round centers to nearest int and calculate rounding errors
        centers_round = numpy.rint(centers).astype(int)
        centers_err = centers_round - centers

        # check if distances between centers > sum of radii
        if check:
            overlaps = cls.findOverlaps(ids=ids, centers=centers_round, 
                                        radii=radii)
            if len(overlaps) > 0:
                logging.warning(
                    "The following disks ovelap when expanded to balls: "
                    + str(overlaps))

        # put centers in a data array and make an instance
        data = numpy.zeros(shape=shape, dtype=dtype)
        for cent, id in zip(centers_round, ids):
            data[tuple(cent)] = id
        inst = cls(data=data, copy=False, ids=ids, clean=False)

        # find top and bottom positions of balls 
        if (cut is not None) and (theta is not None):
            radii = numpy.asarray(radii)
            cut = numpy.asarray(cut)
            cut_norm = cut / numpy.sqrt(numpy.inner(cut, cut))
            displac = numpy.outer(radii, cut_norm) * numpy.cos(theta)
            top = centers + displac
            bottom = centers - displac

        # make and cut balls 
        max_radius = numpy.rint(max(radii) + 0.999)
        mark_dist_gen = inst.markDistance(size=max_radius)
        for index in range(len(ids)):
            distance, slice_, id_ = next(mark_dist_gen)

            # ToDo (?) correct distnce for rounding error of centers

            # make a ball
            ball = numpy.where(distance<=radii[index], id_, 0)

            # cut top and bottom of the ball
            if (cut is not None) and (theta is not None):

                # make top and bottom planes 
                plane_top = Plane.generate_one(vector=cut, point=top[index],
                                               ndslice=slice_, thick=0)
                plane_bottom = Plane.generate_one(vector=-cut, 
                                                  point=bottom[index],
                                                  ndslice=slice_, thick=0)

                # cut the ball with the planes
                ball[(plane_top>0) | (plane_bottom>0)] = 0
                
            # add the ball to the instance
            # overwrites previous segments, use add instead? 
            inst.data[slice_] = numpy.where(ball>0, ball, inst.data[slice_])

        return inst
    
    @classmethod
    def findOverlaps(cls, ids, centers, radii, clearance=1):
        """
        Returns pairs of ids that would overlap when balls of the 
        cooresponding radii are centered to the corresponding centers

        Returns pairs of ids for which the Euclidean distance between centers 
        (distance):

          distance > r1 + r2 + clearance
        
        where r1 and r2 are the corresponding radii.

        Arguments ids, centers and radii have to have the 

        Arguments:
          - ids: list of ids
          - centers: list of coordinate centers (lists), or 2D array where 
          index 0 denotes different centers and index 1 the coordinates
          - radii: list of radii

        Returns list of id pairs 
        """

        # initialize
        centers = numpy.asarray(centers)
        over = []

        # loop over all combinations
        for (id1, c1, r1), (id2, c2, r2) in itertools.combinations(
                list(zip(ids, centers, radii)), 2):
            if numpy.sqrt(((c1 - c2)**2).sum()) < r1 + r2 + clearance:
                over.append([id1, id2])

        return over
