"""

Contains class DistanceTo for calculations of distance between segments of
a segmented images and given region(s)


# Author: Vladan Lucic (MPI for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
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

from .features import Features


class DistanceTo(Features):
    """
    Distance from segments to specified region(s)

    Important attributes:

      - distance: distance for each segment
      - closestRegion: id of the closest region, for each segment

    Basic usage:

      # caclulate
      dist = DistanceTo(segments=segment_object)
      dist.getDistance(regionIds, regions)
    
      # show results
      dist.distance
      dist.closestRegion

    Note: unlike other classes that inherit from Features, the data of this
    class is internally storred in a compact form. That is, elements of 
    self.ids and self._distance directly correspond to each other, both arrays
    are compact. On the other hand, self.distance is dynamically generated 
    (from self._distance) and is indexed by ids (self.distance[i] is the 
    distance for the i-th segment). Important consequence is that changing 
    self.ids effectively changes data (self.distance). The same is true for 
    self.closestRegion and self._closestRegion.

    
    """
    
    #############################################################
    #
    # Initialization
    #
    #############################################################

    def __init__(self, segments=None, ids=None):
        """
        Initializes attributes.

        Arguments:
          - segments: (Segment) segments
          - ids: segment ids (if None segments.ids used)
        """

        # set segments and ids
        #super(DistanceTo, self).__init__(segments=segments, ids=ids)
        self.segments = segments
        self._ids = ids
        if self._ids is None:
            if self.segments is not None:
                self._ids = segments.ids
        else:
            self._ids = numpy.asarray(ids)

        # local data attributes
        self.dataNames = ['distance', 'closestRegion']
        self.compact = True


    #############################################################
    #
    # Attributes
    #
    #############################################################

    def getDistance(self, ids=None):
        """
        Get distance to closest region for each segment. Requires self.ids
        to be set properly. If arg ids is specified it is used instead of
        self.ids.

        If ids is None, None is returned. If ids is [], 0-length ndarray is 
        returned.

        Argument:
          - ids: ids, if not specified self.ids is used
        """

        if ids is None:
            ids = self.ids

        if ids is None:
            dist = None
        elif len(ids) == 0:
            dist = numpy.array([])
        else:
            dist = numpy.zeros(self.maxId+1) -1
            dist[ids] = self._distance

        return dist

    def setDistance(self, distance, ids=None):
        """
        Sets distance to closest region for each segment. Requires self.ids
        to be set properly. If arg ids is specified it is used instead of
        self.ids.

        Argument:
          - distance: (ndarray or list) distances indexed by ids
          - ids: ids, if not specified self.ids is used
        """

        if ids is None:
            ids = self.ids

        if (distance is None) and (ids is None):
            self._distance = None
        else:
            dist = numpy.asarray(distance)
            self._distance = dist[ids]
        
    distance = property(fget=getDistance, fset=setDistance, 
                        doc='Distances from segment to their closest regions')

    def getClosestRegion(self, ids=None):
        """
        Gets closest region id for each segment. Requires self.ids
        to be set properly. If arg ids is specified it is used instead of
        self.ids. 

        If ids is None, None is returned. If ids is [], 0-length ndarray is 
        returned.

        Argument:
          - ids: ids, if not specified self.ids is used
        """
        if ids is None:
            ids = self.ids

        if ids is None:
            res = None
        elif len(ids) == 0:
            res = numpy.array([])
        else:
            res = numpy.zeros(max(ids)+1) -1
            res[ids] = self._closestRegion

        return res

    def setClosestRegion(self, closestRegion, ids=None):
        """
        Sets closest region id for each segment. Requires self.ids
        to be set properly. If arg ids is specified it is used instead of
        self.ids.

        Argument:
          - closestRegion: (ndarray or list) closest region ids indexed by 
          self.ids (segment ids)
          - ids: ids, if not specified self.ids is used
        """
        if ids is None:
            ids = self.ids

        if (closestRegion is None) and (ids is None):
            self._closestRegion = None
        else:
            dist = numpy.asarray(closestRegion)
            self._closestRegion = dist[ids]
        
    closestRegion = property(fget=getClosestRegion, fset=setClosestRegion, 
                             doc='ClosestRegions from segment to their ' \
                                 + 'closest regions')

    #############################################################
    #
    # Calculations
    #
    #############################################################

    def calculate(self, regionIds, regions=None, segments=None, 
                  ids=None, mode='mean', surface=None):
        """
        Calculates distance of each segment to its closest region.

        Regions are specified by (Image or ndarray) region and regionIds. If 
        (arg) region is not specifed this instance is used. If regionIds is not
        given, the region is defined as all positive elements of region array.

        Takes into account positioning of segments and regions.

        If surfaces > 0, only the surfaces (of thickness given by arg surface)
        of the segments are considered. Otherwise whole segments are taken into 
        account. In any case the full region is used.

        If mode is 'center', shortest distances between the region and segment
        centers (note that center may lay outside the segment) are calculated.

        If mode is 'min'/'max'/'mean'/'median', the shortest distance between
        each (surface, if arg surface is not None) element of segments and the
        region are calculated first, and then the min/max/mean/median of these
        values is found for each segment separately. Finally, the closest
        region for each segment is found. 

        If more than one region id is given (arg regionIds is a list with 
        >1 element) the distance between a segment and each region is 
        calculated first (according to the arg mode) and then the closest
        region is found. Note that in case of mean / median mode this means
        that the mean / median is calculated to one (closest) region.

        If ids are not given, distances to all segments are calculated. 

        If the distance to a segment can not be calculated (if the segments does
        not exist, for example) the result for that segment is set to -1.

        Uses self.getDistanceToRegions().

        Sets:
          - self.distance: ndarray of distances from each segment to its
          closest region (indexed by self.ids)
          - self.closestRegion: ndarray of closest region ids for each segment
          (indexed by self.ids)

        Arguments:
          - segments: (Segment)) segmented image, if None self.segments is used
          - ids: segment ids, if None self.ids is used
          - region: (Segments) regions
          - regionIds: (single int, list, tuple or ndarray) id(s) of the region
          - mode: 'center', 'min', 'max', 'mean' or 'median'
          - surface: thickness of segment surfaces 
        """

        # arguments
        if segments is None:
            segments = self.segments
        if ids is None:
            ids = self.ids

        # save arguments as attributes
        self.regionIds = regionIds
        self.mode = mode
        self.surface = surface

        # bring segments and regions to common inset
        seg_data = segments.makeInset(ids=ids, additional=regions, 
                                       additionalIds=regionIds, update=False)
        if regions is not None:
            reg_data = regions.makeInset(ids=regionIds, update=False, 
                            additional=segments, additionalIds=ids) 
        else:
            reg_data = seg_data

        # calculate distances to all regions
        all_dist = self.getDistanceToRegions(
            segments=seg_data, segmentIds=ids, regions=reg_data, 
            regionIds=regionIds, mode=mode, surface=surface)

        # find closest region for each segment and set attributes
        if all_dist is not None:
            if all_dist.ndim == 2:
                self._distance = numpy.min(all_dist, axis=0)
                id_pos = numpy.argmin(all_dist, axis=0)
                self._closestRegion = numpy.asarray(regionIds)[id_pos]
            else:
                self._distance = all_dist
                self._closestRegion = (
#                    numpy.zeros(segments.maxId+1, dtype=int) + regionIds
                    numpy.zeros(len(self.ids), dtype=int) + regionIds)
        else:
            self._distance = None
            self._closestRegion = None
            
    @classmethod
    def getDistanceToRegions(cls, segments, segmentIds, regions,
                             regionIds, mode='mean', surface=None):
        """
        Calculates distance of each segment to each region.

        Segments are specified by args (ndarray) segments and segmentIds. If
        segment ids has no elements None is returned. Regions are specified 
        by args (ndarray) regions and regionIds. If arg regionIds is None, 
        None is returned. Args segments and regions are ndarrays that are 
        expected to have the same shape.

        If surfaces > 0, only the surfaces (of thickness given by arg surface)
        of the segments are considered. Otherwise whole segments are taken into 
        account.

        If mode is 'center', shortest distances between a region and segment
        centers (note that center may lay outside the segment) are calculated.

        If mode is 'min'/'max'/'mean'/'median', the shortest distance between
        each (surface, if arg surface is not None) element of segments and the
        region are calculated first, and then the min/max/mean/median of these
        values is found for each segment separately. 

        If the distance to a segment can not be calculated (if the segments does
        not exist, for example) the result for that segment is set to -1.

        Uses scipy.ndimage.distance_transform_edt.

        Arguments:
          - segments: (ndarray) segments
          - ids: segment ids
          - region: (Segments) regions
          - regionIds: (single int, list, tuple or ndarray) region ids
          - mode: 'center', 'min', 'max', 'mean' or 'median'
          - surface: thickness of segment surfaces 

        Returns: distances (2D ndarray where axis 0 corresponds to regions
        and axis 1 to segments, shape=((len(regionIds), len(segmentIds)) 
        between each segment and each region.
        """

        # trivial cases
        if (regionIds is None):
            return None
        if (segmentIds is None) or (len(segmentIds) == 0):
            return None

        # extract surfaces if required
        if (surface is not None) and (surface > 0):
            from .segment import Segment
            tmp_seg = Segment()
            segments = tmp_seg.makeSurfaces(data=segments, 
                                            size=surface, ids=segmentIds)

        # deal with multiple region ids
        if isinstance(regionIds, (list, tuple, numpy.ndarray)):
            regionIds = numpy.asarray(regionIds)
            distances = \
                numpy.zeros(shape=(len(regionIds), len(segmentIds)), 
                            dtype='float')
            for reg_id, reg_id_index in zip(regionIds, list(range(len(regionIds)))):
                distances[reg_id_index,:] = \
                    cls.getDistanceToRegions(regions=regions, regionIds=reg_id,
                                   segments=segments, segmentIds=segmentIds, 
                                   mode=mode, surface=surface)
            return distances

        # calculate distance (from all elements) to the region
        if regionIds is None:
            distance_input = numpy.where(regions>0, 0, 1)
        else:
            distance_input = numpy.where(regions==regionIds, 0, 1)
        if (distance_input > 0).all():  # workaround for scipy bug 1089
            dist_array = numpy.zeros(shape=distance_input.shape) - 1
        else:
            dist_array = ndimage.distance_transform_edt(distance_input)

        # find distances to segments
        if mode == 'center':

            # distances to the segment centers
            from .morphology import Morphology
            mor = Morphology(segments=segments, ids=segmentIds)
            centers = mor.getCenter(real=False)            
            distances = [dist_array[tuple(centers[id_])] for id_ in segmentIds]

        elif mode == 'median':

            # median distance to segments
            distances = [numpy.median(dist_array[segments==id_]) \
                             for id_ in segmentIds]

        else:

            # any of ndarray methods (no arguments)
            try:
                distances = [getattr(dist_array[segments==id_], mode)() \
                                 for id_ in segmentIds]
            except AttributeError:
                raise ValueError("Mode ", mode, " was not recognized. It can ",
                         "be 'center', 'median' or any appropriate ndarray ",
                         "method name, such as 'min', 'max' or 'mean'.")

        distances = numpy.asarray(distances)
        return distances



            

        

            
