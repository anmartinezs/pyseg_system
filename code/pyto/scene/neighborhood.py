"""
Contains class Neighborhood for the gray-scale analysis of neighborhoods.

A neighborhood of a region on a segment is a subset of the segment whose 
elements lay not further than a given distance to the region. Technically a 
region is also a segment.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from builtins import object

__version__ = "$Revision$"


import logging
from copy import copy, deepcopy

import numpy

from ..util import scipy_plus

class Neighborhood(object):
    """
    Density (greyscale) analysis of neighborhoods of specified regions.

    Typical usage:

      Neighborhood.make()
    """

    ###############################################################
    #
    # Initialization
    #
    ##############################################################

    def __init__(self):
        """
        Initializes attributes
        """

        self.density = None
        self.segmentDensity = None
        self.hood = None


    ###############################################################
    #
    # 
    #
    ##############################################################

    @classmethod
    def make(cls, image, segments, regions, size, maxDistance=None, 
             distanceMode='min', removeOverlap=True):
        """
        Requires two different segmentations, one called segments and the other
        regions.

        Calculates density of the whole segments and of the subsets of segments 
        that lie in the vicinity of regions (neighbourhoods). Also does t-test
        on the mean grey values.

        See segmentation.Grey.getNeighborhoodDensity() for details.

        Returns an instance of this class with attributes:
          - density: (Density) gray scale statistics of neighborhoods
            - mean
            - std
            - min
            - max
            - volume
            - t_value
            - confidence
          - segmentDensity: (Density) gray scale statistics of whole segments
          - hood: (Segment) neighborhoods
        Each attribute of density and segmentDensity is an array indexed as 
        [segment_id, region_id], where region_id = 0 corresponds to all regions
        together and segment_id = 0 to all segments together.
        """

        # calculate whole segment densities
        seg_density = image.getSegmentDensitySimple(segments=segments)

        # calculate neighbourhood densities
        hood_density, hood = image.getNeighbourhoodDensity(segments=segments,
                 regions=regions, size=size, removeOverlap=removeOverlap,
                 maxDistance=maxDistance, distanceMode=distanceMode)

        # compress arrays to contain only segment and regions ids
        seg_ids = numpy.insert(hood_density.ids, 0, 0) 
        reg_ids = numpy.insert(hood_density.regionIds, 0, 0)
        mean = hood_density.mean[numpy.ix_(seg_ids, reg_ids)]
        std = hood_density.std[numpy.ix_(seg_ids, reg_ids)]
        vol = hood_density.volume[numpy.ix_(seg_ids, reg_ids)]
        mean_seg = numpy.expand_dims(seg_density.mean[seg_ids], 1)
        std_seg = numpy.expand_dims(seg_density.std[seg_ids], 1)
        vol_seg = numpy.expand_dims(seg_density.volume[seg_ids], 1)

        # calculate one-tailed t-test variance
        t_value, confidence = pyto.util.scipy_plus.ttest_ind_nodata(\
            mean_1=mean, std_1=std, n_1=vol,
            mean_2=mean_seg, std_2=std_seg, n_2=vol_seg)

        # expand to array indexed by segments and regions ids
        hood_density.t_value = numpy.zeros(
            shape=(segments.maxId+1, regions.maxId+1), dtype='float') - 1
        hood_density.confidence = numpy.zeros(\
            shape=(segments.maxId+1, regions.maxId+1), dtype='float') - 1
        for index in seg_ids:
            hood_density.t_value[seg_ids[index], reg_ids] = t_value[index, :] 
            hood_density.confidence[seg_ids[index], 
                                    reg_ids] = confidence[index, :] 

        # make instance
        neighborhood = cls()
        neighborhood.density = hood_density
        neighborhood.hood = hood
        neighborhood.segmentDensity = seg_density

        return neighborhood

