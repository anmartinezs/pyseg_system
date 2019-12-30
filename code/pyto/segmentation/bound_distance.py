"""
Contains class BoundDistance for the calculation of distances between 
boundaries that contact the same segment.

Applicable only to the segments that contact exactly two boundaries.

# Author: Vladan Lucic
# $Id: bound_distance.py 923 2013-01-18 17:22:23Z vladan $
"""

__version__ = "$Revision: 923 $"


from copy import copy, deepcopy
import warnings

import numpy
import scipy
import scipy.ndimage as ndimage

from features import Features

class BoundDistance(Features):
    """
    Distance between boundaries of given segments.

    Important methods:
      - calculate(): calculates distances

    Attributes holding calculated data (ndarray indexed by segment id):
      - distance: distance between boundaries

    Other important attributes:
      - ids: (list) segment ids

    ToDo: Also save boundary ide for each segments (they're already calculated 
    in calculate()). As these need to be saved in a 2d ndarray, need to make 
    sure that all data-handling methods would work with 2d data. 
    """

    ######################################################
    #
    # Initialization 
    #
    #######################################################
    
    def __init__(self, segments=None, ids=None):
        """
        Initialization of id and data structures.

        If ids is not None, it is saved as self.ids. In addition, data structure
        (self.distance) is set to an ndarray of the appropriate size (maxId+1)
        with values -1 (self.default).

        Note: if arg segments is specified here, it is saved as an attribute,
        so it might significantly increase the size of this object when
        pickled. The alternative is to pass arg segment only in methods that
        use it (but do not save it).

        Arguments:
          - segment: (Labels, or ndarray) segmented image
          - ids: (list, ndarray) ids of segments
        """

        # default distance value
        self.default = -1

        # call super to deal with segments and ids
        super(BoundDistance, self).__init__(segments=segments, ids=ids)

        self.dataNames = ['distance']  # actually not used
        self.initializeData(ids=ids, value=self.default)

    def initializeData(self, ids=None, value=None):
        """
        Initializes id and data structures.

        If ids is not None, it is saved as self.ids. In addition, data structure
        (self.distance) is set to an ndarray of the appropriate size (maxId+1)
        with values -1.

        """

        if value is None:
            value = self.default

        if ids is None:
            self.ids = numpy.array([], dtype=int)
            self.distance = numpy.zeros(0)

        else:
            self.ids = numpy.asarray(ids)
            self.distance = numpy.zeros(self.maxId+1) + value


    ######################################################
    #
    # Basic manipulations of data structures
    #
    #######################################################
    
    def extend(self, ids):
        """
        Extends the data structure to be able to hold the data for all specified
        segment ids. Attribute ids is also updated.

        Note: Should be used only in methods that calculate distances for the 
        ids specified here. That is because using this method alone updates ids
        but leaves the corresponding data at the default, so it makes ids and 
        data are inconsistent.

        Argument:
          - ids: (list, ndarray) segment ids
        """

        # deal with ids being None
        if ids is None:
            return

        # add new ids
        self.ids = numpy.union1d(self.ids, ids)

        # extend data structure
        curr_data_len = self.distance.shape[0]
        if self.maxId+1 > curr_data_len:
            data = numpy.zeros(self.maxId+1) + self.default
            data[0:curr_data_len] = self.distance
            self.distance = data

    def merge(self, new, mode='replace'):
        """
        Merges data from another instance of this class (arg new) with this 
        instance. In case of conflict the data from the other instance is used.

        The reasons why this method is used instead of the one inherited from 
        Features is that this one is simpler and that here the default distance
        value can be set to -1 instead of 0.

        Arguments:
          - new: another instance of this class
          - mode: merge mode, currently only 'replace'
        """
        
        # deal with new being None
        if new is None:
            return

        # prepare for the merge
        self.extend(ids=new.ids)

        # merge
        if mode == 'replace':
            self.distance[new.ids] = new.distance[new.ids]

        else:
            raise ValueError("Argument mode can only have value 'replace'.")


    ######################################################
    #
    # Calculations
    #
    #######################################################
    
    def calculate(self, contacts, boundaries, ids=None, mode='min', 
                  extend=False):
        """
        For each segment specified by segment id (arg ids) calculates the 
        distance between the two boundaries (given by arg boundaries)
        that contact the segment.

        Segments that do not have exactly two contacting boundaries are ignored.
        If arg extend is False, ids of these segments are not added to self.ids.
        If arg extend is true and the arg ids is specified, all ids from arg 
        ids are added to self.ids and the elements of distance structure 
        corresponding to these ids are set to the default value.

        Based on Segment.distanceToRegion().

        Arguments:
          - contacts: (Contact) object that defines contacts between boundaries
          and segments
          - ids: (list, ndarray) segment ids
          - mode: distance mode, can be 'min', 'max', 'mean' or 'median'
          or 'center'
        """

        # get all segment ids if not specified
        if ids is None:
            arg_ids = False
            ids = contacts.segments
        else:
            arg_ids = True

        # get distance for all segments
        new = self.__class__()
        for seg_id in ids:

            # get boundaries and check that only 2
            bound_ids = contacts.findBoundaries(segmentIds=seg_id)
            if len(bound_ids) !=2:
                continue
            
            # update ids
            new.extend(ids=[seg_id])

            # calculate distance
            dists = boundaries.distanceToRegion(
                regionId=bound_ids[0], ids=[bound_ids[1]], mode=mode)
            new.distance[seg_id] = dists[bound_ids[1]]

        # update this instance
        self.merge(new=new)

        # add unused segment ids if needed
        if extend and arg_ids:
            self.extend(ids=ids)
