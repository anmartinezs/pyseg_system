"""
Contains class Distance for the calculation of distances between given segments.

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"


from copy import copy, deepcopy
import warnings

import numpy
import scipy
import scipy.ndimage as ndimage


class Distance(object):
    """
    Distance between given segments.

    The distances calculated are storred in an internal structure (currently 
    dict self.distances). However, it is preferable to use getDistance() and 
    setDistance() to access the data.

    Methods:
      - getDistance():
      - setDistance():
      - calculate(): calculates a distance

    Note that there is no garantee that the distance saved to the internal 
    structure are calculated from the same segmented image and using the same
    mode. That is because calculate() takes a segmented image (arg segment) and
    the distance mode (arg mode) as arguments.
    """

    #############################################################
    #
    # Initialization
    #
    #############################################################

    def __init__(self):
        """
        Initializes internal distance data structure.
        """
        self.dataNames = ['distance']
        self.initializeData()

    def initializeData(self):
        """
        Initializes data
        """
        self.distance = {}

    #############################################################
    #
    # Data manipulation
    #
    #############################################################

    def getDistance(self, ids):
        """
        Returns distance between segments given by id_1 and id_2. Returnes None
        if the distance between these two segments is not found.
        """
        id_1, id_2 = ids
        if (id_1 < id_2):
            ordered_ids = (id_1, id_2)
        else:
            ordered_ids = (id_2, id_1)
        return self.distance.get(ordered_ids)

    def setDistance(self, value, ids):
        """
        Sets distance between segments given by id_1 and id_2. 
        """
        id_1, id_2 = ids
        if (id_1 < id_2):
            ordered_ids = (id_1, id_2)
        else:
            ordered_ids = (id_2, id_1)
        self.distance[ordered_ids] = value

    def merge(self, new):
        """
        Puts data from another instance of this class (arg new) to this class.

        Arguments:
          - new: another instance
        """

        for ids, value in new.distance.items():
            self.setDistance(id_1=ids[0], id_2=ids[1], value=value)

    def calculate(self, segments, ids, force=False, mode='min'):
        """
        Calculates distance between segments specified by a segmented image and 
        their ids (args segments, id_1 and id_2). The calculation is based on
        Segment.distanceToRegion().

        If arg force is False, returns an already calculated value in case it
        exists.

        The distance calculated is saved in the internal distances structure.

        Arguments:
          - segments: (Segment) segmented image
          - id_1, id_2: segment ids
          - force: flag indicating if the distance should be calculated even if
          it already exists
          - mode: 'center', 'min', 'max', 'mean' or 'median'

        Returns: distance

        ToDo: 
          - see about expanding to multiple ids
          - add calculating all distances, but restrict to distances smaller
          than a given value
        """

        # shortcut
        id_1, id_2 = ids

        # check if calculated already
        if not force:
            dist = self.getDistance(ids=(id_1, id_2))
            if dist is not None:
                return dist

        # calculate
        distances = segments.distanceToRegion(ids=[id_1], regionId=id_2, 
                                              mode=mode)
        dist = distances[id_1]
        self.setDistance(ids=(id_1, id_2), value=dist)

        return dist

    def calculateBound(self, contacts, ids):
        """
        Calulcates distances between boundaries that are connected by segments
        specified by arg ids.

        Segments specified by ids that do not contact exactly two boundaries 
        are ignored.

        Arguments:
          - contacts: (Contact) object that defines contacts between boundaries
          and segments
          - ids: (list, ndarray) segment ids
        """
        
        warnings.warn("This method is not implemented yet. Use "
                      + "BoundaryDistance.calculate() instead.")
        
