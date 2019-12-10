"""
Class Shapes provides methods for crating and manipulating geometrical shapes,
in addition to methods defined in its base class Segment.

# Author: Vladan Lucic, last modified 29.09.07
# $Id: shapes.py 1001 2013-12-09 09:41:11Z vladan $
"""

import scipy
import scipy.ndimage as ndimage
import numpy
import sys
import logging
import inspect
from segment import Segment

class Shapes(Segment):
    """
    Provides methods for crating and manipulating geometrical shapes,
    in addition to methods defined in its base class Segment.
    """

    def __init__(self, data=None, copy=True, ids=None, clean=False):
        """
        Calls Segment.__init__ to initialize data, data-related and id-related
        attributes.
        """
        super(Shapes, self).__init__(data, copy, ids, clean)


    #################################################################
    #
    # Methods
    #
    ###################################################################

    @staticmethod
    def makeSphere(center, radius):
        """
        Makes array containing spheres with centhers and radii given by arrays
        center and radius. The spheres are labeled by indices of the center.

        The rest of the array is set to 0. 0 elements of center and radius
        are ignored.

        Arguments:

        Returns an instance of this class.
        """

        # promote radius to array if needed

        # make mask for distances

        # make spheres

        # make instance
