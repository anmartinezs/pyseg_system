"""
Contains class Vector for basic manipulation of one or more N-dim vectors.
Currently, it contains methods for conversion between cartesian and spherical 
coordinates.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: vector.py 1324 2016-07-12 15:03:34Z vladan $
"""

__version__ = "$Revision: 1324 $"


import logging
from copy import copy, deepcopy
import numpy
import scipy


class Vector(object):
    """
    """

    ###############################################################
    #
    # Initialization
    #
    ##############################################################

    def __init__(self, data, oneD=False, coordinates='cartesian'):
        """
        Sets data structure to hold one or more vector given by arg data.

        If arg data is a 2D array it is understood as an array of vectors
        where indices along axis 0 designate vectors and indices along
        axis 1 coordinates.

        If arg data is a 1D vector of length N, it is uderstood as one
        ND vector if arg oneD is False, or as N 1D vectors if oneD is True.

        If arg coordinates is 'cartesian', the coordinates are x, y, z, ... .
        If it is 'spherical', the coordinates are r, phi, theta. 

        Sets attributes:
          - data: (2D ndarray) vector(s) in cartesian coordinates
          - one: flag indicating if only one vector is specified
        """

        if coordinates == 'cartesian':
            self.setCartesian(data=data, oneD=oneD)
        elif coordinates == 'spherical':
            self.setSpherical(data, oneD=oneD)
        else: 
            raise ValueError("Argument cooddinates: ", coordinates, " was not ",
                             "understood. Valid choices are 'cartesian' ",
                             " and 'spherical'.")

    def setCartesian(self, data, oneD=False):
        """
        Sets cartesian coordinates.
        """

        data = numpy.asarray(data)
        if data.ndim == 2:
            self.one = False
            self._data = data
        elif data.ndim == 1:
            if oneD:
                self.one = False
                self._data = data[:, numpy.newaxis]
            else:
                self.one = True
                self._data = data[numpy.newaxis, :]

    def setSpherical(self, data, oneD=False):
        """
        Sets spherical coordinates.
        """

        data = numpy.asarray(data)
        if data.ndim == 2:

            self.one = False
            self._data = self.sphericalToCartesian(data)

        elif data.ndim == 1:
            if oneD:

                self.one = False
                self._data = self.sphericalToCartesian(data[:, numpy.newaxis])

            else:

                self.one = True
                self._data = self.sphericalToCartesian(data[numpy.newaxis, :])
       

    ###############################################################
    #
    # Coordinates
    #
    ##############################################################

    def getData(self):
        """
        Returns cartesian coordinates of vector(s).
        """

        if self.one:
            return self._data[0]
        else:
            return self._data

    data = property(fget=getData, doc="Cartesian coordinates")

    def getR(self):
        """
        Returns absolute value of vector(s). 
        """

        res = numpy.sqrt((self._data ** 2).sum(axis=1))
        if self.one:
            return res[0]
        else:
            return res

    r = property(fget=getR, doc="Absolute value")

    def getPhi(self, units='radians'):
        """
        Returns angle phi, defined as atan(y/x), where x and y are rhe first 
        two coordinates.

        The angle is calculate in radians if arg units is 'radians', or in 
        degrees if it is 'deg' or 'degree'.

        Retruns an angle if there's only one vector, ar an array of angles if
        more that one angle.
        """

        # calculate
        res = numpy.arctan2(self._data[:, 1], self._data[:, 0])
        if (units == 'deg') | (units == 'degree'):
            res = self.radianToDeg(angle=res)

        if self.one:
            return res[0]
        else:
            return res

    phi = property(fget=getPhi, doc="Polar (or spherical) coordinate phi")

    def getPhiDeg(self):
        """
        Returns angle phi, defined as atan(y/x), where x and y are rhe first 
        two coordinates, in degrees.

        Retruns an angle if there's only one vector, ar an array of angles if
        more that one angle.
        """
        phi = self.getPhi(units='deg')
        return phi
        
    phiDeg = property(fget=getPhiDeg, 
                   doc="Polar (or spherical) coordinate phi in degrees")

    def getTheta(self, units='radians'):
        """
        Returns angle theta, defined as atan(r_sub, z) where z is the last 
        coordinate and r_sub is the absolute value of (current) vectors 
        taken without the last coordinate.

        The angle is calculate in radians if arg units is 'radians', or in 
        degrees if it is 'deg' or 'degree'.

        Retruns an angle if there's only one vector, ar an array of angles if
        more that one angle.
        """

        # absolute value of n-1 dim
        subvector = self.__class__(data=self._data[:, :-1])
        sub_r = subvector.r

        # n-dim
        res = numpy.arctan2(sub_r, self._data[:, -1]) 
        if (units == 'deg') | (units == 'degree'):
            res = self.radianToDeg(angle=res)

        if self.one:
            return res[0]
        else:
            return res

    theta = property(fget=getTheta, doc="Spherical coordinate theta")

    def getThetaDeg(self):
        """
        Returns angle theta, defined as atan(r_sub, z) where z is the last 
        coordinate and r_sub is the absolute value of (current) vectors 
        taken without the last coordinate in degrees.

        Retruns an angle if there's only one vector, ar an array of angles if
        more that one angle.
        """
        theta = self.getTheta(units='deg')
        return theta

    thetaDeg = property(fget=getThetaDeg, 
                     doc="Spherical coordinate theta in degrees")


    ###############################################################
    #
    # Conversions
    #
    ##############################################################

    def radianToDeg(self, angle):
        """
        Converts one or more angles from radians to degrees.
        """
        return 180 * angle / numpy.pi 
     
    def degToRadian(self, angle):
        """
        Converts one or more angles from degrees to radians.
        """
        return numpy.pi * angle / 180.
   
    def sphericalToCartesian(self, data):
        """
        Converts vectors given in spherical coordinates to the cartesian. 

        Argument:
          - data: (2D ndarray) with values r, phi, theta (if 3D) along axis 1)
        """

        #
        r = data[:, 0]

        if data.shape[1] == 1:

            result = [r]

        if data.shape[1] == 2:

            phi = data[:, 1]
            x = r * numpy.cos(phi)
            y = r * numpy.sin(phi)
            result = [x, y]

        elif data.shape[1] == 3:

            phi = data[:, 1]
            theta = data[:, 2]
            x = r * numpy.cos(phi) * numpy.sin(theta)
            y = r * numpy.sin(phi) * numpy.sin(theta)
            z = r * numpy.cos(theta)
            result = [x, y, z]

        else:

            raise ValueError("Sorry, don't know how to deal with spherical ",
                             "coordinates in ", data.shape[1], " dimensions.")

        return numpy.array(result).transpose()


    ###############################################################
    #
    # Other methods
    #
    ##############################################################

    @classmethod
    def angleBetween(cls, vec_1, vec_2, units='radians'):
        """
        Calculates angle(s) between two (sets of) vectors. The angle(s)
        are returned in radians or degrees, depending on the arg units. 

        Arguments:
          - vec_1, vec_2: two (sets of vectors), can be ndarrays or instances
          of this class
          - uniits: 'radians' or 'deg'

        Returns ndarray of angles. 
        """

        # get data
        if isinstance(vec_1, cls):
            vec_1 = vec_1.data
        if isinstance(vec_2, cls):
            vec_2 = vec_2.data

        # convert
        vec_1 = numpy.asarray(vec_1)
        if len(vec_1.shape) == 1:
            vec_1 = numpy.expand_dims(vec_1, 0)
        vec_2 = numpy.asarray(vec_2)
        if len(vec_2.shape) == 1:
            vec_2 = numpy.expand_dims(vec_2, 0)

        # calculate
        norm_dot = ((vec_1 * vec_2).sum(axis=1) / 
                    numpy.sqrt(
                        (vec_1 * vec_1).sum(axis=1) * 
                        (vec_2 * vec_2).sum(axis=1)))
        angle = numpy.arccos(norm_dot)
        if units == 'deg':
            angle = 180 * angle / numpy.pi

        return angle

    def bestFitLine(self):
        """
        Determines the best fit line for points specivfed by self.data.

        The best fit line minimizes the sum of squared distances from each 
        point to the line. It is calculated using singular value decomposition.
        Note that there is an global sign ambiguity in the calculated line.

        Returns: cm, direction
          - cm: (Vector) a point on the best fit line, specifically the center  
        of mass (mean) of all points
          - direction: (Vector) direction (vector) of the best fit line
        
        """

        cm = self.data.mean(axis=0)
        cm_wrap = Vector(cm)

        u, lamb, v_star = scipy.linalg.svd(self.data - cm)
        direction = v_star.transpose()[:, lamb.argmax()]
        
        direction_wrap = Vector(direction)

        return cm_wrap, direction_wrap

        
