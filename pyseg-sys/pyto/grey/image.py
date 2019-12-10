"""
Contains class Image for manipulations of grey-scale images.


# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: image.py 1103 2014-12-29 11:36:27Z vladan $
"""

__version__ = "$Revision: 1103 $"

import logging
import numpy
import scipy
import scipy.ndimage as ndimage

from pyto.core.image import Image as BaseImage

class Image(BaseImage):
    """
    Manipulation of greyscale images.
    """

    #############################################################
    #
    # Initialization
    #
    ############################################################

    def __init__(self, data=None):
        """
        Saves data (image)

        Argument:
          - data: (ndarray) image
        """
        super(Image, self).__init__(data)
        

    #############################################################
    #
    # Image manipulations
    #
    ############################################################

    def limit(self, limit, mode, size=3):
        """
        Limits image data.

        Elements of self.data that are outside the limits (see below) are
        replaced by corrected values. A corrected value is obtained as a mean
        value of within-limits elements of a subarray of size given by argument
        size centered at the element to be corrected. For elements near the
        edges the subarray is shifted so that it does it still has the required
        size. If size is even the subarray is shifted towards higher indices
        in respect to the element to be corrected. 

        The low and high limit values are determined from limit and mode. If
        mode is 'abs', the limiting value(s) is (are) given in argument limit.
        If mode is is 'std', the limits are set to limit times image std
        away from the image mean.

        If limit is a single value, it is ised for both low and ligh limits.
        Alternatively, if a list of two elements is given for limits, it
        specifies the low and high limits.

        Arguments:
          - limit: used to determine the upper and the lower limits on image
          values
          - mode: mode used for the determination of the limits
          - size: size of the subarray used to determine the corrected values

        Updates self.data, that is overwerites the uncorrected image.
        """

        # Note: only marginal speedup with data.squeeze() 

        # determine low and high limits
        if mode == 'std':

            # limits expressed as std factors
            mean = self.data.mean()
            std = self.data.std()
            if isinstance(limit, list) or isinstance(limit, tuple):
                low_limit = mean - limit[0] * std
                high_limit = mean + limit[1] * std
            else:
                low_limit = mean - limit * std
                high_limit = mean + limit * std

        elif mode == 'abs':

            # absolute limits
            if isinstance(limit, list) or isinstance(limit, tuple):
                low_limit, high_limit = limit
            else:
                raise TypeError("Argument limit: " + str(limit) + " has to be "\
                                + "a list or a tuple in mode: " + mode + ".")

        else:
            raise ValueError("Mode: " + mode + " is not recognized.")
                
        # find array elements that are outside of the limits
        bad = numpy.zeros(shape=self.data.shape, dtype='bool')
        if low_limit is not None:
            bad = bad | (self.data < low_limit)
        if high_limit is not None:
            bad = bad | (self.data > high_limit)

        # correct the outsiders
        n_corr = 0
        n_uncorr = 0
        new = self.data.copy()
        bad_ind = bad.nonzero()           # much faster than ndenumerate 
        for ind in zip(*bad_ind):         # followed by if val != 0

            # find index limits so they don't extend outside data
            aind = numpy.array(ind)
            shape = numpy.array(self.data.shape)
            low_ind = numpy.maximum(aind - (size - 1) / 2, 0)
            high_ind = numpy.minimum(low_ind + size, shape)

            # enlarge limits on edges (needed?)
            correction = size - (high_ind - low_ind)
            low_ind = numpy.where(high_ind < shape, 
                                  low_ind, low_ind - correction)
            high_ind = numpy.where(low_ind > 0, high_ind, 
                                   high_ind + correction)

            # make index limit slices
            sl = [slice(l, h) for (l, h) in zip(low_ind, high_ind)]

            # correct data
            if numpy.logical_not(bad[sl]).sum() <= 0:
                logging.debug("Element " + str(ind) + 
                              " could not be corrected.")
                n_uncorr += 1
            else:
                mean = ndimage.mean(self.data[sl], bad[sl], index=0)
                new[ind] = mean
                n_corr += 1

        # update self.data
        self.data = new

        # log
        if n_corr > 0:
            logging.info("Corrected " + str(n_corr) + " image elements.")
        if n_uncorr > 0:
            logging.info("Could not correct " + str(n_uncorr) 
                         + " image elements.")

    def getStats(self, apixel=None, counte=None):
        """
        Calculates basic statistics for the data.

        If args apix and counte are specified also calculates mean electrons 
        per A^2.
        """
        self.mean = self.data.mean()
        self.min = self.data.min()
        self.max = self.data.max()
        self.var = self.data.var()
        self.std = self.data.std()

        # calculate mean electrons per A^2
        if (apixel is not None) and (counte is not None):
            conversion = apixel * apixel * counte
            self.mean_ea = self.mean / float(conversion)

        return

