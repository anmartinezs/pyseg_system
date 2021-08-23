"""
Contains class Density for the calculation of densities of gray images along 
segments.

# Author: Vladan Lucic (MPI for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from builtins import zip
from builtins import range
#from past.utils import old_div
from past.builtins import basestring

__version__ = "$Revision$"

import sys
import logging
import inspect
from copy import copy, deepcopy

import numpy
import scipy
import scipy.ndimage as ndimage

from .features import Features
from pyto.core.image import Image
from .segment import Segment
from .statistics import Statistics
from .morphology import Morphology


class Density(Features):
    """
    Calculates basic statistics (mean, std, min, max) and volume for
    segments of a (typically greyscale) image.

    Attributes holding calculated data:
      - mean
      - std
      - min
      - max
      - volume

    Each of these data attributes is an ndarray indexed by a segment id (from 1 
    up). Elements at position 0 contain values for all segments taken together.

    Methods:
      - calculate(): calculated all data arrays
      - aggregate(): combines calculated data for more than one segment

    Usage:
      density = Density()
      density.calculate(image=grey_image, segments_segmented_image)
    """
    
    #############################################################
    #
    # Initialization
    #
    #############################################################

    def __init__(self):
        """
        Initializes segments and ids.
        
        Note (r527): don't want to accept segments as argument because super
        sets the argument segments the the actual array (and loses other info 
        like inset). 
        """

        # call super
        super(Density, self).__init__()

        # local attributes
        self.dataNames = ['mean', 'std', 'min', 'max', 'volume']
        


    #############################################################
    #
    # Calculations
    #
    ############################################################

    def calculate(self, image, segments, ids=None):
        """
        Calculates gray value mean, std, min, max and volume for each segment
        from arg segments specified in arg ids. The calculated values are saved
        as attributes named mean, std, min, max and volume, respectivly.

        If arg segments is not specified self.segments is used. If arg ids is
        not specified self.ids is used if found, or if not segments.ids is used.

        Uses minimal possible size of segments and image arrays to speed up
        the calculations, but these arguments are not changed. This method 
        respects positioning of image and segments (attribute inset) but 
        image.data has to contain the whole inset covered by segments.

        Arguments:
          - image: (core.Image) grayscale image
          - segments: (Segment) segmented image
          - ids: segment ids

        Sets attributes: 
          - mean, std (N-1 degrees of fereedom), min, max and volume to 
          calculated values
          - ids: to arg ids or segments.ids
        """

        self.setIds(ids=ids, segments=segments)
        

        # work on array insets 
        if self.maxId > 0:
            segments_data, segments_inset = segments.getDataInset()
            segments.makeInset(ids=self.ids)
            image_data, image_inset = image.getDataInset()
            image.useInset(inset=segments.inset, mode='abs')

        # calculate density statistics for each segment
        stats = Statistics(data=image.data, labels=segments.data, ids=self.ids)
        stats.calculate()

        # calculate volume
        mor = Morphology(segments=segments.data, ids=self.ids)
        mor.getVolume()
            
        # set attributes
        self.mean = stats.mean
        self.std = stats.std
        self.min = stats.min
        self.max = stats.max
        self.volume = mor.volume

        # revert to the original data
        if self.maxId > 0:
            segments.setDataInset(data=segments_data, inset=segments_inset)
            image.setDataInset(data=image_data, inset=image_inset)

    def aggregate(self, ids):
        """
        Returns a new instance with data corresponding to segments combined
        according to arg ids.

        Elements of ids defines segments in the new instance, element i of
        this list defines segment with id i+1 of the new instance. Each element
        of ids should be a list containing segment ids of the current instance
        that are to be taken together.

        For example:

          aggregate(ids=[[1,2], [3,6,7]])

        returns instance with ids 1 and 2, where the new segment 1 contains data
        for actual segments 1 and 3 together, and the new segment 2 for actual
        3, 6 and 7.

        Note that segment ids can be repeated in the argument ids and that not
        all ids need to be present in the argument ids. In those cases 
        0-elements of data vectors might be meaningless.

        Argument:
          - ids: (list of lists) segment ids

        Returns: new instance of this class.
        """

        # make new instance
        new = self.__class__()
        new_len = len(ids) + 1
        new.setIds(ids=list(range(1, new_len)))

        # initrialize new data arrays
        new.mean = numpy.zeros(new_len) - 1
        new.std = numpy.zeros(new_len) - 1
        new.min = numpy.zeros(new_len) - 1
        new.max = numpy.zeros(new_len) - 1
        new.volume = numpy.zeros(new_len, dtype=int) - 1
        sum_sq = numpy.zeros(new_len) - 1

        # agregated segments
        for old_ids, new_id in zip(ids, new.ids):
            new.volume[new_id] = self.volume[old_ids].sum()
            new.mean[new_id] = (
                numpy.inner(self.mean[old_ids], self.volume[old_ids])
                / new.volume[new_id].astype(float))
            new.min[new_id] = self.min[old_ids].min()
            new.max[new_id] = self.max[old_ids].max()

            # std
            ddof = 0.
            sum_sq[new_id] = \
                ((self.std[old_ids] ** 2 * (self.volume[old_ids] - ddof))
                 + self.mean[old_ids] ** 2 * self.volume[old_ids]).sum()
            new.std[new_id] = numpy.sqrt(
                (sum_sq[new_id] - new.mean[new_id] ** 2 * new.volume[new_id])
                / (new.volume[new_id] - ddof))

        # set total (index 0) values
        new.setTotal()

        return new

    def restrict(self, ids):
        """
        Effectivly removes data that correspond to ids that are not specified 
        in the arg ids. Arg ids should not contain ids that are not in self.ids.

        Sets self.ids to ids and recalculates totals (index 0). Currently
        it doesn't actually remove non-id values from data.

        Argument:
          - ids: new ids
        """

        # set ids
        super(Density, self).restrict(ids=ids)

        # remove non-id values?

        # set total
        self.setTotal()

    def calculateTotal(self):
        """
        Calculates total values based on the individual segments data 
        and saves them at index 0.

        Acts on attributes min, max, volume, mean and std, if in self.dataNames.

        Doesn't do anything if there are no ids (self.ids).
        """

        # don't do anything if no ids
        if len(self.ids) == 0: return 

        if 'min' in self.dataNames:
            self.min[0] = self.min[self.ids].min()
        if 'max' in self.dataNames:
            self.max[0] = self.max[self.ids].max()
        if 'volume' in self.dataNames:
            self.volume[0] = self.volume[self.ids].sum()

        # mean
        if ('mean' in self.dataNames) and ('volume' in self.dataNames):
            self.mean[0] = (
                numpy.dot(self.mean[self.ids], self.volume[self.ids])
                / float(self.volume[0]))

        # std
        if (('std' in self.dataNames) and ('mean' in self.dataNames) 
            and ('volume' in self.dataNames)):
            variances = numpy.dot(
                (self.std[self.ids]**2 + self.mean[self.ids]**2),
                self.volume[self.ids])
            means = numpy.dot(
                self.mean[self.ids], self.volume[self.ids]) 
            self.std[0] = numpy.sqrt(
                variances / float(self.volume[0])
                - (means / float(self.volume[0]))**2)

    def merge(
            self, new, names=None, mode='replace', mode0='consistent',
            volume=None):
        """
        Merges data of this object new with the data of this instance.
        The values of all attributes listed in self.dataNames are added.

        The data attributes whose values are meged are those listed in names or
        in self.dataNames if names is None.

        If mode is 'add' the data is simply added. If mode is 
        'replace' the new values replace the old ones for id that are in 
        new.ids. 

        Recommended value for arg mode0 is 'consistent', because in this case
        index 0 of arrays mean, std min and max will contain appropriate values
        for all segments taken together.

        Arguments:
          - new: instance of Morphology
          - names: (list of strings, or a single string) names of data 
          attributes
          - mode: merge mode for data (indices 1+) 'add' or 'replace'
          - mode0: merge mode for index 0, 'consistent', 'add', 'replace' 
          or a numerical value
        """

        # 
        mode0_consistent = False
        if isinstance(mode0, basestring) and (mode0 == 'consistent'):
            mode0_consistent = True
            mode0 = -1

        # merge data listed in self.dataNames
        super(Density, self).merge(
            new=new, names=names, mode=mode, mode0=mode0)

        # set volume
        if (volume is None) and ('volume' in self.dataNames):
            volume = self.volume

        # calculate index 0 values if needed
        if mode0_consistent:
            #if 'min' in self.dataNames:
            #    self.min[0] = self.min[self.ids].min()
            #if 'max' in self.dataNames:
            #    self.max[0] = self.max[self.ids].max()
            #if 'volume' in self.dataNames:
            #    self.volume[0] = self.volume[self.ids].sum()
            #if ('mean' in self.dataNames) and (volume is not None):
            #    total_dens = numpy.dot(self.mean[self.ids], volume[self.ids]) 
            #    self.mean[0] = total_dens / float(self.volume[0])
            #if ('std' in self.dataNames) and (volume is not None):
            #    variances = numpy.dot(
            #        (self.std[self.ids]**2 + self.mean[self.ids]**2),
            #        self.volume[self.ids])
            #    means = numpy.dot(
            #        self.mean[self.ids], self.volume[self.ids]) 
            #    self.std[0] = numpy.sqrt(
            #        variances / float(self.volume[0]) 
            #        - (means / float(self.volume[0]))**2)
            self.calculateTotal()

    def setTotal(self):
        """
        Sets total values (index 0) for all data.

        In particular, total volume is the sum of all volumes, while total mean,
        std, min and max are values obtained for all segments taken together
        (and not mean of the segment means and so on. Takes into account only
        segments listed in self.ids.

        Sets index 0 of data arrays. 
        """

        ids = self.ids

        # don't do anything if no ids
        if (ids is None) or (len(ids) == 0):
            return

        # volume
        self.volume[0] = self.volume[ids].sum()

        # mean, min, max
        self.mean[0] = (
            numpy.inner(self.mean[ids], self.volume[ids])
            / float(self.volume[0]))
        self.max[0] = self.max[ids].max()
        self.min[0] = self.min[ids].min()

        # std
        ddof = 0
        sum_sq = (self.volume[ids] - ddof) * self.std[ids]**2 \
            + self.volume[ids] * (self.mean[ids] - self.mean[0])**2 
        self.std[0] = numpy.sqrt(sum_sq.sum() / (self.volume[0] - ddof))

