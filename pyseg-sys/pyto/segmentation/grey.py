"""
Contains class Gray for segmentation related analysis of grayscale (3D) images.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: grey.py 1061 2014-10-10 15:30:53Z vladan $
"""

__version__ = "$Revision: 1061 $"


import logging
from copy import copy, deepcopy
import itertools

import numpy
import scipy
import scipy.ndimage as ndimage

from pyto.grey.image import Image
#from density import Density
from statistics import Statistics
#from segment import Segment
#from morphology import Morphology


class Grey(Image):
    """
    Basic segmentation and analysis of grayscale images.

    Methods that create segment:
      - doThreshold(): thresholding
      - labelByBins(): binning based segmentation

    Methods that calculate statistics of segment densities (grey values):
      - getSegmentStats(): basic statistics for density (grey-values) for each 
      segment
      - getSegmentDensity(): like getSegmentStats(), but also statistics for 
      background and total 
      - getSegmentDensitySimple(): like getSegmentStats(), but also gets volume 
      - getNeighborhoodDensity(): density stats for parts of segments that
      are in the neighborhoods of given regions

    Other methods:
      - extractSingle():

    Important attributes:
      - data: (ndarray) n-dim greyscale image
    """

    ##################################################################
    #
    # Initialization
    #
    ##################################################################
        
    def __init__(self, data=None):
        """
        Initializes attributes and sets data.

        Argument:
          - data: (ndarray) greyscale image
        """
        super(Grey, self).__init__(data)


    ##################################################################
    #
    # Segmentation
    #
    ##################################################################
        
    def doThreshold(self, threshold, pickDark=True):
        """
        Basic thresholding of self.data.
        
        Returns a Segment object whose attribute data is the boolean array 
        (of the same size as self.data) with elements <= threshold labeled
        True. If pickDark is False, elements >= threshold are True.

        Arguments:
          - threshold: threshold
          - pickDark: if True elements below threshold are labeled, otherwise
          the element above thresholds are labeled
        """

        # set attributes and check if data exist
        self.threshold = threshold
        self.pickDark = pickDark

        # threshold
        try:
            if pickDark: 
                t_data = (self.data <= threshold)  
            else: 
                t_data = self.data >= threshold
        except AttributeError:
            print "Data not found."
            raise

        # make a Segment instance and preserve offset and inset
        from segment import Segment
        seg = Segment(data=t_data, copy=False, clean=False)
        seg.copyPositioning(self, saveFull=True)

        return seg

    @classmethod
    def labelByBins(cls, values, bins):
        """
        Segments image(s) (arg values) according to arg bins.

        If arg bins is a list of numbers, arg values represents a single image
        to be segmented. Individual bins consist of pairs of consequtive 
        elements of arg bins and are labeled starting wih 1 (increment 1).
        The segmentation consists of assigning a bin id to each element of 
        array value according to the value of the element. The lower bin limit
        is inclusive and the higer exclusive, except for the last binn where 
        the higher bin limit is also inclusive. In addition to the segmented 
        image, a dictionary relating bins and ids (ids a keys and [bin_start, 
        bin_end] are the values) is returned. This is called 1-parameter 
        segmentation.

        If arg bins is a list of list, where each inside list contain numbers,
        arg values has to be composed of a number of images, one for each bin
        element stacked along the axis 0. The shape of arg values is in this 
        case (k, n0, n1, ...) where (n0, n1, ...) is the image shape and k is 
        the number of sublists of arg bin. This is called multi-parameter 
        segmentation, it is a product of k 1-parameter segmentations described
        above. Regarding the id assignment, the last element of bins changes
        the fastest. In addition to the segmented image (shape (n0, n1, ...)),
        a dictionary relating bins and ids (ids a keys and a list of
        [bin_start, bin_end] representing a particular bin product are the 
        values) is returned.

        In the multi-parameter case, it is also possible that the number of
        parameters in arg bins (length o bins) is smaller than the number 
        or stacked images of arg values. In this case the extra stacked images
        are ignored.

        Arguments:
          - values: (ndarray, or masked ndarray) 1- or multi-parameter image 
          to be segmented
          - bins: (list tuple or ndarray) binning values for 1-paramater 
          segmentation, or a list of lists (or tuples or ndarrays) where each 
          sublist contains binning values (multi-parameter). Binning values
          have to be in ascending order.

        Returns: (labels, bin_ids)
          - labels: (ndarray) labeled image
          - bin_ids: dictionary that associated ids with bin limits
        """

        if all([isinstance(x, (list, tuple, numpy.ndarray)) for x in bins]):
            
            # values in multi-parameter 
            if values.shape[0] < len(bins):
                raise ValueError("Number of bins has to be equal or smaller "
                                 + "than the ")

        else:

            # values in 1-parameter
            values_data = numpy.expand_dims(values, axis=0)
            if numpy.ma.isMaskedArray(values):
                values = numpy.ma.array(values_data, mask=values.mask)
            else:
                values = values_data
            labels, bin_ids = cls.labelByBins(values, [bins])
            bin_ids = dict([(key, value[0]) for key, value in bin_ids.items()])
            return labels, bin_ids

        # make explicit bins and max values for each binning
        real_bins = [[[one_bins[ind], one_bins[ind+1]] 
                      for ind in range(len(one_bins)-1)]
                     for one_bins in bins]
        bin_max = [one_bins[-1] for one_bins in bins]

        # initialize variable for the loop
        new_shape = list(values.shape)[1:]
        n_dim = values.shape[0]
        id_ = 0
        labels = numpy.zeros(shape=new_shape, dtype=int)
        bin_ids = {}

        # assign labels from lowest bins up
        for combination in itertools.product(*real_bins):

            # label for the current combination
            tmp_labels = numpy.ones(shape=new_shape, dtype=bool)
            for one_bin, index in zip(combination, range(n_dim)):
                tmp_labels = tmp_labels & (values[index] >= one_bin[0])

            # update
            id_ += 1
            labels[tmp_labels] = id_
            bin_ids[id_] = combination

        # limit lables for highest bins
        for one_max, index in zip(bin_max, range(n_dim)):
            labels[values[index] > one_max] = 0

        # make sure masked elements are 0
        if numpy.ma.isMaskedArray(values):
            labels[values.mask[0]] = 0

        return labels, bin_ids


    ##################################################################
    #
    # Data statistics
    #
    ##################################################################
        
    def getSegmentDensity(self, segment=None, ids=None, offset=None):
        """
        Calculates basic statistics for the grey values (densitity) of segmented
        data.

        The statistical analysis is done for each segment separately, segment
        density means, background and total. This method returns four 
        corresponding instances of Statistics. Each instance contains 
        attributes mean, std, min, max, minPos and maxPos. The type of these 
        attributes for different statistisc is as follows:
          - each segment of data separately: ndarray indexed by ids, where 0-th
          element is the results for all segments taken together
          - segment density means: numbers
          - background: numbers
          - total: numbers

        First parses the segment and ids arguments and sets appropriate
        attributes (self.segment, self.ids, self.segmentOffset).

        Arrays self.data and self.segment (set from segment) are intersected
        taking into account self.segmentOffset The statistical analysis is
        done on the intersection in all cases.

        Arguments:
          - segments: segmented image, either as an Segment object or an ndarray
          - ids: array (or other iterable) of segment ids. If None, the
          analysis is done for segment.ids.

        Returns:
          - Statistics objects containg results for density, meanDensity,
          background, total (in this order). Each of them has attributes: mean,
          std, min, max, minPos, maxPos.
        """

        # make Segment instance and set ids
        from pyto.segmentation.segment import Segment
        if isinstance(segment, numpy.ndarray):
            segment = Segment(data=segment, ids=ids, copy=False)
        if ids is None:
            ids = segment.ids
        else:
            ids = numpy.asarray(ids)

        # get an inset of self.data corresponding to segment
        restricted = self.usePositioning(image=segment, new=True)
        image_inset = restricted.data
            
        # get density for all segments 
        density = self.getSegmentStats(data=image_inset, segment=segment.data,
                                       ids=ids, offset=offset)

        # statistics on segment density means
        try:
            maxId = ids.max()
        except ValueError:
            maxId = 0
        idMask = numpy.zeros(maxId+1, dtype=int)
        idMask[ids] = 1
        meanDensity = self.getSegmentStats(data=density.mean, 
                                           segment=idMask, ids=1)

        # get background density
        # bkg = numpy.where(self.segmentInset==0, 1, 0)
        background = self.getSegmentStats(
            data=image_inset, segment=segment.data, ids=0, offset=offset)

        # get total density of the inset
        all = numpy.ones(shape=segment.data.shape, dtype=int)
        total = self.getSegmentStats(data=image_inset, segment=all,
                                          ids=1, offset=offset)

        # return all stats
        return density, meanDensity, background, total

    def getSegmentStats(self, data=None, segment=None, ids=None, offset=None):
        """
        Calculates basic statistics for each segment of data that is specified
        by ids.

        If data is not specified self.data is used. If segment is not specified
        whole data belong to one segment.

        If data is not specified and segment is an instance of Image, inset of 
        self.data corresponding to segment is used for calculations (to maximize
        speed and minimise memory usage). Otherwise, data and segment have to 
        have the same shape (and no offset betwee each other?). 

        If ids is a single int, the result attributes are single numbers (or
        1d arrays for min/maxPos). In the other case, if inds is an array (even
        if its length is 1), each result attribute is an array indexed by id
        along axis 0. Statistics for all segments taken together are at 0
        positions of the result arrays.

        The sigle int ids is meant for statistics on segment means. For
        statistics on single labels use an array with one element (ids=[2],
        for example)  

        Arguments:
          - data: image (data) array
          - segments: (Label or ndarray) segmented image
          - ids: array (or other iterable) of segment ids, or a single int.
          If None, the analysis is done for all segments (different integers
          in segment define different segments)
          - offset: offset of data and segment arrays in respect to self.data

        Changes attributes:
          - self.fullData
          - self.fullInset

        Returns:
          - Statistics object containg results (attributes: mean, std, min,
          max, minPos, maxPos). 
        """

        from segment import Segment

        # set segment related
        if segment is None: 
            if data is None:
                segment_data=numpy.ones(shape=self.data.shape, dtype=int)
            else:
                segment_data=numpy.ones(shape=data.shape, dtype=int)
            ids = 1
        else:
            if isinstance(segment, Segment):
                segment_data = segment.data
                if ids is None:
                    ids = segment.ids
            else:
                segment_data = segment

        # convert ids to ndarray
        if not isinstance(ids, (list, tuple, numpy.ndarray)):
            ids = numpy.array([ids])
        ids = numpy.asarray(ids)

        # adjust self and segment insets
        revert = False
        if data is None:

            if isinstance(segment, Segment):
                revert = True

                # save initial inset and data
                segment_init_inset = segment.inset
                segment_init_data = segment.data
                self_init_inset = self.inset
                self_init_data = self.data
                
                # make insets
                segment.makeInset(ids=ids)
                self.useInset(inset=segment.inset, mode='abs')

                # set vars to calculate density
                segment_data = segment.data

            # set data
            data = self.data
            
        # basic statistics on density
        stats = Statistics(data=data, labels=segment_data, ids=ids)
        stats.calculate()

        # revert self and segment to the initial state
        if revert:
            self.inset = self_init_inset
            self.data = self_init_data
            segment.inset = segment_init_inset
            segment.data = segment_init_data

        # include offset
        try:
            stats.minPos += offset
            stats.maxPos += offset
        except TypeError:
            pass

        return stats

    def getSegmentDensitySimple(self, segments, ids=None):
        """
        Calculates mean, std, min, max and volume of this instance for segments
        specified by args segment and ids.

        Respects insets of segment and self. 

        Arguments:
          - segments: (Segment) segments
          - ids: segment ids

        Returns (Density) density with attributes mean, std, min, max and volume
        """
        
        from morphology import Morphology 
        from density import Density

        # set ids
        if ids is None:
            ids = segments.ids

        # calculate
        stats = self.getSegmentStats(segment=segments, ids=ids)
        mor = Morphology(segments=segments, ids=ids)
        mor.getVolume()

        # put in a Density instance
        density = Density()
        density.setIds(ids=ids)
        density.mean = stats.mean
        density.std = stats.std
        density.min = stats.min
        density.max = stats.max
        density.volume = mor.volume

        return density

    def getNeighbourhoodDensity(self, segments, ids=None, regions=None,
                      regionIds=None, size=None, maxDistance=None,
                      distanceMode='min', removeOverlap=False):
        """
        Calculates basic statistics on grayscale values of those subsets of 
        segments that form neighborhoods of regions.

        To be precise, for each region (defined by args region and regionIds)
        and each segment (defined by args segments and ids) a neighborhood
        is determined, and a basic statistics is calculated for the grayscale
        values of the neighborhood from arg image. A neighborhood of a given
        region on a given segment is defined as a subset of the segment that 
        contains elements that are at most (arg) size away from the closest 
        segment element to the region, as long as the distance to the region 
        is not above (arg) maxDistance.

        The distance between a region and segments is calculated according to 
        the arg distanceMode. First the (min) distance between segments and 
        each point of regions is calculated. Then the min/max/mean/median of 
        the calculated distances, or the (min) distance between the region 
        center and the segments is used.

        If removeOverlap is True, parts of segments that overlap with regions are 
        removed from the calculated neighborhoods.

        Also calculates statistics for neighborhoods of each region where all 
        segments are taken together, and for all neighborhoods on each segment 
        taken together.

        Respects insets of segment, regions and self.

        If arg regions is None, or no region ids are specified statistics on 
        whole segments is calculated.

        All values for segments that are not specifed in ids, regions that are 
        not specifed in regionIds or are more than maxDistance away from segments 
        are set to -1.

        Arguments:
          - segments: (Segment) segments
          - ids: segment ids
          - regions: (segment) regions
          - regionIds: region ids
          - size: max distance between elements of a neighborhood to the element
          that is the closest to a region (if None whole segments are used)
          - maxDistance: stats are not calculated for regions that are more that
          this value away form (the closest) segment (None means no limit)
          - distanceMode: 'center', 'min', 'max', 'mean', or 'median'
          - removeOverlap: if True remove parts where segments and regions overlap

        Returns density, neighborhood:
          - density: (Density) has attributes: mean, std, min, max, volume. Each 
          attribute an 2D array indexed by segment ids (axis 0) and region ids
          (axis 1). Segment id 0 stands for neighborhoods at all segments of a 
          given region taken together, while region id 0 is for all heighborhoods 
          together on a given segment.
            - density.ids: segment ids
            - density.region_ids: ids of regions for which neighbourhood density 
            was calculated (those region ids that are closer than maxDistance to 
            segments).
          - neighborhood: (Segment) neighbourhoods labeled by segment ids
        """

        # here to avoid circular imports
        from density import Density

        # parse arguments
        if ids is None:
            ids = segments.ids
        ids = numpy.array(ids)
        if regions is not None:
            if regionIds is None:
                regionIds = regions.ids
            region_ids = numpy.array(regionIds)
        else:
            region_ids = None

        # prepare a Density object to hold the results
        density = Density()
        if len(ids) > 0:
            max_id = ids.max()
        else:
            max_id = 0
        if len(region_ids) > 0:
            max_region_id = region_ids.max()
        else:
            max_region_id = 0
        density.mean = numpy.zeros(shape=(max_id+1, max_region_id+1)) - 1
        density.std = numpy.zeros(shape=(max_id+1, max_region_id+1)) - 1
        density.min = numpy.zeros(shape=(max_id+1, max_region_id+1)) - 1
        density.max = numpy.zeros(shape=(max_id+1, max_region_id+1)) - 1
        density.volume = numpy.zeros(shape=(max_id+1, max_region_id+1)) - 1

        # calculate density stats and volume for all region neighborhoods
        reg_ids = []
        for reg_id, hood, all_hoods in \
                segments.generateNeighborhoods(ids=ids, regions=regions, 
                      regionIds=region_ids, size=size, maxDistance=maxDistance, 
                      distanceMode=distanceMode, removeOverlap=removeOverlap):
        
            # add region id
            reg_ids.append(reg_id)

            # calculate density statistics and volume for the current region
            simple = self.getSegmentDensitySimple(segments=hood, ids=ids)
            
            # put results in the Density object
            density.mean[:,reg_id] = simple.mean 
            density.std[:,reg_id] = simple.std
            density.min[:,reg_id] = simple.min
            density.max[:,reg_id] = simple.max
            density.volume[:,reg_id] = simple.volume

        # calculate stats and volume for all regions together on each segment
        simple = self.getSegmentDensitySimple(segments=all_hoods, ids=ids)

        # put results in data arrays
        density.mean[:,0] = simple.mean
        density.std[:,0] = simple.std
        density.min[:,0] = simple.min
        density.max[:,0] = simple.max
        density.volume[:,0] = simple.volume
        density.ids = copy(segments.ids)
        density.regionIds = numpy.asarray(reg_ids)

        return density, all_hoods


    ##################################################################
    #
    # 
    #
    ##################################################################
      
    def extractSingle(self, segments, size, ids=None, binning=0, copyData=True,
                      deepcp=True, noDeepcp=[], segmentNoDeepcp=[]):
        """
        Extracts and yields rectangular subsets of the current image (self.data)
        so that they contain individual segments, as well as the corresponding
        subsets of segments.

        If segments is binned in respect to the current image (binning > 0),
        extracts from segments are enlarged to be the same as extracts from 
        the current image.

        Greyscale image and segmented image have to have the same origin
        (no offset). However, the segmented image can be binned in respect to 
        the grey image. Also, image instances (self and segments) can conatain
        only relevant array insets, that is self.inset and segments.inset
        attributes are taken into account.
        
        Arguments:
          - segments: (Segment)
          - size: (int or a list or ndarray of ints) size of extracted images
          - ids: (int > 0)
          - binning:
          - copyData: frag indicating if the data of extracted images are copied
          from the current data
          - deepcp: flag indicating if a deepcopy or just a copy of the 
          current instance is used to make image extracts
          - noDeepcp: list of attribute names that are not deepcopied 
          from this instance to image extracts (used only if deepcopy is True) 
          - segmentNoDeepcp: list of attribute names that are not deepcopied 
          from segments to segment extracts

        Yields: (extracted_image, extracted_segment)
          - extracted_image: instance of the current class
          - extracted_segment: instance of the segments class

        Not tested
        """

        # set ids
        if ids is None:
            ids = segments.ids
        ids = numpy.asarray(ids)

        # get centers
        from morphology import Morphology 
        from segment import Segment 
        mor = pyto.segmentation.Morphology(segments=segments)
        seg_centers = mor.getCenter(real=True, inset=True)
        
        # change size so that segments can be cleanly extracted (segments
        # array may have different binning) 
        bin_factor = 2**binning
        size = numpy.asarray(size)
        size_increase = numpy.mod(size, bin_factor)
        if size_increase != 0:
            size += size_increase
            logging.info(("Size of extracted images changed to %d so that " \
                             + " the same size segments can be extracted.") \
                             % size)

        # loop 
        for id_ in ids:

            # convert center to (full size) image coordinates
            image_centers = \
                bin_factor * seg_centers + (bin_factor - 1) * 0.5

            # calculate extracted image begin and end positions
            begin = image_centers - (size - 1) / 2.
            begin = numpy.rint(begin).astype(int)
            end = begin + size

            # move the extracted image if it falls beyond the image
            correct_begin = numpy.zeros_like(begin) - begin
            correct_begin[correct_begin < 0] = 0
            begin = begin + correct_begin
            end = end + correct_begin

            # the same as above but for the image array end
            correct_end = end - list(image.data.shape)
            correct_end[correct_end < 0] = 0
            begin = begin - correct_end
            end = end - correct_end

            # adjust so that segment (at higher bining) can be extracted  
            bin_correction = numpy.mod(begin, bin_factor) 
            begin = begin - bin_correction
            end = begin + size

            # calculate extracted segment begin and end positions
            seg_begin = numpy.floor_divide(begin, bin_factor)
            seg_end = numpy.floor_divide(end, bin_factor)

            # check if the repositioning succeeded
            if (begin < 0).any():
                raise ValueError("Segment " + str(id_) + " can't be "
                             + "repositioned so that it fits inside the image.")

            # convert to slices
            image_inset = \
                [slice(begin_1, end_1) for begin_1, end_1 in zip(begin, end)]
            seg_inset = \
                [slice(begin_1, end_1) for begin_1, end_1 \
                     in zip(seg_begin, seg_end)]

            # extract particle
            particle = self.newFromInset(inset=image_inset, mode='abs',
                                     deepcp=deepcp, noDeepcp=noDeepcp)

            # extract segment
            seg = segments.newFromInset(inset=seg_inset, mode='abs',
                                        copyData=True, deepcp=True, 
                                        noDeepcp=segmentNoDeepcp)
            seg.setData(ids=[id_], clean=True)

            # yield particle and segment
            yield particle, seg
        
