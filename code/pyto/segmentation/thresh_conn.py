"""

Provides class ThreshConn for the analysis of multiple threshold and
connectivity based segementations, obtained at different thresholds.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import zip
from builtins import str
from builtins import range

__version__ = "$Revision$"


#from copy import copy, deepcopy
import logging
import numpy
import scipy

from .segment import Segment
from .connected import Connected
from .hierarchy import Hierarchy
from .morphology import Morphology
import pyto.util.nested as nested

class ThreshConn(Hierarchy):
    """
    """
    
    ###############################################################
    #
    # Initialization
    #
    ##############################################################

    def __init__(self, data=None, levelIds=[], higherIds={}):
        """
        Initializes id and data related attributes.

        Specifying args levelIds and higherIds is enough to make a functional 
        id-related attributes (data structures). In this case attributes
        ids, levelIds, _higherIds and _lowerIds are set. These arguments are 
        deepcopied, so they are not changed, nor they can be changed in this
        instance.

        Arguments:
          - levelIds: nested list of ids organized in levels
          - higherIds: dictionary of id : higher_id pairs
          - data: (ndarray) data array
        """

        # initialize super 
        super(ThreshConn, self).__init__(data=data, levelIds=levelIds, 
                                         higherIds=higherIds)


    ###############################################################
    #
    # Properties
    #
    ##############################################################

    def getThresh(self):
        """
        Returns ndarray containing thresholds for all segments. This
        array is indexed by self.ids and is therefore different from
        self.threshold.
        """

        # initialize the array filled with a ridiculous number
        ridiculous = 100 * numpy.asarray(self.threshold[0])
        thresh = numpy.zeros(self.maxId + 1, dtype='float') - ridiculous

        # get thresholds
        for level_ids, curr_thresh in zip(self.levelIds, self.threshold):
            for id_ in level_ids:
                thresh[id_] = curr_thresh

        return thresh

    thresh = property(fget=getThresh, 
                      doc='Array of thresholds for all ids, indexed by' \
                          + ' self.ids (different from self.thresholds)')

    def getLevelFrom(self, threshold=None, precision=1e-10):
        """
        Returns level corresponding to the specified threshold.
        """

        diff = numpy.asarray(self.threshold) - threshold
        good = numpy.abs(diff) <= precision
        level = good.nonzero()[0]
        if len(level) == 0:
            raise ValueError(
                "No level found for threshold " + str(threshold) 
                + " with precision " + str(precision) + ".")
        elif len(level) == 1:
            return level[0]
        else:
            raise ValueError(
                "More than one level (" + str(level) + ")"\
                + " found for threshold " + str(threshold) \
                + " with precision " + str(precision) + ".")
            

    ###############################################################
    #
    # Segmentation related methods
    #
    ##############################################################

    def setConnParam(self, boundary, boundaryIds=None, nBoundary=None,
                     boundCount='exact', mask=0, freeSize=0, freeMode='add', 
                     structElConn=1, contactStructElConn=1, 
                     countStructElConn=1):
        """
        Sets arguments that are needed for the segmentation based on
        thresholding and connectivity, that is for makeLevels and
        makeLevelsGen() methods.
        
        These arguments are passed directly to Connected.make method.
      
        Argument mask can have the same forms as in Segment.makeFree().
  
        Attribute self.connParam is set to a dictionary containing the above
        arguments.
        """

        # find boundaryIds
        if boundaryIds is None:
            boundaryIds = boundary.ids

        # make a segmentation (free) region 
        flat_b_ids = nested.flatten(boundaryIds)
        free = boundary.makeFree(ids=flat_b_ids, size=freeSize, 
                                 mode=freeMode, mask=mask, update=False)

        # put parameters in a dictionary
        self.connParam = {'boundary' : boundary,
                          'structElConn' : structElConn,
                          'contactStructElConn' : contactStructElConn,
                          'countStructElConn' : countStructElConn,
                          'boundaryIds' : boundaryIds,
                          'nBoundary' : nBoundary,
                          'boundCount' : boundCount,
                          'mask' : free,
                          'freeSize': -1}

        # set self.structEl
        self.setStructEl(rank=boundary.data.ndim, connectivity=structElConn)

    def makeLevels(self, image, thresh=None, label=True, props={},
                   check=False, shift=None):
        """
        Performs threshold and connectivity segmentation at specified thresholds
        and puts the resulting segments into this hierarchy. Each level of
        the hierarchy contains all segments obtained at one threshold.

        Image is thresholded, and segmented based on its contacts to regions
        defined in (arg) boundaries. The segmentation consists of finding all
        disjoint segments of thresholded image, analysing contacts between
        the segments and the boundaries and removing segments that do not
        satisfy the contact conditions. Segmentation related parameters need
        to be specified by calling self.setConnParam() before calling this
        method. Based on Connected.make method.
        
        Adds threshold and contacts to properties, so they can be accessed from
        self.threshold (list of all thresholds in the order of increasing 
        levels) and self.contacts (Contacts object containing info for contacts
        at all levels.) Other properties can be specified using argument
        properties. 

        At each level (threshold) segment ids are increased by shift*(level-1)
        if shift is not None, or by self.maxId otherwise.  

        Sets following properties:
          - self.trheshold: list of thresholds indexed by level
          - self.contacts: Contacts object containing contact info for all
          segments

        Arguments:
          - image: (grayscale) image to be segmented (ndarray or Image object)
          - thresh: list of thresholds
          - count: flag indicating if the nuber of contacts between each segment
          and boundary is found, or it is only determined if they have one 
          or more contacts
          - props: dictionary of additional properties
          - check: if True checks if levels fit correctly in the existing
          hierarchy
          - shift: segment ids shift

        Returns level if single threshold is given 

        ToDo: rewrite using makeLevelsGen
        """

        # figure out threshold(s) and set shift
        if isinstance(thresh, list) or isinstance(thresh, numpy.ndarray):
            multi_thresh = True
            if len(thresh) > 1:
                curr_shift = 0
            else:
                curr_shift = shift
        else:
            thresh = [thresh]
            multi_thresh = False
            curr_shift = shift

        # threshold and segment
        all_levels = []
        image_full_inset = image.inset
        for tr in thresh:

            # segment
            seg, con = Connected.make(image=image, thresh=tr, **self.connParam)
        
            # figure out level
            try:
                curr_thresh = numpy.array(self.threshold)
                curr_thresh = numpy.append(curr_thresh, tr)
                tr_sort = curr_thresh.argsort()
                level = tr_sort.argmax()
            except AttributeError:
                level = None

            # deal with properties
            props.update({'contacts':con})
            if tr is not None:
                props.update({'threshold':tr}), 

            # add to local hierarchy
            curr_level = self.addLevel(segment=seg, level=level,
                                 check=check, shift=curr_shift, props=props)
            logging.info("  Added threshold: %f segmentation at level %d" \
                         % (tr, curr_level))
            all_levels.append(curr_level)

            # update shift
            if shift is None:
                curr_shift = self.maxId
            else:
                curr_shift += shift

        # recover image full inset (Connected.make screws it up) 
        # fixed in r241
        #image.useInset(inset=image_full_inset, mode='abs', useFull=True)

        # set positioning
        self.copyPositioning(seg, saveFull=True)

        # return
        if multi_thresh:
            return all_levels
        else:
            return all_levels[0]

    def makeLevelsGen(self, image, thresh=None, order='>', props={},
                      count=False, check=False, shift=None):
        """
        Performs threshold and connectivity segmentation at specified thresholds
        and puts the resulting segments into this hierarchy. Each level of
        the hierarchy contains all segments obtained at one threshold.

        This is a generator, so it should be run within a loop. This method
        makes a segmented image at each threshold, adds it in the current
        hierarchy and yields the segment. Consequently, segmented images at
        each level can be analyzed, which is faster than making a full
        hierarchy first, and then extracting and analyzing individual levels.

        Segementation at each threshold proceeds as follows. Image is
        thresholded, and segmented based on its contacts to regions
        defined in (arg) boundaries. The segmentation consists of finding all
        disjoint segments of thresholded image, analysing contacts between
        the segments and the boundaries and removing segments that do not
        satisfy the contact conditions. Segmentation related parameters need
        to be specified by calling self.setConnParam() before calling this
        method. Based on Connected.make method.
        
        If label is False, (arg) image is expected to be a segmented image, so
        only the connectivity analysis is done.

        Adds threshold and contacts to properties, so they can be accessed from
        self.threshold (list of all thresholds in the order of increasing 
        levels) and self.contacts (Contacts object containing info for contacts
        at all levels.) Other properties can be specified using argument
        properties. 

        At each level (threshold) segment ids are increased by shift*(level-1)
        if shift is not None, or by self.maxId otherwise.  

        Arguments:
          - image: (grayscale) image to be segmented (ndarray or Image object)
          - thresh: list of thresholds
          - order: threshold order, 'ascend', 'descend' or None to keep the
          order given in arg thresh
          - props: dictionary of additional properties
          - count: flag indicating if contacts are counted
          - check: if True checks if levels fit correctly in the existing
          hierarchy
          - shift: segment ids shift

        Sets:
          - self.data: data, has the same positioning as boundaries
          - id-related data structures
          - self.trheshold: list of thresholds indexed by level
          - self.contacts: Contacts object containing contact info for all
          segments
          - self.threshold: array of thresholds indexed by level
          - self.thresh: array of thresholds indexed by ids
          - self.contacts: (Contacts) contacts for all segments

        Yields at each threshold: (segment, level, threshold)
          - segment: (Segment) 
          - level: current level (valid for the final segmentation only if
          arg thresholds are in ascending order)
          - threshold

        where segment has the following attributes (in addition to those se by
        Connected.make):
          - segment.contacts: (Contacts) contacts for segment level
          - segment.threshold: (ndarray, indexed by ids) thresholds
        """

        # figure out threshold(s) and set shift
        if isinstance(thresh, list) or isinstance(thresh, numpy.ndarray):
            if len(thresh) > 1:
                curr_shift = 0
            else:
                curr_shift = shift
        else:
            thresh = [thresh]
            curr_shift = shift

        # order thresholds 
        thresh = list(thresh)
        if order is None:
            pass
        elif (order == 'ascend') or (order == '<'):
            thresh.sort()
        elif (order == 'descend') or (order == '>'):
            thresh.sort()
            thresh.reverse()
        else:
            raise ValueError("Order ", order, " was not understood. Allowed ",
                             "values are 'ascend' ('<'), 'descend ('>') ",
                             " and None.")

        # loop over thresholds
        for tr in thresh:

            # segment
            seg, con = Connected.make(image=image, thresh=tr, count=count,
                                      **self.connParam)
                    
            # figure out level
            try:
                curr_thresh = numpy.array(self.threshold)
                curr_thresh = numpy.append(curr_thresh, tr)
                tr_sort = curr_thresh.argsort()
                level = tr_sort.argmax()
            except AttributeError:
                level = None

            # deal with properties
            props.update({'contacts':con})
            if tr is not None:
                props.update({'threshold':tr}), 

            # add to local hierarchy
            curr_level = self.addLevel(segment=seg, level=level,
                                 check=check, shift=curr_shift, props=props)
            logging.info("  Added threshold: %f segmentation at level %d" \
                         % (tr, curr_level))
            #all_levels.append(curr_level)

            # set some attributes to the Segment
            con.shiftSegmentIds(shift=curr_shift)
            seg.contacts = con
            seg.threshold = numpy.zeros(seg.maxId+1)
            seg.threshold[seg.ids] = tr
            try:
                seg.indexed.append('threshold')
            except AttributeError:
                seg.indexed = ['threshold']

            # prepare shift for the next iteration
            if shift is None:
                curr_shift = self.maxId
            else:
                curr_shift += shift

            yield seg, curr_level, tr

        # set positioning
        #self.copyPositioning(seg)

    def makeByNNew(self, image, thresh, maxNew, minStep=None,
                   between=0.5, check=False):
        """
        Performs threshold and connectivity segmentation for dynamically
        determined thresholds and puts the resulting segments into this
        hierarchy,

        ToDo: make a generator like makeLevelsGen()

        Arguments:
          - image:
          - thresh: list of thresholds
          - minStep: smallest allowed threshold (if None no limit)
          - maxNew: maximum number of new segments
          - between: 
          - check:
        """

        # make level 0
        self.makeLevels(image, thresh=thresh[0], check=check, shift=None)  

        #
        for ind in range(1, len(thresh)):

            # current level
            curr_thresh = thresh[ind]
            curr_level = self.makeLevels(image, thresh=curr_thresh,
                                         check=check, shift=None)  

            # condition
            curr_new = self.getNNew(level=curr_level)
            if curr_new > maxNew:
                thresh_lim = [thresh[ind-1], curr_thresh]
                self.middleByNNew(image=image, prevThresh=thresh_lim, 
                                  between=between, maxNew=maxNew, 
                                  minStep=minStep, check=check, shift=None) 

    def middleByNNew(self, image, maxNew, prevThresh, minStep=None,
                     between=0.5, check=False, shift=None):
        """
        """

        # make a new level between the previous levels
        mid_thresh = (prevThresh[1] - prevThresh[0]) * between + prevThresh[0]
        level = self.makeLevels(image=image, thresh=mid_thresh, check=check,
                                shift=shift)

        # get number of new segement for the current and the higher level
        mid_new = self.getNNew(level=level)
        high_new = self.getNNew(level=level+1)
        logging.debug(("Thresholds (low, mid, high): %f, %f, %f, " \
                      + "n_new (mid, high): %d, %d") \
                      % (prevThresh[0], mid_thresh, prevThresh[1],
                         mid_new, high_new))

        # do middles if needed
        if (mid_new > maxNew) and (mid_thresh - prevThresh[0] > minStep):
            self.middleByNNew(image=image, 
                              prevThresh=[prevThresh[0], mid_thresh],
                              maxNew=maxNew, minStep=minStep, between=between,
                              check=check, shift=shift)
        if (high_new > maxNew) and (prevThresh[1] - mid_thresh > minStep):
            self.middleByNNew(image=image, 
                              prevThresh=[mid_thresh, prevThresh[1]],
                              maxNew=maxNew, minStep=minStep, between=between,
                              check=check, shift=shift)

    def readLevel(self, file, thresh, boundary, boundaryIds=None, check=False,
                  shift=None, byteOrder=None, dataType=None, arrayOrder=None,
                  shape=None):
        """
        in progress
        """

        # read a segmented file
        seg = Segment.read(file=file, byteOrder=byteOrder, dataType=dataType,
                           arrayOrder=arrayOrder, shape=shape)

        # find contacts and add to level
        props = {'threshold':thresh}
        self.addLevel(image=seg, label=False, thresh=None, props=props,
                      boundary=boundary, boundaryIds=boundaryIds,
                      check=check, shift=shift)


    ##############################################################
    # 
    # Other methods
    #
    ###############################################################

    def toSegment(self, copy=False):
        """
        """

        # make segment
        seg = Hierarchy.toSegment(self, copy=copy)

        # add thresholds
        seg.thresh = numpy.zeros(seg.maxId+1)
        seg.thresh[seg.ids] = self.thresh[seg.ids]

        return seg
