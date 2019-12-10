"""
Contains class SegmentationAnalysis

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: segmentation_analysis.py 1485 2018-10-04 14:35:01Z vladan $
"""

__version__ = "$Revision: 1485 $"


import logging
from copy import copy, deepcopy
import numpy
import scipy

from pyto.segmentation.morphology import Morphology
from pyto.segmentation.topology import Topology
from pyto.segmentation.density import Density
from pyto.segmentation.distance_to import DistanceTo
from pyto.segmentation.bound_distance import BoundDistance
from pyto.segmentation.segment import Segment
from ..segmentation.hierarchy import Hierarchy
from ..segmentation.thresh_conn import ThreshConn

 
class SegmentationAnalysis(object):
    """
    Segmentation and morphological, topological and grey-scale (density) 
    analysis of segments.

    Segmentation and analysis example:

      se = SegmentationAnalysis(...)
      se.setSegmentationParameters(...)
      se.setAnalysisParameters(...)
      se.tcSegmentAnalyze(...)

    Important attributes set by the above procedure:
      - labels: hieararchy or segments, whichever one is set (see below)
      - hierarchy: set if hierarchical segmentation
      - segments: set if sigle threshold segmentation
       - density: (Density) density stats for all segments
      - regionDensity: (Density, non-array form) density stats for
      the segmentation region
      - morphology: (Morphology) morphology for all segments, includes
      length at self.morphology.length
      - topology: (Topology) topology for all segments
      - distance: (DistanceTo) distance to a specified region. The name of this
      attribute can be different if it's specified as argument distanceName of
      setAnalysisParameter(). Also, there can be more than one distance if the
      distanceName argument is a list of names.
      - boundDistance: (boundDistance) distances between the boundries 
      contacting segments (for segments with exactly two boundaries)
      - self.nContacts, self.surfaceDensityContacts, 
      self.surfaceDensitySegments: (ndarrays of length 3), the array 
      elements correspond to both boundaries together, boundary 1 and 
      boundary 2. Meant for cleft-like segmentation regions.

    Classification and analysis of an existing segmentation:

      se = SegmentationAnalysis()
      se.addClassificationParameters(...) # one or more times
      se.setAnalysisParameters(...)
      for new, name in se.classifyAnalyze(...):
          new ...

    """

    ###############################################################
    #
    # Initialization
    #
    ##############################################################

    def __init__(self, image=None, boundary=None, segments=None, 
                 hierarchy=None):
        """
        Initializes attributes.

        Arguments:
          - image: () grey-scale image
          - boundary: (Segment) boundaries
          - segments: (Segment) segmented image (non-hierarchical) 
          - hierarchy: (Hierarchy) hierarchical segmentation
        """

        # set attributes
        self.image = image
        self.boundary = boundary
        self.segments = segments
        self.hierarchy = hierarchy

        # internal attributes
        self.classifications = None
        self.resultNames = set([])


    ###############################################################
    #
    # Variables
    #
    ##############################################################

    def getLabels(self):
        """
        Returns labels, that is self.hierarchy if it exists, or 
        self.segments. If neither of the two exists returns None.
        """

        labels = None
        try:
            if self.hierarchy is not None:
                labels = self.hierarchy
        except AttributeError: pass
        try:
            if (labels is None) and (self.segments is not None):
                labels = self.segments
        except AttributeError:  pass

        return labels

    labels = property(fget=getLabels, doc="Labels object")

    def getIds(self):
        """
        Returns segment ids.
        """
        
        if self.labels is not None:
            ids = self.labels.ids
        else:
            ids = None

        return ids

    ids = property(fget=getIds, doc="Segment ids")

    def getNSegments(self):
        """
        Number of segments
        """
        return len(self.ids)

    nSegments = property(fget=getNSegments, doc="Number of segments")


    ###############################################################
    #
    # Main segmentation, classification and analysis methods
    #
    ##############################################################

    def tcSegmentAnalyze(self, thresh, order='>', image=None, boundary=None, 
                  count=False, doDensity=False, doMorphology=False, 
                  doLength=False, doTopology=False,  doDistanceTo=False, 
                  doBoundDistance=False, doCleftContacts=False):
        """
        Segment using thresholding and connectivity and analyze segments.

        Segmentation and analysis parameters need to be set in advance (see
        setSegmentationParameters() and setAnalysisParameters()).

        Makes threshold and connectivity segmentations at each level,  
        analyzes the segments and yields the segmentation. After the 
        segmentation is done at all required levels, yields a (complete) 
        hierarchical segmentation.

        The (complete) analysis results are saved as attributes of this object
        that may include (these attributes are listed in self.resultNames):
          - self.density: (Density) density stats for all segments
          - self.regionDensity: (Density, non-array form) density stats for
          the segmentation region
          - self.morphology: (Morphology) morphology for all segments, includes
          length at self.morphology.length
          - self.topology: (Topology) topology for all segments
          - self.distance: (DistanceTo) distance to a specified region. The 
          name of this attribute can be different if it's specified as
          argument distanceName of setAnalysisParameter(). Also, there can be
          more than one distance if the distanceName argument is a list of
          names.
          - self.boundDistance: (boundDistance) distances between the boundries 
          contacting segments (for segments with exactly two boundaries)
          - self.nContacts, self.surfaceDensityContacts, 
          self.surfaceDensitySegments: (ndarrays of length 3), the array 
          elements correspond to both boundaries together, boundary 1 and 
          boundary 2. Meant for cleft-like segmentation regions.
 
        Note: if doLength is True, doMorphology is effectivly set to True also.

        Yields a new instance of this class at each threshold, that contains 
        the analysis results for the current level only and backgroundDensity
        attribut. In addition, the yielded instance also has several 
        segmentation parameters (lengthContact, lengthLine, nBoundary, 
        boundCount, structElConn, contactStructElConn, countStructElConn, 
        distanceId and distanceMode).

        After the segmentation is completed, yields the hierarchical 
        segmentation. This instance has all analysis results as attributes 
        whose names are listed in self.resultNames.

        Analysis results for an individual level can also be read at the time
        the segmentation for that level is yielded, from attributes 
        that have the same names (as the complete analysis result attributes)
        but with a preceeding underscore (self._density, self._morphology, ...).
        Furthermore the basic density stats for background is in 
        self._backgroundDensity (Density, non-array form).

        Arguments:
          - thresh: all thresholds (saved as self.thresholds)
          - order: threshold order: '>' or 'descend' for descending, '<' or 
          'ascend' for ascending, None to keep the order of thresh 
          - image: (Image) image
          - boundary: (Segment) boundaries
          - do*: flags indicating if particular analysis should be done

        Yields at each level: new_inst, level, threshold
          - new_instance: new instance of this class containing:
            - segments: the current (flat) segmentation (by reference, that 
            is not copied)
            - all analysis results at the current level (deepcopied) 
          - level: current level (valid for the final segmentation only if
          arg thresholds are in ascending order)
          - threshold: current threshold

        Yields, after segmentation is finished for all levels: 
          - tc: (ThreshConn) containing:
            - hierarchy: hierarchical segmentation
            - analysis of all segments

        ToDo: include ThreshConn.makeByNew() (or makeByNewGen()).
        """

        # parse arguments
        if image is None:
            image = self.image
        if boundary is None:
            boundary = self.boundary
        doMorphology = doMorphology or doLength
        self.thresholds = thresh

        # initialize objects to hold final results
        self.initializeAnalysis(doDensity=doDensity, doMorphology=doMorphology,
                                doLength=doLength, doTopology=doTopology, 
                                doDistanceTo=doDistanceTo, 
                                doBoundDistance=doBoundDistance,
                                doCleftContacts=doCleftContacts)

        # prepare for segmentation
        tc = ThreshConn()
        tc.setConnParam(boundary=boundary, boundaryIds=self.boundaryIds, 
                        nBoundary=self.nBoundary, boundCount=self.boundCount, 
                        mask=self.segmentRegion, freeSize=self.freeSize, 
                        freeMode=self.freeMode, 
                        structElConn=self.structElConn,
                        contactStructElConn=self.contactStructElConn,
                        countStructElConn=self.countStructElConn)

        # loop over thresholds
        for segments, level, curr_thresh \
                in tc.makeLevelsGen(image=image, thresh=thresh, order=order,
                                    count=count):

            # analyze current segments 
            self.analyzeSegments(
                image=image, segments=segments, doDensity=doDensity, 
                doMorphology=doMorphology, doLength=doLength, 
                doTopology=doTopology, doDistanceTo=doDistanceTo, 
                doBoundDistance=doBoundDistance,
                doCleftContacts=doCleftContacts)

            # make new instance with current results
            new_inst = self.copyResults(segments=segments)
            new_inst.copyAnalysisParameters(self)

            # merge analysis results with results for other thresholds
            self.mergeAnalysis(
                doDensity=doDensity, doMorphology=doMorphology,
                doLength=doLength, doTopology=doTopology, 
                doDistanceTo=doDistanceTo, doBoundDistance=doBoundDistance,
                doCleftContacts=doCleftContacts)

            # yield current 
            threshold = tc.threshold[level]
            finalize = (yield new_inst, level, curr_thresh)

            # if segmentation done yield hierarchy
            if (finalize is not None) and finalize:
                yield tc

    def classifyAnalyze(
        self, hierarchy, image=None, doDensity=False, doMorphology=False, 
        doLength=False, doTopology=False,  doDistanceTo=False, 
        doBoundDistance=False, doCleftContacts=False):
        """
        Classifies and analyzes a hierarchical segmentation specified by arg
        hierarchy. For each class generated, yields an instance of this class
        that contains a subset of the original hierarchical segmentation 
        (a class of segments) and analysis results for that class.

        One or more classifications are defined by arg self.classifications 
        (set by addClassificationParameters()). Preforms classifications in 
        the order they appear in self.classifications. The classifications are
        done recursively.

        Each class generated according to one classification is then classified 
        according to the next one. In order to minimize memory usage classes
        are generated on demand. This means that after the first class of the 
        first classification is generated, this class is then classified 
        according to the second classification (before the second class of the
        first classification is generated) and so on. Consequently, the 
        classes generated by the last classification 'change' the fastest.

        For example, if classification 1 generates classes a and b, and 
        classification 2 p, q and r, the classes are generated in the order:
        ap, aq, ar, bp, bq, br.

        May modify arg hierarchy. See methods preforming individual
        classifications (keepSegments, classByVolume, classByContactedIds,
        classByNContacted) for details.

        Depending on the arguments, may perform analysis of (grey-scale) 
        density, morphology, topology, distance to region and boundary 
        distance. See the attributes of the yielded object and the docs for the 
        analysis methods.

        If the current instance already contains analysis results and the 
        analysis flags (args do*) are False, the results are retained in the 
        classes but restricted to the segments (ids) that exist in that class.

        Note that even if the current instance already contains nContacted 
        and surfaceDensity*, they are recalculated. In this case arg 
        doCleftContacts doesn't matter, because it is internaly set to True.

        Alternatively, if the analysis flags are True, the analysis is 
        preformed only for segments existing in the classes, and the analysis
        results of the current instance are not used.

        Note: if doLength is True, doMorphology is effectivly set to True also.

        Arguments:
          - hierarchy: (Hierarchy) hierarchy of segments
          - image: (Image) image
          - do*: flags indicating if particular analysis should be done

        Yields: class, name
          - class: (SegmentationAnalysis) contans segmentation and analysis 
          attributes:
            - hierarchy: (Hierarchy) contains a class (subset) of original
            segmentation
            - density: (Density) density stats for all segments
            - regionDensity: (Density, non-array form) density stats for the 
            segmentation region
            - morphology: (Morphology) morphology for all segments
            - topology: (Topology) topology for all segments
            - distance: (DistanceTo) distance to a specified region. The 
            name of this attribute can be different if it's specified as
            argument distanceName of setAnalysisParameter(). Also, there can be
            more than one distance if the distanceName argument is a list of
            names.
            - boundDistance: (boundDistance) distances between the boundries 
            contacting segments (for segments with exactly two boundaries)
            - nContacts, surfaceDensityContacts, surfaceDensitySegments: 
            (ndarrays of length 3), the array elements correspond to both
            boundaries together, boundary 1 and boundary 2. Meant for
            cleft-like segmentation regions.
          - name: class name
        """

        # if doLength then doMorphology also
        doMorphology = doMorphology or doLength

        # classify
        for hi, name in self.classify(hierarchy=hierarchy):

            # make sure that cleft contacts are calculated again if they
            # were already calculated
            if ('nContacts' in self.resultNames) \
                    or ('surfaceDensityContacts' in self.resultNames) \
                    or ('surfaceDensitySegments' in self.resultNames):
                doCleftContacts = True
            
            # initialize objects to hold final results
            new = self.restrict(
                hierarchy=hi, avoid=['regionDensity','nContacts', 
                           'surfaceDensityContacts', 'surfaceDensitySegments'])
            new.initializeAnalysis(
                doDensity=doDensity, doMorphology=doMorphology,
                doLength=doLength, doTopology=doTopology, 
                doDistanceTo=doDistanceTo, doBoundDistance=doBoundDistance,
                doCleftContacts=doCleftContacts)

            # analyze if needed
            if any([doDensity, doMorphology, doLength, doTopology,  
                    doDistanceTo, doBoundDistance, doCleftContacts]):

                # analyze at each level and put together
                local_hi = deepcopy(hi)
                for segments, level in local_hi.extractLevelsGen(order='>'):
                    
                    # analyze level
                    new.analyzeSegments(
                        segments=segments, image=image, doDensity=doDensity, 
                        doMorphology=doMorphology, doLength=doLength, 
                        doTopology=doTopology, doDistanceTo=doDistanceTo, 
                        doBoundDistance=doBoundDistance,
                        doCleftContacts=doCleftContacts)

                    # merge analysis results with results for other thresholds
                    new.mergeAnalysis(
                        doDensity=doDensity, doMorphology=doMorphology,
                        doLength=doLength, doTopology=doTopology, 
                        doDistanceTo=doDistanceTo, 
                        doBoundDistance=doBoundDistance,
                        doCleftContacts=doCleftContacts)

                # analyze at the 

            # yield new class 
            yield new, name

    def setSegmentationParameters(self, nBoundary=None, boundCount='exact', 
                      boundaryIds=None, mask=0, freeSize=0, freeMode='add', 
                      structElConn=1, contactStructElConn=1, 
                      countStructElConn=1):
        """
        Sets segmentation parameters.

        Arguments are saved as attributes of this class under the same names,
        except that mask is renamed to segmentRegion (segmentation region).

        Arguments are sent directly to segmentation.ThreshConn.setConnParam()
        and from there to segmentation.Connected.make(), see those for details.

        Arguments:
          - nBoundary: number of boundaries contacted
          - boundCount: 'exact' for contacting exactly nBoundaries, or
          'at_least' for contacting >= nBoundaries 
          - boundaryIds: ids of boundries in the boundary object
          - mask: (int, list, ndarray or Label) if mask is an int or a list of
          ints, then the segments of self.boundaries having this (those) id(s)
          form the segmentation region. Otherwise, positive elements of mask
          array define the segmentation region
          - free size: restrict segmentation region to the distance given by
          this arg from boundaries (0 for no restriction)
          - free mode: 'add' to add regions around boundaries specified above,
          'intersect' to intersect them 
          - *structEl*: structuring elements, should better not be changed
        """
        
        # set all arguments as attributes (should be a better way)
        self.boundaryIds = boundaryIds
        self.nBoundary = nBoundary
        self.boundCount = boundCount
        self.segmentRegion = mask
        self.freeSize = freeSize
        self.freeMode = freeMode
        self.structElConn = structElConn
        self.contactStructElConn = contactStructElConn
        self.countStructElConn = countStructElConn

        # make connection parameters dictionary (not needed?)
        names = ['structElConn', 'contactStructElConn', 'countStructElConn',
                 'nBoundary', 'boundCount', 'mask', 'freeSize', 'freeMode']
        self.connParam = dict((name, getattr(self, name)) for name in names 
                              if name != 'mask')
        self.connParam['mask'] = self.segmentRegion

    def setAnalysisParameters(self, segmentRegion=None, lengthContact='c2c', 
               lengthLine='mid', distanceRegion=None, distanceId=None,
               distanceMode='mean', distanceName='distance', cleftLayers=None):
        """
        Sets analysis parameters.

        Arguments are saved as attributes of this class under the same names.

        Arguments distanceName, distanceId and distanceMode have to be
        either single values, or they have to be lists (or tuples) with the
        same number of elements.

        See segmentation.Morphology.getLength() and 
        segmentationDistanceTo.calculate() for more details

        Arguments:
          - segmentRegions: segmentation region, if None the value set by
          self.setSegmentationParameters() is used
          - lengthContact: mode of calculating a segment length, 'b2b' (or 
          'boundary') for distance between boundaries, or 'c2c' (or 'contact')
          for distances between contact points on segments for the two boundary
          case, or 'b-max' and 'c-max' for the one boundary case (same as 
          arg distance of segmentation.Morphology.getLength())
          - distanceRegion: (Segment) segmented image containig regions to 
          which distances are calculated
          - distanceId: id(s) of distance regions
          - distanceMode: one or more of 'center', 'min', 'max', 'mean' and
          'median'
          - distanceName: attribute name(s) where the distance results are
          saved
          - cleftLayers: (CleftRegions) cleft layers, used to provide boundary
          edge surfaces
        """

        if segmentRegion is not None:
            self.segmentRegion = segmentRegion
        self.lengthContact = lengthContact
        self.lengthLine=lengthLine
        if distanceRegion is not None:
            self.distanceRegion = distanceRegion
        else:
            self.distanceRegion = self.boundary
        self.distanceId = distanceId
        self.distanceMode = distanceMode
        self.distanceName = distanceName
        self.cleftLayers = cleftLayers


    ###############################################################
    #
    # Analysis methods
    #
    ##############################################################

    def analyzeSegments(self, segments, image=None, boundary=None, 
                        doDensity=False, doMorphology=False, doLength=False, 
                        doTopology=False, doDistanceTo=False, 
                        doBoundDistance=False, doCleftContacts=False):
        """
        Analyze segments specified by arg (Segment) segments. 

        The results are saved as attributes starting with '_' (see below). 
        Previous results (attributes without leading '_') are left unchanged. 
        The new analysis results can be merged with the previous ones using
        mergeAnalysus().

        Note: if doLength is True, doMorphology is effectivly set to True also.

        Arguments:
          - segments: (Segments) segmented image
          - image: image
          - boundaries: (Segments) boundaries
          - do*: flags indicating if the corresponding analysis is done

        Sets:
          - self._density
          - self._backgroundDensity
          - self._regionDensity
          - self._morphology
          - self._topology
          - self._distanceTo: can have other name depending on distanceName 
          attribute of setAnalysisParameters()
          - self._boundDistance
          - self._nContacts, self._surfaceDensityContacts
          - self._surfaceDensitySegments
        """

        # parse arguments
        if image is None:
            image = self.image
        if segments is None:
            segments = self.segments
        if boundary is None:
            boundary = self.boundary

        # if doLength then doMorphology also
        doMorphology = doMorphology or doLength

        if doDensity:

            # segment density
            self._density = self.getDensity(image=image, segments=segments)

            # background and region (total) density
            self._backgroundDensity, self._regionDensity = \
                self.getOtherDensity(segments=segments, boundary=boundary,
                              segmentRegion=self.segmentRegion, image=image)

        # segment morphology
        if doMorphology:
            self._morphology = self.getMorphology(segments=segments, 
                                                    doLength=doLength)

        # topology
        if doTopology:
            self._topology = self.getTopology(segments=segments)

        # distance to region (possibly multiple)
        if doDistanceTo:
            if isinstance(self.distanceName, tuple):

                # multiple
                for _name, distance_id, mode \
                        in zip(self._distanceName, self.distanceId, 
                               self.distanceMode):
                    dist = self.getDistanceToRegion(
                        segments=segments, regions=self.distanceRegion, 
                        mode=mode, distanceId=distance_id)
                    setattr(self, _name, dist)

            else:

                # single
                dist = self.getDistanceToRegion(
                    segments=segments, regions=self.distanceRegion,
                    mode=self.distanceMode, distanceId=self.distanceId)
                setattr(self, self._distanceName, dist)
                
        # distance between boundaries
        if doBoundDistance:
            self._boundDistance = self.getBoundDistance(segments=segments, 
                                                        boundary=boundary)

        # count contacts for cleft
        if doCleftContacts:
            counted = self.countCleftContacts(
                layers=self.cleftLayers, labels=segments)
            (self._nContacts, self._surfaceDensityContacts, 
             self._surfaceDensitySegments) = counted
        
    def getDensity(self, segments, image=None):
        """
        Returns basic density (grey-value) statistics (mean, std, min, max and
        volume for each segment.  

        Arguments:
          - segments: (Segments) segmented image
          - image: image

        Returns: (Density) density
        """

        # parse arguments
        if image is None:
            image = self.image
        if segments is None:
            segments = self.segments

        # return None if no ids
        if (segments.ids is None) or (len(segments.ids) == 0):
            return None

        # density
        dens = Density()
        dens.calculate(image=image, segments=segments)

        return dens
        
    def getOtherDensity(self, segments, segmentRegion, image=None, 
                        boundary=None):
        """
        Calculates basic stats for background (non-segmented parts) and for
        the whole segmentation region.

        Arguments: 
          - segments: (Segment) segmented image
          - segmentRegion: (single int or list) id(s) of segmentation region(s)
          - image: image
          - boundaries: boundaries

        Returns:
          - background_density: (Density, non-array form) background density
          - segmentation_region_density: (Density, non-array form) segmentation
          region density
        """

        # parse arguments
        if image is None:
            image = self.image
        if boundary is None:
            boundary = self.boundary
        if isinstance(segmentRegion, list):
            seg_reg_ids = segmentRegion
        else:
            seg_reg_ids = [segmentRegion]

        # bring segments and boundary to the same positioning 
        seg_data, seg_inset = segments.getDataInset()
        segments.makeInset(additional=boundary, additionalIds=seg_reg_ids,
                           expand=True)
        bound_data, bound_inset = boundary.getDataInset()
        boundary.makeInset(ids=seg_reg_ids, additional=segments, 
                           expand=True)

        # combine all segmentation regions and make an object
        order = dict(zip(seg_reg_ids, [1] * len(seg_reg_ids)))
        seg_reg_data = boundary.reorder(data=boundary.data, order=order,
                                        clean=True)
        seg_reg = Segment(data=seg_reg_data)
        seg_reg.inset = boundary.inset
                                            
        # density of the segmentation region
        seg_reg_dens = Density()
        seg_reg_dens.calculate(image=image, segments=seg_reg)
        seg_reg_dens = seg_reg_dens.extractOne(id_=1, array_=False)

        # make background object and calculate density
        bkg_data = seg_reg_data & (segments.data <= 0)
        bkg_data = numpy.asarray(bkg_data, dtype='int')
        bkg = Segment(data=bkg_data)
        bkg.inset = segments.inset
        bkg_dens = Density()
        bkg_dens.calculate(image=image, segments=bkg)
        bkg_dens = bkg_dens.extractOne(id_=1, array_=False)

        # revert to the old positioning of segments and boundary
        segments.setDataInset(data=seg_data, inset=seg_inset)
        boundary.setDataInset(data=bound_data, inset=bound_inset)

        return bkg_dens, seg_reg_dens

    def getMorphology(self, segments, contacts=None, boundary=None, 
                      doLength=True, lengthContact=None, lengthLine=None):
        """
        Calculates basic morphological properties: volume, surface and length 
        (if arg doLength is True).

        For length calculation lengthContacts and lengthLine have to be
        either given as arguments of this method, or already set as attributes.

        Arguments:
          - segments: (Segment) segmented image
          - contacts: (Contacts) contacts, if None segments.contacts is used
          - boundary: boundary
          - doLength: flag indicating if length is calculated
          - lengthContact: mode of calculating length (same as distance argument
          of Morphology.getLength)
          - lengthLine: mode of calculating length (same as line argument
          of Morphology.getLength)

        Returns (Morphology) morphology
        """

        # parse arguments
        if contacts is None:
            contacts = segments.contacts
        if boundary is None:
            boundary = self.boundary
        if lengthContact is None:
            lengthContact = self.lengthContact
        if lengthLine is None:
            lengthLine = self.lengthLine

        # return None if no ids
        if (segments.ids is None) or (len(segments.ids) == 0):
            return None

        # calculate
        mor = Morphology(segments=segments)
        mor.getVolume()
        mor.getSurface()
        if doLength:
            mor.getLength(segments=segments, boundaries=boundary, 
                          contacts=contacts, distance=lengthContact, 
                          line=lengthLine)
        
        return mor

    def getTopology(self, segments):
        """
        Calculates segment topology.

        Argument:
          - segments: (Segment) segmented image

        Returns (Topology) topology
        """

        # return None if no ids
        if (segments.ids is None) or (len(segments.ids) == 0):
            return None

        # calculate
        topo = Topology(segments=segments)
        topo.calculate()
        return topo

    def getDistanceToRegion(self, segments, regions, distanceId, mode):
        """
        Calculates distance between each segment and its closest distance
        region.

        Distance regions are specified in self.boundary

        Arguments:
          - segments: (Segment) segmented image
          - regions: (Segments) regions
          - distanceId: (int or list of ints) ids of regions used for distance
          calculations
          - mode: distance calculation mode, 'center', 'min', 'max', 'mean' 
          or 'median' as in distanceTo.calculate()

        Returns distance: (ndarray) distance of each segment to its closest 
        distance region
        """
        
        # return None if no ids
        if (segments.ids is None) or (len(segments.ids) == 0):
            return None

        # calculate
        dist = DistanceTo(segments=segments)
        dist.calculate(regions=regions, regionIds=distanceId, mode=mode)
        return dist

    def getBoundDistance(self, segments, contacts=None, boundary=None, 
                         mode='min'):
        """
        Calculates distance between boundaries contacting segments. Calculated
        only for segments that contact exactly two boundaries.

        Arguments:
          - segments: (Segment) segmented image
          - contacts: (Contacts) contacts, if None segments.contacts is used
          - boundary: (Segment) boundary
          - mode: distance mode, can be 'min', 'max', 'mean' or 'median'
          or 'center'
         """

        # parse arguments
        if contacts is None:
            contacts = segments.contacts
        if boundary is None:
            boundary = self.boundary

        # return None if no ids
        if (segments.ids is None) or (len(segments.ids) == 0):
            return None

        # calculate
        bound_dist = BoundDistance()
        bound_dist.calculate(contacts=contacts, boundaries=boundary,
                             ids=segments.ids, mode=mode, extend=True)

        return bound_dist

    def getNContacted(self, contacts, boundary):
        """
        Calculates number of contacted segments for each boundary
        """

        # calculated number of contacts
        n_contacted = numpy.zeros(boundary.maxId+1, dtype=int)
        for b_id in boundary.ids:
            nc = contacts.getN(boundaryId=b_id)
            n_contacted[b_id, 1:len(nc)+1] = nc
            n_contacted[0] = nc.sum()

    def countCleftContacts(self, layers, labels=None):
        """
        Calculates and returns number of contacts, surface density of contacts
        and surface density of segments. Meant for cleft-like
        segmentation regions only.

        The two surface densitites are calculated in respect to surface
        of the edge layer of the two boundaries. An edge layer is the part
        of a boundary that directly contacts cleft region. Consequently,
        boundaries as defined in segmentation.Cleft may not be the best
        choice, because they are often larger than the cleft. Instead,
        a CleftLayers objects is used here.

        Arguments:
          - layers: (CleftLayers) cleft layers defining a cleft
          - labels: (segmentation.Labels) used to provide ids and contacts
          (Contacts) as attributes. If not specified self.labels is used
          instead.

        Returns: (n_contacts, surface_density_contacts, 
        surface_density_segments):
          - n_contacts: (ndarray: axis 0 total, boundary 1, boundary 2, 
          axis 1 indexed by segment id) number of contacts
          - surface_density_contacts: (1-d ndarray, length 3) density of
          contacts on both boundaries / boundary 1 / boundary 2
          - surface_density_segments: (1-d ndarray, length 3) density of
          segments on average boundary area / boundary 1 / boundar 2
        """

        # set contacts and n_segments
        if labels is None:
            labels = self.labels
            n_segments = self.nLabels
        else:
            n_segments = len(labels.ids)
        contacts = labels.contacts

        # get boundary (edge) area
        edges = layers.boundEdgeLayerIds
        sur_1 = layers.regionDensity.volume[edges[0]]
        sur_2 = layers.regionDensity.volume[edges[1]]

        # segment surface density
        surf_dens_seg = numpy.zeros(3)
        surf_dens_seg[1] = n_segments / float(sur_1) 
        surf_dens_seg[2] = n_segments / float(sur_2) 
        surf_dens_seg[0] = 2 * n_segments / float(sur_1 + sur_2)

        # n contacts
        if (labels.ids is not None) and (len(labels.ids) > 0):
            c_1 = numpy.zeros(shape=(labels.maxId+1,), dtype=int)
            c_1[labels.ids] = \
                contacts.getN(boundaryId=self.boundaryIds[0])[labels.ids]
            c_1[0] = c_1[1:].sum()
            c_2 = numpy.zeros(shape=(labels.maxId+1,), dtype=int)
            c_2[labels.ids] = \
                contacts.getN(boundaryId=self.boundaryIds[1])[labels.ids]
            c_2[0] = c_2[1:].sum()
        else:
            c_1 = 0
            c_2 = 0
        c_0 = c_1 + c_2
        n_cont = numpy.vstack([c_0, c_1, c_2])

        # contact surface density
        surf_dens_cont = numpy.zeros(3)
        surf_dens_cont[1] = n_cont[1,0] / float(sur_1) 
        surf_dens_cont[2] = n_cont[2,0] / float(sur_2) 
        surf_dens_cont[0] = n_cont[0,0] / float(sur_1 + sur_2)
        
        return n_cont, surf_dens_cont, surf_dens_seg
    

    ###############################################################
    #
    # Classify methods
    #
    ##############################################################

    def classify(self, hierarchy, classifications=None, name=''):
        """
        Generator for classification of segments from a hierarchical
        segmentation. 

        Reads classification parameters for classifications defined in arg or
        self.classifications (set by addClassificationParameters()). Preforms
        classifications in the order they appear in (self.)classifications.
        The classifications are done recursively.

        Classifications are given by a list where each element is a dict that 
        defines one classification. Each dictionary needs to have in item with 
        name 'type', which determines the method used for classification. 
        Currently, the names: 'keep', 'volume', 'connected_ids' or 
        'n_contacted' specify classification methods keepSegments(),
        classByVolume(), classByContactedIds() and classByNContacted(), 
        respectively. All other items of a classification dictionary are passed
        as (dict) arguments to the corresponding classification method. See
        the docs for the individual classification methods for their
        arguments.

        Each class generated according to one classification is then classified 
        according to the next one. In order to minimize memory usage classes
        are generated on demand. Tha means that after the first class of the 
        first classification is generated, this class is then classified 
        according to the second classification (before the second class of the
        first classification is generated) and so on. Consequently, the 
        classes generated by the last classification 'change' the fastest.

        For example, if classification 1 generates classes a and b, and 
        classification 2 p, q and r, the classes are generated in the order:
        ap, aq, ar, bp, bq, br.

        May modify arg hierarchy. See methods preforming individual
        classifications (keepSegments, classByVolume, classByContactedIds,
        classByNContacted) for details.

        Arguments:
          - hierarchy: (Hierarchy) hierarchy of segments
          - classifications: (list of dicts) each element of this list defines 
          one classification. 
          classification
          - name: prefix of the class names generated by this method (the main
          parts of the class names are generated by classification methods)

        Yields:
          - (Hierarchy) a flat hierarchy of segments
          - class name
        """

        # set classifications
        if classifications is None:
            classifications = self.classifications

        # deal with no classification
        if classifications is None:
            yield hierarchy, name

        # figure out method
        one_classif = classifications[0]
        type = one_classif['type']
        if type == 'keep':
            method = self.keepSegments
        elif type == 'volume':
            method = self.classByVolume
        elif type == 'contacted_ids':
            method = self.classByContactedIds
        elif type == 'n_contacted':
            method = self.classByNContacted
        else:
            raise ValueError("Classification of type ", type, " was not ",
                             "understood. Valid classificatiiions are: 'keep'",
                             " 'volume', 'contacted_ids' and 'n_contacted'.")

        # do one classification
        for cur_hi, cur_name in method(hierarchy=hierarchy, 
                                       **one_classif['args']):
            cur_name = name + '_' + cur_name
            if len(classifications) == 1:

                # last classification
                #print 'last classification: ' + cur_name 
                yield cur_hi, cur_name

            else:

                # more classifications left
                for hi_2, name_2 in self.classify(hierarchy=cur_hi, 
                                        classifications=classifications[1:], 
                                        name=cur_name):
                    #print 'not last classification: ' + cur_name 
                    yield hi_2, name_2

    def addClassificationParameters(self, type, args={}):
        """
        Adds (parameters for) a classification. 

        Arg type specifies the classification type, that is determines the 
        classification method. Possible values are:
          - 'keep': keepSegments()
          - 'volume': classByVolume()
          - 'contacted_ids': classByContacted()
          - 'n_contacted': classByNContacted()

        All arguments for the method chosen, except for hierarchy, need to be
        given in args (see individual methods).

        Appends the values specified to self._classifications.

        Arguments:
          - type: classification type
          - args: (dict) classification method parameters, where keys are 
          argument names and values are argument values
        """

        # initialize if needed
        if self.classifications is None:
            self.classifications = []

        # add
        self.classifications.append({'type':type, 'args':args}) 

    def removeClassificationParameters(self):
        """
        Removes all defined classifications (from self.classifications)
        """
        self.classifications = None

    def keepSegments(self, hierarchy, mode, threshold=None, name=None):
        """
        Yields (hierarchy) object obtained by keeping only some segments from
        those in (arg) hierarchy. 

        If mode is 'new' (or 'leaf') only 'new segments' (or 'leafs'), that is
        the segments that have no segments below them are kept. If in addition
        arg threshold is specifed only the leafs that have higher segments at
        the given threshold are kept.

        If mode is 'new_branch_tops' or 'leaf_branch_tops' only the segments 
        that are the branch tops of the leaf segments, that is the highest 
        segment above each of the leafs that did not merge with any other
        segment is kept.

        In both cases a flat hierarchy is yelded and the hierarchy object is
        modified.

        Class name is either specified in arg name, or the default
        value is used (same as mode).

        Arguments:
          - hierarchy: (Hierarchy) hierarchy of segments
          - mode: determines which segments to keep
          - threshold: threshold used only if mode is 'new' (same as 'leaf')
          - name: name of the class generated by this method

        Yields: 
          - (Hierarchy) a flat hierarchy of segments
          - class name
        """

        # find ids to keep
        if (mode == 'new') or (mode == 'leaf'):

            # keep only new segments
            if threshold is None:
                below = None
            else:
                below = hierarchy.getLevelFrom(threshold=threshold)
            good_ids = hierarchy.findNewIds(below=below)

        elif (mode == 'new_branch_tops') or (mode == 'leaf_branch_tops'):

            # keep only new branch tops
            good_ids = hierarchy.findNewBranchTops()

        # keep ids
        hierarchy.keep(ids=good_ids)
        #hierarchy.contacts.keepSegments(ids=good_ids)

        # get name
        if name is None:
            name = mode

        yield hierarchy, name

    def classByVolume(self, hierarchy, volumes, names=None):
        """
        Classifies segments of arg hierarchy according to their volume and
        yields the corresponding classes. 

        Each class is a Hierarchy objects that contains segments having
        volume between volumes[i] (inclusive) and volumes[i+1] (exclusive). 

        Arg hierarchy is modified so that it contain only segments where
        volumes[0] <= segment_volume < volumes[-1].

        Class names are either specified in arg names, or the default
        values are used (bound-id, and bound-rest if arg rest is True).

        Arguments:
          - hierarchy: (Hierarchy) hierarchy of segments
          - volumes: list of volumes
          - names: list of class names, directly corresponding to the volume 
          intervals (len(volumes) - 1 in total)

        Yields: 
          - (Hierarchy) a hierarchy of segments
          - class name
        """

        # remove volumes that are outside of the range
        good_ids = hierarchy.findIdsByVolume(min=volumes[0], max=volumes[-1])
        hierarchy.keep(ids=good_ids)

        # make classes based on volume
        for ind in range(len(volumes)-1):
            curr_ids = hierarchy.findIdsByVolume(min=volumes[ind], 
                                                 max=volumes[ind+1])
            curr_obj = hierarchy.keep(curr_ids, new=True)
            if names is None:
                name = 'vol-' + str(volumes[ind]) + '-' + str(volumes[ind+1])
            else:
                name = names[ind]
            yield curr_obj, name

    def classByContactedIds(self, hierarchy, ids, contacts=None, rest=False,
                            names=None):
        """
        Classifies segments of arg hierarchy according to the ids of boundaries
        they contact and yields the corresponding classes. 

        Each element of arg ids (can contain one or more boundary ids) 
        determines a class. This class contain all segments that contact (at 
        least one of the) id(s). Note that, a segment may apear in more than 
        one class. If arg rest is True, the last class is formed by segments
        that do not appear in any of the previous classes.

        Class names are either specified in arg names, or the default
        values are used (bound-id, and bound-rest if arg rest is True).

        Arguments:
          - hierarchy: (Hierarchy) hierarchy of segments
          - contacts: (Contacts) contacts between segments and boundaries, if
          None hierarchy.contacts is used
          - ids: list where each element is an boundary id or a list (or tuple)
          of boundary ids
          - rest: flag indicating if the class containing segments that do 
          not belong to any other class as put in a (last) class also.
          - names: list of names, directly corresponding to elements of arg 
          contacts. If rest is True, there has to be an additional name
          that correspond to the 'rest' class.

        Yields: 
          - (Hierarchy) a hierarchy of segments
          - class name
        """
        
        # find contacts
        if contacts is None:
            contacts = hierarchy.contacts

        # check names
        if names is None:
            main_names = [None] * len(ids)
        else:
            if (rest is True) and (len(names) != len(ids)+1):
                raise ValueError("Argument names has to have ", 
                                 len(ids)+1, " elements.")
            elif (rest is False) and (len(names) != len(ids)):
                raise ValueError("Argument names has to have ", 
                                 len(ids), " elements.")
            main_names = names[:-1] 

        # classes according to ids
        all_good_seg_ids = []
        for bound_ids, name in zip(ids, main_names):
            if isinstance(bound_ids, (list, tuple)):
                if len(bound_ids) > 1:
                    bound_ids_local = [bound_ids]
                else:
                    bound_ids_local = bound_ids
            else:
                bound_ids_local = bound_ids
            seg_ids = contacts.findSegments(boundaryIds=bound_ids_local, 
                                             mode='at_least')
            all_good_seg_ids += seg_ids.tolist() 
            curr_obj = hierarchy.keep(seg_ids, new=True)
            if name is None:
                name = 'bound-' + str(bound_ids)
            yield curr_obj, name

        # class of remaining segments
        if rest:
            rest_obj = hierarchy.remove(ids=all_good_seg_ids, new=True)
            if names is None:
                rest_name = 'bound-rest'
            else:
                rest_name = names[-1]
            yield rest_obj, rest_name

    def classByNContacted(self, hierarchy, nContacted, contacts=None, 
                          names=None):
        """
        Classifies segments of arg hierarchy according to the number of 
        contacted boundaries and yields the corresponding classes. 

        Each element of arg ids determines a class. This class contain all
        segments that contact the specified number of boundaries. 

        Class names are either specified in arg names, or the default
        values are used (n-bound-num).

        Arguments:
          - hierarchy: (Hierarchy) hierarchy of segments
          - contacts: (Contacts) contacts between segments and boundaries, if
          None hierarchy.contacts is used
          - nContacted: (list of ints) number of contacted boundaries
          - names: list of class names directly corresponding to arg nContacted

        Yields: 
          - (Hierarchy) a hierarchy of segments
          - class name
        """

        # find contacts
        if contacts is None:
            contacts = hierarchy.contacts

        # classify
        if names is None:
            names = [None] * len(nContacted)
        for num, name in zip(nContacted, names):
            curr_ids = hierarchy.findIdsByNBound(min=num, max=num,
                                                 contacts=contacts)
            curr_obj = hierarchy.keep(curr_ids, new=True)
            if name is None:
                name = 'n-bound-' + str(num)
            yield curr_obj, name


    ###############################################################
    #
    # Other analysis-related methods
    #
    ##############################################################

    def initializeAnalysis(self, doDensity, doMorphology, doLength, 
                           doTopology, doDistanceTo, doBoundDistance,
                           doCleftContacts):
        """
        Initializes objects to hold analysis results.

        Sets attributes of this class to objects corresponding to each of the 
        analysis tasks. These may include: density, morphology, distance (or 
        whatever specified in self.distanceName), boundDistance, nContacts, 
        surfaceDensityContacts, surfaceDensitySegments. However, only those 
        related to the analysis specified by the do* arguments are actually set.

        Each of objects for the analysis tasks is initialized.

        Arguments:
          - do*: Flags indicating id specific analysis should be done
        """

        # deal with flags
        doMorphology = doMorphology or doLength

        # initialize
        if doDensity:
            self.density = Density()
            self.density.initializeData()
            self.resultNames.add('density')
            self.regionDensity = Density()
            self.regionDensity.initializeData()
            self.resultNames.add('regionDensity')
        if doMorphology:
            self.morphology = Morphology()
            self.morphology.initializeData()
            self.resultNames.add('morphology')
            # not good because length not an object
            #if doLength:
            #    self.resultNames.add('length')
        if doTopology:
            self.topology = Topology()
            self.topology.initializeData()
            self.resultNames.add('topology')
        if doDistanceTo:
            if isinstance(self.distanceName, tuple):
                self._distanceName = ['_' + name for name in self.distanceName]
                for name in self.distanceName:
                    setattr(self, name, DistanceTo())
                    getattr(self, name).initializeData()
                    self.resultNames.add(name)
            else:
                self._distanceName = '_' + self.distanceName
                setattr(self, self.distanceName, DistanceTo())
                getattr(self, self.distanceName).initializeData()
                self.resultNames.add(self.distanceName)
        if doBoundDistance:
            self.boundDistance = BoundDistance()
            self.resultNames.add('boundDistance')
        if doCleftContacts:
            self.nContacts = numpy.zeros(shape=(3,1), dtype=int)
            self.surfaceDensityContacts = numpy.zeros(3)
            self.surfaceDensitySegments = numpy.zeros(3)
            self.resultNames.update(['nContacts', 'surfaceDensityContacts',
                                     'surfaceDensitySegments'])

    def mergeAnalysis(self, doDensity, doMorphology, doLength, 
                      doTopology, doDistanceTo, doBoundDistance,
                      doCleftContacts):
        """
        Merges analysis results for the current level segments with the
        already existing results.

        Also sets the following attributes in this objects to the corresponding
        values of the already existing results:
          - morphology.lengthLine
          - morphology.contactMode
          - distance.mode
          - distance.regionIds
        """

        # morphology: data and parameters
        if doMorphology:
            self.morphology.merge(self._morphology, mode='replace', mode0='add')
            if doLength and (self._morphology is not None):
                self.morphology.lengthLine = self._morphology.lengthLine
                self.morphology.contactMode = self._morphology.contactMode

        # density
        if doDensity:
            self.density.merge(
                self._density, mode='replace', mode0='consistent')
            self.regionDensity = self._regionDensity

        # topology
        if doTopology:
            self.topology.merge(self._topology, mode='replace', mode0='add')

        # distance to region: data and parameters 
        if doDistanceTo:
            if isinstance(self.distanceName, tuple):
                name_tup = self.distanceName
                _name_tup = self._distanceName
            else:
                name_tup = (self.distanceName,)
                _name_tup = (self._distanceName,)
            for name, _name in zip(name_tup, _name_tup):
                dist = getattr(self, name)
                _dist = getattr(self, _name)
                try:
                    dist.merge(_dist, mode='replace')
                except (ValueError, AttributeError):
                    if _dist is not None:
                        dist = _dist
                if _dist is not None:
                    dist.mode = _dist.mode
                    dist.regionIds = _dist.regionIds
                setattr(self, name, dist)
                setattr(self, _name, _dist)

        # merge distance between segments
        if doBoundDistance:
            self.boundDistance.merge(self._boundDistance, mode='replace')
    
        # merge cleft contacts
        if doCleftContacts:
            try:
                self.nContacts += self._nContacts
            except ValueError:
                base_len = self.nContacts.shape[1]
                new_len = self._nContacts.shape[1]
                if base_len >= new_len:
                    self.nContacts[:,:new_len] += self._nContacts
                else:
                    new = self._nContacts.copy()
                    new[:,:base_len] += self.nContacts
                    self.nContacts = new
            #self.nContacts = self._nContacts.copy()
            self.surfaceDensityContacts += self._surfaceDensityContacts
            self.surfaceDensitySegments += self._surfaceDensitySegments

    def restrict(self, hierarchy, avoid=['regionDensity']):
        """
        Copies (deep) this object and restricts the analysis results of the 
        new object to segments existing in arg hierarchy.

        Acts on analysis results whose names are specified in self.resultNames,
        except on those listed in arg avoid

        Sets:
          - new.hierarchy to arg hierarchy

        Argument:
          - hierarchy: (Hierarchy) hierarchical segmentation
          - avoid: list of attribute names 

        Returns:
          - new: restricted deepcopy of this object
        """

        # instantiate new object
        new = deepcopy(self)
        new.hierarchy = hierarchy

        # restrict analysis results to existing segments
        for name in new.resultNames:
            if name in avoid:
                continue
            results = getattr(new, name)
            results.restrict(ids=hierarchy.ids)

        return new

    def copyResults(self, segments):
        """
        Makes and returns new instance of this class that contains only the
        new analysis results (deepcopied) and attribute segment (by reference)
        pointing to the arg segments. 

        The new results are those saved in attributes that have names
        listed in self.resultNames but with a leading '_'. In addition,
        '_backgroundDensity' is added.

        Arguments:
          - segments: (Segments) flat segmentation that was analyzed

        Returns: new instance of this class
        """
        
        # instantiate new and add segments
        new_inst = self.__class__()
        new_inst.segments = segments

        # result names
        new_inst.resultNames = deepcopy(self.resultNames)
        if 'density' in self.resultNames:
            new_inst.resultNames.add('backgroundDensity')

        # results
        for name in new_inst.resultNames:
            value = getattr(self, '_' + name)
            setattr(new_inst, name, deepcopy(value))
            
        return new_inst

    def copyAnalysisParameters(self, obj):
        """
        Copies attributes corresponding to analysis parameters (lengthContact,
        lengthLine, nBoundary, boundCount, structElConn, contactStructElConn,
        countStructElConn, distanceId and distanceMode) from obj to this 
        instance.

        Arguments:
          - obj: object of this class
        """

        # length
        self.lengthContact = obj.lengthContact
        self.lengthLine = obj.lengthLine

        # distance to
        self.distanceId = obj.distanceId
        self.distanceMode = obj.distanceMode

        # connectors
        self.nBoundary = obj.nBoundary
        self.boundCount = obj.boundCount

        # structre elements
        self.structElConn=obj.structElConn
        self.contactStructElConn=obj.contactStructElConn
        self.countStructElConn=obj.countStructElConn


    ###############################################################
    #
    # Other methods
    #
    ##############################################################

    def relabel(self, ids):
        """
        Relabels ids.

        Needed for classify_connection script?
        """
        pass
