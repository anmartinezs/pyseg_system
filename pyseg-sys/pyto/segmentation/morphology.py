"""
Class Morphology for calculation of morphological quantities of a segmented
image(label field).

# Author: Vladan Lucic
# $Id: morphology.py 1458 2017-07-04 16:09:33Z vladan $
"""

__version__ = "$Revision: 1458 $"


import sys
import logging
import inspect
from copy import copy

import numpy
import scipy
import scipy.ndimage as ndimage

from features import Features
from statistics import Statistics
from labels import Labels
from segment import Segment

class Morphology(Features):
    """
    Morphological analysis of segments (labels).

    Calculates volume, surface, center and does basic statistics radius
    (distance between the center and the surface elements). 

    Basic usage:

      mor = Morphology(segments=segmented_data_array)
      mor.getVolume()
      ...

    Methods:
      - getVolume():
      - getSurface():
      - getRadius():
      - getCenter():
      - getLength()
      
    Results are stored in the following attributes:
      - self.volume: 1d array containing volumes of all segments (self.volume[i]
      is the volume of segment i)
      - self.surface: array containing surfaces of all segments
      - self.surfaceData
      - self.center:
      - self.radius
      - self.sliceRadius
      - self.length
    """

    ######################################################
    #
    # Initialization 
    #
    #######################################################
    
    # constants
    debug = False

    def __init__(self, segments=None, ids=None):
        """
        Initializes Morphology instance.

        Arguments:
          - segment (Labels object, or ndarray): segments (labels)
          - ids: ids of segments
        """

        # call super to deal with segments and ids
        super(Morphology, self).__init__(segments=segments, ids=ids)

        # declare results data structures
        self.volume = None
        self.surface = None
        self.surfaceData = None
        self.center = None
        self.radius = None
        self.meanRadius = None
        self.sliceRadius = None
        self.meanSliceRadius = None
        self.dataNames = ['volume', 'surface', 'center', 'length']
        self.statDataNames = ['radius', 'meanRadius', 'sliceRadius']

    def setSegments(self, segments, ids=None):
        """
        Sets segments, ids, maxId, mask, rank and structEl.

        Attribute segments is set to segment.data (if segments is an istance
        of Labels), or to segments (if segments is a ndarray). Self.ids
        is set to the first found in ids (argument), segments.ids (Labels
        instance), or to all positive ids present in segments.
        
        Arguments:
          - segments: segments (labels) given as a Labels object or an ndarray
          - ids: segment ids
        """
        
        # call super
        super(Morphology, self).setSegments(segments=segments, ids=ids)

        # make structure element with connectivity ndim
        self.setSurfaceStructEl(connectivity=self.ndim)

    def setSurfaceStructEl(self, connectivity=None):
        """
        Not used anymore.
        
        Sets the structuring element used for surface determination
        (self.surfaceStructEl) and its connectivity (self.surfaceStructElConn).

        Default connectivity is ndim (of the data), that is 26 neighbors in 3d.

        Argument:
          - connectivity: maximum distance squared between the center and the
          elements of the structuring element
        """

        if connectivity is None: connectivity = self.ndim
        self.surfaceStructEl = ndimage.generate_binary_structure(
            rank=self.ndim, connectivity=connectivity)
        self.surfaceStructElConn = connectivity


    ######################################################
    #
    # Basic data manipulation 
    #
    #######################################################
    
    def reorder(self, order, data=None):
        """
        Reorders elements of data array(s).

        If data (1d numarray) is given, its elements are reordered according to
        the dictionary order, where keys are old array indices (segment ids) and
        values are new array indices (segment ids).

        If data is not given, arrays self.volume, self.surface, 
        self.surfaceData, self.center are reordered in the same way.

        Arguments:
          - order: dictionary with old (keys) and new ids (values)
          - data: array to be reordered

        Sets: self.volume, self.surface, self.surfaceData, self.center and
        self.radii, if data is None.

        Returns (new) reordered array if data is not None.
        """

        # merge data from seld.dataNames
        super(Morphology, self).reorder(order=order, data=data)

        if data is None:

            # surfaceData
            if self.surfaceData is not None:
                seg = Segment()
                self.surfaceData = seg.reorder(order=order, 
                                               data=self.surfaceData)

            # reorder self.radius (single or a list of Statistics object(s)) 
            if self.radius is not None:
                if isinstance(self.radius, Statistics):
                    self.radius.reorder(order=order)
                else:
                    for stat in self.radius:
                        stat.reorder(order=order)

    def merge(self, new, names=None, mode='replace', mode0='add'):
        """
        Merges data of Morphology object new with the data of this instance.
        The values of all attributes listed in self.dataNames are added.

        The data attributes whose values are meged are those listed in names or
        in self.dataNames if names is None.

        If mode is 'add' the data is simply added. If mode is 
        'replace' the new values replace the old ones for id that are in 
        new.ids.

        Note: surfaceData, radius and radius-related data are not merged.

        Arguments:
          - new: instance of Morphology
          - names: (list of strings, or a single string) names of data 
          attributes
          - mode: merge mode for data (indices 1+) 'add' or 'replace'
          - mode0: merge mode for index 0, 'add' or 'replace' or a numerical
          value
        """

        # merge data listed in self.dataNames
        super(Morphology, self).merge(
            new=new, names=names, mode=mode, mode0=mode0)

        # ToDo: merge radius and related

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
        super(Morphology, self).restrict(ids=ids)

        # remove non-id values?

        # set total
        self.setTotal()

    def setTotal(self):
        """
        Sets total values (index 0) for volume and surface. The total value
        is the sum of all elements corresponding to self.ids.
        """

        ids = self.ids

        if (ids is None) or (len(ids) == 0):
            return

        if self.volume is not None:
            self.volume[0] = self.volume[ids].sum()
        if self.surface is not None:
            self.surface[0] = self.surface[ids].sum()

    ######################################################
    #
    # Volume surface and center
    #
    #######################################################    

    def getVolume(self, ids=None):
        """
        Calculates volumes of segments given by ids, or all segments if 
        ids=None, and the total volume.

        Sets (int) self.volume.

        Returns:
          - array containing all volumes calculated so far, that is the volumes
          for ids as well as volumes calculated earlier.  
        """

        # set ids
        if ids is not None: self.setIds(ids)
        
        # make or enlarge self.volume
        if self.volume is None:
            self.volume = numpy.zeros(self.maxId+1, dtype=int)
        elif self.maxId >= self.volume.shape[0]:
            newVolume = numpy.zeros(self.maxId+1, dtype=int)
            newVolume[0:self.volume.shape[0]] = self.volume
            self.volume = newVolume

        # return if no ids (otherwise segfaults) 
        if self.maxId == 0: return self.volume
            
        # calculate volume for each element of ids and convert to int
        volume = ndimage.sum(self.segments, self.segments, self.ids) / self.ids
        volume = numpy.rint(volume).astype(int)

        # put data in self.volume
        self.volume[self.ids] = volume

        # calculate total volume
        self.volume[0] = 0
        self.volume[0] = numpy.sum(self.volume)

        # return all volumes
        return self.volume

    def getSurface(self, ids=None, size=1, saveData=False, copy=True):
        """
        Calculates surfaces of all segments.

        Elements of self.data that are closer than size to the background are
        considered to form surfaces of segments.

        The default size value of 1 gives the same result as making surfaces
        by erosion using structuring element with connecectivity = 1 (only 
        elements sharing a face with outside elements form the surface). Note 
        that a size used for a structuring element corresponds to the square of 
        the size used here. 

        Modifies self.surface if copy is False.        

        Arguments:
          - ids: segment ids
          - size: maximum distance from background for a surface element
          - saveData: if true, surface array is saved in self.surfaceData
          - copy: if True works on a copy of self.data, so that self.data is not
          modified by this method

        Returns (ndarray) self.surface
        """

        # set ids
        if ids is not None: self.setIds(ids)
        
        # make or enlarge self.surface
        if self.surface is None:
            self.surface = numpy.zeros(self.maxId+1, dtype=int)
        elif self.maxId >= self.surface.shape[0]:
            newSurface = numpy.zeros(self.maxId+1, dtype=int)
            newSurface[0:self.surface.shape[0]] = self.surface
            self.surface = newSurface
            
        # return if no ids (otherwise segfaults) 
        if self.maxId == 0: return self.surface
            
        # make surfaces
        surfaceData = self.makeSurface(size=size, save=saveData, copy=copy)
        
        # calculate surface for each element of ids and covert to int
        surface = ndimage.sum(surfaceData, surfaceData, self.ids) / self.ids
        surface = numpy.rint(surface).astype(int)

        # put data in self.surface
        if self.surface is None:
            self.surface = numpy.zeros(self.maxId+1, dtype=int)
        self.surface[self.ids] = surface

        # calculate total surface
        self.surface[0] = 0
        self.surface[0] = numpy.sum(self.surface)

        return self.surface

    def makeSurface(self, size=1, save=False, copy=True):
        """
        Extracts surfaces of all segments.

        Elements of self.data that are closer than size to the background are
        considered to form surfaces of segments (see Segmentation.makeSurfaces) 
        for details).

        Modifies self.surface if copy is False.

        Arguments:
          - size: maximum distance from background for a surface element
          - save: if true, surface array is saved in self.surfaceData
          - copy: if True works on a copy of self.data, so that self.data is not
          modified by this method
        """

        # make copy of data if needed
        if copy:
            data = numpy.array(self.segments)
        else:
            data = self.segments

        # make surfaces
        seg = Segment()
        surface = seg.makeSurfaces(data=data, size=size, ids=self.ids)
        
        #surface = self.segments - \
        #      ndimage.grey_erosion(self.segments, 
        #                           footprint=self.surfaceStructEl,
        #                           mode='constant', cval=0)
        if save: self.surfaceData = surface

        return surface
        
    def getCenter(self, ids=None, segments=None, real=False, inset=False):
        """
        Calculates centers of all segments. Segments are given by arg
        segments or self.segments.

        Sets (int or float) ndarray self.center, where self.center[i] contains
        position of the i-th segment coordinates, and self.center[0] is the
        center of all segments taken together.

        If arg inset is True, the calculated centers take into account inset 
        info of the segments image (self.segments.inset). Otherwise centers
        are calculated in respect to the actual array (self.segments.data).

        In inset is true it is necessary that arg segments or self.segments 
        (if arg segments is None) is an instance of Labels. 

        Elements of self.center corresponding to ids that are not in ids (but 
        are smaller than max(ids) are set to -1.

        Arguments:
          - ids: segment ids
          - real: flag indicating if centers should be real numbers (floats) or
          they should be converted to integers
          - segments: (Labels or ndarray) segmented image
          - inset: flag indicating if segments image inset is taken into account

        Sets and Returns:
          - self.center: array where element i contains coordinates of the 
          center of segment i
        """

        # set ids
        if ids is not None: self.setIds(ids)

        # set segments and inset
        inset_value = None
        if segments is not None:
            if isinstance(segments, Labels):
                segments_data = segments.data
                inset_value = segments.inset
            else:
                segments_data = segments
        else:
            if isinstance(self.segments, Labels):
                segments_data = self.segments.data
                inset_value = self.segments.inset
            else:
                segments_data = self.segments
        if inset and (inset_value is None):
            raise ValueError(
                "Inset correction requested, but the value of inset "
                + "is not present.")
        
        # set dtype
        if real:
            dtype = float
        else:
            dtype = int
        
        # make or enlarge self.center
        if self.center is None:
            self.center = numpy.zeros(shape=(self.maxId+1, self.ndim), 
                                      dtype=dtype)
            self.center = self.center - 1
        elif self.maxId >= self.center.shape[0]:
            newCenter = numpy.zeros(shape=(self.maxId+1, self.ndim), 
                                    dtype=dtype)
            newCenter[0:self.center.shape[0],:] = self.center
            newCenter[self.center.shape[0]:,:] = -1
            self.center = newCenter
            
        # return if no ids (otherwise segfaults) 
        if self.maxId == 0: return self.center
            
        # calculate center positions
        center = ndimage.center_of_mass(segments_data, segments_data, self.ids)
        if not real:
            center = numpy.rint(center).astype(int)
        # changed (fixed) in scipy
        #if self.ids.size == 1: center = numpy.array([center])
        self.center[self.ids] = center

        # center for all segments
        center0 = ndimage.center_of_mass(input=segments_data, 
                                         labels=segments_data)
        if not real:
            center0 = numpy.rint(center0).astype(int)             
        self.center[0,:] = center0

        # use inset
        if inset:
            self.center = (self.center + 
                           numpy.asarray([x.start for x in inset_value]))

        return self.center


    ######################################################
    #
    # Radius
    #
    #######################################################    

    def getRadius(self, ids=None, surface=None, doSlices=False, axis=None):
        """
        Calculates basic statistics (mean, std, min and max) for distances 
        between a center and surface elements for each segment.

        If argument surface is not given, self.surfaceData is used if it exists.
        Otherwise, the surfaces are determined using self.makeSurface method.
        Strictly speeking, surfaces can contain segments of any form, not just
        surfaces of self.data segments.

        Note that even if center position is calculated as float, in order to
        calculate the distances the center is moved to int. 

        If doSlices is true, the same distances are calculated for slices
        of segments along the major axis given by axis (in 3d axis=0 means along
        the x-axis, that is a yz slice). If axis is None the distances are
        calculated for slices along all majos axes.

        Arguments:
          - ids: segment ids
          - surface: ndarray containig surfaces
          - doSlices: flag indicating if the statistics is also done on the
          slices along major axes
          - axis: if doSlices is True, axis indicates along which major axis
          slice radius is calculated. 

        Sets:
          - self.radius: instance of Statistics containig radius statistics,
          that is attributes mean, std, min and max. Each of these attributes
          is an array indexed by ids
          - self.sliceRadius: array of Statistics instances containing radius
          statistics for slices along the major axis (in the same form as
          self.radius)
          - self.surfaceData: surface array
        """

        # set ids
        if ids is not None: self.setIds(ids)
        
        # return if no ids (otherwise segfaults) 
        if self.maxId == 0: return 
            
        # make surface array
        if surface is None:
            if self.surfaceData is None:
                self.makeSurface(save=True)
            surface = self.surfaceData

        # calculate center if needed
        if (self.center is None) or (self.maxId >= self.center.shape[0]):
            self.getCenter()

        # make slices (subarrays) for each segment
        seg_slices = ndimage.find_objects(self.segments)

        # prepare input array for distance_transform function 
        distances = numpy.zeros(shape=surface.shape, dtype='float32')

        # find distances for each segment (slice) separately
        for id_ in self.ids:

            # a_list[(numpy.int32)id_] throws list indices must be ints (19.05.08)
            id_ = int(id_)

            # make background (not-this--surface) id to be different than id and
            # 0, but beware of overflow
            if id_ > 1:
                bkg_id = id_ - 1
            else:
                bkg_id = id_ + 1            

            # generate inset for this segment
            try:
                inputInset = surface[seg_slices[id_-1]].copy()
            except IndexError:
                logging.warning('(%s:%u) Segment with id_ = %u does not ' \
                                + 'seem to exist. Skipped.',
                        inspect.stack()[0][1], sys.exc_info()[2].tb_lineno, id_)
                continue
                
            # set all elements that are not on the suface of this element
            # to tmpBkgid
            inputInset[inputInset!=id_] = bkg_id

            # set center position to int and put 0 there
            try:
                insetCoord = [sl.start for sl in seg_slices[id_-1]]
            except TypeError:
                logging.warning('(%s:%u) Segment with id_ = %u does not ' \
                                + 'seem to exist. Skipped.',
                        inspect.stack()[0][1], sys.exc_info()[2].tb_lineno, id_)
                continue
            centers = numpy.rint(self.center[id_]).astype('int')
            insetCenter = centers - numpy.array(insetCoord, dtype=int)
            inputInset[tuple(insetCenter)] = 0

            # generate distances
            if (inputInset > 0).all():  # workaround for scipy bug 1089
                raise ValueError("Can't calculate distance_function ",
                                 "(no background)")
            else:
                distanceInset = ndimage.distance_transform_edt(input=inputInset)
            
            # save distances for this segement only (important!) 
            distanceInset[inputInset!=id_] = 0
            distances[seg_slices[id_-1]] += distanceInset

        # save distances if needed
        if Morphology.debug: self.distances = distances

        # do statistics
        self.doStatistics(data=distances, segments=surface,
                          ids=self.ids, centers=self.center, output='radius')
        if doSlices:
            self.doStatistics(data=distances, segments=surface, ids=self.ids,
                              centers=self.center, sliceCoord=self.center, 
                              axes=[axis], output='sliceRadius')
            
    def doStatistics(self, data, segments=None, ids=None, centers=None,
                          sliceCoord=None, axes=None, output=None):
        """
        Do statistics on segmented data (according to Statistics.calculate()).

        If centers is given calculate also positions and angles (spherical
        coordinates) of min and max in respect to center.

        If sliceCoord is None the statistics are calculated for (whole) labels.
        Otherwise, the statistics are done on ndim-1 slices of data that contain
        sliceCoord coordinate, along axis.

        if output is a name of an attribute holding an existing instance
        of Statistics (or a list od instances) the results are merged with
        the existing results in that (list of) instance(s).  

        Arguments:
          - data: data to be analyzed
          - segments: define segments
          - ids: ids of segments to be analyzed, array or a single int
          - centers: centers for each segment
          - sliceCoord: coordinates of points defining slices for each segment
          - axes: axes along which the segments are sliced (all axes if None)
          - output: name of the attribute (string) pointing to the object
          holding the results (instance of Statistics or a list of Statistics
          instances). 

        Returns an instance (sliceCoord=None), or a list of Statistics instances
        (for each axis) containing the statistics for each slice.
        """

        # parse arguments
        if segments is None: segments = self.segments
        if ids is None: ids = self.ids

        # initialize output if needed
        try: self.__dict__[output] is None
        except KeyError: self.__dict__[output] = None

        # initialize mean output
        meanOutput = 'mean' + output[0].capitalize() + output[1:len(output)]
        try: self.__dict__[meanOutput] is None
        except KeyError: self.__dict__[meanOutput] = None

        # make id mask
        idMask = numpy.zeros(self.maxId+1, dtype=int)
        idMask[ids] = 1

        # work
        if sliceCoord is None:

            # instantiate output if needed and do statistics on each segment
            if self.__dict__[output] is None:
                self.__dict__[output] = Statistics()
            self.__dict__[output].calculate(data=data, labels=segments,
                                            ids=ids, centers=centers)

            # do statistics on the mean values
            if self.__dict__[meanOutput] is None:
                self.__dict__[meanOutput] = Statistics()
            self.__dict__[meanOutput].calculate(data=self.__dict__[output].mean,
                                                labels=idMask, ids=1)

        else:

            # make axes if needed
            if axes is None: axes = range(data.ndim)

            # instantiate output if needed 
            if self.__dict__[output] is None:
                self.__dict__[output] = [None] * data.ndim
                for axis in axes:
                    self.__dict__[output][axis] = \
                        Statistics(data=data, labels=segments, ids=ids,
                                   sliceCoord=sliceCoord, axis=axis)

            # do statistics on each sliced segment for each axis
            for axis in axes:
                # labels argument should not be given because it would
                # overwrite the sliced labels generated by the constructor
                self.__dict__[output][axis].calculate(centers=centers)
                
            # instantiate mean output if needed 
            if self.__dict__[meanOutput] is None:
                self.__dict__[meanOutput] = [None] * data.ndim
                for axis in axes:
                    self.__dict__[meanOutput][axis] = \
                        Statistics(data=self.__dict__[output][axis].mean,
                                   labels=idMask, ids=1)

            # do statistics on each sliced segment for each axis
            for axis in axes:
                # labels argument should not be given because it would
                # overwrite the sliced labels generated by the constructor
                self.__dict__[meanOutput][axis].calculate()
                
        return self.__dict__[output]
        
    def labelExtrema(self, stats, ids=None, minLabel=None, maxLabel=None):
        """
        Makes labeles that indicate psitions of extrema.

        Arguments:
          - stats: Statistics object holding min and max positions
          - ids: segment ids 
          - minLabel: number used to label min positions
          - maxLabel: number used to label max positions

        Returns: ndarray with min and max positions labeled
        """

        # parse arguments
        if ids is None: ids = self.ids
        if minLabel is None: minLabel = 1
        if maxLabel is None: maxLabel = 2

        # initialize extrema labels array
        extreme = numpy.zeros_like(self.segments)

        # label all min and max positions 
        extreme[ tuple(numpy.transpose(stats.minPos[ids])) ] = minLabel
        extreme[ tuple(numpy.transpose(stats.maxPos[ids])) ] = maxLabel

        return extreme
    

    ######################################################
    #
    # Length
    #
    #######################################################    

    def getLength(
            self, segments, boundaries, contacts, ids=None, distance='b2b', 
            structElConn=1, line='straight', position=False):
        """
        Calculates lengts of segments specified by (args) segments and ids. The
        segments can contact exactly one or two boundaries. 

        In the one boundary case, the length is calculated as the maximal 
        distance between segment points and the contact region, that is using 
        the 'straight' line mode. (The distance between a point and a contact 
        region is the distance between the point and its closest contact point.)

        In the two boundary case, there are two possibilities. If the line mode 
        is 'straight', the length is calculated as a smallest straight 
        (Euclidean) distance between points on the two contact regions. 
        Otherwise, in the 'mid' or 'mid-seg' line modes, the length is 
        calculated as a smallest sum of distances between a 'central' and two 
        contact points. A central point has to belong to the intersection of 
        the segment and a central layer formed exactly in the middle between 
        the two boundaries. In other words, the sum of distances is minimized 
        over all contact and mid points. 

        The difference between the 'mid' and the 'mid-seg' modes is that in the 
        'mid-seg' mode the layers between the boundaries are formed on the 
        segment alone, while in the 'mid' mode they are formed on both the 
        segment and the neighboring inter-boundary region. Consequently, the 
        layers formed using the 'mid-seg' mode and the distance calculated, 
        might be a bit more precise.  

        If argument distance is 'b2b' (two boundaries) or 'b-max' 
        (one boundary), contact points are elements of boundaries that contact 
        a segment, so the length is calculated between the boundaries. If it is 
        'c2c' (two boundaries) or 'c-max' (one boundary), contact points are 
        elements of segments that contact a boundary. Consequently, the lengths 
        calculated in the 'b2b' or 'b-max' modes are expected to be up to two 
        pixels longer than those calculated in the 'c2c' or 'c-max' modes. 

        In the case of two boundaries, the length is calculated between
        the contact points on the boundary (first end) and the segment 
        (second end) dor arg distance 'b2c' and the other way round for 'c2b'.

        Arguments line and distance are saved as attributes lengthLine and 
        contactMode, respectivly.

        If arg position is True, the positions of contact points (contact and 
        end point for one boundary) are also calculated, generally increasing 
        the run-time.
        
        Segments and boundaries objects have to have the same positioning 
        (attributes offset and inset). 

        Arguments:
          - segments: (Segment) object containing segments whose langths are
          calculated
          - bondaries: (Segment) object defining boundaries
          - contacts: (Contact) object containing the info about the contacts
          between the segments and the boundaries
          - ids: segment ids
          - distance: for two boundaries: 'b2b' (or 'boundary') for 
          distance between contact points on boundaries, 'c2c' (or 'contact') 
          between contact points on segments, 'b2c' between boundary and 
          segment contacts and 'c2b' between segment and boundary contacts. 
          For one boundary: 'b-max' and 'c-max'.
          - structElConn: (square) connectivity of the structuring element used
          to detect contacts (can be 1 to ndim).
          - line: The type of the line used to calculate the distance in the
          2-boundary cases: 'straight', 'mid' or 'mid-seg'
          - position: flag indicating if the positions of the contact points
          used for the length measurment are calculated and returned, used only
          
        Return:
          - length: if pos is False
          - length, position_contact, position_other: in the one boundary case
          - length, position_1, position_2: in the two boundary case

        ToDo: make something like an average of b2b and c2c
        """

        # set segments and ids
        self.setSegments(segments=segments, ids=ids)
        ids = self.ids
        
        # set structuring element
        struct_el = ndimage.generate_binary_structure(rank=self.ndim,
                                                     connectivity=structElConn)
        
        # figure out expected number of boundaries
        if (distance == 'b-max') or (distance == 'c-max'):
            n_bound = 1
        elif ((distance == 'b2b') or (distance == 'boundary') 
              or (distance == 'c2c') or (distance == 'contact') 
              or (distance == 'b2c') or (distance == 'c2b')):
            n_bound = 2
        else:
            n_bound = -1
            raise ValueError(
                "Argument distance: " + str(distance) + "  was not understood."
                + "Defined values are 'b2b', 'boundary', 'c2c', 'contact', "
                + "'b2c' and 'c2c', 'b-max' and 'c-max'.")

        # save original insets
        seg_data, seg_inset = segments.getDataInset()
        seg_inset = copy(seg_inset)
        bound_data, bound_inset = boundaries.getDataInset()
        bound_inset = copy(bound_inset)

        # find length for each segment
        length = numpy.zeros(self.maxId+1) - 1
        if position:
            position_1 = numpy.zeros((self.maxId+1, self.ndim), dtype='int') - 1
            position_2 = numpy.zeros((self.maxId+1, self.ndim), dtype='int') - 1
        for seg_id in ids:

            # find boundaries
            b_ids = contacts.findBoundaries(segmentIds=seg_id, nSegment=1)

            # use appropriate method to get the length of this segment
            if len(b_ids) == n_bound:

                # use smaller arrays for calculations
                
                segments.makeInset(ids=[seg_id], extend=1, expand=True)
                boundaries.useInset(
                    inset=segments.inset, mode='abs', expand=True)

                # calculate 
                if n_bound == 1:
                    res = self._getSingleLength1Bound(
                        segments=segments, boundaries=boundaries, 
                        boundaryIds=b_ids, id_=seg_id, distance=distance, 
                        structEl=struct_el, line=line, position=position)
                elif n_bound == 2:
                    res = self._getSingleLength2Bound(
                        segments=segments, boundaries=boundaries, 
                        boundaryIds=b_ids, id_=seg_id, distance=distance, 
                        structEl=struct_el, line=line, position=position)
                    
                # parse result
                if position:
                    length[seg_id] = res[0]
                    position_1[seg_id] = [pos + ins.start for pos, ins \
                                              in zip(res[1], segments.inset)]
                    position_2[seg_id] = [pos + ins.start for pos, ins \
                                              in zip(res[2], segments.inset)]
                else:
                    length[seg_id] = res

                # recover full data
                segments.setDataInset(data=seg_data, inset=seg_inset)
                boundaries.setDataInset(data=bound_data, inset=bound_inset)

        # assign attributes
        self.length = length
        self.lengthLine = line
        self.contactMode = distance
        if position:
            self.end1 = position_1
            self.end2 = position_2

        # return
        if position:
            return length, position_1, position_2
        else:
            return length

    def _getSingleLength1Bound(
            self, segments, id_, boundaries, boundaryIds, distance, 
            structEl, line='straight', position=False):

        """
        Calculate length of a given segment that contacts exactly one boundary.

        The length is calculated as the maximal distance between segment points
        and the contact region. (The distance between a point and a contact 
        region is the distance between the point and its closest contact point.)
       """

        # alias
        b_ids = boundaryIds

        # restrict to a subarray that contains current segment and boundaries 
        region = (segments.data == id_) \
            | ((boundaries.data == b_ids[0]))
        inset = ndimage.find_objects(region)[0]
        local_seg = Segment(data=segments.data[inset], copy=True, ids=[id_], 
                            clean=True)
        local_bound = Segment(data=boundaries.data[inset], copy=True, 
                              ids=b_ids, clean=True)

        # make contacts
        if (distance == 'b-max'):
            dilated = ndimage.binary_dilation(input=local_seg.data==id_, 
                                              structure=structEl)
            contact_1 = dilated & (local_bound.data == b_ids[0]) 
        elif (distance == 'c-max'):
            dilated_1 = ndimage.binary_dilation(
                input=local_bound.data==b_ids[0], structure=structEl)
            contact_1 = dilated_1 & (local_seg.data == id_) 

        # distances from contacts
        if (~contact_1 > 0).all():  # workaround for scipy bug 1089
            raise ValueError("Can't calculate distance_function ",
                             "(no background)")
        else:
            dist_1 = ndimage.distance_transform_edt(input=~contact_1)

        # max distances between an element of segment and contacts
        length = ndimage.maximum(input=dist_1, labels=local_seg.data) 

        # line
        if line != 'straight':
            logging.warning(
                'Sorry, the only implemented line mode for segment '
                + "length calculation is 'straight'. Continuing "
                + "using line mode 'straight'.")

        # find positions of the points used to calculate length
        if position:
            
            # position of the non-contact maximum position
            pos_2 = ndimage.maximum_position(input=dist_1, 
                                             labels=local_seg.data)

            # distance from the non-contact maximum point
            point_2 = numpy.zeros_like(contact_1)
            point_2[pos_2] = 1
            if (~point_2 > 0).all():  # workaround for scipy bug 1089
                raise ValueError("Can't calculate distance_function ",
                                 "(no background)")
            else:
                dist = ndimage.distance_transform_edt(input=~point_2)
            
            # position of the contact max point
            pos_1 = ndimage.maximum_position(input=dist, labels=local_seg.data)

            return length, pos_1, pos_2

        return length

    def _getSingleLength2Bound(
            self, segments, id_, boundaries, boundaryIds,
            distance, structEl, line='straight', position=False):

        """
        Calculate length of a given segment that contacts exactly two 
        boundaries.

        The length is calculated as a shortest path between a contact point with
        one boundary, a segment point lying on the middle layer between the
        boundaries and a contact point on the other boundary. 

        If the line mode is 'straight', the length is calculated as a 
        smallest straight (Euclidean) distance between points on the two 
        contact regions. 
        
        Otherwise, in the 'mid' or 'mid-seg' line modes, the length is 
        calculated as a smallest sum of distances between a 'central' and two 
        contact points. A central point has to belong to the intersection of 
        the segment and a central layer formed exactly in the middle between 
        the two boundaries. In other words, the sum of distances is minimized 
        over all contact and mid points. 
        """

        # alias
        b_ids = boundaryIds

        # restrict to a subarray that contains current segment and boundaries 
        region = (segments.data == id_) \
            | ((boundaries.data == b_ids[0]) | (boundaries.data == b_ids[1]))
        inset = ndimage.find_objects(region)[0]
        local_seg = Segment(data=segments.data[inset], copy=True, ids=[id_], 
                            clean=True)
        local_bound = Segment(data=boundaries.data[inset], copy=True, 
                              ids=b_ids, clean=True)

        # make contacts
        if (distance == 'b2b') or (distance == 'boundary'):
            dilated = ndimage.binary_dilation(input=local_seg.data==id_, 
                                              structure=structEl)
            contact_1 = dilated & (local_bound.data == b_ids[0]) 
            contact_2 = dilated & (local_bound.data == b_ids[1])
        elif (distance == 'c2c') or (distance == 'contact'):
            dilated_1 = ndimage.binary_dilation(
                input=local_bound.data==b_ids[0], structure=structEl)
            contact_1 = dilated_1 & (local_seg.data == id_) 
            dilated_2 = ndimage.binary_dilation(
                input=local_bound.data==b_ids[1], structure=structEl)
            contact_2 = dilated_2 & (local_seg.data == id_)
        elif (distance == 'b2c'):
            dilated_1 = ndimage.binary_dilation(input=local_seg.data==id_, 
                                              structure=structEl)
            contact_1 = dilated_1 & (local_bound.data == b_ids[0]) 
            dilated_2 = ndimage.binary_dilation(
                input=local_bound.data==b_ids[1], structure=structEl)
            contact_2 = dilated_2 & (local_seg.data == id_)
        elif (distance == 'c2b'):
            dilated_1 = ndimage.binary_dilation(
                input=local_bound.data==b_ids[0], structure=structEl)
            contact_1 = dilated_1 & (local_seg.data == id_) 
            dilated_2 = ndimage.binary_dilation(input=local_seg.data==id_, 
                                              structure=structEl)
            contact_2 = dilated_2 & (local_bound.data == b_ids[1])
        else:
            raise ValueError(
                "Argument distance: " + str(distance) + "  was not understood."
                + "Defined values are 'b2b', 'boundary', 'c2', 'contact', "
                + "'b2c' and 'c2c'.")

        # distances from contacts 1
        if (~contact_1 > 0).all():  # workaround for scipy bug 1089
            raise ValueError("Can't calculate distance_function ",
                             "(no background)")
        else:
            dist_1 = ndimage.distance_transform_edt(input=~contact_1)

        if line == 'straight':

            # get straight length
            length = ndimage.minimum(input=dist_1, labels=contact_2) 

            # get position
            if position:
                pos_2 = ndimage.minimum_position(input=dist_1, labels=contact_2)
                point_2 = numpy.zeros_like(contact_2)
                point_2[pos_2] = 1
                if (~point_2 > 0).all():  # workaround for scipy bug 1089
                    raise ValueError("Can't calculate distance_function ",
                                     "(no background)")
                else:
                    dist_2 = ndimage.distance_transform_edt(input=~point_2)
                pos_1 = ndimage.minimum_position(input=dist_2, labels=contact_1)

                return length, pos_1, pos_2

            return length

        elif (line == 'mid') or (line == 'mid-seg'):

            if (~contact_2 > 0).all():  # workaround for scipy bug 1089
                raise ValueError("Can't calculate distance_function ",
                                 "(no background)")
            else:
                dist_2 = ndimage.distance_transform_edt(input=~contact_2)

            # make layers 
            if (line == 'mid'):
                layers, lay_dist = local_bound.makeLayersBetween(
                    bound_1=b_ids[0], bound_2=b_ids[1], mask=0, between='min')
            elif (line == 'mid-seg'):
                layers, lay_dist = local_bound.makeLayersBetween(
                    bound_1=b_ids[0], bound_2=b_ids[1], mask=local_seg.data, 
                    between='min')

            # make sure the middle layer id is at least 1
            if lay_dist <= 1:
                half = 1
            else:
                half = int(numpy.rint(lay_dist / 2))

            # keep only the middle layer(s)
            layers.keep(ids=[half])
            middle = (local_seg.data == id_) & (layers.data > 0)

            # min sum of distances to both contacts
            length = ndimage.minimum(input=dist_1+dist_2, labels=middle) 

            # find positions of points used to calculate length
            if position:

                # position of the point on the middle layer having min distance
                mid_position = ndimage.minimum_position(input=dist_1+dist_2, 
                                                        labels=middle)

                # distances to the mid point
                mid_point = numpy.zeros_like(contact_1)
                mid_point[mid_position] = 1
                if (~mid_point > 0).all():  # workaround for scipy bug 1089
                    raise ValueError("Can't calculate distance_function ",
                                     "(no background)")
                else:
                    mid_dist = ndimage.distance_transform_edt(input=~mid_point)

                # positions of points having min distances to contacts
                pos_1 = ndimage.minimum_position(input=mid_dist, 
                                                 labels=contact_1)
                pos_2 = ndimage.minimum_position(input=mid_dist, 
                                                 labels=contact_2)

                return length, pos_1, pos_2

            return length

        else:
            raise ValueError("Line mode: " + line + " was not recognized. " \
                             + "Available line modes are 'straight' and 'mid'.")
