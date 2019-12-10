"""
Class Statistics for basic statistical analysis of labeled (segmented) data.

# Author: Vladan Lucic
# $Id: statistics.py 885 2012-09-26 13:33:26Z vladan $
"""

__version__ = "$Revision: 885 $"


import scipy
import scipy.ndimage as ndimage
import numpy


class Statistics(object):
    """
    Basic statistical analysis of labeled (segmented) data.

    Basic usage for calculating statistics:

      st = Statistics()
      st.calculate(data=data_array, labels=labels_array, ids=[1,3,7])

    The results (mean, std, min, max, minPos, maxPos) are stored in arrays
    with the same names (mean, std, ...). The individual values can be obtained
    as: st.mean[id], st.std[id], ... and the total values (all segments taken
    together) are in st.mean[0], st.std[0], ... .

    Slightly more complicated usage:

      st = Statistics(data=data_array, labels=labels_array, ids=[1,3,7])
      st.calculate(centers=array_of_positions)

    In addition to the above results, positions of min/max in respect to
    centers are calculated in cartesian (minVec, maxVec) and spherical
    coordinates if appropriate (minPhi, maxPhi in 2-3d, and minTheta,
    maxTheat in 3d).

    Even more complicated:

      st = Statistics(data=data_array, labels=labels_array, ids=[1,3,7],
                      sliceCoord=array_of_positions, axis=1)
      st.calculate(centers=array_of_positions)

    The same results are calculated as above, but instead of segments as
    specified in labels, each segment is restricted to a ndim-1 dimensional
    slice defined by the position given as the corresponding element of
    sliceCoord array and axis.
    """

    ##################################################################
    #
    # Initialization of data structures and related attributes
    #
    ##################################################################
        
    def __init__(self, data=None, labels=None, ids=None, sliceCoord=None, 
                 axis=0):
        """
        Sets self.data, self.labels and related attributes.

        Attributes data and labels can be changed using setData method. If
        ids are not given, all ids present in labels are considered.

        Arrays data and labels are not modified by methods of this class. 

        If sliceCoord is given, the statistics are not calculated on the 
        segments (labels), but on a (ndim-1 dimensional) slices of labels. 
        The slice used for a given label is defined by the corresponding 
        position given in sliceCoord and by axis. SliceCoords and and axis 
        can't be changed.

        If ids is a single number a flag self._numIds is set to True. If ids 
        is an array (even if it has 0, or 1 element) self._numInd = False

        Arguments:
          - data: (ndarray) image to be analyzed
          - labels: ndarray that defines segements
          - ids: array of ids, or a single int
          - sliceCoord: array where each element specifies coordinates of  
          - axis:
        """
        # initial values
        self.calculated = None
        self.data = None
        self.labels = None
        self._ids = None

        # declare results data structures
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.minPos = None
        self.maxPos = None
        self.minVec = None
        self.maxVec = None
        self.minPhi = None
        self.maxPhi = None
        self.minTheta = None
        self.maxTheta = None

        # parse arguments
        self.setData(data=data, labels=labels, ids=ids,
                         sliceCoord=sliceCoord, axis=axis)

    def setData(self, data=None, labels=None, ids=None,
                    sliceCoord=None, axis=0):
        """
        Sets self.data, self.labels and related attributes (_maxId, ids,
        calculated) and initializes arrays that hold results.

        However, inconsistencies may arise if the dimensions of shape and labels
        are changed. Also, it does not reset the results data structures,
        so the results may contain values for both previous and current
        data and labels for different ids.

        If sliceCoord is given, each segment (from labels) is restricted to
        a ndim-1 subarray defined by sliceCoord element corresponding to the
        segment and axis. Attribute self.labels is changed to contain only
        the ndim-1 dimensional segments.

        Arguments:
          - data: array to be analyzed
          - labels: labels (segmentation) array), default all 1's
          - ids: array (or other iterrable)  of ids. Can be a single int for
          1d data only
          - sliceCoord: array of positions that (together with axes) define
          the ndim-1 dimensional slices of labels
          - axis: axis perpendicular to the ndim-1 dimensional slices
        """

        # set self.data and self.ndim
        if data is not None: self.data = data
        try:
            self.ndim = self.data.ndim
        except AttributeError:
            pass

        # set self.labels to labels, or ones (if self.data exists)
        try: 
            if labels is not None:
                self.labels = labels
            elif self.labels is None:
                self.labels = numpy.ones(shape=self.data.shape, dtype=int)
        except AttributeError:
            pass

        # set ids, _maxId and calculated
        self._setIds(ids)

        # set self.labels to a slice through self.labels if needed
        if sliceCoord is not None: self.setSlicedLabels(sliceCoord, axis)

    def _setIds(self, ids=None):
        """
        Sets self._ids (type ndarray) either to ids if ids given, or to
        the array of all ids present in self.labels. self._maxId is then set
        to the max id of self_.ids

        Also sets self._singleId to True if ids is a single int.

        Arguments:
          - ids: list of ids, or a single int 
        """

        # set self._ids, self._maxId
        if ids is not None:

            # from ids
            if isinstance(ids, int):
                self._numberId = True
                ids = [ids]
            else:
                self._numberId = False
            try: 
                self._ids = numpy.array(ids)
                self._maxId = self._ids.max()
            except ValueError:
                # ids is []
                self._ids = numpy.array(ids, dtype='int_')
                self._maxId = 0

        elif self._ids is None and self.labels is not None:

            # from self.labels  
            try:
                all = numpy.unique(self.labels)
                self._ids = all.compress(all>0)
                self._maxId = self._ids.max()
            except (AttributeError, ValueError):
                self._ids = numpy.array([], dtype='int_')
                self._maxId = 0

        # create or enlarge self.calculated
        if self._ids is not None:
            self._prepareArrays(arrays=('calculated',), dtypes=(bool,))

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

        Sets all data attributes (self.mean, self.std, self.min, ...) if data
        is None.

        Returns (new) reordered array if data is not None.
        """

        if data is None:

            # reorderes all data of this instance
            vars = ['mean', 'std', 'min', 'max', 'minPos', 'maxPos', 'minVec',
                    'maxVec', 'minPhi', 'maxPhi', 'minTheta', 'maxTheta']
            for var in vars:
                if self.__dict__[var] is not None:
                    self.__dict__[var] = self.reorder(order=order,
                                                      data=self.__dict__[var])

        else:

            # reorderes data array
            reordered = data.copy()
            reordered[order.values()] = data[order.keys()]
            return reordered

    def _prepareArrays(self, arrays, dtypes, widths=0):
        """
        Creates or extends 1D and 2D arrays along axis 0.

        For each array, if self.array is None, a new array of dimension
        self.maxId+1 along axis 0 is created. If an array already exist,
        it is extended along axis 0 (the new dimension is a new value
        of self._maxId+1).

        If an array is created, its data type is taken from dtypes. The new
        array is 1d if the corresponding width <= 1, and 2d (dimension along
        axis 1 is given by the width) otherwise.

        An extended array keeps all the elements of the old one. The new 
        elements are set to 0. It also keeps the dtype and the shape from the 
        old array (arguments dtypes and widths are not used).

        Arguments:
          - arrays: list of attribute names (strings) of the arrays to be
          initialized or extended
          - dtypes: list of dtypes of arrays (used only for initialization)
          - widths: list of (or single int) dimensions of the array along axis 1
          For a width <= 1, 1d an array is created, otherwise an 2d array. Used
          only for initializat
        """

        # parse widths
        if isinstance(widths, int): widths = [widths] * len(arrays)

        for [attr, dtp, wid] in zip(arrays, dtypes, widths):
            arr = self.__dict__[attr]
            if wid <= 1:

                # make 1D arrays           
                if arr is None:
                    self.__dict__[attr] = numpy.zeros(self._maxId+1, dtype=dtp)
                elif self._maxId >= arr.shape[0]:
                    new = numpy.zeros(self._maxId+1, dtype=arr.dtype)
                    new[0:arr.shape[0]] = arr
                    self.__dict__[attr] = new

            else:

                # make 2D arrays
                if arr is None:
                    self.__dict__[attr] = \
                        numpy.zeros(shape=(self._maxId+1,wid), dtype=dtp)
                elif self._maxId >= arr.shape[0]:
                    new = numpy.zeros(shape=(self._maxId+1,arr.shape[1]),
                                      dtype=arr.dtype)
                    new[0:arr.shape[0],:] = arr
                    self.__dict__[attr] = new
        
    def setSlicedLabels(self, sliceCoord, axis):
        """
        Extracts ndim-1 dimensional (sub)arrays out of self.labels for each id.

        The dimensions of self.labels are actually kept the same, but all labels
        that do not lie on the ndim-1 dimensional arrays are erased.
 
        The ndim-1 arrays are defined by axis and sliceCoord. Each element of
        sliceCoors contains coordinates of a point for a corrsponding id

        Arguments:
          - sliceCoord: list of coordinates for each segment
          - axis: axis defining the direction of the ndim-1 dimensional planes
        """

        # initialize
        slicedLabels = numpy.zeros(self.labels.shape, dtype=self.labels.dtype)
        sliceObj = [slice(None)] * self.labels.ndim

        # slice each segment (label) separately
        for id in self._ids:

            # make a mask defining the slice
            sliceObj[axis] = slice(sliceCoord[id,axis], sliceCoord[id,axis]+1) 
            mask = numpy.zeros(self.labels.shape, dtype=bool)
            mask[sliceObj] = True

            # intersect the mask with (this id) labels and add to sliceLabels
            slicedLabels += (mask & (self.labels == id)) * id

        # set self.labels
        self.labels = slicedLabels


    ##################################################################
    #
    # Statistical analysis
    #
    ##################################################################
        
    def calculate(self, data=None, labels=None, ids=None, centers=None):
        """
        Makes basic statistics of self.data for each segment defined in
        self.labels whose id is specified in ids (or self._ids if ids=None).

        The statistics done are: mean, std, min, max, position of min (minPos)
        and position of max (maxPos). They are saved in self.mean, self.std,
        self.min, self.max, self.minPos and self.maxPos attributes, respectivly.

        If ids (as given here or in the constructor) is an int, the result
        attributes are just numbers. This choice is appropriate only for
        1d data (such as an array of mean values).

        Otherwise, if ids is not an int (presumably an array or other
        iterrable), the result attributes are stored in 1d ndarrays indexed
        by ids. In this case, the same statistical analysis is done (on the
        same data but) on all segments together and the results are stored
        in 0-elements of the result arrays. If ids is [], all statistics
        attributes (self.mean, self.std, ...) are set to None.

        The values calculate here (mean, std, ...) are entered to the result
        arrays (self.mean, self.std, ...) at the positions given by ids. The
        values at other positions of the result arrays are kept unchanged, or
        set to zero if they were not calculated. The zero position of these
        arrays contains the results obtained by taking all segments (given by
        the argument ids) together. Consequently, they are correct for the
        current ids, but may not be consistent with all non-zero elements of
        the result arrays.
        
        Based on scipy.ndimage object measurements functions, self.data and
        self.labels are directly passed to those functions. Argument ids is the
        same, except that ids=None here means do the calculations for
        each segment (defined by self._ids) seprarately.

        If centers is given, the vectors defined by minimum and maximum
        positions of each segment are calculated in respect to centers, in
        cartesian (minVec and maxVec) and spherical (phi, theta, for 2D and
        3D only) coordinates.

        Arguments:
          - data: data, if None self.data is used
          - labels: labels (segments), if None self.labels is used.
          - ids: array of ids for which the statistics are calculated. It can
          be a single int, but only for 1d data. If none self._ids is used
          - centers: list of segment centers (indexed by ids), used for
          the determination of min/mac vectors and directions in spherical
          coordinates. It has to contain elements correspondig to all ids, but
          it can additionally contain elements for other ids.
        """

        # parse arguments
        self.setData(data=data, labels=labels, ids=ids)

        # do nothing if self._ids is empty
        if len(self._ids) == 0: return

        # treat single integer ids and array ids cases separately
        if self._numberId:
            self._calculateSingleId()
        else:
            self._calculateArrayId()
            if centers is not None: self._extremaVec(centers=centers)

    def _calculateArrayId(self):
        """
        Calculates statistics (see methods calculate) when ids are array, even
        if ids have one element only.
        """

        # make sure the result data structures are ok
        self._prepareArrays(
            arrays=('mean','std','min','max','minPos', 'maxPos'),
            widths=(1, 1, 1, 1, self.ndim, self.ndim),
            dtypes=(float, float, float, float, int, int))

        # do calculations for each segment separately
        self.mean[self._ids] = ndimage.mean(
            input=self.data, labels=self.labels, index=self._ids)
        self.std[self._ids] = ndimage.standard_deviation(
            input=self.data, labels=self.labels, index=self._ids)
        extr = ndimage.extrema(input=self.data, labels=self.labels, 
                               index=self._ids)

        # figure out if 0 is the only id
        zero_id = False
        if isinstance(self._ids, (numpy.ndarray, list)):
            zero_id = (self._ids[0] == 0)
        else:
            zero_id = (self._ids == 0)

        # do calculations for all segments together
        if not zero_id:
            from segment import Segment
            seg = Segment()
            locLabels = seg.keep(ids=self._ids, data=self.labels.copy())
            self.mean[0] = ndimage.mean(input=self.data, labels=locLabels,
                                        index=None)
            self.std[0] = ndimage.standard_deviation(
                input=self.data, labels=locLabels, index=None)
            extr0 = ndimage.extrema(input=self.data, labels=locLabels, 
                                    index=None)

        # put extrema for individual labels in data arrays
        (self.min[self._ids], self.max[self._ids]) = (extr[0], extr[1])
        #self.max[self._ids] = -numpy.array(extr_bug[0])
        if (self._ids.size == 1) and not isinstance(extr[2], list):
            self.minPos[self._ids] = [extr[2]]
            self.maxPos[self._ids] = [extr[3]]
            #self.maxPos[self._ids] = [extr_bug[2]]
        else:
            self.minPos[self._ids] = extr[2]
            self.maxPos[self._ids] = extr[3]
            #self.maxPos[self._ids] = extr_bug[3]

        # put total extrema in data arrays at index 0  
        if not zero_id:
            (self.min[0], self.max[0]) = (extr0[0], extr0[1])
            #self.max[0] = -extr0_bug[0]
            if self.ndim == 1:
                self.minPos[0] = extr0[2][0]
                self.maxPos[0] = extr0[3][0]
                #self.maxPos[0] = extr0_bug[2][0]
            else:
                self.minPos[0] = extr0[2]
                self.maxPos[0] = extr0[3]
                #self.maxPos[0] = extr0_bug[2]
        
    def _calculateSingleId(self):
        """
        Calculates statistics (see method calculate) when ids is a single 
        (int) id.

        Does not calculate statistics on the mean.

        """

        # do calculations for all segments
        self.mean = ndimage.mean(input=self.data, labels=self.labels,
                                            index=self._ids)
        self.std = ndimage.standard_deviation(input=self.data,
                                                         labels=self.labels,
                                                         index=self._ids)
        # maximum bug fix
        extr = ndimage.extrema(input=self.data, labels=self.labels, 
                               index=self._ids)
        extr_bug = ndimage.extrema(input=-self.data, labels=self.labels,
                                   index=self._ids)

        # put extrema results in data arrays
        (self.min, self.max) = (extr[0], extr[1])
        self.max = -extr_bug[0]
        self.minPos = extr[2][0]
        #self.maxPos = extr[3][0]
        self.maxPos = extr_bug[2][0]
        
    def _extremaVec(self, centers):
        """
        Calculates the positions of extrema (min, max) in respect to centers.

        Calculations done for self_ids.

        Argument:
          - centers: list of segment centers (indexed by ids). It has to
          contain elements correspondig to all self._ids, but it can also
          contain elements for other ids.

        Sets:
          - minVec, maxVec: position vectors of min and max in cartesian 
          coordinates
          - minPhi, maxPhi: phi angles on the position vectors (2d and 3d only)
          - minTheta, maxTheta: theta angles on the position vectors (2d and 3d
          only)
        """

        # prepend 0 to self._ids to calculate values for all segments together
        ids = numpy.insert(arr=self._ids, obj=0, values=0)
        pi = numpy.pi

        # calculate positions (vectors) of min and max in respect to center
        self._prepareArrays(arrays=('minVec', 'maxVec'),
                            dtypes=(int, int), widths=self.ndim)
        self.minVec[ids] = self.minPos[ids] - centers[ids]
        self.maxVec[ids] = self.maxPos[ids] - centers[ids]

        # calculate directions (spherical coordinates) for 2D and 3D
        if self.data.ndim == 2:

            self._prepareArrays(arrays=('minPhi', 'maxPhi'),
                                dtypes=(float, float), widths=1)
            self.minPhi[ids] = \
                numpy.arctan2(self.minVec[ids,1], self.minVec[ids,0]) * 180 / pi
            self.maxPhi[ids] = \
                numpy.arctan2(self.maxVec[ids,1], self.maxVec[ids,0]) * 180 / pi

        elif self.data.ndim == 3:

            self._prepareArrays(arrays=('minPhi', 'maxPhi', 
                                        'minTheta', 'maxTheta'),
                                dtypes=(float, float, float, float), widths=1)
            self.minPhi[ids] = \
                numpy.arctan2(self.minVec[ids,1], self.minVec[ids,0]) * 180 / pi
            self.maxPhi[ids] = \
                numpy.arctan2(self.maxVec[ids,1], self.maxVec[ids,0]) * 180 / pi
            self.minTheta[ids] = \
                numpy.arccos(self.minVec[ids,2] / self.min[ids]) * 180 / pi
            self.maxTheta[ids] = \
                numpy.arccos(self.maxVec[ids,2] / self.max[ids]) * 180 / pi

    def getValues(self, id):
        """
        """

        if self.calculated[id]:
            return None
        else:
            return (self.mean[id], self.std[id], self.min[id], self.max[id])


        
        

        
        
