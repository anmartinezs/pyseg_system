"""
Contains class Segments for manipulation of segmented image where segments do
not touch each other.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: segment.py 1396 2017-03-15 13:08:51Z vladan $
"""

__version__ = "$Revision: 1396 $"


import warnings
import sys
import logging
import inspect
import itertools
import numpy
import scipy
import scipy.ndimage as ndimage

from struct_el import StructEl
import pyto.util.numpy_plus as numpy_plus
import pyto.util.nested as nested
from labels import Labels

class Segment(Labels):
    """
    Manipulations and analysis of segmented images (label fields).

    All segments are expected to be spatially separated, that is they should
    not touch each other.

    Important instance attributes:
      - data (ndarray): segmented image data. Read-only, use setData() to
      write new data.
      - ids (ndarray): all ids. Read-only, use setIds method to set new ids.
      - offset: offset of data in respect to the inputData used to make
      segments (depreciated)
      - boundaryIds: list of all boundaries (boundary ids) (remove?)
      - freeIds: list of all free region ids
      - _free: dictionary where each key is a list of boundaries
      used to make a free region and a corresponding value is the free
      region id
      - structEl: instance of StructEl, structuring element used for the
      connectivity of segments
      - contactStructEl: SE used to find segement - boundary contacts
      - countContactStructEl: SE used to count contacts
      - fillStructEl: SE used to fill interior af a segment
      - contact: instance describing contacts

    Methods that set the data:
      - setData
      - setOffset (depreciated)
      - setDefaults

    Methods that affect segment ids:
      - setIds
      - extractIds
      - reorder    

    Mathods for basic manipulation of segments:
      - add
      - remove
      - keep

    Methods that change shape of segments:
      - makeFree
      - makeLayersBetween
      - makeLayersFrom
      - makeSurfaces
      - fillSegments
      - markDistance
      - generateNeighborhoods

    Segmentation methods:
      - label
      - connectivity

    Distance between segments:
      - markDistance()
      - distanceToRegion()
      - pairwiseDistance()

    Individual distances between elements of a segment and another region
    (segment):
      - elementDistanceToRegion()
      - elementGeodesicDistanceToRegion()
      - elementDistanceToRim()

    """
    
    ##################################################################
    #
    # Constants
    #
    ##################################################################

    # numerical factor used in _remove method. The current value is used because
    #'remove' mode is 20% faster than the 'keep mode'
    _remove_or_keep_factor = 0.55


    ##################################################################
    #
    # Constructor
    #
    #################################################################

    def __init__(self, data=None, copy=True, ids=None, clean=False):
        """
        Sets data, data-related and id-related attributes.

        If copy is True data is copied so that (the original) data can not be
        changed in this class. Otherwise only a reference is passed saving
        memory, but there is no guarantee that the data would not be changed.

        Data-related attributes: structure elements and _labelMask are derived
        from the shape and the type of data, or self.data.

        The id-related attributes: ids (list containing all ids), maxId,
        nextId and n (number of ids) are determined in the following order:
        from ids, data, or self.data.

        If clean is True, all segments having ids that are not in the argument
        ids are removed.

        Arguments:
          - data: (ndarray) segments (labels)
          - copy: if True makes a copy of data
          - ids: list of ids
          - clean: if True only the segments given in ids are retained in 
        self.data
        """

        # call super
        super(Segment, self).__init__()

        # initialize attributes that are expected to be defined
        self.ids = None
        #self.ndim = None
        #self._labelMask = None
        #self._labelMaskOffset = None

        # structuring elements defaults
        #self.structEl = None
        #self.structElConn = 1
        #self.structElSize = 3
        #self.contactStructEl = None
        #self.countContactStructEl = None
        self.fillStructEl = None

        # initial values of other (not always used) attributes
        self.boundaryIds = []
        self.freeIds = []
        self._free = {}

        # set data and related attributes if data given
        self.data = None
        if data is not None:
            self.setData(data=data, copy=copy, ids=ids, clean=clean)


    ########################################################
    #
    # Data and defaults
    #
    #########################################################
    
    def setData(self, data=None, copy=True, ids=None, clean=True):
        """
        Sets data and the attributes determined from data.

        If copy is True data is copied so that (the original) data can not be
        changed in this class. Otherwise only a reference is passed saving
        memory, but there is no guarantee that the data would not be changed.
        
        The id-related attributes: ids (list containing all ids), maxId,
        nextId and n (number of ids) are determined in the following order:
        from ids, data, or self.data.

        If clean is True, all segments heving ids that are not in the argument
        ids are removed.

        Data should always be set using this method (not by the assignment
        to self.data).

        Arguments:
          - data: (ndarray) segments (labels)
          - copy: flag indication if data is copied to self.data.
          - ids: list of ids
          - clean: flag indicating if only the segments given in ids are
          retained
        """

        # check and set data
        if data is not None:
            if copy:
                self.data = data.copy()
            else:
                self.data = data

        # set attributes that depend on the shape of data
        self.setDefaults()

        # set id related attributes based on self.data
        self.setIds(ids=ids)

        # keep only the segments that are specified in ids
        if clean and ids is not None: self.keep(ids=ids)
                    
    def setDefaults(self, data=None):
        """
        Sets structure elements and _labelMask attributes derived from (the
        shape and the type of) data that haven't been set already to their
        default values. Does not set self.data.

        Data has to be provided as an argument, or self.data has to be set, or
        self.ndim has to be set.

        Arguments:
          - data: (ndarray) segment (labels) array.

        Sets:
          - all structuring element related stuff
        """

        # SE that determines segment connectivity (default connectivity 1)
        self.setStructEl()

        # ToDo: make the same for all other structuring element

        # SE used for contact determination (default connectivity 1)
        #if self.contactStructEl is None:
        #    self.contactStructElConn = 1
        #    self.contactStructEl = \
        #        ndimage.generate_binary_structure(rank=self.ndim, connectivity=1)

        # SE used to count contacts (default connectivity equals self.ndim)
        #if self.countContactStructEl is None:
        #    self.countContactStructElConn = self.ndim
        #    self.countContactStructEl = \
        #        ndimage.generate_binary_structure(rank=self.ndim,
        #                                          connectivity=self.ndim)

        # SE used to fill interior of a segment
        if self.fillStructEl is None:
            self.fillStructElConn = 1
            self.fillStructEl = ndimage.generate_binary_structure(
                rank=self.ndim,  connectivity=1)

        # SE used to find the surface of a segment
        #if self.surfaceStructEl is None:
        #    self.setSurfaceStructEl(connectivity=self.ndim)

    def setSurfaceStructEl(self, connectivity=None):
        """
        Not used anymore.

        Sets the structuring element used for surface determination
        (self.surfaceStructEl) and its connectivity (self.surfaceStructElConn).

        Default connectivity is ndim (of the data), that is 26 neighbors in 3d.
        In this case, a surface is connected in the default self.structEl sense.

        Argument:
          - connectivity: maximum distance squared between the center and the
          elements of the structuring element
        """

        if connectivity is None: connectivity = self.ndim
        self.surfaceStructEl = ndimage.generate_binary_structure(
            rank=self.ndim, connectivity=connectivity)
        self.surfaceStructElConn = connectivity

    def parseInput(self, data=None, ids=None):
        """
        Returns data, ids and update. If data is not given returns self.data. If
        ids is not given tries self.ids (if data was not given), or extracts ids
        from data (or self.data).

        Update is True if self.data is used.

        Arguments:
          - data: (ndarray) containing segments
          - ids: array of segment ids

        Returns data, ids, update
        """

        update = False

        # if data not given use self data
        if data is None:
            update = True
            data = self.data
            if ids is None:
                ids = self.ids

        # extract ids if needed
        if ids is None:
            ids = self.extractIds(data)

        return data, ids, update
        

    ########################################################
    #
    # Ids and other attribute related
    #
    #########################################################
    
    def removeIdFromBound(self, ids):
        """
        Removes ids from boundaryIds, freeIds and _free.
        """

        if isinstance(ids, int): ids = [ids]
        
        # remove id from boundaryIds and freeIds
        for id in ids:
            if self.boundaryIds.count(id) > 0: self.boundaryIds.remove(id)
            if self.freeIds.count(id) > 0: self.freeIds.remove(id)
        
        # remove id from _free
        freeInv = {}
        freeInv.update([(val, key) for (key, val) in self._free.items()])
        for id in ids:
            if freeInv.has_key(id): freeInv.pop(id)
        self._free.clear()
        self._free.update([(val, key) for (key, val) in self._free.items()])

    def setProperties(self, props):
        """
        Sets attributes of this instance from (arg) properties. The attribute
        names are given as keys and attribute valus are determined from values
        of dictionary (arg) props.

        Each of the props.values() has to be an array (or list) with elements
        directly corresponding to elements of self.ids (that is the lengths 
        of these arrays have to be the same as the length of self.ids). The
        attributes are then set to ndarray indexed by ids (that is 
        attr[3] is a value correspoinding to id = 3).

        Only property with name 'contacts' is expected to be an instance
        of Contacts

        Also sets attribute indexed, that contains the names of all indexed
        properties (all properties except 'contacts').

        Argument:
          - props: distionary of properties
        """

        # set indexed
        self.indexed = [key for key in props.keys() if key != 'contacts']

        for name, value in props.items():

            if name == 'contacts':

                # set contacts
                self.contacts = value

            else:

                # set other properties
                if isinstance(value, numpy.ndarray):
                    dtype = value.dtype
                else:
                    dtype = float
                var = numpy.zeros(self.maxId+1, dtype=dtype)
                var[self.ids] = value
                setattr(self, name, var)
                
    def findNonUnique(self):
        """
        Finds segments that are disconnected and segments that do not exist
        (id in self.ids but have no elements) and returns their ids.

        Returns dictionary with the following key value pairs:
          - 'many' : (list) ids of disconnected segments
          - 'empty' : (list) ids of non-existing segments
        """

        from topology import Topology

        # find ids that don't exist in data
        existing_ids = self.extractIds()
        empty = numpy.setdiff1d(self.ids, existing_ids)

        # topology should not have nonexistant ids
        ids = numpy.intersect1d(self.ids, existing_ids)

        # find ids of disconnected segments 
        topo = Topology(segments=self, ids=ids)
        n_connected = topo.getHomologyRank(dim=0)[ids]
        many = topo.ids[numpy.nonzero(n_connected>1)[0]]
        #empty = topo.ids[numpy.nonzero(n_connected<1)[0]]
 
        return {'many' : many, 'empty' : empty}


    ########################################################
    #
    # Simple segment manipulations
    #
    #########################################################
    
    def clean(self, value=0):
        """
        Makes self.data and self.ids consistent by removing segments self.data
        that are not in self.ids, and removing ids from self.ids corresponding
        to non-existing segments.

        Prints logging.INFO message for removed ids. 

        Arguments:
          - value: segments that are removed are replaced by this value

        Sets self.data and self.ids
        """

        # remove segments whose ids are not in self.ids
        self.keep(ids=self.ids, value=value)

        # remove ids of non-existing segments
        existing = self.extractIds()
        empty = numpy.setdiff1d(self.ids, existing)
        self.setIds(existing)

        # print message if ids removed
        if len(empty) > 0:
            empty = numpy.array2string(empty)
            fil, line, meth = logging.Logger(name="").findCaller()
            logging.warning('(%s:%d in %s) Segments with ids: %s do not' \
                            + ' seem to exist. Removed from self.ids.', \
                            self.__module__, line, meth, empty)

    def remove(self, ids, data=None, all=None, value=0, mode='auto'):
        """
        Removes segments labeled by (elements of) ids, by replacing the
        corresponding elements of data by value.

        If data is None, specified segments are removed from self.data. In 
        addition id related and boundary related attributes are updated. If 
        data is given a modified segment ndarray is returned. The original 
        data array may be changed.

        If argument all is None, all ids are determined from data. If all is
        provided it may increase the speed a bit.

        Argument mode determines the method used. If mode is 'remove' the
        segments are removed from the array. If it is 'keep', the segments with
        ids that are not in the argument ids are kept. Finally, the mode 'auto'
        finds the better strategy between the two, based on the numbers  of ids
        that should be removed or kept. See self._remove for details.

        Arguments:
          - ids: ids of segments to be removed (int or list/array)
          - data: segment ndarray
          - all: all ids (determined from data if not given)
          - value: value denoting background (no segments)
          - mode: determines the method used, 'auto' is recommended

        Returns:
          - modified array if data is given
        """

        # set ids
        if isinstance(ids, int): ids = [ids]

        # set data
        update = False
        if data is None:
            data = self.data
            update = True

        # remove the segments
        data = self._remove(data=data, remove=ids, all=all, value=value, 
                            mode=mode)

        # update or return
        if update:
            remain_ids = numpy.setdiff1d(self.ids, ids)
            self.setData(data=data, copy=False, ids=remain_ids, clean=False)
            self.removeIdFromBound(ids)
        else:
            return data

    def keep(self, ids, data=None, all=None, value=0, mode='auto'):
        """
        Keeps only the segments labeled by elements of ids. Elements of other
        segments are replaced by value.

        If data is None, self.data is modified. In addition, the id related and
        boundary related attributes are updated. If data is given a modified
        segment ndarray is returned (may change the original data array).

        If argument all is None, all ids are determined from data. If all is
        provided it may increase the speed a bit.

        Argument mode determines the method used. If mode is 'remove' the
        segments are removed from the array. If it is 'keep', the segments with
        ids that are not in the argument ids are kept. Finally, the mode 'auto'
        finds the better strategy between the two, based on the numbers  of ids
        that should be removed or kept. See self._remove for details.

        Arguments:
          - ids: ids of segments to be removed (int or list/array)
          - data: segment ndarray
          - all: all segment ids (determined if not given)
          - value: value denoting background (no segments)
          - mode: determines the method used, 'auto' is recommended

        Returns:
          - modified array if data is given
        """

        # set ids
        if isinstance(ids, int): ids = [ids]

        # set data and all ids
        update=False
        if data is None:
            data = self.data
            update = True

        # remove the segments that are not to be kept
        data = self._remove(data=data, keep=ids, all=all, value=0, mode=mode)

        # update or return
        if update:
            self.setData(data=data, copy=False, ids=ids, clean=False)
            remove_ids = numpy.setdiff1d(self.ids, ids)
            self.removeIdFromBound(remove_ids)
        else:
            return data

    def add(self, new, shift=None, remove_overlap=False, dtype=None, 
            relabel=False, clean=True):
        """
        Adds new segments to the existing data.

        Ids of new segments are increased by shift (or self.maxId if shift is 
        None) so they do not collide with the existing segemnts self.data. All
        existing segments of self.data whose ids are not in self.ids are 
        removed, to avoid possible confusion with the new segments. If clean 
        is true and new is an instance of this class all segments from new that
        are not in new.ids are also removed. Finally, id-related attributes 
        (maxId, nextId, ids and n) of this instance are updated.

        If an id from the added segments do is above the limit of the self.data
        data type an exception is raised.

        If the new and old labels overlap, the new labels are used. If this
        happens, the the old labels may become disconnected.

        If remove_overlap is True, segments from self.data that overlap (have
        at least one element in common with the new segments) are 
        completely removed.

        If relabel is True, the existing and the new segments are put together
        and relabeled (renumbered) from scratch. In this case all segments 
        are properly distinguished, but their ids are most likely changed.

        Data in new are copied to self.data, so new is not modified, nor it is
        referenced by this instance.
        
        Argument:
          - new: instance of Segment, or array whose positive elements define
          new segments
          - shift: new segments ids are increased by this amount, default
          self.maxId
          - remove_overlap: if True, segments from self.data that overlap (have
          at least one element in common with the new segments) are removed
          - dtype: current and new data arrays are converted to this dtype
          - relabel: flag indicating what to do when new and old segments
          overlap
          - clean: if True segments from new.data that are not listed in new.ids
          are removed
        """

        # extract new data and new ids and keep only segments with new ids
        if isinstance(new, Segment):
            try:
                if new.ids is not None:
                    new_ids = numpy.array(new.ids)
                if clean: new.keep(ids=new.ids)
            except AttributeError:
                new_ids = None
            new_data = new.data
        elif isinstance(new, numpy.ndarray):
            new_data = new
            new_ids = None
        else:
            raise TypeError("Argument new has to be an instance of ",
                            + "Segment or numpy.ndarray.")

        # data did not exist before
        if self.data is None:
            if shift is None: shift = 0
            new_data = numpy.where(new_data>0, new_data+shift, 0)
            if new_ids is None:
                new_ids = self.extractIds(data=new_data)
            new_ids += shift
            self.setData(data=new_data, copy=True, ids=new_ids, clean=clean)
            return

        # remove all self.data segments that are not in self.ids
        self.keep(ids=self.ids)

        # remove segments from self.data that overlap with new segments
        if remove_overlap:
            overlap = numpy.where((self.data>0) & (new_data>0), self.data, 0)
            overlap_ids = numpy.unique(overlap)
            self.remove(ids=overlap_ids)

            # log overlapping segments
            if len(overlap_ids) > 0:
                if logging.getLogger().isEnabledFor(logging.INFO):
                    new_overlap_ids = numpy.unique(new_data[overlap>0])
                    logging.info(
                        ("Segments %s were removed because they overlap with" 
                         + " segments %s") % (overlap_ids, new_overlap_ids)) 
            
        # combine current and new data
        if relabel: 

            # relabel everything
            data =  new_data + self.data
            data, nSeg = StructEl.label(input=(data>0), structure=self.structEl)
            self.setData(data=data, copy=False)
            
        else:

            # set shift and new_ids
            if shift is None:
                shift = self.maxId
            if new_ids is None:
                new_ids = self.extractIds(data=new_data)

            # combine ids
            ids = numpy.union1d(self.ids, new_ids+shift)

            # change dypes of data arrays
            if dtype is not None:
                self.data = numpy.asarray(self.data, dtype=dtype)
                new_data = numpy.asarray(new_data, dtype=dtype)

            # complain if max id is higher than can fit in the requested dtype 
            try:
                data_max = numpy.iinfo(self.data.dtype).max
                new_data_max = numpy.iinfo(new_data.dtype).max
                if (ids.max() > data_max) or (ids.max() > new_data_max):
                    if data_max < new_data_max:
                        dtype_ = self.data.dtype
                    else:
                        dtype_ = new_data.dtype
                    raise TypeError(
                      ("Maximum new id: %d is higher that the maximum for the " 
                       + "data type (%s) which is %d. Please set argument dtype"
                       + " to an appropriate dtype.") \
                        % (ids.max(), dtype_, min(data_max, new_data_max)))
            except AttributeError:
                pass
                #logging.warning('Could not figure out if dtype ' + dtype_.name 
                #      + ' can fit number ' + str(ids.max()) + '. Proceeding' )

            # add shifted data 
            data = numpy.where(new_data>0, new_data+shift, self.data)
            self.setData(data=data, copy=False, ids=ids, clean=False)


    ###########################################################
    #
    # Reordering segments and ids
    #
    ##########################################################

    def reorder(self, order=None, data=None, clean=False):
        """
        Changes ids of segments according to the order, or orderes ids from
        1 up without gaps if order is not given.

        Returns (dictionary) order if argument order is None. 

        If data is None segments in self.data are reordered and nothing is
        returned. Otherwise, segments in data are reordered and the resulting
        array is returned (data is not modified).

        If clean is True, the segments whose ids are not in order.keys() are
        removed. This option does have any effect if order is None.

        Arguments:
          - data: labels array 
          - order: dictionary with old (keys) and new ids (values), where keys
          and values have to be nonnegative integers 
          - clean: flag indicating if ids that are not in order.keys
          are removed or kept.

        Sets: self.data, self.ids and related if data is None.

        Returns:
          - (dictionary) order, if data is None and order is None
          - nothing, if data is None and order is not None
          - reordered_array, order, if data is not None and order is None
          - reordered_array, if data is not None and order is not None
        """

        res = super(Segment, self).reorder(order=order, data=data, clean=clean)

        # update id-lists
        if data is None:

            # find new order
            if order is None:
                new_order = res
            else:
                new_order = order

            # need to check the following
            if self.boundaryIds is not None:
                self.boundaryIds = [new_order[old] for old in self.boundaryIds]
            if self.freeIds is not None:
                self.freeIds = [ new_order[old] for old in self.freeIds ]
            if self._free is not None:
                self._free = [(key, new_order[self.free[key]]) 
                              for key in self._free ]
                self._free = dict(self._free)

        return res

    def shiftIds(self, shift, data=None):
        """
        Shifts all (positive) ids of segments by (arg) shift.

        If data is None segments in self.data are reordered, self.ids is
        adjusted and nothing is returned. Otherwise, segments in data are 
        reordered and the resulting array is returned (data is modified).

        Arguments:
          - shift: value by which ids are increased
          - data: labels array 

        Sets: self.data, self.ids and related if data is None.

        Returns:
          - nothing, if data is None and order is not None
          - reordered array, if data is not None and order is not None
        """

        res = super(Segment, self).shiftIds(shift=shift, data=data)

        # update id-lists
        if data is None:
            
            # (5.09.08) not sure if needed
            if (self.boundaryIds is not None) and (len(self.boundaryIds) > 0):
                self.boundaryIds = self.boundaryIds + shift
            if (self.freeIds is not None) and (len(self.freeIds) > 0):
                self.freeIds = self.freeIds + shift
            if (self._free is not None) and (len(self._free) > 0):
                self._free = self._free + shift

        return res


    ###########################################################
    #
    # Manipulations that generate new or change shape of existing segments
    #
    ##########################################################

    def makeFree(self, ids, size, mask=None, mode='intersect',
                 update=False):
        """
        Makes a free region between segments of this instance. 

        The free region can be formed where mask > 0 only. The main idea behind
        mask is to set it so that the existing boundaries are off-limits to
        the formation of free regions. If mask is a Label it's positioning
        (inset) is adjusted and it may be expanded to match self.data. 
        If mask is a ndarray it is applied to self.data directly. Finally, 
        if mask is an int, or a list of ints the free region is formed 
        where self.data equals any of the values of mask.

        If mode is 'add' and size is an int, finds all elements of self.data
        whose Euclidean distance to a nearest boundary defined by ids is
        smaller or equal to size (the fastest variant, calls
        ndimage.distance_transform_edt once). If size is an array,
        all elements of self.data whose Euclidian distance to any of the
        boundaries is smaller or equal to the corresponding size (calls
        ndimage.distance_transform_edt once for each id). In both cases
        the selected elements of self.data form a free region.

        If mode is 'intersect', the free region is obtained by the intersection
        of regions (of self.data) surrounding each boundary (given by ids).
        Each element of these surrounding regions can have at most size (for
        the corresponding id) Euclidean distance to the boundary (calls
        ndimage.distance_transform_edt once for each id). 

        If size is a single number <= 0, the free region is formed based on
        mask only.

        If update is True, adds the free region to self.data, and puts the id
        of the free region to self.freeIds, make a corresponding entry in
        self._free and returns the id or the free region. If update is false
        returns a Labels object that contains the free region, and has the same
        positioning attributes (offset, inset) as this instance.

        Note: does not make an entry in self._free, because not sure if 
        that's needed for anything.

        Arguments:
          - ids: list of boundaryIds, or a single int
          - size: list (or ndarray) of maximum distances to each boundary. If
          single number then it's used for all boundaries. If it is <= 0 free
          region is obtained from mask only.
          - mask: (ndarray, int, list or Label) Positive elements of mask 
          define a region where a free region is allowed to be created.
          Alternatively, if mask is an int (list of ints), then the free
          region can be formed where elements of self.data equal (any of the
          ids from) the mask.
          - mode: 'intersect' or 'add' determines how the new region is
          formed
          - update: flag that determines if a free region is added to this
          instance.

        Returns: Labels object whose data attribute is a ndarray (of the same 
        shape and position as self.data) defining the free region.
        """

        # make mask array if the argument mask is not an array
        if mask is None:
            mask = numpy.ones(shape=self.data.shape, dtype='bool')
        elif isinstance(mask, int):
            maskId = mask
            mask = (self.data == maskId)
        elif isinstance(mask, list):
            mask_ids = mask
            mask = numpy.zeros(shape=self.data.shape, dtype='bool')
            for id_ in mask_ids:
                mask = mask | (self.data == id_)
        elif isinstance(mask, Labels):
            mask.useInset(
                inset=self.inset, mode='absolute', expand=True, value=0)
            mask = mask.data

        # set ids
        if ids is None:
            ids = self.ids
        elif isinstance(ids, int): 
            ids = [ids]

        if size <= 0:

            # size <= 0, free determined by mask
            free = (mask > 0)
                
        else:

            # make a list of (id, size) pairs
            if isinstance(size, list) or isinstance(size, numpy.ndarray):
                size_list = zip(ids, size) 
            else:
                size_list = [[id, size] for id in ids] 

            # make free region according to the requested mode
            if mode == 'intersect':

                # find a neighboring region for each boundary and intersect them
                free = numpy.ones(self.data.shape, dtype='bool') & mask
                for (id, size) in size_list:
                    inv_data = numpy.where(self.data==id, 0, 1)
                    if (inv_data > 0).all():  # workaround for scipy bug 1089
                        raise ValueError("Can't calculate distance_function ",
                                         "(no background)")
                    else:
                        distance = ndimage.distance_transform_edt(inv_data)
                    free = free & (distance <= size)

            elif mode == 'add':

                # calculate distances to the nearest boundary and keep 
                # those < size
                if isinstance(size, list) or isinstance(size, numpy.ndarray):

                    # different sizes
                    free = numpy.ones(self.data.shape, dtype='bool') 
                    for (id, size) in size_list:
                        pos = (self.keep(ids=id, data=self.data.copy()) > 0)
                        inv_data = numpy.where(pos, 0, 1)
                        if (inv_data > 0).all(): # workaround for scipy bug 1089
                            raise ValueError(
                                "Can't calculate distance_function ",
                                "(no background)")
                        else:
                            distance = ndimage.distance_transform_edt(inv_data)
                        free = free | (distance <= size)
                    free = free & mask

                else:

                    # same size for all
                    pos = self.keep(ids=ids, data=self.data.copy()) > 0
                    inv_data = numpy.where(pos, 0, 1)
                    if (inv_data > 0).all():  # workaround for scipy bug 1089
                        raise ValueError("Can't calculate distance_function ",
                                         "(no background)")
                    else:
                        distance = ndimage.distance_transform_edt(inv_data)
                    free = (distance <= size) & mask

            else:
                raise ValueError, "Mode " + mode + " is not valid. Allowed " \
                  + " modes are intersect and add."

        # manage output
        if update:
            
            # update self.data, bounded table and other attributes
            self.add(new=free, relabel=False)
            newId = self.maxId
            self.freeIds.append(newId)
            #self._free.update({tuple(boundaryIds): newId}) 
            return newId

        else: 
            
            # wrap free in a Labels object and keep the positioning
            free = Labels(free)
            free.copyPositioning(self, saveFull=True)
            return free

    def makeLayersBetween(
        self, bound_1, bound_2, mask, nLayers=None, width=None, offset=0.5,
        between='median', maxDistance=None, fill=True,
        nExtraLayers=0, extra_1=None, extra_2=None):
        """
        Makes layers between boundaries labeled by bound_1 and bound_2 and 
        calculates the width between boundaries.

        The layers are made on a region specified by mask (layers region). If 
        mask is an int (or a list of int's) it denotes id(s) of segment(s) of 
        this instance that make a mask. Alternatively, if it is a ndarray 
        positive elements definem the region where layers are made.

        If arg maxDistance is specified, the mask is adjusted so that the
        elements that have sum of distances to both boundaries larger than
        maxDistance are discarded.

        If fill is True, holes in the mask that were created by adjusting 
        the maxDistance (see paragraph above) are filled. In order to detect
        holes both layer region and boundaries are taken into account, and
        the default structuring element (see scipy.ndimage.binary_fill_holes)
        is used. 

        If the arument width is None, first the distance between boundaries is 
        calculated as the mean/min/max/median (see arg between) value of the 
        distance from each element of boundary 2 that touches the adjusted mask 
        region to boundary 1. Then the width is calculated as the distance 
        between boundaries minus 1. Note: Cleft.getWidth() uses a bit nicer way
        to calculate this distance.

        Array elements are assigned to layers based on their eucliden distances
        to (the closest element of) both segments bound_1 and bound_2. In total 
        nLayers are made between the boundaries. If nLayers is not given it is 
        set to the rounded (int) value of the width.

        In case nExtraLayers is 0, layers are numbered from 1 (closest to
        bound_1) to nLayers (closest to bound_2). Numerical parameter offset
        determines the distribution of layers. It should be larger than 0 and
        smaller than 1. The bigger this parameter the more elements are
        included in the outside layers. The formula used to calculate the
        layer position is:

        layer_no = floor{(d_1-offset) * nLayers / (d_1+d_2-2*offset) + 1}

        where d_1 and d_2 are the shortest distances to segments bound_1 and 
        bound_2. 

        In case nExtraLayers is > 0, extra layers are formed in addition to the
        layers between the boundaries (explained above). The extra layers are 
        formed on the boundaries and the extra regions (args extra_1 and 
        extra_2). The extra layers are formed based on their euclidean distance 
        to the closest 'between' layer, and they have the same thickness as the 
        'between' layers. This is done using self.makeLayersFrom() method. 
        The additional layers over the first boundary and the first extra 
        region are numbered from 1 to nExtraLayers, the ids of the 'between' 
        layers are shifted by nExtraLayers and the layers over the second 
        boundary and the second extra region go from nLayers_nExtraLayers+1 
        to nLayers+2*nExtraLayers.

        Arguments:
          - bound_1, bound_2: (int, list or ndarray) id(s) of boundaries
          list of ints
          - mask: (ndarray, int or a list of int's) region where layers are
          formed
          - nLayers: number of layers between the boundaries
          - width: distance between boundaries
          - between: denotes how the distance between segments is calculated. It
          can 'mean', 'min', 'max', median', the same as the mode argument in
          distanceToRegion() method.
          - nExtraLayers: number of additional layers formed over each boundary.
          - extra_1, extra_2: (int, list or ndarray of int's) ids of extra 
          regions 1 and 2
          - maxDistance: highest allowed sum of distances to segments bound_1
          and bound_2
          - fill: flag indicating if holes created by maxDistance procedure
          are filled (used only if maxDistance is not None)
          - offset: numerical parameter that compensates for the discrete 
          positions in an array and so determins the exact position of layers,
          has to be 0 - 1, default 0.5 makes the positions symmetrical

        Returns layer_obj, dist_between:
          - layer_obj: instance of this class containing layers. The data array
          of the this instance has the smallest shape that contains all layers
          and boundaries (to speed up calculations and decrease memory usage), 
          and the inset attribute is set to define the position of the returned 
          data.
          - width: width of the region between the boundaries calculated from
          edge to edge (for example, if the bounaries touch each other the
          width is 0)
        """

        # constants
        id_1 = 1
        id_2 = 2
        e_id_1 = 11
        e_id_2 = 12
        id_seg = 3
        id_tmp = 6

        # prepare for relabeling boundary ids
        if isinstance(bound_1, (list, numpy.ndarray)):
            order = dict([[b_id, id_1] for b_id in bound_1])
        else:
            order = {bound_1:id_1}
        if isinstance(bound_2, (list, numpy.ndarray)):
            order.update(dict([[b_id, id_2] for b_id in bound_2]))
        else:
            order.update({bound_2:id_2})
            
        # prepare for relabeling extra ids
        if extra_1 is not None:
            if isinstance(extra_1, (list, numpy.ndarray)):
                order.update(dict([[e_id, e_id_1] for e_id in extra_1]))
            else:
                order.update({extra_1:e_id_1})
        # don't understand what the idea here was (r 550)
        #if (extra_2 is not None) and (extra_2 != extra_1):
        if (extra_2 is not None):
            if isinstance(extra_2, (list, numpy.ndarray)):
                order.update(dict([[e_id, e_id_2] for e_id in extra_2]))
            else:
                order.update({extra_2:e_id_2})
            
        # relabel boundaries with id_1 and id_2, mask by id_seg and extra
        # regions by e_id_1 and e_id_2
        if isinstance(mask, numpy.ndarray):

            # mask is an ndarray
            label = self.reorder(order=order, data=self.data.copy(), clean=True)
            label[mask>0] = id_seg
            
        else:

            # mask is (list of) int(s)
            if isinstance(mask, list):
                order.update(dict([[m_id, id_seg] for m_id in mask]))
            else:
                order.update({mask:id_seg})
            label = self.reorder(order=order, data=self.data.copy(), clean=True)

        # relabeled self, make inset if needed
        ids = [id_1, id_2, id_seg, e_id_1, e_id_2]
        relabeled = Segment(data=label, copy=False, ids=ids)
        #relabeled.copyPositioning(self, saveFull=True)
        relabeled.copyPositioning(self)
        relabeled.makeInset()

        # adjust layers region for maxDistance and fill args
        if maxDistance is not None:

            # remove layers region elements that are too far from boundaries
            dist_to_1 = relabeled.distanceToRegion(ids=[id_seg], regionId=id_1, 
                                                   mode='all')
            dist_to_2 = relabeled.distanceToRegion(ids=[id_seg], regionId=id_2, 
                                                   mode='all')
            relabeled_adjust = numpy.where(
                (relabeled.data == id_seg) \
                    & (dist_to_1 + dist_to_2 > maxDistance),
                0, 
                relabeled.data)               
            if (relabeled_adjust==id_seg).sum() == 0:
                fil, line, meth = logging.Logger(name="").findCaller()
                logging.warning("(%s:%d) No segmentation region: "\
                    + "boundaries are further apart than maxDistance parameter",
                    self.__module__, line)
                
            # fill holes in the segmentation region
            if fill:
                filled = ndimage.binary_fill_holes(relabeled_adjust)
                relabeled_adjust[filled & (relabeled.data == id_seg) 
                                 & (relabeled_adjust == 0)] = id_seg

            # adjust relabeled instance
            relabeled.data = relabeled_adjust
            relabeled.makeInset()

        # extract elements of boundaries directly touching the segmentation
        # region
        dilated = ndimage.binary_dilation(relabeled.data==id_seg)
        thinned = numpy.where(dilated & (relabeled.data==id_1), id_1, 0)
        thinned[dilated & (relabeled.data==id_2)] = id_2
        thinned = Segment(data=thinned, ids=[id_1, id_2], copy=False)

        # calculate distance between boundaries if not specified
        if width is None:
            dist_all = thinned.distanceToRegion(ids=[id_2], regionId=id_1, 
                                                mode=between)
            width = dist_all[id_2] - 1.

        # find number of layers and thickness, if needed
        if nLayers is None:
            nLayers = int(numpy.rint(width))
        if nExtraLayers > 0:
            thick = float(width) / nLayers

        # calculate distances to the boundaries
        label_1 = self.reorder(data=relabeled.data.copy(), clean=True,
                               order={0:id_tmp, id_1:0, id_2:id_tmp, id_seg:1})
        if (label_1 > 0).all():  # workaround for scipy bug 1089
            raise ValueError("Can't calculate distance_function ",
                             "(no background)")
        else:
            dist_1 = ndimage.distance_transform_edt(label_1)
        label_2 = self.reorder(data=relabeled.data.copy(), clean=True,
                               order={0:id_tmp, id_1:id_tmp, id_2:0, id_seg:1})
        if (label_2 > 0).all():  # workaround for scipy bug 1089
            raise ValueError("Can't calculate distance_function ",
                             "(no background)")
        else:
            dist_2 = ndimage.distance_transform_edt(label_2) 

        # convert to layers (see about rounding)
        dist_denom = (dist_1 + dist_2 - 2*offset)
        dist_denom[(dist_denom==0) & (relabeled.data!=id_seg)] = -1
        layers = nLayers * (dist_1 - offset) / dist_denom
        layers = numpy.where(relabeled.data==id_seg,
                             layers.astype('int') + 1, 0)

        # remove those that are very far
        #if maxDistance is not None:
        #    layers[dist_1 + dist_2 > maxDistance] = 0

        # convert layers to an instance of this class
        lay_obj = Segment(data=layers, ids=range(1,nLayers+1), copy=False)
        
        # add extra layers
        if nExtraLayers > 0:

            # relabel layers between
            lay_obj.data[lay_obj.data>0] += nExtraLayers
            between_ids = range(1+nExtraLayers, 1+nLayers+nExtraLayers)
                      
            # layers on boundary_1
            mask_1 = (relabeled.data == id_1) | (relabeled.data == e_id_1)
            e_lay_1 = lay_obj.makeLayersFrom(bound=between_ids, thick=thick, 
                                             nLayers=nExtraLayers, mask=mask_1)
            e_lay_1.usePositioning(image=lay_obj, intersect=False, expand=True)
            lay_obj.data = numpy.where(e_lay_1.data>0,
                                       nExtraLayers - e_lay_1.data + 1, 
                                       lay_obj.data)
                      
            # layers on boundary_2
            # don't understand what the idea here was (r 550)
            #if extra_2 == extra_1:
            #    mask_2 = (relabeled.data == id_2) | (relabeled.data == e_id_1)
            #else:
            mask_2 = (relabeled.data == id_2) | (relabeled.data == e_id_2)
            e_lay_2 = lay_obj.makeLayersFrom(bound=between_ids, thick=thick, 
                                             nLayers=nExtraLayers, mask=mask_2)
            e_lay_2.usePositioning(image=lay_obj, intersect=False, expand=True)
            lay_obj.data = numpy.where(e_lay_2.data>0,
                                       e_lay_2.data + nExtraLayers + nLayers, 
                                       lay_obj.data)
            
            # set ids
            lay_obj.setIds(ids=range(1, nLayers + 2*nExtraLayers + 1))

        # lay_obj positioning 
        #lay_obj.copyPositioning(image=relabeled, saveFull=True)
        lay_obj.copyPositioning(image=relabeled)

        return lay_obj, width

    def makeLayersFrom(self, bound, thick, nLayers, mask=0, offset=0.5,
                       nExtraLayers=0, extra=None):
        """
        Makes layers starting from a boundary (one or more segments specified 
        by argument bound) on a region specified by arg mask and returns them 
        as an object of this class.

        Array elements are assigned to layers based on their euclidean distance 
        to (the closest element) of segment bound. The thickness of each layer 
        is given by argument thick. In total nLayers are made.
        
        Layers are numbered from 1 (closest to bound) to nLayers. Numerical
        parameter offset determines the distribution of layers. It should
        be larger than 0 and smaller than 1. The bigger this parameter the more
        elements are included in the first layer. The formula used to calculate
        the layer position is:

        layer_no = rint{(d - offset) / thick + offset} + 1

        where d is the shortest distance to segment bound. For example, if 
        thick = 1 the layer assignment for different distances are:

                       layer 1  layer 2  layer 3  ...
          offset=0       1-2      2-3      3-4
          offset=0.5   0.5-1.5  1.5-2.5  2.5-3.5

        The layers are made on a region specified by mask. If mask is an int (or
        a list of int's) it denotes id(s) of segment(s) of this instance that 
        make a mask. Alternatively, if it is a ndarray positive elements define 
        the region where layers are made.

        In case nExtraLayers is > 0, extra layers are formed on the boundary 
        and the extra region (arg extra). The extra layers are formed based on 
        their euclidean distance to the closest 'normal' layer, with the same
        thickness. The extra layers are numbered from 1 to nExtraLayers, while
        the ids of the 'normal' layers are shifted by nExtraLayers.

        Arguments:
          - bound: (int, list or ndarray) segment id(s) or the boundary used 
          to start making layers
          - thick: thickness of each layer
          - mask: (ndarray, int or a list of int's) region where layers are
          formed
          - nLayers: number of layers
          - nExtraLayers: number of additional layers
          - extra_1: (int, list or ndarray of int's) id(s) of the extra region
          - offset: numerical parameter that compensates for the discrete 
          positions in an array and so determins the exact position of layers,
          has to be 0 - 1, default 0.5 makes the positions symmetrical

        Returns (Segment) instance containing layers. The data array of the 
        returned instance might have smaller shape than self.data (reduced to
        use less memory), but the inset attribute is set to define the position
        of the returned data. 
        """

        # constants
        id_1 = 1
        e_id_1 = 11
        id_seg = 3
        id_tmp = 6
        
        # prepare for relabeling boundary
        if isinstance(bound, (list, numpy.ndarray)):
            order = dict([[b_id, id_1] for b_id in bound])
        else:
            order = {bound:id_1}

        # prepare for relabeling extra ids
        if extra is not None:
            if isinstance(extra, (list, numpy.ndarray)):
                order.update(dict([[e_id, e_id_1] for e_id in extra]))
            else:
                order.update({extra:e_id_1})

        # relabel boundary with id_1, the mask by id_seg and extra region 
        # by e_id_1
        if isinstance(mask, numpy.ndarray):

            # mask is an ndarray
            label = self.reorder(order=order, data=self.data.copy(), 
                                 clean=True)
            label[mask>0] = id_seg
            
        else:

            # mask is (list of) int(s)
            if isinstance(mask, list):
                order.update(dict([[m_id, id_seg] for m_id in mask]))
            else:
                order.update({mask:id_seg})
            label = self.reorder(order=order, data=self.data.copy(), clean=True)

        # relabeled self, make inset if needed
        relabeled = Segment(data=label, copy=False, ids=[id_1, id_seg, e_id_1])
        relabeled.copyPositioning(self, saveFull=True)
        relabeled.makeInset()

        # calculate distances to the boundary
        label = self.reorder(data=relabeled.data.copy(), 
                             order={0:6, id_1:0, id_seg:1}, clean=True)
        if (label > 0).all():  # workaround for scipy bug 1089
            raise ValueError("Can't calculate distance_function ",
                             "(no background)")
        else:
            dist = ndimage.distance_transform_edt(label) 

        # convert to 'normal' layers (see about rounding)
        n_layer = numpy.rint((dist - offset) / float(thick) + offset)
        #n_layer = numpy.ceil(numpy.rint(dist) / float(thick))
        layers = numpy.where(relabeled.data==id_seg, n_layer.astype('int'), 0)
        layers[layers>nLayers] = 0

        # convert 'normal' layers to an instance of this class
        lay_obj = Segment(data=layers, ids=range(1,nLayers+1), copy=False)
        
        # add extra layers
        if nExtraLayers > 0:

            # relabel layers between
            lay_obj.data[lay_obj.data>0] += nExtraLayers
            normal_ids = range(1+nExtraLayers, 1+nExtraLayers+nLayers) 
            
            # layers on boundary_1
            mask_1 = (relabeled.data == id_1) | (relabeled.data == e_id_1)
            extra_1 = lay_obj.makeLayersFrom(bound=normal_ids, thick=thick, 
                                             nLayers=nExtraLayers, mask=mask_1)
            extra_1.usePositioning(image=lay_obj, intersect=False, expand=True)
            lay_obj.data = numpy.where(extra_1.data>0,
                                       nExtraLayers - extra_1.data + 1, 
                                       lay_obj.data)
                      
            # set ids
            lay_obj.setIds(ids=range(1, nLayers + nExtraLayers + 1))
            
        # lay_obj positioning
        lay_obj.copyPositioning(relabeled, saveFull=True)

        return lay_obj

    def makeSurfaces(self, data=None, size=1, ids=None):
        """
        Creates surfaces of thickness given by size of all segments specified
        by ids.

        Segements are specified by data, or by self.data if data is not given.
        In both cases the segment arrays are modified to contain only the 
        segments with given ids (uses self.keep).

        The surfacess of segments specifed by ids are labeled. If ids is not 
        given, self.ids is used if self.data specifes the segments (data not 
        given). If data is given and ids are not, all the surfaces of all
        segments are determined.

        Surface elements are those whose Euclidian distance to a nearest 
        background element is <= size (size < 1 does not give any surface).
        Consequently, elements that lie on edges of data array are not
        considered surface elements (unless they happen to have a backround
        element for a neighbor. Also, segments should not touch each other.
        Ids of surfaces are the same as the ids of corresponding segments.

        Arguments:
          - size: thickness of surfaces, as defined by the number of erosions
          needed to remove surface from a segment
          - data: array containing segments
          - ids: ids of segments whose surfaces are to be created
 
        Returns:
          - surfaces if data was given
        """
        
        # set data and ids and keep only ids
        if data is None:
            if ids is None: 
                ids = self.ids
            self.keep(ids=ids)
            data = self.data
            extern_data = False
        else:
            if ids is not None:
                data = self.keep(ids=ids, data=data)
            extern_data = True

        # find the surfaces and interior 
        if (data > 0).all():  # workaround for scipy bug 1089
            raise ValueError("Can't calculate distance_function ",
                             "(no background)")
        else:
            dist = ndimage.distance_transform_edt(data)
        surfaces = numpy.where((dist>0) & (dist<=size), data, 0)

        # either update or return the surfaces
        if extern_data:
            return surfaces
        else:
            self.setData(data=surfaces, copy=False, ids=ids, clean=False)

    def labelInteriors(self, data=None, ids=None, surfaces=None, surfaceSize=1):
        """
        Labels segment interiors.

        Segements are specified by data, or by self.data if data is not given.
        In both cases the segment arrays are modified to contain only the 
        segments with given ids (uses self.keep).

        The interiors of segments specifed by ids are labeled. If ids is not 
        given, self.ids is used in case self.data was used (data not given). 
        If data is given and ids is not given interiors of all segments 
        are determined.

        If surfaces are not given uses self.makeSurfaces to label surfaces 
        first. In both case the interiors are determined by subtraction of 
        the surfaces from segments.

        Arguments:
          - data: segments array
          - ids: ids of segments 
          - surfaces: surfaces array
          - surfaceSize: thickness of surfaces (the same as size argument in  
          self.makeSurfaces)

        Returns:
          - surfaces if data was given
        """

        # parse data
        if data is None:
            if ids is None:
                ids = self.ids
            self.keep(ids=ids)
            data = self.data
            extern_data = False
        else:
            if ids is not None:
                data = self.keep(ids=ids, data=data)
            extern_data = True

        # label surfaces if not specified
        if surfaces is None:
            surfaces = self.makeSurfaces(data=data, ids=ids, size=surfaceSize)

        # label interiors 
        interiors = data - surfaces

        # make appropriate output format
        if extern_data:
            return interiors
        else:
            self.setData(data=interiors, copy=False, ids=self.ids, clean=False)
            
    def fillSegments(self, size, ids, update=False):
        """
        Fill interior of segments whose ids are given in ids.

        Dne by closing with self.fillStructEl (connectivity=1). Perhaps more
        precise than rank=connectivity.

        Arguments:
          - size: holes <= size will be filled
          - ids: segments to be filled
          - update: if True updates self.data
        """

        # set ids
        if isinstance(ids, int): ids = [ids]
        
        # do closing for each segment separatelly, in order to avoid closing
        # regions between segments
        filledData = numpy.zeros_like(self.data)
        for id in ids:
            label = (self.data==id)
            iterations = int( numpy.ceil(size / 2.) )
            filled = ndimage.binary_closing(input=label,
                                            structure=self.fillStructEl,
                                            iterations=iterations)
            filledData[filled] = id        

        # update if needed
        if update: self.data = filledData


    ###########################################################
    #
    # Distance between segments
    #
    ##########################################################

    def markDistance(self, size, data=None, ids=None, slices=None):
        """
        Generator that returns distances to individual segments.

        Makes subarrays that extend around the minimal subarrays containing each
        segment (as given by ndimage.find_objects, for example) by size along
        all coordinate axes in both directions. The distances of all non-segment
        array elements to a given segment are returned, together with slices
        defining the subarrays.

        Can be used for iterations:

          for subarray, slice in seg_instance.markDistance(...):
              ...

        If data is not given uses self.data. If ids is not given uses self.ids,
        or extracts all ids from data.

        Uses slices as the minimal segment subarrays. If slices is not given
        they are generated using ndimage.find_objects method.

        Arguments:
          - size: distance by which the segment subarrays are expanded
          - data: ndarray containig segments
          - ids: ids of segment for which the distances are calculated  
          - slices: array of slices containing segments

        Yields distances, extended_slices for each id in ids:
          - distances: size-extended subarray containing the segment
          - extended_slices: slice defining the position of distances
          - ids: segment ids
        """

        # set slices
        data, ids, update = self.parseInput(data=data, ids=ids)
        if slices is None:
            slices = ndimage.find_objects(data)

        # keep only the slices corresponding to ids
        slices = [slices[int(ind)-1] for ind in ids]
        
        # loop over slices
        for (id_, slice_nd) in zip(ids, slices):

            # enlarge the current subvolume, but not over data.shape
            size = int(numpy.rint(size + 0.999))
            try:
                new_slice_nd = \
                  tuple([slice(max(0, sl.start-size), min(len_, sl.stop+size)) \
                         for sl, len_ in zip(slice_nd, data.shape)])
            except TypeError:
                logging.warning('(%s:%u) Segment with id_ = %u does not ' \
                                + 'seem to exist. Skipped.',
                        inspect.stack()[0][1], sys.exc_info()[2].tb_lineno, id_)
                yield None, None, id_
                continue                

            # find distance to the current segment 
            inset = data[new_slice_nd]
            dist_mask = numpy.ones(shape=inset.shape, dtype='int8')
            dist_mask[inset==id_] = 0
            if (dist_mask==0).sum() == 0:
                raise ValueError, "Slice: " + str(slice_nd) + \
                      " does not contain id_ " + str(id_) + "." 
            if (dist_mask > 0).all():  # workaround for scipy bug 1089
                raise ValueError("Can't calculate distance_function ",
                                 "(no background)")
            else:
                distance = ndimage.distance_transform_edt(input=dist_mask)
            #distance[inset>0] = -1

            # correct the new_slice_nd to fit data.shape
            #new_slice_nd, tmp = numpy_plus.trim_slice(slice_nd=new_slice_nd,
            #                                          shape=data.shape)

            yield distance, new_slice_nd, id_

    def distanceToRegion(self, ids=None, region=None, regionId=None,
                         surface=None, mode='center'):
        """
        Calculates distance of all segments specified by ids to a given region.

        If mode is 'min'/'max'/'mean'/'median', the shortest distance between
        each (surface, if arg surface is not None) element of segments and the
        region are calculated first, and then the min/max/mean/median of these
        values is found for each segment separately.

        If mode is 'center', shortest distances between the region and segment
        centers (note that center may lay outside the segment) are calculated.

        The region is specified by (Image or ndarray) region and regionId. If 
        (arg) region is not specifed this instance is used. If regionId is not
        given, the region is defined as all positive elements of region array.

        If surfaces > 0, only the surfaces (of thickness given by arg surface)
        of the segments are considered. Otherwise whole segments are taken into 
        account. In any case the full region is used.

        If ids are not given, distances to all ids are calculated. 

        If the distance to a segment can not be calculated (if the segments does
        not exist, for example) the result for that segment is set to numpy.nan.

        Arguments:
          - ids: segment ids
          - region: (core.Image or ndarray) instance specifying a region
          - regionId: (single int, list, tuple or ndarray) id(s) of the region
          - mode: 'center', 'min', 'max', 'mean' or 'median'
          - surface: thickness of segment surfaces 

        Returns: 1d array of distances to segments indexed by segment ids if 
        regionId is a single int, 2d array of distances indexed by region id 
        and segment id if regionId is a list, tuple or array, or None if there i
        s no ids.

        ToDo: 
          - see about using insets (see Density.calculateNeighborhoods)
        """

        logging.debug("segment.py (distanceToRegion): ids: %s, regionId: %s",
                      str(ids), str(regionId)) 

        # parse ids
        if ids is None:
            ids = self.ids
        else:
            ids = numpy.asarray(ids)

        # return if no ids
        if (ids is None) or (len(ids) == 0):
            return None

        ############ The following should be used instead, but need to fix
        ###########  backcompatibility (return type) first #############

        # use DistanceTo to do the calculations
        #from distance_to import DistanceTo
        #dist = DistanceTo(segments=self, segmentIds=ids)
        #dist.getDistance(region=region, regionId=regionIds, surface=surfaces, 
        #                 mode=mode)
        #return dist

        ############# End of the new way ################################

        # deal with multiple region ids
        if isinstance(regionId, list) or isinstance(regionId, tuple) \
           or isinstance(regionId, numpy.ndarray):
            regionId = numpy.asarray(regionId)
            distances = \
                numpy.zeros(shape=(regionId.max()+1,ids.max()+1), 
                            dtype='float') - 1
            for reg_id in regionId:
                distances[reg_id,:] = self.distanceToRegion(region=region,
                          regionId=reg_id, ids=ids, surface=surface, mode=mode)
            return distances

        # figure out region
        if region is None:
            region = self
        if isinstance(region, numpy.ndarray):
            region = Labels(region)

        # extract surfaces if required
        if (surface is not None) and (surface > 0):
            seg_data = numpy.array(self.data)
            seg_data = self.makeSurfaces(data=seg_data, size=surface, ids=ids)
        else:
            seg_data = self.data
        reg_data = region.data

        # ToDo: make insets for reg_data and seg_data

        # input for distance functions: array where the region is labeled by 0's
        if regionId is None:
            distance_input = numpy.where(reg_data>0, 0, 1)
        else:
            distance_input = numpy.where(reg_data==regionId, 0, 1)

        # calculate distance to all elements outside the region
        if (distance_input > 0).all():  # workaround for scipy bug 1089
            raise ValueError("Can't calculate distance_function ",
                             "(no background)")
        else:
            dist_array = ndimage.distance_transform_edt(distance_input)

        # find distances to segments
        distances = numpy.zeros(shape=(ids.max()+1,), dtype='float') - 1
        if mode == 'all':

            # ndarray containing all distances 
            return dist_array

        elif mode == 'center':

            # distances to the segment centers
            from morphology import Morphology
            mor = Morphology(segments=seg_data, ids=ids)
            centers = mor.getCenter(real=False)
            for id_ in ids:
                distances[id_] = dist_array[tuple(centers[id_])]

        elif mode == 'min':

            # min distance to segments
            for id_ in ids:
                try:
                    distances[id_] = (dist_array[seg_data==id_]).min()
                except ValueError:
                    if (seg_data==id_).sum() == 0:
                        distances[id_] = numpy.nan
                    else:
                        raise

        elif mode == 'max':

            # max distance to segments
            for id_ in ids:
                try:
                    distances[id_] = (dist_array[seg_data==id_]).max()
                except ValueError:
                    if (seg_data==id_).sum() == 0:
                        distances[id_] = numpy.nan
                    else:
                        raise

        elif mode == 'mean':

            # mean distance to segments
            for id_ in ids:
                distances[id_] = (dist_array[seg_data==id_]).mean()

        elif mode == 'median':

            # mean distance to segments
            for id_ in ids:
                distances[id_] = numpy.median(dist_array[seg_data==id_])

        else:
            raise ValueError, "Sorry, mode: " + mode + " is not recognized. " \
                "Mode can be 'center', 'min', 'max', 'mean' or 'median'."

        logging.debug("segment.py (distanceToRegion): end") 

        return distances

    def pairwiseDistance(self, ids=None, mode='min'):
        """
        Calculate pairwise distances between segments with ids.

        Arguments:
          - ids: (list or ndarray) segment ids (default self.ids)
          - mode: currently only 'min' implemented

        Return:
          - ndarray (length n*(n-1)/2, where n = len(ids)) of distances, where 
          the first n-1 elements are distances between ids[0] and each of 
          ids[1:], the next n-2 elements are distances between ids[1] and each
          of ids[2:], and so on.
          - None if ids is None or contain no elements
        """

        # parse input
        if ids is None:
            ids = self.ids

        # copy ids and check if there are >1 ids
        ids = numpy.array(ids)
        if (ids is None) or (len(ids) == 0):
            return None

        # reduce and copy data (not using makeInset to avoid copying whole 
        # segments.data)
        inset = ndimage.find_objects(self.data>0)[0]
        seg = Segment(data=self.data[inset], ids=ids, copy=True, clean=True)

        # find distances
        distance = numpy.array([])
        while(len(ids) > 1):

            # set ids and main_id 
            main_id = seg.ids[0]
            ids = seg.ids[1:]

            # find distances for current main_id
            if mode == 'min':
                dist = seg.distanceToRegion(ids=ids, regionId=main_id, 
                                            mode=mode)
            else:
                raise ValueError, "Currently only 'min' mode is implemented."

            # append current distances
            try:
                distance = numpy.append(distance, dist[ids])
            except NameError:
                distance = dist[ids]

            # remove main_id and adjust inset
            seg.remove(ids=main_id, all=seg.ids, mode='remove')
            seg.makeInset()

        return distance

    def generateNeighborhoods(
        self, regions, ids=None, regionIds=None, size=None, 
        maxDistance=None, distanceMode='min', removeOverlap=False):
        """
        Generates neighborhoods of each specified region on each of the 
        segments.

        Regions are specified by args region and regionIds, and segments by ids.
        A neighborhood of a given region on a segment is defined as a 
        subset of the segment that contains elements that are at most (arg) 
        size/2 away from the closest segment element to the region, as long as 
        the distance to the region is not above (arg) maxDistance.

        The distance between a region and segments is calculated according to 
        the arg distanceMode. First the (min) distance between segments and 
        each point of regions is calculated. Then the min/max/mean/median of 
        the calculated distances, or the (min) distance between the region 
        center and the segments is used.

        If removeOverlap is True, parts of segments that overlap with regions 
        are removed from the calculated neighborhoods.

        Arguments:
          - ids: segment ids
          - regions: (Segment) regions
          - regionIds: region ids
          - size: size of a neighborhood in the direction perpendicular to
          the direction towards the corresponding region (diameter-like)
          - maxDistance: max distance between segments and a given region
          - distanceMode: how a distance between layers and a region is
          calculated (min/max/mean/median)
          - removeOverlap: if True a neighbor can not contain a part of a region

        Yields:
          - region id: id of current neighborhood
          - neighborhoods: (Segment) neighborhood corresponing to region id
          - all neighborhoods: (Segment) all neighborhoods together
        """

        # parse arguments
        if ids is None:
            ids = self.ids
        ids = numpy.asarray(ids)
        if regionIds is None:
            regionIds = regions.ids
        region_ids = numpy.asarray(regionIds)

        # save initial insets and data of segments
        self_inset = self.inset
        self_data = self.data
        reg_inset = regions.inset
        reg_data = regions.data

        # make a working copy of an inset of this instance and clean it
        self.makeInset(ids=ids, additional=regions, additionalIds=region_ids)
        seg = Segment(data=self.data, ids=ids, copy=True, clean=True)
        seg.inset = self.inset

        # revert self to initial state (won't be used further down)
        self.inset = self_inset
        self.data = self_data

        # save current seg
        seg_inset = seg.inset
        seg_data = seg.data

        # find regions that are not further than maxDistance to segments
        if maxDistance is not None:

            # find min distances from each region to segments 
            regions.useInset(inset=seg.inset, mode='abs')
            dist = regions.distanceToRegion(ids=region_ids, 
                                            region=1*(seg.data>0), 
                                            regionId=1, mode=distanceMode)
            regions.inset = reg_inset
            regions.data = reg_data

            # get ids of close regions 
            region_ids = ((dist <= maxDistance) & (dist >= 0)).nonzero()[0]
            region_ids = region_ids.compress(region_ids>0)

        # make Segment to hold all neighborhoods
        all_hoods = Segment(data=seg.data, ids=ids, copy=False, clean=False)
        all_hoods.inset = seg.inset
        all_hoods.makeInset(ids=ids)
        all_hoods.data = numpy.zeros_like(all_hoods.data)

        # make nighborhoods for each region id
        for reg_id in region_ids:

            # make insets
            seg.makeInset(ids=ids, additional=regions, additionalIds=[reg_id])
            regions.useInset(inset=seg.inset, mode='abs')

            # distances to the current region
            reg_dist_in = numpy.where(regions.data==reg_id, 0, 1)
            if (reg_dist_in > 0).all():  # workaround for scipy bug 1089
                raise ValueError("Can't calculate distance_function ",
                                 "(no background)")
            else:
                reg_dist = ndimage.distance_transform_edt(reg_dist_in)

            # if a region overlaps with a segment remove the overlap and warn
            if removeOverlap and \
                   ((seg.data > 0) & (regions.data == reg_id)).any():
                seg.data = numpy.where(regions.data==reg_id, 0, seg.data)
                logging.warning("Density.calculateNeighbourhood: region " \
                                    + str(reg_id) + " overlap with segments." \
                                    + "Removed the overlap from segments." )

            # make a neighbourhood of current region on each segment
            if size is not None:

                hood_data = numpy.zeros_like(seg.data)
                for seg_id in ids:

                    # find the closest point on the current segment
                    min, max, min_pos, max_pos = \
                        ndimage.extrema(input=reg_dist, labels=seg.data,
                                        index=seg_id)

                    # find inset that contains only the current segment 
                    fine_inset = ndimage.find_objects(seg.data==seg_id)
                    try:
                        fine_inset = fine_inset[0]
                    except IndexError:
                        continue

                    # prepare input distance array for hood 
                    hood_in = numpy.ones(shape=seg.data.shape)
                    hood_in[min_pos] = 0

                    # make hood for this region and segment
                    if (fine_inset > 0).all():  # workaround for scipy bug 1089
                        raise ValueError("Can't calculate distance_function ",
                                         "(no background)")
                    else:
                        hood_dist = ndimage.distance_transform_edt(
                            hood_in[fine_inset])
                    fine_hood_data = hood_data[fine_inset]
                    fine_seg = seg.data[fine_inset]
                    fine_hood_data[(fine_seg == seg_id) \
                                       & (hood_dist <= size/2.)] = seg_id

            else:
                hood_data = seg.data

            # make hood instance
            hood = Segment(data=hood_data, ids=ids, copy=False)
            hood.inset = seg.inset

            # add the current hood to all hoods
            hood.useInset(inset=all_hoods.inset, mode='abs')
            all_hoods.data = numpy.where(hood.data>0, 
                                         hood.data, all_hoods.data)
                
            # yield
            yield reg_id, hood, all_hoods

            # revert to initial insets and data
            seg.inset = seg_inset
            seg.data = seg_data
            regions.inset = reg_inset
            regions.data = reg_data

    ###########################################################
    #
    # Distances from/to elements of segments
    #
    ##########################################################

    def elementDistanceToRegion(self, ids, region, metric='euclidean', 
                                connectivity=1, noDistance=-1):
        """
        Calculates distances between each element of segments (specified by
        arg ids) to a given region (arg region). 

        If arg metric is 'euclidean', the Euclidean distance is calculated.

        If arg metric is 'geodesic', the geodesic distance is calculated, 
        using the structure element defined by arg connectivity. 

        If arg metric is 'euclidean-geodesic', something like the geodesic 
        distance is calculated. Specifically, the distance between two 
        neighboring elements in N dim that share an N-i dim surface is i. The
        distance between two elements that are further apart is obtained by 
        adding the distance between neighbors for the shortest path between 
        the two elements.

        All array elements for which distance was not calculated (because their
        ids are not in arg ids) are set to the value of arg noDistance.

        Arguments:
          - ids: ids of segments for which the distance is calculated
          - region: (ndarray) True, or values >1 define region to which 
          distances are calculated. Has to have the same shape as self.data
          - metric: distance metric, can be 'euclidean', 'geodesic' or
          'euclidean-geodesic'
          - connectivity: connectivity of the structure element (as in
        scipy.ndimage.generate_binary_structure() where rank is self.ndim) 
        for geodesic distance calculation (int). 
          - noDistance: value used for array elements where the distance is
          not calculated

        Returns: (ndarray, shape the same as self.data) distances to the region.
        """

        # convert region to boolean
        if region.dtype == bool:
            loc_region = region
        else:
            loc_region = region > 0

        if metric == 'euclidean':

            # euclidean
            distance = numpy.zeros(shape=self.data.shape, 
                                   dtype=float) + noDistance
            for id_ in ids:
            
                # calculate for all elements and restrict to segments
                distance_in = ~loc_region
                curr_dist = scipy.ndimage.distance_transform_edt(distance_in)
                distance = numpy.where(self.data==id_, curr_dist, distance)

        elif metric == 'geodesic':

            # geodesic
            distance = self.elementGeodesicDistanceToRegion(
                ids=ids, region=region, connectivity=connectivity,
                noDistance=noDistance)
 
        elif metric == 'euclidean-geodesic':

            # Euclidean within geodesic
            connectivity = range(1, self.ndim+1)
            distance = self.elementGeodesicDistanceToRegion(
                ids=ids, region=region, connectivity=connectivity,
                noDistance=noDistance)
 
        else:
            raise ValueError(
                "Argument metric: " + metric + " not understood. Currently "
                + "implemented are 'euclidean', 'geodesic' and "
                + "'euclidean-geodesic'.")
            
        return distance

    def elementGeodesicDistanceToRegion(
        self, ids, region, structure=None, footprint=None, connectivity=None, 
        noDistance=-1):
        """
        Calculates geodesic distances between each element of segments 
        (specified by arg ids) to a given region (arg region). 

        If args structure and footprint are specified, they are used directly
        to calculate distances. The calculations are based on 
        scipy.ndimage.grey_dilation().

        If arg connectivity is an int, the standard geodesic distance for the 
        specified connectivity is calculated. Specifically, the structuring 
        element and the footprint are determined from it using rank that equals
        the dimensionality of self.data (see self.getStructureFootprint()).

        If arg connectivity is a list of ints, the structuring element and the 
        footprint are determined using self.getStructureFootprint(). This is
        useful for calculation Euclidean distance within a geodesic mask.

        All array elements for which distance was not calculated (because their
        ids are not in arg ids) are set to the value of arg noDistance.

        Arguments:
          - ids: ids of segments for which the distance is calculated
          - region: (ndarray) True, or values >1 define region to which 
          distances are calculated. Has to have the same shape as self.data
          - structure: (ndarray of size ndim x ndim) structuring element
          - footprint: (ndarray of size ndim x ndim, type bool) footprint
          of the structuring element
          - connectivity: connectivity of the structure element (as in
        scipy.ndimage.generate_binary_structure() where rank is self.ndim)
          - noDistance: value used for array elements where the distance is
          not calculated

        Returns: (ndarray, shape the same as self.data) distances to the region.
        """

        # deal with arguments
        if (structure is not None) and (footprint is not None):
            pass

        elif ((structure is None) and (footprint is None) and 
            (connectivity is not None)):

            # calculate structure and footprint
            structure, footprint = self.getStructureFootprint(
                connectivity=connectivity)

        else:
            raise ValueError("Either both structure and footprint arguments "
                             + "have to be specified, or just connectivity.") 

        # initialize distances
        # there might be a better way 'cause numpy.infty makes float array
        dist = numpy.zeros(shape=self.data.shape, 
                           dtype=structure.dtype) - numpy.infty
        dist[region] = 0

        # calculate distances for each id
        for id_ in ids:

            # initialize distances for this segment
            dist_id = numpy.zeros(shape=self.data.shape, 
                                  dtype=structure.dtype) - numpy.infty
            dist_id[region] = 0

            # make a mask for the current segment 
            seg = numpy.zeros(shape=self.data.shape, dtype=bool)
            seg[self.data == id_] = True

            # dilate as many times as needed
            for index in range(dist.size):

                # dilate but keep only the current segment
                dil = scipy.ndimage.grey_dilation(
                    dist_id, structure=-structure, footprint=footprint, 
                    mode='constant', cval=-numpy.infty)
                dist_new = numpy.where(seg, dil, dist_id)

                # finish if not changed
                if (dist_new == dist_id).all():
                    break
                else:
                    dist_id = dist_new

            else:
                raise RuntimeError("Dilations don't stop")

            # add to the main dist
            dist[seg] = dist_id[seg]

        # make distance positive and set elements outside segments
        dist = -dist
        dist[dist==numpy.infty] = noDistance

        # set distances at region
        all_seg = numpy.zeros(shape=self.data.shape, dtype=bool)
        for id_ in ids:
            all_seg[self.data == id_] = True
        dist[region & ~all_seg] = noDistance
        dist[region & all_seg] = 0

        # check if distance calculated at all elements
        if (dist[all_seg] == noDistance).any():
            logging.debug("Distance at some elements eas not calculated.")  

        return dist
        
    def getStructureFootprint(self, connectivity):
        """
        Returns special (non-flat) forms of structuring element and footprint.

        The length of each dimesnion is 3, the origin is at position [1,1,...]. 

        If connectivity is a single int, the footprint is just the binary
        structuring element corresponding to self.data.ndim and the 
        specified connectivity. The structure is the same as the footprint
        except that it has int type and the origin is set to 0.

        If connectivity is a list (of one or more ints), the footprint is the 
        binary structuring element corresponding to self.data.ndim and the 
        max of the specified connectivities. Each element of structure has
        value that equals its Euclidean distance to the origin, except that
        those that correspond to False elements of footprint are set to 0.
        The type of connectivity is float.

        Used for the calculation of geodesic distance 
        (self.elementGeodesicDistanceToRegion and related).

        Arguments:
          - connectivity: connectivity

        Returns: (structure, footprint)
        """

        if not isinstance(connectivity, (list, tuple, numpy.ndarray)):

            # single connectivity
            if connectivity == -1:
                connectivity = self.ndim
            footprint = scipy.ndimage.generate_binary_structure(
                self.ndim, connectivity)
            structure = numpy.where(footprint, 1, 0)

        else:

            # multiple connectivities: footprint
            conn_max = numpy.asarray(connectivity).max()
            footprint = scipy.ndimage.generate_binary_structure(
                self.ndim, conn_max)

            # set structuring element as a combination of all 
            structure = numpy.zeros(shape=footprint.shape)
            for conn in numpy.sort(numpy.array(connectivity)):
                se_bin = scipy.ndimage.generate_binary_structure(
                    self.ndim, conn)
                se_curr = numpy.where(se_bin, numpy.sqrt(conn), 0)
                structure = numpy.where(structure==0, se_curr, structure)

        # set 0 at the origin of structure
        ones = tuple([1] * self.ndim)
        structure[ones] = 0

        return structure, footprint

    def elementDistanceToRim(
        self, ids, metric='euclidean', connectivity=1, rimId=0, 
        rimLocation='out', rimConnectivity=1, noDistance=-1):
        """
        For each element of specified segments (arg ids) calculates minimal 
        distance to the bounding rim of the segment the element belongs to. 

        The bounding rim of a segment consists of all elements or rim region
        (id given by arg rimId) that contact the layer (in the sense of the
        structure element specified by arg rimConnectivity) if arg rimLocation 
        is 'out'. If it's 'in', the boundary rim consists of the segment 
        elements that contact the rim region.

        If arg metric is 'euclidean', the Euclidean distance is calculated.

        If arg metric is 'geodesic', the geodesic distance is specified, 
        using the structure element defined by arg connectivity. 

        If arg metric is 'euclidean-geodesic', something like the geodesic 
        distance is calculated (see self.elementDistanceToRegion()). 

        All array elements for which distance was not calculated (because their
        ids are not in arg ids) are set to the value of arg noDistance.

        Uses self.elementDistanceToRegion() for distance calculations.

        Arguments:
          - ids: ids of segments for which the distance to rim is calculated
          - metric: distance metric, can be 'euclidean', 'geodesic' or
          'euclidean-geodesic'
          - connectivity: connectivity of the structure element (as in
        scipy.ndimage.generate_binary_structure() where rank is self.ndim)
        for geodesic distance calculation (int). Not used for euclidean.
          - rimId: id or rim region
          - rimLocation: specifies if the rim is just outside of segments
          ('out') or on the segment boundary ('in')
          - rimConnectivity: (int) connectivity of the structure element
          that defines contact elements between a segment and the rim region.
          - noDistance: value used for array elements where the distance is
          not calculated

        Returns: (ndarray, shape the same as self.data) distances to the region.
        """

        # initialize distance matrix
        if (metric == 'euclidean') or (metric == 'euclidean-geodesic'):
            distance = numpy.zeros(shape=self.data.shape, 
                                   dtype=float) + noDistance
        elif(metric == 'geodesic'):
            distance = numpy.zeros(shape=self.data.shape, 
                                   dtype=int) + noDistance
 
        # set connectivities:
        if rimConnectivity == -1:
            rimConnectivity = self.ndim
        rim_se = scipy.ndimage.generate_binary_structure(self.ndim, 
                                                         rimConnectivity)

        for id_ in ids:

            # form rim
            if rimLocation == 'out':
                mask = (self.data == rimId)
                rim = scipy.ndimage.binary_dilation(self.data==id_, 
                                                    structure=rim_se, mask=mask)
                rim[~mask] = False
            elif rimLocation == 'in':
                mask = (self.data == id_)
                rim = scipy.ndimage.binary_dilation(self.data==rimId, 
                                                    structure=rim_se, mask=mask)
                rim[~mask] = False
            else:
                raise ValueError(
                    "Argument rimLocation: " + rimLocation + " not understood."
                    + " Acceptable values are 'out' and 'in'.")

            # distances to rim
            curr_dist = self.elementDistanceToRegion(
                ids=[id_], region=rim, metric=metric, connectivity=connectivity,
                noDistance=noDistance)
            distance = numpy.where(self.data==id_, curr_dist, distance) 

        return distance

    def distanceFromOrigin(self, origins, metric='euclidean', connectivity=1, 
                           noDistance=-1):
        """
        For each element of specified segments (keys of arg origins) calculates
        distance from the origin of the segment the element belongs to. 

        If arg metric is 'euclidean', the Euclidean distance is calculated.

        If arg metric is 'geodesic', the geodesic distance is specified, 
        using the structure element defined by arg connectivity. 

        If arg metric is 'euclidean-geodesic', something like the geodesic 
        distance is calculated (see self.elementDistanceToRegion()). 

        Uses self.elementDistanceToRegion() for distance calculations.

        All array elements for which distance was not calculated (because their
        ids are not in arg ids) are set to the value of arg noDistance.

        Arguments:
          - origins: dictionary where ids are keys and origin coordinates
          are values
          - metric: distance metric, can be 'euclidean', 'geodesic' or
          'euclidean-geodesic'
          - connectivity: connectivity of the structure element (as in
        scipy.ndimage.generate_binary_structure() where rank is self.ndim)
        for geodesic distance calculation (int). Not used for euclidean.
          - noDistance: value used for array elements where the distance is
          not calculated

        Returns: (ndarray, shape the same as self.data) distances to the 
        respective origins.
        """
        
        # initialize distance array
        if (metric == 'euclidean') or (metric == 'euclidean-geodesic'):
            distance = numpy.zeros(shape=self.data.shape, 
                                   dtype=float) + noDistance
        elif(metric == 'geodesic'):
            distance = numpy.zeros(shape=self.data.shape, 
                                   dtype=int) + noDistance
 
        for id_, coord in origins.items():

            # make origins array
            orig_array = numpy.zeros(shape=self.data.shape, dtype=bool)
            orig_array[tuple(coord)] = True

            # get distances for the current id
            dist = self.elementDistanceToRegion(
                ids=[id_], region=orig_array, metric=metric, 
                connectivity=connectivity, noDistance=noDistance)
            
            # update
            distance = numpy.where(self.data==id_, dist, distance)

        return distance


    ########################################################
    #
    # Input / output
    #
    #########################################################

    @classmethod
    def read(cls, file, ids=None, clean=True, byteOrder=None, dataType=None,
             arrayOrder=None, shape=None, memmap=False):
        """
        Reads segmented image (label filed) from a file.

        If file is in em or mrc format (file extension em or mrc) only the file
        argument is needed. If the file is in the raw data format (file 
        extension raw) all arguments are required.

        If arg memmap is True, instead into a nparray, the data is read to
        a memory map. That means that the complete data is not read into
        the memory, but the required parts are read on demand. This is useful
        when working with large images, but might not always work properly 
        because the memory map is not quite properly a subclass of 
        numpy.ndarray (from Numpy doc).

        Sets attributes:
          - data: (ndarray) image data
          - pixelsize: (float or list of floats) pixel size in nm, 1 if pixel 
          size not known
          - length: (list or ndarray) length in each dimension in nm
          - fileFormat: file format ('em', 'mrc', or 'raw')
          - memmap: from the argument

        Arguments:
          - file: file name
          - ids: list of segment ids
          - clean: if true, only the segments corresponding to ids are kept
          - byteOrder: '<' (little-endian), '>' (big-endian)
          - dataType: any of the numpy types, e.g.: 'int8', 'int16', 'int32',
            'float32', 'float64'
          - arrayOrder: 'C' (z-axis fastest), or 'FORTRAN' (x-axis fastest)
          - shape: (x_dim, y_dim, z_dim)
          - memmap: Flag indicating if the data is read to a memory map,
          instead of reading it into a ndarray

        Returns:
          - instance of Segment
        """

        # call super to read the file 
        image = Labels.read(
            file=file, byteOrder=byteOrder, dataType=dataType,
            arrayOrder=arrayOrder, shape=shape, memmap=memmap)

        # deal with ids
        seg = cls(data=image.data, copy=False, ids=ids, clean=clean)

        # copy attributes
        seg.length = image.length
        seg.pixelsize = image.pixelsize
        seg.memmap = image.memmap

        return seg

   
