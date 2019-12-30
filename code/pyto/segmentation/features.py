"""
Contains class Features that provides basic functionality for the analysis of 
segmented images. 

# Author: Vladan Lucic
# $Id: features.py 1216 2015-08-12 16:40:17Z vladan $
"""

__version__ = "$Revision: 1216 $"


import sys
import logging
import inspect
from copy import copy, deepcopy

import numpy
import scipy
import scipy.ndimage as ndimage

from labels import Labels
from segment import Segment
from hierarchy import Hierarchy

class Features(object):
    """
    Should not be used directly, please use subclasses of this class.

    Each subclass of this class should have self.dataNames set to a list
    of attribute names (strings) that represent the analysis results. Each of
    these attributes has to be a ndarray with first index denoting a segment id.

    Subclassing:
      - If a subclass data is stored neither in the usual (indexed by ids) nor
      in compact form (see DistanceTo, foe example), changeIds() need to be 
      overridden.
      - If some of the data structures are more tha 1-dimensional, or if data
      type is not float, initializeData() needs to be overridden

    ToDo: Make it work for data in 2D ndarray (for example, index at position 0
    should correspond to segment ids).
    """

    #############################################################
    #
    # Initialization
    #
    #############################################################

    def __init__(self, segments=None, ids=None):
        """
        Initializes segments, ids, and few other attributes.

        Arguments:
          - segments: (subclass of Labels, or ndarray) segmented image
          - ids: segment ids

        Note: attribute segments is set to the data array of arg segments, so
        other info (such as inset) is lost. Perhaps should not use. But then
        Morphology is normally instantiated with segments and doesn't care
        about positioning.
        """

        self.dataNames = []
        self.compact = False
        self.segments = None
        self._ids = None

        # used when data is 2+dimensional
        self.regionIds = None

        # set attributes segments and ids
        if segments is not None:
            self.setSegments(segments=segments, ids=ids)
        elif ids is not None:
            self.ids = ids

    def setSegments(self, segments, ids=None):
        """
        Sets attribute segments and ids.

        Attribute segments is set to segments.data (if segments is an istance
        of Segments), or to segments (if segments is a ndarray). Self.ids
        is set to the first found in ids (argument), segments.ids, or to all 
        positive ids present in segments.
        
        Arguments:
          - segments: (subclass of Labels, or ndarray) segmented image
          - ids: segment ids

        Note: attribute segments set to the data array of arg segments, so
        other info (such as inset) is lost. Perhaps should not use.
        """
        
        # read segments (labels) and set ids and related 
        if isinstance(segments, Segment):
            self.segments = segments.data
            if ids is None: 
                self.setIds(segments.ids)
            else: 
                self.setIds(ids)
            #if len(self.ids) > 0:
            #    segments.makeInset()
            if segments.data is not None:
                self.ndim = segments.data.ndim

        elif isinstance(segments, Hierarchy):
            self.segments = None
            self.hierarchy = segments
            if ids is None: 
                self.setIds(segments.ids)
            else: 
                self.setIds(ids)
            if segments.data is not None:
                self.ndim = segments.data.ndim

        elif isinstance(segments, numpy.ndarray):
            self.segments = segments
            self.setIds(ids)
            self.ndim = segments.ndim

        else:
            raise TypeError, "Argument segments is neither an ndarray nor " + \
              "an instance of a Labels subclass (but a " \
              + segments.__class__.__name__ + " )."

    def getIds(self):
        """
        Returns ids
        """
        return self._ids

    def setIds(self, ids=None, segments=None):
        """
        Sets self.ids, If data is internally stored in the compact form, the
        data is changed appropriately. 

        If arg ids is not specified, but arg segments is, ids are extracted
        from segments. This should not be used with the comact data. 

        Ids are saved as ndarray of dtype 'int32' (scipy.ndimage has problems
        with 'int64').

        Arguments:
          - ids: list of ids
          - segments: (Labels) segmented image

        ToDo: perhaps remove arg segments, to avoid confusion with compact data.
        """
        # initialize self.ids if ids=None, or set them to given values
        if ids is None:

            # ids not specified, figure out from segments
            if segments is None:
                all = numpy.unique(self.segments)
                self._ids = all.compress(all>0)
            else:
                self._ids = segments.ids

        else:

            # ids specified, compact data needs to be changed also
            self.changeIds(ids=ids)

        # some of ndimage functions don't accept int64
        # should be ok now (scipy 0.11)
        if len(self._ids) > 0:
            #self._ids = numpy.asarray(self._ids, dtype='int32')
	    self._ids = numpy.asarray(self._ids, dtype='int64')
	    
    ids = property(fget=getIds, fset=setIds, doc="Ids")

    def changeIds(self, ids):
        """
        Sets self.ids to arg ids. In case data is internally stored in the 
        compact form (self.compact=True) the data is modified accordingly.

        If a subclass has data stored in another form, this method should be
        overrided.
        """

        # change data if compact
        if self.compact:
            for name in self.dataNames:
                data = getattr(self, name)
                if data is not None:
                    setattr(self, '_' + name, data[ids])

        # set ids
        self._ids = ids



    def findIds(self, ids=None):
        """
        Returns ids and max id.

        If arguments ids is not specified self.ids is used. If there is no ids
        max id is set to 0.
        """

        # get ids
        if ids is None:
            ids = self.ids

        # convert ids to ndarray and find m,ax id
        if (ids is None) or (len(ids) == 0):
            max_id = 0
        else:
            #ids = numpy.asarray(ids, dtype='int32')
            ids = numpy.asarray(ids, dtype='int64')
            max_id = ids.max()

        return ids, max_id
    

    #############################################################
    #
    # Useful attributes
    #
    #############################################################

    def getNdim(self):
        """
        Gets ndim from self.data, or if the data does not exists returns
        self._ndim.
        """
        try:
            ndim = self.segments.ndim
        except AttributeError:
            ndim = self._ndim
        return ndim

    def setNdim(self, ndim):
        """
        Sets self._ndim.
        """
        self._ndim = ndim
    
    ndim = property(fget=getNdim, fset=setNdim, doc="Dimensionality of data")

    def getMaxId(self):
        """
        Returns maximum id, or 0 if ids don't exist.
        """
        if (self.ids is None) or (len(self.ids) == 0):
            max_id = 0
        else:
            max_id = self.ids.max()
        return  max_id

    maxId = property(fget=getMaxId, doc="Maximum id")


    #############################################################
    #
    # Dealing with data
    #
    #############################################################

    def initializeData(self):
        """
        Sets ids and all data structures to length 0 ndarrays
        """
        self._ids = numpy.array([], dtype=int)

        for name in self.dataNames:
            setattr(self, name, numpy.array([]))

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

        if data is None:

            # reorder all internal data
            for name in new.dataNames:
                var = getattr(self, name, None)
                if var is not None:
                    setattr(self, name, self.reorder(order=order, data=var))

        else:

            # reorderes a given data array according to the first index
            reordered = data.copy()
            reordered[order.values()] = data[order.keys()]
            return reordered

    def merge(self, new, names=None, mode='replace', mode0='replace'):
        """
        Merges data of object new with the data of this instance.
        
        The data attributes whose values are meged are those listed in names
        or in self.dataNames if names is None.

        If mode is 'add' the data is simply added for all ids. If mode is 
        'replace' the new values replace the old ones for id that are in 
        new.ids.

        If new is None, the current object is returned unchanged

        The ids are also merged (and ordered). The mearging works fine even
        if getting and setting data involves needs attribute ids.

        ToDo: make it work for the case data has ndim>1 (perhaps add attribute
        specifying which axis isindexed by ids).

        Arguments:
          - new: instance of Morphology
          - names: (list of strings, or a single string) names of data
          attributes
          - mode: merge mode for data (indices 1+) 'add' or 'replace'
          - mode0: merge mode for index 0, 'add' or 'replace' or a numerical
          value
        """

        # deal with new being None
        if new is None:
            return

        # set data attribute names
        if names is not None:
            if isinstance(names, str):
                names = [names]
        else:
            names = new.dataNames

        # merge ids, take care of self.ids being None or not set at all
        try:
            self_ids = self.ids
        except AttributeError:
            self_ids = None
        if self_ids is None:
            self_ids_old = numpy.array([], dtype=int)
            self_ids_new = numpy.asarray(new.ids)
        else:
            self_ids_old = numpy.asarray(self.ids)
            if new.ids is not None:
                self_ids_new = numpy.union1d(self.ids, new.ids)
        self._ids = self_ids_new

        # merge data
        for name in names:

            # get current variable from new
            new_var = getattr(new, name, None)

            # get current variable from self, but set self.ids to the old
            # value before geting the value because ids might be used to
            # get the variable (the variable might be a property)
            self._ids = self_ids_old
            self_var = getattr(self, name, None)
            self._ids = self_ids_new

            if (self_var is None) or (len(self_var) == 0):

                # no current data, use new data
                setattr(self, name, new_var)
                
            elif (new_var is not None) and (len(new_var) > 0):

                # extend cuurent array
                new_len = new_var.shape[0]
                if new_len > self_var.shape[0]:
                    modified = numpy.zeros(shape=new_var.shape, 
                                           dtype=self_var.dtype)
                    modified[0:self_var.shape[0]] = self_var
                else:
                    modified = self_var

                # merge data
                if mode == 'add':
                    modified[1:new_len] = \
                        modified[1:new_len] + new_var[1:new_len]

                elif mode=='replace':
                    modified[new.ids] = new_var[new.ids]

                else:
                    raise ValueError("Mode: ", mode, " not understood. ",
                              "Currently 'add' and 'replace' are implemented.")

                # merge 0 element
                if isinstance(mode0, str):
                    if mode0 == 'add':
                        modified[0] = modified[0] + new_var[0]
                    elif mode0=='replace':
                        modified[0] = new_var[0]
                else:
                    modified[0] = mode0

                # set attribute
                setattr(self, name, modified)
                    

        # set ndim from new if not set already
        try:
            self.ndim
        except AttributeError:
            try: 
                self.ndim = new.ndim
            except AttributeError:
                pass
        
    def add(self, new):
        """
        Adds data, The same as self.merge but mode='add'. 

        Arguments:
          - new: instance of Morphology
        """
        self.merge(new=new, mode='add')

    def expand(self, axis, names=None):
        """
        Expands dimension of specified attributes (given by arg names or by
        self.dataNames) by adding an axis before arg axis.

        Values of the specified attributes have to be of ndarray type.

        Arguments:
          - axis: data axis
          - names: names of data attributes
        """

        # set attribute names
        if names is None:
            names = self.dataNames

        # expand
        for nam in names:
            value = getattr(self, nam)
            value = numpy.expand_dims(value, axis)
            setattr(self, nam, value)

    def extractOne(self, id_, array_=True):
        """
        Returns a new instance that contains data only for id given by arg id_. 
        If arg array_ is true, each data structure is an ndarray of length 1.
        Otherwise it is a simple number. For example:

          self.prop = numpy.array([0, 3, 5, 6])
          self.extractOne(id_=2, array=True).prop -> numpy.array([2])
          self.extractOne(id_=1, array=False).prop -> 3

        The new instance has self.ids set to [0] and self.dataNames copied from
        self.dataNames

        Returns None if id_ is not in self.ids.

        Argument:
          - id_: id
          - array_: flag indicating if data attributes are arrays or single
          numbers
        """

        # sanity check
        if id_ not in self.ids: 
            return None 

        # new instance
        inst = self.__class__()
        inst.setIds([0])
        inst.dataNames = copy(self.dataNames) 

        # set data
        if array_:
            
            # array form
            for name in self.dataNames:
                data_array = getattr(self, name)
                setattr(inst, name, data_array[[id_]])

        else:
            
            # single number form
            for name in self.dataNames:
                data_array = getattr(self, name)
                setattr(inst, name, data_array[id_])

        return inst

    def nameIds(self, names, array_=False):
        """
        Makes a new attribute from each value of arg names. These new attributes
        are instances of the same class as self. Their data attributes are set
        to the values of the corresponding ids.

        Example:
          self.prop = numpy.array([0, 3, 5, 6])
          names = {1:'one', 2:'two', 3:'three', 4:'four'}
          self.nameIds(name=names)
          self.one.prop = 1
          self.three.prop = 3

        Argument:
          - names: dictionary with group names as values and ids as keys

        Note: not sure if better to reverse ide and values of names 
        """

        for id_, group_name in names.items():
            value = self.extractOne(id_=id_, array_=array_)
            setattr(self, group_name, value) 

    def restrict(self, ids):
        """
        Effectivly removes data that correspond to ids that are not specified 
        in the arg ids. Arg ids should not contain ids that are not in self.ids.

        Actually, only sets self.ids to arg ids.    

        Argument:
          - ids: ids

        Sets:
          - self.ids
        """

        # check
        if len(numpy.setdiff1d(numpy.asarray(ids), self.ids)) > 0:
            raise FeaturesError("Ids specified in the argument have to be a ",
                                "subset of ids given in self.ids.")

        # restrict
        self.setIds(ids=ids)

        # set other elements to default values ?

        # recalculate total


class FeaturesError(ValueError):
    """
    Exception class for Features
    """
    pass
