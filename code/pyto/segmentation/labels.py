"""
Contains class Labels for general manipulations of an image that contains 
integer values (label field).

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import zip
from builtins import range

__version__ = "$Revision$"


import warnings
import sys
import logging
from copy import copy, deepcopy

import numpy
import scipy
import scipy.ndimage as ndimage

from .struct_el import StructEl
import pyto.util.numpy_plus as np_plus
import pyto.util.nested as nested
from pyto.core.image import Image 

class Labels(Image):
    """
    This class concerns an integer-labeled image that contains one or more 
    segments. However, it is not defined how exactly are segments represened 
    in the label image. Therefore, the following subclasses of this
    class should be used for the following cases:

      - Segments: segments are non-overlaping, a segment is represented as 
    a set of image elements having the same value
      - Hierarchy: hierarchical (overlaping) organization of segments

    It is recomended to use objects of the above two subclasses instead
    of instantiating this class.

    The essential attribute of this calss is data, which holds an (n-dim)
    image that takes integer values (label field). In general, 0 labels 
    background and positive integers are for different labels (segments),
    while negative integers are normally not used.

    Important attributes:

      - data (ndarray): labeled n-dim image
      - ids (list): label ids, not necessarily set

    Positioning of the image data array (methods in addition to those from
    core.Image:

    - findInset: returns the smallest inset that contains specified data
    - makeInset: returns the smallest data inset that contains specified data
    and optionally sets self.data to the new data inset

    Example: speeding up calculations without using additional memory

      new_data = self.makeInset(... update=False)   # self.data unchanged

      (calculations using new_data)

    Id and segments-related methods:

    - remove: removes / keeps specified segments
    - restrict(mask): removes elements that are not present in mask
    - reorder: changes segment ids
    - shiftIds: shifts segment ids
    - order*: reorders ids according to different criteria
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
    # Initialization
    #
    ##################################################################
        
    def __init__(self, data=None):
        """
        Initializes attributes and sets data.
        """
        super(Labels, self).__init__(data)

        # initialize attributes
        self.structEl = None


    #############################################################
    #
    # Positioning of the image
    #
    ############################################################

    def makeInset(self, ids=None, extend=0, additional=None, additionalIds=None,
                  expand=True, value=0, update=True):
        """
        Finds the smallest inset that contains all elements of self.data 
        labeled by ids (or self.ids if ids is None). This inset is extended at 
        all sides by (arg) extend (but not outside of the current self.data) 
        and saves it as self.data.

        If arg update is True also sets self.data to the data inset found and
        data.inset to the absolute inset corresponding to the arguments.
        
        If additonal is specified, the smallest inset that contains segments of
        both self.data and additional.data (labeled by additionalIds or 
        additional.ids) is calculated. 

        If some of the additional segments fall outside the current data and if
        expand is True, the data is expanded (a new array is created with 
        the additional elements set at arg value). If expand is False a 
        ValueError is raised.

        If expand is True or no additional segment falls outside the current
        data the data array itself is not changed, only the view of this array 
        (self.data) is changed.

        If there are no segments or no ids, inset in all dimenssions is set to
        slice(0,0). 

        The original inset can be recovered using self.recoverFull().

        Arguments:
          - ids: segment ids (of self.data) 
          - extend: length of extension in all directions
          - additional: (Segment) additional segmented image
          - additionalIds: segment ids of additional
          - expand: flag indicating if data can be extended if needed
          - value: value assigned to the expanded part of data

        Sets (only if update is True):
          - self.data: new data
          - self.inset: inset

        Return: (ndarray) data inset found
        """

        # find inset
        inset = self.findInset(
            ids=ids, extend=extend, additional=additional,
            additionalIds=additionalIds, mode='abs', expand=expand)

        # use this inset
        if inset is None:
            inset = [slice(0,0)] * self.ndim
        new_data = self.useInset(
            inset, mode='abs', expand=expand, update=update)

        return new_data

    def findInset(
            self, ids=None, mode='rel', extend=0, additional=None,
            additionalIds=None, expand=True):
        """
        Returns the smallest inset (list of slice objects of length self.ndim)
        that contains all elements of self.data labeled by ids (or self.ids if
        ids is None).

        If additonal is specified, the smallest inset that contains segments of
        both self.data and additional.data (labeled by additionalIds or 
        additional.ids) is calculated. 

        If arg extend is >0, the inset is extended on all sides by arg extend.
        However, arg expand is False, the inset is not extended outside of 
        what self.inset was when this method was called. Even if arg expand
        is True, the resulting inset is extended to negative array indices. 

        The returned inset is absolute (in respect to 0 array index) if
        arg mode is 'absolute' or 'abs'. The returned inset is relative to 
        the current inset (self.inset) if arg mode is 'relative' or 'rel'.
        Note that in the relative case the resulting inset can have negative
        array indices, but they have to be positive when converted to 
        the absolute mode.

        Returns None if the data (including additional data) do not contain 
        labels specified by ids (or additionalIds). 

        Arguments:
          - ids: segment ids (of self.data), if not specified self.ids has
          to exist (exists generally only in subclasses of this class)
          - mode: 'relative' (same as 'rel') or 'absolute' (same as 'abs')
          - extend: length of extension in all directions
          - additional: (Segment) additional segmented image
          - additionalIds: segment ids of additional
          - expand: flag indicating if the resulting inset is allowed to 
          extend beuond the initial inset (self.inset)

        Returns: inset (list of slices)
        """

        # ids have to be specified if instance of this class
        # not good because it's fine if data contains 0 and >0 elements 
        #if (ids is None) and (type(self) == Labels):
        #    raise ValueError("If this method is called on Labels type, "
        #                     "arg ids has to be specified.")

        # set ids
        if ids is None:
            try:
                ids = self.ids
            except AttributeError:
                ids = None
        if ids is not None:
            if not isinstance(ids, (list, tuple, numpy.ndarray)):
                ids = [ids]
            ids = numpy.asarray(ids)

        # save initial inset
        init_inset = copy(self.inset)

        #  find smallest relative inset that contains all segments of self
        try:

            # no ids or no data
            if (ids is None) or (len(ids) == 0) or (self.data.size == 0):
                raise _LocalException()

            # insets corresponding to data
            all_insets = ndimage.find_objects(self.data)
            
            # extend insets if more ids than seen in data
            if len(all_insets) < ids.max():
                all_insets.extend([None] * (ids.max() - len(all_insets)))

            # keep only insets corresponding to ids
            insets = [all_insets[id_-1] for id_ in ids]

            # remove nonexisting insets and convert to ndarray
            inset_array = []
            for ins_list in insets:
                if ins_list is None: continue
                inset_array.append([[ins.start, ins.stop] \
                                        for ins in ins_list])
            inset_array = numpy.asarray(inset_array)

            # no insets corresponding to ids
            if inset_array.size == 0:
                raise _LocalException()

            # join insets
            inset_start = inset_array[:,:,0].min(axis=0)
            inset_stop = inset_array[:,:,1].max(axis=0)
            inset = [slice(ins_start, ins_stop) for ins_start, ins_stop \
                         in zip(inset_start, inset_stop)]

        except _LocalException:
            inset = None

        # convert to absolute
        inset = self.relativeToAbsoluteInset(inset=inset) 

        # find smallest (absolute) inset for the additional image
        if additional is not None:

            # find inset for additional
            add_inset = additional.findInset(ids=additionalIds, mode='abs')

            # find enclosing inset
            if (inset is not None) and (add_inset is not None):
                # Note: before r1186 included offset 
                inset = self.findEnclosingInset(inset=inset, inset2=add_inset)

            # deal with inset or additional inset being None
            elif (inset is None) and (add_inset is None):
                return None
            elif add_inset is None:
                pass
            elif inset is None:
                inset = add_inset

        # skip the rest if inset is None
        if inset is None: return None

        # extend 
        if extend > 0:
            inset = [slice(sl.start-extend, sl.stop+extend) for sl in inset]

        # convert to absolute
        #inset = self.relativeToAbsoluteInset(inset=inset)
        
        # if not expand do not allow larger than initial
        if not expand:
            inset = [slice(max(init_sl.start, sl.start), 
                           min(init_sl.stop, sl.stop)) 
                     for (sl, init_sl) in zip(inset, init_inset)]
            
        # in any case negative absolute inset values are not allowed
        inset = [slice(max(0, sl.start), sl.stop) for sl in inset]

        # convert to relative if needed
        if ((mode == 'rel') or (mode == 'relative')):
          inset = self.absoluteToRelativeInset(inset=inset)  

        return inset


    ########################################################
    #
    # Basic data and id-related methods
    #
    #########################################################
    
    def setData(self, data=None, copy=True, ids=None):
        """
        Sets data and the attributes determined from data.

        If copy is True data is copied so that (the original) data can not be
        changed in this class. Otherwise only a reference is passed saving
        memory, but there is no guarantee that the data would not be changed.
        
        The id-related attributes: ids (list containing all ids), maxId,
        nextId and n (number of ids) are determined in the following order:
        from ids, data, or self.data.

        Data should always be set using this method (not by the assignment
        to self.data).

        Arguments:
          - data: (ndarray) segments (labels)
          - copy: flag indication if data is copied to self.data.
          - ids: list of ids
        """

        # check and set data
        if data is not None:
            if copy:
                self.data = data.copy()
            else:
                self.data = data

        # set id related attributes based on self.data
        self.setIds(ids=ids)


    def setIds(self, ids=None):
        """
        Sets id and related attributes: self.ids (ndarray of flattened ids),
        self.originalIds (self.maxId, self.nextId and self.n).

        Attribute self.ids is set to (its own copy of) ids. If ids is None the
        ids are extracted from self.data.

        Argument:
          - ids: segment (label) ids
        """

        # keep the original form of ids 
        self.originalIds = ids

        # set ids
        if ids is None:
            self.ids = self.extractIds(self.data)
        elif isinstance(ids, list) or isinstance(ids, tuple) \
                 or isinstance(ids, numpy.ndarray):
            self.ids = nested.flatten(ids)
            self.ids = numpy.asarray(self.ids, dtype='int')
        else:
            self.ids = numpy.array([ids], dtype='int')

        # set other attributes
        self.n = len(self.ids)
        if self.n == 0:
            self.maxId = 0
        else:
            self.maxId = self.ids.max()
        self.nextId = self.maxId + 1

    def extractIds(self, data=None, positive=True):
        """
        Finds segment ids, that is positive unique elements of data array.

        If data is None, self.data is used.

        Argument:
          - data: segments ndarray
          - positive: if True only positive ids are returned

        Returns: ordered ndarray of segment ids. 
        """

        # use self.data if data is not given
        if data is None:
            data = self.data

        # get all unique elements and keep those that are positive if required
        ids = numpy.unique(data)
        if positive:
            ids = ids.compress(ids>0)
        ids = numpy.asarray(ids, dtype='int')

        # return ids 
        return ids

    def getMaxId(self):
        """
        Maximum id
        """
        if (self.ids is None) or (len(self.ids) == 0):
            max_id = 0
        else:
            max_id = self.ids.max()
        return  max_id

    def setMaxId(self, id_):
        """
        Doesn't do anything, needed for compatibility with super.
        """ 
        pass

    maxId = property(fget=getMaxId, fset=setMaxId, doc="Maximum id")

    def remove(self, ids, value=0):
        """
        Removes segments specified in arg ids, by both removing them from
        the data array (self.data). Thus it modifies self.data.

        Uses _remove(remove=ids) to remove segments from data array, see 
        _remove() doc for a more detailed explanation.

        Doesn't do anything to other attributes. This means it doesn't even
        remove the removed ids from self.ids.

        Argument:
          - ids: list of ids to remove
          - value: the replaced ids are replaced by this value
        """

        # remove from data array
        new_data = self._remove(data=self.data, remove=ids, value=value)

        # 
        self.setData(data=new_data)

        # set ids
        #self.setIds(ids=ids)

    def _remove(self, data, remove=None, keep=None, all=None, mode='auto',
                       value=0, slices=None):
        """
        Removes segments labeled by elements of remove, or those that are not
        labeled by keep, by replacing the removed elements of data by value.

        A modified segment ndarray is returned. The original data array may
        be changed (depends on the mode).

        If argument all is None, all ids are determined from data. If all is
        provided it may increase the speed a bit.

        Argument mode determines the method used. If mode is 'remove' the
        segments are removed from the array. If it is 'keep', the segments with
        ids that are not in the argument ids are kept. Finally, the mode 'auto'
        finds the better strategy between the two, based on the numbers  of ids
        that should be removed or kept. Specifically if:

            len(remove) <= len(all) * Segment._remove_or_keep_factor

        the 'remove' mode is used. If not, the 'keep' mode is used. The factor
        Segment._remove_or_keep_factor is chosen to minimize the computational
        time. Currently (17.09.07) it is 0.55, because the remove mode is about
        20% faster than the keep mode.

        If this method is called many times and with the same data (as in 
        hierarchy.removeLoverLevels), it is likely that its preformance will be 
        improved if (arg) slices is given. Slices is a result of 
        ndimage.find_objects that has to contain all relevant id, that is all
        ids to be removed if mode is 'remove' and all ids to be kept if mode
        is 'keep'. If mode is 'auto' it is the safest to have slice objects
        for all ids.

        Arguments:
          - data: segment ndarray
          - remove: ids of segments to be removed (int or list/array, positive)
          - keep: ids of segments not to be removed (int or list/array, 
          positive)
          - all: ids of all segments in data (determined from data if not given)
          - value: value denoting background (no segments)
          - mode: determines the method used, 'auto' is recommended
          - slices: result of ndimage.find_objects()

        Returns:
          - modified array 
        """

        # determine all and remove if not specified
        if all is None:
            all = self.extractIds(data=data)
        if remove is None:
            if keep is not None:
                remove = numpy.setdiff1d(all, keep)
            else:
                raise ValueError(
                    "Either remove or keep argument need to be specified.")  

        # make remove and keep lists if needed
        if remove is not None:
            if not isinstance(remove, (list, tuple, numpy.ndarray)):
                remove = [remove]
        if keep is not None:
            if not isinstance(keep, (list, tuple, numpy.ndarray)):
                keep = [keep]

        if mode == 'auto':

            # decide which mode to use
            if len(remove) <= len(all) * self._remove_or_keep_factor:
                mode = 'remove'
            else:
                mode = 'keep'
        
        # do the work
        if mode == 'remove':

            # check
            if len(remove) == 0:
                return data

            # remove segments from data
            if slices is None:
                try:
                    max_remove = max(remove)
                except ValueError:
                    max_remove = 1
                slices = ndimage.find_objects(data, max_label=max_remove)
            condition = numpy.zeros(data.shape, dtype=bool)
            for id in remove:
                try:
                    curr_slice = slices[id-1]
                except IndexError:
                    continue
                if curr_slice is None:
                    continue
                condition[curr_slice] = condition[curr_slice] \
                        | (data[curr_slice] == id)
            # alternative to the above 3 lines, but no speed improvement
            #def xxx(condition, id): return condition | (data == id)
            #condition = reduce(xxx, ids, numpy.zeros(data.shape, dtype=bool))
            data[condition] = value

        elif mode == 'keep':

            # determine keep if needed
            if keep is None:
                keep = numpy.setdiff1d(all, remove)

            # keep segments
            if slices is None:
                try:
                    max_keep = max(keep)
                except ValueError:
                    max_keep = 1
                slices = ndimage.find_objects(data, max_label=max_keep)
            #condition = numpy.zeros(data.shape, dtype=bool)
            condition = (data == 0)
            for id in keep:
                try:
                    curr_slice = slices[id-1]
                except IndexError:
                    continue
                if curr_slice is None:
                    continue
                condition[curr_slice] = condition[curr_slice] \
                    | (data[curr_slice] == id)
            data = numpy.where(condition, data, value)

        else:
            raise ValueError("Mode can be 'auto', 'delete', or 'keep'.")
            
        #
        return data
        
    def restrict(self, mask, ids=None, update=True):
        """
        Removes (sets to 0) data elements of this instance that have one of 
        the specified ids and where mask.data is 0 (if arg update is True).

        If arg update is False, the data of this instance is not modified,
        instead a new ndarray is returned.

        Respects inset attributes. Acts only on elements of the current inset.
        Elements of mask that are outside mask's inset (mask.data) are 
        considered to be 0.  

        Arguments:
          - mask: (Labels) image used to restrict this instance
          - ids: list of ids of this instance that are affected, if None all
          ids are affected
          - update: flag indicating if the data of this instance is modified

        Retruns (if update is True): (ndarray) modified (restricted) data
        """

        # expand mask to self.inset by padding with 0's if needed
        mask_data = mask.useInset(inset=self.inset, mode='abs', useFull=False,
                                  expand=True, update=False) 

        # copy data if update
        if update:
            data = self.data
        else:
            data = self.data.copy()

        if ids is None:

            # remove all unmasked elements
            data[mask_data == 0] = 0

        else:

            # remove unmasked elements for specified ids only
            for id_ in ids:
                data[(mask_data == 0) & (data == id_)] = 0

        # returne if update
        if not update:
            return data

    ###########################################################
    #
    # Reordering segments and ids
    #
    ###########################################################
    
    def reorder(self, order=None, data=None, clean=False):
        """
        Changes ids of segments according to the order, or orderes ids from
        1 up without gaps if order is not given.

        Returns (dictionary) order where keys are old and values are new ids,
        if argument order is None. 

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
        
        # set data and ids
        if data is None:
            data = self.data
            ids = self.ids
            update = True
        else:
            ids = self.extractIds(data=data)
            update = False

        # set initial value of reordered data array
        if clean:
            reordered = numpy.zeros_like(data)
        else:
            reordered = data.copy()

        # make default order if not given and get new_ids
        if order is None:
            new_ids = numpy.arange(1, ids.shape[0]+1)
            new_order = dict(list(zip(ids, new_ids)))
        else:
            new_order = order
            if clean:
                new_ids = numpy.array(list(new_order.values())).sort()
            else:
                new_ids = None
           
        # reorder id 0, if needed
        try:
            reordered[data==0] = new_order.pop(0) 
        except KeyError:
            # 0 need not be reordered
            pass

        # reorder other ids
        slices = ndimage.find_objects(data)
        for old in new_order:
            try:
                curr_slice = slices[old-1]
            except IndexError:
                logging.debug("labels.py.reorder(): Most likely one or more "\
                              + "of the highest ids don't exist, id = %d", old)
                continue
            reordered[curr_slice][data[curr_slice]==old] = new_order[old]
                   
        # update
        if update:
            self.setData(data=reordered, copy=False, ids=new_ids)

        # return
        if update:
            if order is None:
                return new_order
            else:
                return            
        else:
            if order is None:
                return reordered, new_order
            else:
                return reordered
        
    def shiftIds(self, shift, data=None):
        """
        Changes all (positive) ids of segments by shifting them by 
        (arg) shift.

        If data is None segments in self.data are reordered, self.ids is
        adjusted and nothing is returned. Otherwise, segments in data are 
        reordered and the resulting array is returned (data is modified).

        Arguments:
          - shift: value by which ids are increased
          - data: labels array 

        Sets: self.data and self.ids if data is None.

        Returns:
          - nothing, if data is None and order is not None
          - reordered array, if data is not None and order is not None
        """

        # set data and ids
        if data is None:
            data = self.data
            ids = self.ids
            update = True
        else:
            ids = self.extractIds(data=data)
            update = False

        # reorder
        reordered = numpy.where(data>0, data+shift, data)
        if len(ids) > 0:
            new_ids = ids + shift
        
        # update and return
        if update and (len(ids) > 0):
            self.setData(data=reordered, copy=False, ids=new_ids)
            return            
        else:
            return reordered
        
    def orderByBoundaries(self, contacts):
        """
        Orders segments both here and in the contacts object by ids of
        contacted boundaries. Both contacts data structure and the current
        instance are updated. Calls contacts.orderSegmentsByContactedBoundaries
        to get the order.

        Arguments:
          - contacts: Contact object

        Returns:
          - dictionary where keys are old ids and values are new ids.
        """

        # order contacts
        new_order = contacts.orderSegmentsByContactedBoundaries(argsort=True)

        # order segments
        self.reorder(new_order)

        return new_order

    def orderIdsByBoundaries(self, contacts, argsort=False):
        """
        Orders self.ids according to the ids of contacted boundaries. Calls
        contacts.orderSegmentsByContactedBoundaries to get the order.

        Arguments:
          - contacts: Contact object
          - argsort: if True, also returns the list used for ordering

        Returns:
          - if argsort is False: ordered_ids
          - if argsort is True: ordered_ids, sort_array array (as returned from
          ndarray.argsort)
        """

        # order contacts
        order = contacts.orderSegmentsByContactedBoundaries(argsort=True)

        if argsort:
            return self.ids[order], order
        else:
            return self.ids[order]

    def orderByValues(self, values, contacts=None):
        """
        Orders segments, that is changes their ids according to values.

        The elements of values correspond to elements of self.ids.
        
        If contacts is given, the segments in contacts are ordered in the same
        way. Contacts have to contain the contact data structure already and
        the segments in contacts are expected to be the same as segments in
        this instance.

        Arguments:
          - values: (list or ndarray) of values corresponding to self.ids
          - contacts: instance of Contacts 

        Returns dictionary where keys are old ids and values are new ids.
        """

        # make sort list
        values = numpy.asarray(values)
        sort_list = values.argsort()

        # sort segments
        new_order = dict(list(zip(self.ids[sort_list], self.ids)))
        self.reorder(new_order)

        # sort contacts
        if contacts is not None:
            contacts.reorderSegments(new_order)

        return new_order

    def orderIdsByValues(self, values, contacts=None, argsort=False):
        """
        Orders ids by values.

        The elements of values correspond to elements of self.ids. The same
        permuation that orders values is then used to order ids.

        If contacts is given, the segments in contacts are ordered in the same
        way. Contacts have to contain the contact data structure already and
        the segments in contacts are expected to be the same as segments in
        this instance.

        Arguments:
          - values: (list or ndarray) of values corresponding to self.ids
          - contacts: instance of Contacts 

        Returns dictionary where keys are old ids and values are new ids.
        """

        # make sort list
        values = numpy.asarray(values)
        sort_list = values.argsort()

        if argsort:
            return self.ids[sort_list], sort_list
        else:
            return self.ids[sort_list]
        

    ###########################################################
    #
    # Structuring elements needed for many subclasses
    #
    ############################################################

    def setStructEl(self, rank=None, connectivity=None, size=None,
                    mode=None, axis=None):
        """
        Sets or updates structuring element (self.structEl).

        If self.structEl is None, it is set to a new instance of StructEl
        and sets attributes of this instance to appropriate arguments.
        Attribute self.structEl.rank is set to the argument rank, or if the
        argument is None to self.ndim. If connectivity is None, it is set to 1.
        If size is None it is set to 3 if connectivity is 1, 2 or 3, or to 5
        if connectivity is 4.

        If self.structEl is not None, updates the attributes of this instance
        to the values of non-None attributes. If self.structEl.rank is None,
        sets it to self.ndim (useful when self.data is set after self.structEl).

        Arguments  (see StructEl):
          - rank: dimensionality of the structuring element
          - connectivity: square of the distance from the center that defines 
          which elements belong to the structuring element (as in
          scipy.ndimage.generate_binary_structure)
          - size: size of the structuring element (same in each dimension) 
          - mode: type of structuring element
          - axis: structuring element axis if mode is '1d'

        Sets:
          - self.structElConn: connectivity
          - self.structElSize: size
        """

        if self.structEl is None:

            # instantiate new structuring element
            if rank is None: rank = self.ndim
            if connectivity is None: connectivity = 1
            #if size is None: size = 3
            self.structEl = StructEl(
                rank=rank, mode=mode, connectivity=connectivity, size=size, 
                axis=axis)
        else:

            # update the existing structuring element
            if rank is not None:
                self.structEl.rank = rank
            elif self.structEl.rank is None:
                self.structEl.rank = self.ndim
            if mode is not None: self.structEl.mode = mode
            if connectivity is not None: 
                self.structEl.connectivity = connectivity
            if axis is not None: self.structEl.axis = axis
            if size is not None: self.structEl.size = size
            
    def setContactStructElConn(self, connectivity=None):
        """
        Sets self._contactStructElConn.
        """
        if connectivity is not None:
            self._contactStructElConn = connectivity

    def getContactStructElConn(self):
        """
        Returns self._contactStructElConn. If it is not found returns the 
        default value (1).
        """
        try:
            conn = self._contactStructElConn
        except AttributeError:
            conn = 1
        return conn

    contactStructElConn = property(
        fget=getContactStructElConn,  fset=setContactStructElConn,
        doc="Contact structuring element connectivity, default 1")
    
    def getContactStructEl(self):
        """
        Generates and returns contact structuring element.
        """

        # generate se
        se = ndimage.generate_binary_structure(
            rank=self.ndim, connectivity=self.contactStructElConn)
        return se

    contactStructEl = property(
        fget=getContactStructEl,
        doc="Contact structuring element, default connectivity 1")
    
    def setCountStructElConn(self, connectivity=None):
        """
        Sets self._countStructElConn.
        """
        if connectivity is not None:
            self._countStructElConn = connectivity

    def getCountStructElConn(self):
        """
        Returns self._countStructElConn. If it is not found returns the default
        value (1).
        """
        try:
            conn = self._countStructElConn
        except AttributeError:
            conn = 1
        return conn

    countStructElConn = property(
        fget=getCountStructElConn, fset=setCountStructElConn,
        doc="Count structuring element connectivity, default ndim")
    
    def getCountStructEl(self):
        """
        Generates and returns count structuring element.
        """

        # generate se
        se = ndimage.generate_binary_structure(
            rank=self.ndim, connectivity=self.countStructElConn)
        return se

    countStructEl = property(
        fget=getCountStructEl,
        doc="Count structuring element, default connectivity ndim")

    
    ###########################################################
    #
    # Methods dealing with coordinates
    #
    ############################################################

    def getPoints(self, ids, mode='all', connectivity=1, distance=2,
                  format_='coordinates'):
        """
        Returns coordinates of selected elements (points) of segments labeled 
        by arg ids.

        If mode is 'all', coordinates of all points are returned.

        If mode is 'geodesic', the points are selected so that they are not
        closer than the argument distance, using the geodesic metric with
        the structuring element determined by arg connectivity.

        Respects inset attribute, that is the returned coordinates are given 
        for the full size array self.data. In addition, it works internally 
        with the smallest subarray of self.data that contains all ids.

        Note

        Arguments:
          - ids: (list or ndarray) ids, or a sigle (int) id 
          - mode: determines how the points are selected
          - distance: min distance between selected points (needed if mode is
          'geodesic')
          - connectivity: connectivity for structuring element (needed if mode 
          is 'geodesic')
          - format_: output format; 'numpy' for the format used by 
          numpy.nonzero(), or 'coordinates' for a 2D array where the first 
          index denotes different points, and the second the coordinates of the
          point.
        """
        
        # make an array that labels all given ids
        local_data = numpy.zeros_like(self.data, dtype='bool')
        if type(ids) == int:
            ids = [ids]
        for id_ in ids:
            local_data = local_data | (self.data == id_)
        # use the smallest inset
        local_labels = Labels(data=local_data)
        local_labels.setInset(inset=self.inset)
        local_labels.makeInset(ids=[1])
        inset = local_labels.inset

        if mode == 'all':

            # mode all
            coords = local_labels.data.nonzero()
 
            # adjust for inset
            coords_list = [coords_one + one_slice.start 
                           for coords_one, one_slice in zip(coords, inset)]
            coords = tuple(coords_list)

            # output format
            if format_ == 'coordinates':
                coords = numpy.array(coords).transpose()

        elif mode == 'geodesic':

            # mode geodesic
            coords = self.getPointsGeodesic(
                data=local_labels.data, connectivity=connectivity, 
                distance=distance)
           
            # adjust for inset
            for one_slice, index in zip (inset, list(range(len(inset)))):
                coords[:, index] += one_slice.start
                
            # output format
            if format_ == 'numpy':
                coords_numpy = [x for x in coords.transpose()]
                coords = tuple(coords_numpy)

        else:

            raise ValueError('Argument mode ', mode, ' was not understood.')

        return coords
                                  
    def getPointsGeodesic(self, data, connectivity, distance):
        """
        Returns coordinates of selected elements (points) of segments labeled 
        by arg ids. The points are selected so that they are not
        closer than the argument distance, using the geodesic metric with
        the structuring element determined by arg connectivity.

        Respects inset attribute, that is the returned coordinates are given 
        for the full size array self.data. In addition, it works internally 
        with the smallest subarray of self.data that contains all ids.

        Arguments:
          - data (bool ndarray):  
          - distance (int): min distance between selected points 
          - connectivity: connectivity for structuring element 

        Returns: coordinates
        """

        # make structure element and dilation of a point
        struct_el = scipy.ndimage.generate_binary_structure(
            rank=data.ndim, connectivity=connectivity)
        n_iter = distance - 1
        shape = [2*distance - 1] * data.ndim
        center = [distance - 1] * data.ndim
        dilated = numpy.zeros(shape, dtype='bool')
        dilated[tuple(center)] = True
        dilated = scipy.ndimage.binary_dilation(
                dilated, structure=struct_el, iterations=n_iter)

        # initialize arrays that will show points that are taken
        occupied = numpy.zeros_like(data, dtype='bool')
        points = []

        # get coordinates of all points
        free_coords = numpy.array((data & ~occupied).nonzero()).transpose() 
        
        # find points
        while len(free_coords) > 0:

            # pick a random point
            random_index = numpy.random.randint(0, len(free_coords))
            random_point = free_coords[random_index]
            points.append(random_point)
 
            # mark neighborhood
            if n_iter < 1:
                break
            # deleted part slow
            #new_occupied = numpy.zeros_like(data, dtype='bool')
            #new_occupied[tuple(random_point)] = True
            #new_occupied = scipy.ndimage.binary_dilation(
            #    new_occupied, structure=struct_el, iterations=n_iter)
            #occupied = occupied | new_occupied
            dilated_inset = [slice(x-distance+1, x+distance) 
                             for x in random_point]
            try:
                occupied[tuple(dilated_inset)] = \
                    occupied[tuple(dilated_inset)] | dilated
            except ValueError:
                occupied_slice, dilated_slice = \
                    np_plus.trim_slice(slice_nd=dilated_inset, 
                                       shape=occupied.shape)
                occupied[occupied_slice] = \
                    occupied[occupied_slice] | dilated[dilated_slice]

            # get coordinates of points that are still free
            # performance: nonzero() call major bottleneck
            free_coords = numpy.array((data & ~occupied).nonzero()).transpose() 

        points = numpy.array(points)

        return points
        

    ###########################################################
    #
    # Other methods
    #
    ############################################################

    def magnify(self, factor):
        """
        Magnifies (increases the size of) the data array (self.data) by 
        an int factor.

        All other attributes remain the same.
        
        Argument:
          - factor: magnification factor (int)
        """

        # initialize
        new_shape = numpy.array(self.data.shape) * factor
        new_data = numpy.zeros(shape=new_shape, dtype=self.data.dtype)

        # set values in the new data 
        for old_coords, old_value in numpy.ndenumerate(self.data):
            new_coords_low = factor * numpy.array(old_coords)
            new_coords = tuple(slice(low, low+factor) for low in new_coords_low)
            new_data[new_coords] = old_value

        # update data
        self.data = new_data
 

###########################################################
#
# Local exception class
#
##########################################################

class _LocalException(Exception):
    """
    Exception used for flow control in Labels
    """
