"""
Contains class Image that defines basic properties of an (gray-scale, integer, 
...) image represented by an ndarray.

This is a base class for specialized classes dealing with different types
of images. This class should not be instantiated


# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from builtins import zip
from builtins import object
from past.builtins import basestring

__version__ = u"$Revision$"


from copy import copy, deepcopy

import numpy
import scipy
import scipy.ndimage as ndimage

#from pyto.io.image_io import ImageIO

class Image(object):
    """
    Methods for creating new images:
      - tile: tile current image to get a bigger one

    Input/output methods (see pyto.io.image_io for details): 
      - read: reads data array from a file in em, mrc or raw-data format
      - write: writes data to a file

    Positioning of the image data array:

      - useInset(): main method to use, sets data to a specified inset
      - newFromInset(): makes new instance from this instance after data is set
      to s specified inset
      - inset: property defining current inset in respect tosome absolute 
      reference
      - usePositioning() and copyPositioning(): based on useInset(), also 
      takes property offset into account (depreciated)
      - saveFull(), recoverFull(), clearFull(): store, recall and clear 
      another (previous, for example) inset and data array (properties
      fullInset and fullData)
      - fullInset, fullData: properties than can be set, but if there're
      not set they take the values from the deepest base of self.data
      (see getDeepBase())


    Example, speeding up calculations, memory not increased:

      new_data = self.useInset(inset=new_inset, update=False)

      (calculations using new_data)

    Alternative, the calculations can use the whole object:

      data, inset = self.getDataInset()   # keep reference to the original array
      self.useInset(inset=smaller_inset)  

        (calculations using self)

      self.setDataInset(data=data, inset=inset) # self.data points to the 
                                                # original array again 

      Note: smaller_inset has to be smaller than self.inset. In this way 
      calculations are done on a view of self.data (to increase speed) and 
      self.data is never copied (so the memory is not increased). 

    Example: memory use reduction

      new = self.newFromInset(inset=smaller_inset, copyData=True, deepcp=True)
      self = None     # so that self.data can be removed by garbage collection
      
        (calculations involving new)

      Again, small_inset has to be smaller than self.inset. 
    """

    #############################################################
    #
    # Initialization
    #
    ############################################################

    def __init__(self, data=None):
        """
        """

        # initialize attributes
        self.data = data
        self._offset = None
        self._inset = None

        self._fullInset = None
        self._fullData = None

        # file io
        self.file_io = None


    #############################################################
    #
    # Generate images
    #
    ############################################################

    def tile(self, shape, mode=u'simple'):
        """
        Use the current instance (image) as a tile (pattern) to make a (bigger) 
        image in arbitrary dimensions.

        If arg mode is 'simple', the tile (this instance) is simply repeated
        in all dimensions.

        If arg mode is 'mirror', each tile is mirrored before it is placed so 
        that its (ndim-1 dimensional) faces have the same values as the 
        neghboring faces of other tiles and it is placed so that the same faces
        overlap. For example:

        tile: [[0, 1, 2, 3],
               [4, 5, 6, 7]]

        big:  [[0, 1, 2, 3, 2, 1, 0, 1],
               [4, 5, 6, 7, 6, 5, 4, 5],
               [0, 1, 2, 3, 2, 1, 0, 1]]

        Arguments:
          - shape: shape of the newly created image
          - mode: tailing method, 'simple' or 'mirror'

        Return:
          - instance of the same class containing the newly created image
        """

        # make unit tile for different modes
        if mode == u'simple':

            unit_tile = self.data

        elif mode == u'mirror':

            # prepare slices for reflection
            identity = slice(None)
            reflect = slice(None, None, -1)
            geometry = {0 : identity,
                        1 : reflect}

            # slice object for trimmed (last position in each dim removed) tile 
            trim_shape = numpy.array(self.data.shape) - 1
            trim_slice = tuple([slice(0, x) for x in trim_shape])

            # initialize unit tile (almost double in each dimension)
            unit_tile = numpy.zeros(shape=2*numpy.array(self.data.shape)-2, 
                                     dtype=self.data.dtype)

            # generate and put identity/reflect tiles to make the unit tile 
            for nd_item in numpy.ndindex(tuple(self.data.ndim * [2])):

                # mirror tile 
                slice_obj = tuple([geometry[x] for x in nd_item])
                curr_tile = self.data[slice_obj]
                curr_tile = curr_tile[trim_slice]

                # put mirrored tile at the right position
                position = tuple([slice(x * length, (x+1) * length) 
                                  for x, length in zip(nd_item, trim_shape)]) 
                unit_tile[position] = curr_tile

        else:
            raise ValueError(
                u'Mode: ' + mode + u" not understood. Available " 
                + u"modes are 'simple' and 'mirror'.")

        # find number of repeats
        repeat = numpy.true_divide(shape, unit_tile.shape)
        repeat = numpy.cast[int](numpy.ceil(repeat))

        # tile with the unit tile and trim to the required shape 
        big = numpy.tile(unit_tile, repeat)
        big_slice = tuple([slice(0, x) for x in shape]) 
        big = big[big_slice]

        return self.__class__(data=big)


    #############################################################
    #
    # Positioning of the image
    #
    ############################################################

    def getNdim(self):
        """
        Gets ndim from self.data, or if the data does not exists returns 
        self._ndim.
        """
        try:
            ndim = self.data.ndim
        except AttributeError:
            ndim = self._ndim
        return ndim

    def setNdim(self, ndim):
        """
        Sets self._ndim.
        """
        self._ndim = ndim
    
    ndim = property(fget=getNdim, fset=setNdim, doc=u"Dimensionality of data")

    def setOffset(self, offset=None):
        """
        Sets offset of  (ndarray) self.offset.

        Argument:
          - offset: position of offset (1d array or similar)
        """
        if offset is None:
            self._offset = None
        else:
            self._offset = numpy.asarray(offset)

    def getOffset(self):
        """
        Returns current offset value.

        If offset has not been set, it is set to 0's if self.data is set 
        """
        try:
            if self._offset is None:
                self._offset = numpy.zeros(shape=self.ndim, dtype=u'int_')
        except AttributeError:
            self._offset = None
        return self._offset

    offset = property(
        fget=getOffset, fset=setOffset, 
        doc=u"Position of the origin in respect to some reference")

    def getTotalOffset(self):
        """
        Returns total offset, which is a sum of self.offset and the start 
        positions of self.inset.
        """
        if self.inset is None:
            return self.offset
        else:
            return self.offset + [sl.start for sl in self.inset]

    totalOffset = property(
        fget=getTotalOffset,
        doc=(u"Total offset, that is a sum of self.offset and " 
             + u"the start positions of self.inset."))

    def setInset(self, inset, mode=u'absolute'):
        """
        Sets inset for self.data (does not modify self.data).

        If mode is 'absolute' self.inset is set to (arg) inset. If it is 
        'relative' self inset is obtained by applying (arg) inset on the 
        existing self.inset.

        Note: start, stop and step for each dimension are saved as a 2d array
        in self._inset, because deepcopy doesn't work on slices. 

        Argument:
          - inset: list of slice objects
          - mode: 'relative' (same as 'rel') or 'absolute' (same as 'abs')
        """

        if inset is None:
            self._inset = None
            
        elif (mode == u'absolute') or (mode == u'abs') or (self.inset is None):
            self._inset = [[sl.start, sl.stop, sl.step] for sl in inset]

        elif (mode == u'relative') or (mode == u'rel'):
            self._inset = \
                [[old.start + new.start, old.start + new.stop, new.step] \
                     for old, new in zip(self.inset, inset)]

    def getInset(self):
        """
        Returns inset position as a list of slice objects.

        If inset was not defined, it is determined from data shape. If data
        doesn't exist either, None is returned.
        """

        if self._inset is None:

            # determine inset
            try:
                slice_list = [slice(0, shape_1d) for shape_1d \
                                  in self.data.shape]
            except AttributeError:
                slice_list = None

        else:
                
            # convert to slice form
            slice_list = [slice(sl[0], sl[1], sl[2]) for sl in self._inset]
                
        return slice_list

    inset = property(
        fget=getInset, fset=setInset, 
        doc=(u"(list of slice objects) a current view of self.data "
             + u"array in respect to the underlying base reference array."))

    def relativeToAbsoluteInset(self, inset):
        """
        Converts relative inset (in respect to the current inset (self.inset))
        to an absolute inset.

        The resulting inset may fall outside the current inset, or even have 
        negative slice values.

        If (arg) inset is None, None is returned.

        Argument:
          - inset: inset relative to the current inset

        Returns: absolute inset
        """

        if inset is None:
            return None

        else:
            abs_ins = [slice(rel_ins.start + self_ins.start, 
                             rel_ins.stop + self_ins.start) 
                       for rel_ins, self_ins in zip(inset, self.inset)] 
            return abs_ins

    def absoluteToRelativeInset(self, inset):
        """
        Converts given absolute inset into inset relative to the current inset
        (self.inset).

        The resulting inset may fall outside the current inset. If (arg) inset
        is None, None is returned. 

        Argument:
          - inset: absolute inset

        Returns: inset relative to the corrent inset
        """
        if inset is None:
            return None

        else:
            rel_ins = [slice(abs_ins.start - self_ins.start, 
                             abs_ins.stop - self_ins.start) 
                       for abs_ins, self_ins in zip(inset, self.inset)] 
            return rel_ins

    def setFullInset(self, inset):
        """
        Sets full inset for self.data (does not modify self.data).

        Note: start, stop and step for each dimension are saved as a 2d array
        in self._fullInset, because deepcopy doesn't work on slices. 

        Argument:
          - inset: list of slice objects
        """

        if inset is None:
            self._fullInset = None
            
        else:
            self._fullInset = [[sl.start, sl.stop, sl.step] for sl in inset]

    def getFullInset(self):
        """
        Returns a previously defined inset (list of slice objects) of a 
        full-size array underlying (the current view) of self.data.
        
        If fullInset was not defined, returns the inset corresponding to the 
        deepest base of array underlying self.data that still has the same 
        dimensionality as self.data.

        If self.data doesn't exist None is returned.
        """

        if self._fullInset is None:

            # find base data shape
            base_array = self.getDeepBase()
            if base_array is None:
                return None
            inset = [slice(0, base_sh) for base_sh in base_array.shape]
            return inset

        else:
                
            # convert to slice form
            slice_list = [slice(sl[0], sl[1], sl[2]) for sl in self._fullInset]
                
        return slice_list

    fullInset = property(fget=getFullInset, fset=setFullInset, 
                         doc=u"Full inset that underlies current self.data.") 

    def getFullData(self):
        """
        Returns a previously defined full-size array underlying (the current
        view of) self.data.  
        
        If fullData was not defined, returns the deepest base array of 
        self.data that still has the same dimensionality as self.data. 
        Returns None if self.data wasn't defined.
        """

        if self._fullData is None:

            # find base data
            return self.getDeepBase()
        
        else:
            return self._fullData

    def setFullData(self, data):
        """
        Sets full size data (the array is not copied).
        """

        self._fullData = data

    fullData = property(
        fget=getFullData, fset=setFullData, doc=u'Full size data array')

    def getDeepBase(self, data=None):
        """
        Finds the deepest base of array data that still has the same 
        dimensionality as data.

        If data is None, self.data is used. Returns None if neither data nor 
        self.data exist.
        """

        if data is None:
            data = self.data
        if data is None:
            return None
            
        ndim = data.ndim
        try:
            while (data.base is not None) and (data.base.ndim == ndim):
                data = data.base
        except AttributeError:
            if isinstance(data.base, basestring):
                # data.base is a string, happens after unpickling
                pass
            else:
                raise

        return data
        
    def saveFull(self):
        """
        Sets self.fullInset and self.fullData to the current self.inset and 
        self.data.
        """

        self.fullInset = self.inset
        self.fullData = self.data
        
    def recoverFull(self):
        """
        Recovers the full array (shape self.fullShape) that is a base for the
        current self.data. Consequently, it does not allocate new memory for 
        the self.data array.

        Resets self.inset to None.
        """

        self.inset = self.fullInset
        self.data = self.fullData
        
    def clearFull(self):
        """
        Sets fullInset and fullData to None.
        """
        self.fullInset = None
        self.fullData = None

    def getDataInset(self):
        """
        Returns current data (self.data) and inset (self.inset). 

        The returned data is only a reference to the self.data, so this
        method is meant to be used before useInset() (or related), so that
        after the work on data is finished setDataInset() can be used to
        recover the original data, without using additional space.
        """
        return self.data, self.inset

    def setDataInset(self, data, inset):
        """
        Sets self.data to data and self.inset to inset. 
        """
        self.data = data
        self.inset = inset

    def useInset(self, inset, mode=u'relative', intersect=False, useFull=False,
                 expand=False, value=0, update=True):
        """
        Finds data inset of the current data that corresponds to the 
        arguments specified. If arg update is True also sets self.data to the
        data inset found and data.inset to the absolute inset corresponding
        to the arguments.

        If mode is relative (arg) inset is relative to the current inset. If 
        mode is absolute self.inset and (arg) inset are given in respect to 
        the same reference. 

        If the new inset falls inside the current one (self.inset), self.data
        (if update is True) or the returned array (if update is False) is set 
        to the appropriate view of self.data.

        If intersect is True, the new inset is obtained as an intersection
        between self.inset and (arg) inset (according to the mode). Self.data 
        (if update is True) or the returned array (if update is False) is set 
        to the appropriate view of self.data.

        Because in the previous two cases only the view is modified, changing
        the new or the returned data changes the original data.

        If the new inset is outside self.inset, but inside self.fullInset and
        intersect is False, the behavior depends on useFull. If it's False an
        exception is raised. Otherwise self.data is set to the appropriate view
        of self.fullData.
        
        If the new inset is outside self.fullData, useFull is True and 
        intersect is False, the behavior depends on expand. If it is False an 
        exception is raised. Otherwise, a new data array is created that 
        contains self.fullData and it is large enough to fit the new inset 
        (elements outside self.fullData are set to arg value.

        Important: in the cases when self.fullData is used (the last two cases)
        self.data and self.fullData need to be consistent, that is self.data 
        should be a view of self.fullData.

        Important: in case the data array is expanded and _fullInset and 
        _fullData were set, properties fullInset and fullData might end up 
        smaller than inset and data. To avoid that execute self.saveFull() 
        after this method (removes old _fullInset and _fullData, but does not 
        generate any additional array).

        Unless self.data is expanded (prevented by expand=False or 
        intersect=True) no new array is formed in the memory, only the view 
        of the array is changed. 

        In all cases when a new array is created, changing self.data
        (if update is True) or the returned array (if update is False) does
        not change the original array.
        
        Can be used only on subclasses of this class that have data attribute.

        Sets self.inset and self.data.

        Arguments:
          - inset: list of slice objects that defines the inset
          - mode: 'relative' (same as 'rel') or 'absolute' (same as 'abs')
          - intersect: flag indicating if the (agr) inset is to be intersected 
          with the current inset.
          - useFull: flag indicating if self.fullData is used
          - expand: flag indicating if self.data array should be expanded if 
          needed
          - value: used to fill the expanded part of self.data
          - update: flag indicating if the current object is updated, otherwise
          new data array is returned

        Sets (only if update is True):
          - self.data: new data
          - self.inset: inset

        Return: (ndarray) data inset found; it can be a view of the initial 
        data or a new data array, see above.
        """
        
        # this instance need to use a copy of inset
        #print 'id(inset)' + str(id(inset))
        #inset = copy(inset)
        #print 'id(inset copy)' + str(id(inset))

        # not sure if it should be here
        #if inset is None:
        #    inset = [slice(0,0)] * self.ndim
        #    return

        # find absolute inset
        if (mode == u'relative') or (mode == u'rel'):

            # convert to absolute inset
            abs_inset = [slice(rel_in.start + self_in.start,
                               rel_in.stop + self_in.start) 
                         for rel_in, self_in in zip(inset, self.inset)]
                
        elif (mode == u'absolute') or (mode == u'abs'):

            # new inset is absolute
            abs_inset = inset

        else: 
            raise ValueError(
                u"Argument mode can be either u'relative' or u'absolute'.")

        # check if requested inset lies withing the inset and the full self.data
        inside = min((abs_in.start >= self_in.start) 
                     and (abs_in.stop <= self_in.stop) 
                     for abs_in, self_in in zip(abs_inset, self.inset))
        inside_full = min((abs_in.start >= full_in.start) 
                          and (abs_in.stop <= full_in.stop) 
                          for abs_in, full_in in zip(abs_inset, self.fullInset))

        # adjust absolute inset if needed and check if it is inside the 
        # full shape
        if inside or intersect:

            # if outside inset, intersect with current inset
            if not inside:
                abs_inset = [slice(max(abs_in.start, self_in.start),
                                   min(abs_in.stop, self_in.stop)) 
                             for abs_in, self_in in zip(abs_inset, self.inset)] 

            # calculate inset relative to self.inset
            rel_inset = [slice(abs_in.start - full_in.start,
                               abs_in.stop - full_in.start) 
                         for abs_in, full_in in zip(abs_inset, self.inset)]

            # set inset and data
            new_data = self.data[tuple(rel_inset)]
            if update:
                self.inset = abs_inset
                self.data = new_data
                
        elif inside_full and useFull:

            # calculate inset relative to the full inset
            rel_inset = [slice(abs_in.start - full_in.start,
                               abs_in.stop - full_in.start) 
                         for abs_in, full_in in zip(abs_inset, self.fullInset)]

            # recover full and set inset and data
            before_recover_inset = self.inset 
            self.recoverFull()
            new_data = self.data[tuple(rel_inset)]
            if update:
                self.inset = abs_inset
                self.data = new_data
            else:
                self.useInset(inset=before_recover_inset, mode=u'abs')

        elif expand:

            # create new array
            new_data = self.expandInset(inset=abs_inset, 
                                        value=value, update=update)

        else:

            # outside, but can't expand or outside full but can't use full
            raise ValueError(u"Inset falls outside the full self.data array.")

        return new_data

    def expandInset(self, inset, value=0, update=True):
        """
        Expands curent self.data array so that it corresponds to inset. 

        A new array is created, and the old self.data is copied on the
        appropriate position. Parts of old self.data that fall outside of 
        inset are removed, so that the new data and inset correspond to arg 
        inset.

        If self.inset and inset do not overlap at all, self inset is set to 
        arg inset and the shape of self.data is set so that it corresponrs
        to the inset.

        If arg update is True, data and inset attributes of the current
        object are set to the new data and inset, respectively.

        Arguments:
          - inset: absolute inset
          - value: value to fill the added parts of self.data
          - update: flag indicating if the current object is updated

        Sets (only if update is True):
          - self.data: new data
          - self.inset: inset

        Return: (ndarray) new data
        """

        # common inset relative to the new (inset)
        rel_inset_new = [slice(max(old.start - new.start, 0),
                               min(old.stop, new.stop) - new.start)
                         for new, old in zip(inset, self.inset)]

        # common inset relative to the old (self.inset)
        rel_inset_old = [slice(max(new.start - old.start, 0), 
                               min(old.stop, new.stop) - old.start)
                         for new, old in zip(inset, self.inset)]

        # initialize new data array
        new_shape = [in_.stop - in_.start for in_ in inset]
        new_data = numpy.zeros(shape=new_shape, dtype=self.data.dtype) + value

        # get new data
        if self.hasOverlap(inset=inset):
            new_data[tuple(rel_inset_new)] = self.data[tuple(rel_inset_old)]
        else:
            pass

        # update current object if required
        if update:
            self.data = new_data
            self.inset = inset

        return new_data

    def usePositioning(self, image, intersect=False, useFull=False, 
                       expand=False, value=0, new=False):
        """
        Depreciated

        Adjusts inset and data attributes (but not offset) by applying the 
        positioning (offset and inset) of (arg) image on the positioning and 
        data of this instance. Offsets and insets of both (arg) image and this 
        instance are taken into account.

        If the new inset falls inside the current one (self.inset), self.data
        is set to the appropriate view of self.data.

        If intersect is True, the new inset is obtained as an intersection
        between self.inset and (arg) inset (according to the mode). Self.data is
        set to the appropriate view of self.data.

        If the new inset is outside self.inset, but inside self.fullInset and
        intersect is False, the behavior depends on useFull. If it's False an
        exception is raised. Otherwise self.data is set to the appropriate view
        of self.fullData.
        
        If the new inset is outside self.fullData, useFull is True and 
        intersect is False, the behavior depends on expand. If it is False an 
        exception is raised. Otherwise, a new data array is created that 
        contains self.fullData and it is large enough to fit the new inset 
        (elements outside self.fullData are set to arg value.

        Important: in the cases when self.fullData is used (the last two cases)
        self.data and self.fullData need to be consistent, that is self.data 
        should be a view of self.fullData.

        Unless self.data is expanded (prevented by expand=False or 
        intersect=True) no new array is formed in the memory, only the view 
        of the array is changed. 
                
        If new is True a new instance is created having restricted data and 
        adjusted inset attributes. All other attributes are the same as in this 
        instance, and this instance is not modified. Otherwise modifies 
        self.inset and self.data.
        
        Can be used only on subclasses of this class that have data attribute.

        Argument:
          - image: (Image) 
          - intersect: flag indicating if the (agr) inset is to be intersected 
          with the current inset.
          - useFull: flag indicating if self.fullData is used          
          - expand: flag indicating if self.data array should be expanded if 
          needed
          - value: used to fill the expanded part of self.data
          - new: flag indicating if a new object that uses the given 
          positioning is generated 
        
        Returns: new instance if new is True. 
        """

        # copy this instance if new instance is requested
        if new:
            inst = deepcopy(self)
        else:
            inst = self

        # adjust image.inset for offsets
        inset_start = [image_sl.start + image_off - self_off 
                       for image_sl, image_off, self_off 
                       in zip(image.inset, image.offset, self.offset)]  
        adj_inset = [slice(st, st + ins.stop - ins.start) 
                     for st, ins in zip(inset_start, image.inset)]

        # use the adjusted inset
        inst.useInset(inset=adj_inset, mode=u'abs', intersect=intersect,
                      expand=expand, value=value)

        # return if new
        if new: return inst

    def copyPositioning(self, image, saveFull=False):
        """
        Depreciated

        Sets offset and inset attributes of this instance to the corresponding
        values of image. 

        Attributes self.fullInset and self.fullData are not set, unless 
        saveFull is True, in which case they are set to the current values of 
        inset and data.

        Argument:
          - image: (Image) image
          - saveFull: flag indication if fullInset and fullData are set
        """
        self.offset = deepcopy(image.offset)

        # don't need deepcopy cause int list -> slice list -> int list 
        self.inset = image.inset

        if saveFull:
            self.saveFull()

    def isInside(self, inset):
        """
        Returns True if arg inset lies completely inside self.inset.

        Note: Does not take offset into account

        Argument:
          - inset: absolute inset
        """

        # 
        res_list = [
            (ins.start >= self_ins.start) and (ins.stop <= self_ins.stop) 
            for self_ins, ins in zip(self.inset, inset)]
        res = numpy.asarray(res_list)

        return res.all()

    def hasOverlap(self, inset):
        """
        Returns True if arg inset self.inset have an overlap.

        Note: Does not take offset into account

        Argument:
          - inset: absolute inset
        """

        # 
        res_list = [
            (ins.stop > self_ins.start) and (ins.start < self_ins.stop) 
            for self_ins, ins in zip(self.inset, inset)]
        res = numpy.asarray(res_list)

        return res.all()

    def findEnclosingInset(self, inset, inset2=None):
        """
        Returns minimal inset that fully contains inset and inset2. If
        inset2 is None, self.inset is used instead.
        
        Argument:
          - inset, inset2: absolute insets

        Returns the absolute enclosing inset.
        """
        
        # parse args
        if inset2 is None:
            inset2 = self.inset

        # deal with no insets
        if inset2 is None:
            return inset
        elif inset is None:
            return inset2

        # both insets exist
        result = [
            slice(min(self_ins.start, ins.start), max(self_ins.stop, ins.stop)) 
            for self_ins, ins in zip(inset2, inset)]
        return result

    def findIntersectingInset(self, inset, inset2=None):
        """
        Returns intersection between inset and inset2. If
        inset2 is not specified, self.inset is used instead.

        If inset is None, returns inset2 if specified or self.inset.

        If inset2 and inset.data are None, returns inset.

        If insets do not intersett at all, returns None
        
        Argument:
          - inset, inset2: absolute insets

        Returns the absolute intersecting inset.
        """
        
        # parse args
        if inset2 is None:
            inset2 = self.inset

        # deal with no insets
        if inset2 is None:
            return inset
        elif inset is None:
            return inset2

        # both insets exist
        result = [
            slice(max(self_ins.start, ins.start), min(self_ins.stop, ins.stop)) 
            for self_ins, ins in zip(inset2, inset)]
        return result

    def newFromInset(self, inset, mode=u'relative', intersect=False, 
                     useFull=False, expand=False, value=0, copyData=True, 
                     deepcp=True, noDeepcp=[]):
        """
        Makes an instance that has values of all attributes the same as this 
        instance, except that the data array (attribute data) is set to  
        the data array inset defined by the arguments, and the inset
        attribute is set accordingly.

        The current instance is left unchanged (actaully changed and then
        brought back to its original state).

        If arg copyData is True, the new instance data array is a copy of the 
        data inset from the current instance. Otherwise the new instance data 
        is a view of the data array self.data (providing that arg inset fits
        within the current data array.

        Args deepcp and noDeepcp determine which attributes of the current
        instance have their values simply copied to the new instance (so that
        modifying one modifies other in case they are references to other 
        objects) and which are deepcopied (modifying one does not modify the 
        other). In any case, attribute inset is different in the two instances.

        See useInset() for the full description of the the role of other 
        arguments in detemining the inset.

        Arguments:
          - inset: list of slice objects that defines the inset
          - mode: 'relative' (same as 'rel') or 'absolute' (same as 'abs')
          - intersect: flag indicating if the (agr) inset is to be intersected 
          with the current inset.
          - useFull: flag indicating if self.fullData is used
          - expand: flag indicating if self.data array should be expanded if 
          needed
          - value: used to fill the expanded part of self.data
          - copyData: flag indicating if data array is coppied
          - deepcp: flag indicating if a deepcopy or just a copy of the 
          current instance is used to make the new instance
          - noDeepcp: list of attribute names that are not deepcopied 
          (used only if deepcp is True) 

        Returns the new instance.
        """

        # clean noDeepcp
        if deepcp:
            no_deepcp = [attr for attr in noDeepcp \
                               if getattr(self, attr, None) is not None] 

        # save initial data array and inset attribute
        initial_data_array = self.data
        initial_inset = self.inset
        if deepcp:
            initial_values = [getattr(self, attr) for attr in no_deepcp] 

        # make inset and copy the array inset
        self.useInset(inset=inset, mode=mode, intersect=intersect, 
                      useFull=useFull, expand=expand, value=value)
        if copyData:
            new_data_array = self.data.copy()
        else:
            new_data_array = self.data

        # make a copy/deepcopy of self
        self.data = None
        if deepcp:
            for attr in no_deepcp:
                setattr(self, attr, None)
            new_instance = deepcopy(self)
        else:
            new_instance = copy(self)

        # set data of new instance to the copy of data inset        
        new_instance.data = new_data_array
        if deepcp:
            for attr, value in zip(no_deepcp, initial_values):
                setattr(new_instance, attr, value)

        # bring self to the initial state (recover initial data and inset)
        self.data = initial_data_array
        self.inset = initial_inset
        if deepcp:
            for attr, value in zip(no_deepcp, initial_values):
                setattr(self, attr, value)
            
        return new_instance


    ########################################################
    #
    # Input / output
    #
    #########################################################

    @classmethod
    def read(cls, file, fileFormat=None, byteOrder=None, dataType=None, 
             arrayOrder=None, shape=None, header=False, memmap=False):
        """
        Reads image from a file and saves the data in numpy.array format.

        Images can be in em, mrc and raw format. Unless arg fileFormat is
        specified, file format is determined from the extension: 
          - 'em' and 'EM' for em format
          - 'mrc', 'rec' and 'mrcs' for mrc format
          - 'raw', 'dat' and 'RAW' for raw format

        If file is in em or mrc format (file extension em or mrc) only the file
        argument is needed. If the file is in the raw data format (file 
        extension raw) all arguments (except fileFormat) are required.

        If arg header is True, file header (em, mrc or raw) is saved as 
        attribute header. In any case attribute fileFormat is set.

        For mrc and em files, array order is determined from arg arrayOrder,
        from self.arrayOrder, or it is set to the default ("F") in 
        this order. Data is read according the determined array order.
        That is, array order is not read from the file header.

        If arg memmap is True, instead into a nparray, the data is read to
        a memory map. That means that the complete data is not read into
        the memory, but the required parts are read on demand. This is useful
        when working with large images, but might not always work properly 
        because the memory map is not quite properly a subclass of 
        numpy.ndarray (from Numpy doc).

        Data from mrc files is always read as 3D because mrc header always
        contains lengths for all three dimensions (the length of the last
        dimeension is 1 for 2D images). In such cases one can obtain the 
        2D ndarray using:  

          self.data.reshape(self.data.shape[0:2])

        Sets attributes:
          - data: (ndarray) image data
          - pixelsize: (float or list of floats) pixel size in nm, 1 if pixel 
          size not known
          - length: (list or ndarray) length in each dimension in nm
          - fileFormat: file format ('em', 'mrc', or 'raw')
          - memmap: from the argument
          - header: header

        Arguments:
          - file: file name
          - fileFormat: 'em', 'mrc', or 'raw'
          - byteOrder: '<' (little-endian), '>' (big-endian)
          - dataType: any of the numpy types, e.g.: 'int8', 'int16', 'int32',
            'float32', 'float64'
          - arrayOrder: 'C' (z-axis fastest), or 'F' (x-axis fastest)
          - shape: (x_dim, y_dim, z_dim)
          - header: flag indicating if file header is read and saved
          - memmap: Flag indicating if the data is read to a memory map,
          instead of reading it into a ndarray

        Returns:
          - instance of this class
        """

        # read file
        from pyto.io.image_io import ImageIO as ImageIO
        fi = ImageIO()
        fi.read(
            file=file, fileFormat=fileFormat, byteOrder=byteOrder, 
            dataType=dataType, arrayOrder=arrayOrder, shape=shape, 
            memmap=memmap)

        # instantiate object and set fullInset
        object = cls(data=fi.data)
        # is this needed (ugly)
        object.saveFull()

        # save file header
        object.header = None
        if header:
            object.header = fi.header
            try:
                object.extendedHeaderString = fi.extendedHeaderString
            except AttributeError:
                object.extendedHeaderString = None
        object.fileFormat = fi.fileFormat

        # save pixel size
        try:
            object.pixelsize = fi.pixelsize
        except (AttributeError, ValueError):
            object.pixelsize = 1

        # save length
        try:
            object.length = fi.length
        except (AttributeError, ValueError):
            object.length = numpy.asarray(fi.data.shape) \
                * numpy.asarray(object.pixelsize)

        # save memmap
        object.memmap = fi.memmap

        # save file io object
        # removed because if object.data is changed, object.file_io.data will
        # still hold a copy of data 
        #object.file_io = fi
        
        # save file type
        #try:
        #    object.fileType = fi.fileType
        #except (AttributeError, ValueError):
        #    object.fileType = None

        return object

    def write(
        self, file, byteOrder=None, dataType=None, fileFormat=None,
        arrayOrder=None, shape=None, length=None, pixel=1, casting=u'unsafe',
            header=False, existing=False):

        """
        Writes image to a file in em, mrc or raw format.

        If fileFormat is not given it is determined from the file extension.

        Data (image) has to be specified by arg data or previously set 
        self.data attribute.

        Data type and shape are determined by args dataType and shape, 
        or by the data type and shape of the data, in this order..

        If data type (determined as described above) is not one of the 
        data types used for the specified file format (ubyte, int16, float32, 
        complex64 for mrc and uint8, uint16, int32, float32, float64, 
        complex64 for em), then the value of arg dataType has to be one of the 
        appropriate data types. Otherwise an exception is raised.

        If data type (determined as described above) is different from the 
        type of actual data, the data is converted to the data type. Note that
        if these two types are incompatible according to arg casting, an 
        exception is raised. 

        The data is converted to the (prevously determined) shape and array
        order and then written. That means that the shape and the array order 
        may be changed in the original array (argument data or self.data).
        However, array order is not written the header. 

        Additional header parameters are determined for mrc format. Nxstart, 
        nystart and nzstart are set to 0, while mx, my and mz to the 
        corresponding data size (grid size). xlen, ylen and zlen are taken from
        arg length if given, or obtained by multiplying data size with pixel 
        size (in nm).

        If arg header is True and if the file type corresponding to arg file
        is the same as self.fileFormat, a header consisting of self.header 
        (list) and self.extendedHeaderString (str) are written as file 
        header. The extended header should be used only for mrc format.

        Arguments:
          - file: file name
          - fileFormat: 'em', 'mrc', or 'raw'
          - byteOrder: '<' (little-endian), '>' (big-endian)
          - dataType: any of the numpy types, e.g.: 'int8', 'int16', 'int32',
            'float32', 'float64'
          - arrayOrder: 'C' (z-axis fastest), or 'F' (x-axis fastest)
          - shape: (x_dim, y_dim, z_dim)
          - length: (list aor ndarray) length in each dimension in nm (used 
          only for mrc format)
          - pixel: pixel size in nm (used only for mrc format if length is 
          None)
          - casting: Controls what kind of data casting may occur: 'no', 
          'equiv', 'safe', 'same_kind', 'unsafe'. Identical to numpy.astype()
          method.
          - header: flag indicating if self.header is written as file header
          - existing: flag indicating whether the already existing file_io
          attribute is used for writting

        Returns file instance.
        """

        # write file
        from pyto.io.image_io import ImageIO as ImageIO
        if existing and (self.file_io is not None):
            fi = self.file_io
        fi = ImageIO()

        # get file format of the file to be written
        fi.setFileFormat(file_=file, fileFormat=fileFormat)
        new_file_format = fi.fileFormat

        # set header to pass as argument, if new and old file types are the same
        header_arg = None
        extended = None
        try:
            if header and (new_file_format == self.fileFormat):
                header_arg = self.header
                extended = self.extendedHeaderString
        except AttributeError:
            pass

        # write
        fi.write(
            file=file, data=self.data, fileFormat=fileFormat, 
            byteOrder=byteOrder, dataType=dataType, arrayOrder=arrayOrder, 
            shape=shape, length=length, pixel=pixel, casting=casting, 
            header=header_arg, extended=extended)

        return fi.file_

    @classmethod
    def modify(cls, old, new, fun, fun_kwargs={}, memmap=True):
        """
        Reads an image (arg old), modifies old.data using function
        passed as arg fun and writes an image containing the modified
        data as a new image (arg new).

        The function passes (arg fun) has to have signature
          fun(Image, **fun_kwargs)
        and to return image data (ndarray).

        Meant for mrc files. The new image will have exactly the same 
        header as the old image, except for the shape, length and 
        min/max/mean values, which are set according to the new image 
        data.

        Also works if old is an mrc and new is a raw file.  
        
        Arguments:
          - old: old mrc image file name
          - new: new (subtomogram) mrc image file name
          - fun: function that takes old.data as an argument 
          - memmap: if True, read memory map instead of the whole image

        Returns an instance of this class that holds the new image. This
        instance contains attribute image_io (pyto.io.ImageIO) that
        was used to write the new file.
 
        """
        # read header
        from pyto.io.image_io import ImageIO as ImageIO
        image_io = ImageIO()
        image_io.readHeader(file=old)

        # read image 
        image = cls.read(file=old, memmap=memmap)

        # modify data
        data = fun(image, **fun_kwargs)

        # write new (modified data)
        image_io.setData(data=data)
        image_io.write(file=new, pixel=image_io.pixel)
        #image_io.setFileFormat(file_=new)
        #if image_io.fileFormat == 'mrc':
        #    image_io.write(file=new, pixel=image_io.pixel)
        #elif image_io.fileFormat == 'raw':
        #    image_io.write(file=new, pixel=image_io.pixel)
            
        #
        image.image_io = image_io
        return image
             
    @classmethod
    def cut(cls, old, new, inset, memmap=True):
        """
        Reads an image (arg old), cuts a subtomo defined by arg inset
        and writes the new image (arg new).

        Meant for mrc files. The new image will have exactly the same 
        header as the old image, except for the shape, length and 
        min/max/mean values, which are set according to the new image 
        data.

        Arguments:
          - old: old mrc image file name
          - new: new (subtomogram) mrc image file name
          - inset: defines the subtomogram
          - memmap: if True, read memory map instead of the whole image

        Returns an instance of this class that holds the new image. This
        instance contains attribute image_io (pyto.io.ImageIO) that
        was used to write the new file.
        """

        # read header
        from pyto.io.image_io import ImageIO as ImageIO
        image_io = ImageIO()
        image_io.readHeader(file=old)

        # read image and cut inset
        image = cls.read(file=old, memmap=memmap)
        image.useInset(inset=inset)

        # write new
        image_io.setData(data=image.data)
        image_io.write(file=new, pixel=image_io.pixel)

        #
        image.image_io = image_io
        return image
        
