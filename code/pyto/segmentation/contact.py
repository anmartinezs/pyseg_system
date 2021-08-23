"""
Contains class Contact for analysis of contacts between segments (from a 
segmented image) and boundaries specified on the same image.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import zip
from builtins import range
from builtins import object

__version__ = "$Revision$"


from copy import copy, deepcopy
import logging
import numpy
try:
    import numpy.ma as ma
except ImportError:
    import numpy.core.ma as ma  # numpy <= 1.0.4
#import numpy.ma as ma
import scipy
import scipy.ndimage as ndimage

# not good here because it makes circular import
#from segment import Segment
import pyto.util.nested as util_nested
import pyto.util.numpy_plus as numpy_plus

class Contact(object):
    """
    Analysis of contacts between segments and boundaries.

    Important methods:
      - findContacts: detects and optionally counts contacts between
      segments and boundaries (needs to be run before the methods given below)
      - findSegments: finds segments that satisfy given condition (such as
      number of contacted boundaries) among all contacts previously detected
      by findContacts
      - findBoundaries: finds boundaries that satisfy given condition (such as
      number of contacted segments) among all contacts previously detected
      by findContacts

    The following example finds all contancts between segments 2,3,5 and 7
    with boundaries 1,3,4 and 6, and then returns ids of segments that
    contact both boundary 1 and 4 but do not contact any other boundary:

      con = Contact()  # instantiation
      con.findContacts(segment=seg, segmentIds=[2,3,5,7], boundary=bound,
                       boundaryIds=[1,3,4,6])
      con.findSegments(boundaryIds=[1,4], mode='exact')

    Important instance attributes:
      - labels: Connected object containing an image of contacts labeled
      by segment ids
      - segment: (ndarray) segmented image containing segments
      - segmentIds: ids of segments
      - boundary: (ndarray) segmented image containing boundaries
      - boundaryIds: ids of boundaries
      - contactStructEl: structure element used for finding contacts between
      boundaries and structures
      - countStructEl: structure element used to distinguish and count
      conntacts between a given segment and a given region.

    Internal structures (should not be manipulated directly):
      - _n: contact data structure currently implemented as a masked array
      where _n[boundId, segId] is a number of contacts segment segId makes
      with boundary boundId. Shape = (nBoundaries+1, nSegments+1), so that
      boundary (segment) i as accessed as _n[i,:] (_n[;,i]). Data with
      index 0 (any) are not used.
    """

    ##################################################################
    #
    # Initialization of data structures and other attributes
    #
    ##################################################################
        
    def __init__(self, nBoundaries=0, nSegments=0, maskValue=True):
        """
        Initializes data structure that holds number of contacts between
        given segments and boundaries, as well as some attributes.

        Elements of maskValue that are True are masked (not available),
        following the scheme of numpy.ma. Regardless of the value of
        maskValue argument given, mask for all elements with index 0 is set.
        """

        # initialize attributes
        self.offset = None
        self.segment = None
        self.segmentIds = None
        self.boundary = None
        self.boundaryIds = None
        self.contactStructEl = None
        self.countStructEl = None
        
        self.compactified = False

        self.fillValue = -1
        self._initializeData(shape=(nBoundaries+1, nSegments+1),
                             maskValue=maskValue)

        # required by numpy.ndimage.label()
        #self.labelOutDtype = 'int32'
        # perhaps not anymore (scipy 0.11)
        self.labelOutDtype = 'int64'

    @classmethod
    def recast(cls, obj):
        """
        Returns a new instance of this class and sets all attributes of the new 
        object to the attributes of obj.

        Useful for objects of this class loaded from a pickle that were pickled
        before some of the current methods of this class were coded. In such a 
        case the returned object would contain the same attributes as the 
        unpickled object but it would accept recently coded methods. 
        """

        # make a new instance
        new = cls()

        # set data
        new._n = obj._n

        # set other attributes
        new.offset = obj.offset
        new.segment = obj.segment
        new.segmentIds = obj.segmentIds
        new.boundary = obj.boundary
        new.boundaryIds = obj.boundaryIds
        new.contactStructEl = obj.contactStructEl
        new.countStructEl = obj.countStructEl
        new.compactified = obj.compactified
        new.fillValue = obj.fillValue

        # required by numpy.ndimage.label()
        new.labelOutType = obj.labelOutDtype        
     
        return new

    def _initializeData(self, shape=(1,1), maskValue=True):
        """
        Initializes data structure that holds number of contacts between
        given segments and boundaries.

        Elements of maskValue that are True are masked (not available),
        following the scheme of numpy.ma. Regardless of the value of
        maskValue argument given, mask for all elements with index 0 is set.
        """

        # make the data structure
        data = numpy.zeros(shape=shape, dtype=int)
        if maskValue: maskValue = 1
        else: maskValue = 0
        self._n = ma.array(data=data, mask=maskValue,
                           fill_value=self.fillValue)

        # make sure mask is set for all elements with index 0
        self._n._mask[0,:] = True
        self._n._mask[:,0] = True

    def _enlargeData(self, shape, maskValue=True):
        """
        Enlarges the size of the data structure.

        The shape of the new structure is the max in each dimension between
        shape and the shape of the current (old) data shape. Mask is set
        for all newly added elements with index 0. 
        
        """
        
        # find max size in each dimension
        newShape = [max(old,new) for (old,new) in zip(self._n.shape, shape)]

        # make new data structure
        oldData = self._n._data
        oldMask = self._n._mask
        self._initializeData(shape=newShape, maskValue=maskValue)

        # make sure mask is set for all elements with index 0
        self._n._mask[0,:] = True
        self._n._mask[:,0] = True

        # copy the old data and mask separately (preserves data values that
        # are masked in numpy 1.0.1) 
        self._n._data[0:oldData.shape[0], 0:oldData.shape[1]] = oldData[:,:]
        self._n._mask[0:oldData.shape[0], 0:oldData.shape[1]] = oldMask[:,:]


    ##################################################################
    #
    # Basic manipulation of the contact data structure
    #
    ##################################################################

    def getSegments(self):
        """
        Returns (ndarray) segment ids of all segments that have at least one 
        unmasked element.

        Note: perhaps this should be changed to return ids of segments that 
        contact at lease one boundary.
        """
        # make a mask
        shifted = ~numpy.bitwise_and.reduce(self._n._mask[1:,1:], 0)

        # convert to segment ids
        return numpy.nonzero(shifted)[0] + 1

    segments = property(fget=getSegments, doc="Ids of all segments")
        
    def getBoundaries(self):
        """
        Returns (ndarray) boundary ids of all boundaries that have at least 
        one unmasked element.

        Note: perhaps this should be changed to return ids of boundaries that 
        contact at lease one segment.
        """
        # make a mask
        shifted = ~numpy.bitwise_and.reduce(self._n._mask[1:,1:], 1)

        # convert to segment ids
        return numpy.nonzero(shifted)[0] + 1
        
    boundaries = property(fget=getBoundaries, doc="Ids of all boundaries")
        
    def getMaxSegment(self):
        """
        Returns the highest segment id.
        """
        return self._n.shape[1]-1

    maxSegment = property(fget=getMaxSegment, doc='Max segment id')

    def getMaxBoundary(self):
        """
        Returns the highest boundary id.
        """
        return self._n.shape[0]-1

    maxBoundary = property(fget=getMaxBoundary, doc='Max boundary id')

    def add(self, contacts, shift=0):
        """
        Adds contacts to the current contacts. Ids of segments in (argument)
        contacts are shifted by (argument) shift.

        Argumens:
          - contacts: instance of Contacts
          - shift: all segment ids in contacts are increased by this amount
        """

        # shift new contacts
        contacts = deepcopy(contacts)
        contacts.shiftSegmentIds(shift)

        # enlarge current data structure
        max_seg = max(self.maxSegment, contacts.maxSegment)
        max_bound = max(self.maxBoundary, contacts.maxBoundary)
        self._enlargeData(shape=(max_bound+1, max_seg+1))

        # add shifted contacts
        new_seg_ids = contacts.segments
        if len(numpy.intersect1d(self.segments, new_seg_ids)) > 0:
            logging.warning("Ids of the segments that are added collide with"\
                           + " with the current segments.")
        self._n[:, new_seg_ids] = contacts._n[:, new_seg_ids]

    def addBoundary(self, id, nContacts):
        """
        Adds contact data given in nContacts for boundary with id.

        Argument nContacts has to be a list (or a ndarray) indexed by segment
        ids. That is, nContacts[3] is the number of contacts this boundary makes
        with segment with ids 3 (nContacts[0] is not used).

        If the length of nContacts is larger than the max_segmentId+1 the
        max segmentId (as returned by getMaxSegment() is increased
        to hold all segments. If it is smaller, the missing nContact
        elements are set to 0.

        Arguments:
          - id: boundary id whose contacts are added
          - nContacts: list or ndarray containing number of contacts for each
          segment (indexed by segmentIds)
        """

        # add nContacts
        try: 
            self._n[id, 0:len(nContacts)] = nContacts

        except (IndexError, ValueError):

            # enlarge _n 
            newShape = (id+1, len(nContacts)) 
            self._enlargeData(shape=newShape)

            # put data
            self._n[id, 0:len(nContacts)] = nContacts

        # set mask at 0 and pad the remainer with 0's
        self._n._mask[id, 0] = True
        self._n[id, len(nContacts):] = 0

    def addSegment(self, id, nContacts):
        """
        Adds contact data given in nContacts for segment with id.

        Argument nContacts has to be a list (or a ndarray) indexed by boundary
        ids. That is, nContacts[3] is the number of contacts this segment makes
        with boundary with ids 3 (nContacts[0] is not used).

        If the length of nContacts is larger than the max_boundaryId+1 the
        max boundaryId (as returned by getMaxBoundary() is increased
        to hold all boundaries. If it is smaller, the missing nContact
        elements are set to 0.

        Arguments:
          - id: segment id whose contacts are added
          - nContacts: list or ndarray containing number of contacts for each
          boundary (indexed by boundaryIds)
        """

        # add nContacts
        try: 
            self._n[0:len(nContacts), id] = nContacts

        except (IndexError, ValueError):

            # enlarge _n if needed
            newShape = (len(nContacts), id+1)
            self._enlargeData(shape=newShape)

            # put data
            self._n[0:len(nContacts), id] = nContacts

        # set mask at 0 and pad the remainer with 0's
        self._n._mask[0, id] = True
        self._n[len(nContacts):, id] = 0

    def setN(self, boundaryId=None, segmentId=None, nContacts=None):
        """
        Adds specified contacts to the contact data structure.
        
        The number of contacts nContacts, between bounday boundaryId and
        segment segmentId is added to the contact data structure.

        Arguments:
          - boundaryId: a boundary
          - segmentId: a segment
          - nContacts: number of contacts between the boundary and the segment
        """

        try:
            self._n[boundaryId, segmentId] = nContacts

        except IndexError:

            # enlarge data structure first
            newShape = (max(boundaryId, self.getMaxBoundary()) + 1,
                        max(segmentId, self.getMaxSegment()) + 1)
            self._enlargeData(shape=newShape)
            self._n[boundaryId, segmentId] = nContacts
            
    def getN(self, boundaryId=None, segmentId=None):
        """
        Returns number of contacts between segment segmentId and boundary
        boundaryId.

        If boundaryId (segmentId) is None, number of contacts of the
        segmentId (boundaryId) with all boundaries (segments) is returned.
        The array returned is indexed by boundaryId (segmentId), so the
        element at position 0 has no meaning.

        At least one of the arguments need to be supplied.

        Returns a single integer or an array containig number(s) of contacts.
        """

        if boundaryId is None: 
            if segmentId not in self.getSegments():
                return []
            boundaryId = slice(0,None)
        if segmentId is None: 
            if boundaryId not in self.getBoundaries():
                return []
            segmentId = slice(0,None)
        
        return self._n[boundaryId, segmentId]

    def removeBoundaries(self, ids):
        """
        Removes all contacts of boundaries with ids.

        Boundaries with ids are only masked (the contact data is not
        removed), so they can be recovered if needed.
        """
        if (not isinstance(ids, list)) and (not isinstance(ids, tuple)) \
               and (not isinstance(ids, numpy.ndarray)):
            ids = [ids]
        for id in ids: self._n._mask[id, 1:] = True

    def keepBoundaries(self, ids):
        """
        Removes all contacts of boundaries whose ids are not in ids.

        Boundaries with ids are only masked (the contact data is not
        removed), so they can be recovered if needed.
        """
        if (not isinstance(ids, list)) and (not isinstance(ids, tuple)) \
               and (not isinstance(ids, numpy.ndarray)):
            ids = [ids]
        all = numpy.arange(1, self.getMaxBoundary()+1)
        self.removeBoundaries( numpy.setdiff1d(all, ids) )

    def removeSegments(self, ids):
        """
        Removes all contacts of segments with ids.

        Segments with ids are only masked (the contact data is not
        removed), so they can be recovered if needed.

        Note: if it's needed to reduce the size of the contact data structure
        (self._n) use reorderSegments().
        """

        # Note: Removing the segments completely from self._n array does
        # not make much sense because the size of self._n is determined
        # by highest id (and not by the number of segements)

        if (not isinstance(ids, list)) and (not isinstance(ids, tuple)) \
               and (not isinstance(ids, numpy.ndarray)):
            ids = [ids]
        for id in ids: self._n._mask[1:, id] = True

    def keepSegments(self, ids):
        """
        Removes all contacts of segments whose ids are not in ids.

        Segments with ids are only masked (the contact data is not
        removed), so they can be recovered if needed.
        """
        if (not isinstance(ids, list)) and (not isinstance(ids, tuple)) \
               and (not isinstance(ids, numpy.ndarray)):
            ids = [ids]
        all = numpy.arange(1, self.getMaxSegment()+1)
        self.removeSegments( numpy.setdiff1d(all, ids) )

    def removeContact(self, segmentId, boundaryId):
        """
        Removes contact between boundaryId and segmentID.

        The contact is only masked (the contact data is not removed), so
        it can be recovered if needed.
        """
        self._n._mask[boundaryId, segmentId] = True

    def _recoverBoundary(self, boundaryId):
        """
        Recovers previously removed boundary.
        """
        self._n._mask[boundaryId, 1:] = False
    
    def _recoverSegment(self, segmentId):
        """
        Recovers previously removed segment.
        """
        self._n._mask[1:, segmentId] = False
    
    def _recoverContact(self, segmentId, boundaryId):
        """
        Recoveres previously removed contact between boundaryId and segmentID.
        """
        self._n._mask[boundaryId, segmentId] = False

    def reorderSegments(self, order):
        """
        Reorders segments in both contact data structure (self._n) and contact
        labels (self.labels) according to the order dictionary.

        Only the elements corresponding to the new ids (and 0) are set. Elements
        corresponding to old ids that are not given in order are removed. 

        Argument:
          - order: dictionary where keys are old ids and values are new ids.
        """

        # make new array
        if len(order) > 0:
            new_shape = (self.getMaxBoundary()+1, max(order.values())+1)
        else:
            new_shape = (self.getMaxBoundary()+1, 1)
        new_data = numpy.zeros(shape=new_shape, dtype=int)
        new = ma.array(data=new_data, mask=True, fill_value=self.fillValue)

        # reorder contact data structure (self._n)
        for old in order:
            new[:,order[old]] = self._n[:,old]
        self._n = new

        # reorder contact array if it exists
        try:
            self.labels.reorder(order=order, clean=True)
        except AttributeError:
            pass

    def shiftSegmentIds(self, shift):
        """
        Increases all segment ids by shift in both contacts data structure and
        contacts labels.
        """
        
        if shift is not None:

            # make old-new ids dictionary
            old = numpy.array(list(range(1, self.getMaxSegment()+1)))
            new = old + shift
            order = dict(list(zip(old, new)))

            # reorder
            self.reorderSegments(order)

    def orderSegmentsByContactedBoundaries(self, argsort=False):
        """
        There are two modes. If argsort is False, orders segments according
        to the boundaries contacted by the segments (changes the contacts
        data structure). If argsort is True, behaves like ndarray.argsort,
        that is the segments are not reordered, but an array that can be used
        to order them is returned.

        Only the existing (not masked) segments are reordered.

        For example, if:

          segment 1 contacts boundaries [2,3]
          segment 3 contacts [1,2,5]
          segment 4 contacts [1,4]

        then the segment ids are rearanged as follows:

          3 -> 1
          4 -> 3
          1 -> 4

        and the dictionary {1:4, 3:1, 4:3} is returned if argsort is False.
        Otherwise [2, 0, 1] is returned.

        Argument:
          - argsort: if True the segments are not reordered, only the array
          that can be used for sorting is returned, similar to what
          ndarray.argsort does.  

        Returns:
          - if argsort is False: dictionary where keys are old ids and values
          are new ids.
          - if argsort is True: array that can be used for sorting (like
          ndarray.argsort)
        """

        # make array where each element contains ids of boundaries that contact
        # segment whose id is given at the same positin in self.segments
        s_ids = self.segments
        b_ids = numpy.zeros(len(s_ids), dtype='object_')
        for (index, id) in zip(list(range(len(s_ids))), s_ids):
            b_ids[index] = self.findBoundaries(segmentIds=[id],
                                               update=False).tolist()

        # make sort array
        sort_list = b_ids.argsort()

        if argsort:
            return sort_list

        else:

            # reorder segments in self._n
            new_order = dict(list(zip(s_ids[sort_list], s_ids)))
            self.reorderSegments(new_order)            
            return new_order


    ##################################################################
    #
    # Converting data to/from the compact form 
    #
    #################################################################

    def compactify(self):
        """
        Saves data structure in a compact form. This is only useful for storing
        an object, because all methods that require access to the data 
        structure need the data in the expanded form. The data can be
        expanded by self.expand().

        Sets:
          - self.compactified: flag indicating if data is in the compact form
          - self._boundaries: boundary ids (as given by self.boundaries)
          - self._segments: boundary ids (as given by self.segments)
          - self._n: (expanded data) set to None
          - self._compact: data in the compact form
        """

        # asign values
        self._boundaries = self.boundaries
        self._segments = self.segments
        self._compact = self._n[numpy.ix_(self._boundaries, self._segments)]

        # finish and clean up
        self.compactified = True
        self._n = None

    def expand(self):
        """
        Expands data structure form the compact (using self.compactify()) to 
        the normal (expanded) form. If self.compactified is False, or if it 
        doesn't exist, this method doesn't do anything. 

        Sets:
          - self.compactified: flag indicating if data is in the compact form
          - self._boundaries: to None
          - self._segments: to None
          - self._n: expanded data
          - self._compact: (data in the compact form) to None
        """
        
        # check if compactified exists and if it's True
        if getattr(self, 'compactified', False):

            # create new self._n
            if (self._boundaries is not None) and (len(self._boundaries) > 0):
                bound_max = self._boundaries.max()
            else:
                bound_max = 0
            if (self._segments is not None) and (len(self._segments) > 0):
                seg_max = self._segments.max()
            else:
                seg_max = 0
            self._initializeData(shape=(bound_max + 1, seg_max + 1))
            
            # assign values
            self._n[numpy.ix_(self._boundaries, self._segments)] = self._compact

            # finish and clean up
            self.compactified = False
            self._compact = None
            self._boundaries = None
            self._segments = None


    ##################################################################
    #
    # Finding contacts
    #
    #################################################################

    def findContacts(self, segment, segmentIds=None, boundary=None,
                  boundaryIds=None, contactStructEl=None, saveLabels=False, 
                  countStructEl=None, count=False):
        """
        Finds (but doesn't count) contacts between each segment and each
        boundary.

        A contact is comosed of mutually connected elements of segment that
        lie in the neighborhood of a boundary.

        For each contact found between a segment and a boundary 1 is entered
        to the data structure that holds number of contacts. Flag
        self.counted is set to False to indicate that the contact were not
        counted.

        If saveLabels is true, the contacts are labeled by segments and the
        contacts image (labels) is saved in (Connected object) self.labels.
                
        Attribute self.segment is set to arg segment.

        Attribute self.segmentIds is set to the first value found in:
        segmentIds, segment.segmentIds, self.segmentIds.

        Attribute self.boundary is set to the first value found in: boundary,
        self.boundary.

        Attribute self.boundaryIds is set to the first value found in:
        boundaryIds, boundary.boundaryIds, self.boundaryIds.

        Note: does not respect inset of segment and boundary objects.

        Arguments:
          - segment: instance of class Segment defining all segments, Required
          unless self.segment is already set
          - segmentIds: list (or an int) of segment ids
          - boundary: instance of class Segment defining boundaries
          - boundaryIds: list (or an int) of boundary ids
          - contactStructEl: structure element used to find contacts (default
          connectivity = 1, or 6 neighbors in 3D )
          - saveLabels: flag indicating if an array of contacts is saved
          - countStructEl: structure element used to count contacts (default
          connectivity = ndim, or 26 neighbors in 3D )
          - count: flag indicating if contacts are also counted (or just
          detected)
        """

        # ther might be a better place for this somewhere else
        #from segment import Segment

        # extract segment, segmentIds
        self.setSegment(segment=segment, ids=segmentIds)

        # extract boundary, boundaryIds
        self.setBoundary(boundary=boundary, ids=boundaryIds)

        # return if no segmentIds or no boundaryIds
        to_return = False
        if (self.segmentIds is None) or (len(self.segmentIds) == 0):
            max_segment_id = 0 
            to_return = True
        else:
            max_segment_id = max(self.segmentIds)
        if (self.boundaryIds is None) or (len(self.boundaryIds) == 0):
            max_boundary_id = 0
            to_return = True
        else:
            max_boundary_id = max(self.boundaryIds)

        # make sure that self._n has proper shape (even if one does not exist)
        newShape = (max_boundary_id+1, max_segment_id+1)
        self._enlargeData(shape=newShape, maskValue=True)

        # return if no segments or no boundaries
        if to_return: return

        # set structure elements
        self.setStructEls(segment=self.segment, contactStructEl=contactStructEl,
                          countStructEl=countStructEl)

        # find contacts
        # Note: it's not good to use grey_dilation (and avoid this for loop)
        # because dilated boundaries overlap if they are close (distance = 2)
        if saveLabels:
            contacts = numpy.zeros_like(self.segment)
        for bound in self.boundaryIds:
        
            # dilate the current boundary and intersect with all segments
            dilated_bound = ndimage.binary_dilation(self.boundary==bound,
                                         structure=self.contactStructEl)
            curr_contacts = numpy.where(dilated_bound, self.segment, 0)

            # put contacts into array
            # Note: labeling by boundary id is wrong for close boundaries 
            if saveLabels:
                contacts += curr_contacts 
 
            # find segments with contacts
            segs = numpy.unique(curr_contacts)

            # count or just find contacts
            conts = numpy.zeros(self.getMaxSegment()+1, dtype='int_')
            if count:

                # count number of contacts between current boundary and each 
                # segment
                if curr_contacts.max() > 0:
                    insets = ndimage.find_objects(curr_contacts)
                    for seg_id in segs:
                        seg_inset = insets[seg_id-1]
                        tmp, n_contacts = ndimage.label(
                            curr_contacts[seg_inset]==seg_id,
                            structure=self.countStructEl)
                        conts[seg_id] = n_contacts

            else:

                # don't count, just put 1
                conts[segs] = 1

            # add contacts to the contacts data structure 
            self.addBoundary(id=bound, nContacts=conts)

        # contacts found but not counted
        #self.counted = False

        # make Connected instance from contacts
        if saveLabels:
            from .connected import Connected
            labels = Connected(data=contacts, copy=False)
            labels.copyPositioning(image=segment, saveFull=True)
            #labels.makeInset()  problem if no contacts
            self.labels = labels

    def countContacts(self, labels=None, contactStructEl=None):
        """
        Work in progress.
        
        Counts number of contacts.

        Use findCOntacts instead?
        """

        # parse arguments
        if labels is None: labels = self.labels
        if contactStructEl is None: contactStructEl = self.contactStructEl

        self.counted = True

        raise NotImplementedError("Sorry, not finished yet")
        
        # count number of contacts for each segment and put in nContacts
        for seg in self.segmentIds:
            labels, nCont = ndimage.label(contacts==seg,
                                          structure=self.countStructEl)
            self.setN(boundaryId=bound, segmentId=seg, nContacts=nCont) 


    def setSegment(self, segment, ids=None):
        """
        Sets attributes (ndarray) segment, segmentIds from arguments.

        Attribute self.segmentIds is set to argument ids if specified (no type
        change), otherwise to segment.ids. If segment.ids is also None, the
        current self.segmentIds is retained.

        Agruments:
          - segment: instance of Segment or ndarray
          - ids: iterable (ndarray, flat or nested list) of segment ids. 
        """

        # use ids argument if given 
        if ids is not None:
            if (not isinstance(ids, list)) and (not isinstance(ids, tuple)) \
                   and (not isinstance(ids, numpy.ndarray)):
                ids = [ids]
            self.segmentIds = ids

        # ther might be a better place for this somewhere else
        #import pyto.segmentation.segment

        # set self.segment 
        #if isinstance(segment, segment.Segment):
        if isinstance(segment, numpy.ndarray):

            # segment is an array
            self.segment = segment

        else:

            # segment is an instance of Segment
            self.segment = segment.data
            if ids is None:
                self.segmentIds = segment.ids

    def setBoundary(self, boundary, ids=None):
        """
        Sets attributes (ndarray) boundary, boundaryIds from arguments.

        Attribute self.boundaryIds is set to argument ids if specified (no type
        change), otherwise to boundary.ids. If boundary.ids is also
        None, the current self.boundaryIds is retained.

        Agruments:
          - boundary: instance of Segment or ndarray defining boundaries
          - ids: iterable (ndarray, flat or nested list) of boundary ids. 
        """

        # use ids argument if given 
        if ids is not None:
            if (not isinstance(ids, list)) and (not isinstance(ids, tuple)) \
                   and (not isinstance(ids, numpy.ndarray)):
                ids = [ids]
            self.boundaryIds = ids

        # ther might be a better place for this somewhere else
        from .segment import Segment

        # set self.boundary, self.boundaryIds
        if isinstance(boundary, Segment):

            # boundary is an instance of Segment
            self.boundary = boundary.data
            if ids is None:
                self.boundaryIds = boundary.ids
        else:

            # boundary is an array
            self.boundary = boundary

    def setStructEls(self, segment, contactStructEl=None, countStructEl=None):
        """
        By default contactStructEl has connectivity 1, and countStructEl
        has connectivity equal to ndim.

        Sets self.contactStructEl and countStructEl.

        Arguments:
          - segment: (ndarray) segment, used only to get ndim
          - contactStructEl: structuring element used for contact detetion 
          - countStructEl: structuring element used for counting contacts
        """

        # get dimensionality of segment array
        ndim = segment.ndim

        # structuring element for contact detection
        if contactStructEl is not None:
            self.contactStructEl = contactStructEl
        if self.contactStructEl is None:
            self.contactStructEl = \
                ndimage.generate_binary_structure(rank=ndim, connectivity=1)

        # structuring element for contact counting
        if countStructEl is not None:
            self.countStructEl = countStructEl
        if self.countStructEl is None:
            self.countStructEl = \
                ndimage.generate_binary_structure(rank=ndim, connectivity=ndim)


    ##################################################################
    #
    # Analysis of contact data structure
    #
    ##################################################################

    def findSegments(self, boundaryIds=None, nBoundary=None, mode='at_least',
                    update=False):
        """
        Returns segment ids for all segements that satisfy the criterium
        specified by boundaryIds, nBoundary and mode arguments.

          1) If boundaryIds is given, but nBoundary isn't, segments that contact
          all boundaries specified as elements of boundaryIds are selected.

          2) If boundaryIds isn't given, but nBoundary is, segments that contact
          nBoundary boundaries are selected.

          3) If both boundaryIds and nBoundary are given, segments that contact
          nBoundary of the boundaries specified as elements of boundaryIds are
          selected.

          4) If both boundaryIds and nBoundary are None, ids of all segments
          are returned

        Each of the above conditions is modified by mode. Mode='exact' means
        that the segments whose ids are returned satisfy the condition exactly.
        That is, these segments contact all boundaries in boundaryIds and no
        other boundary, or exactly nBoundary boundaries. In mode='at_least'
        ('at_most') all the segments contact at least (at most) all boundaries
        with boundaryIds but may contact other boundaries also, or the segments
        contact nBoundary or more boundaries.

        Example:

          findSegments(boundaryIds=[1,3,6], nBoundary=2, mode='exact')

        returns ids of segments that contact only two of the boundaries 1,3 or 6
        and do not contact any other boundary.

        Furthemore, boundaryIds can be a nested list. In that case, boundary ids
        that are given in a sub-list are taken in the 'or' sense, as if those
        boundaries were merged together. 
    
        Another example:

          findSegments(boundaryIds=[[1,2],4], nBoundary=2, mode='exact')

        returns ids of segments that contact boundary 4 and any (or both) of
        1 and 2 and none of the other boundaries.

        If update=True the other segments are removed from the contacts data
        structure.

        Arguments:
          - boundaryIds: (nested) list or ndarray of boundary ids that are
          checked for contact with segments. Can be given as int if only one id.
          - nBoundary: (int) number of contacted boundaries
          - mode: 'exact' (or 'eq'), 'at_least' (or 'ge'), or 'at_most' (or 'le')
          - update: flag indicating if the contact data structure is updated

        Returns: ndarray of segments that satisfy the condition.
        """

        # no condition
        if (boundaryIds is None) and (nBoundary is None):
            return self.segments

        # if boundaryIds is a number convert to a list
        if boundaryIds is not None:
            if not isinstance(boundaryIds, (list, tuple, numpy.ndarray)):
                boundaryIds = [boundaryIds]

        # required no. of contacted boundaries that are in boundaryIds
        if nBoundary is None:
            nb = len(boundaryIds)
        else:
            nb = nBoundary

        # number of boundaries among boundaryIds contacted by each segments
        if boundaryIds is not None:
            selected = self.countContactedBoundaries(ids=boundaryIds)
        elif nBoundary is not None:
            selected = self.countContactedBoundaries()
        #else:
        #    raise ValueError, """At least one of boundaryIds and nBoundary
        #    arguments have to be given."""

        # find good segments for both modes 
        if (mode == 'exact') or (mode == 'eq'):
            good = numpy.where((selected._data==nb) \
                                   & (selected._mask==False))[0] 

        elif (mode == 'at_least') or (mode == 'ge'):
            good = numpy.where((selected._data>=nb) \
                                   & (selected._mask==False))[0]
            
        elif (mode == 'at_most') or (mode == 'le'):
            good = numpy.where((selected._data<=nb) \
                                   & (selected._mask==False))[0]
            
        else:
            raise ValueError("Mode argument " + mode + \
                  " is not recognized. Available values are: " + \
                  "'exact' (same as 'eq'), 'at_least' (same as 'ge') " + \
                  "and 'at_most' (same as 'le').")  

        # remove segments that do not satisfy the conditions, if required
        good = numpy.asarray(good, dtype='int')
        if update: self.keepSegments(ids=good)

        return good

    def findBoundaries(self, segmentIds=None, nSegment=None, mode='at_least',
                    update=False):
        """
        Returns boundary ids for all segements that satisfy the criterium
        specified by segmentIds, nSegment and mode arguments.

          1) If segmentIds is given, but nSegment isn't, boundaries that contact
          all segments specified as elements of segmentIds are selected.

          2) If segmentIds isn't given, but nSegment is, boundaries that contact
          nSegment segments are selected.

          3) If both segmentIds and nSegment are given, boundaries that contact
          nSegment of the segments specified as elements of  segmentIds are
          selected.

          4) If both segmentIds and nSegment are None, all boundary ids
          are returned.
    
        Each of the above conditions is modeified by mode. Mode='exact' means
        that the boundaries whose ids are returned satisfy the condition 
        exactly. That is, these boundaries contact all segments in segmentIds 
        and no other segment, or exactly nSegment segments. In mode='at_least'
        ('at_most') all the boundaries contact at least (at most) all segments
        with segmentIds but may contact other segments also, or the boundaries
        contact nSegment or more segments.

        Example:

          findBoundaries(segmentIds=[1,3,6], nSegment=2, mode='exact')

        returns ids of boundaries that contact only two of the segments 1,3 or 6
        and do not contact any other segment.

        Furthemore, segmentIds can be a nested list. In that case, segment ids
        that are given in a sub-list are taken in the 'or' sense, as if those
        segments were merged together. 
    
        Another example:

          findBoundaries(segmentIds=[[1,2],4], nSegment=2, mode='exact')

        returns ids of boundaries that contact segment 4 and any (or both) of
        1 and 2 and none of the other segments.

        If update=True the other boundaries are removed from the contacts data
        structure.

        Arguments:
          - segmentIds: list of segment ids that are checked for contact
          with boundaries. Can be given as int if only one id.
          - nSegment: (int) number of contacted segments
          - mode: 'exact' (or 'eq'), 'at_least' (or 'ge'), or 'at_most' 
          (or 'le')
          - update: flag indicating if the contact data structure is updated

        Returns: ndarray of boundaries that satisfy the condition.
        """

        # no condition
        if (segmentIds is None) and (nSegment is None):
            return self.boundaries

        # if segmentIds is a number convert to a list
        if segmentIds is not None:
            if not isinstance(segmentIds, (list, tuple, numpy.ndarray)):
                segmentIds = [segmentIds]

        # required no. of contacted segments that are in segmentIds
        if nSegment is None:
            nb = len(segmentIds)
        else:
            nb = nSegment

        # number of contacted elements for all boundaries
        if segmentIds is not None:
            selected = self.countContactedSegments(ids=segmentIds)
        elif nSegment is not None:
            selected = self.countContactedSegments()
        #else:
        #    raise ValueError, """At least one of segmentIds and nSegment
        #    arguments have to be given."""
            
        # find good boundaries for both modes 
        if (mode == 'exact') or (mode == 'eq'):
            good = numpy.where((selected._data==nb) \
                                   & (selected._mask==False))[0]

        elif (mode == 'at_least') or (mode == 'ge'):
            good = numpy.where((selected._data>=nb) \
                                   & (selected._mask==False))[0]
            
        elif (mode == 'at_most') or (mode == 'le'):
            good = numpy.where((selected._data<=nb) \
                                   & (selected._mask==False))[0]
            
        else:
            raise ValueError("Mode argument " + mode + \
                  " is not recognized. Available values are: " + \
                  "'exact' (same as 'eq'), 'at_least' (same as 'ge'), " +\
                  "and 'at_most' (same as 'le').")  

        # remove boundaries that do not satisfy the conditions, if required
        good = numpy.asarray(good, dtype='int')
        if update: self.keepBoundaries(ids=good)

        return good

    def countContactedBoundaries(self, ids=None):
        """
        Calculates and returns the number of boundaries, among those specified
        by arg ids, that contact each existing segment. 

        Returns number of contacted boundaries with given ids for each segment.

        Masked contacts are not counted. Ids can be a nested list in which 
        case each sublist is counted as a single boundary. 

        Argument:
          - ids: (list) boundary ids. If None, all boundaries are used. Note:
          in numpy 1.9.0+ single int can be specified.

        Returns: masked array indexed by segment ids, where each element 
        specifies the number of boundaries contacted by the corresponding 
        segment. Non-existing boundaries specified by arg ids are ignored.
        """

        # figure out if ids is a nested list
        if ids is None:
            ids_nested = False
        else:
            ids_nested = util_nested.is_nested(ids)

        # count contacted boundaries
        if ids is None:
            ids = self.getBoundaries()
        res = self._countContacted(axis=0, indices=ids, nested=ids_nested)

        return res

    def countContactedSegments(self, ids=None):
        """
        Calculates and returns the number of segments among those specified
        by arg ids that contact each existing boundary. 

        Masked contacts are not counted. Ids can be a nested list in which 
        case each sublist is counted as a single segment. 

        Argument:
          - ids: (list) segments ids. If None, all segments are used. Note:
          in numpy 1.9.0+ single int can be specified.

        Returns: masked array indexed by boundary ids, where each element 
        specifies the number of segments contacted by the corresponding 
        boundary. Non-existing segments specified by arg ids are ignored.
        """
        
        # We don't check for (nor allow) nested lists here to improve 
        # preformance of _countContacted and because there's no need for
        # nested segment ids yet (VL 08.10.09) 
        # Changed my mind (VL 11.10.08)

        # figure out if ids is a nested list
        if ids is None:
            ids_nested = False
        else:
            ids_nested = util_nested.is_nested(ids)

        # count contacted segments
        if ids is None:
            ids = self.getSegments()
        res = self._countContacted(axis=1, indices=ids, nested=ids_nested)

        return res

    def _countContacted(self, indices, axis, nested=False):
        """
        Counts number of positive elements of self._n._data along axis.

        Only the elements that are not masked by self._n._mask and have indices
        along axis are counted. 

        Elements of indices can be lists, in which case they are considered
        to form one boundary/segment. 

        Arguments:
          - axis: axis along which the elements are counted (0 or 1)
          - indices: (list) elements with this indices along axis are used. 
          If None, all elements are used. Note: in numpy 1.9.0+ single int 
          can be specified.
          - nested: flag indication if indices is a nested list

        Returns: masked array of calculated numbers. Non-existing boundaries /
        segments specified by arg ids are ignored.
        """

        # get max index
        if axis == 0:
            max_index = self.getMaxBoundary()
        elif axis==1:
            max_index = self.getMaxSegment()
        
        # remove indices for boundaries or segments that are outside the range
        if nested:

            # clean each element for nested
            clean_indices = []
            for element in indices: 
                if isinstance(element, (list, tuple)):
                    element = numpy.asarray(element)
                    clean_element = element[element<=max_index].tolist()
                    if len(clean_element) > 0:
                        clean_indices.append(clean_element)
                else:
                    if element<=max_index:
                        clean_indices.append(element)
            indices = clean_indices

        else:

            # not nested, simple
            np_indices = numpy.asarray(indices)
            indices = np_indices[np_indices<=max_index].tolist()
                
        # flatten indices
        if nested:
            nested_indices = indices
            indices = util_nested.flatten(nested_indices)

        # Note: need to check if multiple comactify / expend occur, or
        # if this should be in other functions
        # expand if compactified
        #was_compact = False
        #if self.compactified:
        #    was_compact = True
        #    self.expand()

        # keep only specified indices along axis
        try:
            reduced_n = self._n.take(indices=indices, axis=axis)
        except(IndexError, ValueError):
            raise

        # Note: see above
        # compactify if it was compactified
        #if was_compact: self.compactify()

        # calculate by how much the count should be decreased  
        if nested:
            for ind in nested_indices:
                if isinstance(ind, (list, tuple, numpy.ndarray)): 
                    line_n = self._countContacted(indices=ind, axis=axis, 
                                                 nested=False)
                    line_1 = 1 * (line_n > 0)
                    try:
                        overcount += line_n - line_1
                    except NameError:
                        overcount = line_n - line_1

        # find number of positives along axis; return all masked if no indices 
        if len(indices) > 0:
            # reduced_n is 2-dim because in take() arg indices is a list (above) 
            res = numpy.sum(reduced_n > 0, axis=axis)
        else:
            if axis == 0:
                max_other_ind = self.getMaxSegment()
            elif axis == 1:
                max_other_ind = self.getMaxBoundary()
            res = ma.zeros(max_other_ind+1, fill_value=self.fillValue)
            res.mask = True

        # reduce by overcount
        if nested:
            res = res - overcount
        #try:
        #    res = res - overcount
        #except UnboundLocalError:
        #    pass

        return res

    def countPositive(self, arr, indices=None, axis=0):
        """
        Counts positive entries of array arr at indices along axis.

        If indices is None all integers starting with 1 and ending with 
        the size of arr in the axis dimension are used

        Arguments:
          - arr: array
          - indices: list of indices 
          - axis: axis

        Returns ndim-1 dimensional array
        """

        # keep only indices in arr
        if indices is not None:
            all_indices = list(range(1, arr.shape[axis]))
            bad_indices = numpy.setdiff1d(all_indices, indices)
            arr = numpy.delete(arr, bad_indices, axis=axis)

        # calculate
        res = numpy.add.reduce(arr>0, axis=axis)
         
        return res 
    
    def findLinkedBoundaries(self, ids=None, distance=1, mode='exact'):
        """
        Returns ids of boundaries that are linked (through segments) to and are
        at (arg) distance to each of the boundaries specified by arg ids. 

        The distance between two boundaries is defined as the smallest number of
        segments needed to link them. For example, the distance between 
        boundaries that contact the same segment is 1 while the boundaries
        that are connected through two segments (and one other boundary) are
        at distance 2.

        If (arg) mode is 'exact' only the boundaries that are exactly at (arg) 
        distance from the specified boundaries are returned. If (arg) mode is 
        'all' boundaries at or below (arg) distance are returned.

        Arguments:
          - ids: boundary ids
          - distance: distance between boundaries
          - mode: 'exact' for exact distance or 'all' eact or lower distance

        Return: array of ids if arg ids is a single number, or a list of arrays
        if arg ids is a list, tuple or ndarray, where each element correspond 
        to the id in the argument ids at the same position. 
        """
        
        # set ids
        if ids is None:
            ids = self.getBoundaries()

        # check if one id or more
        if isinstance(ids, (list, tuple, numpy.ndarray)):
            one_id = False
        else:
            ids = [ids]
            one_id = True

        # find linked boundaries for each specified boundary
        result = []
        for id_ in ids:
            bound_ids = [id_]
            old_bound_ids = numpy.array([id_])

            for dist in range(distance):
                old_bound_ids = numpy.append(old_bound_ids, bound_ids)

                # find segments that contact the current boundary
                seg_ids = self.findSegments(boundaryIds=bound_ids, nBoundary=1)
                seg_ids = numpy.unique(seg_ids)

                # find boundaries that contact these segments
                bound_ids = self.findBoundaries(segmentIds=seg_ids, nSegment=1)
                bound_ids = numpy.unique(bound_ids)

                # remove the initial boundaries 
                if mode == 'exact':
                    bound_ids = numpy.setdiff1d(bound_ids, old_bound_ids)

            result.append(bound_ids)

        # return in the same form as argument ids
        if one_id:
            return result[0]
        else:
            return result
        
    def findLinkedSegments(self, ids=None, distance=1, mode='exact'):
        """
        Returns ids of segments that are linked (through boundaries) to and are
        at (arg) distance to each of the segments specified by arg ids. 

        The distance between two segments is defined as the smallest number of
        boundaries needed to link them. For example, the distance between 
        segments that contact the same boundary is 1 while the segments
        that are connected through two boundaries (and one other segment) are
        at distance 2.

        If (arg) mode is 'exact' only the segments that are exactly at (arg) 
        distance from the specified segments are returned. If (arg) mode is 
        'all' segments at or below (arg) distance are returned.

        Arguments:
          - ids: segment ids
          - distance: distance between segments
          - mode: 'exact' for exact distance or 'all' eact or lower distance

        Return: array of ids if arg ids is a single number, or a list of arrays
        if arg ids is a list, tuple or ndarray, where each element correspond 
        to the id in the argument ids at the same position. 
        """
        
        # set ids
        if ids is None:
            ids = self.getSegments()

        # check if one id or more
        if isinstance(ids, (list, tuple, numpy.ndarray)):
            one_id = False
        else:
            ids = [ids]
            one_id = True

        # find linked segments for each specified segment
        result = []
        for id_ in ids:
            seg_ids = [id_]
            old_seg_ids = numpy.array([id_])

            for dist in range(distance):
                old_seg_ids = numpy.append(old_seg_ids, seg_ids)

                # find boundaries that contact the current segment
                bound_ids = self.findBoundaries(segmentIds=seg_ids, nSegment=1)
                bound_ids = numpy.unique(bound_ids)

                # find segments that contact these boundaries
                seg_ids = self.findSegments(boundaryIds=bound_ids, nBoundary=1)
                seg_ids = numpy.unique(seg_ids)

                # remove the initial segments 
                if mode == 'exact':
                    seg_ids = numpy.setdiff1d(seg_ids, old_seg_ids)

            result.append(seg_ids)

        # return in the same form as argument ids
        if one_id:
            return result[0]
        else:
            return result
        
