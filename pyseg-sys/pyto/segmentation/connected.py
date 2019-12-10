"""

Provides class Connected for the manipulation of segments defined on the basis
of their connectivity. Each of these segments is defined as a (single)
connected cluster of array elements. The segments do not touch each other.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: connected.py 1333 2016-09-27 08:01:37Z vladan $
"""

__version__ = "$Revision: 1333 $"


import logging
import numpy
import scipy
import scipy.ndimage as ndimage

import pyto.util.numpy_plus as numpy_plus
import pyto.util.nested as nested
from struct_el import StructEl
from contact import Contact
from pyto.core.image import Image
from grey import Grey
from labels import Labels
from segment import Segment

class Connected(Segment):
    """
    """

    ##################################################################
    #
    # Constructor
    #
    #################################################################

    def __init__(self, data=None, copy=True, ids=None, clean=False):
        """
        Initializes instance and sets data, data-related and id-related
        attributes by calling super constructor.  

        Arguments:
          - data: (ndarray) segments (labels)
          - copy: if True makes a copy of data
          - ids: list of ids
          - clean: if True only the segments given in ids are retained in 
          self.data
        """
        super(Connected, self).__init__(data, copy, ids, clean)


    ##################################################################
    #
    # Other methods
    #
    #################################################################

    @classmethod
    def make(cls, image, boundary, thresh=None, label=True, count=False, 
             structElConn=1, contactStructElConn=1, countStructElConn=None,
             boundaryIds=None, nBoundary=None, boundCount='exact',
             mask=0, freeSize=0, freeMode='add', saveContactLabels=False):
        """
        Performs threshold and connectivity segmentation at specified threshold.

        Image is thresholded, and segmented based on contacts to regions
        defined in (arg) boundaries. The segmentation consists of finding all
        disjoint segments of thresholded image (see label), analyzing contacts
        between the segments and the boundaries (see findContacts) and removing
        segments that do not satisfy the contact conditions (see findSegments). 

        If threshold is None, the image is assumed to be either a thresholded
        or a fully segmented image having background (0 or False) and foreground
        (elements>0 or True), so thresholding is not performed.

        If label is False, the image is assumed to contain segments already, so
        the labeling is skipped.

        Each segment is allowed to contact exactly/at least/at most (argument
        boundCount) nBoundary boundaries. If count is True, the number of
        contacts between each segment and each boundaries are counted. 
        
        Unless attributes contactStructElConn and countStructElConn are set
        explicitly before calling this method, the default values (1) are used
        to generate the structuring elements.

        The segmentation is preformed on a segmentation (free) region (see 
        Segment.makeFree). A free region is either given by the 
        positive elements of mask (or mask.data), or if mask is an integer 
        the free region is a segment of boundary with that id. A free 
        region is further restricted to a union (freeMode 'add') or an 
        intersection (freeMode 'intersect') of neighborhoods of boundaries 
        that are at most freeSize away from boundaries. 

        Internally, this method works on insets of boundary, image and mask 
        (free region) that are just large enough to contain the free region 
        and the boundaries. The returned Connected data have the same 
        positioning as boundary.data.

        Data and inset of args boundary and image are not changed by this 
        method. 

        Arguments:
          - image: (grayscale) image to be segmented (ndarray or Image object)
          if arg label is True, or labeled (Label) image if arg label is False
          - boundary: (Segment): defines boundaries
          - thresh: a threshold
          - label: flag indicating if labeling should be done
          - count: flag indicating if contacts are counted
          - structElConn: structuring element used for finding disjoint segments
          - contactStructElConn: structuring element used to detect contacts
          between segments and boundaries
          - countStructElConn: structuring element used to count contacts
          between each segment and each boundary
          - boundaryIds: ids of boundaries in boundary object
          - nBoundary: number of boundaries that a segment contacts 
          - boundCount: 'exact' (or 'eq'), 'at_least' (or 'ge'), or 'at_most'
          (or 'le')
          - mask: (ndarray, Label or int) determines a free region
          - freeSize: max distance to boundaries that is included in a free
          region
          - freeMode: 'intersect' or 'add' determines how the boundary
          neighborhoods are combined
          - saveContactLabels: flag indicating if the array containing contact
          elements is saved (in contacts.labels)

        Returns (conn, contacts):
          - conn: an instance of this class containing the segments found
          - contacts: the corresponding Contacts instance 
        """

        # check image
        if isinstance(image, numpy.ndarray):
            image = Grey(data=image)

        # save insets
        boundary_inset = boundary.inset
        image_inset = image.inset
        image_expanded = False

        # make segmentation (free) region if needed and bring image, boundary
        # and free to the same size
        if label:

            # make segmentation (free) region
            if boundaryIds is None:
                flat_b_ids = None
            else:
                flat_b_ids = nested.flatten(boundaryIds)
            free = boundary.makeFree(
                ids=flat_b_ids, size=freeSize, mode=freeMode, mask=mask, 
                update=False)
            if isinstance(free, numpy.ndarray):
                free = Segment(data=free, copy=False)

            # reduce free.inset to intersection with image
            free.useInset(inset=image.inset, mode='abs', intersect=True)

            # expand free and boundary to union between them
            large_inset = boundary.findEnclosingInset(inset=free.inset)
            free.useInset(inset=large_inset, mode='abs', expand=True)
            boundary.useInset(inset=large_inset, mode='abs', expand=True)

            # check if new boundary inside image, if not expand image  
            boundary_inside_image = image.isInside(inset=large_inset)
            if boundary_inside_image:

                # don't need to expand image data
                image.useInset(
                    inset=large_inset, mode='abs', expand=False) 

            else:

                # expand image to (new) boundary if needed and save data 
                image_data = image.data.copy()
                # value doesn't matter because segmentation limited by free
                image.useInset(
                    inset=large_inset, mode='abs', expand=True, 
                    value=image.data.max())
                image_expanded = True

        else:

            # insets based on image 
            image.makeInset()
            large_inset = boundary.findEnclosingInset(inset=image.inset)
            boundary.useInset(inset=large_inset, mode='abs', expand=True)
            image.useInset(inset=large_inset, mode='abs', expand=True)

        # threshold, if needed
        if thresh is not None:
            if not label:
                raise ValueError(
                    "If argument label is not None, arg label should be True.")
            thresh_image = image.doThreshold(threshold=thresh)
        else:
            thresh_image = image

        # make a new instance of this class, use boundary positioning
        conn = cls()
        conn.copyPositioning(boundary, saveFull=True)

        # set structuring element connectivities 
        conn.setStructEl(rank=image.data.ndim, connectivity=structElConn)
        conn.contactStructElConn = contactStructElConn

        # segment by connectivity if needed, and detect contacts
        contacts = conn.findContacts(
            input=thresh_image, label=label, boundary=boundary,
            boundaryIds=boundaryIds, mask=free, freeSize=0)
        logging.debug("connected.py (make): %d segments before constraints",
                      len(contacts.getSegments()) ) 

        # impose constraints on contacts unless image is already fully segmented
        if label:
            contacts = conn.findSegments(contacts, boundaryIds=boundaryIds,
                                     nBoundary=nBoundary, countMode=boundCount,
                                     update=True, reorder=True)
            logging.debug("connected.py (make): %d segments after constraints",
                          len(contacts.getSegments()) ) 

        # set count structuring connectivity and count contacts
        if count:
            conn.countStructElConn = countStructElConn
            contacts.findContacts(segment=conn, boundary=boundary,
                         boundaryIds=boundaryIds,
                         contactStructEl=conn.contactStructEl,
                         countStructEl=conn.countStructEl,
                         count=True, saveLabels=saveContactLabels)

        # recover insets so that the arguments passed are not changed
        boundary.useInset(inset=boundary_inset, mode='abs', useFull=True,
                          expand=True)
        if image_expanded:
            image.data = image_data
            image.setInset(inset=image_inset, mode='abs')
        else:
            image.useInset(inset=image_inset, mode='abs', useFull=True)

        # conn: use inset from boundary and adjust fullInset/Data to inset/data
        conn.useInset(inset=boundary.inset, mode='abs', expand=True)
        #conn.saveFull()      # removed in r912, don't see any reason for it

        # return
        return conn, contacts

    def label(self, input, mask=None, structEl=None, update=False,
              relabel=False):
        """
        Segments input by identifying (connected) segments in input. The 
        connected segments are formed from elements of input that are >0 or 
        True (if input is binary)

        Only the elements of input that are >0 and where mask >0 are labeled. 
        The segmented image has the same offset and inset as mask 

        If update is False self.data is set to the new segments array and the
        positioning attributes (offset, inset, fullShape) are set to the 
        same as in mask. 

        If update is True the new segments are added to the exisiting self.data,
        after they are repositioned to fit the current offset and inset.
        In this case and if relabel is False, new segemnts are simply placed
        on top the exsisting may overwrite the already existing segments of 
        self.data. Otherwise, the existing and the new segments are combined and
        relabeled (the old segments are likely to change their ids). See
        self.add.

        Arguments:
          - input: (pyto.core.Image, or ndarray) image to be segmented
          - mask: (pyto.core.Image or ndarray, mask that defines the 
          segmentation region
          - structEl: instance of StructEl, structuring elements that defines
          the connectivity of segments (in None, self.structEl is used) 
          - update: if True new segments are added to the existing self.data
          - relabel: determines what to do if the segments generated here
          overlap with the existing segments (self.data).

        Sets:
          - self.data: array containing labeled segments with shape of the
          intersection of input and mask.data with mask.offset.
        """

        # set input and structEl 
        if isinstance(input, numpy.ndarray):
            input = Image(input)
        if structEl is None: structEl = self.structEl

        # mask
        if isinstance(mask, numpy.ndarray):
            mask = Image(mask)

        # restrict input to an inset corresponding to mask
        adj_input_inst  = input.usePositioning(image=mask, new=True)
        adj_input = adj_input_inst.data
        #input_inset = mask.findImageSlice(image=input)
        #adj_input = input.data[input_inset]

        # mask (but don't overwrite) input
        adj_input = numpy.where(mask.data>0, adj_input, 0)
    
        # find segments
        labeled_data, nSegments = \
           StructEl.label(input=adj_input, structure=structEl)

        # add new segments and update
        if update:

            # position labeled_slice to this instance
            adj_input_inst.usePositioning(image=self, intersect=False)
            #labeled = self.__class__(data=labeled_data)
            #labeled.copyPositioning(mask)
            #labeled_slice = self.findImageSlice(labeled)

            # add labeled_slice
            self.add(new=adj_input_inst.data, relabel=relabel)
            #self.add(new=labeled_data[labeled_slice], relabel=relabel)

        else:

            # set data to labeled_data and positioning to be the same as 
            # in mask 
            self.setData(data=labeled_data, copy=False)
            self.copyPositioning(adj_input_inst)
            #self.copyPositioning(mask)

    def findContacts(
            self, input, boundary, boundaryIds=None, boundaryLists=None,
            label=True, mask=0, freeSize=0, freeMode='add',
            count=False, saveLabels=False):
        """
        Finds all contacts between segements of this instance and boundaries. 

        If label is False, the segments are taken directly from input. 

        If label is True, segments are formed by identifying clusters of 
        connected elements of input among the forground (>0) elements (see 
        self.label and scipy.ndimage.label methods). The connectivity is 
        given by structEl or self.structEl structuring element (default 
        connectivity 1).

        The segmentation is preformed only in the segmentation region defined
        as the intersection of the region defined by mask and the "free" region.
        The segmentation region is restricted to bound=mask if mask is an int,
        or to the positive elements of mask if mask is an array.

        If freeSize<=0, then the free regions is determined by mask only. If it
        is >0, areas around each boundary with max (Euclidean) distance
        freeSize are intersected/summmed (determined by freeMode) to define
        the free region. Uses self.makeFree method.

        If boundaryLists is None, the free region is formed using all 
        boundaryIds. Otherwise, for each element of boundaryList (containing 
        an array of boundary ids) a free region (and a segmentation region) 
        is defined and the segmentation is preformed. In this case all 
        segments are added up and then relabeled (to avoid the overwritting 
        of some segments in case different segmentation regions overlap). 
        However, some segments may be formed by addition of separate pieces 
        and therefore be larger than what would be expected based on 
        freeSize and freeMode parameters. 

        Note: the data for input, boundary and mask has to have the same shape,
        their positioning (inset) is not taken into account,

        Arguments:
          - input: (pyto.core.Image, or ndarray) image to be segmented
          - boundary: ndarray with labeled boundaries, or a Segment instance
          - boundaryIds: list of ids defining boundaries in boundary
          - boundaryLists: an iterable (a list, tuple or a generator) containing
          lists (subsets) of boundaryIds.
          - label: if True label input, if False input already contains 
          segments 
          - freeSize: size of the free region. Either a list corresponding 
          to ids
          (different size for each boundary) or a single number (used for all
          boundaries)
          - freeMode: Can be 'add' or 'intersect' and specifies that the free
          region is formed by the addition or intersection of the enlarged
          boundaries
          - mask: defines the region where new segments are allowed to be
          established. If int, the allowed region is where boundary==mask.
          Otherwise it's an array and the positive or True elements define the
          allowed region.
          - saveLabels: flag indicating if the array containing contact
          elements is saved (in contacts.labels)

        Sets attributes data, ids and id-related and creates a Contact object.

        Returns instance of Contacts.
        """

        # wrap input if needed
        if isinstance(input, numpy.ndarray):
            input = Image(input)
        
        # set boundary object and flat boundaryIds
        if isinstance(boundary, Segment):
            if boundaryIds is not None:
                flat_bids = nested.flatten(boundaryIds)
            else:
                flat_bids = boundary.ids
        else:
            boundary = Segment(data=boundary, ids=boundaryIds)
            flat_bids = boundary.ids
        
        # make free regions and find segments within free regions
        if label:
            if boundaryLists is None:

                # for all boundaries at the same time
                free = boundary.makeFree(ids=flat_bids, size=freeSize,
                                         mode=freeMode, mask=mask, update=False)
                self.label(input=input, mask=free)

            else:

                # for each group of boundaries separately
                for bounds in boundaryLists:
                    # ToDo: avoid making free if boundaries are too far 
                    free = boundary.makeFree(ids=bounds, size=freeSize,
                                         mode=freeMode, mask=mask, update=False)
                    self.label(input=input, mask=free)
                self.label()

        else:
            self.setData(input.data, copy=False)

        # find all contacts
        contacts = Contact()
        contacts.findContacts(
            segment=self, boundary=boundary, boundaryIds=flat_bids,
            contactStructEl=self.contactStructEl, saveLabels=saveLabels)

        return contacts

    def findSegments(self, contacts, boundaryIds=None, nBoundary=None,
                     countMode='exact', update=True, reorder=True):
        """
        Analyzes contacs to find segments that satisfy  the criterium
        specified by boundaryIds, nBoundary and mode arguments.

        See Contact.findSegments for details regarding the criterium.

        If update is true, only the segments that satisfy the criterium are
        retained both in the contacts object and in this instance.

        Arguments:
          - contacts: instance of Contact that already contains the contact data
          structure
          - boundaryIds: (nested) list or ndarray of boundary ids that are
          checked for contact with segments. Can be given as int if only one id.
          - nBoundary: (int) number of contacted boundaries
          - mode: 'exact' (or 'eq'), 'at_least' (or 'ge'), or 'at_most'
          (or 'le')
          - update: flag indicating if the contact data structure and the
          segments of this instance are updated to contain only the segments
          that satisfy the criterium.
          - reorder: (used only if update is True) flag indicating if the
          segments remaining after update are ordered from 1 up

        Returns: instance of Contact 
        """

        # analyze contacts and remove bad ones if required
        good = contacts.findSegments(
            boundaryIds=boundaryIds, nBoundary=nBoundary,
            mode=countMode, update=update)

        # remove bad segments and reorder both here and in contacts
        if update:
            self.keep(ids=good)
            if reorder:
                new_order = self.reorder(clean=True)
                contacts.reorderSegments(order=new_order)

        return contacts

