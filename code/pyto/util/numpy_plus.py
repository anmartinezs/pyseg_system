"""
Numpy related utility functions.

ToDo: 
  - see if label_ids, remove_labels and keep_labels should be removed.
  - not sure if intersect_arrays, intersect_slices and trim_slices are made
  redundant by positioning in core.Image.

# Author: Vladan Lucic, Max Planck Institute for Biochemistry
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import zip
#from past.utils import old_div

__version__ = "$Revision$"


import numpy

def intersect_arrays(shape_1, shape_2, offset_1=None, offset_2=None):
    """
    Finds intersection of two (N-dim) arrays represented by their shapes
    and by offsets to some common origin.
    
    Arguments:
      - shape_1, shape_2: shapes of the arrays
      - offset_1, offset_2: array offsets to a common origin

    Returns (slice_1, slice_2, offset)
      - slice_1, slice_2: slices defining the intersection
      - offset: offset of the intersection. Meaningles if the arrays do not
      intersect.

    Note: should probably use trim_slices or intersect_slicesinstead.
    """

    # default data type for shapes and offsets
    defaultType = 'int_'

    # convert shapes and offsets in numpy.array forms if needed
    if not isinstance(shape_1, numpy.ndarray):
        shape_1 = numpy.array(shape_1, dtype=defaultType)
    if not isinstance(shape_2, numpy.ndarray):
        shape_2 = numpy.array(shape_2, dtype=defaultType)
    if offset_1 is None:
        offset_1 = numpy.zeros_like(shape_1)
    if not isinstance(offset_1, numpy.ndarray):
        offset_1 = numpy.array(offset_1, dtype=defaultType)
    if offset_2 is None:
        offset_2 = numpy.zeros_like(shape_2)
    if not isinstance(offset_2, numpy.ndarray):
        offset_2 = numpy.array(offset_2, dtype=defaultType)
    
    # find beginning and end indices
    origin = numpy.zeros( len(shape_1), dtype=defaultType )
    begin_1 = [ max(o21, 0) for o21 in offset_2 - offset_1 ]
    begin_2 = [ max(o12, 0) for o12 in offset_1 - offset_2 ]
    end_1 = [ min(s1, tmp) for (s1, tmp) in \
              zip(shape_1, shape_2+offset_2-offset_1) ]
    end_2 = [ min(s2, tmp) for (s2, tmp) in \
              zip(shape_2, shape_1+offset_1-offset_2) ]

    # check: don't complain if begin > end, but set to 0 if negative
    end_1 = [ max(e1, 0) for e1 in end_1 ]
    end_2 = [ max(e2, 0) for e2 in end_2 ]
    
    # make slices
    slice_1 = [slice(be, en) for (be, en) in zip(begin_1, end_1)]
    slice_2 = [slice(be, en) for (be, en) in zip(begin_2, end_2)]

    # make offsets
    offset = offset_1 + numpy.array(begin_1, dtype=defaultType)

    return slice_1, slice_2, offset

def intersect_slices(slice_1, slice_2):
    """
    Finds intersection of n-dim slices slice_1 and slice_2.

    Arguments:
      - slice_1, slice_2: tuple or list of slices (one element for each dimension)

    Returns
      - slice: tuple containing intersecting slices (one element for each dimension)
    """

    result = [slice(max(sl_1.start, sl_2.start), min(sl_1.stop, sl_2.stop)) \
              for sl_1, sl_2 in zip(slice_1, slice_2)]
    return tuple(result)

def trim_slice(slice_nd, shape):
    """
    Makes intersection of an array (defined by arg shape) with a slice (arg 
    slice_nd).

    This method is useful for operations between two non-overlaping arrays.
    This can happen when the two arrays have different shapes, or when one is 
    moved to a specific position with respect to the other one.

    For example, the following line adds an array ar_2 to array ar_1 at the
    position given by n-dim slice sl (assuming that ar_2.shape is consistent
    with sl):

      ar_1[slice_nd] + ar_2

    However, this doesn't work if slice_nd "sticks out" of ar_1. The remedy is
    to use this function:

      new_slice, other_slice = trim_slice(slice_nd, ar_1.shape)
      ar_1[new_slice] + ar_2[other_slice]

    Note that not only ar1 may need to be sliced but also ar2 may need to.

    Slice start and stop may be negative integers, in which case they are 
    considered simply to start (stop) before the beginning of the array ar_1, 
    and not the way negative indices are normally used in lists and arrays.  

    If the two arrays do not overlap, Null is returned for each dimension.

    Arguments:
      - sl: tuple or list of slices (one element for each dimension)
      - shape: shape in which modified sl (new_slice) should fit

    Return:
      - new_slice: tuple of trimmed slices (one element for each dimension)
      - other_slice: slice that needs to be applied to 
    """

    # sanity check
    if any(sl.start >= sha for sl, sha in zip(slice_nd, shape)):
        return None, None
    if any(sl.stop < 0 for sl in slice_nd):
        return None, None
    
    # work
    new_slice = [slice(max(0, sl.start), min(sha, sl.stop)) 
                 for sl, sha in zip(slice_nd, shape)]
    
    other_slice = [slice(max(0, -sl.start), 
                         new_sl.stop - new_sl.start + max(0, -sl.start))
                   for sl, new_sl in zip(slice_nd, new_slice)]
    
    return tuple(new_slice), tuple(other_slice)

def label_ids(array):
    """
    Returns (sorted ndarray of) unique positive elements of array.

    Made obsolete by segment.extractIds?
    """

    # get all unique elements and keep those that are positive
    all = numpy.unique(array)
    pos = all.compress(all>0)

    return pos

def remove_labels(array, ids, value=0):
    """
    Removes from array all elements that equal any of ids.

    Made obsolete by segment.remove?
    """
    condition = numpy.zeros(array.shape, dtype=bool)
    for id in ids:
        condition = condition | (array == id)
    array[condition] = value

def keep_labels(array, keep, value=0, all=None):
    """
    Keep only those elements of array that equal one of elements of keep.
    The remaining elements are set to value.

    Made obsolete by segment.keep?
    """

    # make a list of all unique elements if all is not given
    if all is None: all = label_ids(array=array)

    remove = numpy.setdiff1d(all, keep)

    remove_labels(array=array, ids=remove, value=value)
    
def wvar(array, weight, ddof=0):
    """
    Weighted variance of array:

      sum(weight_i (array_i - array_w_mean)^2) / (len(array) - ddof)

    Arguments:
      - array: 1d array
      - weight: (1d array) weights
      - ddof: delta degrees of freedom
    """

    array = numpy.asarray(array)
    weight = numpy.asarray(weight)

    x2w = numpy.inner(array*array, weight)
    xw = numpy.inner(array, weight)
    res = (x2w - xw * xw / weight.sum()) / float(len(array) - ddof)
    return res

def update(a, b, mask, value=0):
    """
    Replaces elements of array a with elements of array b where mask is True.

    The shape of resulting array (allways a new array) is extended to fit b.
    Elements that are created by this extension are set to value.

    Arguments:
      - a: array to be updated
      - b: update array
      - mask: (boolean array of the same shape as b) indicates which elements 
      of b are used for updating
      - value: value for padding new elements

    Returns: array of the shape obtained form max shape of a and b in each
    dimension.
    """

    # one of a and b can be None
    if a is None:
        a = numpy.zeros_like(b) + value
    if b is None:
        b = numpy.zeros_like(a) + value

    # find shape that fit both a and b
    max_shape = [max(a_ah, b_sh) for a_sh, b_sh in zip(a.shape, b.shape)]

    # enlarge a
    a_slice = [slice(0, sh1) for sh1 in a.shape] 
    a_big = numpy.zeros(shape=max_shape, dtype=a.dtype) + value
    a_big[a_slice] = a[a_slice]

    # enlarge b
    b_slice = [slice(0, sh1) for sh1 in b.shape] 
    b_big = numpy.zeros(shape=max_shape, dtype=b.dtype)
    b_big[b_slice] = b[b_slice]

    # enlarge mask
    mask_big = numpy.zeros(shape=max_shape, dtype=mask.dtype)
    mask_big[b_slice] = mask[b_slice]

    # update a
    a_big[mask] = b_big

    return a_big
