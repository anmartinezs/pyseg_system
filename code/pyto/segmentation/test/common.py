"""

Common stuff for tests

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import zip
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"

import numpy
import numpy.testing as np_test 
import scipy

import pyto
from pyto.segmentation.grey import Grey
from pyto.segmentation.segment import Segment


##############################################################
#
# Example 1: image, boundaries and tc segmentation
#
# Hierarchy:
#
#              14
#               |
#              13
#               |
#              12
#         |     -    |
#        10          11
#      |  -  |    |  -  |
#      6     8    7     9
#    | - |        | 
#    3   5        4
#    |            |
#    1            2    


# image 1
image_ar_inset_1 = numpy.array(\
    [[9, 1, 9, 9, 4, 1, 1, 9],
     [9, 2, 9, 9, 4, 9, 2, 9],
     [3, 3, 4, 5, 4, 6, 1, 4],
     [2, 9, 3, 9, 3, 9, 2, 5]])
image_ar_1 = numpy.zeros((10,10)) + 9
image_1 = Grey(image_ar_1)
image_ar_1[2:6, 1:9] = image_ar_inset_1
image_1in = Grey(image_ar_inset_1)
image_1in.inset = [slice(2, 6), slice(1, 9)]
image_1in2 = Grey(image_ar_1[1:7, 1:9])
image_1in2.inset = [slice(1, 7), slice(1, 9)]

# boundaries 1
bound_ar_inset_1 = numpy.array(\
    [[3, 3, 3, 3, 3, 3, 3, 3],
     [5, 5, 5, 5, 5, 5, 5, 5],
     [5, 5, 5, 5, 5, 5, 5, 5],
     [5, 5, 5, 5, 5, 5, 5, 5],
     [5, 5, 5, 5, 5, 5, 5, 5],
     [4, 4, 4, 4, 4, 4, 4, 4]])
bound_ar_1 = numpy.zeros((10,10), dtype=int)
bound_ar_1[1:7, 1:9] = bound_ar_inset_1
bound_1 = Segment(bound_ar_1)
bound_1in = Segment(bound_ar_inset_1)
bound_1in.inset = [slice(1, 7), slice(1, 9)]

# expected segmentation
threshold_1 = list(range(8))
ids_1 = list(range(1,15))
levelIds_1 = [[], [1,2], [3,4,5], [6,7,8,9], [10,11], 
                 [12], [13], [14]]
thresh_1 = [0, 1,1, 2,2,2, 3,3,3,3, 4,4, 5, 6, 7]
data_1 = numpy.zeros((8,4,8), dtype='int')
data_1[0] = numpy.zeros((4, 8), dtype='int')
data_1[1] = numpy.array([[0, 1, 0, 0, 0, 2, 2, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]])
data_1[2] = numpy.array([[0, 3, 0, 0, 0, 4, 4, 0],
                            [0, 3, 0, 0, 0, 0, 4, 0],
                            [0, 0, 0, 0, 0, 0, 4, 0],
                            [5, 0, 0, 0, 0, 0, 4, 0]])
data_1[3] = numpy.array([[0, 6, 0, 0, 0, 7, 7, 0],
                            [0, 6, 0, 0, 0, 0, 7, 0],
                            [6, 6, 0, 0, 0, 0, 7, 0],
                            [6, 0, 8, 0, 9, 0, 7, 0]])
data_1[4] = numpy.array([[0, 10, 0, 0, 11, 11, 11, 0],
                            [0, 10, 0, 0, 11, 0, 11, 0],
                            [10, 10, 10, 0, 11, 0, 11, 11],
                            [10, 0, 10, 0, 11, 0, 11, 0]])
data_1[5] = numpy.array([[0, 12, 0, 0, 12, 12, 12, 0],
                            [0, 12, 0, 0, 12, 0, 12, 0],
                            [12, 12, 12, 12, 12, 0, 12, 12],
                            [12, 0, 12, 0, 12, 0, 12, 12]])
data_1[6] = numpy.array([[0, 13, 0, 0, 13, 13, 13, 0],
                            [0, 13, 0, 0, 13, 0, 13, 0],
                            [13, 13, 13, 13, 13, 13, 13, 13],
                            [13, 0, 13, 0, 13, 0, 13, 13]])
data_1[7] = numpy.array([[0, 14, 0, 0, 14, 14, 14, 0],
                            [0, 14, 0, 0, 14, 0, 14, 0],
                            [14, 14, 14, 14, 14, 14, 14, 14],
                            [14, 0, 14, 0, 14, 0, 14, 14]])
slice_1 = [slice(2,6), slice(1,9)]
hi_data_1 =  numpy.zeros((10,10), dtype=int)
hi_data_1[tuple(slice_1)] = numpy.array(\
    [[0, 1,  0,  0, 11,  2,  2, 0],
     [0, 3,  0,  0, 11,  0,  4, 0],
     [6, 6, 10, 12, 11, 13,  4, 11],
     [5, 0,  8,  0,  9,  0,  4, 12]])
bound_ge2_1 = [[], [], [4], [6, 7], [10, 11], [12], [13], [14]]
n_contact_1 = numpy.array([[0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2],
                           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 4]]) 

#expected analysis
density_1 = numpy.array([-1., 1., 1., 1.5, 1.4, 2., 2.2, 1.4, 3., 3., 
                          2.57142857, 2.6, 2.84210526, 3., 3.])
#region_density_1 = 5.25
bkg_density_1 = [5.25, 5.68965517, 6.5, 7.2, 8.26666667, 8.76923077, 9, 9]
volume_1 = numpy.array([0, 1, 2, 2, 5, 1, 5, 5, 1, 1, 7, 10, 19, 20, 20])
euler_1 = numpy.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
n_faces_1 = numpy.array([[ 3,  1,  0],
                        [ 1,  0,  0],
                        [ 2,  1,  0],
                        [ 2,  1,  0],
                        [ 5,  4,  0],
                        [ 1,  0,  0],
                        [ 5,  4,  0],
                        [ 5,  4,  0],
                        [ 1,  0,  0],
                        [ 1,  0,  0],
                        [ 7,  6,  0],
                        [10,  9,  0],
                        [19, 19,  1],
                        [20, 21,  1],
                        [20, 21,  1]])
distance_to_3_min_1 = numpy.array([-1, 1, 1, 1, 1, 4, 1, 1, 4, 4, 1, 
                                    1, 1, 1, 1])
distance_to_3_4_mean_1 = numpy.array([-1, 1, 1, 1.5, 11./5, 1., 
                                       12./5, 11./5, 1, 1, 15./7,
                                       24./10, 44./19, 46./20, 46./20])
closest_region_1 = numpy.array([-1, 3, 3, 3, 3, 4, 4, 3, 4, 4, 4,
                                 3, 4, 4, 4])


##############################################################
#
# Example 2: boundaries
#

# boundaries 2
bound_ar_inset_2 = numpy.array(\
    [[3, 3, 3, 3, 6, 6, 6, 6],
     [5, 5, 5, 5, 9, 9, 9, 9],
     [5, 5, 5, 5, 9, 9, 9, 9],
     [5, 5, 5, 5, 9, 9, 9, 9],
     [5, 5, 5, 5, 9, 9, 9, 9],
     [4, 4, 4, 4, 8, 8, 8, 8]])
bound_ar_2 = numpy.zeros((10,10), dtype=int)
bound_ar_2[1:7, 1:9] = bound_ar_inset_2
bound_2 = Segment(bound_ar_2)
bound_2in = Segment(bound_ar_inset_2)
bound_2in.inset = [slice(1, 7), slice(1, 9)]

[slice(2,6), slice(1,9)]
hi_data_1 =  numpy.zeros((10,10), dtype=int)
hi_data_1[tuple(slice_1)] = numpy.array(\
    [[0, 1,  0,  0, 11,  2,  2, 0],
     [0, 3,  0,  0, 11,  0,  4, 0],
     [6, 6, 10, 12, 11, 13,  4, 11],
     [5, 0,  8,  0,  9,  0,  4, 12]])
bound_ge2_1 = [[], [], [4], [6, 7], [10, 11], [12], [13], [14]]

#expected analysis
density_1 = numpy.array([-1., 1., 1., 1.5, 1.4, 2., 2.2, 1.4, 3., 3., 
                          2.57142857, 2.6, 2.84210526, 3., 3.])
volume_1 = numpy.array([0, 1, 2, 2, 5, 1, 5, 5, 1, 1, 7, 10, 19, 20, 20])


##############################################################
#
# Example 3: 3D segments
#

# segments of different homology ranks
segment_ar_in_3 = numpy.zeros((8,9,5), dtype='int')
segment_ar_in_3[0,0,0] = 1
segment_ar_in_3[1,1:4,0] = 2
segment_ar_in_3[1,3,1:2] = 2
segment_ar_in_3[0:3,0:2,2] = 3

segment_ar_in_3[4:7,0:4,1] = 11
segment_ar_in_3[5,1:3,1] = 0
segment_ar_in_3[4,3,0] = 11
segment_ar_in_3[7,0:3,0:5] = 12
segment_ar_in_3[7,1,1] = 0 
segment_ar_in_3[7,1,3] = 0 

segment_ar_in_3[5:8,6:9,2:5] = 21
segment_ar_in_3[6,7,3] = 0
segment_ar_in_3[0:3,6:9,2:5] = 22
segment_ar_in_3[1,7,3] = 0
segment_ar_in_3[0,6:9,0:2] = 22
segment_ar_in_3[0,7,1] = 0

# make objects
segment_ar_3 = numpy.zeros((10,10,6), dtype='int')
inset_3 = [slice(1,9), slice(0,9), slice(1,6)]
segment_ar_3[tuple(inset_3)] = segment_ar_in_3
segment_3 = Segment(segment_ar_3)
segment_3in = Segment(segment_ar_in_3)
segment_3in.inset = inset_3

# expected 
#euler_3 = numpy.array([-5, 1, 1, 1, -5, -5, -5, -5, -5, -5, 
#                       -5, 0, -1, -5, -5, -5, -5, -5, -5, -5,
#                       -5, 2, 1])
euler_3 = numpy.array([5, 1, 1, 1, 0, -1, 2, 1])
objects_3 = numpy.insert(numpy.ones(7, dtype=int), 0, 7)
loops_3 = numpy.array([4, 0, 0, 0, 1, 2, 0, 1])
holes_3 = numpy.array([2, 0, 0, 0, 0, 0, 1, 1])
faces_3 = numpy.array([[0, 0, 0, 0],
                       [1, 0, 0, 0],
                       [4, 3, 0, 0],
                       [6, 7, 2, 0],
                       [11, 11, 0, 0],
                       [13, 14, 0, 0],
                       [26, 48, 24, 0],
                       [31, 54, 24, 0]])
faces_3[0,:] = numpy.sum(faces_3, axis=0)



##############################################################
#
# Other
#

def id_correspondence(actual, desired, ids=None, current=None):
    """
    Checks that data (given in desired and actual) overlap completely (but the 
    actual ids may be different) and returns dictionary with 
    desired_id : actual_id pairs if ids is None, or an array of actual ids 
    ordered according to ids.

    If a dicitionary of desired - actual ids is given as arg current (arg ids
    has to be None) the new dictionary is updated with current.

    The intended use is to check a segmentation when segment ids may be 
    different. However it may not besuitable for hierarchical segmentation
    because only the ids that actually exist in the data arrays are mapped.
    A way to overcome this problem is to run this function on each level
    and pass the already existing dictionary as arg current.

    Arguments:
      - desired: (ndarray) actual data array
    """

    # check overall agreement 
    np_test.assert_equal(actual>0, desired>0)

    # check that individual segments agree
    actual_ids = numpy.unique(actual[actual>0])
    desired_ids = numpy.unique(desired[desired>0])
    id_dict = dict(list(zip(desired_ids, actual_ids)))
    #id_dict = {}
    #for d_id in actual_ids:
    #    a_id = desired[actual==d_id][0]
    #    np_test.assert_equal(desired==a_id, actual==d_id)
    #    id_dict[d_id] = a_id

    if ids is None:

        if current is not None:
            id_dict.update(current)
        return id_dict

    else:

        return numpy.array([id_dict.get(id_) for id_ in ids])

    

def make_shapes():
    """
    Makes an array containing few different shapes, makes a Segmentation object 
    and returns it.
    """

    # empty array
    shapes = numpy.zeros(shape=(10,10), dtype=int)

    # add square
    shapes[1:4,1:4] = 1

    # add circle
    shapes[0:5,5:10] = 3
    shapes[0,5] = 0
    shapes[0,9] = 0
    shapes[4,9] = 0
    shapes[4,5] = 0

    # add elipse
    shapes[7:10,1:7] = [[0,4,4,4,0,0],
                        [4,4,4,4,4,0],
                        [0,4,4,4,4,4]]

    # add uneven
    shapes[6:10,5:9] = [[6,6,0,0],
                       [0,6,6,0],
                       [4,0,6,6],
                       [4,4,0,6]]
    
    # instantiate and return
    return Segment(shapes)

def make_grey():
    """
    Makes an array with greyscale-like values, puts it in a Grey object
    and returns the object.
    """

    grey_data = numpy.arange(100).reshape((10,10))
    # not good because for some strange reason Grey can not be initialized
    # after it's edited and reloaded
    #grey = Grey(grey_data)
    grey = pyto.segmentation.Grey(grey_data)
    return grey
    
