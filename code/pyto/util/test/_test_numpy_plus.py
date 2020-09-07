"""
ToDo: convert to proper format

Tests for modules in this directory
"""
from __future__ import print_function
# Author: Vladan Lucic, last modified 05.04.07

import scipy
import scipy.ndimage
import numpy
import pyto.util.numpy_plus as np_plus 

# define test arrays
aa = numpy.arange(12, dtype='int32')
aa = aa.reshape((3,4))
bb = numpy.arange(6, dtype='int32')
bb = bb.reshape((2,3))

def run():

    print("Checking numpy_plus.intersect_arrays ...")

    # run
    print("\n\taa: ")
    print(aa)
    print("\tbb: ")
    print(bb)

    res = np_plus.intersect_arrays(aa.shape, bb.shape)
    print("\n\tno offset: ")
    print("\t", res)
    print(aa[res[0]])
    print(bb[res[1]])

    offset_1 = (0,0)
    offset_2 = (1,0)
    res = np_plus.intersect_arrays(aa.shape, bb.shape,
                                    offset_1=offset_1, offset_2=offset_2)
    print("\n\toffset_1 = ", offset_1, ", offset_2 = ", offset_2, ":")
    print("\t", res)
    print(aa[res[0]])
    print(bb[res[1]])

    offset_1 = (0,0)
    offset_2 = (0,2)
    res = np_plus.intersect_arrays(aa.shape, bb.shape,
                                    offset_1=offset_1, offset_2=offset_2)
    print("\n\toffset_1 = ", offset_1, ", offset_2 = ", offset_2, ":")
    print("\t", res) 
    print(aa[res[0]])
    print(bb[res[1]])

    offset_1 = (1,0)
    offset_2 = (0,2)
    res = np_plus.intersect_arrays(aa.shape, bb.shape,
                                    offset_1=offset_1, offset_2=offset_2)
    print("\n\toffset_1 = ", offset_1, ", offset_2 = ", offset_2, ":")
    print("\t", res) 
    print(aa[res[0]])
    print(bb[res[1]])

    offset_1 = (2,2)
    offset_2 = (1,4)
    res = np_plus.intersect_arrays(aa.shape, bb.shape,
                                    offset_1=offset_1, offset_2=offset_2)
    print("\n\toffset_1 = ", offset_1, ", offset_2 = ", offset_2, ":")
    print("\t", res) 
    print(aa[res[0]])
    print(bb[res[1]])

