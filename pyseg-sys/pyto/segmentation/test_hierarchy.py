"""

Tests class Hiarrchy.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: test_hierarchy.py 88 2008-04-05 16:48:43Z vladan $
"""

__version__ = "$Revision"


import numpy
import scipy

from hierarchy import Hierarchy
from segment import Segment
from connected import Connected

level_0 = numpy.array([[0,1,0,2,0,0],
                       [0,0,0,0,3,0],
                       [4,0,5,0,0,0],
                       [0,0,0,6,6,0],
                       [8,0,0,0,0,7],
                       [8,0,9,0,0,0],
                       [0,0,0,0,0,0]])

level_1 = numpy.array([[11,11,11,11,0,0],
                       [0, 0, 0, 0,13,13],
                       [14,0,15,0, 0,0],
                       [0, 0, 0,16,16,16],
                       [18,18,18,0,0, 16],
                       [18,0,18,0, 0,0],
                       [0,19, 0, 0, 0,0]])

level_2 = numpy.array([[51,51,51,51,0,0],
                       [0, 0, 0, 51,51,51],
                       [54,54,54,0, 0,0],
                       [0, 0, 0,56,56,56],
                       [58,58,58, 0, 0,56],
                       [58,58,58, 0,57, 0],
                       [ 0,58, 0, 0, 0,0]])
 
level_3 = numpy.array([[21,21,21,21,0,0],
                       [0, 0,21,21,21,21],
                       [21,21,21,0, 0,0],
                       [0,0,0, 26,26,26],
                       [28,28,28,0, 0,26],
                       [28,28,28,0,27,0],
                       [0, 28,0,29,0,0]])

level_4 = numpy.array([[31,31,31,31,0,0],
                       [0, 0,31,31,31,31],
                       [31,31,31,0, 0,0],
                       [0, 0, 0,36,36,36],
                       [37,37,37,0, 0,36],
                       [37,37,37,37,37,0],
                       [0,37,37,37,0,0]])

                      
def test_1(self):
    """
    """

    s0 = Segmentation(level_0)
    s1 = Segmentation(level_1)
    s2 = Segmentation(level_2)
    s3 = Segmentation(level_3)
    s4 = Segmentation(level_4)

    # add levels
    hi = Hierarchy()
    hi.addLevel(segment=s1, check=True, shift=0)
    hi.addLevel(segment=s0, check=True, shift=0)
    hi.addLevel(segment=s4, check=True, shift=0)
    hi.addLevel(segment=s3, check=True, shift=0)
    hi.addLevel(segment=s2, check=True, shift=0)

    print hi.data
    print hi._lowerIds

    # remove levels
    hi.remove(level=0)
    hi.remove(level=3)
    hi.remove(level=1)

    print data
    print hi._lowerIds
    
image = numpy.array([[0,  0,  0,  0,  0,  2,  0,  0,  0,  0],
                     [1.8,1.2,1.4,1.8,1.6,1.4,1.8,1.2,1.6,1.4],
                     [1.2,1.2,1.4,1.8,1.6,1.4,1.4,1.2,1.6,1.4],
                     [1.2,1.6,1.8,1.8,1.4,1.8,1.8,1.4,1.8,1.4],
                     [1.2,1.8,1.2,1.6,1.6,1.6,1.4,1.8,1.6,1.4],
                     [1.2,1.8,1.2,1.8,1.8,1.4,1.8,1.2,1.6,1.4],
                     [0,  0,  0,  0,  0,  2,  0,  0,  0,  0]])

bound = numpy.array([[1,1,1,1,1,0,2,2,2,2],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [3,3,3,3,3,0,4,4,4,4]])

def test_2(self):

    #
    hi = Hierarchy()

    # add levels from threshold and connectivity
    for tr in numpy.arange(1.2, 2):
        seg, contacts = \
            Connected.make(input=image, boundary=bound, thresh=tr, 
                           boundaryIds=[1,2,3,4], nBoundary=2, 
                           boundCount='at_least', mask=0, freeSize=0,
                           freeMode='add')
                    
        props = {'threshold':tr, 'contacts':contacts}
        hi.addLevel(segment=seg, props=props, check=True, shift=10)

    # analyze
    print hi.findIdsByNBound(min=2, max=3)
    print hi.findIdsByVolume(min=6, max=10)

