"""

Common stuff for tests

# Author: Vladan Lucic
# $Id: common.py 695 2010-04-11 21:01:57Z vladan $
"""

__version__ = "$Revision: 695 $"

import numpy
import scipy

from pyto.core.image import Image
from pyto.segmentation.cleft import Cleft
from pyto.segmentation.segment import Segment
from pyto.segmentation.grey import Grey
import pyto.segmentation.test.common as seg_cmn


##############################################################
#
# Example 1: image, boundaries and tc segmentation
#

# input
image_1 = seg_cmn.image_1
bound_1 = seg_cmn.bound_1
threshold_1 = seg_cmn.threshold_1

# expected
levelIds_1 = seg_cmn.levelIds_1
data_1 = seg_cmn.data_1
bound_ge2_1 = seg_cmn.bound_ge2_1

##############################################################
#
# CleftLayers Example 
#

def make_cleft_layers_example():
    """
    Makes and returns Cleft and Image objects that can be used together
    """

    # make image
    image_2 = numpy.zeros((10,10))
    image_2[1,:] = 5
    image_2[2,:] = 10
    for ind in range(5):
        image_2[ind+3, :] = ind + 3
    image_2[8,:] = 12
    image_2[9,:] = 6
    image_cleft_2 = Image(image_2)

    # make cleft
    cleft_2 = numpy.zeros((10, 10), dtype=int)
    cleft_2[1:3, 2:8] = 1
    cleft_2[3:8, 2:8] = 3
    cleft_2[8:, 2:8] = 2
    cleft_2 = Cleft(data=cleft_2, cleftId=3, bound1Id=1, bound2Id=2)
    
    return cleft_2, image_cleft_2

# expected
cleft_layers_density_mean = numpy.array([5, 10, 3, 4, 5, 6, 7, 12, 6])
cleft_layers_density_volume = numpy.ones(9, dtype=float) * 6
cleft_layers_width = 5.
