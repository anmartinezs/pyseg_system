"""
Analysis of scenes (features, or parts of segmented images).

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import

__version__ = "$Revision$"

from .em_lm_correlation import EmLmCorrelation
from .neighborhood import Neighborhood
from .cleft_regions import CleftRegions
from .segmentation_analysis import SegmentationAnalysis
from .multi_cluster import MultiCluster


# Not good because it makes pyto.scene.test a method, so importing 
# pyto.scene.test.common doesn't work
#import test
#from numpy.testing import Tester
#test = Tester().test
