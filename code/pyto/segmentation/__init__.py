"""
Segmentation and segment analysis for n-dimensional images.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import

__version__ = "$Revision$"

from .struct_el import StructEl
from .statistics import Statistics
from .grey import Grey
from .labels import Labels
from .contact import Contact
from .cluster import Cluster
from .segment import Segment
from .ball import Ball
from .plane import Plane
from .connected import Connected
from .hierarchy import Hierarchy
from .thresh_conn import ThreshConn
from .features import Features
from .morphology import Morphology
from .topology import Topology
from .distance_to import DistanceTo
from .distance import Distance
from .bound_distance import BoundDistance
from .density import Density
from .cleft import Cleft
#import test_old
#import test_hierarchy

#import test
#from numpy.testing import Tester
#test = Tester().test
