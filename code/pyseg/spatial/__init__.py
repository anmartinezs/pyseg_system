"""
Set of specific utilities for spatial analysis

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 12.06.15
"""

__author__ = 'Antonio Martinez-Sanchez'

# Analysis from coordinates DEPRECATED
from mb import CMCAnalyzer
from plane import SetClouds
from plane import SetClusters
from plane import SlA
from plane import SetCloudsP
from plane import SetClustersP
from plane import PairClouds
from plane import NetFilCloud
from plane import SetPairClouds
from plane import GroupClouds
from plane import GroupPlotter
from globals import FuncComparator

# Analysis based on sparse arrays
from sparse import CSRSimulator, LinkerSimulator, RubCSimulation, CSRTSimulator, gen_sin_points, gen_rand_in_mask
from sparse import CPackingSimulator, AggSimulator, AggSimulator2
from sparse import UniStat, BiStat
from sparse import PlotUni, PlotBi
from stack import TomoUni