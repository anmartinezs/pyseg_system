"""
Analysis of one or more n-dimensional images.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: __init__.py 1006 2013-12-18 10:26:23Z vladan $
"""

__version__ = "$Revision: 1006 $"

# observations
from experiment import Experiment
from observations import Observations

# groups of observations
from catalog import Catalog
from groups import Groups
from vesicles import Vesicles
from connections import Connections
from layers import Layers
from clusters import Clusters
from cleft_regions import CleftRegions

#import test
# Not good because it makes pyto.analysis.test a method which interferes 
# with the module of the same name
#from numpy.testing import Tester
#test = Tester().test
