"""
Set of utilities for segmentation and analysis of n-dimensional images.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import

__version__ = "$Revision$"

from . import util
from . import core
from . import grey
from . import io
from . import segmentation
from . import clustering
from . import scene
from . import analysis
from . import geometry
from . import tomo
from . import correlative
try:
    from . import particles
except (ModuleNotFoundError, ImportError):
    pass

#import scripts

from numpy.testing import Tester
test = Tester().test
