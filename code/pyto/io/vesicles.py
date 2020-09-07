"""
Contains class Vesicles for reading vesicles results from a file and
parsing the results.

Note: Depreciated because vesicles results are red from pickle files.

After executing the following statements:

ves = pyto.io.Vesicles()
ves.read('vesicles_file')

all data from the vesicles is saved in the following attributes:

  - ves.ids: ndarray of vesiclke ids
  - ves.vesDensMean: vesicle
  - ves.vesDensStd
  - ves.vesDensMin
  - ves.vesDensMax
  - ves.memDensMean: membrane
  - ves.memDensStd
  - ves.memDensMin
  - ves.memDensMax
  - ves.intDensMean: interior
  - ves.intDensStd
  - ves.intDensMin
  - ves.intDensMax
  - ves.vesVolume: vesicle volume
  - ves.memVolume: membrane volume
  - ves.center: vesicle center
  - ves.radiusMean: vesicle radius
  - ves.radiusStd
  - ves.radiusMin
  - ves.radiusMax

All attributes are ndarrays ordered in the same way as ids in ves.ids.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import

__version__ = "$Revision$"


import numpy
import scipy
import scipy.io
from .results import Results


class Vesicles(Results):
    """
    Contains methods to read and write vesicle results.

    """

    def __init__(self):
        """
        Initializes attributes.
        """

        # from super
        super(self.__class__, self).__init__()

        # local
        self.names = ['id', 
                'vesDensMean', 'vesDensStd', 'vesDens.Min', 'vesDensMax',
                'memDensMean', 'memDensStd', 'memDens.Min', 'memDensMax',
                'intDensMean', 'intDensStd', 'intDens.Min', 'intDensMax',
                'vesVolume', 'memVolume', 'center'
                'radiusMean', 'radiusStd', 'radiusMin', 'radiusMax']
        self.columns = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,),
                        (9,), (10,), (11,), (12,), 
                        (13,), (14,), (15,16,17), 
                        (18,), (19,), (20,), (21,)]

        

