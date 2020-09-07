"""

Common stuff for tests

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

import pickle
import numpy
import scipy

import pyto.scene.test.common as scene_cmn
from pyto.scene.cleft_regions import CleftRegions as SceneCleftRegions


# catalogs directory, relative to this 
rel_catalogs_dir = 'all_catalogs'

def make_and_pickle(file_, data=0, mode='layers'):
    """
    Makes cleft layers (scene.CleftRegions) objects and pickles them. Needed 
    for testing methods that read and analyze these pickles.

    Arguments:
      - file_: pickle file name 
      - data: adds this number to the image data 
    """

    # make cleft and image
    cleft, image = scene_cmn.make_cleft_layers_example()

    # change image data
    image.data += data

    # make layers or columns (scene.CleftRegions) 
    clayers = SceneCleftRegions(image=image, cleft=cleft)
    if mode == 'layers':
        clayers.makeLayers(nBoundLayers=2)
        cregions = clayers
    elif mode == 'columns':
        clayers.makeLayers(nBoundLayers=2)
        cregions = clayers.makeColumns(bins=[0, 0.5, 1], normalize=True)
    cregions.findDensity(regions=cregions.regions)

    # pickle it
    fd = open(file_, 'wb')
    pickle.dump(cregions, fd, -1)

# expected
#cleft_layers_density_mean = numpy.array([5, 10, 3, 4, 5, 6, 7, 12, 6])
cleft_layers_density_mean = scene_cmn.cleft_layers_density_mean
cleft_layers_density_volume = scene_cmn.cleft_layers_density_volume
#cleft_layers_width = 5.
cleft_layers_width = scene_cmn.cleft_layers_width

cleft_columns_density_mean = numpy.array([5, 5])
cleft_columns_volume = numpy.array([15, 15])
