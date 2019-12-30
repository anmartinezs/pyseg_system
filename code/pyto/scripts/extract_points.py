#!/usr/bin/env python
"""
Extract coordinates of (some) points of one or more segments of a segmented 
image. 

Currently implemented for the attribute regions of pyto.scene.CleftRegions
object.

This script may be placed anywhere in the directory tree.

$Id: extract_points.py 1485 2018-10-04 14:35:01Z vladan $
Author: Vladan Lucic 
"""
__version__ = "$Revision: 1485 $"

import sys
import os
import os.path
import time
import platform
import pickle
import logging
from copy import copy, deepcopy

import numpy
import scipy

import pyto

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')


##############################################################
#
# Parameters (please edit)
#
##############################################################

# pickled file that contains the object
pickle_file_name = 'pickle.pkl'

# ids of segments (labels) that should be selected, can be a single int or 
# an array (or list) of ints
ids = 7

# point selection mode: 
#  - 'all' to pick all points 
#  - 'geodesic' to pick as many points as possible according to the distance
#  variable. Points are selected in a random manner
mode = 'geodesic'

# distance (int): minimum distance between selected points, used in the
# 'geodesic' mode only
distance = 2

# structuring element connectivity used to calculate the geodesic distance
struct_el_connectivity = 1

# format in which the coordiantes of selected points are organized:
#  - 'numpy': tuple of ndarrays, where each array contains coordinates of all 
#  selected points in one dimension, as the output of numpy.nonzero()
#  - 'coordinates': array of shape n_points x ndim
points_format = 'coordinates'

# file name where the result is saved
save_file = 'points.pkl'

# save format: 
#  - 'pickle': pickle the array in the stadard python format
#  - 'numpy': the numy way (filoe extension should be '.npy'), uses numpy.save()
save_format = 'pickle'


################################################################
#
# Main function
#
###############################################################

def main():
    """
    Main function
    """
           
    # unpickle
    cl = pickle.load(open(pickle_file_name))

    # select points
    points = cl.getPoints(
        ids=ids, mode=mode, distance=distance, 
        connectivity=struct_el_connectivity, format_=points_format)

    # save points
    if save_format == 'pickle':
        pickle.dump(points, open(save_file, 'wb'), -1)
    elif save_format == 'numpy':
        numpy.save(save_file, points)


# run if standalone
if __name__ == '__main__':
    main()
