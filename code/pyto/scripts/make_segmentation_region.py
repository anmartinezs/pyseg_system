#!/usr/bin/env python
"""
Makes a region around the specified boundaries. 

Used to make the segmentation region around vesicles and the active zone
membrane for the presynaptic analysis.

Specifically, given a label image, makes a region around the labels 
specified by bounday_ids. The extent of this region is defined by paramers
free_size and free_mode. This region is not formed over any of the labels
specified by all_ids. The newly formed label image that contain the new
region as well as all labels specified by all_ids is saved as a file.

If a label that exists in the input label image is not specified among
all_ids, it will be ignored. That means that it will not be present in the
output label image and that the new region may be formed over it. 

$Id$
Author: Vladan Lucic 
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

#import sys
#import os
#import os.path
#import time
#import platform
#import pickle
from copy import copy, deepcopy
import logging

import numpy as np

import pyto
import pyto.scripts.common as common

# import ../common/tomo_info.py
tomo_info = common.__import__(name='tomo_info', path='../common')

# to debug replace INFO by DEBUG
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')

##############################################################
#
# Parameters
#
##############################################################
 
###############################################################
#
# Input labels file, specifies boundaries and possibly other regions.
# If the file is in em or mrc format shape, data type, byte order and array
# order are not needed (should be set to None). If these variables are
# specified they will override the values specified in the headers.
#

# name of the input labels file containing boundaries
labels_file_name = "pre_labels.mrc"   

# labels file dimensions (size in voxels)
labels_shape = None   # shape given in header of the labels file (if em or mrc)
#labels_shape = (512, 512, 190) # shape given here

# labels file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float16', 
# 'float64') 
if tomo_info is not None: labels_data_type = tomo_info.labels_data_type
#labels_data_type = 'uint16'

# labels file byteOrder ('<' for little-endian, '>' for big-endian)
labels_byte_order = '<'

# labels file array order ('F' for x-axis fastest, 'C' for z-axis fastest)
labels_array_order = 'F'

###############################################################
#
# Input labels file, specifies boundaries and possibly other regions.
#

# name of the output labels file 
if tomo_info is not None: out_labels_file_name = tomo_info.labels_file_name
#out_labels_file_name = "labels_final.mrc"

# labels file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float16', 
# 'float64'), or None to keep the data type of the input labels file 
out_labels_data_type = None

###########################################################
#
# Boundaries and other segments in labels
#
# Important: 
#   - All ids in this section should be >0
#   - Voxels at the tomogam side should not belong to any boundaries
#
# Important note: Incorrect id specification (all_ids, boundary_ids and 
# conn_region) accounts for a majority of problems encountered while running 
# this script. 

# ids of all segments defined in boundary labels file, including boundary ids.
# Segments of the boundary file that are not listed here are discarded, that is
# set to 0 for segmentation purposes (and consequently may be used for the 
# segmentation region if conn_region=0)
# Single labels file forms:
#all_ids = [2,3,5,6]       # individual ids
if tomo_info is not None: all_ids = tomo_info.all_ids

# ids of all boundaries.
# Nested list can be used where ids in a sublist are 
# uderstood in the "or" sense, that is all boundaries listed in a sublist form 
# effectivly a single boundary
#boundary_ids = [2,3,5]       # individual ids
if tomo_info is not None: boundary_ids = tomo_info.boundary_ids

# Id of the segmentation region (where connectors can be formed)
#
# Note that decreasing segmentation region decreases computational time.
# Not used if read_connections is True.
if tomo_info is not None: conn_region = tomo_info.segmentation_region
#conn_region = 3        # segmentation region not specified in boundary file

# check if specified boundaries exist and if they are not disconnected
check_boundaries = True  

# Maximal distance to the boundaries for the segmentation region. Connectors are
# formed in the area surrounding boundaries (given by all ids
# specified in boundary_ids) that are not further than free_size from the
# boundaries and that are still within conn_region. If free_size = 0, the area
# surrounding boundaries is maximaly extended (to the limits of conn_region).
free_size = 50  # the same for all boundaries, one labels file

# Defines the manner in which the areas surrounding individual boundaries 
# (defined by free_size) are combined. These areas can be added 
# (free_mode='add') or intersected (free_mode='intersect'). 
# Not used if free_size = 0
free_mode = 'add'


################################################################
#
# Work
#
################################################################

######################################################
#
# Functions
#

################################################################
#
# Main function
#

def main():
    """
    """

    # remove segmentation region id from all ids
    all_ids.remove(conn_region)
    

    # read boundaries from an labels file
    bound, nested_boundary_ids = common.read_labels(
        file_name=labels_file_name, ids=all_ids, label_ids=boundary_ids, 
        shape=labels_shape, 
        byte_order=labels_byte_order, data_type=labels_data_type,
        array_order=labels_array_order,
        clean=True, check=check_boundaries)
    flat_boundary_ids = pyto.util.nested.flatten(nested_boundary_ids)

    # make segmentation region
    free = bound.makeFree(
        ids=flat_boundary_ids, size=free_size, 
        mode=free_mode, mask=None, update=False)
    bound.data[free.data & (bound.data == 0)] = conn_region

    # save labels
    write_head = False
    if bound.header is not None:
        write_head = True
    bound.write(
        file=out_labels_file_name, dataType=out_labels_data_type,
        pixel=bound.pixelsize, header=write_head)


# run if standalone
if __name__ == '__main__':
    main()

