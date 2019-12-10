#!/usr/bin/env python
"""
Adds segments form different segment (labels) files.

The segment ids in the resulting file are calculated as follows:
  - segment ids of the first label file are left unchanged 
  - segment ids of the second label file are increased by the specified
    value of variable shift
  - segment ids of the third label file are increased by 2*shift
    ...

Typically used to combine multiple boundary files into one. The resulting 
boundary file is then used for the segmentation and analysis procedures.  

This script may be placed anywhere in the directory tree.

$Id: add_segments.py 1430 2017-03-24 13:18:43Z vladan $
Author: Vladan Lucic 
"""
__version__ = "$Revision: 1430 $"

import numpy
import pyto


##############################################################
#
# Parameters
#
##############################################################

############################################################
#
# Segment (labels) files related.
#

# segment (labels) file names, order may be important (see remove_overlap)
labels_file_names = ("labels_1.dat",
                     "labels_2.dat",
                     "labels_3,dat")

# labels file dimensions
labels_shape = (100, 120, 90)

# labels file data type (e.g. 'int8', 'int16', 'int32', 'float16', 'float64') 
labels_data_type = 'int8'

# labels file byte order ('<' for little-endian, '>' for big-endian)
labels_byte_order = '<'

# labels file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis fastest)
labels_array_order = 'FORTRAN'

#################################################################
#
# Ids related
#

# ids of the segments that need to be added, one entry for each labels file
ids = ([3,5,7],
       range(2,8) + [4],
       range(3,5) + range(6,10))

# id shift in each subsequent labels (in case of multiple labels files).
# Important: Shift has to be larger than the largest id (of all labels file) 
#shift = None     # shift is determined automatically
shift = 300

# removes segments (from the earlier file) if it intersects with a segment
# from a later file.
remove_overlap = True

# renumbers ids after the files are added
relabel = False

#####################################################################
#
# Output file related
#

# output file name
out_labels_file_name = "combined_labels.dat"

# data type of combined labels, should be chosen so that the max id is still
# below the data type limit (e.g. 'uint8', 'uint16', 'uint32' for an em file)
out_labels_data_type = 'uint16'


################################################################
#
# Work
#
################################################################

# initialize
result = pyto.segmentation.Segment()
curr_shift = 0

# loop over labels files
for (single_file, single_ids) in zip(labels_file_names, ids):

    # read current labels file
    labels_file = pyto.io.ImageIO()
    labels_file.read(
        file=single_file, byteOrder=labels_byte_order, 
        dataType=labels_data_type, arrayOrder=labels_array_order, 
        shape=labels_shape)
    labels = pyto.segmentation.Segment(
        data=labels_file.data, copy=False, ids=single_ids)

    # add current segments
    result.add(
        new=labels, shift=curr_shift, remove_overlap=remove_overlap, 
        dtype=out_labels_data_type, relabel=relabel)
    if shift is None:
        curr_shift = None
    else:
        curr_shift += shift

# write output file
out_file = pyto.io.ImageIO()
out_file.write(file=out_labels_file_name, data=result.data)

