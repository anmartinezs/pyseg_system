#!/usr/bin/env python
"""
Blanks (sets to a given value) those voxels of a tomogram that correspond to
given segments. 

This script may be placed anywhere in the directory tree.

$Id$
Author: Vladan Lucic (Max Planck Institute for Biochemistry) 
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

import os

import numpy

import pyto
import pyto.scripts.common as common


##############################################################
#
# Parameters
#
##############################################################

############################################################
#
# Image (grayscale) file. If it isn't in em or mrc format, format related
# variables (shape, data type, byte order and array order) need to be given
# (see labels file section below).
#

# name of the image file
image_file_name = "../3d/tomo.mrc"

#####################################################
#
# Labels file. If the file is in em or mrc format shape, data type, byte order
# and array order are not needed (should be set to None). If these variables
# are specified they will override the values given in em or mrc headers.
#

# name of the labels file (file containing segmented volume) 
labels_file_name = "labels_1.raw"

# labels file dimensions
labels_shape = None     # shape given in header of the disc file (if em
                        # or mrc), or in the tomo (grayscale image) header   
#shape = (100, 120, 90) # shape given here

# labels file data type (e.g. 'int8', 'int16', 'int32', 'float16', 'float64') 
labels_data_type = 'int8'

# labels file byteOrder ('<' for little-endian, '>' for big-endian)
labels_byte_order = '<'

# labels file array order ('F' for x-axis fastest, 'C' for z-axis fastest)
labels_array_order = 'F'

#########################################
#
# Calculation related parameters
#

# ids of all labels that need to be blanked (removed) 
ids = [2,3,4]       # individual ids
#ids = range(2,5)   # range of ids, same as above

# value to which the blanked elements are set
value = -1.

###########################################################
#
# Output (blanked) file. The output file name is formed as:
#
#   labels_directory + labels_base_name + <connSuffix> + labels_extension
#

# blanked tomo file name suffix
blank_suffix = '_1'


################################################################
#
# Work
#
################################################################

################################################
#
#  Main function
#

def main():

    # read image file
    #imageFile = pyto.io.ImageIO()
    #imageFile.read(file=image_file_name)
    image = common.read_image(
        file_name=image_file_name, header=True, memmap=True)
    new_shape = common.find_shape(
        file_name=image_file_name, shape=labels_shape,
        suggest_shape=image.data.shape)

    # read labels file
    #labels = pyto.io.ImageIO()
    #labels.read(
    labels = pyto.segmentation.Segment.read(
        file=labels_file_name, byteOrder=labels_byte_order, 
        dataType=labels_data_type, arrayOrder=labels_array_order, 
        shape=new_shape)

    # calculations
    blank_data = image.data.copy()
    for id in ids:
        blank_data[labels.data==id] = value

    # extract parts from the image_file_name and make blank image name
    (dir, base) = os.path.split(image_file_name)
    (root, ext) = os.path.splitext(base)
    blank_file_name = os.path.join(dir, root+blank_suffix+ext)

    # write the blank file with the same header as in imageFile
    image_io = pyto.io.ImageIO()
    image_io.readHeader(file=image_file_name)
    image_io.setData(data=blank_data)
    image_io.write(file=blank_file_name)


# run if standalone
if __name__ == '__main__':
    main()


