#!/usr/bin/env python
"""
Blanks (sets to a given value) those voxels of a tomogram that correspond to
given segments. 

This script may be placed anywhere in the directory tree.

$Id: blank.py 21 2007-12-23 11:38:03Z vladan $
"""

import os
import numpy
import pyto


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
imageFileName = "and-1-6.mrc"

#####################################################
#
# Labels file. If the file is in em or mrc format shape, data type, byte order
# and array order are not needed (should be set to None). If these variables
# are specified they will override the values given in em or mrc headers.
#

# name of the labels file (file containing segmented volume) 
labelsFileName = "../viz/and-1-6_labels_sv.raw"

# labels file dimensions
labelsShape = (260, 260, 100)

# labels file data type (e.g. 'int8', 'int16', 'int32', 'float16', 'float64') 
labelsDataType = 'int8'

# labels file byteOrder ('<' for little-endian, '>' for big-endian)
labelsByteOrder = '<'

# labels file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis fastest)
labelsArrayOrder = 'FORTRAN'

# offset of labels in respect to the image (None means 0-offset)
labelsOffset = None


#########################################
#
# Calculation related parameters
#

# ids of all labels that need to be blanked (removed) 
#vesicleIds = range(2,64)  # range of ids
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

# out file name suffix
blankSuffix = '_blank-test'


################################################################
#
# Work
#
################################################################

################################################
#
# Read files
#

# read image file
imageFile = pyto.io.ImageIO()
imageFile.read(file=imageFileName)

# read labels file
labels = pyto.io.ImageIO()
labels.read(file=labelsFileName, byteOrder=labelsByteOrder, dataType=labelsDataType,
            arrayOrder=labelsArrayOrder, shape=labelsShape)

#################################################
#
# Calculations
#

blankData = imageFile.data.copy()
for id in ids:
    blankData[labels.data==id] = value

##################################################
#
# Write output file
#

# extract parts from the imageFileName and make blank image name
(dir, base) = os.path.split(imageFileName)
(root, ext) = os.path.splitext(base)
blankFileName = os.path.join(dir, root+blankSuffix+ext)

# write the blank file with the same header as in imageFile
imageFile.write(file=blankFileName, data=blankData)


