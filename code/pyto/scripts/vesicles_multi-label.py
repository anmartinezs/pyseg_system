#!/usr/bin/env python
"""
Calculates some morphological quantities (volume, surface, radius) for each 
vesicle (segment) and does basic statistical analysis on the density of the
vesicles.

This script may be placed anywhere in the directory tree.

ToDo:
  - excentricity

$Id: vesicles_multi-label.py 30 2008-01-08 16:17:52Z vladan $
Author: Vladan Lucic 
"""
__version__ = "$Revision: 30 $"

import os
import os.path
import time
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
image_file_name = "../3d/tomo.em"

#####################################################
#
# Labels file. If the file is in em or mrc format shape, data type, byte order
# and array order are not needed (should be set to None). If these variables
# are specified they will override the values given in em or mrc headers.
#

# name of the labels file (file containing segmented volume)
labels_file_name = "labels.dat"    # single labels file

# use this if multiple labels files are used (all need to have same format)
#labels_file_name = ["labels_1.dat, "labels_2.dat", "labels_3.dat"]

# id shift in each subsequent labels (in case of multiple labels files) 
#shift = None     # shift is determined automatically
shift = 254

# labels file dimensions
labels_shape = (100, 120, 90)

# labels file data type (e.g. 'int8', 'int16', 'int32', 'float16', 'float64') 
labels_data_type = 'int8'

# data type of combined labels, should be chosen so that the max id is still
# below the data type limit (e.g. 'int16', 'int32', 'int64')
labels_final_data_type = 'int16'

# labels file byte order ('<' for little-endian, '>' for big-endian)
labels_byte_order = '<'

# labels file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis fastest)
labels_array_order = 'FORTRAN'

# offset of labels in respect to the image (None means 0-offset)
labels_offset = None

# ids of all vesicles in segments (labels)
vesicle_ids = [2,3,6]       # individual ids, one labels file
#vesicle_ids = range(2,64)  # range of ids, one labels file
#vesicle_ids = None         # all segments are to be used, one labels file
#vesicle_ids = [[2,3,6], None, range(3,9)]  # multiple labels files

#########################################
#
# Calculation related parameters
#

# calculate radius statistics for major 2d slices
do_slices = False

# thickness of vesicle membranes 
membrane_thick = 2

##############################################
#
# Output (results) file. The results file name is formed as:
#   results_directory/results_prefix + labels root + results_suffix
#

# results directory
results_directory = ''

# results file name prefix (without directory)
results_prefix = ''

# results file name suffix
results_suffix = '_vesicles.dat'

# results file name (use this only if you do not want this name to be composed
# using the above variables
results_file_name = None

# include total values (for all segments taken together) in the results, id 0
include_total = True


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
image_file = pyto.io.ImageIO()
image_file.read(file=image_file_name)
image = pyto.segmentation.Image(data=image_file.data)

# read multiple label files
if isinstance(labels_file_name, str):
    labels_file_name = [labels_file_name]
    vesicle_ids = [vesicle_ids]
ves = pyto.segmentation.Segment()
curr_shift = 0
for (l_name, ids) in zip(labels_file_name, vesicle_ids):

    # read current labels file
    labels = pyto.io.ImageIO()
    labels.read(file=l_name, byteOrder=labels_byte_order, dataType=labels_data_type,
            arrayOrder=labels_array_order, shape=labels_shape)
    curr_ves = pyto.segmentation.Segment(data=labels.data, copy=False, ids=ids)

    # add to the ves
    ves.add(new=curr_ves, shift=curr_shift)
    ves.data = numpy.asarray(ves.data, dtype=labels_final_data_type)
    if shift is None:
        curr_shift = None
    else:
        curr_shift += shift

#################################################
#
# Calculations
#

# vesicles density and morphology
ves.density, ves.meanDensity, ves.bkg, ves.tot = \
             image.getSegmentDensity(segment=ves, offset=labels_offset)
ves.mor = pyto.segmentation.Morphology(segments=labels.data, ids=ves.ids)
ves.mor.getVolume()
ves.mor.getRadius(doSlices=do_slices)
del ves.data

# make vesicle membranes and lumens (interiors)
mem = pyto.segmentation.Segment(data=labels.data, copy=True,
                                ids=ves.ids, clean=True)
mem.makeSurfaces(size=membrane_thick)
lum = pyto.segmentation.Segment(data=labels.data, copy=True,
                                ids=mem.ids, clean=True)
lum.labelInteriors(surfaces=mem.data)


# lumen density
lum.density, lum.meanDensity, lum.bkg, lum.tot = \
    image.getSegmentDensity(segment=lum, ids=ves.ids, offset=labels_offset)
del lum.data

# membranes density and morphology
mem.density, mem.meanDensity, mem.bkg, mem.tot = \
    image.getSegmentDensity(segment=mem, ids=ves.ids, offset=labels_offset)
mem.mor = pyto.segmentation.Morphology(segments=mem.data, ids=ves.ids)
mem.mor.getVolume()
del mem.data

###############################################
#
# Write results
#

# make results file name and open the results file
if results_file_name == None:
    (dir, base) = os.path.split(labels_file_name[0])
    (root, ext) = os.path.splitext(base)
    results_file_name = os.path.join(results_directory,
                                   results_prefix + root + results_suffix)
results_file = open(results_file_name, 'w')

# results file header
image_time = \
    '(' + time.asctime(time.localtime(os.path.getmtime(image_file_name))) + ')'
labels_time = [time.asctime(time.localtime(os.path.getmtime(l_file))) \
             for l_file in labels_file_name]
labels_name_time = [fi + ' (' + ti + ') ' \
                        for (fi, ti) in zip(labels_file_name, labels_time)] 
header = ("#",
          "# Image: " + image_file_name + " " + image_time,
          "# Labels: " + str(tuple(labels_name_time)),
          "# Working directory: " + os.getcwd(),
          "")

# write header and parameters
param = ("# Parameters:",
         "#\t Membrane thickness: " + str(membrane_thick), 
         "")
for line in header + param: results_file.write(line + os.linesep)

# write results table head
tabHead = ["# Id      Vesicle density              Membrane density            Interior density      Vesicle Membrane      Center               Radius         ",
           "#     mean    std    min    max    mean   std    min    max    mean   std    min    max                    x     y     z    mean std    min   max  "]
tabHeadExt = ["    X-slice radius          Y-slice radius           Z-slice radius",
              "mean std    min   max   mean std    min   max   mean std    min   max"]
if do_slices:
    for [line, lineExt] in zip(tabHead, tabHeadExt):
        results_file.write(line + lineExt + os.linesep)
else:
    for line in tabHead: results_file.write(line + os.linesep)

# list of all output variables (starts with id id prependIndex = True)
outVars = [ves.density.mean, ves.density.std, ves.density.min, ves.density.max,
           mem.density.mean, mem.density.std, mem.density.min, mem.density.max,
           lum.density.mean, lum.density.std, lum.density.min, lum.density.max,
           ves.mor.volume, mem.mor.volume,
           ves.mor.center[:,0], ves.mor.center[:,1], ves.mor.center[:,2],
           ves.mor.radius.mean, ves.mor.radius.std,
           ves.mor.radius.min, ves.mor.radius.max]
outFormat = "%3u  %6.2f %5.2f %6.2f %6.2f  %6.2f %5.2f %6.2f %6.2f  %6.2f %5.2f %6.2f %6.2f  %6u %6u %6u %5u %5u %5.1f %5.2f %5.1f %5.1f"  
if do_slices:
    for srad in ves.mor.sliceRadius:
        outVars += [srad.mean, srad.std, srad.min, srad.max]
        outFormat += "%5.1f %5.2f %5.1f %5.1f "
                     
# write the results for each vesicle
resTable = pyto.io.util.arrayFormat(arrays=outVars, format=outFormat,
                                    indices=ves.ids, prependIndex=True)
for line in resTable: results_file.write(line + os.linesep)

# write results for all vesicles together
if include_total:
    tot_line = pyto.io.util.arrayFormat(arrays=outVars, format=outFormat,
                                        indices=[0], prependIndex=True)
    results_file.write(os.linesep + tot_line[0] + os.linesep)

# write background results
bkgOutVars = [ves.bkg.mean, ves.bkg.std, ves.bkg.min, ves.bkg.max]
bkgOutFormat = " bkg %6.2f %5.2f %6.2f %6.2f"
results_file.write((bkgOutFormat % tuple(bkgOutVars)) + os.linesep)

# write total results
totOutVars = [ves.tot.mean, ves.tot.std, ves.tot.min, ves.tot.max]
totOutFormat = " bkg %6.2f %5.2f %6.2f %6.2f"
results_file.write((totOutFormat % tuple(totOutVars)) + os.linesep)

# write the average values 
results_file.write(os.linesep + "# Average values: " + os.linesep)
avOutVars = [ves.meanDensity.mean, ves.meanDensity.std,
             ves.meanDensity.min, ves.meanDensity.max,
             mem.meanDensity.mean, mem.meanDensity.std,
             mem.meanDensity.min, mem.meanDensity.max,
             lum.meanDensity.mean, lum.meanDensity.std,
             lum.meanDensity.min, lum.meanDensity.max,
             ves.mor.meanRadius.mean, ves.mor.meanRadius.std,
             ves.mor.meanRadius.min, ves.mor.meanRadius.max]
avOutFormat = " av  %6.2f %5.2f %6.2f %6.2f  %6.2f %5.2f %6.2f %6.2f  %6.2f %5.2f %6.2f %6.2f                                   %5.1f %5.2f %5.1f %5.1f"
results_file.write((avOutFormat % tuple(avOutVars)) + os.linesep)

# flush
results_file.flush()
