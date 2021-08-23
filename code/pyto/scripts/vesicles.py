#!/usr/bin/env python
"""
Basic greyscale density and morphology analysis of vesicle-like segments. 
These segments are typically manually / Amira segmented. 

This script may be placed anywhere in the directory tree.

Calculates:
  - greyscale density statistics
  - surface (membrane) and lumen greyscale density statistics 
  - distance to a specified region
  - center and radius 

Output files:

  - result file: text file containing all results and relevant parameters
  - vesicle pickle: pickle containing resulting analysis of (entire) vesicles
  (pyto.segmentation.Segment object), intended for all further analyses. 
  Also contains vesicles labels image.  
  - membrane and lumen pickles: pickles containing resulting analysis of 
  vesicles membrane and lumen, respectively (type pyto.segmentation.Segment), 
  intended for all further analyses

Important attributes of the (output) pickled objects:

  - vesicleIds: vesicle ids
  - data: labels image

Notes about specifying parameters:

  - In order to make setting up multiple scripts easier, parameters common 
to these scripts are expected to be read from tomo_info.py file. The location 
of this file is given as argument path of common.__import__(). These parameters
are set in this script in lines having the following form:

  if tomo_info is not None: labels_file_name = tomo_info.labels_file_name

Specifiying another value for the same parameter (here labels_file_name) 
overrides the value from tomo_info.

  - Incorrect id specification accounts for a majority of problems encountered 
while running this script. In case an exception is thrown, please check the ids
carefully first.

  - For many variables more than one option is given. Uncomment the desired one
and adjust the value if needed, and comment out the other options.

Other notes:

  - Currently (r997) division by 0 warnings are printed. These might be ignored
because they most likely arise from calculating denisty of a 0-volume segment
and are also a consequence of the changed behavior of scipy. 

  - Currently (r1118) pickled object attribute ids contains all boundaries,
  while vesicleIds only ids of vesicles. Attribute ids may be changed in future.


$Id$
Author: Vladan Lucic 
"""
from __future__ import print_function
from builtins import zip
#from builtins import str
from past.builtins import basestring

__version__ = "$Revision$"

import sys
import os
import os.path
import pickle
import platform
import time
import logging

import numpy
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

############################################################
#
# Image (grayscale) file. If it isn't in em or mrc format, format related
# variables (shape, data type, byte order and array order) need to be given
# (see labels file section below).
#

# name of the image file
if tomo_info is not None: image_file_name = tomo_info.image_file_name
#image_file_name = "../3d/tomo.em"

#####################################################
#
# Labels file. If the file is in em or mrc format shape, data type, byte order
# and array order are not needed (should be set to None). If these variables
# are specified they will override the values given in em or mrc headers.
#
# If multiple labels files are used, labels_file_name, all_ids and boundary_ids
# have to be tuples (or lists) of the same lenght
#
# Note: The use of multiple label files is depreciated. Use add_segments.py
# to combine multiple label files into one.
#

# name of the labels file (file containing segmented volume)
if tomo_info is not None: labels_file_name = tomo_info.labels_file_name
#labels_file_name = "labels.dat"    # single labels file
#labels_file_name = ("labels_1.dat", "labels_2.dat", "labels_3.dat")  # more labels

# labels file dimensions
labels_shape = None   # shape given in header of the labels file (if em
                      # or mrc), or in the tomo (grayscale image) header   
#labels_shape = (100, 120, 90) # shape given here

# labels file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float16', 'float64') 
if tomo_info is not None: labels_data_type = tomo_info.labels_data_type
#labels_data_type = 'uint8'

# labels file byte order ('<' for little-endian, '>' for big-endian)
labels_byte_order = '<'

# labels file array order ('F' for x-axis fastest, 'C' for z-axis fastest)
labels_array_order = 'F'

# offset of labels in respect to the image (None means 0-offset)
labels_offset = None

# offset of labels in respect to the data (experimental)
labels_offset = None             # no offset
#labels_offset = [10, 20, 30]    # offset specified

#########################################
#
# Calculation related parameters
#

# ids of all segments for single labels file. Segments with other ids are 
# removed 
#all_ids = [1,3,4,6]       # individual ids, single file
#all_ids = range(1,119)  # range of ids, single file
#all_ids = None         # all ids are to be used, single file
if tomo_info is not None: all_ids = tomo_info.all_ids

# ids of all segments for multiple labels files. All above forms can be used,
# the following means ids 2, 3 and 6 from labels_1, ids 15-94 from labels_2
# and ids 3, 4, ... 8 from labels_3
# Warning: it is likely to work, this form is not actively maintaned
#all_ids = ([2, 3, 6], range(15,95), range(3,9))  # multiple files

# ids of vesicles in the labels file, all formats available for all_ids are
# accepted here also.
#vesicle_ids = [3,4,6]       # individual ids
#vesicle_ids = range(3,96)  # range of ids
#vesicle_ids = None         # all segments are to be used
if tomo_info is not None: vesicle_ids = tomo_info.vesicle_ids
#vesicle_ids = (range(3,15), range(15,95), range(1,52)) # multiple files

# check if specified boundaries exist and if they are not disconnected
check_vesicles = True  

# id shift in each subsequent labels (in case of multiple labels files) 
#shift = None     # shift is determined automatically
shift = 256

# id of a segment to which distances are caluculated. In case of multiple
# labels files this id is understood after the ids of the labels files are
# shifted.
if tomo_info is not None: distance_id = tomo_info.distance_id
#distance_id = None   # distances are not calculated
#distance_id = 6

# the way distance is calculated 
# Note: if 'max' or 'center' are added, they has to be added to the presynaptic
# analysis script as 
# pyto.analysis.Vesicles.read(additional=['maxDistance', 'centerDistance'], ...)
#distance_mode = 'mean'            # mean distance calculated
distance_mode = ('mean', 'min')  # mean and min distances calculated

# calculate radius statistics for major 2d slices; if True axis has to be
# specified (experimental, meant for cryo-sections)
do_slices = False

# determines which 2d slices are used: 0 for yz, 1 for xz, 2 for xy
# used only if do_slices is True (experimental, meant for cryo-sections)
axis = None
#axis = 0

# thickness of vesicle membranes (in pixels)
membrane_thick = 2

###########################################################
#
# Pickle files. The file name is formed as:
#
#   <pkl_directory>/<pkl_prefix> + labels root + <pkl_suffix>
#

# hierarchy directory
pkl_directory = ''

# hierarchy file name prefix (no directory name)
pkl_prefix = ''

# hierarchy file name suffix
pkl_suffix = ".pkl"

##############################################
#
# Output (results) file. The results file name is formed as:
#
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

################################################################
#
# Analysis
#

def analyze_vesicles(image, ves, ves_ids):
    """
    Analyzes whole vesicles, membranes and lumens.

    In case a vesicle does not have a lumen, its density is set to nan.

    Arguments:
      - image: greyscale image
      - ves: vesicle labels image
      - ves_ids: vesicle ids
    """

    # vesicles density and morphology
    ves.density, ves.meanDensity, ves.bkg, ves.tot = (
        image.getSegmentDensity(segment=ves, offset=labels_offset, ids=ves_ids))
    ves.mor = pyto.segmentation.Morphology(segments=ves.data, ids=ves_ids)
    ves.mor.getVolume()
    ves.mor.getRadius(doSlices=do_slices, axis=axis)

    return ves

def analyze_membranes_lumens(image, ves, ves_ids):
    """
    Analyses vesicle membranes and lumens.

    In case a vesicle does not have a lumen, its density is set to nan.

    Arguments:
      - image: greyscale image
      - ves: vesicle labels image
      - ves_ids: vesicle ids
    """

    # make vesicle membranes and lumens (interiors)
    mem = pyto.segmentation.Segment(data=ves.data, copy=False,
                                    ids=ves_ids, clean=True)
    mem.copyPositioning(ves)
    mem.makeSurfaces(size=membrane_thick)
    lum = pyto.segmentation.Segment(data=ves.data, copy=True,
                                    ids=mem.ids, clean=True)
    lum.copyPositioning(ves)
    lum.labelInteriors(surfaces=mem.data)

    # lumen density
    lum.density, lum.meanDensity, lum.bkg, lum.tot = \
        image.getSegmentDensity(segment=lum, ids=ves_ids, offset=labels_offset)
    write_pickle(segment=lum, name='lum')
    del lum.data

    # membranes density and morphology
    mem.density, mem.meanDensity, mem.bkg, mem.tot = \
        image.getSegmentDensity(segment=mem, ids=ves_ids, offset=labels_offset)
    mem.mor = pyto.segmentation.Morphology(segments=mem.data, ids=ves_ids)
    mem.mor.getVolume()
    write_pickle(segment=mem, name='mem')
    del mem.data

    return mem, lum

###########################################
#
# Read image and boundary files
#

def read_image():
    """
    Reads image file and returns an segmentation.Image object

    Depreciated
    """
    image_file = pyto.io.ImageIO()
    image_file.read(file=image_file_name)
    image = pyto.segmentation.Grey(data=image_file.data)
    return image

def read_boundaries(check=True, suggest_shape=None):
    """
    Reads labels file(s), makes (Segment) boundaries and makes inset.

    Works on single and multiple boundaries files. In the latter case, ids
    are shifted so that they do not overlap and the boundaries from different 
    files are merged.

    The label file shape is determined using the first found of the following:
      - variable labels_shape
      - shape given in the labels file header (em im or mrc format) 
      - argument suggest_shape

    Arguments:
      - check: if True checks if there are ids without a boundary (data 
      elements), or disconnected boundaries
      - suggest_shape: suggested image shape (see above)
 
    Returns (bound, shifted_boundary_ids) where:
      - bound: (Segment) boundaries, from single file or merged
      - boundary_ids: (list of ndarrays) boundry ids, each list element contains
      ids from one boundary file (shifted in case of multiple files)
   """

    # read
    #if is_multi_boundaries():
    #    bound, multi_boundary_ids = read_multi_boundaries(
    #        suggest_shape=suggest_shape)
    #else:
    #    bound = read_single_boundaries(suggest_shape=suggest_shape)
    #    multi_boundary_ids = [vesicle_ids]

    # offset
    #bound.offset = labels_offset

    bound, multi_boundary_ids = common.read_labels(
        file_name=labels_file_name, ids=all_ids, label_ids=vesicle_ids, 
        shift=shift, shape=labels_shape, suggest_shape=suggest_shape, 
        byte_order=labels_byte_order, data_type=labels_data_type,
        array_order=labels_array_order,
        clean=True, offset=labels_offset, check=check)

    # make inset
    bound.makeInset()

    return bound, multi_boundary_ids

def is_multi_boundaries():
    """
    Returns True if multiple boundaries (labels) files are given.

    Depreciated
    """
    if isinstance(labels_file_name, basestring):
        return False
    elif isinstance(labels_file_name, (tuple, list)):
        return True
    else:
        raise ValueError(
            "Argument labels_file_name has to be either a string (one " 
            + "labels file) or a tuple (multiple labels files).")    

def read_single_boundaries(suggest_shape=None):
    """
    Reads and initializes boundaries form a sigle labels file.

    Argument:
      - suggest_shape: suggested image shape, used only of variable 
      labels_shape is None and boundaries file is neither em nor mrc

    Depreciated
    """

    # find shape
    shape = common.find_shape(file_name=labels_file_name, shape=labels_shape,
                             suggest_shape=suggest_shape)

    # read labels file and make a Segment object
    bound = pyto.segmentation.Segment.read(
        file=labels_file_name, ids=all_ids, clean=False, 
        byteOrder=labels_byte_order, dataType=labels_data_type,
        arrayOrder=labels_array_order, shape=shape)

    return bound

def read_multi_boundaries(suggest_shape=None):
    """
    Reads and initializes boundaries form a sigle labels file.

    Argument:
      - suggest_shape: suggested image shape, used only of variable 
      labels_shape is None and boundaries file is neither em nor mrc

    Depreciated
    """

    # read all labels files and combine them in a single Segment object
    bound = pyto.segmentation.Segment()
    curr_shift = 0
    shifted_vesicle_ids = []
    for (l_name, a_ids, v_ids) in zip(labels_file_name, all_ids, vesicle_ids):

        # find shape
        shape = common.find_shape(file_name=l_name, shape=labels_shape,
                             suggest_shape=suggest_shape)

        # read
        curr_bound = pyto.segmentation.Segment.read(
            file=l_name, ids=a_ids,
            clean=True, byteOrder=labels_byte_order, dataType=labels_data_type,
            arrayOrder=labels_array_order, shape=shape)

        bound.add(new=curr_bound, shift=curr_shift, dtype='int16')
        shifted_vesicle_ids.append(numpy.array(v_ids) + curr_shift)
        if shift is None:
            curr_shift = None
        else:
            curr_shift += shift

    return bound, shifted_vesicle_ids
    
################################################################
#
# Output
#

def get_out_pickle_name(name=''):
    """
    Returns output pickle file name
    """

    # extract root from the image_file_name
    base, root = get_image_base()

    # figure out hierarchy file name
    pkl_base = pkl_prefix + root + '_' + name + pkl_suffix
    pkl_file_name = os.path.join(pkl_directory, pkl_base)

    return pkl_file_name

def write_pickle(segment, name=''):
    """
    Writes (pickles) a Segment object
    """

    # file name
    pkl_file_name = get_out_pickle_name(name=name)

    # write 
    out_file = open(pkl_file_name, 'wb')
    pickle.dump(segment, out_file, -1)
    out_file.close()

def get_image_base():
    """
    Returns base and root of the image file name
    """
    (dir, base) = os.path.split(image_file_name)
    (root, ext) = os.path.splitext(base)
    return base, root

def get_labels_base():
    """
    Returns base and root of the labels file name
    """
    (dir, base) = os.path.split(labels_file_name)
    (root, ext) = os.path.splitext(base)
    return base, root

def open_results():
    """
    Opens a results file name and returns it.
    """
    
    # extract root from the labels_file_name
    base, root = get_image_base()

    # figure out results file name
    res_base = results_prefix + root + results_suffix
    res_file_name = os.path.join(results_directory, res_base)

    # open file
    res_file = open(res_file_name, 'w')
    return res_file

def machine_info():
    """
    Returns machine name and machine architecture strings
    """
    mach = platform.uname() 
    mach_name = mach[1]
    mach_arch = str([mach[0], mach[4], mach[5]])

    return mach_name, mach_arch

def write_res(ves, dist, mem, lum, file_name, ids=None, multi_ves_ids=None):
    """
    Writes results
    """

    # check ids
    if ids is None:
        ids = pyto.util.nested.flatten(multi_ves_ids)

    # machine info
    mach_name, mach_arch = machine_info()

    # get file names
    ves_pkl_file_name = get_out_pickle_name('vesicles')
    mem_pkl_file_name = get_out_pickle_name('mem')
    lum_pkl_file_name = get_out_pickle_name('lum')
    in_file_name = sys.modules[__name__].__file__

    # file times
    image_time = '(' + time.asctime(time.localtime(os.path.getmtime(
                image_file_name))) + ')'
    ves_pkl_time = time.asctime(time.localtime(os.path.getmtime(
                ves_pkl_file_name)))
    if membrane_thick is not None:
        mem_pkl_time = time.asctime(time.localtime(os.path.getmtime(
                    mem_pkl_file_name)))
        lum_pkl_time = time.asctime(time.localtime(os.path.getmtime(
                    lum_pkl_file_name)))
    in_time = time.asctime(time.localtime(os.path.getmtime(in_file_name)))

    # vesicle (labels) file(s) name(s), time(s) and vesicle ids
    #if is_multi_boundaries():
    if common.is_multi_file(file_name=labels_file_name):
        vesicle_lines = [
            "#     " + l_file + " (" + 
            time.asctime(time.localtime(os.path.getmtime(l_file))) + ")"
            for l_file in labels_file_name]
        vesicle_lines.insert(0, "# Vesicles: ")
        vesicle_ids_lines = ["#     " + str(b_ids) for b_ids in multi_ves_ids]
        vesicle_ids_lines.insert(0, 
                                 "# Vesicle ids (shift = " + str(shift) + "): ")
    else:
        vesicles_time = time.asctime(time.localtime(
                os.path.getmtime(labels_file_name)))
        vesicle_lines = ["# Vesicles: ",
                   "#     " + labels_file_name + " (" + vesicles_time + ")"]
        vesicle_ids_lines = ["# Vesicle ids: ",
                       "#     " + str(ids)]

    # header
    header = ["#",
              "# Machine: " + mach_name + " " + mach_arch,
              "# Date: " + time.asctime(time.localtime()),
              "#",
              "# Image: " + image_file_name + " " + image_time]
    header.extend(vesicle_lines)
    header.extend(
        ["# Out pickle files:",
         "#     - vesicles: " + ves_pkl_file_name + " (" + ves_pkl_time + ")"])
    if membrane_thick is not None:
        header.extend(
            ["#     - membrane: " + mem_pkl_file_name 
             + " (" + mem_pkl_time + ")",
             "#     - lumen: " + lum_pkl_file_name + " (" + lum_pkl_time + ")"])
    header.extend(
        ["# Input script: " + in_file_name + " (" + in_time + ") " 
         + __version__,
         "# Working directory: " + os.getcwd(),
         "#"])
    header.extend(vesicle_ids_lines)
    header.extend([\
              "#",
              "# Membrane thickness: " + str(membrane_thick),
              "#",
              "# Distance region id: " + str(distance_id),
              "# Distance mode(s): " + str(distance_mode),
              "#",
              ""])
    for line in header: file_name.write(line + os.linesep)

    # make distance 
    dist_0 = ''
    dist_1 = ''
    if len(dist) > 0:
        if isinstance(distance_mode, (list, tuple)):
            d_modes = distance_mode
        else:
            d_modes = (distance_mode,)
        for dist_mode in d_modes:
            dist_0 += 'Distance '
            dist_1 += ' %6s  ' % dist_mode

    # write results table head
    tabHead = ["# Id      Vesicle density                  Membrane density              Interior density         Vesicle Membrane   " + dist_0 + "    Center              Radius         ",
               "#     mean    std    min    max        mean   std    min    max        mean   std    min    max                      " + dist_1 + " x     y     z   mean  std    min   max  "]
    tabHeadExt = ["    X-slice radius          Y-slice radius           Z-slice radius",
                  "mean std    min   max   mean std    min   max   mean std    min   max"]
    if do_slices:
        for [line, lineExt] in zip(tabHead, tabHeadExt):
            file_name.write(line + lineExt + os.linesep)
    else:
        for line in tabHead: file_name.write(line + os.linesep)

    # list of all output variables (starts with id id prependIndex = True)
    out_vars = [\
        ves.density.mean, ves.density.std, ves.density.min, ves.density.max,
        mem.density.mean, mem.density.std, mem.density.min, mem.density.max,
        lum.density.mean, lum.density.std, lum.density.min, lum.density.max,
        ves.mor.volume, mem.mor.volume]
    for curr_dist in dist:
        out_vars.append(curr_dist)
    out_vars += [\
        ves.mor.center[:,0], ves.mor.center[:,1], ves.mor.center[:,2],
        ves.mor.radius.mean, ves.mor.radius.std,
        ves.mor.radius.min, ves.mor.radius.max]
    out_format = "%3u  %6.2f %5.2f %6.2f %6.2f  %6.2f %5.2f %6.2f %6.2f  "\
        + "%6.2f %5.2f %6.2f %6.2f  %6u %6u" + '   %6.1f' * len(dist) \
        + " %5u %5u %5u %5.1f %5.2f %5.1f %5.1f"  
    if do_slices:
        for srad in ves.mor.sliceRadius:
            out_vars += [srad.mean, srad.std, srad.min, srad.max]
            out_format += "%5.1f %5.2f %5.1f %5.1f "

    # write the results for each vesicle
    resTable = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                        indices=ids, prependIndex=True)
    for line in resTable: file_name.write(line + os.linesep)

    # write results for all vesicles together
    if include_total:
        tot_line = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                            indices=[0], prependIndex=True)
        file_name.write(os.linesep + "# All together: " + os.linesep 
                        + "#" + tot_line[0] + os.linesep)

    # write background results
    bkgOutVars = [ves.bkg.mean, ves.bkg.std, ves.bkg.min, ves.bkg.max]
    bkgOutFormat = "# bkg %6.2f %5.2f %6.2f %6.2f"
    file_name.write((bkgOutFormat % tuple(bkgOutVars)) + os.linesep)

    # write total results
    totOutVars = [ves.tot.mean[-1], ves.tot.std[-1], 
                  ves.tot.min[-1], ves.tot.max[-1]]
    totOutFormat = "# tot %6.2f %5.2f %6.2f %6.2f"
    file_name.write((totOutFormat % tuple(totOutVars)) + os.linesep)

    # write the average values 
    file_name.write(os.linesep + "# Average values: " + os.linesep)
    avOutVars = [ves.meanDensity.mean[-1], ves.meanDensity.std[-1],
                 ves.meanDensity.min[-1], ves.meanDensity.max[-1],
                 mem.meanDensity.mean[-1], mem.meanDensity.std[-1],
                 mem.meanDensity.min[-1], mem.meanDensity.max[-1],
                 lum.meanDensity.mean[-1], lum.meanDensity.std[-1],
                 lum.meanDensity.min[-1], lum.meanDensity.max[-1],
                 ves.mor.meanRadius.mean, ves.mor.meanRadius.std,
                 ves.mor.meanRadius.min, ves.mor.meanRadius.max]
    avOutFormat = "# av  %6.2f %5.2f %6.2f %6.2f  %6.2f %5.2f %6.2f %6.2f" \
        + "  %6.2f %5.2f %6.2f %6.2f            " \
        + '         ' * len(dist) \
        + "                      %5.1f %5.2f %5.1f %5.1f"
    file_name.write((avOutFormat % tuple(avOutVars)) + os.linesep)

    # flush
    file_name.flush()


################################################################
#
# Main function
#
###############################################################

def main():
    """
    Main function
    """

    # log machine name and architecture
    mach_name, mach_arch = machine_info()
    logging.info('Machine: ' + mach_name + ' ' + mach_arch)
    logging.info('Begin (script ' + __version__ + ')')

    # read image and vesicles
    image = common.read_image(file_name=image_file_name)
    vesicles, nested_vesicle_ids = read_boundaries(
        check=check_vesicles, suggest_shape=image.data.shape)
    flat_vesicle_ids = pyto.util.nested.flatten(nested_vesicle_ids)
    image.useInset(inset=vesicles.inset, mode='absolute', intersect=False)

    # analyze vesicles
    logging.info('Starting vesicle analysis')
    vesicles = analyze_vesicles(image=image, ves=vesicles, 
                                ves_ids=flat_vesicle_ids)

    # calculate distance to distance_id
    if distance_id is not None:
        if isinstance(distance_mode, (tuple, list)):
            dist_mode = distance_mode
        else:
            dist_mode = (distance_mode,)
        dist = []
        for curr_dist_mode in dist_mode:
            curr_dist = vesicles.distanceToRegion(regionId=distance_id, 
                                  ids=flat_vesicle_ids, mode=curr_dist_mode)
            setattr(vesicles, curr_dist_mode + 'Distance', curr_dist)
            dist.append(curr_dist)
        vesicles.distanceId = distance_id
    else:
        dist = []
        #dist = numpy.zeros(vesicles.ids.max()+1) - 1

    # make and analyze membranes and lumens
    logging.info('Starting membrane and lumen analysis')
    mem, lum = analyze_membranes_lumens(image=image, ves=vesicles,
                                        ves_ids=flat_vesicle_ids)

    # adjust center for inset
    vesicles.mor.center += [sl.start for sl in vesicles.inset]

    # add attributes and pickle vesicles 
    vesicles.vesicleIds = numpy.asarray(flat_vesicle_ids)
    write_pickle(segment=vesicles, name='vesicles')

    # write results
    res_file = open_results()
    write_res(ves=vesicles, dist=dist, mem=mem, lum=lum, file_name=res_file,
              multi_ves_ids=nested_vesicle_ids)

    logging.info('Done')

# run if standalone
if __name__ == '__main__':
    main()
