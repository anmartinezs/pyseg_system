#!/usr/bin/env python
"""

Makes layers in a given region starting from one or between two specified 
boundaries. Then it analyzes ovelap of given segments with the layers and
determines the position of gegment centers in repect to the layers.  

Important notes:

  - In order to make setting up multiple scripts easier, parameters common 
to these scripts are expected to be read from tomo_info.py file. The location 
of this file is given as argument path of common.__import__(). These parameters
are set in this script in lines having the following form:

  if tomo_info is not None: boundaries_file = tomo_info.labels_file_name

Specifiying another value for the same parameter (here labels_file_name) 
overrides the value from tomo_info.

  - Incorrect id specification accounts for a majority of problems encountered 
while running this script. In case an exception is thrown, please check the ids
carefully first.

  - For many variables more than one option is given. Uncomment the desired one
and adjust the value if needed, and comment out the other options.

  - Currently (r997) division by 0 warnings are printed. These might be ignored
because they most likely arise from calculating denisty of a 0-volume segment
and are also a consequence of the changed behavior of scipy. 

$Id$
Author: Vladan Lucic 
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import zip
#from builtins import str
#from past.utils import old_div

__version__ = "$Revision$"

import sys
import os
import os.path
import time
import platform
import pickle
from copy import copy, deepcopy 
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
#image_file_name = "../3d/and-1-6.mrc"

###############################################################
#
# Boundaries file, used to make layers 
#
# Can be either an pickle file containing Segments object (extension pkl), or
# (one or more) file(s) containing data array.  
#
# If the file is a pickle file, parameters shape, data type, byte order and 
# array order are disregarded. If the file is in em or mrc format, these 
# parameters are not necessary and should be set to None. If any of these 
# parameters is specified, it will override the corresponding value specified 
# in the header.
#
# If multiple boundaries files are used, boundaries_file, boundary_ids have to 
# be tuples (or lists) of the same lenght
#

# name of (one or more) file(s) containing boundaries. It can be a pickle file 
# containing Segment object, or (one or more) files containing boundaries 
# array(s).
if tomo_info is not None: boundaries_file = tomo_info.labels_file_name
#boundaries_file = 'boundaries.pkl'   # Segment object pickle file
#boundaries_file = '../viz/reconstruction_and-1-6_int-1-label_vesicles-membrane_2AZ_psd.raw'   # one boundaries file
#boundaries_file = ("bound_1.dat", "bound_2.dat", "bound_3.dat")  # more boundaries

# boundaries file dimensions (size in voxels)
boundaries_shape = None   # shape given in header of the boundaries file (if em
                          # or mrc), or in the tomo (grayscale image) header   
#boundaries_shape = (512, 512, 190) # shape given here

# boundaries file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float64') 
if tomo_info is not None: boundaries_data_type = tomo_info.labels_data_type
#boundaries_data_type = 'uint8'

# boundaries file byteOrder ('<' for little-endian, '>' for big-endian)
boundaries_byte_order = '<'

# boundaries file array order ('F' for x-axis fastest, 'C' for z-axis 
# fastest)
boundaries_array_order = 'F'

# offset of boundaries in respect to the data 
boundaries_offset = None             # no offset
#boundaries_offset = [10, 20, 30]    # offset specified

# ids of all segments in the boundaries file 
if tomo_info is not None: boundary_ids = tomo_info.all_ids
#boundary_ids = 1           # 
#boundary_ids = [1,66,67,68,69] + range(2,15)

# id shift in each subsequent boundaries file (in case of multiple boundary 
# files) 
boundaries_shift = None    # shift is determined automatically
#boundaries_shift = 300

########################################################
#
# Layer parameters (set boundary_id_1 to None for no layers)
#

# segment id of a boundary 1 used to make layers
if tomo_info is not None: boundary_id_1 = tomo_info.distance_id
#boundary_id_1 = None       # don't make layers at all
#boundary_id_1 = 1           # make layers

# segment id of a boundary 2 used to make layers
boundary_id_2 = None       # make layers from boundary_id_1
#boundary_id_2 = 4         # make layers between boundary_id_1 and boundary_id_2

# one or more ids of regions where layers are formed (0 is not recommended)
# if using multiple bondary files, enter shifted ids here
if tomo_info is not None: 
    layers_mask_id = [tomo_info.segmentation_region] + tomo_info.vesicle_ids
#layers_mask_id = [69] + range(2,15)

# layer thickness
# Needed if boundary_2 is None, or if boundary_id_2 is not None and 
# num_layers is None
#layer_thickness = None    # thickness from num_layers (for layers between only)
layer_thickness = 1        # for layers from only

# number of layers 
# Needed if boundary_2 is None, or if boundary_id_2 is not None and 
# layer_thickness is None
num_layers = 200

# number of extra layers (on each side if layers between) formed on 
# boundary(ies) and on extra_layers mask(s)
#num_extra_layers = 0    # no extra regions
num_extra_layers = 0

# ids of a region where extra layers are formed (in addition to boundary_id_1),
# or None for no extra layers on the boundary 1 side
extra_layers_mask_id_1 = 67

# ids of a region where extra layers are formed (in addition to boundary_id_2),
# or None for no extra layers on the boundary 2 side
extra_layers_mask_id_2 = None

###############################################################
#
# Segments file 
#
# These segments are used to determine the overlap with the layers and
# the position of their centers in respect to the layers.
#
# Can be either one pickle file containing Segments object (extension pkl), or
# (one or more) file(s) containing data array.  
#
# If the file is a pickle file, parameters shape, data type, byte order and 
# array order are disregarded. If the file is in em or mrc format, these 
# parameters are not necessary and should be set to None. If any of these 
# parameters is specified, it will override the corresponding value specified 
# in the header.
#
# If multiple segments files are used, segments_file, segments_ids have to be 
# tuples (or lists) of the same lenght
#

# name of (one or more) file(s) containing segments. It can be a pickle file 
# containing Segment object, or (one or more) files containing segments 
# array(s).
if tomo_info is not None: segments_file = tomo_info.labels_file_name
#segments_file = 'segments.pkl'   # Segment object pickle file
#segments_file = 'segments.em'   # one segments file
#segments_file = ("segments_1.dat", "segments_2.dat", "segments_3.dat")  # more segments (experimental)

# segments file dimensions
segments_shape = boundaries_shape   # segments same as boundaries file
#segments_shape = None              # pickle
#segments_shape = (260, 260, 100)   # image

# segments file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float16', 
# 'float64') 
if tomo_info is not None: segments_data_type = tomo_info.labels_data_type
#segments_data_type = None
#segments_data_type = 'uint8'

# segments file byteOrder ('<' for little-endian, '>' for big-endian)
segments_byte_order = None
#segments_byte_order = '<'

# segments file array order ('F' for x-axis fastest, 'C' 
# for z-axis fastest)
segments_array_order = None
#segments_array_order = 'F'

# offset of segments in respect to the data 
segments_offset = boundaries_offset # segments same as boundaries file
#segments_offset = None             # no offset
#segments_offset = [10, 20, 30]     # offset specified

# ids of segments in the segments file
if tomo_info is not None: segment_ids = tomo_info.vesicle_ids
#segment_ids = None         # use all segments
#segment_ids = range(2,15) # use only these segments

# id shift in each subsequent segments file (in case of multiple segment files) 
segments_shift = boundaries_shift # segments same as boundaries file
#segments_shift = None    # shift is determined automatically
#segments_shift = 400

###############################################################
#
# Output files: layers and results, Results file name is formed as:
#
#    <results_directory>/<results_prefix> + segments root + <results_suffix>
#

# layers file (pickle, em or mrc file)
layers_file = 'layers.mrc'

# layers file data type (int16 for mrc, uint16 for em file)
layers_data_type = 'int16'

# results directory
results_directory = ''

# results file name prefix (no directory name)
results_prefix = ''

# results file name suffix
results_suffix = '_layers.dat'


################################################################
#
# Functions
#
################################################################

###########################################
#
# Read image file
#

def read_image():
    """
    Reads image file and returns an segmentation.Image object

    Depreciated
    """
    image_file_o = pyto.io.ImageIO()
    image_file_o.read(file=image_file_name)
    image = pyto.segmentation.Grey(data=image_file_o.data)
    return image

###########################################
#
# Read segments and boundaries files
#

def read_segments(
        file_name, ids=None, shape=None, suggest_shape=None, byte_order=None,  
        data_type=None, array_order=None, shift=None, offset=None):
    """
    Reads file(s) containing boundaries.

    Works on single and multiple segments files. In the latter case, ids
    are shifted so that they do not overlap and the segments from different 
    files are merged.

    The label file shape is determined using the first found of the following:
      - variable labels_shape
      - shape given in the labels file header (em im or mrc format) 
      - argument suggest_shape

    Arguments:
      - shape, suggest_shape: actual or suggested image shape (see above)

    Returns (segments, segment_ids) where:
      - segments: (Segment) segments, from single file or merged
      - segment_ids: (list of ndarrays) segment ids, each list element contains
      ids from one segments file (shifted in case of multiple files)
    """
                  
    if os.path.splitext(file_name)[-1] == '.pkl':

        # read pickle file
        #segments = common.read_pickle(file_name=file_name).toSegment()
        segments = common.read_pickle(file_name=file_name)
        if isinstance(segments,  pyto.segmentation.ThreshConn):
            segments = segments.toSegment()
        elif isinstance(segments, pyto.scene.SegmentationAnalysis):
            # necessary for cleft layers
            segments = segments.labels.toSegment()
        else:
            segments = segments.toSegment()
        if ids is not None:
            segments.setIds(ids=ids)
        multi_ids = [ids]

    else:

        # read array (labels) file(s)
        segments, multi_ids = common.read_labels(
            file_name=file_name, ids=ids, shift=shift, shape=shape, 
            suggest_shape=suggest_shape, 
            byte_order=byte_order, data_type=data_type, array_order=array_order,
            clean=True, offset=offset, check=False)

    return segments, multi_ids

def read_array_segments(
    file_name, offset=None, ids=None, byte_order=None,  
    data_type=None, array_order=None, shape=None, shift=None):
    """
    Reads segments (labels) file(s)

    Depreciated
    """

    # read
    if common.is_multi_file(file_name=file_name):
        segments, multi_ids = read_multi_segments(
            file_name=file_name, ids=ids, byte_order=byte_order, 
            data_type=data_type, array_order=array_order, shape=shape, 
            shift=shift)
    else:
        segments = read_single_segments(
            file_name=file_name, ids=ids, byte_order=byte_order, 
            data_type=data_type, array_order=array_order, shape=shape)
        multi_ids = [ids]

    # offset
    segments.offset = offset

    return segments, multi_ids

def read_multi_segments(file_name, ids=None, byte_order=None, data_type=None, 
                        array_order=None, shape=None, shift=None):
    """
    Reads and initializes segments form multiple labels files.

    Depreciated
    """

    # read all labels files and combine them in a single Segment object
    segments = pyto.segmentation.Segment()
    curr_shift = 0
    shifted_segment_ids = []
    for (l_name, single_ids, s_ids) in zip(file_name, ids, segment_ids):
        curr_segments = pyto.segmentation.Segment.read(
            file=l_name, ids=single_ids, clean=True, byteOrder=byte_order, 
            dataType=data_type, arrayOrder=array_order, shape=shape)
        segments.add(new=curr_segments, shift=curr_shift, dtype='int16')
        shifted_segment_ids.append(numpy.array(s_ids) + curr_shift)
        if shift is None:
            curr_shift = None
        else:
            curr_shift += shift

    return segments, shifted_segment_ids
    
def read_single_segments(file_name, ids=None, byte_order=None, data_type=None, 
                         array_order=None, shape=None):
    """
    Reads and initializes segments form a sigle labels file.

    Depreciated
    """

    # read file and make a Segment object
    segments = pyto.segmentation.Segment.read(
        file=file_name, clean=True, ids=ids, byteOrder=byte_order, 
        dataType=data_type, arrayOrder=array_order, shape=shape)

    return segments

###########################################
#
# Analysis
#

def make_layers(segments, mask):
    """
    """
    
    # save inset
    #segment_full_inset = segment.inset
    
    if boundary_id_2 is None:

        # segments from
        layers =  segments.makeLayersFrom(
            bound=boundary_id_1, mask=mask, thick=layer_thickness, 
            nLayers=num_layers, nExtraLayers=num_extra_layers, 
            extra=extra_layers_mask_id_1) 

    else:

        # segments between
        layers, dist =  segments.makeLayersBetween(
            bound_1=boundary_id_1, bound_2=boundary_id_2, mask=mask, 
            nLayers=num_layers, nExtraLayers=num_extra_layers, 
            extra_1=extra_layers_mask_id_1, extra_2=extra_layers_mask_id_2)

    return layers

def analyze(image, layers, segments):
    """
    """

    # layer density and volume
    layer_density = pyto.segmentation.Density()
    layer_density.calculate(image=image, segments=layers)

    # segments centers
    seg_mor = pyto.segmentation.Morphology(segments=segments)
    seg_mor.getCenter()

    # count segment centers in each layer
    segment_count = numpy.zeros(layers.maxId+1, dtype='int')
    for seg_id in segments.ids:
        layer_id = layers.data[tuple(seg_mor.center[seg_id])]
        segment_count[layer_id] += 1

    # get density of segment centers in each volume
    segment_count_per_volume = (1. * segment_count) / layer_density.volume

    # get volume occupied by all segments for each layer
    segments_vol = numpy.zeros(layers.maxId+1, dtype='int') - 1
    segments_bin = segments.data > 0
    for lay_id in layers.ids:
        segments_vol[lay_id] = (segments_bin & (layers.data == lay_id)).sum()

    # get a fraction of each layer volume that is occupied by segments
    overlap = numpy.true_divide(segments_vol, layer_density.volume)

    #
    return layer_density, segment_count, segment_count_per_volume, overlap

###########################################
#
# Write files
#

def get_results_file(extension=''):
    """
    
    """

    # extract root from the segments file, if multiple use only the first 
    if isinstance(segments_file, tuple):
        seg_file = segments_file[0]
    else:
        seg_file = segments_file
    (dir, base) = os.path.split(seg_file)
    (root, ext) = os.path.splitext(base)

    # figure out hierarchy file name
    results_base = results_prefix + root + results_suffix + extension
    results_file = os.path.join(results_directory, results_base)

    return results_file

def write_layers(layers):
    """
    """

    ext = os.path.splitext(layers_file)
    if ext == '.pkl':

        # pickle
        pickle.dump(layers, open(layers_file, 'w'))

    else:

        # write image file
        layers.data = numpy.asarray(layers.data, dtype=layers_data_type)
        layers.write(file=layers_file)

def format_layer_ids():
    """
    """

    lines = []
    lines.append("# Layer ids:")
    lines.append("#     - layers region: %d - %d" %
                 (num_extra_layers + 1, num_extra_layers + num_layers)) 
    lines.append("#     - extra layers:      %d - %d" % (1, num_extra_layers)) 

    if boundary_id_2 is not None:
        lines.append("#     - extra layers:      %d - %d" %
                     (num_extra_layers + num_layers + 1,
                      2 * num_extra_layers + num_layers)) 

    return lines
                    
def format_header(nested_segment_ids):
    """
    """

    lines = []

    # machine info
    mach_name, mach_arch = common.machine_info()
    lines.append("# Machine: " + mach_name + " " + mach_arch)

    # files
    lines.extend(common.format_file_info(name=image_file_name, 
                                         description="Image"))
    lines.extend(common.format_file_info(name=boundaries_file, 
                                         description="Boundaries"))
    lines.extend(common.format_file_info(name=segments_file, 
                                         description="Segments"))
    lines.extend(common.format_file_info(name=layers_file, 
                                         description="Layers (output)"))
    lines.extend(common.format_file_info(name=sys.modules[__name__].__file__,
                                         description="Input script"))
    lines.append("# Working directory: " + os.getcwd())

    # segment ids:
    lines.append("#")
    if common.is_multi_file(file_name=segments_file):
        lines.append("# Segment ids:")
        lines.extend(["#     " + str(one_ids) 
                      for one_ids in nested_segment_ids])
    else:
        lines.append("# Segment ids: " + str(nested_segment_ids))
        
    # layer ids
    lines.append('#')
    lines.extend(format_layer_ids())

    # parameters

    return lines


def format_data(
    layers_density, segment_count, segment_count_per_volume, overlap, 
    layer_ids):
    """
    Formats data section of result file.
    """

    lines = ["#"]

    # table head
    lines.extend([\
        "#                  Layers                          Segments      ",
        "# Id            Grey value        Volume   Num Num/Vol  Vol fract",
        "#       mean   std    min    max                                 "]) 

    # data for individual neighbourhoods and whole segments
    out_vars = [layers_density.mean, layers_density.std, layers_density.min,
                layers_density.max, layers_density.volume,
                segment_count, segment_count_per_volume, overlap]
    out_format = '%4u  %6.2f %6.3f %6.2f %6.2f  %5d %4d %8.6f %7.5f'  
    lines.extend(pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                          indices=layer_ids, prependIndex=True))

    return lines

def write_results(layer_density, layer_ids, nested_segment_ids, segment_count,
                  segment_count_per_volume, overlap):

    # make header
    lines = ["#"]
    lines.extend(format_header(nested_segment_ids=nested_segment_ids))

    # make data part
    lines.append("#")
    lines.extend(format_data(layers_density=layer_density, layer_ids=layer_ids,
                             segment_count=segment_count,
                             segment_count_per_volume=segment_count_per_volume,
                             overlap=overlap))

    # write
    res_file = open(get_results_file(), 'w')
    for line in lines:
        res_file.write(line + os.linesep)
    res_file.flush()


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
    mach_name, mach_arch = common.machine_info()
    logging.info('Machine: ' + mach_name + ' ' + mach_arch)
    logging.info('Begin (script ' + __version__ + ')')

    # read image
    image = common.read_image(file_name=image_file_name)

    # read boundaries
    boundaries, nested_boundary_ids = read_segments(
        file_name=boundaries_file, offset=boundaries_offset, ids=boundary_ids, 
        byte_order=boundaries_byte_order, data_type=boundaries_data_type, 
        array_order=boundaries_array_order, shape=boundaries_shape,
        suggest_shape=image.data.shape, shift=boundaries_shift)
    boundaries_full_inset = boundaries.inset

    # make layers
    logging.info('Making layers')
    if isinstance(layers_mask_id, tuple):
        layers_mask =  pyto.util.nested.flatten(layers_mask_id)
    else:
        layers_mask = layers_mask_id
    layers = make_layers(segments=boundaries, mask=layers_mask)
    layers.useInset(
        inset=boundaries.inset, mode='abs', useFull=True, expand=True)

    # pickle layers
    write_layers(layers)

    # read segments
    segments, nested_segment_ids = read_segments(
        file_name=segments_file, offset=segments_offset, ids=segment_ids, 
        byte_order=segments_byte_order, data_type=segments_data_type, 
        array_order=segments_array_order, shape=segments_shape, 
        suggest_shape=image.data.shape, shift=segments_shift)
    segments.useInset(
        inset=boundaries.inset, mode='abs', useFull=True, expand=True)

    # analyze
    logging.info('Starting analysis')
    lay_dens, seg_count, seg_count_vol, overlap = \
              analyze(image=image, layers=layers, segments=segments)

    # write
    write_results(
        layer_density=lay_dens, layer_ids=layers.ids, segment_count=seg_count,
        segment_count_per_volume=seg_count_vol, overlap=overlap,
        nested_segment_ids=nested_segment_ids)

    logging.info('Done')

# run if standalone
if __name__ == '__main__':
    main()

