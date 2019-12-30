#!/usr/bin/env python
"""

Gray value (density) analysis of parts of segments (typically layers) that are
in the vicinity of some specified regions.

$Id: neighborhood_grey.py 1009 2014-01-21 15:19:48Z vladan $
Author: Vladan Lucic 
"""
__version__ = "$Revision: 1009 $"

import sys
import os
import os.path
import time
import platform
import pickle
from copy import copy, deepcopy
import logging

import numpy
import scipy
import scipy.stats as stats

import pyto

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
image_file = "../3d/tomo.em"

# set to True if image file is a segmented file (all segments are labeled by 1) 
image_binary = False

###############################################################
#
# Segments file 
#
# Can be either a pickle file containing Segments object (extension pkl), or
# (one or more) file(s) containing data array.  
#
# In any case, segments used for density determination can be obtained either 
# directly from these file(s), or by making layers based on the segment(s) read
# from segment file(s) (see layers section). 
#
# If the file is a pickle file, parameters shape, data type, byte order and 
# array order are disregarded. If the file is in em or mrc format, these 
# parameters are not necessary and should be set to None. If any of these 
# parameters is specified, it will override the corresponding value specified 
# in the header.
#
# If multiple segments files are used, segments_file, segment_ids have to be 
# tuples (or lists) of the same lenght
#

# name of (one or more) file(s) containing segments. It can be a pickle file 
# containing Segment object, or (one or more) files containing segments 
# array(s).
#segments_file = 'segments.pkl'   # Segment object pickle file
segments_file = "segments.dat"   # one segments file
#segments_file = ("segments_1.dat", "segments_2.dat", ...) # multiple segments 

# segments file dimensions
segments_shape = (100, 120, 90)

# segments file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float16', 
# 'float64') 
segments_data_type = 'uint8'

# segments file byteOrder ('<' for little-endian, '>' for big-endian)
segments_byte_order = '<'

# segments file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis 
# fastest)
segments_array_order = 'FORTRAN'

# offset of segments in respect to the data 
segments_offset = None             # no offset
#segments_offset = [10, 20, 30]    # offset specified

# ids of all segments in the segments file that need to be kept 
#segment_ids = 1           # usual from for ma
segment_ids = range(1,24)

# id shift in each subsequent segments file (in case of multiple segment files) 
segments_shift = None    # shift is determined automatically
#segments_shift = 300

########################################################
#
# Layer parameters (set boundary_id_1 to None for no layers)
#

# segment id of a boundary 1 used to make layers
#boundary_id_1 = None       # don't make layers at all
boundary_id_1 = 1           # make layers

# segment id of a boundary 2 used to make layers
boundary_id_2 = None      # make layers from boundary_id_1
#boundary_id_2 = 4        # make layers between boundary_id_1 and boundary_id_2

# id of a region where layers are formed (if 0 make sure segment_ids has all 
# ids)
layers_mask_id = 22

# layer thickness
#layer_thickness = None   # thickness from num_layers (for layers between only)
layer_thickness = 1       # for layers from boundary_1 only

# number of layers (only if layer_thickness is None and if layers are formed 
# between boundary_id_1 and boundary_id_2 
num_layers = 6

# number of extra layers (on each side if layers between) formed on boundary
# (ies) and on extra_layers mask(s)
#num_extra_layers = 0    # no extra regions
num_extra_layers = 3

# ids of a region where extra layers are formed (in addition to boundary_id_1)
# (if 0 make sure segment_ids has all ids)
extra_layers_mask_id_1 = 23

# ids of a region where extra layers are formed (in addition to boundary_id_2)
# (if 0 make sure segment_ids has all ids)
extra_layers_mask_id_2 = None

###############################################################
#
# Regions file 
#
# Can be either a pickle file containing Segments object (extension pkl), or
# (one or more) file(s) containing data array.  
#
# If the file is a pickle file, parameters shape, data type, byte order and 
# array order are disregarded. If the file is in em or mrc format, these 
# parameters are not necessary and should be set to None. If any of these 
# parameters is specified, it will override the corresponding value specified 
# in the header.
#
# If multiple regions files are used, regions_file, regions_ids have to be 
# tuples (or lists) of the same lenght
#

# name of (one or more) file(s) containing regions. It can be a pickle file 
# containing Segment object, or (one or more) files containing regions array(s).
regions_file = 'regions.pkl'   # Segment object pickle file
regions_file = "regions.dat"   # one regions file
#regions_file = ("regions_1.dat", "regions_2.dat", ...)  # multiple regions

# regions file dimensions
regions_shape = (100, 120, 90)

# regions file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float16', 
# 'float64') 
regions_data_type = 'uint8'

# regions file byteOrder ('<' for little-endian, '>' for big-endian)
regions_byte_order = '<'

# regions file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis 
# fastest)
regions_array_order = 'FORTRAN'

# offset of regions in respect to the data 
regions_offset = None             # no offset
#regions_offset = [10, 20, 30]    # offset specified

# ids of segments in the regions file
region_ids = range(1,22)

# id shift in each subsequent regions file (in case of multiple region files) 
regions_shift = None    # shift is determined automatically
#regions_shift = 400

###############################################################
#
# Density related parameters
#

# diameter of a neighbourhood (a neighbourhood is formed as an intersection of a
# segmentand a ball of this radius centered at the segment element that is the 
# closest to a region
#neighbourhood_size = None        # no limit
neighbourhood_size = 5

# maximum allowed distance between a neighbourhood and a region
#max_distance_to_region = None    # no limit
max_distance_to_region = 20

# mode of calculating distance between a neighborhood and a region: 'min' for
# the minimal distance, 'center' for the min distance between region center
# and segment (also 'max', 'mean', 'median', see 
# pyto.segmentationSegment.distanceToRegion())
max_distance_mode = 'min'

###############################################################
#
# Output files: density (pickle) and results, The file names are formed as:
#
#    <out_directory>/<out_prefix> + image root + <out_suffix>  + extension
#

# out directory
out_directory = ''

# out file name prefix (no directory name)
out_prefix = ''

# out file name suffix
out_suffix = ''

# density file extension
density_ext = '.pkl'

# write neighborhood array in a file
write_layers = True 

# neighborhood file extension
layers_ext = '_layers.em'

# write neighborhood array in a file
write_hood = True 

# neighborhood file extension
hood_ext = '_hood.em'

# results file extension
results_ext = '.dat'


################################################################
#
# Functions
#
################################################################

###########################################
#
# Read image file
#

def read_image(binary=False):
    """
    Reads image file and returns an segmentation.Grey object
    """
    image_file_o = pyto.io.ImageIO()
    image_file_o.read(file=image_file)
    image = pyto.segmentation.Grey(data=image_file_o.data)

    if binary:
        image.data = numpy.where(image.data>0, 1, 0)

    return image

###########################################
#
# Read segments and regions files
#

def read_segments(file_name, offset=None, ids=None, byte_order=None,  
                   data_type=None, array_order=None, shape=None, shift=None):
    """
    Reads and cleans segments from a pickle of an image (array) file.
    """
                  
    if os.path.splitext(segments_file) == '.pkl':

        # read pickle file
        segments = pickle.load(open(segments_file))
        if ids is not None:
            segment.setIds(ids=ids)
        multi_ids = [ids]

    else:

        # read array (labels) file(s)
        segments, multi_ids = read_array_segments(file_name=file_name, 
                     ids=ids, byte_order=byte_order, data_type=data_type, 
                     array_order=array_order, shape=shape, shift=shift)

    return segments, multi_ids

def read_array_segments(
    file_name, offset=None, ids=None, byte_order=None,  
    data_type=None, array_order=None, shape=None, shift=None):
    """
    Reads and cleans segments (labels) file(s)
    """

    # read
    if is_multi_file(file_name=file_name):
        segments, multi_ids = read_multi_segments(
            file_name=file_name, ids=ids,
            byte_order=byte_order, data_type=data_type,
            array_order=array_order, shape=shape, shift=shift)
    else:
        segments = read_single_segments(
            file_name=file_name, ids=ids,
            byte_order=byte_order, data_type=data_type,
            array_order=array_order, shape=shape)
        multi_ids = ids

    # offset
    segments.offset = offset

    return segments, multi_ids

def read_multi_segments(file_name, ids=None, byte_order=None, data_type=None, 
                        array_order=None, shape=None, shift=None):
    """
    Reads, cleans and initializes segments form multiple labels files.
    """

    # read all labels files and combine them in a single Segment object
    segments = pyto.segmentation.Segment()
    curr_shift = 0
    shifted_segment_ids = []
    for (l_name, single_ids, s_ids) in zip(file_name, ids, segment_ids):
        curr_segments = pyto.segmentation.Segment.read(
            file=l_name, ids=single_ids,
            clean=True, byteOrder=byte_order, dataType=data_type,
            arrayOrder=array_order, shape=shape)
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
    Reads, cleans and initializes segments form a sigle labels file.
    """

    # read file and make a Segment object
    segments = pyto.segmentation.Segment.read(
        file=file_name, clean=True, ids=ids,
        byteOrder=byte_order, dataType=data_type,
        arrayOrder=array_order, shape=shape)

    return segments

def is_multi_file(file_name):
    """
    Returns True if maultiple files are given.
    """
    if isinstance(file_name, str):
        return False
    elif isinstance(file_name, tuple) or isinstance(file_name, list):
        return True
    else:
        raise ValueError, str(file_name) + " has to be aither a string (one " \
              + "file) or a tuple (multiple files)."    

###########################################
#
# Analysis
#

def make_layers(segments):
    """
    """
    
    # save inset
    #segment_full_inset = segment.inset
    
    if boundary_id_2 is None:

        # segments from
        layers =  segments.makeLayersFrom(
            bound=boundary_id_1, mask=layers_mask_id, 
            thick=layer_thickness, nLayers=num_layers,
            nExtraLayers=num_extra_layers, extra=extra_layers_mask_id_1) 

    else:

        # segments between
        layers, dist =  segments.makeLayersBetween(
            bound_1=boundary_id_1, 
            bound_2=boundary_id_2, mask=layers_mask_id, nLayers=num_layers,
            nExtraLayers=num_extra_layers, extra_1=extra_layers_mask_id_1, 
            extra_2=extra_layers_mask_id_2)

    return layers

###########################################
#
# Write files
#

def machine_info():
    """
    Returns machine name and machine architecture strings
    """
    mach = platform.uname() 
    mach_name = mach[1]
    mach_arch = str([mach[0], mach[4], mach[5]])

    return mach_name, mach_arch
       
def get_out_file(extension=''):
    """
    
    """

    # extract root from the image_file
    (im_dir, im_base) = os.path.split(image_file)
    (im_root, im_ext) = os.path.splitext(im_base)

    # figure out hierarchy file name
    out_base = out_prefix + im_root + out_suffix + extension
    out_file = os.path.join(out_directory, out_base)

    return out_file


def write_density(density, suffix):
    """
    """
    density_file = get_out_file(extension=suffix)
    pickle.dump(density, open(density_file, 'wb'), -1)

def write_label(label, suffix, inset=None):
    """
    Writes label file data. Before writing it, the data is converted to uint8
    if max id < 512, or to uint16 otherwise, and it is expanded to the given 
    inset. The file is written in the format corresponding to the extension,
    as given in arg suffix.
    """
    file_ = get_out_file(extension=suffix)
    if label.maxId < 512:
        label.data = numpy.asarray(label.data, dtype='uint8')
    else:
        label.data = numpy.asarray(label.data, dtype='uint16')
    if inset is not None:
        label.useInset(inset=inset, mode='abs', expand=True)
    label.write(file_)

def make_file_header(file_name, file_type, ids=None, extra=''):
    """
    """

    if file_name is None: return []

    if is_multi_file(file_name=file_name):

        # multi file
        lines = ["# " + file_type + ":"]
        lines.extend(
            ["#     " + name + " (" + 
             time.asctime(time.localtime(os.path.getmtime(name))) + ")"
             for name in file_name])
        if ids is not None:
            lines.extend(
                ["#     Ids:" + name + " (" + 
                 time.asctime(time.localtime(os.path.getmtime(name))) + ")"
                 for name in file_name])

    else:

        # single_file
        file_time = time.asctime(time.localtime(os.path.getmtime(file_name)))
        lines = ["# " + file_type + ": " + file_name + " (" + file_time + ") " \
                     + extra]

    return lines

def make_segment_ids(shifted_seg_ids=None):
    """
    """

    lines = []
    if boundary_id_1 is None:

        # segment ids
        if is_multi_file(file_name=segments_file):
            lines.append("# Segment ids:")
            lines.extend(
                ["#     " + str(one_ids) for one_ids in shifted_seg_ids])
        else:
            lines.append("# Segment ids: " + str(shifted_seg_ids))

    else:

        # layer ids
        lines.append("# Layer ids:")
        lines.append("#     - on layers region: %d - %d" %
                     (num_extra_layers + 1, num_extra_layers + num_layers)) 
        lines.append(
            "#     - extra layers:      %d - %d" % (1, num_extra_layers)) 

        if boundary_id_2 is not None:
            lines.append("#     - extra layers:      %d - %d" %
                         (num_extra_layers + num_layers + 1,
                          2 * num_extra_layers + num_layers)) 

    return lines
                    
def make_header(shifted_seg_ids, shifted_reg_ids, reg_ids):
    """
    """

    # machine info
    mach_name, mach_arch = machine_info()
    lines = ["#",
             "# Machine: " + mach_name + " " + mach_arch,
             "# Date: " + time.asctime(time.localtime()),
             "#"]

    # files
    image_type = "Image"
    if image_binary:
        image_type += " (binary)"
    lines.extend(make_file_header(file_name=image_file, file_type=image_type))
    lines.extend(make_file_header(file_name=segments_file, 
                                  file_type="Segments"))
    lines.extend(make_file_header(file_name=regions_file, file_type="Regions"))
    density_file = get_out_file(extension=density_ext)
    lines.extend(make_file_header(file_name=density_file, 
                                  file_type="Density (output)"))
    layers_file = get_out_file(extension=layers_ext)
    try:
        lines.extend(make_file_header(file_name=layers_file, 
                                      file_type="Layers (output)"))
    except OSError:
        pass
    hood_file = get_out_file(extension=hood_ext)
    try:
        lines.extend(make_file_header(file_name=hood_file, 
                                      file_type="Neighborhoods (output)"))
    except OSError:
        pass
    lines.extend(make_file_header(file_name=sys.modules[__name__].__file__,
                                  file_type="Input script", extra=__version__)) 
    lines.append("# Working directory: " + os.getcwd())

    # region ids:
    lines.append("#")
    if is_multi_file(file_name=regions_file):
        lines.append("# Region ids (input):")
        lines.extend(["#     " + str(one_ids) for one_ids in shifted_reg_ids])
    else:
        lines.append("# Region ids (input): " + str(shifted_reg_ids))
    lines.append("# Region ids (selected regions): " + str(reg_ids))
        
    # segment ids and layers
    lines.append("#")
    lines.extend(make_segment_ids(shifted_seg_ids=shifted_seg_ids))

    # parameters

    # neighborhood parameters
    lines.extend([\
        "#",
        "# Distance parameters:",
        "#     - neighbourhood size: " + str(neighbourhood_size),
        "#     - max distance to region: " + str(max_distance_to_region)])
                 
    return lines

def format_data(data, name, format, segment_ids=None, region_ids=None):

    lines = ["#"]

    # ids
    if segment_ids is None:
        segment_ids = range(1, data.shape[0]+1)
    n_segments = len(segment_ids)
    if region_ids is None:
        region_ids = range(1, data.shape[1]+1)

    # table head
    segment_ids_str = ''.join([('%5i' % id_) + '   ' for id_ in segment_ids])
    if boundary_id_1 is not None:
        lay_seg = 'Layers'
    else:
        lay_seg = 'Segments'
    lines.extend([\
        "# " + name + ":",
        "# Region           " + lay_seg,
        "#       " + segment_ids_str])

    # data for individual neighbourhoods and whole segments
    out_vars = [data[seg_id,:] for seg_id in segment_ids]
    out_format = ' %4u ' + ''.join([format] * len(segment_ids))
    #all_region_ids = region_ids + [0]
    lines.extend(pyto.io.util.arrayFormat(
            arrays=out_vars, format=out_format,
            indices=region_ids, prependIndex=True))

    return lines

def write_results(neighborhood, shifted_seg_ids, shifted_reg_ids):
    """
    """

    # notation
    hood_density = neighborhood.density
    seg_density = neighborhood.segmentDensity

    # set ids
    segment_ids = numpy.insert(hood_density.ids, 0, 0)
    region_ids = numpy.insert(hood_density.regionIds, 0, 0)

    # make header
    lines = ["#"]
    lines.extend(make_header(shifted_seg_ids=shifted_seg_ids,
                             shifted_reg_ids=shifted_reg_ids, 
                             reg_ids=hood_density.regionIds))

    # make data part
    lines.append("#")
    lines.extend(\
        format_data(
            data=hood_density.mean, name='Density mean', format=' %6.3f ',
            segment_ids=segment_ids, region_ids=region_ids))
    lines.append("#")
    lines.extend(\
        format_data(data=hood_density.std, name='Density standard deviation',
                    format=' %6.3f ', segment_ids=segment_ids, 
                    region_ids=region_ids))
    lines.append("#")
    lines.extend(\
        format_data(data=hood_density.min, name='Density min', format=' %6.3f ',
                    segment_ids=segment_ids, region_ids=region_ids))
    lines.append("#")
    lines.extend(\
        format_data(data=hood_density.max, name='Density max', format=' %6.3f ',
                    segment_ids=segment_ids, region_ids=region_ids))
    lines.append("#")
    lines.extend(\
        format_data(data=hood_density.volume, name='Volume', format=' %6i ',
                    segment_ids=segment_ids, region_ids=region_ids))
    lines.append("#")
    lines.extend(\
        format_data(data=hood_density.t_value, 
                    name="Student's t-value between neighbourhood and segment " \
                        'means', format=' %6.3f ',
                    segment_ids=segment_ids, region_ids=region_ids))
    lines.append("#")
    confidence = hood_density.confidence * numpy.sign(hood_density.t_value)
    lines.extend(\
        format_data(
            data=confidence, 
            name='Confidence level that neighbourhood mean is different '\
                + 'from the segment mean (two tails t-test), negative if '\
                + 'neighborhood mean < segment mean', 
            format=' %6.3f ', 
            segment_ids=segment_ids, region_ids=region_ids))

    # adjust segment density array for format_data
    seg_density.expand(axis=1)

    # add segment density data
    lines.append("#")
    lines.append("Whole layers")
    lines.extend(\
        format_data(data=seg_density.mean, name='Segment density mean', 
                    format=' %6.3f ', segment_ids=segment_ids, region_ids=[0]))
    lines.extend(\
        format_data(data=seg_density.std, name='Segment density std', 
                    format=' %6.3f ', segment_ids=segment_ids, region_ids=[0]))
    lines.extend(\
        format_data(data=seg_density.min, name='Segment density min', 
                    format=' %6.3f ', segment_ids=segment_ids, region_ids=[0]))
    lines.extend(\
        format_data(data=seg_density.max, name='Segment density max', 
                    format=' %6.3f ', segment_ids=segment_ids, region_ids=[0]))
    lines.extend(\
        format_data(data=seg_density.volume, name='Segment volume', 
                    format=' %6i ', segment_ids=segment_ids, region_ids=[0]))
    

    # write
    res_file = open(get_out_file(extension=results_ext), 'w')
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
    mach_name, mach_arch = machine_info()
    logging.info('Machine: ' + mach_name + ' ' + mach_arch)
    logging.info('Begin (script ' + __version__ + ')')

    # read and clean segments 
    logging.info("Reading segments")
    segments, shifted_segment_ids = read_segments(
        file_name=segments_file, offset=segments_offset, 
        ids=segment_ids, byte_order=segments_byte_order,  
        data_type=segments_data_type, array_order=segments_array_order, 
        shape=segments_shape, shift=segments_shift)
    segments_full_inset = segments.inset

    # make layers if needed 
    if boundary_id_1 is not None:
        logging.info("Making layers from segments")
        segments = make_layers(segments=segments)
        shifted_segment_ids = segments.ids

    # read image and read and clean regions
    logging.info("Reading image and regions")
    image = read_image(binary=image_binary)
    regions, shifted_region_ids = read_segments(
        file_name=regions_file, offset=segments_offset, 
        ids=region_ids, byte_order=regions_byte_order,  
        data_type=regions_data_type, array_order=regions_array_order, 
        shape=regions_shape, shift=regions_shift)

    # calculate both hood and whole segment densities
    logging.info("Calculating density")
    neighborhood = pyto.scene.Neighborhood.make(
        image=image, segments=segments, regions=regions, 
        size=neighbourhood_size, 
        maxDistance=max_distance_to_region, distanceMode=max_distance_mode, 
        removeOverlap=True)

    # write hood density pickle
    logging.info("Writing output files")
    write_density(density=neighborhood, suffix=density_ext)

    # write layers and neighborhood arrays
    if (boundary_id_1 is not None) and write_layers:
        write_label(label=segments, suffix=layers_ext, 
                    inset=segments_full_inset)
    if write_hood:
        write_label(label=neighborhood.hood, suffix=hood_ext, 
                    inset=segments_full_inset)
    
    # write results
    write_results(
        neighborhood=neighborhood, shifted_seg_ids=shifted_segment_ids,
        shifted_reg_ids=shifted_region_ids)
    logging.info('End')

# run if standalone
if __name__ == '__main__':
    main()

