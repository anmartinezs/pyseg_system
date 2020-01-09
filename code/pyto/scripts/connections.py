#!/usr/bin/env python
"""
Depreciated. Please use connectors.py instead.

Makes segmentation based on threshold and connectivity, and finds connections
(segments) that contact specified boundaries.

The main applcations of this script are the following:

  1) Find and analyze connections between given boundaries, such as between synaptic
  vesicles. Each connections is typically required to make contacts with at least
  two boundaries.

  2) Find and analyze connections that contact a given boundary, such as a cell
  membrane. Each connection is required to make a contact with one boundary.

  In this case layers along the boundary are made 

This script may be placed anywhere in the directory tree.

$Id: connections.py 937 2013-01-29 11:39:05Z vladan $
Author: Vladan Lucic 
"""
__version__ = "$Revision: 937 $"

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
image_file_name = "../3d/tomo.em"

###############################################################
#
# Labels file, specifies boundaries and possibly other regions. If the file
# is in em or mrc format shape, data type, byte order and array order are not
# needed (should be set to None). If these variables are specified they will
# override the values specified in the headers.
#
# If multiple labels files are used, labels_file_name, all_ids and boundary_ids
# have to be tuples (or lists) of the same lenght
#

# name of (one or more) labels file(s) containing boundaries
labels_file_name = "labels.dat"   # one labels file
#labels_file_name = ("labels_1.dat", "labels_2.dat", "labels_3.dat")  # more labels

# labels file dimensions
labels_shape = (100, 120, 90)

# labels file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float16', 'float64') 
labels_data_type = 'uint8'

# labels file byteOrder ('<' for little-endian, '>' for big-endian)
labels_byte_order = '<'

# labels file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis fastest)
labels_array_order = 'FORTRAN'

# offset of labels in respect to the data 
labels_offset = None             # no offset
#labels_offset = [10, 20, 30]    # offset specified

###########################################################
#
# Segmentation parameters
#

# if True connections are read from connections file(s) If False, connections
# are obtained by segmentation (labeling) of the image. The connection file
# name(s) is (are) specified in the connections file name section below. 
read_connections = False

# threshold list 
threshold = numpy.arange(-0.2, -0.09, 0.02)  

# in addition to threshold list thresholds are also chosen dynamically, such that
#the number of new connections at each level is limited to this number
#max_new = 2        # dynamical thresholds
max_new = None     # thresholds from threshol list 

# smallest threshold step allowed for the dynamical threshold scheme
min_step = 0.001       

# if True checks if new levels (thresholds) fit the existing levels
check_hierarchy = False

###############################################################
#
# Structuring element connectivities. 
#

# segment determination (can be 1-3, 4 is experimental)
struct_el_connectivity = 1

# detection of contacts between segments and boundaries
contact_struct_el_connectivity = 1

# counting contacts between segments and boundaries (currently not used)
count_struct_el_connectivity = 1

###########################################################
#
# Boundaries and other segments in labels
#

# ids of all segments for single labels file, segments with other ids are removed 
#all_ids = [2,3,5,6]       # individual ids, single file
all_ids = range(2,64)  # range of ids, single file
#all_ids = None         # all ids are to be used, single file

# ids of all segments for multiple labels files can use all above forms,
# the following means ids 2, 3 and 6 from labels_1, all ids from labels_2 and
# ids 3, 4, ... 8 from labels_3
#all_ids = ([2,3,5,6], None, range(3,9))  # multiple files

# ids of all boundaries. All formats available for all_ids are accepted here also.
# In addition, nested list can be used where ids in a sublist are uderstood in the
# "or" sense, that is all boundaries listed in a sublist form effectivly a single
# boundary
#boundary_ids = [2,3,5]       # individual ids, single file
boundary_ids = range(2,64)  # range of ids, single file
#boundary_ids = None         # all segments are to be used, single file
#boundary_ids = [[2,3], 4, 5, 6]  #  2 and 3 taken together, single file

# ids of all boundaries for multiple labels files. All formats available for
# boundary_ids are accepted here also.
#boundary_ids = ([2,3,5], range(2,64), [[6,7], 8])  # multiple files

# id shift in each subsequent labels (in case of multiple labels files) 
shift = None     # shift is determined automatically
#shift = 300

# number of boundaries that each segment is required to contact
n_boundary = 2      # for connections between 2 boundaries
#n_boundary = 1      # for connections contacting one boundary

# 'exact' to require that segments contact exactly n_boundary boundaries, or
# 'at_least' to alow n_boundary or more contacted boundaries 
count_mode = 'at_least'

# id of region where connections can be formed, has to be the same in all
# labels files. Not used if read_connections is True.
conn_region = 0

# Connections are formed in the area surrounding boundaries (given by all ids
# specified in boundary_ids) that are not further than free_size from the
# boundaries and that are still within conn_region. If free_size = 0, the area
# surrounding boundaries is maximaly extended (to the limits of conn_region).
# Not used if read_connections is True.
free_size = 10  # the same for all boundaries, one labels file
#free_size = [10, 12, 15, 12, 10] # one for each boundary

# Defines the manner in which the areas surrounding individual boundaries are
# combined. These areas can be added (free_mode='add') or intersected
# (free_mode='intersect'). Not used if read_connections is True. 
free_mode = 'add'

# If None, the region where the connections are made is determined by using all
# boundaries together. Alternatively, if an iterable of lists is
# given the segmentation is done in a free region established between
# (usually two) boundaries given in each element of this list.
# Works only for single labels file.
#boundary_lists = pyto.util.probability.combinations(elements=boundary_ids,
#                                                   size=n_boundary,
#                                                   repeat=False, order=False)
# simpler version of the above
#boundary_lists = [(2,3), (2,6), (3,6)]
boundary_lists = None

###########################################################
#
# Analysis
#

# calculate segment length (if length already exists in the input hierarchy and
# all length-related parameters are the same the old length values are kept)
do_length = False

# segment length calculation mode:
#   - (one boundary) 'b-max' or 'c-max': between contacts and the most distant
#     element of a segment, where contacts belong to a boundary or a segment,
#     respectivly  
#   - (two boundaries) 'b2b' and 'c2c: minimim distance that connect a contact with one
#     boundary, element of a segment on the central layer between the boundaries, and
#     a contact with the other boundary, where contacts belong to a boundary or a
#     segment, respectivly
length_mode = 'c2c'

# calculate distance between connected boundaries 
# Note: used only for connections that contact exactly two boundaries 
do_boundary_distance = True

# calculate topological properties
do_topology = True

# id(s) of a region(s) in labels to which distances from each segment are caluculated.
# If more than one distance region is specified, for each segment the distance to
# the closest region is calculated. In case of multiple labels files this id is
# understood after the ids of the labels files are shifted.
#distance_id = None   # get distances are not calculated
distance_id = 114   # get distance between each segment and one region
#distance_id = range(2,20)   # get distance between each segment and the closest region

# the way distance is calculated ('center', 'mean' or 'min')
distance_mode = 'mean' 

############################################################
#
#
# Ordering of connections (only the first True ordering is applied)

# order segments by the boundaries contacted
order_by_boundaries = True

# order segments by volume
order_by_volume = False

###########################################################
#
# Hierarchy file. The file name is formed as:
#
#   <hi_directory>/<hi_prefix> + image root + <hi_suffix>
#

# write segmentation hierarchy to a file
write_hierarchy = True

# hierarchy directory
hi_directory = ''

# hierarchy file name prefix (no directory name)
hi_prefix = ''

# include image file root (filename without directory and extension)
insert_root_hi = True

# hierarchy file name suffix
hi_suffix = ".pkl"

# read hierarchy (perhaps better to use classify_connections)
read_hierarchy = False

# name of the input hierarchy file 
hierarchy_file_name = ''

###########################################################
#
# Connections file. If connections_file_name is None, the file name is formed as:
#
#   <conn_directory>/<conn_prefix> + image root + _tr-<threshold> + <conn_suffix>
#

# write connections to a file at each threshold
write_connections = False

# If given, and if read_connections is True, the connections are read from this
# file name and the remaining variables in this section are not used. If None, the
# connections file name is determined as specified above.
# Useful for single threshold only.
#connections_file_name = "conn.em"  # use this file as input
connections_file_name = None       # name defined below

# connections directory
conn_directory = ''

# connections file name prefix (no directory name)
conn_prefix = ''

# include image file root (filename without directory and extension)
insert_root_conn = True

# connections file name suffix
conn_suffix = ".em"

# number of decimal points for threshold (used for forming file names)
threshold_precision = 3

# connections data type, 'uint8' , or
# conn_data_type = 'uint8'        # if max segment id is not bigger than 255
conn_data_type = 'uint16'         # more than 255 segments

############################################################
#
# Results file. If results_file_name is None, the results file name is formed as:
#
#   <res_directory>/<res_prefix> + image root + _tr-<threshold> + <res_suffix>
#

# if True at each threshold the results are written to a file 
write_results = True

# File name for results. If given the remaining variables in this section are
# not used. Otherwise, the results file name is determined as specified above.
# Useful for single threshold only.
#results_file_name = "res.dat"  # use this name
results_file_name = None       # get name from connections section

# results directory
res_directory = ''

# results file name prefix (without directory)
res_prefix = ''

# include image file root (filename without directory and extension)
insert_root_res = True

# results file name suffix
res_suffix = "_con.dat"

# include total values (for all segments taken together) in the results, id 0
include_total = True


################################################################
#
# Work
#
################################################################

###########################################
#
# Read image and boundary files
#

def read_image():
    """
    Reads image file and returns an segmentation.Image object
    """
    image_file = pyto.io.ImageIO()
    image_file.read(file=image_file_name)
    image = pyto.segmentation.Grey(data=image_file.data)
    return image

def read_boundaries():
    """
    Reads labels file(s), makes (Segment) boundaries and makes inset.
    """

    # read
    if is_multi_boundaries():
        bound, multi_boundary_ids = read_multi_boundaries()
    else:
        bound = read_single_boundaries()
        multi_boundary_ids = [boundary_ids]

    # offset
    bound.offset = labels_offset

    # make inset
#    if free_size > 0:
#        if conn_region > 0:
#            bound.makeInset(extend=0)
#        else:
#            bound.makeInset(extend=free_size)
#    else:
#        slices = ndimage.find_objects(bound.data==conn_region)
#        bound.useInset(slices[0])

    return bound, multi_boundary_ids

def is_multi_boundaries():
    """
    Returns True if maultiple boundaries (labels) files are given.
    """
    if isinstance(labels_file_name, str):
        return False
    elif isinstance(labels_file_name, tuple) or isinstance(labels_file_name, list):
        return True
    else:
        raise ValueError, "labels_file_name has to be aither a string (one " \
              + "labels file) or a tuple (multiple labels files)."    

def read_single_boundaries():
    """
    Reads and initializes boundaries form a sigle labels file.
    """

    # read labels file and make a Segment object
    bound = pyto.segmentation.Segment.read(file=labels_file_name, ids=all_ids,
               clean=True, byteOrder=labels_byte_order, dataType=labels_data_type,
               arrayOrder=labels_array_order, shape=labels_shape)

    return bound

def read_multi_boundaries():
    """
    Reads and initializes boundaries form a sigle labels file.
    """

    # read all labels files and combine them in a single Segment object
    bound = pyto.segmentation.Segment()
    curr_shift = 0
    shifted_boundary_ids = []
    for (l_name, a_ids, b_ids) in zip(labels_file_name, all_ids, boundary_ids):
        curr_bound = pyto.segmentation.Segment.read(file=l_name, ids=a_ids,
               clean=True, byteOrder=labels_byte_order, dataType=labels_data_type,
               arrayOrder=labels_array_order, shape=labels_shape)
        bound.add(new=curr_bound, shift=curr_shift, dtype='int16')
        shifted_boundary_ids.append(numpy.array(b_ids) + curr_shift)
        if shift is None:
            curr_shift = None
        else:
            curr_shift += shift

    return bound, shifted_boundary_ids
    
def read_hi():
    """
    Reads (unpickles) a hierarchy object.
    """

    # read
    in_file = open(hierarchy_file_name, 'rb')
    hi = pickle.load(in_file)
    in_file.close()

    return hi

def get_out_hierarchy_name():
    """
    Returns output hierarchy file name
    """

    # extract root from the image_file_name
    base, root = get_image_base()

    # figure out hierarchy file name
    hi_base = hi_prefix + root + hi_suffix
    hi_file_name = os.path.join(hi_directory, hi_base)

    return hi_file_name

def machine_info():
    """
    Returns machine name and machine architecture strings
    """
    mach = platform.uname() 
    mach_name = mach[1]
    mach_arch = str([mach[0], mach[4], mach[5]])

    return mach_name, mach_arch

def write_hi(hi, full_inset=None):
    """
    Writes (pickles) a hierarchy object
    """

    # file name
    hi_file_name = get_out_hierarchy_name()

    # make inset
    inset = hi.inset
    hi.makeInset()
    if full_inset is None:
        hi.fullInset = inset
    else:
        hi.fullInset = full_inset

    # write 
    out_file = open(hi_file_name, 'wb')
    pickle.dump(hi, out_file, -1)
    out_file.close()

def format_threshold(tr):
    """
    Makes threshold strings.
    """
    if tr is not None:
        tr_str = (('%6.' + str(threshold_precision) + 'f') % tr).strip()
        tr_long_str = "_tr-" + tr_str
    else:
        tr_str = ''
        tr_long_str = ''

    return tr_str, tr_long_str

def get_image_base():
    """
    Returns base and root of the image file name
    """
    (dir, base) = os.path.split(image_file_name)
    (root, ext) = os.path.splitext(base)
    return base, root

def get_connections_name(tr):
    """
    Returns the connections file name
    """

    # format threshold
    tr_str, tr_long_str = format_threshold(tr)

    # extract root from the image_file_name
    base, root = get_image_base()

    # determine connections file name 
    if connections_file_name is not None:
        conn_file_name = connections_file_name
    else:
        if insert_root_conn:
            conn_base = conn_prefix + root + tr_long_str + conn_suffix
        else:
            conn_base = conn_prefix + tr_Long_str + conn_suffix
        conn_file_name = os.path.join(conn_directory, conn_base)

    return conn_file_name
    
def read_conn(tr):
    """
    Reads connections from a file and return a segmentation object.
    """

    # get connections file name
    conn_file_name = get_connections_name(tr)

    # read
    seg = pyto.segmentation.Connected.read(file=conn_file_name)

    return seg

def open_results(tr):
    """
    Opens a results file name and returns it.
    """
    
    # format threshold
    tr_str, tr_long_str = format_threshold(tr)

    # extract root from the image_file_name
    base, root = get_image_base()

    # figure out results file name
    if results_file_name is None:    
        if insert_root_res:
            res_base = res_prefix + root + tr_long_str + res_suffix
        else:
            res_base = res_prefix + tr_long_str + res_suffix
        res_file_name = os.path.join(res_directory, res_base)
    else:
        res_file_name = results_file_name

    # open file
    res_file = open(res_file_name, 'w')
    return res_file

def analyze_segment(image, segment, contacts, boundary, bound_dist_db):
    """
    Calculates volume, surface and density of a segment
    """

    # get density stats
    logging.info("  density")
    dens = pyto.segmentation.Statistics(data=image.data, labels=segment.data,
                                        ids=segment.ids)
    dens.calculate()

    # calculate morphology
    logging.info("  morphology")
    mor = pyto.segmentation.Morphology(segments=segment.data)
    mor.getVolume()
    mor.getSurface()
    if do_length:
        logging.info("  length")
        mor.getLength(segments=segment, boundaries=boundary, contacts=contacts,
                      distance=length_mode)
    else:
        mor.length = None

    # calculate topology
    if do_topology:
        logging.info("  topology")
        topo = pyto.segmentation.Topology(segments=segment)
        topo.calculate()
    else:
        topo = None

    # calculate distance to region(s)
    if distance_id is not None:
        curr_dist = segment.distanceToRegion(region=boundary, regionId=distance_id, 
                                        ids=segment.ids, mode=distance_mode)

    # find distances to closest region
    if (distance_id is not None) and (curr_dist is not None):
        logging.info("  distance to region")
        dist = numpy.zeros(shape=segment.ids.max()+1) - 1
        dist_reg = numpy.zeros(shape=segment.ids.max()+1, dtype='int') - 1
        if curr_dist.ndim == 2:
            distance_ids = numpy.asarray(distance_id)
            dist_strip = curr_dist[distance_ids, :]
            dist_reg[segment.ids] = \
                  distance_ids[dist_strip.argmin(axis=0)][segment.ids]
            dist[segment.ids] = dist_strip.min(axis=0)[segment.ids]
        else:
            dist[segment.ids] = curr_dist[segment.ids]
            dist_reg[segment.ids] = distance_id
        
    else:
        dist = None
        dist_reg = None

    # calculate distance between connected boundaries
    bound_dist = None
    if do_boundary_distance:
        logging.info("  distance between connected boundaries")
        bound_dist = numpy.zeros(segment.maxId+1) - 1
        for seg_id in segment.ids:
            b_ids = contacts.findBoundaries(segmentIds=seg_id, nSegment=1)
            if len(b_ids) == 2:
                try:
                    bound_dist[seg_id] = bound_dist_db[(b_ids[0], b_ids[1])]
                except KeyError:
                    bound_dist[seg_id] = boundary.distanceToRegion(regionId=b_ids[0],
                                               ids=[b_ids[1]], mode='min')[b_ids[1]]
                    bound_dist_db[(b_ids[0], b_ids[1])] = bound_dist[seg_id]
                    bound_dist_db[(b_ids[1], b_ids[0])] = bound_dist[seg_id]
            else:
                bound_dist[seg_id] = -1

    else:
        bound_dist = None 

    return dens, mor, topo, dist, dist_reg, bound_dist, bound_dist_db

def write_res(seg, dens, mor, topo, dist, dist_reg, contacts, bound_dist, 
              res_file, ids=None, multi_b_ids=None):
    """
    Writes results for one level (threshold) in the results file.
    
    If ids is not given seg.ids is used instead.

    Arguments:
    """
    
    # check ids
    if ids is None:
        ids = seg.ids

    # format threshold
    tr = seg.threshold[0]
    tr_str, tr_long_str = format_threshold(tr)

    # machine info
    mach_name, mach_arch = machine_info()

    # get file names
    conn_file_name = get_connections_name(tr)    
    out_hi_file_name = get_out_hierarchy_name()
    in_file_name = sys.modules[__name__].__file__

    # file times
    image_time = \
        '(' + time.asctime(time.localtime(os.path.getmtime(image_file_name))) + ')'
    in_time = time.asctime(time.localtime(os.path.getmtime(in_file_name)))
    if write_hierarchy:
        try:
            hi_time = time.asctime(time.localtime(os.path.getmtime(out_hi_file_name)))
        except OSError:
            hi_time = "not yet"
        hi_line = "# Out hierarchy: " + out_hi_file_name + " (" + hi_time + ")"
    else:
        hi_line = "# Out hierarchy: not written"
    try:
        conn_time = time.asctime(time.localtime(os.path.getmtime(conn_file_name)))
    except OSError:
        conn_time = "not yet"

    # boundary (labels) file(s) name(s), time(s) and boundary ids
    if is_multi_boundaries():
        boundary_lines = ["#     " + l_file + " (" + \
                       time.asctime(time.localtime(os.path.getmtime(l_file))) + ")"\
                   for l_file in labels_file_name]
        boundary_lines.insert(0, "# Boundaries: ")
        boundary_ids_lines = ["#     " + str(b_ids) for b_ids in multi_b_ids]
        boundary_ids_lines.insert(0, "# Boundary ids (shift = " + str(shift) + "): ")
    else:
        labels_time = time.asctime(time.localtime(os.path.getmtime(labels_file_name)))
        boundary_lines = ["# Boundaries: ",
                   "#     " + labels_file_name + " (" + labels_time + ")"]
        boundary_ids_lines = ["# Boundary ids: ",
                       "#     " + str(boundary_ids)]

    # other file related info
    if read_connections:
        in_out = "input"
    elif write_connections:
        in_out = "output"
    else:
        in_out = "not used"
    if seg.structEl is None:
        se_conn = 'none'
        se_size = 'none'
    else:
        se_conn = str(seg.structEl.connectivity)
        se_size = str(seg.structEl.size)

    # results file header
    header = ["#",
        "# Machine: " + mach_name + " " + mach_arch,
        "# Date: " + time.asctime(time.localtime()),
        "#",
        "# Image: " + image_file_name + " " + image_time]
    header.extend(boundary_lines)
    header.extend([\
        "# Connections (" + in_out + "): " + conn_file_name + " (" + conn_time + ")",
        hi_line,
        "# Input script: " + in_file_name + " (" + in_time + ") " + __version__,
        "# Working directory: " + os.getcwd(),
        "#"])
    header.extend(boundary_ids_lines)
    header.extend([\
        "#",
        "# Number of boundaries contacted: " + str(n_boundary),
        "# Contact mode: " + count_mode,
        "#",
        "# Connection region: " + str(conn_region),
        "# Free region size: " + str(free_size),
        "# Free region mode: " + free_mode,
        "#",
        "# Segmentation structuring element:",
        "#   - connectivity: " + se_conn,
        "#   - size: " + se_size,
        "#",
        "# Connectivities of other structuring elements:",
        "#    - detect contacts: " + str(seg.contactStructElConn),
        "#    - topology: 1"])
    if do_length:
        header.extend([\
                "#",
                "# Length calculation mode: " + str(length_mode)])
    if distance_id is not None:
        header.extend([\
                "#",
                "# Distance:",
                "#   - region(s): " + str(distance_id),
                "#   - mode: " + distance_mode])
    header.extend([\
        "#",
        "# Threshold: " + tr_str,
        "#",
        "# Number of segments: " + str(len(ids)),
        "#"])
    for line in header: res_file.write(line + os.linesep)

    # write results table head
    tabHead = ["# Id           Density             Volume Surface ",
               "#         mean   std    min    max                " ]
    if mor.length is not None:
        tabHead[0] = tabHead[0] + ' Length '
        tabHead[1] = tabHead[1] + '        '
    if do_topology:
        tabHead[0] = tabHead[0] + 'Euler Loops Holes '
        tabHead[1] = tabHead[1] + '                  '
    if do_boundary_distance:
        tabHead[0] = tabHead[0] + 'Boundary '
        tabHead[1] = tabHead[1] + 'distance '
    if dist is not None:
        tabHead[0] = tabHead[0] + '  Distance   '
        tabHead[1] = tabHead[1] +('         to  ')  
    tabHead[0] = tabHead[0] + " Boundaries "
    for line in tabHead: res_file.write(line + os.linesep)

    # prepare results
    out_vars = [ dens.mean, dens.std, dens.min, dens.max, mor.volume, mor.surface]
    out_format = ' %6u %6.2f %5.2f %6.2f %6.2f %5u %5u  '
    if mor.length is not None:
        out_vars.append(mor.length)
        out_format = out_format + '  %5.1f '
    if do_topology:
        out_vars.extend([topo.euler, topo.nLoops, topo.nHoles])
        out_format = out_format + '%4i  %4i  %4i '
    if do_boundary_distance:
        out_vars.append(bound_dist)
        out_format = out_format + '  %6.1f  '        
    if dist is not None:
        out_vars.extend([dist, dist_reg])
        out_format = out_format + ' %6.1f %4i '

    # write the results
    resTable = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                        indices=ids, prependIndex=True)
    for (sid, line) in zip(ids, resTable):
        boundIds = numpy.array2string(contacts.findBoundaries(segmentIds=sid,
                                                              nSegment=1))
        res_file.write(line + "  %s" % boundIds + os.linesep)

    # write results for all connections together
    if include_total:
        try:
            tot_line = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                                indices=[0], prependIndex=True)
            res_file.write(os.linesep + tot_line[0] + os.linesep)
        except TypeError:
            print "Could not write total results for threshold " + tr_str + \
                  ". Continuing to the next threshold."

    res_file.flush()

def order(ids, contacts=None, volume=None):
    """
    Returns ordered ids.
    """
    if order_by_boundaries:
        sort_list = contacts.orderSegmentsByContactedBoundaries(argsort=True)
    elif order_by_volume:
        sort_list = volume[ids].argsort()
    else:
        return ids

    return ids[sort_list]


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

    # read image and boundaries
    image = read_image()
    bound, nested_boundary_ids = read_boundaries()
    bound_full_inset = bound.inset
    flat_boundary_ids = pyto.util.nested.flatten(nested_boundary_ids)

    # read or make segmentation
    if read_hierarchy:

            ##########################################
            #
            # Read hierarchy (already segmented)
            #
            tc = read_hi()

    elif read_connections:
        
        ###########################################
        #
        # Read connections and find contacts for all thresholds
        #

        # loop over thresholds
        if connections_file_name is not None:
            thresh = [None]
        else:
            thresh = threshold
        tc = pyto.segmentation.ThreshConn()
        for tr in thresh:

            # read connections and find contacts
            try:
                seg = read_conn(tr)
            except IOError:
                conn_file_name = get_connections_name(tr)
                print "Connections file " + conn_file_name \
                      + " not found. Continuing to the next file."
                continue

            # find contacts
            contacts = seg.findContacts(input=seg.data, boundary=bound,
                         boundaryIds=flat_boundary_ids, boundaryLists=boundary_lists,
                         label=False)

            # add to hierarchy
            props = {}
            props['contacts'] = contacts
            if tr is not None:
                props['threshold'] = tr
            tc.addLevel(segment=seg, props=props)

            # analyze?

        # write hierarchy
        if write_hierarchy: write_hi(tc)

    else:

        ##########################################
        #
        # Segment from scratch
        #

        # threshold and connectivity segmentation
        logging.info('Segmentation start')
        tc = pyto.segmentation.ThreshConn()
        tc.setConnParam(boundary=bound, structElConn=struct_el_connectivity,
                        contactStructElConn=contact_struct_el_connectivity,
                        countStructElConn=count_struct_el_connectivity,
                        boundaryIds=flat_boundary_ids, nBoundary=n_boundary,
                        boundCount=count_mode, mask=conn_region,
                        freeSize=free_size, freeMode=free_mode)
        if (max_new is None) or (max_new <= 0):
            tc.makeLevels(image=image, thresh=threshold, check=check_hierarchy)
        else:
            tc.makeByNNew(image=image, thresh=threshold, maxNew=max_new, minStep=min_step, check=check_hierarchy)

        # write hierarchy
        if write_hierarchy:
            #tc_inset = tc.inset
            #tc.useInset(inset=bound_full_inset, mode='abs', useFull=False, expand=True)
            #tc.saveFull()
            write_hi(tc)
            #tc.useInset(inset=tc_inset, mode='abs')

    ###############################################
    #
    # Analyze and write results for each level separately
    #
    
    # apply boundary inset to image
    logging.info('Preparing for analysis')
    tc.makeInset(extend=1)
    tc_inset = tc.inset 
    bound.useInset(inset=tc.inset, mode='abs')
    image.useInset(inset=bound.inset, mode='abs', useFull=True)

    # prepare for analysis
    mor = pyto.segmentation.Morphology()
    if do_topology:
        topo = pyto.segmentation.Topology()
    dist = numpy.zeros(tc.maxId+1, dtype='float')
    dist_reg = numpy.zeros(tc.maxId+1, dtype='int')
    bound_dist_db = {}

    # analyze from top level down to make it faster
    all_levels = range(tc.topLevel+1)
    all_levels.reverse()
    for level in all_levels:

        # extract current level
        tc_level = tc.extractLevel(level, new=True)

        # restrict contacts to ids belonging to the current class
        contacts = deepcopy(tc_level.contacts)
        # workaround for bug #906 in numpy 1.1.1
        if numpy.__version__ == '1.1.1':
            try:
                contacts._n._mask = deepcopy(tc_level.contacts._n._mask)
            except AttributeError:
                pass
        contacts.keepSegments(ids=tc_level.ids)

        # analyze
        segment = tc_level.toSegment(copy=False)
        logging.info("Analysis start for level %d (threshold %6.3f)", 
                      level, tc.threshold[level])    
        curr_dens, curr_mor, curr_topo, curr_dist, curr_dist_reg, \
            bound_dist, bound_dist_db \
                   = analyze_segment(image=image, segment=segment, 
                                     contacts=contacts, boundary=bound, 
                                     bound_dist_db=bound_dist_db)

        # add current to data structures 
        mor.merge(curr_mor)
        if do_topology:
            topo.merge(curr_topo)
        if curr_dist is not None:
            dist[segment.ids] = curr_dist[segment.ids]
            dist_reg[segment.ids] = curr_dist_reg[segment.ids]

        # write connections
        if write_connections:
            logging.info("  Write connections")
            conn_file_name = get_connections_name(tr=tc.threshold[level])
            tc_level.useInset(inset=bound_full_inset, mode='abs', useFull=False,
                              expand=True)
            #tc_level.saveFull()
            tc_level.write(file=conn_file_name, dataType=conn_data_type)
            tc_level.useInset(inset=tc_inset, mode='abs')

        # write results
        if write_results:
            logging.info("  Ordering")
            ordered_ids = order(segment.ids, contacts=contacts,
                                volume=curr_mor.volume)
            res_file = open_results(tr=tc.threshold[level])
            logging.info("  Write resultss")
            write_res(seg=tc_level, dens=curr_dens, mor=curr_mor, topo=curr_topo,
                      dist=curr_dist, dist_reg=curr_dist_reg,
                      contacts=tc_level.contacts, bound_dist=bound_dist,
                      res_file=res_file, ids=ordered_ids,
                      multi_b_ids=nested_boundary_ids)

        logging.info("  Finished with level %d", level)    

    # write hierarchy, positioning like bound
    tc.morphology = mor
    if do_topology:
        tc.topology = topo
    tc.distance = dist
    if write_hierarchy:
        #tc.useInset(inset=bound_full_inset, mode='abs', useFull=True, expand=True)
        #tc.saveFull()
        write_hi(tc, full_inset=bound_full_inset)

    logging.info('End')

    return tc

# run if standalone
if __name__ == '__main__':
    main()
