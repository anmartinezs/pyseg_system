#!/usr/bin/env python
"""
Depreciated. Please use connectors.py instead.

Classifies connections obtained by connections.py and analyzes them.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: classify_connections.py 918 2012-10-31 10:24:45Z vladan $
"""
__version__ = "$Revision: 918 $"

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


###########################################################
#
# Classification parameters
#

# determines which segments to keep, one of the following options has to be chosen
keep_segments = 'new'  # new segments
#keep_segments = 'new_branch_tops'  # top segment on each new branch
#keep_segments = None    # keep all segments

# restrict segments as if only one threshold was used (but keep hierarchy), used
# only for keep_segments = 'new'
one_threshold = None        # do not restrict segments
#one_threshold = -0.15       # restrict

#
# At most one of the following (volume, ids_class or number_contacted_class) can be
# used for classification. Also, class_names have to defined in exactly one of the 
# following stubs.
#

# volume: segments below the lowest and above the highest volume are removed,
# while other segments are classified based on their volume  
volumes = [10, 50, 100, 500]
class_names = ['small', 'medium', 'big']  # one element less than volumes
#volumes = [None, None]  # if no classification is needed
#class_names = ['clean']  # if no classification is needed
#volumes = None     # no volume based classification 

# segments classified by connections to the given boundaries
#ids_class = [26, 29]
#class_names = ['26', '31', 'rest']
#ids_class = [[26,27], 31]   # boundaries 26 and 27 taken together
ids_class = None

# classification based on the number of contacted boundaries
#number_contacted_class = [1, 2]
#class_names = ['1', '2']
number_contacted_class = None

# don't classify
class_names = ['all']

###########################################################
#
# Hierarchy input and output
#

# name of the file containing hierarchy object  
in_hierarchy_file_name = 'thresh_conn.pkl'

# out file is formed as: hierarchy_base + class_name + hierarchy_extension 

############################################################
#
# Image (grayscale) file. It should be in em or mrc format.
#

# name of the image file
image_file_name = "../3d/tomo.em"

###############################################################
#
# Boundary file defines a region for distance determination. If the file
# is in em or mrc format shape, data type, byte order and array order are not
# needed (should be set to None). If these variables are specified they will
# override the values specified in the headers.
#

# name of (one or more) boundary file(s)
boundary_file_name = "labels.dat"   # one boundaries file
#boundary_file_name = ("bound_1.dat", "bound_2.dat", "bound_3.dat")  # multiple files

# boundary file dimensions
boundary_shape = (512, 512, 200)

# boundary file data type (e.g. 'int8', 'int16', 'int32', 'float16', 'float64') 
boundary_data_type = 'uint8'

# boundary file byteOrder ('<' for little-endian, '>' for big-endian)
boundary_byte_order = '<'

# boundary file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis fastest)
boundary_array_order = 'FORTRAN'

# offset of boundary in respect to the data (None means 0-offset)
boundary_offset = None

###############################################################
#
# Boundaries
#
# Note: all_ids and boundary_ids have to be the same or subsets of the ids used
# in the classify script that generated the input hierarchy pickle file.
#

# ids of all segments for a single boundaries file, segments with other ids are removed 
#all_ids = [2,3,5,6]       # individual ids, single file
all_ids = range(2,64)  # range of ids, single file
#all_ids = None         # all ids are to be used, single file

# ids of all segments for multiple boundaries files can use all above forms,
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

# id shift in each subsequent boundary file (in case of multiple boundaries files) 
shift = None     # shift is determined automatically
#shift = 300

###########################################################
#
# Analysis
#

# calculate segment length (if length already exists in the input hierarchy and
# all length-related parameters are the same the old length values are kept)
do_length = False

# calculate segment length even if lengths for same parameters already exists in
# the input hierarchy (used only if do_length is True)
force_length = False

# segment length calculation mode:
#   - (one boundary) 'b-max' or 'c-max': between contacts and the most distant
#     element of a segment, where contacts belong to a boundary or a segment,
#     respectivly  
#   - (two boundaries) 'b2b' and 'c2c: minimim distance that connect a contact 
#     with one boundary, element of a segment on the central layer between the
#     boundaries, and a contact with the other boundary, where contacts belong 
#     to a boundary or a segment, respectivly
length_contact_mode = 'c2c'

# segment length line mode:
#   - 'straight': straight-line distance between contacts (two boundaries) or
#     between a contact and an end point
#   - 'mid': sum of distances between contact points and a mid point (two 
#     boundaries only)
length_line_mode = 'mid'

# id(s) of a region(s) in labels to which distances from each segment are caluculated.
# If more than one distance region is specified, for each segment the distance to
# the closest region is calculated. In case of multiple labels files this id is
# understood after the ids of the labels files are shifted.
distance_id = None  # distance is not calculated 
#distance_id = 1     # calculate distances between each segment and one region
#distance_id = [1,2,3,5]   # get distance between each segment and the closest region

# the way distance is calculated ('center', 'mean' or 'min')
distance_mode = 'mean' 

# calculate topological properties
do_topology = False

# calculate distance between connected boundaries 
do_boundary_distance = False

# cluster by connectivity
do_cluster_by_connectivity = False

########################################################
#
# hierarchical vesicle clustering
#

# cluster or not
do_hierarchical_clustering = False

# criterion for forming flat clusters from cluster hierarchy: 'distance' (maximum
# distance within each cluster), 'maxclust' (max number of clusters), 'inconsistent'
# (
hi_cluster_criter = 'distance'

# threshold for forming flat clusters with the following meaning: maximum
# distance within each cluster for 'distance' criterion, max number of clusters
# for 'maxclust', or maximum inconsistency for 'inconsistent'
hi_cluster_thresh = 10

# depth used for the flat cluster formation according to 'inconsistent' criterion
hi_cluster_depth = 2

############################################################
#
#
# Ordering of connections (only the first True ordering is applied)

# order segments by the boundaries contacted
order_by_boundaries = True

# order segments by volume
order_by_volume = False

# order by clusters
order_by_connected_clusters = False

###########################################################
#
# Connections file, formed as:
#
#   <conn_directory>/<conn_prefix> + hierarchy_root + class + <conn_suffix>
#

# if True the connections array is written to a file 
write_connections = False

# connections directory
conn_directory = ''

# connections file name prefix (no directory name)
conn_prefix = ''

# include image file root (filename without directory and extension)
insert_root_conn = True

# connections file name suffix
conn_suffix = ".em"

# connections data type: 'uint8' if max segment id is not bigger than 255, 
# 'uint16' for up to 65538, or 'int32' for up to 2147483647 segments
conn_data_type = 'uint16'         

###########################################################
#
# Clusters files, formed as:
#
#   <clust_directory>/<clust_prefix> + hierarchy_root + class + <clust_suffix>
#

# if True the clusters array is written to a file 
write_clusters = False

# clusters directory
clust_directory = ''

# clusters file name prefix (no directory name)
clust_prefix = ''

# include image file root (filename without directory and extension)
insert_root_clust = True

# connection clusters file name suffix
connect_clust_suffix = "_conn-clust.em"

# connection clusters file name suffix
bound_clust_suffix = "_bound-clust.em"

# clusters data type, 'uint8' , or
# clust_data_type = 'uint8'        # if max segment id is not bigger than 255
clust_data_type = 'uint16'         # more than 255 segments

############################################################
#
# Results file. If results_file_name is None, the results file name is formed as:
#
#   <res_directory>/<res_prefix> + hierarchy_root + class + <res_suffix>
#

# results directory
res_directory = ''

# results file name prefix (without directory)
res_prefix = ''

# include image file root (filename without directory and extension)
insert_root_res = True

# results file name suffix
res_suffix = "_con.dat"


################################################################
#
# Work
#
################################################################

###########################################
#
# Segment manipulations
#

def classify(hi):
    """
    Removes unwanted segments and separates the remaining segments in differnt
    cathegories.
    """

    # check ids
    extra_boundary_ids = numpy.setdiff1d(hi.boundaryIds, hi.contacts.boundaries)
    if len(extra_boundary_ids) > 0:
        raise ValueError("it is not allowed to introduce new boundaries. Boundaries " \
                  + numpy.array2string(extra_boundary_ids) + "are given in " \
                  + " boundary_ids variable, but do not exist in contacts.")
    bad_boundary_ids = numpy.setdiff1d(hi.contacts.boundaries, hi.boundaryIds)

    # remove boundaries that are not in hi.boundaryIds from hi.contacts
    if len(bad_boundary_ids) > 0:
        bad_ids = hi.contacts.findSegments(boundaryIds=bad_boundary_ids,  
                                           nBoundary=1, mode='at_least')
        hi.contacts.removeBoundaries(ids=bad_boundary_ids)
        hi.contacts.removeSegments(ids=bad_ids)

    # pick segments based on hierarchy
    if keep_segments is not None:

        if keep_segments == 'new':

            # keep only new segments
            if one_threshold is None:
                below = None
            else:
                below = hi.getLevelFrom(threshold=one_threshold)
            good_ids = hi.findNewIds(below=below)

        elif keep_segments == 'new_branch_tops':

            # keep only new branch tops
            good_ids = hi.findNewBranchTops()

        hi.keep(ids=good_ids)
        hi.contacts.keepSegments(ids=good_ids)

    # classify
    classes = []
    if volumes is not None:

        # warn if needed
        if len(bad_boundary_ids) > 0:
            logging.warning("Some boundary ids are removed in this run, so " \
                      + " classifications from previous runs (by number connected, " \
                      + "for example) might be affected.") 

        # remove volumes that are outside of the range
        good_ids = hi.findIdsByVolume(min=volumes[0], max=volumes[-1])
        hi.keep(ids=good_ids)

        # make classes based on volume
        for ind in range(len(class_names)):
            curr_ids = hi.findIdsByVolume(min=volumes[ind], max=volumes[ind+1])
            curr_obj = hi.keep(curr_ids, new=True)
            classes.append(curr_obj)

    elif ids_class is not None:

        # warn if needed
        if len(bad_boundary_ids) > 0:
            logging.warning("Some boundary ids are removed in this run, so " \
                      + " classifications from previous runs (by number connected, " \
                      + "for example) might be affected.") 

        # classify based on ids of contacted boundaries
        all_good_ids = []
        for ids in ids_class:
            curr_ids = hi.contacts.findSegments(boundaryIds=[ids], mode='at_least')
            all_good_ids += curr_ids.tolist() 
            curr_obj = hi.keep(curr_ids, new=True)
            classes.append(curr_obj)
        curr_obj = hi.remove(ids=all_good_ids, new=True)
        classes.append(curr_obj)
    
    elif number_contacted_class is not None:

        # classify based on number of contacted
        for num, name in zip(number_contacted_class, class_names):
            curr_ids = hi.findIdsByNBound(min=num, max=num)
            curr_obj = hi.keep(curr_ids, new=True)
            classes.append(curr_obj)

    else:

        # no classification
        classes.append(deepcopy(hi))

    return classes

def analyze(hi, image, boundary, bound_dist_db):
    """
    For each segment calculates volume, surface, density and distance to a region.
    """

    logging.info("  Starting analysis")

    # get density stats
    logging.info("    density ")
    dens = hi.getDensity(image=image.data)

    # check if morphology calculated
    try:
        mor = hi.morphology
        do_morphology = False
    except AttributeError:
        mor = pyto.segmentation.Morphology()
        do_morphology = True

    # check if to do morphology.length
    if not do_length:
        do_morphology_length = False
    elif force_length or do_morphology:
        do_morphology_length = True
    else:
        # don't calculate only if length exists and all params are the same
        try:
            if (hi.lengthContactMode == length_contact_mode) \
                    and (hi.lengthLineMode == length_line_mode) \
                    and (mor.length >= 0).any(): 
                do_morphology_length = False
            else:
                do_morphology_length = True
        except AttributeError:
            do_morphology_length = True

    # save length parameters
    if do_morphology_length:
        hi.lengthContactMode = length_contact_mode
        hi.lengthLineMode = length_line_mode

    # check if topology calculated
    try:
        topo = hi.topology
        topology_done = True
    except AttributeError:
        topo = pyto.segmentation.Topology()
        topology_done = False

    # calculate distance to a region, morprhology and topology
    # note: distances may be calculated already, but the id might be different
    dist = numpy.zeros(hi.maxId+1, dtype='float') - 1 
    dist_reg = numpy.zeros(hi.maxId+1, dtype='int') - 1
    for level in range(hi.topLevel+1):

        logging.info("  level: " + str(level))

        # extract level
        hi_level = hi.extractLevel(level=level, new=True)
        segment = hi_level.toSegment(copy=False)

        # morphology volume and surface
        if do_morphology:
            logging.info("    morphology")
            curr_mor = pyto.segmentation.Morphology(segments=segment)
            curr_mor.getVolume()
            curr_mor.getSurface()
            mor.merge(new=curr_mor)

        # morphology.length
        if do_morphology_length:
            logging.info("    morphology length ")
            curr_mor = pyto.segmentation.Morphology(segments=segment)
            curr_mor.getLength(segments=segment, boundaries=boundary,
                               contacts=hi.contacts, distance=length_contact_mode, 
                               line=length_line_mode)
            mor.merge(new=curr_mor)

        # calculate topology
        if do_topology and not topology_done:
            logging.info("    topology ")
            curr_topo = pyto.segmentation.Topology(segments=segment)
            curr_topo.calculate()
            topo.merge(new=curr_topo)

        # distances
        if distance_id is not None:
            logging.info("    distance to region ")
            curr_dist = segment.distanceToRegion(region=boundary, regionId=distance_id, 
                                                 ids=segment.ids, mode=distance_mode)

        # find distances to closest region
        if (distance_id is not None) and (curr_dist is not None):
            if curr_dist.ndim == 2:
                distance_ids = numpy.asarray(distance_id)
                dist_strip = curr_dist[distance_ids, :]
                dist_reg[segment.ids] = \
                     distance_ids[dist_strip.argmin(axis=0)][segment.ids]
                dist[segment.ids] = dist_strip.min(axis=0)[segment.ids]
            else:
                dist[segment.ids] = curr_dist[segment.ids]
                dist_reg[segment.ids] = distance_id

    # calculate distance between connected boundaries
    bound_dist = None 
    if do_boundary_distance:
        logging.info("  distance between connected boundaries")
        bound_dist = numpy.zeros(hi.maxId+1) - 1
        for seg_id in hi.ids:
            b_ids = hi.contacts.findBoundaries(segmentIds=seg_id, nSegment=1)
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

    logging.info("  Analysis done")

    # return
    return dens, mor, topo, dist, dist_reg, bound_dist, bound_dist_db

def find_boundaries(hi):
    """
    Finds boundaries that each segment contacts.
    """
    boundaries = numpy.array([None] * (hi.maxId + 1))
    for sid in hi.ids:
        boundaries[sid] = hi.contacts.findBoundaries(segmentIds=sid, nSegment=1)
    return boundaries

def cluster_by_connectivity(contacts):
    """
    Cluster by connectivity
    """

    if do_cluster_by_connectivity:

        # find clusters
        logging.info("  connectivity clustering ")
        b_clusters = \
            pyto.segmentation.Cluster.clusterBoundaries(contacts=contacts)
        c_clusters = pyto.segmentation.Cluster.clusterConnections(\
                boundClusters=b_clusters, contacts=contacts)
        
    else:
        b_clusters = None
        c_clusters = None

    return b_clusters, c_clusters
    
def hierarchical_clustering(boundary, ids, contacts, threshold, 
                            criterion='maxclust', depth=2):
    """
    Hierarchical segment clustering
    """

    if do_hierarchical_clustering:

        # make boundary clusters
        logging.info("  hierarchical clustering ")
        b_clusters = pyto.segmentation.Cluster.hierarchical(segments=boundary, 
                                                             ids=ids)
        b_clusters.extractFlat(threshold, criterion, depth)

        # convert to connection clusters
        c_clusters = pyto.segmentation.Cluster.clusterConnections(\
            boundClusters=b_clusters, contacts=contacts)

    else:
        b_clusters = None
        c_clusters = None

    return b_clusters, c_clusters


################################################################
#
# Input/output
#

def read_hi(inset=None):
    """
    Reads (unpickles) a hierarchy object and expands it to inset.

    Argument:
      - inset: absolute inset
    """

    # figure out hierarchy file name

    # read
    in_file = open(in_hierarchy_file_name, 'rb')
    hi = pickle.load(in_file)
    #in_file.close()

    # use given inset
    if inset is not None:
        hi.useInset(inset=inset, mode='absolute', expand=True)
    hi.clearFull()

    # expand contacts
    hi.contacts.expand()

    return hi

def get_out_hierarchy_name(class_name):
    """
    Returns output hierarchy name
    """
    
    # extract root from the hierarchy file name
    (dir, base) = os.path.split(in_hierarchy_file_name)
    (root, ext) = os.path.splitext(base)

    # figure out hierarchy file name
    out_hi_base = root + '_' + class_name + ext 
    out_hi_file_name = os.path.join(dir, out_hi_base)

    return out_hi_file_name

def write_hi(hi, class_name, full_inset=None):
    """
    Writes (pickles) a hierarchy object
    """
    # get name
    out_hi_file_name = get_out_hierarchy_name(class_name)

    # make inset
    inset = hi.inset
    hi.makeInset()
    if full_inset is None:
        hi.fullInset = inset
    else:
        hi.fullInset = full_inset

    # compactify contacts
    hi.contacts.compactify()

    # write 
    out_file = open(out_hi_file_name, 'wb')
    pickle.dump(hi, out_file, -1)

    # recover inset and contacts
    hi.useInset(inset=inset, mode='absolute', expand=True)
    hi.contacts.expand()

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
    Reads boundaries file(s) and makes (Segment) boundaries.

    Keeps all ids given in all_ids, discards other.

    Also adjusts boundary ids (from variable boundary_ids) so that in the case of 
    multiple boundary files the ids are properly shifted.

    Returns:
      - bound: (segmentation.Segment) boundaries
      - multi_boundary_ids: list of adjusted (shifted) boundary ids
    """

    # read
    if is_multi_boundaries():
        bound, multi_boundary_ids = read_multi_boundaries()
    else:
        bound = read_single_boundaries()
        multi_boundary_ids = [boundary_ids]

    # offset
    bound.offset = boundary_offset

    return bound, multi_boundary_ids

def is_multi_boundaries():
    """
    Returns True if maultiple boundaries files are given.
    """
    if isinstance(boundary_file_name, str):
        return False
    elif isinstance(boundary_file_name, tuple) or isinstance(boundary_file_name, list):
        return True
    else:
        raise ValueError, "boundary_file_name has to be aither a string (one " \
              + "boundary file) or a tuple (multiple boundary files)."    

def read_single_boundaries():
    """
    Reads and initializes boundaries form a sigle file.
    """

    # read boundaries file and make a Segment object
    bound = pyto.segmentation.Segment.read(file=boundary_file_name, ids=all_ids,
               clean=True, byteOrder=boundary_byte_order, dataType=boundary_data_type,
               arrayOrder=boundary_array_order, shape=boundary_shape)

    return bound

def read_multi_boundaries():
    """
    Reads and initializes boundaries form a sigle file.
    """

    # read all boundaries files and combine them in a single Segment object
    bound = pyto.segmentation.Segment()
    curr_shift = 0
    shifted_boundary_ids = []
    for (l_name, a_ids, b_ids) in zip(boundary_file_name, all_ids, boundary_ids):
        curr_bound = pyto.segmentation.Segment.read(file=l_name, ids=a_ids,
               clean=True, byteOrder=boundary_byte_order, dataType=boundary_data_type,
               arrayOrder=boundary_array_order, shape=boundary_shape)
        bound.add(new=curr_bound, shift=curr_shift, dtype='int16')
        shifted_boundary_ids.append(numpy.array(b_ids) + curr_shift)
        if shift is None:
            curr_shift = None
        else:
            curr_shift += shift

    return bound, shifted_boundary_ids
    
def get_base():
    """
    Returns base and root of the hierarchy input file name
    """
    (dir, base) = os.path.split(in_hierarchy_file_name)
    (root, ext) = os.path.splitext(base)
    return base, root

def get_connections_name(class_name):
    """
    Returns the connections file name
    """

    # extract root from the hierarchy file name
    base, root = get_base()

    # determine connections file name 
    conn_base = conn_prefix + root + '_' + class_name + conn_suffix
    conn_file_name = os.path.join(conn_directory, conn_base)

    return conn_file_name
    
def get_clusters_name(class_name, cluster_suffix=None):
    """
    Returns the clusters file name
    """

    # extract root from the hierarchy file name
    base, root = get_base()

    # determine cluster suffix
    if cluster_suffix is None:
        cluster_suffix = clust_suffix

    # determine connections file name 
    clust_base = clust_prefix + root + '_' + class_name + cluster_suffix
    clust_file_name = os.path.join(clust_directory, clust_base)

    return clust_file_name

def machine_info():
    """
    Returns machine name and machine architecture strings
    """
    mach = platform.uname() 
    mach_name = mach[1]
    mach_arch = str([mach[0], mach[4], mach[5]])

    return mach_name, mach_arch

def write_clus(labels, class_name, clusters, cluster_suffix=None):
    """
    Writes data that was relabeled according to clusters.
    """

    # get clusters file name
    clust_file_name = get_clusters_name(class_name, cluster_suffix)

    # relabel segment ids according to clusters
    cluster_order = {}
    for clust_id in range(1, clusters.getNClusters()+1):
        for data_id in clusters.getCluster(clusterId=clust_id):
            cluster_order[data_id] = clust_id
    clust_data = labels.reorder(order=cluster_order, 
                                  data=labels.data, clean=True)

    # write
    clust_seg = pyto.segmentation.Segment(data=clust_data)
    clust_seg.write(file=clust_file_name, dataType=clust_data_type)
    
def open_results(class_name):
    """
    Opens a results file name and returns it.
    """
    
    # extract root from the input hierarchy file name
    base, root = get_base()

    # figure out results file name
    res_base = res_prefix + root + '_' + class_name + res_suffix
    res_file_name = os.path.join(res_directory, res_base)

    # open file
    res_file = open(res_file_name, 'w')
    return res_file

def write_res(hi, dens, mor, topo, dist, dist_reg, contacts, bound_dist, con_clusters, 
              res_file, class_name, ids=None, multi_b_ids=None):
    """
    Writes results for one level (threshold) in the results file.
    
    Arguments:
    """
    
    # check ids
    if ids is None:
        ids = hi.ids

    # machine info
    mach_name, mach_arch = machine_info()

    # get file names
    conn_file_name = get_connections_name(class_name)    
    out_hi_file_name = get_out_hierarchy_name(class_name)

    # file times
    image_time = \
        '(' + time.asctime(time.localtime(os.path.getmtime(image_file_name))) + ')'
    in_hi_time = time.asctime(time.localtime(os.path.getmtime(in_hierarchy_file_name)))
    try:
        out_hi_time = time.asctime(time.localtime(os.path.getmtime(out_hi_file_name)))
    except OSError:
        out_hi_time = 'not written'
    try:
        conn_time = time.asctime(time.localtime(os.path.getmtime(conn_file_name)))
    except OSError:
        conn_time = "not written"
    in_file_name = sys.modules[__name__].__file__
    in_time = time.asctime(time.localtime(os.path.getmtime(in_file_name)))

    # boundary file(s) name(s), time(s) and boundary ids
    if is_multi_boundaries():
        boundary_lines = ["#     " + b_file + " (" + \
                       time.asctime(time.localtime(os.path.getmtime(b_file))) + ")"\
                   for b_file in boundary_file_name]
        boundary_lines.insert(0, "# Boundaries: ")
        boundary_ids_lines = ["#     " + str(b_ids) for b_ids in multi_b_ids]
        boundary_ids_lines.insert(0, "# Boundary ids (shift = " + str(shift) + "): ")
    else:
        boundary_time = \
                  time.asctime(time.localtime(os.path.getmtime(boundary_file_name)))
        boundary_lines = ["# Boundaries: ",
                   "#     " + boundary_file_name + " (" + boundary_time + ")"]
        boundary_ids_lines = ["# Boundary ids: ",
                       "#     " + str(boundary_ids)]

    # structuring element
    if hi.structEl is None:
        se_conn = 'none'
        se_size = 'none'
    else:
        se_conn = str(hi.structEl.connectivity)
        se_size = str(hi.structEl.size)

    # length parameters
    try:
        len_cont_mode = hi.lengthContactMode
    except AttributeError:
        len_cont_mode = "Unknown"
    try:
        len_line_mode = hi.lengthLineMode
    except AttributeError:
        len_line_mode = "Unknown"

    # results file header
    header = ["#",
        "# Machine: " + mach_name + " " + mach_arch,
        "# Date: " + time.asctime(time.localtime()),
        "#",
        "# Image: " + image_file_name + " " + image_time,
        "# Input hierarchy: " + in_hierarchy_file_name + " (" + in_hi_time + ")"]
    header.extend(boundary_lines)
    header.extend([\
        "# Output hierarchy: " + out_hi_file_name + " (" + out_hi_time + ")",
        "# Connections (output): " + conn_file_name + " (" + conn_time + ")",
        "# Input script: " + in_file_name + " (" + in_time + ") " + __version__,
        "# Working directory: " + os.getcwd(),
        "#"])
    header.extend(boundary_ids_lines)
    header.extend([\
        "#",
        "# Structuring element:",
        "#   - connectivity: " + se_conn,
        "#   - size: " + se_size,
        "#",
        "# Connectivities of other structuring elements:",
        "#   - detect contacts: " + str(hi.contactStructElConn),
        "#   - topology: 1",
        "#",
        "# Keep segments:",
        "#   - mode: " + str(keep_segments),
        "#   - one threshold: " + str(one_threshold),
        "#",
        "# Segment length: ",
        "#   - contact mode: " + str(len_cont_mode),
        "#   - line mode: " + str(len_line_mode),
        "#",
        "# Distance to region: ", 
        "#   - region id: " + str(distance_id),
        "#   - mode: " + distance_mode,
        "#"])
    if do_hierarchical_clustering:
        header.extend([\
                "# Hierarchical clustering:",
                "#   - clustering method: minimal euclidean distance",
                "#   - flat clusters criterion: " + hi_cluster_criter,
                "#   - flat clusters threshold: " + str(hi_cluster_thresh),
                "#   - flat clusters depth: " + str(hi_cluster_depth),
                "#"])
    header.extend([\
        "# Number of segments: " + str(len(ids)),
        "#"])
    if con_clusters is not None:
        header.extend([\
                "# Number of connectivity clusters (some clusters may contain no "\
                    + " connections): " + str(len(con_clusters.clusters)),
                "#"])
    for line in header: 
        res_file.write(line + os.linesep)

    # write results table head
    tab_head = ["# Id   Threshold       Density             Volume Surface ",
                "#                 mean   std    min    max                " ]
    if mor.length is not None:
        tab_head[0] = tab_head[0] + 'Length '
        tab_head[1] = tab_head[1] + '       '
    if do_topology:
        tab_head[0] = tab_head[0] + 'Euler Loops Holes '
        tab_head[1] = tab_head[1] + '                  '
    if do_boundary_distance:
        tab_head[0] = tab_head[0] + 'Boundary '
        tab_head[1] = tab_head[1] + 'distance '
    if dist is not None:
        tab_head[0] = tab_head[0] + 'Distance    '
        tab_head[1] = tab_head[1] + '        to  '  
    if con_clusters is not None:
        tab_head[0] = tab_head[0] + 'Clusters'
        tab_head[1] = tab_head[1] + 'connect '
    if do_hierarchical_clustering:
        tab_head[0] = tab_head[0] + 'Clusters '
        tab_head[1] = tab_head[1] + 'hierarchy'
    tab_head[0] = tab_head[0] + " Boundaries "
    for line in tab_head: res_file.write(line + os.linesep)

    # make threshold array
    thresh = numpy.zeros(hi.maxId+1)
    for id_ in hi.ids:
        thresh[id_] = hi.threshold[hi.getIdLevels(id_)]

    # prepare the results table
    out_vars = [ thresh, dens.mean, dens.std, dens.min, dens.max, mor.volume, \
                mor.surface]
    out_format = ' %6u %7.3f %6.2f %5.2f %6.2f %6.2f %5u  %5u '
    if mor.length is not None:
        out_vars.append(mor.length)
        out_format = out_format + '  %5.1f'
    if do_topology:
        out_vars.extend([topo.euler, topo.nLoops, topo.nHoles])
        out_format = out_format + ' %4i  %4i  %4i '
    if do_boundary_distance:
        out_vars.append(bound_dist)
        out_format = out_format + ' %6.1f  '        
    if dist is not None:
        out_vars.extend([dist, dist_reg])
        out_format = out_format + ' %6.1f %4i '
    if con_clusters is not None:
        con_cluster_array = con_clusters.clustersData
        out_vars.append(con_cluster_array)
        out_format = out_format + '   %4i  ' 
    if do_hierarchical_clustering:
        hi_cluster_array = hi.hierarchicalConnectionClusters.clustersData
        out_vars.append(hi_cluster_array)
        out_format = out_format + '   %4i   '
    out_vars.append(hi.boundaries)
    out_format = out_format + '%s'

    # write the results table
    res_table = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                        indices=ids, prependIndex=True)
    for line in res_table:
        res_file.write(line + os.linesep)

    res_file.flush()

def order(ids, contacts=None, volume=None, connect_clusters=None):
    """
    Returns ordered ids.
    """
    if order_by_boundaries:
        sort_list = contacts.orderSegmentsByContactedBoundaries(argsort=True)
    elif order_by_volume:
        sort_list = volume[ids].argsort()
    elif order_by_connected_clusters:
        clusters_array = connect_clusters.clustersData
        sort_list = clusters_array[ids].argsort()
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

    # read boundary (needed for distances and if the current boundary ids are
    # a subset of those in connections script used to make tc file)
    bound, nested_boundary_ids = read_boundaries()
    flat_boundary_ids = pyto.util.nested.flatten(nested_boundary_ids)
    full_inset = bound.inset
    bound.saveFull()

    # read hierarchy (already segmented) and clean it (just in case)
    tc = read_hi(inset=full_inset)
    #tc.clearFull()
    tc.data = pyto.segmentation.Segment().keep(ids=tc.ids, data=tc.data)

    # make insets
    tc.makeInset(extend=1)
    bound.useInset(inset=tc.inset, mode='abs')

    # remove unwanted segments and classify the remaining segments
    logging.info('Starting classification')
    tc.boundaryIds = numpy.asarray(flat_boundary_ids)
    tc_classes = classify(tc)

    # read image
    image = read_image()
    image.useInset(inset=tc.inset, mode='absolute')

    # initialize variables used for all classes
    bound_dist_db = {}

    #
    for tc_class, tc_name in zip(tc_classes, class_names):

        logging.info('Class: ' + tc_name)

        # restrict contacts to those connecting current segments and boundaries 
        contacts = deepcopy(tc_class.contacts)
        contacts.keepSegments(ids=tc_class.ids)
        #contacts.keepBoundaries(ids=flat_boundary_ids)
        tc_class.contacts = contacts

        # analyze segments and save results
        dens, mor, topo, dist, dist_reg, bound_dist, bound_dist_db = \
            analyze(hi=tc_class, image=image, boundary=bound,
                    bound_dist_db=bound_dist_db)
        tc_class.distanceToRegion = dist
        if bound_dist is not None:
            tc_class.boundaryDistance = bound_dist        

        # find boundaries
        logging.info('  Boundaries for each segment')
        tc_class.boundaries = find_boundaries(hi=tc_class)

        # cluster and save results
        bound_clusters, connect_clusters = \
            cluster_by_connectivity(contacts=tc_class.contacts)
        bound.recoverFull()
        hi_bound_clusters, hi_connect_clusters = \
                hierarchical_clustering(boundary=bound, contacts=tc_class.contacts,
                     ids=flat_boundary_ids, threshold=hi_cluster_thresh, 
                     criterion=hi_cluster_criter, depth=hi_cluster_depth)
        tc_class.connectivityBoundaryClusters = bound_clusters
        tc_class.connectivityConnectionClusters = connect_clusters
        tc_class.hierarchicalBoundaryClusters = hi_bound_clusters
        tc_class.hierarchicalConnectionClusters = hi_connect_clusters

        # write hierarchy
        logging.info('  Writing ')
        if len(tc_class.ids) > 0:
            write_hi(tc_class, tc_name, full_inset)

        # set to full size (just fow writing data)
        tc_inset = tc_class.inset
        tc_class.useInset(inset=full_inset, mode='abs', useFull=False, expand=True)
        #tc_class.saveFull()

        # write hierarchy, connections, clusters
        if write_connections and (len(tc_class.ids) > 0):
            conn_file_name = get_connections_name(tc_name) 
            tc_class.write(file=conn_file_name, dataType=conn_data_type)
        if write_clusters and (len(tc_class.ids) > 0):
            if connect_clusters is not None:
                write_clus(labels=tc_class, class_name=tc_name, 
                           clusters=connect_clusters, 
                           cluster_suffix=connect_clust_suffix)
            if bound_clusters is not None:
                bound.recoverFull()
                write_clus(labels=bound, class_name=tc_name, 
                           clusters=bound_clusters, 
                           cluster_suffix=bound_clust_suffix)

        # use inset again
        tc_class.useInset(inset=tc_inset, mode='abs')
        bound.useInset(inset=tc_inset, mode='abs')

        # write results
        logging.info('  Ordering ')
        ordered_ids = order(ids=tc_class.ids, contacts=tc_class.contacts,
                            volume=mor.volume, connect_clusters=connect_clusters)
        res_file = open_results(tc_name)
        write_res(hi=tc_class, dens=dens, mor=mor, topo=topo, dist=dist,
                  dist_reg=dist_reg, contacts=tc_class.contacts, 
                  bound_dist=bound_dist, res_file=res_file, class_name=tc_name, 
                  ids=ordered_ids, multi_b_ids=nested_boundary_ids,
                  con_clusters=connect_clusters)

        # reduce memory usage
        tc_class.data = None
        tc_class.fullData = None

        logging.info('  Finished with class: ' + tc_name)

    return tc

# run if standalone
if __name__ == '__main__':
    main()
