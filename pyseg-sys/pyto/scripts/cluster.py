#!/usr/bin/env python
"""
Makes and analyzes clusters of (already existing) boundaries and segments.

For example, in case of presynaptic terminal, vesicles are boundaries and
connectors are segments.

Clustering can be done based on connectivity (see "Clustering by connectivity"
parameter section) or on distance (see "Hierarchical clustering of connections"
and "Hierarchical clustering of boundaries" parameter sections).

In short, clustering based on connectivity makes:
  - clusters of boundaries that are connected (linked) by segments 
  - clusters of segments connected by boundaries. 
Clustering based on distance (hierarchical clustering) makes:
  - clusters of boundaries
  - clusters of segments
In addition, clusters dual to distance based clusters are also made 
(experimental feature):
  - clusters of segments that contact distance boundary clusters (dual to
  distance boundary clusters)
  - clusters of boundariess that contact distance segment clusters (dual to
  distance segment clusters)

For more info how these clusters are made see class pyto.scene.MultiCluster.

Important notes:

  - In order to make setting up multiple scripts easier, parameters common 
to these scripts are expected to be read from tomo_info.py file. The location 
of this file is given as argument path of common.__import__(). These parameters
are set in this script in lines having the following form:

  if tomo_info is not None: labels_file_name = tomo_info.labels_file_name

Specifiying another value for the same parameter (here labels_file_name) 
overrides the value from tomo_info.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: cluster.py 1430 2017-03-24 13:18:43Z vladan $
"""
__version__ = "$Revision: 1430 $"

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
import scipy.ndimage as ndimage

# to debug replace INFO by DEBUG
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')

import pyto
import pyto.scripts.common as common

# import ../common/tomo_info.py
tomo_info = common.__import__(name='tomo_info', path='../common')


##############################################################
#
# Parameters
#
##############################################################

###########################################################
#
# Clustering by connectivity
#

# cluster by connectivity 
cluster_by_connectivity = True

###########################################################
#
# Hierarchical clustering of boundaries (vesicles)
#

# hierarchically cluster boundaries
hi_cluster_boundaries = True

# linkage method: 'single', 'complete', 'average', 'weighted', or whatever else 
# is accepted by scipy.cluster.hierarchy.linkage()
hi_bound_linkage = 'single'

# criterion for forming flat clusters from cluster hierarchy: 'distance', 
# 'maxclust', or 'inconsistent'.
hi_bound_criter = 'distance'

# depth used for the flat cluster formation when criterion is 'inconsistent'
hi_bound_depth = 2

# Threshold for forming flat clusters. Its meaning depends on the criterion 
# for forming flat clusters: 
#   - 'distance': maximum distance within each cluster 
#   - 'maxclust': max number of clusters
#   - 'inconsistent': maximum inconsistency
#hi_bound_thresh = 10              # single threshold
# Alternatively, if multiple values are given, a flat cluster is calculated for
# each value and the one that is the most similar to the connectivity cluster
# (according to the similarity calculation method) is selected.
hi_bound_thresh = range(6,15)    # find best threshold

# Similarity calculation method used to select the best threshold value. The
# possible values are: 'vi', 'b-flat', 'rand' or 'rand_same_cluster'
hi_bound_similarity = 'b-flat'

# use one-item clusters for similarity calculations
hi_bound_single = True

###########################################################
#
# Hierarchical clustering of connections
#

# hierarchically cluster segments
hi_cluster_connections = True

# linkage method: 'single', 'complete', 'average', 'weighted', or whatever 
# else is accepted by scipy.cluster.hierarchy.linkage()
hi_conn_linkage = 'single'

# criterion for forming flat clusters from cluster hierarchy: 'distance', 
# 'maxclust', or 'inconsistent'.
hi_conn_criter = 'distance'

# depth used for the flat cluster formation when criterion is 'inconsistent'
hi_conn_depth = 2

# Threshold for forming flat clusters. Its meaning depends on the criterion 
# for forming flat clusters: 
#   - 'distance': maximum distance within each cluster 
#   - 'maxclust': max number of clusters
#   - 'inconsistent': maximum inconsistency
#hi_conn_thresh = 20              # single threshold
# Alternatively, if multiple values are given, a flat cluster is calculated for
# each value and the one that is the most similar to the connectivity cluster
# (according to the similarity calculation method) is selected.
hi_conn_thresh = range(10,30,2)    # find best threshold

# similarity calculation method: 'b-flat', 'rand' or 'rand_same_cluster'
hi_conn_similarity = 'rand'

# use one-item clusters for similarity calculations
hi_conn_single = True

###########################################################
#
# Threshold and connectivity pickle file input 
#

# name of the pickle file containing hierarchy object  
in_seg_file_name = 'thresh_conn.pkl'

###############################################################
#
# Boundary (labels) file 
#
# Boundary file defines a region for distance determination. If the file
# is in em or mrc format shape, data type, byte order and array order are not
# needed (should be set to None). If these variables are specified they will
# override the values specified in the headers.
#

# name of (one or more) boundary file(s)
if tomo_info is not None: boundary_file_name = tomo_info.labels_file_name
#boundary_file_name = "labels.dat"   # one boundaries file
#boundary_file_name = ("bound_1.dat", "bound_2.dat", "bound_3.dat")  # multiple

# boundary file dimensions
boundary_shape = (512, 512, 200)

# boundary file data type (e.g. 'int8', 'int16', 'int32', 'float16', 'float64') 
if tomo_info is not None: boundary_data_type = tomo_info.labels_data_type
#boundary_data_type = 'uint8'

# boundary file byteOrder ('<' for little-endian, '>' for big-endian)
boundary_byte_order = '<'

# boundary file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis)
boundary_array_order = 'FORTRAN'

# offset of boundary in respect to the data (None means 0-offset) (experimental)
boundary_offset = None

# ids of all boundaries. In addition to the single and multiple boundary files
# format, nested list can be used where ids in a sublist are understood in the
# "or" sense, that is all boundaries listed in a sublist form effectively a 
# single boundary. 
# Note: These ids can be all or a subset of boundary ids in the hierarchy 
# pickle, but there shouldn't be any id that's not in the hierarchy pickle
#in_boundary_ids = [2,3,5]       # individual ids, single file
in_boundary_ids = range(2,64)  # range of ids, single file
#in_boundary_ids = None         # all segments are to be used, single file
#in_boundary_ids = [[2,3], 4, 5, 6]  #  2 and 3 taken together, single file

# boundary ids that should not be used, same formats as obove apply
# Note: doesn't work if multiple boundary files and some boundaries taken
# together
#ex_boundary_ids = []         # do not exclude any
ex_boundary_ids = [20, 21]

# id shift in each subsequent boundary file (in case of multiple boundaries 
# files) (experimental)
shift = None     # shift is determined automatically
#shift = 254

###########################################################
#
# Output files
#
# Result files are formed as:
#
#   <clust_directory>/<clust_prefix> + tc_root + <result_suffix>
#
# Image (array) clusters files are formed as:
#
#   <clust_directory>/<clust_prefix> + tc_root + <image_suffix>
#
# Clusters pickle file name is formed as:
#
#   <clust_directory>/<clust_prefix> + tc_root + <pickle_suffix>
#

# clusters directory
clust_directory = ''

# clusters file name prefix (no directory name)
clust_prefix = ''

# clustering connections results file suffix
result_conn_suffix = '_cluster-conn.dat'

# clustering boundaries results file suffix
result_bound_suffix = '_cluster-bound.dat'

# connectivity clustering results file suffix
result_conn_clust_suffix = '_conn-cluster.dat'

# pickle all clusters
pickle_clusters = True

# add contacts to the pickle 
pickle_contacts = False

# cluster pickle file suffix
pickle_suffix = '_cluster.pkl'

# if True the clusters array is written to a file 
write_images = False

# clustering connections by connectivity image file suffix
conn_conn_suffix = "_cluster-conn-conn.em"

# clustering boundaries by connectivity image file suffix
conn_bound_suffix = "_cluster-conn-bound.em"

# herarchical clustering of connections image file suffix
hi_conn_suffix = "_cluster-hi-conn.em"

# hierarchical clustering of boundaries image file suffix
hi_bound_suffix = "_cluster-hi-bound.em"

# dual clustering of connections from hierarchical clustering of boundaries
dual_hi_conn_suffix = "_cluster-dual-hi-conn.em"

# dual clustering of boundaries from hierarchical clustering of connections
dual_hi_bound_suffix = "_cluster-dual-hi-conn.em"

# clusters data type, 'uint8' , or
clust_data_type = 'uint8'        # if max segment id is not bigger than 255
#clust_data_type = 'uint16'         # more than 255 segments

###########################################################
#
# Distances file (both for input and output)
#

# distance file prefix
distance_prefix = ''

# name of the pickle containing boundary distances (both for input and output)
bound_distance_suffix = '_bound_distances.pkl'

# name of the pickle containing connection distances (both for input and output)
conn_distance_suffix = '_conn_distances.pkl'

# if True distances are read from a pickle if possible, otherwise they're 
# calculated
# Warning: boundary_ids used here and those used to generate the distances have 
# to be the same and to be in the same order.
read_distance = False


################################################################
#
# Work
#
################################################################

################################################################
#
# Input/output
#

def clean_ids(include, exclude):
    """
    Removes exclude ids from include ids
    """

    # deal with nothing to exclude
    if (exclude is None) or (len(exclude) < 1):
        return include

    # check if nested
    if pyto.util.nested.is_nested(exclude):
        nested = True
    else:
        nested = False
        include = [include]
        exclude = [exclude]

    # work
    ids = [list(set(inc).difference(set(ex))) \
               for inc, ex in zip(include, exclude)]
    ids = [x for x in ids if len(x) > 0]
    
    # return in the form arguments were given 
    if nested:
        return ids
    else:
        return ids[0]

def read_segments(name, bound_ids, inset=None):
    """
    Reads segments from Hierarchy pickle.

    Arguments:
    
    """

    # read threshold and connectivity
    pickled_obj = common.read_pickle(file_name=name)
    if isinstance(pickled_obj, pyto.segmentation.Labels):
        segments = pickled_obj
    elif isinstance(pickled_obj, pyto.scene.SegmentationAnalysis):
        segments = pickled_obj.labels
    else:
        raise ValueError(
            "Pickle file " + name + " has to be instance of"
            + "pyto.segmentation.Labels or pyto.scene.SegmentationAnalysis")

    # expand contacts
    segments.contacts.expand()

    # convert to segments
    if isinstance(segments, pyto.segmentation.Hierarchy):
        contacts = segments.contacts
        segments = segments.toSegment()  
        segments.contacts = contacts

    # clean segments (just in case)
    segment_ids = segments.contacts.findSegments(boundaryIds=bound_ids, 
                                                 nBoundary=2)
    segments.keep(ids=segment_ids)

    # clean contacts (necessary)
    segments.contacts.keepBoundaries(ids=bound_ids)
    segments.contacts.keepSegments(ids=segment_ids)

    # use inset
    segments.clearFull()
    if inset is not None:
        segments.useInset(inset=inset, mode='absolute', expand=True)

    return segments

def read_boundaries(boundary_ids):
    """
    Reads boundaries file(s) and makes (Segment) boundaries.
    """

    # read
    if is_multi_boundaries():
        bound, multi_boundary_ids = read_multi_boundaries(
            boundary_ids=boundary_ids)
    else:
        bound = read_single_boundaries(boundary_ids=boundary_ids)
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
    elif isinstance(boundary_file_name, tuple) \
            or isinstance(boundary_file_name, list):
        return True
    else:
        raise ValueError, "boundary_file_name has to be aither a string (one " \
              + "boundary file) or a tuple (multiple boundary files)."    

def read_single_boundaries(boundary_ids):
    """
    Reads and initializes boundaries form a sigle file.
    """

    # read boundaries file and make a Segment object
    bound = pyto.segmentation.Segment.read(
        file=boundary_file_name, ids=boundary_ids,
        clean=True, byteOrder=boundary_byte_order, dataType=boundary_data_type,
        arrayOrder=boundary_array_order, shape=boundary_shape)

    return bound

def read_multi_boundaries(boundary_ids):
    """
    Reads and initializes boundaries form a sigle file.
    """

    # read all boundaries files and combine them in a single Segment object
    bound = pyto.segmentation.Segment()
    curr_shift = 0
    shifted_boundary_ids = []
    for (l_name, b_ids) in zip(boundary_file_name, boundary_ids):
        curr_bound = pyto.segmentation.Segment.read(
            file=l_name, ids=b_ids, clean=True, byteOrder=boundary_byte_order, 
            dataType=boundary_data_type, arrayOrder=boundary_array_order, 
            shape=boundary_shape)
        bound.add(new=curr_bound, shift=curr_shift, dtype='int16')
        shifted_boundary_ids.append(numpy.array(b_ids) + curr_shift)
        if shift is None:
            curr_shift = None
        else:
            curr_shift += shift

    return bound, shifted_boundary_ids
    
def get_base(file_name):
    """
    Returns base and root of the given file name
    """
    (dir, base) = os.path.split(file_name)
    (root, ext) = os.path.splitext(base)
    return base, root

def get_clusters_file_name(base_file, cluster_directory, cluster_suffix):
    """
    Returns the clusters file name
    """

    # get base
    base, root = get_base(file_name=base_file)

    # cluster file name
    clust_base = clust_prefix + root + cluster_suffix
    clust_file_name = os.path.join(cluster_directory, clust_base)

    return clust_file_name

def get_file_name(base_file, directory, prefix, suffix):
    """
    Returns file name in the form:

      directory/prefix + root_of_base_file + suffix
    """

    # get base
    foo, root = get_base(file_name=base_file)

    # cluster file name
    base = prefix + root + suffix
    file_name = os.path.join(directory, base)

    return file_name

def find_distances(file_, read=True, segments=None, ids=None):
    """
    Read distances from a pickle file, or calculate them if the file does not
    exist or if arg read is False. If distances are calculated they're saved
    to a pickle file.

    Returns distances
    """

    try:

        # read from pickle
        if not read:
            raise IOError 
        in_file = open(file_)
        logging.info('Reading distance file')
        distances = pickle.load(in_file)

    except IOError:

        # calculate
        logging.info('Calculating distances')
        distances = segments.pairwiseDistance(ids=ids, mode='min')

        # save 
        out_file = open(file_, 'wb')
        pickle.dump(distances, out_file, -1)

    return distances

def write_cluster_image(clusters, labels, base_file, cluster_directory, 
                        cluster_suffix, clusters_data_type):
    """
    Writes clusters image
    """

    # don't do anything if no clustering
    if clusters is None:
        return

    # get clusters image name
    file_name = get_clusters_file_name(base_file, cluster_directory, cluster_suffix)

    # relabel segment ids according to clusters
    cluster_order = {}
    for clust_id in range(1, clusters.getNClusters()+1):
        for data_id in clusters.getCluster(clusterId=clust_id):
            cluster_order[data_id] = clust_id
    clust_data = labels.reorder(order=cluster_order, 
                                  data=labels.data, clean=True)

    # write
    clust_image = pyto.segmentation.Segment(data=clust_data)
    file_ = clust_image.write(file=file_name, dataType=clusters_data_type)
    
    return file_

def pickle_all_clusters(multi_clust, base_file, directory, suffix, contacts=None):
    """
    Pickles multi cluster.
    """

    if contacts is not None:
        contacts.compactify()
        multi_clust.contacts = contacts

    # write pickle
    file_name = get_clusters_file_name(base_file, directory, suffix)
    pickle.dump(multi_clust, open(file_name, 'wb'), -1)

    if contacts is not None:
        contacts.expand()

    return file_name

def write_cluster_results(multi_cluster, multi_cluster_name, segments, bound, 
           multi_bound_ids, contacts, base_file, cluster_directory, 
           result_bound_suffix, result_conn_suffix, cluster_files,
           hi_bound_thr=None, hi_conn_thr=None, distance_files={}):
    """
    Writes cluster results file
    """

    # open results file
    bound_res_file_name = get_clusters_file_name(base_file, cluster_directory, 
                                           result_bound_suffix)
    bound_res_file = open(bound_res_file_name, 'w')
    conn_res_file_name = get_clusters_file_name(base_file, cluster_directory, 
                                           result_conn_suffix)
    conn_res_file = open(conn_res_file_name, 'w')
    conn_clust_res_file_name = get_clusters_file_name(
        base_file, cluster_directory, result_conn_clust_suffix)
    conn_clust_res_file = open(conn_clust_res_file_name, 'w')

    # machine info
    mach_name, mach_arch = common.machine_info()
    header = ["#",
        "# Machine: " + mach_name + " " + mach_arch,
        "# Date: " + time.asctime(time.localtime())]

    # script and working directory
    in_file_name = sys.modules[__name__].__file__
    in_time = time.asctime(time.localtime(os.path.getmtime(in_file_name)))
    header.extend([
            "#",
            "# Input script: " + in_file_name + " (" + in_time + ") " 
            + __version__,
            "# Working directory: " + os.getcwd()])

    # file names and times
    in_seg_time = time.asctime(time.localtime(os.path.getmtime(base_file)))
    header.extend([
            "#",
            "# Connections: " + base_file + " (" + in_seg_time + ")"])

    # boundary file(s)
    if is_multi_boundaries():
        boundary_lines = [
            "#     " + b_file + " (" 
            + time.asctime(time.localtime(os.path.getmtime(b_file))) + ")"
            for b_file in boundary_file_name]
        boundary_lines.insert(0, "# Boundaries: ")
        boundary_ids_lines = [
            "#     " + str(b_ids) for b_ids in multi_bound_ids]
        boundary_ids_lines.insert(
            0, "# Boundary ids (shift = " + str(shift) + "): ")
    else:
        boundary_time = time.asctime(
            time.localtime(os.path.getmtime(boundary_file_name)))
        boundary_lines = [
            "# Boundaries: ",
            "#     " + boundary_file_name + " (" + boundary_time + ")"]
        boundary_ids_lines = [
            "# Boundary ids: ",
            "#     " + str(bound.ids)]
    header.extend(boundary_lines)

    # multi cluster file name and time
    try:
       clus_time = time.asctime(
           time.localtime(os.path.getmtime(multi_cluster_name))) 
    except OSError:
        clus_time = 'not written' 
    cluster_lines = [
        "# Multi cluster pickle:",
        "#     " + multi_cluster_name + " (" + clus_time + ")"]
    header.extend(cluster_lines)    

    # cluster image file names and times
    header.append("# Output cluster images:")
    for file_ in cluster_files:
        file_name = cluster_files[file_].name
        try:
            file_time = time.asctime(
                time.localtime(os.path.getmtime(file_name)))
        except OSError:
            con_bound_time = 'not written'
        header.append("#     " + file_name + " (" + file_time + ")")

    # distance file names and times
    header.append("# Distance files:")
    for file_name in distance_files.values():
        try:
            file_time = time.asctime(
                time.localtime(os.path.getmtime(file_name)))
        except OSError:
            con_bound_time = 'does not exist'
        header.append("#     " + file_name + " (" + file_time + ")")

    # results file names
    header.extend(common.format_file_info(
            name=[conn_clust_res_file_name],
            description="Connectivity clustering results"))
    header.extend(common.format_file_info(
            name=[bound_res_file_name],
            description="Hierarchical clustering of boundaries results"))
    header.extend(common.format_file_info(
            name=[conn_res_file_name],
            description="Hierarchical clustering of connectors results"))

    # write boundary ids
    header.extend("#")
    header.extend(boundary_ids_lines)

    # 
    header.extend([
            "#",
            " Clustered items:",
            "#   - number of boundaries: " + str(len(bound.ids)),
            "#   - number of connections: " + str(len(segments.ids))])

    # hierarchical boundary clustering parameters
    if hi_cluster_boundaries:
        header.extend([
                "#",
                "# Hierarchical boundary clustering parameters:",
                "#   - clustering method: minimal euclidean distance",
                "#   - linkage: " + hi_bound_linkage,
                "#   - flat clusters criterion: " + hi_bound_criter,
                "#   - flat clusters threshold: " + str(hi_bound_thresh)])
        if hi_bound_criter == 'inconsistent':
            header.append(
                "#   - flat clusters depth: " + str(hi_bound_depth))
        header.extend([
                "#   - similarity method: " + hi_bound_similarity,
                "#   - use single-item clusters for similarity: " \
                    + str(hi_bound_single)])

    # hierarchical connection clustering parameters
    if hi_cluster_connections:
        header.extend([
                "#",
                "# Hierarchical connection clustering parameters:",
                "#   - clustering method: minimal euclidean distance",
                "#   - linkage: " + hi_conn_linkage,
                "#   - flat clusters criterion: " + hi_conn_criter,
                "#   - flat clusters threshold: " + str(hi_conn_thresh)])
        if hi_conn_criter == 'inconsistent':
            header.append(
                "#   - flat clusters depth: " + str(hi_conn_depth))
            header.extend([
                "#   - similarity method: " + hi_conn_similarity,
                "#   - use single-item clusters for similarity: " 
                + str(hi_conn_single)])

    # connectivity clustering results
    if cluster_by_connectivity:
        header.extend([
                "#",
                "# Connectivity clustering results:",
                "#   - number of clusters: " 
                + str(len(multi_cluster.connectivityBoundaries.clusters))])

    # hierarchical boundary clustering results
    if hi_cluster_boundaries:
        header.extend([
                "#",
                "# Hierarchical boundary clustering results:",
                "#   - number of clusters: "
                + str(multi_cluster.hierarchyBoundaries.nClusters), 
                "#   - similarity index: " 
                + ('%6.3f' % multi_cluster.hierarchyBoundaries.similarity),
                "#   - threshold: " + str(hi_bound_thr)])
        try:
           header.append(
               "#   - rand similarity index: " 
               + ('%6.3f' % multi_cluster.hierarchyBoundaries.rand))
        except AttributeError:
            pass
        try:
           header.append(
               "#   - b-flat similarity index: " 
               + ('%6.3f' % multi_cluster.hierarchyBoundaries.bflat))
        except AttributeError:
            pass
        try:
            header.append(
                "#   - vi similarity index: " 
                + ('%6.3f' % multi_cluster.hierarchyBoundaries.vi))
        except AttributeError:
            pass

    # hierarchical connection clustering results
    if hi_cluster_connections:
        header.extend([
                "#",
                "# Hierarchical connection clustering results:",
                "#   - number of clusters (some clusters may contain no "
                + " boundaries): " 
                + str(multi_cluster.hierarchyConnections.nClusters), 
                "#   - similarity index: " 
                + ('%6.3f' % multi_cluster.hierarchyConnections.similarity),
                "#   - threshold: " + str(hi_conn_thr)])
        try:
            header.append(
                "#   - rand similarity index: " 
                + ('%6.3f' % multi_cluster.hierarchyConnections.rand))
        except AttributeError:
            pass
        try:
            header.append(
                "#   - b-flat similarity index: "
                + ('%6.3f' % multi_cluster.hierarchyConnections.bflat))
        except AttributeError:
            pass
        try:
            header.append(
                "#   - vi similarity index: " 
                + ('%6.3f' % multi_cluster.hierarchyConnections.vi))
        except AttributeError:
            pass

    # write header
    for line in header:
        bound_res_file.write(line + os.linesep)
        conn_res_file.write(line + os.linesep)
        conn_clust_res_file.write(line + os.linesep)

    # write tables
    write_boundary_cluster_table(
        file_=bound_res_file, multi_cluster=multi_cluster,
        bound=bound, contacts=contacts)
    write_connection_cluster_table(
        file_=conn_res_file, multi_cluster=multi_cluster,
        conn=segments, contacts=contacts)
    write_connectivity_cluster_table(
        file_=conn_clust_res_file, multi_cluster=multi_cluster)

def write_boundary_cluster_table(file_, multi_cluster, bound, contacts):
    """
    Writes data table for boundary clusters.
    """

    # start head
    table_head = []
    table_head.extend([\
            "# Boundary ",
            "#    id    ",
            "#"])

    # prepare head and data
    out_vars = []
    out_format = '    %5u  '
    try:
        out_vars.append(multi_cluster.connectivityBoundaries.clustersData) 
        out_format = out_format + '     %4u     '
        table_head[0] = table_head[0] + ' Connectivity ' 
        table_head[1] = table_head[1] + '  cluster id  ' 
    except AttributeError: pass
    try:
        hi_data = multi_cluster.hierarchyBoundaries.clustersData
        out_vars.append(hi_data) 
        out_format = out_format + '     %4u     '
        table_head[0] = table_head[0] + '  Hierarchy   ' 
        table_head[1] = table_head[1] + '  cluster id  ' 
    except AttributeError: pass
    try:
        dh_data = multi_cluster.dualHierarchyBoundaries.clustersData
        # make sure the lengths are the same
        dh_data = numpy.append(dh_data, [-1] * (len(hi_data) - len(dh_data)))
        out_vars.append(dh_data) 
        out_format = out_format + '     %4u     '
        table_head[0] = table_head[0] + 'Dual Hierarchy' 
        table_head[1] = table_head[1] + '  cluster id  ' 
    except AttributeError: pass

    # make results
    results_tab = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                           indices=bound.ids, prependIndex=True)

    # append connection ids
    table_head[0] += ' Connection ids'
    table_head[1] += '              '
    for (id_, line_index) in zip(bound.ids, range(len(results_tab))):
        conn_ids = numpy.array2string(contacts.findSegments(boundaryIds=id_,
                                                            nBoundary=1))
        results_tab[line_index] = results_tab[line_index] + '   ' + conn_ids
        
    # write data
    table = ["#"]
    table.extend(table_head)
    table.extend(results_tab)
    for line in table:
        file_.write(line + os.linesep)

def write_connection_cluster_table(file_, multi_cluster, conn, contacts):
    """
    Writes data table for connection clusters.
    """

    # start head
    table_head = []
    table_head.extend([\
            "# Segment  ",
            "#    id    ",
            "#"])

    # prepare head and data
    out_vars = []
    out_format = '    %5u  '
    try:
        out_vars.append(multi_cluster.connectivityConnections.clustersData) 
        out_format = out_format + '     %4u     '
        table_head[0] = table_head[0] + ' Connectivity ' 
        table_head[1] = table_head[1] + '  cluster id  ' 
    except AttributeError: pass
    try:
        out_vars.append(multi_cluster.hierarchyConnections.clustersData) 
        out_format = out_format + '     %4u     '
        table_head[0] = table_head[0] + '  Hierarchy   ' 
        table_head[1] = table_head[1] + '  cluster id  ' 
    except AttributeError: pass
    try:
        out_vars.append(multi_cluster.dualHierarchyConnections.clustersData) 
        out_format = out_format + '     %4u     '
        table_head[0] = table_head[0] + 'Dual Hierarchy' 
        table_head[1] = table_head[1] + '  cluster id  ' 
    except AttributeError: pass

    # make results
    results_tab = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                           indices=conn.ids, prependIndex=True)

    # append boundary ids
    table_head[0] += ' Boundary ids'
    table_head[1] += '            '
    for (id_, line_index) in zip(conn.ids, range(len(results_tab))):
        conn_ids = numpy.array2string(contacts.findBoundaries(segmentIds=id_,
                                                              nSegment=1))
        results_tab[line_index] = results_tab[line_index] + '   ' + conn_ids
        
    # write data
    table = ["#"]
    table.extend(table_head)
    table.extend(results_tab)
    for line in table:
        file_.write(line + os.linesep)

def write_connectivity_cluster_table(file_, multi_cluster):
    """
    Writes data table for connectivity-based clustering
    """

    # shortcuts
    bound_clust = multi_cluster.connectivityBoundaries
    conn_clust = multi_cluster.connectivityConnections

    # start head
    table_head = []
    table_head.extend([
            "# Cluster   N bound   N conn   N link   Euler   Num loops" + 
            "  Branches   Euler   Num loops   Branches  ",
            "#    id                                 (Conn)    (Conn) " + 
            "   (Conn)    (Link)    (Link)     (Link)   ",
            "#"])

    # make results
    out_vars = [
        bound_clust.nItems, bound_clust.nConnections, bound_clust.nLinks,
        bound_clust.euler, bound_clust.nLoops, bound_clust.branches,
        bound_clust.eulerLinks, bound_clust.nLoopsLinks, 
        bound_clust.branchesLinks]
    out_format = ('  %5u    %5u    %5u    %5u    %5d     %5u      %5d' +
                  '     %5d     %5u     %5d')
    #indices = numpy.insert(bound_clust.ids, 0, 0)
    results_tab = pyto.io.util.arrayFormat(
        arrays=out_vars, format=out_format, indices=bound_clust.ids, 
        prependIndex=True)

    # make total
    table_total_head = [
        "#",
        "# Total:"] 
    results_total_tab = pyto.io.util.arrayFormat(
        arrays=out_vars, format=out_format, indices=[0], prependIndex=True)
    
    # write data
    table = ["#"]
    table.extend(table_head)
    table.extend(results_tab)
    table.extend(table_total_head)
    table.extend(results_total_tab)
    for line in table:
        file_.write(line + os.linesep)


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
    logging.info("Reading and cleaning input files")

    # read boundaries (vesicles)
    boundary_ids = clean_ids(include=in_boundary_ids, exclude=ex_boundary_ids)
    bound, nested_bound_ids = read_boundaries(boundary_ids=boundary_ids)

    # read threshold and connectivity and clean it together with contacts
    flat_bound_ids = pyto.util.nested.flatten(nested_bound_ids)
    flat_bound_ids = numpy.asarray(flat_bound_ids)
    segments = read_segments(name=in_seg_file_name, inset=bound.inset, 
                             bound_ids=flat_bound_ids)

    # initialize
    cluster_files = {}
    distance_files = {}
    multi_clust = pyto.scene.MultiCluster()

    # cluster by connectivity and write images
    if cluster_by_connectivity:

        # cluster
        logging.info("Starting connectivity clustering")
        multi_clust.connectivity(contacts=segments.contacts)

        # write
        if write_images:
            conn_bound_file = write_cluster_image(
                clusters=multi_clust.connectivityBoundaries, 
                labels=bound, base_file=in_seg_file_name, 
                cluster_directory=clust_directory, 
                cluster_suffix=conn_bound_suffix, 
                clusters_data_type=clust_data_type)
            cluster_files['conn_bound_file'] = conn_bound_file
            conn_conn_file = write_cluster_image(
                clusters=multi_clust.connectivityConnections, 
                labels=segments, base_file=in_seg_file_name, 
                cluster_directory=clust_directory, 
                cluster_suffix=conn_conn_suffix, 
                clusters_data_type=clust_data_type)
            cluster_files['conn_conn_file'] = conn_conn_file

    else:
        (conn_bound, conn_conn) = (None, None)

    # hierarchical clustering of boundaries
    if hi_cluster_boundaries:

        logging.info("Starting hierarchical clustering of boundaries")

        # get distances
        bound_dist_file = get_file_name(base_file=in_seg_file_name, 
                            directory=clust_directory, prefix=distance_prefix, 
                            suffix=bound_distance_suffix)
        bound_dist = find_distances(file_=bound_dist_file, read=read_distance,
                                segments=bound, ids=flat_bound_ids)
        distance_files['bound_dist_file'] = bound_dist_file

        # cluster
        hi_bound_thr, bound_simil = multi_clust.hierarchicalBoundaries(
            linkage=hi_bound_linkage, distances=bound_dist, ids=flat_bound_ids, 
            threshold=hi_bound_thresh, criterion=hi_bound_criter, 
            depth=hi_bound_depth, contacts=segments.contacts, 
            reference='connectivity', similarity=hi_bound_similarity, 
            single=hi_bound_single)
        # ToDo: deal with no segments situation 
        multi_clust.hierarchyBoundaries.findSimilarity(
            reference=multi_clust.connectivityBoundaries, method=None)

        # write
        if write_images:
            hi_bound_file = write_cluster_image(
                clusters=multi_clust.hierarchyBoundaries, 
                labels=bound, base_file=in_seg_file_name, 
                cluster_directory=clust_directory, 
                cluster_suffix=hi_bound_suffix, 
                clusters_data_type=clust_data_type)
            cluster_files['hi_bound_file'] = hi_bound_file
            dual_hi_conn_file = write_cluster_image(
                clusters=multi_clust.dualHierarchyConnections, 
                labels=segments, base_file=in_seg_file_name, 
                cluster_directory=clust_directory, 
                cluster_suffix=dual_hi_conn_suffix, 
                clusters_data_type=clust_data_type)
            cluster_files['dual_hi_conn_file'] = dual_hi_conn_file

    else:
        hi_bound_thr, hi_bound, dual_hi_conn = (None, None, None)

    # hierarchical clustering of connections
    if hi_cluster_connections:

        logging.info("Starting hierarchical clustering of connectors")

        # get distances
        conn_dist_file = get_file_name(base_file=in_seg_file_name, 
                            directory=clust_directory, prefix=distance_prefix, 
                            suffix=conn_distance_suffix)
        conn_dist = find_distances(file_=conn_dist_file, read=read_distance,
                                segments=segments)
        distance_files['conn_dist_file'] = conn_dist_file

        # cluster
        hi_conn_thr, bound_simil = multi_clust.hierarchicalConnections(
            linkage=hi_conn_linkage, distances=conn_dist, ids=segments.ids, 
            threshold=hi_conn_thresh, criterion=hi_conn_criter, 
            depth=hi_conn_depth, contacts=segments.contacts, 
            reference='connectivity', similarity=hi_conn_similarity, 
            single=hi_conn_single)
        multi_clust.hierarchyConnections.findSimilarity(
            reference=multi_clust.connectivityConnections, method=None)

        # write
        if write_images:
            hi_conn_file = write_cluster_image(
                clusters=multi_clust.hierarchyConnections, 
                labels=segments, base_file=in_seg_file_name, 
                cluster_directory=clust_directory, 
                cluster_suffix=hi_conn_suffix, 
                clusters_data_type=clust_data_type)
            cluster_files['hi_conn_file'] = hi_conn_file
            dual_hi_bound_file = write_cluster_image(
                clusters=multi_clust.dualHierarchyBoundaries, 
                labels=bound, base_file=in_seg_file_name, 
                cluster_directory=clust_directory, 
                cluster_suffix=dual_hi_bound_suffix, 
                clusters_data_type=clust_data_type)
            cluster_files['dual_hi_bound_file'] = dual_hi_bound_file

        logging.info("Hierarchical connection clustering done")

    else:
        hi_conn_thr, hi_conn, dual_hi_bound = (None, None, None)

    # pickle all clusters
    if pickle_clusters:
        if pickle_contacts:
            cont = segments.contacts
        else:
            cont = None
        multi_clust_name = pickle_all_clusters(
            multi_clust=multi_clust, base_file=in_seg_file_name, 
            directory=clust_directory, suffix=pickle_suffix, contacts=cont)

    # write results
    write_cluster_results(multi_cluster=multi_clust, 
        hi_bound_thr=hi_bound_thr, hi_conn_thr=hi_conn_thr, 
        multi_cluster_name=multi_clust_name, segments=segments, bound=bound, 
        multi_bound_ids=nested_bound_ids, contacts=segments.contacts,
        base_file=in_seg_file_name, cluster_directory=clust_directory, 
        result_bound_suffix=result_bound_suffix, 
        result_conn_suffix=result_conn_suffix, cluster_files=cluster_files,
        distance_files=distance_files)


# run if standalone
if __name__ == '__main__':
    main()
