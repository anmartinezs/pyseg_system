#!/usr/bin/env python
"""
There are two tasks preformed by this script:
    1) An image is segmented using the hierarchical connectivity procedure 
to yield connectors (segments). 
    2) These connectors are classified according the specified 
classification criteria. 
This script can be used to execute these two tasks together, or separately. 
Both tasks can be followed by the analysis of the connectors. 

The analysis can be done on all connectors (after the segmentation), on 
connectors after classification, or both times. 

The main applications of this script is to find and analyze connectors between 
given boundaries, such as among vesicles and between vesicles and other 
membranes. Each connector is typically required to make contacts with at least 
two boundaries.

This script supersedes connections.py, as it has almost all functionality of 
connections.py (except dynamic segmentation and making vesicle clusters) while 
it has several software improvements.

This script may be placed anywhere in the directory tree.

Important notes:

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

  - Currently (r997) division by 0 warnings are printed. These might be ignored
because they most likely arise from calculating denisty of a 0-volume segment
and are also a consequence of the changed behavior of scipy. 

$Id: connectors.py 1493 2018-11-08 16:23:38Z vladan $
Author: Vladan Lucic 
"""
__version__ = "$Revision: 1493 $"

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
from pyto.scene.segmentation_analysis import SegmentationAnalysis

# import ../common/tomo_info.py
tomo_info = common.__import__(name='tomo_info', path='../common')

# to debug replace INFO by DEBUG
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')

##############################################################
#
# Parameters
#
##############################################################

############################################################
#
# Image (grayscale) file, should be in the mrc or em format (recognized 
# extensions are 'mrc' and 'em'). 

# name of the image file
if tomo_info is not None: image_file_name = tomo_info.image_file_name
#image_file_name = "tomo.mrc"

###############################################################
#
# Labels file, specifies boundaries and possibly other regions. If the file
# is in em or mrc format shape, data type, byte order and array order are not
# needed (should be set to None). If these variables are specified they will
# override the values specified in the headers.
#
# If multiple labels files are used, labels_file_name, all_ids and boundary_ids
# have to be tuples (or lists) of the same lenght.
# Note: This feature was working in the past. However, it was not used nor 
# tested for quite some time now, so it is possible that it is incompatible 
# with some of the more recent developments. 
#
# Variables of this section are not needed if do_segmentation_flag is False 
#
# Note: The use of multiple label files is depreciated. Use add_segments.py
# to combine multiple label files into one.
#

# name of (one or multiple) labels file(s) containing boundaries
if tomo_info is not None: labels_file_name = tomo_info.labels_file_name
#labels_file_name = "labels.raw"   # one 
#labels_file_name = ("labels_1.dat", "labels_2.dat", "labels_3.dat") # multiple 

# labels file dimensions (size in voxels)
labels_shape = None   # shape given in header of the labels file (if em
                      # or mrc), or in the tomo (grayscale image) header   
#labels_shape = (512, 512, 190) # shape given here

# labels file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float16', 
# 'float64') 
if tomo_info is not None: labels_data_type = tomo_info.labels_data_type
#labels_data_type = 'uint16'

# labels file byteOrder ('<' for little-endian, '>' for big-endian)
labels_byte_order = '<'

# labels file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis fastest)
labels_array_order = 'FORTRAN'

# offset of labels in respect to the data (experimental)
labels_offset = None             # no offset
#labels_offset = [10, 20, 30]    # offset specified

###########################################################
#
# Segmentation parameters
#

# if True do segmentation, otherwise read pickle containing segmentation 
do_segmentation_flag = True

# threshold list (not needed if do_segmentation_flag is False)
threshold = numpy.arange(-0.3, 0, 0.01)  

# Currently not implemented
# in addition to threshold list thresholds are also chosen dynamically, so that
# the number of new connectors at each level is limited to this number
#max_new = 2        # dynamical thresholds
#max_new = None     # thresholds from threshol list 

# Currently not implemented
# smallest threshold step allowed for the dynamical threshold scheme
#min_step = 0.001       

# segmentation pickle name, used only if do_segmentation_flag is False. If None,
# this name is formed based on the image_file_name.
segmentation_file_name = 'sa.pkl'

###############################################################
#
# Structuring element connectivities. 
#
# These parameters define what are considered neghboring voxels:
#   - 1: those that have common face in 3D (edge in 2D)
#   - 2: those that have common edge or face in 3D (vertex or edge in 2D)
#   - 3: those that have common vertex, edge or face in 3D  
#
# Note: values different from 1 should be considered experimental. Values
# 2 and 3 should work in 3D, but its not clear how to interpret results,
# especially those obtained by topological analysis. 

# Segment connectivity, determines what is a connected segment (can be 1-3, 
# 4 is experimental)
struct_el_connectivity = 1

# Segment-boundary connectivity, detection of contacts between segments and 
# boundaries
contact_struct_el_connectivity = 1

# counting contacts between segments and boundaries (currently not used)
count_struct_el_connectivity = 1

###########################################################
#
# Boundaries and other segments in labels
#
# Important: 
#   - All ids in this section should be >0
#   - Voxels at the tomogam side should not belong to any boundaries
#
# Important note: Incorrect id specification (all_ids, boundary_ids and 
# conn_region) accounts for a majority of problems encountered while running 
# this script. 
#
# Variables of this section are not needed if do_segmentation_flag is False

# ids of all segments defined in boundary labels file, including boundary ids.
# Segments of the boundary file that are not listed here are discarded, that is
# set to 0 for segmentation purposes (and consequently may be used for the 
# segmentation region if conn_region=0)
# Single labels file forms:
#all_ids = [2,3,5,6]       # individual ids, single file
if tomo_info is not None: all_ids = tomo_info.all_ids
#all_ids = range(1, 145)  # range of ids, single file
#all_ids = None         # all ids are to be used, single file
#
# ids of all segments for multiple labels files can use all above forms,
# the following means ids 2, 3 and 6 from labels_1, all ids from labels_2 and
# ids 3, 4, ... 8 from labels_3
#all_ids = ([2,3,5,6], None, range(3,9))  # multiple files

# ids of all boundaries. All formats available for all_ids are accepted here 
# also. In addition, nested list can be used where ids in a sublist are 
# uderstood in the "or" sense, that is all boundaries listed in a sublist form 
# effectivly a single boundary
#boundary_ids = [2,3,5]       # individual ids, single file
if tomo_info is not None: boundary_ids = tomo_info.boundary_ids
#boundary_ids = range(1, 144)  # range of ids, single file
#boundary_ids = None         # all segments are to be used, single file
#boundary_ids = [[2,3], 4, 5, 6]  #  2 and 3 taken together, single file

# ids of all boundaries for multiple labels files. All formats available for
# boundary_ids are accepted here also.
#boundary_ids = ([2,3,5], range(2,64), [[6,7], 8])  # multiple files

# check if specified boundaries exist and if they are not disconnected
check_boundaries = True  

# id shift in each subsequent labels (in case of multiple labels files) 
#shift = None     # shift is determined automatically
shift = 254

# number of boundaries that each segment is required to contact
n_boundary = 2      # for connectors between 2 boundaries
#n_boundary = 1      # for connectors contacting one boundary

# 'exact' to require that segments contact exactly n_boundary boundaries, or
# 'at_least' to alow n_boundary or more contacted boundaries 
count_mode = 'exact'

# Id of the segmentation region (where connectors can be formed), has to be the
# same in all labels files. 
#
# While it is still possible to use 0 (if segmentation region is not 
# specified in boundary file), this is strongly discouraged.
# 
# Note that decreasing segmentation region decreases computational time.
# Not used if read_connections is True.
#conn_region = 144      # segmentation region specified
if tomo_info is not None: conn_region = tomo_info.segmentation_region
#conn_region = 3        # segmentation region not specified in boundary file

# Maximal distance to the boundaries for the segmentation region. Connectors are
# formed in the area surrounding boundaries (given by all ids
# specified in boundary_ids) that are not further than free_size from the
# boundaries and that are still within conn_region. If free_size = 0, the area
# surrounding boundaries is maximaly extended (to the limits of conn_region).
# Not used if read_connections is True.
#free_size = 10  # the same for all boundaries, one labels file
free_size = 0   # segmentation region on the whole background
#free_size = [10, 12, 15, 12, 10] # one for each boundary

# Defines the manner in which the areas surrounding individual boundaries 
# (defined by free_size) are combined. These areas can be added 
# (free_mode='add') or intersected (free_mode='intersect'). 
# Not used if free_size = 0
# Not used if read_connections is True. 
free_mode = 'add'

# If None, the region where the connectors are made is determined by using all
# boundaries together. Alternatively, if an iterable of lists is
# given the segmentation is done in a free region established between
# (usually two) boundaries given in each element of this list.
# Works only for single labels file. (experimental)
#boundary_lists = pyto.util.probability.combinations(elements=boundary_ids,
#                                                   size=n_boundary,
#                                                   repeat=False, order=False)
# simpler version of the above 
#boundary_lists = [(2,3), (2,6), (3,6)]
boundary_lists = None

#####################################################################
#
# Classification parameters
#
# Segments can be classified by one or more classification methods (defined 
# below). All classes generated by one classification are classified separately
# by a subsequent classification. 
#
# All possible classification are listed below. They can be executed in any 
# order, the order is determined by their numbering. For each classification 
# variable class_i_type (i>=1) has to be defined as well as variables that 
# hold classification parameters, as follows:
#   class_i_type = 'keep':
#     - class_i_mode: 'new' for smallest or 'new_branch_tops' for biggest
#       segments before merging
#     - class_i_name: name of the class
#   class_i_type = 'volume':
#     - class_i_volumes: list of volumes (in voxels)
#   class_i_type = 'contacted_ids':
#     - class_i_ids: ids of contacted boundaries (such as pre and post 
#       membranes)
#     - class_i_rest: if True a class of segments not used in other classes
#       is generated also
#   class_i_type = 'n_contacted'
#     - class_i_nContacted: list of numbers of contacted boundaries 
# For all classifications except in 'keep', variable parameter class_i_names
# can be defined, it has to be a list of class names (has to have the same 
# number of elements as there are classes)

# flag indicating if classifications is done
do_classify_flag = True

# classification 1
class_1_type = 'keep'
class_1_mode = 'new'

# classification 2
class_2_type = 'contacted_ids'
class_2_ids = [1]
class_2_rest = True
class_2_names = ['AZ', 'rest']

# classification 3
#class_3_type = 'volume'
#class_3_volumes = [5, 100, 1000]
#class_3_names = ['small', 'big']


###########################################################
#
# Analysis
#
# For each of these analysis tasks a flag do_<task> specifies if the 
# corresponding ananlysis task should be done, while where_<task> determines
# if it should be calculated after segmentation ('segmentation'),
# after classification ('classification') or after both segmentatation and
# classification ('both').

# calculate segment length (if length already exists in the input hierarchy and
# all length-related parameters are the same the old length values are kept)
do_length = True
# Note: currently (r921) the values calculated for 'segmentation' are not
# propagated to classification
where_length = 'both'   

# segment length calculation mode:
#   - (one boundary) 'b-max' or 'c-max': between contacts and the most distant
#     element of a segment, where contacts belong to a boundary or a segment,
#     respectivly  
#   - (two boundaries) 'b2b' for minimal distance between contact
#     points on boundaries, 'c2c' on segments, 'b2c' between boundary contact
#     points on one end and segment contact points on the other end and
#     'c2b' the other way round
length_contact_mode = 'c2c'
#length_contact_mode = 'b2c' expected to become the default in the future

# segment length line mode:
#   - 'straight': straight-line distance between contacts (two boundaries) or
#     between a contact and an end point
#   - 'mid' and 'mid-seg': sum of distances between contact points and a 
#      mid point (two boundaries only), 'mid-seg' might be slightly more 
#      precise
length_line_mode = 'mid'

# topological properties (takes more time than other analysis tasks)
do_topology = True
where_topology = 'classification'

# distance of each segment to the specified region. 
do_distance_to = True
where_distance_to = 'segmentation'

# Region id, or if more than one distance region is specified, for each segment 
# the distance to the closest region is calculated. In case of multiple labels 
# files this id is understood after the ids of the labels files are shifted.
distance_id = 1   # one region
#distance_id = range(2,20)   # multiple regions

# the way distance is calculated ('center', 'mean' or 'min')
distance_mode = 'mean' 

# distance between connected boundaries 
# Note: used only for connectors that contact exactly two boundaries 
do_boundary_distance = True
where_boundary_distance = 'segmentation'

############################################################
#
#
# Ordering of connectors (only the first True ordering is applied)
#
# This determines the order in which the segments are ordered in the results 
# file, and has no influence on analysis
#

# order segments by the boundaries contacted
order_by_boundaries = True

# order segments by volume
order_by_volume = False

###########################################################
#
# Segments (connectors) files. The file name is formed as:
#
#   <conn_directory>/<conn_prefix> + image root + <threshold_label> 
#       + <threshold> + <class_names> + <conn_suffix>
#
# Notes: 
#   - Final connectors file is always written
#   - Threshold label and threshold are used only for individual threshold
#     segmentations
#   - Class names is used only when classification is done

# write connectors flag
write_connections_flag = True

# connectors directory
conn_directory = ''

# connectors file name prefix (no directory name)
conn_prefix = ''

# include image file root (filename without directory and extension)
conn_insert_root = True

# connectors file name suffix
conn_suffix = ".mrc"

# connectors data type, 'uint8' , or
#conn_data_type = 'uint8'       # if max segment id is not bigger than 255
#conn_data_type = 'uint16'      # more than 255 segments, em file
conn_data_type = 'int16'       # more than 255 segments, mrc file (not tested)

# controls what kind of data casting may occur: 'no', 'equiv', 'safe', 
# 'same_kind', 'unsafe'. Identical to numpy.astype()
# Warning: Unless using 'safe', check results to see if label ids are too high 
# for conn_data_type. However, 'safe' will most likely fail for conn_data_type
# that have less than 32 or 64 bits
conn_casting = 'unsafe'

############################################################
#
# Segmentation and analysis results and pickles. The results file name is:
#
#   <res_directory>/<res_prefix> + image root + <threshold_label> 
#       + <threshold> + <class_names> + <res_suffix>
#
# while the pickle file name is:
#
#   <res_directory>/<res_prefix> + image root + <threshold_label> 
#       + <threshold> + <class_names> + <sa_suffix>
#
# Notes: 
#   - Threshold label and threshold are used only for individual threshold
#     segmentations
#   - Class names is used only when classification is done

# write results flag
write_results_flag = True

# write pickles at individual thresholds
pickle_all_thresholds = True

# results directory
res_directory = ''

# results file name prefix (no directory name)
res_prefix = ''

# include image file root (filename without directory and extension)
res_insert_root = True

# results file name suffix
res_suffix = ".dat"

# include total values (for all connectors taken together) in the results, id 0
res_include_total = True

# threshold label 
threshold_label = '_thr-'

# threshold format used to include threshold in file names and for reporting
threshold_format = '%6.3f'

############################################################
#
# Segmentation and analysis pickle file. The file name is formed as:
#
#   <sa_directory>/<sa_prefix> + image root + <sa_suffix>
# 

# directory
sa_directory = ''

# layer results file name prefix (no directory name)
sa_prefix = ''

# include image file root (filename without directory and extension)
sa_insert_root = True

# segmentation and analysis results file name suffix
sa_suffix = ".pkl"


################################################################
#
# Work
#
################################################################

######################################################
#
# Main functions
#

def do_segmentation(image, bound, bound_ids, sa_file_name, inset):
    """
    Makes hierarchical segmentation
    """

    # open already existing summary file or write header

    # make threshold list
    if isinstance(threshold, (numpy.ndarray, list)):
        thresh = threshold
    else:
        thresh = [threshold]

    # prepare for segmentation and analysis
    sa = SegmentationAnalysis(boundary=bound)
    sa.setSegmentationParameters(
        nBoundary=n_boundary, boundCount=count_mode, boundaryIds=bound_ids, 
        mask=conn_region, freeSize=free_size, freeMode=free_mode,
        structElConn=struct_el_connectivity, 
        contactStructElConn=contact_struct_el_connectivity,
        countStructElConn=count_struct_el_connectivity)
    sa.setAnalysisParameters(
        lengthContact=length_contact_mode, lengthLine=length_line_mode,
        distanceId=distance_id, distanceMode=distance_mode)

    # fugure out what to analyze 
    do_now = what_to_do(where='segmentation')

    # analysis iterator
    tc_iter = sa.tcSegmentAnalyze(
        image=image, thresh=thresh, order='<', count=False, doDensity=True, 
        doMorphology=True, doLength=do_now['length'], 
        doTopology=do_now['topology'], doDistanceTo=do_now['distance_to'], 
        doBoundDistance=do_now['boundary_distance'])
    
    # segment and analysis
    thresh_lines = []
    for tr in thresh:
        
        tr_str, tr_long_str = common.format_param(
            value=tr, name=threshold_label, format=threshold_format)
        logging.info('Starting segmentation and analysis for threshold ' 
                     + tr_str)

        # segment at this threshold
        sa_level, level, curr_thresh = tc_iter.next()

        # set analysis parameters
        sa.setAnalysisParameters(
            lengthContact=length_contact_mode, lengthLine=length_line_mode,
            distanceId=distance_id, distanceMode=distance_mode)

        # write connectors
        if write_connections_flag and (len(sa_level.segments.ids) > 0):
            conn_file_name = common.make_file_name(
                directory=conn_directory, prefix=conn_prefix,
                insert_root=conn_insert_root, reference=image_file_name,
                param_name=threshold_label, param_value=tr, 
                param_format=threshold_format, suffix=conn_suffix)
            common.write_labels(
                labels=sa_level.segments, name=conn_file_name,
                data_type=conn_data_type, inset=inset, length=image.length,
                casting=conn_casting)

        # pickle the results
        if pickle_all_thresholds:
            pkl_file_name = common.make_file_name(
                directory=sa_directory, prefix=sa_prefix, 
                insert_root=sa_insert_root, reference=image_file_name, 
                param_value=tr, param_name=threshold_label,
                param_format=threshold_format, suffix=sa_suffix)
            common.write_pickle(obj=sa_level, file_name=pkl_file_name, 
                                image=['labels'], compact=['labels.contacts'])

        # write single threshold results file
        if write_results_flag:
            logging.info("  Ordering")
            ordered_ids = order(analysis=sa_level)
            logging.info("  Writing results")
            write_results(analysis=sa_level, thresh=tr, ids=ordered_ids) 

        # append to all threshold results file
        one_line = append_all_thresh_results(analysis=sa_level, thresh=tr) 
        if one_line is not None:
            thresh_lines.append(one_line)

    # hierarchy
    tc = tc_iter.send(True) 
    sa.hierarchy = tc

    # check if needed
    sa.hierarchy.full_inset = inset

    logging.info('Writing segmentation and analysis pickle and result files')

    # pickle
    common.write_pickle(obj=sa, file_name=sa_file_name, image=['hierarchy'],
                        compact=['hierarchy.contacts'])

    # hierarchy image
    if write_connections_flag and (len(sa.labels.ids) > 0):
        hi_file_name = common.make_file_name(
            directory=sa_directory, prefix=sa_prefix, 
            insert_root=sa_insert_root, reference=image_file_name, 
            suffix=conn_suffix)
        common.write_labels(
            labels=sa.labels, name=hi_file_name,
            data_type=conn_data_type, inset=inset, length=image.length,
            casting=conn_casting)    

    # open all thresholds file and write header
    all_thresh_file = open_results()
    write_all_thresh_header(fd=all_thresh_file, analysis=sa)

    # write individual threshold lines
    for line in thresh_lines:
        all_thresh_file.write(line + os.linesep)

    # write bottom for the all thresholds
    #write_all_thresh_bottom(fd=all_thresh_file, analysis=sa)

    return sa

def do_classification(sa, image=None, inset=None, ref_name=None):
    """
    Iterator that classifies hierarchical segments

    Arguments:
      - sa: (Hierarchy) hierarchical segmentation
      - image
      - inset: connectors image is adjusted to this inset before it's written

    Yelds (for each class):
      - cls: (SegmentationAnalysis) segmentation and analysis for one 
      class (classification result)
      - cls_name: class name
      - cls_file_name: pickle file name
    """

    # set inset
    if inset is None:
        inset = sa.labels.full_inset

    # reference file name
    if ref_name is None:
        ref_name = image_file_name

    # set classifications
    set_classifications(analysis=sa)

    # set analysis parameters
    sa.setAnalysisParameters(
        lengthContact=length_contact_mode, lengthLine=length_line_mode,
        distanceId=distance_id, distanceMode=distance_mode)
    #do_distance_to = (distance_id is not None)

    # classify 
    if sa.classifications is not None:

        # fugure out what to analyze 
        do_now = what_to_do(where='classification')

        # analyze
        for cls, cls_name in sa.classifyAnalyze(
            hierarchy=sa.hierarchy, image=image, doDensity=True, 
            doMorphology=True, doLength=do_now['length'], 
            doTopology=do_now['topology'], doDistanceTo=do_now['distance_to'], 
            doBoundDistance=do_now['boundary_distance']):

            logging.info('  Writing segmentation and analysis pickle and ' 
                         + 'result files for class ' + cls_name + '.')

            # pickle
            suffix = cls_name + sa_suffix
            cls_file_name = common.make_file_name(
                directory=sa_directory, prefix=sa_prefix, suffix=suffix,
                insert_root=sa_insert_root, reference=ref_name)
            common.write_pickle(obj=cls, file_name=cls_file_name, 
                      image=['hierarchy'], compact=['hierarchy.contacts'])

            # connectors (labels)
            suffix = cls_name + conn_suffix
            conn_file_name = common.make_file_name(
                directory=sa_directory, prefix=sa_prefix, suffix=suffix,
                insert_root=sa_insert_root, reference=ref_name)
            common.write_labels(
                labels=sa.labels, name=conn_file_name,
                data_type=conn_data_type, inset=inset, ids=cls.labels.ids,
                length=image.length, casting=conn_casting)

            # write results            
            logging.info("  Ordering")
            ordered_ids = order(analysis=cls)
            logging.info("  Writing results")
            write_results(analysis=cls, ref_name=ref_name, 
                          name=cls_name, ids=ordered_ids) 

            yield cls, cls_name, cls_file_name

def what_to_do(where):
    """
    Returns dictionary that specifies which analysis task should be calculated.

    Theis decision is made based on the requirement to do a secific task 
    (do_<task> variable), the requirement at what stage to do it (where_<task>
    variable) and the current position of the calling method specified by
    arg where. 
    """

    res = {}

    if do_length and ((where_length == where) or (where_length == 'both')):
        res['length'] = True
    else:
        res['length'] = False
    if do_topology and ((where_topology == where) 
                        or (where_topology == 'both')):
        res['topology'] = True
    else:
        res['topology'] = False
    if do_distance_to and ((where_distance_to == where) 
                           or (where_distance_to == 'both')):
        res['distance_to'] = True
    else:
        res['distance_to'] = False
    if do_boundary_distance and ((where_boundary_distance == where) 
                                 or (where_boundary_distance == 'both')):
        res['boundary_distance'] = True
    else:
        res['boundary_distance'] = False

    return res

def set_classifications(analysis, max_class=100):
    """
    Parses input variables and sets classification parameters.

    Does up to max_class classifications.
    """

    for ind in range(1,max_class):

        # finish if no class type
        # Note: sys.modules[__name__].locals() fails with iPython 0.10
        class_type = globals().get('class_' + str(ind) + '_type')
        if class_type is None:
            continue

        # parse classification parameters
        prefix = 'class_' + str(ind) + '_'
        args = {}
        for name, value in globals().items():
            if name.startswith(prefix):
                class_arg_name = name.lstrip(prefix)
                if class_arg_name != 'type':
                    args[class_arg_name]  = value

        # set parameters
        analysis.addClassificationParameters(type=class_type, args=args)
    
###########################################
#
# Read image and boundary files
#

def read_image():
    """
    Reads image file and returns an segmentation.Grey object

    Depreciated
    """
    image = pyto.segmentation.Grey.read(file=image_file_name)
    return image

def read_boundaries(check=True, suggest_shape=None):
    """
    Reads file(s) containing boundaries.

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
    #    bound, multi_boundary_ids = read_multi_boundaries()
    #else:
    #    bound = read_single_boundaries()
    #    multi_boundary_ids = [boundary_ids]

    # offset
    #bound.offset = labels_offset

    # check
    #if check:
    #    nonun = bound.findNonUnique()
    #    if len(nonun['many']) > 0:
    #        logging.warning(
    #            "The following discs are disconnected: " + str(nonun.many))
    #    if len(nonun['empty']) > 0:
    #        logging.warning(
    #            "The following discs do not exist: " + str(nonun.empty))

    bound, multi_boundary_ids = common.read_labels(
        file_name=labels_file_name, ids=all_ids, label_ids=boundary_ids, 
        shift=shift, shape=labels_shape, suggest_shape=suggest_shape, 
        byte_order=labels_byte_order, data_type=labels_data_type,
        array_order=labels_array_order,
        clean=True, offset=labels_offset, check=check)

    return bound, multi_boundary_ids

def is_multi_boundaries():
    """
    Returns True if maultiple boundaries (labels) files are given.

    Depreciated
    """
    if isinstance(labels_file_name, str):
        return False
    elif (isinstance(labels_file_name, tuple) 
          or isinstance(labels_file_name, list)):
        return True
    else:
        raise ValueError("Labels_file_name has to be either a string (one " 
                         + "labels file) or a tuple (multiple labels files).")

def read_single_boundaries():
    """
    Reads and initializes boundaries from a sigle labels file.

    Returns (Segment) boundaries.

    Depreciated
    """

    # read labels file and make a Segment object
    bound = pyto.segmentation.Segment.read(
        file=labels_file_name, ids=all_ids, clean=True, 
        byteOrder=labels_byte_order, dataType=labels_data_type,
        arrayOrder=labels_array_order, shape=labels_shape)

    return bound

def read_multi_boundaries():
    """
    Reads and initializes boundaries form multple labels file. The label ids
    are shifted so that they do not overlap and the labels (boundaries) are
    merged.

    Returns (bound, shifted_boundary_ids) where:
      - bound: (Segment) merged boundaries
      - shifted_boundary_ids: (list of ndarrays) shifted ids

    Depreciated
    """

    # read all labels files and combine them in a single Segment object
    bound = pyto.segmentation.Segment()
    curr_shift = 0
    shifted_boundary_ids = []
    for (l_name, a_ids, b_ids) in zip(labels_file_name, all_ids, boundary_ids):
        curr_bound = pyto.segmentation.Segment.read(
            file=l_name, ids=a_ids, clean=True, 
            byteOrder=labels_byte_order, dataType=labels_data_type,
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

    Depreciated
    """

    # read
    in_file = open(hierarchy_file_name, 'rb')
    hi = pickle.load(in_file)
    in_file.close()

    return hi

###########################################
#
# Write output files
#

def open_results(thresh=None, ref_name=None, name=''):
    """
    Opens a results file name and returns it.

    Arguments:
      - thresh: current threshold
      - ref_name: reference file name
      - name: inserted before suffix in the results file name, usually a 
      class name
    """
    
    # threshold
    if thresh is None:
        thresh_name = ''
    else:
        thresh_name = threshold_label
    suffix = name + res_suffix

    # reference file name
    if ref_name is None:
        ref_name = image_file_name

    # make results file name
    res_file_name = common.make_file_name(
        directory=res_directory, prefix=res_prefix, 
        insert_root=res_insert_root, reference=ref_name, 
        param_value=thresh, param_name=thresh_name, 
        param_format=threshold_format, suffix=suffix)

    # open file
    res_file = open(res_file_name, 'w')
    return res_file

def write_results(analysis, thresh=None, ref_name=None, name='', ids=None):
    """
    Writes segmentation and analysis results for a single threshold.

    If arg ids is not given, analysis.labels.ids is used.

    Arguments:
      - analysis: segmentation analysis, need to contain either hierarchy
      (hierarchical segmentation) or segments (flat segmentation) attribute
      - threshold: threshold
      - ref_name: reference file name
      - name: inserted before suffix in the results file name, usually a
      class name
      - ids: segment ids, the results are printed in this order
    """

    # open results file
    fd = open_results(thresh=thresh, ref_name=ref_name, name=name)

    # reference file name
    if ref_name is None:
        ref_name = image_file_name

    # top of the header
    header = make_top_header(ref_name=ref_name)

    # reference file name
    if ref_name is None:
        ref_name = image_file_name

    # rest of file names
    segments_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, suffix=name+conn_suffix,
        insert_root=sa_insert_root, reference=ref_name,
        param_value=thresh, param_name=threshold_label,
        param_format=threshold_format)
    header.extend(common.format_file_info(
            name=segments_file_name, 
            description="Segmented image (out)"))
    pkl_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, suffix=name+sa_suffix,
        insert_root=sa_insert_root, reference=ref_name,
        param_value=thresh, param_name=threshold_label,
        param_format=threshold_format)
    header.extend(common.format_file_info(
            name=pkl_file_name, 
            description="SegmentationAnalysis pickle (out)"))

    # figure out type
    labels = analysis.labels
    if labels is not None:
        if isinstance(labels, pyto.segmentation.Hierarchy):
            mode = 'hierarchy'
        if isinstance(labels, pyto.segmentation.Segment):
            mode = 'segments'
    else:
        raise ValueError("Argument analysis has neither hierarchy nor "
                         + "segments attribute")

    # variable shorhands
    if ids is None:
        ids = labels.ids
    dens = analysis.density
    mor = analysis.morphology

    # make parameters header 
    param_header = make_header(analysis=analysis, thresh=thresh, name=name)
    header.extend(param_header)

    # write header
    for line in header: 
        fd.write(line + os.linesep)

    # return if no segments
    if len(ids) == 0:
        fd.flush()
        return
    
    # id
    tab_head = ["# Id ",
                "#    "]    
    out_vars = []
    out_format = ' %3u'

    # threshold
    if mode is 'hierarchy':
        tab_head[0] += " Thresh"
        tab_head[1] += "       "
        out_vars += [labels.thresh]
        out_format += ' %6.3f'

    # density and morphology
    tab_head[0] += "         Density               Volume Surface  S/V "
    tab_head[1] += "  mean    std     min     max                      "
    volume = mor.volume.astype(float).copy()
    for index in range(len(mor.volume)):
        if (mor.volume[index] == 0) and (index not in ids):
            volume[index] = -1.
    volume_float = numpy.asarray(mor.volume, dtype='float')
    volume_float[(mor.surface==0) & (volume_float==0)] = 1
    out_vars += [dens.mean, dens.std, dens.min, dens.max, 
                 mor.volume, mor.surface, mor.surface / volume_float]
    out_format += ' %7.3f %6.3f %7.3f %7.3f %6u %6u %6.3f'

    # topology
    if 'topology' in analysis.resultNames:
        topo = analysis.topology
        tab_head[0] += " Euler Loops Holes"
        tab_head[1] += "                  "
        out_vars += [topo.euler, topo.nLoops, topo.nHoles]
        out_format += ' %5i %5i %5i'

    # length
    if 'morphology' in analysis.resultNames: 
        try:
            if mor.length is not None:
                tab_head[0] += " Length"
                tab_head[1] += "       "
                out_vars += [mor.length]
                out_format += " %6.1f"
        except AttributeError:
            # no length
            pass

    # boundary distance
    if 'boundDistance' in analysis.resultNames:
        tab_head[0] = tab_head[0] + ' Boundary'
        tab_head[1] = tab_head[1] + ' distance'
        out_vars += [analysis.boundDistance.distance]
        out_format += ' %6.1f   ' 

    # distance to
    if 'distance' in analysis.resultNames:
        distance_to = analysis.distance
        dist_reg = numpy.zeros(distance_to.distance.shape[0], dtype='int')
        dist_reg = dist_reg + distance_id
        tab_head[0] = tab_head[0] + '  Distance  '
        tab_head[1] = tab_head[1] +('         to ')  
        out_vars += [distance_to.distance, dist_reg]
        out_format += ' %5.1f %4i ' 

    # contacts
    contacts = labels.contacts
    tab_head[0] += "  Contacts "
    tab_head[1] += "           "

    # write the table head 
    for line in tab_head: 
        fd.write(line + os.linesep)

    # write the individual results with boundaries added
    res_table = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                         indices=ids, prependIndex=True)
    for (sid, line) in zip(ids, res_table):
        boundIds = numpy.array2string(
            contacts.findBoundaries(segmentIds=sid, nSegment=1))
        fd.write(line + "  %s" % boundIds + os.linesep)
        
    # write summary results
    if res_include_total:

        # write results for all connectors together
        if (mode == 'segments') or (mode == 'hierarchy'):
            fd.write(os.linesep + "# All segments together:" + os.linesep)
            try:
                tot_line = pyto.io.util.arrayFormat(arrays=out_vars, 
                                                format=out_format,
                                                indices=[0], prependIndex=True)
                fd.write('#' + tot_line[0] + os.linesep)
            except TypeError:
                logging.warning(
                    "Could not write all segments results.") 

        # write background results
        if 'backgroundDensity' in analysis.resultNames:
            bkg_dens = analysis.backgroundDensity
            fd.write(os.linesep + "# Background:" + os.linesep)
            out_vars = (bkg_dens.mean, bkg_dens.std, bkg_dens.min, 
                        bkg_dens.max, bkg_dens.volume)
            out_format = '    %7.3f %6.3f %7.3f %7.3f %6u'
            bkg_line = out_format % out_vars
            fd.write('#' + bkg_line + os.linesep)

        # write total cleft results
        if 'regionDensity' in analysis.resultNames:
            reg_dens = analysis.regionDensity
            fd.write(os.linesep + "# Whole segmentation region:" + os.linesep)
            out_vars = (reg_dens.mean, reg_dens.std, reg_dens.min, 
                        reg_dens.max, reg_dens.volume)
            out_format = '    %7.3f %6.3f %7.3f %7.3f %6u'
            if mode is 'hierarchy':
                out_format = '       ' + out_format
            cleft_line = out_format % out_vars
            fd.write('#' + cleft_line + os.linesep)

    fd.flush()

def make_top_header(ref_name=None):
    """
    Returns header lines containing machine and files info
    """

    # machine info
    mach_name, mach_arch = common.machine_info()

    # out file names
    script_file_name = sys.modules[__name__].__file__

    # general 
    header = ["#",
        "# Machine: " + mach_name + " " + mach_arch,
        "# Date: " + time.asctime(time.localtime()),
        "#"]
    header.extend(common.format_file_info(
            name=script_file_name, description="Input script", 
            extra=("  "+__version__)))
    header.append("# Working directory: " + os.getcwd())
    header.append("#")

    # files
    if (not do_segmentation_flag):
        if ref_name is None:
            ref_name = image_file_name
        header.extend(common.format_file_info(name=ref_name, 
                                              description="Segmentation (in)"))
    header.extend(common.format_file_info(name=image_file_name, 
                                          description="Image (in)"))
    header.extend(common.format_file_info(name=labels_file_name, 
                                          description="Boundaries (in)"))

    return header
 
def make_header(analysis=None, thresh=None, name=''):
    """
    Returns part of the header for a result file. Can be used for a single 
    threshold or hierarchical segmentation, as well as for all thresholds.

    Arguments:
      - seg: segmentation, either hierarchical (Hierarchy) or flat (Segment)
      - thresh: threshold
      - name: inserted before suffix in the results and pickle file names,
      usually a class name
    """

    # set labels, hierarchy or segments, ids and threshold
    ids = None
    seg = None
    hierarchy = None
    if analysis is not None:
        labels = analysis.labels
        try:
            hierarchy = analysis.hierarchy
            ids = analysis.hierarchy.ids
        except AttributeError:
            try:
                seg = analysis.segments
                ids = analysis.segments.ids
            except AttributeError:
                pass
    if thresh is not None:
        tr_str, tr_long_str = common.format_param(
            value=thresh, name=threshold_label, format='%6.3f')
    header = []

    # parameters
    header = [
        "#",
        "# Boundary ids: ",
        "#     " + str(boundary_ids),
        "#",
        "# Number of boundaries contacted: " + str(n_boundary),
        "# Contact mode: " + count_mode,
        "#",
        "# Segmentation region: " + str(conn_region),
        "# Free region size: " + str(free_size),
        "# Free region mode: " + free_mode]
    
    # structuring elements
    header.extend([
            "#",
            "# Structuring element connectivities"])
    if labels is not None:
        header.extend([
                ("#     - connector formation: " 
                 + str(labels.structEl.connectivity)),
                "#     - size: " + str(labels.structEl.size),
                "#     - contact detection: " + str(labels.contactStructElConn),
                "#     - contact counting: " + str(labels.countStructElConn)])
    else:
        header.extend([
                "#     - connector formation: " + str(struct_el_connectivity),
                "#     - contact detection: " \
                    + str(contact_struct_el_connectivity),
                "#     - contact counting: " \
                    + str(count_struct_el_connectivity)])
    header.extend(["#     - topology: 1"])

    # length
    try:
        analysis.morphology.length
        header.extend([
            "#",
            "# Length",
            "#     - contact mode: " + length_contact_mode,
            "#     - line mode: " + length_line_mode,
            "#"])
    except AttributeError:
        pass

    # distance
    try:
        analysis.distance
        header.extend([
                "#",
                "# Distance:",
                "#   - region(s): " + str(distance_id),
                "#   - mode: " + distance_mode])
    except AttributeError:
        pass

    # threshold
    if thresh is not None:
        header.extend([
            "#",
            "# Threshold: " + tr_str,
            "#"])
    else:
        if isinstance(threshold, (list, numpy.ndarray)):
            n_thresholds = len(threshold)
            threshold_var = tuple(threshold)
        else:
            n_thresholds = 1
            threshold_var = threshold
        header.extend([
            "#",
            ("# Thresholds:" +
             (' ' + threshold_format) * n_thresholds) % threshold_var,
            "#"])

    # N segments
    header.extend([\
        "#",
        "# Number of segments: " + str(len(ids)),
        "#"])

    # classification
    if hierarchy is not None:
        header.extend([
                "# Classification: "])
        if analysis.classifications is None:
            header.extend([
                    "# \tNone",
                    "#"])        
        else:
            for one_class in analysis.classifications:
                header.append(
                    "# \t- type: " + one_class['type'] + ",  arguments: " 
                    + str(one_class['args']))
            header.append("#")        
            
    return header

def write_all_thresh_header(fd, analysis):
    """
    Writes header for all threshold results. This includes files, parameters
    and table (data) head.

    Table head should be formated the same way as data is formated in
    append_all_thresh_results().
    """

    # top of the header
    header = make_top_header()

    # rest of file names
    hi_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, 
        insert_root=sa_insert_root, reference=image_file_name, 
        suffix=conn_suffix)
    header.extend(common.format_file_info(
            name=hi_file_name, 
            description="Hierarchy image (out)"))
    pkl_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, suffix=sa_suffix,
        insert_root=sa_insert_root, reference=image_file_name)
    header.extend(common.format_file_info(
            name=pkl_file_name, 
            description="SegmentationAnalysis pickle (out)"))

    # parameters header 
    param_head = make_header(analysis=analysis)
    header.extend(param_head)

    # results table head for density and morphology
    tab_head = [
        ("# Threshold        Density               Volume "
         + "Surface  S/V  "),
        ("#          mean    std     min     max          "
         + "               ")]

    # topology
    if 'topology' in analysis.resultNames:
        tab_head[0] += "  Euler Loops Holes"
        tab_head[1] += "                   "

    # n segments
    tab_head[0] += " N segments"
    tab_head[1] += "           "

    # write
    for line in header + tab_head: 
        fd.write(line + os.linesep)
        
def append_all_thresh_results(analysis, thresh, fd=None):
    """
    Makes a line containing total results for one threshold. 

    If arg fd is given writes the line to the all thresholds results file.
    Otherwise returns the line.

    The results should be formated the same way as the table head is formated in
    write_all_thresh_header().

    Arguments:
      - analysis: analysis object
      - thresh: current threshold
      - fd: file descriptor
    """

    # don't write anything if no segments
    n_complexes = len(analysis.segments.ids)
    if n_complexes == 0:
        return

    # density and morphology
    dens = analysis.density
    mor = analysis.morphology
    out_vars = [thresh, dens.mean[0], dens.std[0], dens.min[0], dens.max[0], 
                mor.volume[0], mor.surface[0], 1.*mor.surface[0]/mor.volume[0]] 
    out_format = '%7.4f  %7.3f %6.3f %7.3f %7.3f %7u %7u %6.3f'

    # topology
    if 'topology' in analysis.resultNames:
        topo = analysis.topology
        out_vars += [topo.euler[0], topo.nLoops[0], topo.nHoles[0]]
        out_format += ' %5i %5i %5i'

    # n segments
    n_segments = len(analysis.labels.ids)
    out_vars += [n_segments]
    out_format += '   %5i   '

    # write or return line
    line = out_format % tuple(out_vars)
    if fd is not None:
        fd.write(line + os.linesep)
        fd.flush()
    else:
        return line

def write_all_thresh_bottom(fd, analysis):
    """
    Writes the bottom part of the all thrreshold results file
    """

    if res_include_total:

        dens = analysis.regionDensity
        
        # write total cleft results
        fd.write(os.linesep + "# Whole segmentation region:" + os.linesep)
        out_vars = (dens.mean, dens.std, dens.min, dens.max, dens.volume)
        out_format = '        %7.3f %6.3f %7.3f %7.3f %6u'
        cleft_line = out_format % out_vars
        fd.write('#' + cleft_line + os.linesep)
        
    fd.flush()

def order(analysis):
    """
    Returns ordered ids.
    """

    # current ids
    ids = analysis.labels.ids

    if order_by_boundaries:
        contacts = analysis.labels.contacts
        sort_list = contacts.orderSegmentsByContactedBoundaries(argsort=True)
    elif order_by_volume:
        volume = analysis.morphology.volume
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
    Under reconstruction

    Main function
    """

    # log machine name and architecture
    mach_name, mach_arch = common.machine_info()
    logging.info('Machine: ' + mach_name + ' ' + mach_arch)
    logging.info('Begin (script ' + __version__ + ')')


    ##########################################
    #
    # Segmentation
    #

    # read image
    image = common.read_image(file_name=image_file_name)

    # read boundaries and set inset
    if do_segmentation_flag:

        # read boundaries from an labels file
        bound, nested_boundary_ids = read_boundaries(
            check=check_boundaries, suggest_shape=image.data.shape)
        flat_boundary_ids = pyto.util.nested.flatten(nested_boundary_ids)
        bound_full_inset = bound.inset
        free = bound.makeFree(ids=flat_boundary_ids, size=free_size, 
                              mode=free_mode, mask=conn_region, update=False)
        free.makeInset(additional=bound, additionalIds=flat_boundary_ids)
        bound.useInset(inset=free.inset, mode='abs')

    else:

        # read pickle and extract boundaries
        if segmentation_file_name is not None:
            logging.info('Reading segmentation')
            sa_file_name = segmentation_file_name 
            sa = common.read_pickle(file_name=sa_file_name, 
                                    compact=['hierarchy.contacts'])
            sa.classifications = []
            bound = sa.boundary
            bound_full_inset = sa.hierarchy.full_inset
        else:
            raise ValueError("If do_segmentation_flag is False, " +
                             "segmentation_file_name has to be specified")

    # set image inset
    image.useInset(inset=bound.inset, mode='abs')

    # segmentation and analysis pickle file name
    sa_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, suffix=sa_suffix,
        insert_root=sa_insert_root, reference=image_file_name)

    if do_segmentation_flag:

        # do segmentation and analysis
        logging.info('Starting segmentation')
        sa = do_segmentation(
            image=image, bound=bound, bound_ids=flat_boundary_ids,
            sa_file_name=sa_file_name, inset=bound_full_inset)

    else:

        # read segmentation and analysis
        #logging.info('Reading segmentation')
        #if segmentation_file_name is not None:
        #    sa_file_name = segmentation_file_name 
        #sa = common.read_pickle(file_name=sa_file_name, 
        #                        compact=['hierarchy.contacts'])
        pass

    ######################################################
    # 
    # Classification and segment analysis
    #

    if do_classify_flag:
        
        logging.info('Starting classification')
        if do_segmentation_flag:
            ref_name = None
        else:
            ref_name = segmentation_file_name
        for sa_cls, sa_cls_name, sa_cls_file_name \
                in do_classification(sa=sa, image=image, inset=bound_full_inset,
                                     ref_name=ref_name):
            pass
        logging.info('End')

    return sa


# run if standalone
if __name__ == '__main__':
    main()
