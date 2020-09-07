#!/usr/bin/env python
"""
This script performs the following tasks:
    1) Makes cleft regions and analyzes their greyscale density. If they 
alreadyexist, cleft layers can be read. The following regions can be made 
(from 0 to all 3):
        a) Layers (parallel to the cleft)
        b) Columns (perpendicular to the cleft, each column is trans-cleft)
        c) Layers on columns
    2) Hierarchical connectivity segmentation of the cleft and the
connnector analysis
    3) Connector classification (optional)
    4) Makes layers on segments and analyzes their greyscale density 
(optional, requires layers).

The main applications of this script is to find and analyze connectors between 
giventwo relatively parallel boundaries, such as the membranes of a synaptic
cleft.

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

This script may be placed anywhere in the directory tree.

$Id$
Author: Vladan Lucic 
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from builtins import next
from builtins import zip
#from builtins import str
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"

import sys
import os
import os.path
import time
import platform
import pickle
import logging
from copy import copy, deepcopy

import numpy
import scipy

import pyto
import pyto.scripts.common as common
from pyto.segmentation.segment import Segment
from pyto.segmentation.cleft import Cleft
from pyto.scene.cleft_regions import CleftRegions
from pyto.scene.segmentation_analysis import SegmentationAnalysis

# import ../common/tomo_info.py
tomo_info = common.__import__(name='tomo_info', path='../common')

# to debug replace INFO by DEBUG
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')


##############################################################
#
# Parameters (please edit)
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
#image_file_name = "../3d/image.mrc"

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
# Note: The use of multiple label files is depreciated. Use add_segments.py
# to combine multiple label files into one.
#

# name of the labels file containing boundaries
if tomo_info is not None: labels_file_name = tomo_info.labels_file_name
#labels_file_name = "../viz/labels.raw"   

# labels file dimensions
labels_shape = (100, 120, 110)  # use these values
#labels_shape = None             # use image shape   

# labels file data type (e.g. 'int8', 'int16', 'int32', 'float16', 'float64') 
labels_data_type = 'uint8'

# labels file byteOrder ('<' for little-endian, '>' for big-endian)
labels_byte_order = '<'

# labels file array order ('F' for x-axis fastest, 'C' for z-axis fastest)
labels_array_order = 'F'

# offset of labels in respect to the data 
labels_offset = None             # no offset
#labels_offset = [10, 20, 30]    # offset specified

#####################################################################
#
# Cleft regions, general 
#
# Important: 
#   - The following ids should be >0
#   - Voxels at the tomogam side should not belong to any regions

# id of the first boundary (the postsynaptic terminal)
id_1 = 3

# id of the second boundary (the presynaptic terminal)
id_2 = 2

# id (or a list of ids) of the segmentation region between boundaries (cleft)
seg_id = 4

#####################################################################
#
# Cleft layers 
#

# if True layers are calculated, otherwise layes are read from a layers file 
do_layers_flag = True

# number of layers between the boundaries (on the segmetnation region)
n_layers = None  # detemined automatically, layers approximatly 1 voxel thick
# n_layers = 5   # 5 layers

# number of layers on each boundary, adjacent to the segmentation region
# Note: has to be at least 1
n_extra_layers = 3

# maximum distance (sum of the shortest distances to both boundaries) that a
# voxel is allowed to have in order to be included in this analysis
#max_distance = None  # no max distance, the whole segmentation region is used
max_distance = 15

# the way the distance between the boundaries is calculated 
# (mean/min/max/median)
distance_mode='median'

# number of layers that cover membranes (has to be <= n_extra_layers)
# Note: has to be at least 1
membrane_thickness = 2

# number of layers on each side of the segmentation region that are excluded
# from density calculation (and possibly some other) because their density 
# may not be right due to the CTF
cleft_exclude_thickness = 0

# if True the cleft (segmentation) and cleft boundary regions are adjusted to
# contain only elements that belong to layers
adjust_cleft_by_layers = True

#####################################################################
#
# Cleft columns
#

# if True columns are calculated, otherwise they are read from a file 
do_columns_flag = True

# id of the region around the clef region (rim region)
rim_id = 0

# coordinate system for cleft parametrization and making columns. Can be
# 'radial', 'polar', 'cartesian'.
column_system = 'radial'

# bins used to make columns
column_bins = [0, 0.25, 0.5, 0.75, 1]     # for radial, normalized
#column_bins = [[0,2,4], [-pi, 0, pi]]     # for polar, normalized
#column_bins = [[0, 0.25, 0.5, 1], [0, 0.25, 0.5, 1]] # for cartesian, norm.

# normalize column parametrization 
column_normalize = True

# column parametrization metric ('euclidean', 'geodesic')
column_metric = 'euclidean'

# column parametrization connectivity (needed if column_metric is 'geodesic')
column_connectivity = 1

#####################################################################
#
# Cleft layers on columns
#

# if True layers are analyzed on each of the columns separately
do_layers_on_columns_flag = True

#####################################################################
#
# Segmentation parameters
#

# if True do (calculate) segmentation 
do_segmentation_flag = True

# if True and do_segmentation_flag=False, read segmentation from a pickle
read_segmentation_flag = True

# threshold list 
threshold = numpy.arange(-0.12, -0.041, 0.01)  

# number of boundaries that segments contacts
n_boundary = 2

# 'exact' to require that segments contact exactly n_boundary boundaries, or
# 'at_least' to alow n_boundary or more contacted boundaries 
bound_count = 'exact'

# Note: Next three parameters are experimental and should not be changed

# connectivity of the structuring element used for segment determination
struct_el_conn = 1 

# connectivity of the structuring element used to detect contacts
contact_struct_el_conn = 1

# connectivity of the structuring element used to count contacts
count_struct_el_conn = 1

#####################################################################
#
# Classification parameters
#
# Classifications can be executed in any order, the way they are numbered 
# determins the order. For each classification variable class_i_type (i>=1) 
# has to be defined as well as variables that hold classification parameters,
# as follows:
#   class_i_type = 'keep':
#     - class_i_mode: 'new' for smallest or 'new_branch_tops' for biggest
#       segments before merging
#     - class_i_name: name of the class
#   class_i_type = 'volume':
#     - class_i_volumes: list of volumes
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
class_2_type = 'volume'
class_2_volumes = [0, 100, 1000]
class_2_names = ['small', 'big']

#####################################################################
#
# Analysis parameters
#

# segment length calculation mode:
#   - (one boundary) 'b-max' or 'c-max': between contacts and the most distant
#     element of a segment, where contacts belong to a boundary or a segment,
#     respectivly  
#   - (two boundaries) 'b2b' and 'c2c: minimim distance that connect a contact 
#     with one boundary, element of a segment on the central layer between the
#     boundaries, and a contact with the other boundary, where contacts belong 
#     to a boundary or a segment, respectivly
length_contact_mode = 'c2b'

# segment length line mode:
#   - 'straight': straight-line distance between contacts (two boundaries) or
#     between a contact and an end point
#   - 'mid': sum of distances between contact points and a mid point (two 
#     boundaries only)
length_line_mode = 'mid'

# unit cleft area, used for number of segments per unit cleft area
unit_cleft_area = 1000

#####################################################################
#
# Layers on segments parameters
#

# flag indicating if classifications is done
do_layers_on_segments_flag = True


##############################################################
#
# Output file names (can be left unchaged, but ok to edit)
#
##############################################################

###########################################################
#
# Cleft layers label (data) file. The file name is formed as:
#
#   <lay_directory>/<lay_prefix> + image root + <lay_suffix>
#

# write layers flag
write_layers_flag = True

# layers directory
lay_directory = ''

# layers file name prefix (no directory name)
lay_prefix = ''

# include image file root (filename without directory and extension)
lay_insert_root = True

# layers file name suffix
lay_suffix = "_layers.mrc"

# layers data type, 'uint8' , or
lay_data_type = 'uint8'        # if max segment id is not bigger than 255
#lay_data_type = 'int16'         # more than 255 segments

############################################################
#
# Cleft layers results file. The file name is formed as:
#
#   <lay_res_directory>/<lay_res_prefix> + image root + <lay_res_suffix>
#

# directory
lay_res_directory = ''

# file name prefix (no directory name)
lay_res_prefix = ''

# include image file root (filename without directory and extension)
lay_res_insert_root = True

# file name suffix
lay_res_suffix = "_layers.dat"

# include total values (for all layers taken together) in the results, id 0
lay_res_include_total = True

############################################################
#
# Writing cleft layers (CleftRegions) object. The file name is formed as:
#
#   <lay_pkl_directory>/<lay_pkl_prefix> + image root + <lay_pkl_suffix>
#

# directory
lay_pkl_directory = ''

# file name prefix (no directory name)
lay_pkl_prefix = ''

# include image file root (filename without directory and extension)
lay_pkl_insert_root = True

# file name suffix
lay_pkl_suffix = "_layers.pkl"

###########################################################
#
# Thin cleft layers label (data) file. The file name is formed as:
#
#   <thin_lay_directory>/<thin_lay_prefix> + image root + <thin_lay_suffix>
#

# write thin layers flag
write_thin_layers_flag = True

# thin layers directory
thin_lay_directory = ''

# thin layers file name prefix (no directory name)
thin_lay_prefix = ''

# include image file root (filename without directory and extension)
thin_lay_insert_root = True

# thin layers file name suffix
thin_lay_suffix = "_layers-thin.mrc"

# thin layers data type, 'uint8' , or
thin_lay_data_type = 'uint8'        # if max segment id is not bigger than 255
#thin_lay_data_type = 'int16'         # more than 255 segments

############################################################
#
# Thin cleft layers results file. The file name is formed as:
#
#   <thin_lay_res_directory>/<thin_lay_res_prefix> + image root 
#             + <thin_lay_res_suffix>
#

# directory
thin_lay_res_directory = ''

# file name prefix (no directory name)
thin_lay_res_prefix = ''

# include image file root (filename without directory and extension)
thin_lay_res_insert_root = True

# file name suffix
thin_lay_res_suffix = "_layers-thin.dat"

# include total values (for all layers taken together) in the results, id 0
thin_lay_res_include_total = True

############################################################
#
# Writing thin cleft layers (CleftRegions) object. The file name is formed as:
#
#   <thin_lay_pkl_directory>/<thin_lay_pkl_prefix> + image root 
#               + <thin_lay_pkl_suffix>
#

# directory
thin_lay_pkl_directory = ''

# file name prefix (no directory name)
thin_lay_pkl_prefix = ''

# include image file root (filename without directory and extension)
thin_lay_pkl_insert_root = True

# file name suffix
thin_lay_pkl_suffix = "_layers-thin.pkl"

###########################################################
#
# Cleft columns label (data) file. The file name is formed as:
#
#   <col_directory>/<col_prefix> + image root + <col_suffix>
#

# write columns flag
write_columns_flag = True

# columns directory
col_directory = ''

# columns file name prefix (no directory name)
col_prefix = ''

# include image file root (filename without directory and extension)
col_insert_root = True

# columns file name suffix
col_suffix = "_columns.mrc"

# columns data type, 'uint8' , or
col_data_type = 'uint8'        # if max segment id is not bigger than 255
#col_data_type = 'int16'         # more than 255 segments

############################################################
#
# Cleft columns results file. The file name is formed as:
#
#   <col_res_directory>/<col_res_prefix> + image root + <col_res_suffix>
#

# directory
col_res_directory = ''

# file name prefix (no directory name)
col_res_prefix = ''

# include image file root (filename without directory and extension)
col_res_insert_root = True

# file name suffix
col_res_suffix = "_columns.dat"

# include total values (for all layers taken together) in the results, id 0
col_res_include_total = True

############################################################
#
# Cleft columns (CleftRegions) object. The file name is formed as:
#
#   <col_pkl_directory>/<col_pkl_prefix> + image root + <col_pkl_suffix>
#

# directory
col_pkl_directory = ''

# file name prefix (no directory name)
col_pkl_prefix = ''

# include image file root (filename without directory and extension)
col_pkl_insert_root = True

# file name suffix
col_pkl_suffix = "_columns.pkl"

###########################################################
#
# Cleft layers on column label (data) file. The file name is formed as:
#
#   <lay_col_directory>/<lay_col_prefix> + image root 
#   + <lay_col_suffix> + column_id + <lay_col_suffix_2>
#

# write layers flag
write_layers_on_columns_flag = True

# layers directory
lay_col_directory = ''

# layers file name prefix (no directory name)
lay_col_prefix = ''

# include image file root (filename without directory and extension)
lay_col_insert_root = True

# layers file name suffix
lay_col_suffix = "_layers-4_on_column-"
lay_col_suffix_2 = "-of-4.mrc"

# layers data type, 'uint8' , or
lay_col_data_type = 'uint8'        # if max segment id is not bigger than 255
#lay_col_data_type = 'int16'         # more than 255 segments

############################################################
#
# Cleft layers on columns results file. The file name is formed as:
#
#   <lay_col_res_directory>/<lay_col_res_prefix> + image root 
#   + <lay_col_res_suffix> + column_id + <lay_col_res_suffix_2>
#

# directory
lay_col_res_directory = ''

# file name prefix (no directory name)
lay_col_res_prefix = ''

# include image file root (filename without directory and extension)
lay_col_res_insert_root = True

# file name suffix 
lay_col_res_suffix = "_layers-4_on_column-"
lay_col_res_suffix_2 = "-of-4.dat"

# include total values (for all layers taken together) in the results, id 0
lay_col_res_include_total = True

############################################################
#
# Writing cleft layers on columns (CleftRegions) object. The file name is 
# formed as:
#
#   <lay_col_pkl_directory>/<lay_col_pkl_prefix> + image root 
#   + <lay_col_pkl_suffix> + column_id + <lay_col_pkl_suffix_2>
#

# directory
lay_col_pkl_directory = ''

# file name prefix (no directory name)
lay_col_pkl_prefix = ''

# include image file root (filename without directory and extension)
lay_col_pkl_insert_root = True

# file name suffix 
lay_col_pkl_suffix = "_layers-4_on_column-"
lay_col_pkl_suffix_2 = "-of-4.pkl"

###########################################################
#
# Segments (connections) files. The file name is formed as:
#
#   <conn_directory>/<conn_prefix> + image root + <threshold_label> 
#       + <threshold> + <conn_suffix>
#
# Note: Final connections file is always written

# write connections flag
write_connections_flag = True

# connections directory
conn_directory = ''

# connections file name prefix (no directory name)
conn_prefix = ''

# include image file root (filename without directory and extension)
conn_insert_root = True

# connections file name suffix
conn_suffix = ".mrc"

# connections data type, 'uint8' , or
#conn_data_type = 'uint8'        # if max segment id is not bigger than 255
conn_data_type = 'int16'         # more than 255 segments

# controls what kind of data casting may occur: 'no', 'equiv', 'safe', 
# 'same_kind', 'unsafe'. Identical to numpy.astype()
# Warning: Unless using 'safe', check results to see if label ids are too high 
# for conn_data_type. However, 'safe' will most likely fail for conn_data_type
# that have less than 32 or 64 bits
# Note: used also for writing layers and columns
conn_casting = 'unsafe'

############################################################
#
# Segmentation and analysis results and pickles. The results file name is:
#
#   <res_directory>/<res_prefix> + image root + <threshold_label> 
#       + <threshold> + <res_suffix>
#
# while the pickle file name is:
#
#   <res_directory>/<res_prefix> + image root + <threshold_label> 
#       + <threshold> + <sa_suffix>
#

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

# include total values (for all connections taken together) in the results, id 0
res_include_total = True

# threshold label 
threshold_label = '_thr-'

# threshold format used for forming file names and reporting
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

###########################################################
#
# Layers on segments (connections) files
#
#
# Label (data) file:
#
#   <lay_directory>/<lay_prefix> + segmentation_and_analysis_root + <lay_suffix>
#
# Results file:
#
#   <lay_res_directory>/<lay_res_prefix> + segmentation_and_analysis_root 
#     + <lay_res_suffix>
#
# Pickle:
#
#   <lay_pkl_directory>/<lay_pkl_prefix> + segmentation_and_analysis_root 
#     + <lay_pkl_suffix>
#
# where:
#  - segmentation_and_analysis_root is the segmetnation and analysis file 
#  name without the directory and extension parts


################################################################
#
# Work (edit only if you know what you're doing)
#
################################################################

######################################################
#
# Main components
#

def do_regions(inset, mode, reference, regions=None, image=None, bound=None, 
               layers=None, mask=None, in_suffix=None):
    """
    Makes and analyzes regions (such as layers or columns) on a cleft and 
    optionally on boundaries.

    Regions can be specified in arg regions (not modified in this function). If
    arg regions is not given, regions are calculated using args image and 
    bounds. 

    If mask is given, it is used to restrict cleft (bot not boundary) regions 
    to elements that are present (>0) in mask.data.

    Arguments:
      - image: grey scale image
      - bound: boundaries
      - inset: regions are written with this inset
      - reference: reference file name
      - regions: (CleftRegions) cleft regions
      - layers: (CleftRegions) layers used to determine region centers (only
      in 'columns' mode)
      - mask: (Labels) segments used to mask cleft regions
      - in_suffix: (str) added between suffix and suffix_2 in layers_on_columns
      more, in order to distinguish file names related to different columns
    
    Returns: (CleftRegions) cleft regions, the most important attributes being:
      - regions: (segmentation.Segment) regions
      - regionsDensity: (segmentation.Density) regions density
    """

    if regions is None:

        # make regions
        cleft = Cleft(data=bound.data, cleftId=seg_id, bound1Id=id_1, 
                      bound2Id=id_2, copy=False)
        cleft.inset = bound.inset
        cleft_reg = CleftRegions(image=image, cleft=cleft)

        if mode == 'layers':
            cleft_reg.makeLayers(
                nLayers=n_layers, widthMode=distance_mode, fill=True,
                nBoundLayers=n_extra_layers, maxDistance=max_distance,
                adjust=adjust_cleft_by_layers, refine=True)
        elif mode == 'thin_layers':
            cleft_reg.makeLayers(
                nLayers=None, widthMode=distance_mode, fill=True,
                nBoundLayers=n_extra_layers, maxDistance=max_distance,
                adjust=False)
        elif mode =='columns':
            cleft_reg = layers.makeColumns(
                bins=column_bins, ids=layers.cleftLayerIds, 
                system=column_system, normalize=column_normalize, 
                originMode='one', startId=None, metric=column_metric, 
                connectivity=column_connectivity, rimId=rim_id)

    else:

        # deepcopy regions and remove density
        cleft_reg = deepcopy(regions)
        cleft_reg.regionsDensity = None

    # restrict cleft regions if needed
    if mask is not None:
        cleft_reg.regions.restrict(
            mask=mask, ids=cleft_reg.getCleftLayerIds(), update=True)

    # define groups
    if ((mode == 'layers') or (mode == 'thin_layers') 
        or (mode == 'layers_on_columns')):
        groups = {
            'cleft' : cleft_reg.getCleftLayerIds(),
            'cleft_ex' : cleft_reg.getCleftLayerIds(
                exclude=cleft_exclude_thickness),
            'bound_1' : cleft_reg.getBound1LayerIds(thick=membrane_thickness),
            'bound_2' : cleft_reg.getBound2LayerIds(thick=membrane_thickness),
            'all' : cleft_reg.getLayerIds()}
    else:
        groups = None

    # analyze
    if ((mode == 'layers') or (mode == 'thin_layers') or 
        (mode == 'layers_on_columns')):
        analysis_mode = 'layers'
    else:
        analysis_mode = mode
    cleft_reg.findDensity(
        regions=cleft_reg.regions, mode=analysis_mode, groups=groups, 
        boundThick=membrane_thickness, exclude=cleft_exclude_thickness)

    logging.info('  Writing ' + mode + ' formation and analysis results')

    # set write parameters
    if mode == 'layers':
        directory = lay_directory
        prefix = lay_prefix
        suffix = lay_suffix
        insert_root = lay_insert_root
        data_type = lay_data_type
        res_directory = lay_res_directory
        res_prefix = lay_res_prefix
        res_insert_root = lay_res_insert_root
        res_suffix = lay_res_suffix
        pkl_directory = lay_pkl_directory
        pkl_prefix = lay_pkl_prefix
        pkl_insert_root = lay_pkl_insert_root
        pkl_suffix = lay_pkl_suffix
    elif mode == 'thin_layers':
        directory = thin_lay_directory
        prefix = thin_lay_prefix
        suffix = thin_lay_suffix
        insert_root = thin_lay_insert_root
        data_type = thin_lay_data_type
        res_directory = thin_lay_res_directory
        res_prefix = thin_lay_res_prefix
        res_insert_root = thin_lay_res_insert_root
        res_suffix = thin_lay_res_suffix
        pkl_directory = thin_lay_pkl_directory
        pkl_prefix = thin_lay_pkl_prefix
        pkl_insert_root = thin_lay_pkl_insert_root
        pkl_suffix = thin_lay_pkl_suffix
    elif mode == 'columns':
        directory = col_directory
        prefix = col_prefix
        suffix = col_suffix
        insert_root = col_insert_root
        data_type = col_data_type
        res_directory = col_res_directory
        res_prefix = col_res_prefix
        res_insert_root = col_res_insert_root
        res_suffix = col_res_suffix
        pkl_directory = col_pkl_directory
        pkl_prefix = col_pkl_prefix
        pkl_insert_root = col_pkl_insert_root
        pkl_suffix = col_pkl_suffix
    elif mode == 'layers_on_columns':
        directory = lay_col_directory
        prefix = lay_col_prefix
        suffix = lay_col_suffix + in_suffix + lay_col_suffix_2
        insert_root = lay_col_insert_root
        data_type = lay_col_data_type
        res_directory = lay_col_res_directory
        res_prefix = lay_col_res_prefix
        res_insert_root = lay_col_res_insert_root
        res_suffix = lay_col_res_suffix + in_suffix + lay_col_res_suffix_2
        pkl_directory = lay_col_pkl_directory
        pkl_prefix = lay_col_pkl_prefix
        pkl_insert_root = lay_col_pkl_insert_root
        pkl_suffix = lay_col_pkl_suffix + in_suffix + lay_col_pkl_suffix_2

    # write regions
    if (write_layers_flag or write_thin_layers_flag or write_columns_flag):
        reg_file_name = common.make_file_name(
            directory=directory, prefix=prefix, suffix=suffix,
            insert_root=insert_root, reference=reference)
        common.write_labels(
            labels=cleft_reg.regions, name=reg_file_name, 
            data_type=data_type, inset=inset, casting=conn_casting)

    # pickle cleft regions
    pkl_file_name =  common.make_file_name(
        directory=pkl_directory, prefix=pkl_prefix, suffix=pkl_suffix,
        insert_root=pkl_insert_root, reference=reference)
    common.write_pickle(obj=cleft_reg, file_name=pkl_file_name, 
                        image=['regions'])

    # write layer results
    res_file_name =  common.make_file_name(
        directory=res_directory, prefix=res_prefix, suffix=res_suffix,
        insert_root=res_insert_root, reference=reference)
    write_cleft_results(
        cleft_regions=cleft_reg, reference=reference, mode=mode,
        regions_name=reg_file_name, pickle_name=pkl_file_name, 
        results_name=res_file_name)

    return cleft_reg

def intersect_layers_columns(layers, columns):
    """
    Iterator that makes layers on each column and yields the corresponding 
    pyto.scene.CleftRegion object.

    Simply intersects layers with each of the columns. Respects inset attribute.

    Arguments:
      - layers: (CleftRegions) cleft layers
      - columns: (CleftRegions) cleft columns

    Yields: 
      - (CleftRegions) layers on a column. The following attributes are set:
        - regions
        - regionsDensity
        - nLayers
        - nBoundLayers
      - (str) column_id
    """

    # keep layers for each column
    for col_id in columns.regions.ids:

        # current column mask
        mask_data = (columns.regions.data == col_id) * 1
        mask = Segment(data=mask_data, ids=[1])
        mask.inset = columns.regions.inset

        # intersect with layers
        layers_on_col_data = layers.regions.restrict(mask=mask, update=False)

        # make new instance
        layers_on_col = CleftRegions(image=layers.image, cleft=layers.cleft)
        layers_on_col.regions = deepcopy(layers.regions)
        layers_on_col.regions.data = layers_on_col_data

        # copy most important attributes
        layers_on_col.nLayers = layers.nLayers
        layers_on_col.nBoundLayers = layers.nBoundLayers

        yield layers_on_col, str(col_id) 

def do_segmentation(image, bound, cleft_lay, sa_file_name, inset):
    """
    Makes hierarchical segmentation
    """

    # open already existing summary file or write header

    # make threshold list
    if isinstance(threshold, (numpy.ndarray, list)):
        thresh = threshold
    else:
        thresh = [threshold]

    # prepare for segmentation
    sa = SegmentationAnalysis(image=image, boundary=bound)
    sa.setSegmentationParameters(
        nBoundary=n_boundary, boundCount=bound_count, boundaryIds=[id_1, id_2],
        mask=seg_id, structElConn=struct_el_conn, 
        contactStructElConn=contact_struct_el_conn, 
        countStructElConn=count_struct_el_conn)
    sa.setAnalysisParameters(
        lengthContact=length_contact_mode, lengthLine=length_line_mode,
        cleftLayers=cleft_lay)
    tc_iter = sa.tcSegmentAnalyze(
        thresh=thresh, order='<', count=True, doDensity=True, 
        doMorphology=True, doLength=True, doTopology=True, doCleftContacts=True)
    
    # segment and analysis
    thresh_lines = []
    for tr in thresh:
        
        tr_str, tr_long_str = common.format_param(
            value=tr, name=threshold_label, format='%6.3f')
        logging.info('Starting segmentation and analysis for threshold ' 
                     + tr_str)

        # segment at this threshold
        sa_level, level, curr_thresh = next(tc_iter)
    
        # write connections
        if write_connections_flag and (len(sa_level.segments.ids) > 0):
            conn_file_name = common.make_file_name(
                directory=conn_directory, prefix=conn_prefix,
                insert_root=conn_insert_root, reference=image_file_name,
                param_name=threshold_label, param_value=tr, 
                param_format='%6.3f', suffix=conn_suffix)
            common.write_labels(
                labels=sa_level.segments, name=conn_file_name,
                data_type=conn_data_type, inset=inset, casting=conn_casting)

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
            write_results(analysis=sa_level, thresh=tr) 

        # append to all threshold results file
        one_line = append_all_thresh_results(analysis=sa_level, thresh=tr) 
        if one_line is not None:
            thresh_lines.append(one_line)

    # hierarchy
    tc = tc_iter.send(True) 
    sa.hierarchy = tc

    logging.info('Writing segmentation and analysis pickle and result files')

    # pickle
    common.write_pickle(obj=sa, file_name=sa_file_name, image=['hierarchy'],
                        compact=['hierarchy.contacts'])

    # open all thresholds file and write header
    all_thresh_file = open_results()
    write_all_thresh_header(fd=all_thresh_file, cleft_layers=cleft_lay, 
                            analysis=sa)

    # write individual threshold lines
    for line in thresh_lines:
        all_thresh_file.write(line + os.linesep)

    # write bottom for the all thresholds
    write_all_thresh_bottom(fd=all_thresh_file, analysis=sa)

    return sa

def do_classification(sa, inset, cleft_lay):
    """
    Iterator that classifies hierarchical segments

    Arguments:
      - sa: (Hierarchy) hierarchical segmentation
      - inset: connections image is adjusted to this inset before it's written
      - cleft_lay: (CleftRegions) cleft regions object

    Yelds (for each class):
      - cls: (SegmentationAnalysis) segmentation and analysis for one 
      class (classification result)
      - cls_name: class name
      - cls_file_name: pickle file name
    """

    setClassifications(analysis=sa)
    if sa.classifications is not None:
        for cls, cls_name in sa.classifyAnalyze(
            hierarchy=sa.hierarchy, doDensity=False, doMorphology=False, 
            doLength=False, doTopology=False):

            logging.info('  Writing segmentation and analysis pickle and ' 
                         + 'result files for class ' + cls_name + '.')

            # pickle
            suffix = cls_name + sa_suffix
            cls_file_name = common.make_file_name(
                directory=sa_directory, prefix=sa_prefix, suffix=suffix,
                insert_root=sa_insert_root, reference=image_file_name)
            common.write_pickle(obj=cls, file_name=cls_file_name, 
                      image=['hierarchy'], compact=['hierarchy.contacts'])

            # connections (labels)
            suffix = cls_name + conn_suffix
            conn_file_name = common.make_file_name(
                directory=sa_directory, prefix=sa_prefix, suffix=suffix,
                insert_root=sa_insert_root, reference=image_file_name)
            common.write_labels(
                labels=sa.labels, name=conn_file_name, data_type=conn_data_type,
                inset=inset, casting=conn_casting)

            # write results            
            write_results(analysis=cls, name=cls_name, cleft_layers=cleft_lay) 

            yield cls, cls_name, cls_file_name

######################################################
#
# Analysis
#

def setClassifications(analysis):
    """
    Parses input variables and sets classification parameters
    """

    for ind in range(1,100):

        # finish if no class type
        # Note: sys.modules[__name__].locals() fails with iPython 0.10
        class_type = globals().get('class_' + str(ind) + '_type')
        if class_type is None:
            break

        # parse classification parameters
        prefix = 'class_' + str(ind) + '_'
        args = {}
        for name, value in list(globals().items()):
            if name.startswith(prefix):
                class_arg_name = name.lstrip(prefix)
                if class_arg_name != 'type':
                    args[class_arg_name]  = value

        # set parameters
        analysis.addClassificationParameters(type=class_type, args=args)
    

###########################################
#
# Reading or modifying image, boundary or related files
#

def read_image():
    """
    Reads image file and returns an segmentation.Image object
    """
    return pyto.segmentation.Grey.read(file=image_file_name)

def read_boundaries(shape=None, segments=None):
    """
    Reads labels file, makes (Segment) boundaries and makes inset.

    Arguments:
      - shape: shape of the boundaries
    """

    #global seg_id

    # if shape is not given and the file is raw, use arg shape
    if labels_shape is None:
        bound_file = pyto.io.ImageIO()
        bound_file.setFileFormat(file_=labels_file_name)
        if bound_file.fileFormat == 'raw':
            local_labels_shape = shape
        else:
            local_labels_shape = labels_shape
    else:
        local_labels_shape = labels_shape

    # read labels file and make a Segment object
    all_ids = pyto.util.nested.flatten([id_1, id_2, seg_id])
    bound = pyto.segmentation.Segment.read(
        file=labels_file_name, ids=all_ids, clean=True, 
        byteOrder=labels_byte_order, dataType=labels_data_type,
        arrayOrder=labels_array_order, shape=local_labels_shape)

    # positioning 
    if labels_offset is not None:
        bound.inset = [slice(offset, offset+size) for offset, size 
                       in zip(labels_offset, bound.shape)]
        
    return bound

###########################################
#
# Write output files
#

def open_results(thresh=None, name=''):
    """
    Opens a results file name and returns it.

    Arguments:
      - name: inserted before suffix in the results file name, usually a 
      class name
    """
    
    if thresh is None:
        thresh_name = ''
    else:
        thresh_name = threshold_label
    suffix = name + res_suffix

    # make results file name
    res_file_name = common.make_file_name(
        directory=res_directory, prefix=res_prefix, 
        insert_root=res_insert_root, reference=image_file_name, 
        param_value=thresh, param_name=thresh_name, 
        param_format=threshold_format, suffix=suffix)

    # open file
    res_file = open(res_file_name, 'w')
    return res_file

def write_cleft_results(cleft_regions, reference, mode, regions_name, 
                        pickle_name, results_name=None):
    """
    Writes cleft geometry and layer results (including density profile)

    Arguments:
      - cleft_regions: (CleftRegions) cleft regions object
      - reference: reference file name
      - mode: cleft regions mode, 'layers', 'thin_layers' or 'co;umns'
      - regions_name: name of the file where regions are saved
      - pickle_name: pickle file name
      - results_name: results file name; if None generated by this function
      but only for 'layers', 'thin_layers' and 'columns' modes
    """

    # set write parameters
    if (mode == 'layers'):
        res_directory = lay_res_directory
        res_prefix = lay_res_prefix
        res_insert_root = lay_res_insert_root
        res_suffix = lay_res_suffix
        res_include_total = lay_res_include_total
        regions_description = "Layers image (out)"
    elif (mode == 'thin_layers'):
        res_directory = thin_lay_res_directory
        res_prefix = thin_lay_res_prefix
        res_insert_root = thin_lay_res_insert_root
        res_suffix = thin_lay_res_suffix
        res_include_total = thin_lay_res_include_total
        regions_description = "Thin layers image (out)"
    elif (mode == 'columns'):
        res_directory = col_res_directory
        res_prefix = col_res_prefix
        res_insert_root = col_res_insert_root
        res_suffix = col_res_suffix
        res_include_total = col_res_include_total
        regions_description = "Columns image (out)"
    elif (mode == 'layers_on_columns'):
        res_include_total = lay_col_res_include_total
        regions_description = "Layers on columns image (out)"

    # open region results file
    if results_name is None:
        results_name = common.make_file_name(
            directory=res_directory, prefix=res_prefix, suffix=res_suffix,
            insert_root=res_insert_root, reference=reference)
    fd = open(results_name, 'w')

    # top of the header
    header = make_top_header()

    # rest of file names
    header.extend(common.format_file_info(
            name=pickle_name, description="CleftLayers pickle (out)"))
    header.extend(common.format_file_info(
            name=regions_name, description=regions_description))

    # variables
    ids = cleft_regions.regions.ids

    # parameters and results 
    if ((mode == 'layers') or (mode == 'thin_layers') or 
        (mode == 'layers_on_columns')):

        # parameter lines for layers
        header.extend([
            "#",
            "# Boundary ids:",
            "#     - presynaptic: " + str(id_2),
            "#     - postsynaptic: " + str(id_1),
            "#     - cleft: " + str(seg_id),
            "#",
            "# Layer ids:",
            "#     - on the postsynaptic terminal: " + "1 - " \
                + str(n_extra_layers),
            "#     - in the synaptic cleft: " + str(n_extra_layers+1) + " - " \
                  + str(len(ids) - n_extra_layers),
            ("#     - on the presynaptic terminal: " 
             + str(len(ids) - n_extra_layers + 1) + " - " + str(len(ids))),
            "#",
            "# Cleft edges:",
            "#      - max distance to membranes: " + str(max_distance),
            "#      - cleft adjusted by layers: "\
                + str(adjust_cleft_by_layers),
            "#",        
            "#",
            "# Results:"])

    if (mode == 'layers') or (mode == 'thin_layers'):

        # result lines for layers
        header.extend([
            "#",
            ("#   Cleft thickness (edge to edge): %7.1f,  mode %s" 
             % (cleft_regions.width, distance_mode)),
            "#",
            "#   Cleft direction (from pre- to post- side):",
            ("#     - theta: %6.1f deg" 
             % cleft_regions.widthVector.getTheta('deg')),
            "#     - phi: %6.1f deg" % cleft_regions.widthVector.getPhi('deg'),
            "#",
            "#   Density dip in the cleft:",
            ("#     - layer id: " + str(cleft_regions.minCleftDensityId) 
             + ", fractional position: "
             + str(cleft_regions.minCleftDensityPosition)),
            "#     - relative density: %7.2f" \
                % cleft_regions.relativeMinCleftDensity])

    elif mode == 'columns':

        # parameters for columns
        header.extend(
            ['#',
             '# Parameters:',
             "#"])
        header.extend(['#\t' + line for line 
                       in cleft_regions.getParameterStrings()])
        header.extend(
            ['#',
             "# Ids and bins",
             "#"])
        header.extend(['#\t' + line for line 
                       in cleft_regions.getBinIdsStrings()])

    # write header
    header.extend(2 * ['#'])
    for line in header: 
        fd.write(line + os.linesep)

    # write results table head
    tab_head = ["# Id             Density               Volume  ",
               "#        mean    std     min     max" ]
    for line in tab_head: 
        fd.write(line + os.linesep)

    # write the region results
    dens = cleft_regions.regionDensity
    out_vars = [dens.mean, dens.std, dens.min, dens.max, dens.volume]
    out_format = '%3u   %7.3f %6.3f %7.3f %7.3f %6u'
    res_table = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                         indices=ids, prependIndex=True)
    for line in res_table:
        fd.write(' ' + line + os.linesep)

    # additional for layers
    if (mode == 'layers') or (mode == 'thin_layers'):

        # write total membrane densities
        fd.write(os.linesep + '# Membranes: ' + os.linesep)
        ids = [cleft_regions.getBound1LayerIds(thick=membrane_thickness), 
               cleft_regions.getBound2LayerIds(thick=membrane_thickness)]
        keys = ['bound_1', 'bound_2']
        for g_ids, g_key in zip(ids, keys):
            g_dens = cleft_regions.groupDensity[g_key]
            mem_vars = (g_ids[0], g_ids[-1], g_dens.mean, g_dens.std, 
                        g_dens.min, g_dens.max, g_dens.volume)
            mem_format = '#%2d-%2d %7.3f %6.3f %7.3f %7.3f %6u'
            fd.write((mem_format % mem_vars) + os.linesep)

        # write total cleft densities
        fd.write(os.linesep + '# Cleft with and without excluded layers: ' 
                 + os.linesep)
        ids = [cleft_regions.getCleftLayerIds(), 
               cleft_regions.getCleftLayerIds(exclude=cleft_exclude_thickness)]
        keys = ['cleft', 'cleft_ex']
        for g_ids, g_key in zip(ids, keys):
            g_dens = cleft_regions.groupDensity[g_key]
            cleft_vars = (g_ids[0], g_ids[-1], g_dens.mean, g_dens.std, 
                        g_dens.min, g_dens.max, g_dens.volume)
            cleft_format = '#%2d-%2d %7.3f %6.3f %7.3f %7.3f %6u'
            fd.write((cleft_format % cleft_vars) + os.linesep)

    # total results
    if res_include_total:
        fd.write(os.linesep + "# All regions together:" + os.linesep)
        try:
            tot_line = pyto.io.util.arrayFormat(
                arrays=out_vars, format=out_format, 
                indices=[0], prependIndex=True)
            fd.write('#' + tot_line[0] + os.linesep)
        except TypeError:
            logging.info("Could not write total results.")

    fd.flush()

def write_results(analysis, cleft_layers=None, thresh=None, name=''):
    """
    Writes segmentation and analysis results for a single threshold

    Arguments:
      - analysis: segmentation analysis, need to contain either hierarchy
      (hierarchical segmentation) or segments (flat segmentation) attribute
      - n_contact
      - complex_density
      - cleft_layers: (CleftRegions) cleft layer ananlysis
      - threshold: threshold
      - name: inserted before suffix in the results file name, usually a
      class name
    """

    # open results file
    fd = open_results(thresh=thresh, name=name)

    # top of the header
    header = make_top_header()

    # rest of file names
    suffix = name + sa_suffix
    pkl_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, suffix=suffix,
        insert_root=sa_insert_root, reference=image_file_name,
        param_value=thresh, param_name=threshold_label,
        param_format=threshold_format)
    header.extend(common.format_file_info(
            name=pkl_file_name, 
            description="SegmentationAnalysis pickle (out)"))
    segments_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, suffix=name+conn_suffix,
        insert_root=sa_insert_root, reference=image_file_name)
    header.extend(common.format_file_info(
            name=segments_file_name, 
            description="Segmented image (out)"))

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
    ids = labels.ids
    dens = analysis.density
    mor = analysis.morphology

    # parameters header 
    param_header = make_header(analysis=analysis, cleft_layers=cleft_layers,
                               thresh=thresh, name=name)
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
    if mode == 'hierarchy':
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
    out_vars += [dens.mean, dens.std, dens.min, dens.max, 
                 mor.volume, mor.surface, mor.surface / volume]
    out_format += ' %7.3f %6.3f %7.3f %7.3f %6u %6u %6.3f'

    # length
    if ('morphology' in analysis.resultNames) and (mor.length is not None):
        tab_head[0] += " Length"
        tab_head[1] += "       "
        out_vars += [mor.length]
        out_format += " %6.1f"

    # topology
    topo = analysis.topology
    if 'topology' in analysis.resultNames:
        tab_head[0] += "  Euler Loops"
        tab_head[1] += "             "
        out_vars += [topo.euler, topo.nLoops]
        out_format += ' %5i %5i'

    # contacts
    tab_head[0] += "  Contacts "
    tab_head[1] += "  post pre "
    out_vars += [analysis.nContacts[1], analysis.nContacts[2] ]
    out_format += ' %4u  %4u'

    # write the table head and the results
    for line in tab_head: 
        fd.write(line + os.linesep)
    res_table = pyto.io.util.arrayFormat(arrays=out_vars, format=out_format,
                                         indices=ids, prependIndex=True)
    for line in res_table:
        fd.write(line + os.linesep)
        
    # write summary results
    if res_include_total:

        # write results for all connections together
        if (mode == 'segments') or (mode == 'hierarchy'):
            fd.write(os.linesep + "# All segments together:" + os.linesep)
            try:
                tot_line = pyto.io.util.arrayFormat(arrays=out_vars, 
                                                format=out_format,
                                                indices=[0], prependIndex=True)
                fd.write('#' + tot_line[0] + os.linesep)
            except TypeError:
                print(("Could not write all segments results for threshold ", 
                      tr_str, ", continuing to the next threshold.")) 

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
            fd.write(os.linesep + "# Whole cleft:" + os.linesep)
            out_vars = (reg_dens.mean, reg_dens.std, reg_dens.min, 
                        reg_dens.max, reg_dens.volume)
            out_format = '    %7.3f %6.3f %7.3f %7.3f %6u'
            if mode == 'hierarchy':
                out_format = '       ' + out_format
            cleft_line = out_format % out_vars
            fd.write('#' + cleft_line + os.linesep)

    fd.flush()

def make_top_header():
    """
    Returns header lines containing machine and files info
    """

    # machine info
    mach_name, mach_arch = common.machine_info()

    # out file names
    script_file_name = sys.modules[__name__].__file__

    # general and file info lines
    header = ["#",
        "# Machine: " + mach_name + " " + mach_arch,
        "# Date: " + time.asctime(time.localtime()),
        "#"]
    header.extend(common.format_file_info(name=script_file_name, 
                     description="Input script", extra=("  "+__version__)))
    header.append("# Working directory: " + os.getcwd())
    header.extend(common.format_file_info(name=image_file_name, 
                                   description="Image (in)"))
    header.extend(common.format_file_info(name=labels_file_name, 
                                   description="Boundaries (in)"))

    return header
 
def make_header(analysis=None, cleft_layers=None, thresh=None, name=''):
    """
    Returns part of the header for a result file. Can be used for a single 
    threshold or hierarchical segmentation, as well as for all thresholds.

    Arguments:
      - analysis: segmentation analysis
      - cleft_layers: cleft layers
      - thresh: threshold
      - name: inserted before suffix in the results and pickle file names,
      usually a class name
    """

    # set seg, ids and threshold
    ids = None
    seg = None
    hierarchy = None
    if analysis is not None:
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

    # rest of file names
    lay_file_name = common.make_file_name(
        directory=lay_directory, prefix=lay_prefix,
        insert_root=lay_insert_root, reference=image_file_name,
        suffix=lay_suffix)
    suffix = name + sa_suffix
    sa_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, suffix=suffix,
        insert_root=sa_insert_root, reference=image_file_name)
    header.extend(common.format_file_info(name=lay_file_name, 
                                   description="Layers image"))
    header.extend(common.format_file_info(name=sa_file_name, 
                            description="SegmentationANalysis pickle (out)"))
    header.append("# Working directory: " + os.getcwd())

    # parameters
    header = [
        "#",
        "# Boundary ids",
        "#     - presynaptic: " + str(id_2),
        "#     - postsynaptic: " + str(id_1),
        "#     - cleft: " + str(seg_id),
        "#",
        "# Cleft edges",
        "#      - max distance to membranes: " + str(max_distance),
        "#      - cleft adjusted by layers: "\
            + str(adjust_cleft_by_layers),
        "#      - cleft unit area: " + str(unit_cleft_area)]

    # cleft 
    if cleft_layers is not None:
        header.extend([
            "#",
            "# Cleft layer results",
            "#",
            ("#   Thickness (edge to edge): %7.1f,  mode %s" 
             % (cleft_layers.width, distance_mode)),
            "#   Direction (from pre- to post- side):",
            "#     - theta: %6.1f deg" % \
                cleft_layers.widthVector.getTheta('deg'),
            "#     - phi: %6.1f deg" % cleft_layers.widthVector.getPhi('deg'),
            "#",
            "#   Density dip in the cleft:",
            "#     - layer id: %d, fractional position: %5.2f" \
                 % (cleft_layers.minCleftDensityId, 
                   cleft_layers.minCleftDensityPosition),
            "#     - relative density: %7.2f" \
                % cleft_layers.relativeMinCleftDensity])
    
    # structuring elements
    header.extend([
            "#",
            "# Structuring element connectivities"])
    if analysis is not None:
        header.extend([
                ("#     - connection formation: " 
                 + str(analysis.structElConn)),
                "#     - contact detection: " 
                + str(analysis.contactStructElConn),
                "#     - contact counting: " 
                + str(analysis.countStructElConn)])
    header.extend([
        "#     - topology: 1",
        "#"])

    # connectors
    if analysis is not None:
        header.extend([
                "# Connectors:",
                "#     - number of boundaries: " + str(analysis.nBoundary),
                "#     - boundary count: " + analysis.boundCount,
                "#"])

    # length
    if ((analysis is not None) and ('morphology' in analysis.resultNames)):
        header.extend([
                "# Length",
                "#     - contact mode: " + length_contact_mode,
                "#     - line mode: " + length_line_mode,
                "#"])

    # threshold
    if thresh is not None:
        header.extend([
                "# Threshold: " + tr_str,
                "#"])
    elif hierarchy is not None:
        header.extend([
                ("# Thresholds:" +
                 (' ' + threshold_format) * len(threshold) 
                 % tuple(hierarchy.threshold)),
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
            
    # number of segments and contacts
    if ids is not None:
        header.extend([
                "# Segments and contacts: ",
                "#     - Total number of segments: " + str(len(ids))])
        if getattr(analysis, 'surfaceDensitySegments', None) is not None: 
            seg_dens = analysis.surfaceDensitySegments[0] * unit_cleft_area
            contact_dens_1 = \
                analysis.surfaceDensityContacts[1] * unit_cleft_area
            contact_dens_2 = \
                analysis.surfaceDensityContacts[2] * unit_cleft_area
            header.extend([
                    "#     - Surface density of segments per unit area: %6.2f" \
                        % seg_dens,
                    "#     - Surface density of contacts at postsynaptic " \
                        + "membrane per unit area: %6.2f" \
                        % contact_dens_1,
                    "#     - Surface density of contacts at presynaptic " \
                        + "membrane per unit area: %6.2f" \
                        % contact_dens_2,
                    "#     - Unit area (pix^2): " + str(unit_cleft_area)])
        header.append("#")

    return header

def write_all_thresh_header(fd, cleft_layers=None, analysis=None):
    """
    Writes header for all threshold results

    Arguments:
      - fd: file descriptor
      - cleft_layers: (scene.CleftRegions) cleft layers
      - analysis: (scene.SegmentationAnalysis) segmentation analysis
    """

    # top of the header
    header = make_top_header()

    # rest of file names
    pkl_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, suffix=sa_suffix,
        insert_root=sa_insert_root, reference=image_file_name)
    header.extend(common.format_file_info(
            name=pkl_file_name, 
            description="SegmentationAnalysis pickle (out)"))

    # parameters header 
    header.extend(make_header(cleft_layers=cleft_layers, analysis=analysis))

    # write header
    #write_header(fd=fd)
    
    # write results table head
    tab_head = [("# Threshold        Density                  Volume    "
                 + "Surface  S/V    Complexes  Euler Loops Contacts"),
                ("#          mean    std     min     max     abs  rel   "
                 + "                N unit area            post pre" )]
    for line in header + tab_head: 
        fd.write(line + os.linesep)
        
def append_all_thresh_results(analysis, thresh, fd=None):
    """
    Makes a line containing total results for one threshold. 

    If arg fd is given writes the line to the all thresholds results file.
    Otherwise returns the line

    Arguments:
      - analysis: analysis object
      - thresh: current threshold
      - fd: file descriptor
    """

    # don't write anything if no segments
    n_complexes = len(analysis.segments.ids)
    if n_complexes == 0:
        return

    # variables
    dens = analysis.density
    mor = analysis.morphology
    topo = analysis.topology
    complex_density = analysis.surfaceDensitySegments[0] * unit_cleft_area
    n_contact_1 = analysis.nContacts[1]
    n_contact_2 = analysis.nContacts[2]
    cleft_volume = analysis.regionDensity.volume
    
    # write line
    out_vars = (thresh, dens.mean[0], dens.std[0], dens.min[0], dens.max[0], 
                mor.volume[0], 1.*mor.volume[0]/cleft_volume, 
                mor.surface[0], 1.*mor.surface[0]/mor.volume[0], 
                n_complexes, complex_density, topo.euler[0], topo.nLoops[0], 
                n_contact_1[0], n_contact_2[0])
    out_format = ('%7.4f  %7.3f %6.3f %7.3f %7.3f %6u %6.3f %6u %6.3f '
                  + '%3u %6.2f  %5i %5i  %3u  %3u')
    line = out_format % out_vars

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
        fd.write(os.linesep + "# Whole cleft:" + os.linesep)
        out_vars = (dens.mean, dens.std, dens.min, dens.max, dens.volume)
        out_format = '        %7.3f %6.3f %7.3f %7.3f %6u'
        cleft_line = out_format % out_vars
        fd.write('#' + cleft_line + os.linesep)
        
    fd.flush()


################################################################
#
# Main function
#
###############################################################

def main():
    """
    Main function
    """
                       
    # begin logging
    mach_name, mach_arch = common.machine_info()
    logging.info('Machine: ' + mach_name + ' ' + mach_arch)
    logging.info('Begin (script ' + __version__ + ')')

    # read image and boundaries
    image = read_image()
    bound = read_boundaries(shape=image.data.shape)

    # set insets
    full_inset = bound.inset
    bound.makeInset(extend=1+n_extra_layers)
    image.useInset(inset=bound.inset, mode='absolute')
                       
    #######################################################
    #
    # Formation and grey-scale analysis of layers
    #

    if do_layers_flag:

        # calculate layers
        logging.info('Starting layer formation and analysis')
        cleft_lay = do_regions(
            mode='layers', image=image, bound=bound, inset=full_inset, 
            reference=image_file_name)

    else:

        # read layers pickle file 
        logging.info('Reading layers')
        cl_file_name = common.make_file_name(
            directory=lay_pkl_directory, prefix=lay_pkl_prefix, 
            suffix=lay_pkl_suffix, insert_root=lay_pkl_insert_root, 
            reference=image_file_name)
        cleft_lay = pickle.load(open(cl_file_name, 'rb'), encoding='latin1')
        
    # adjust cleft region and boundaries according to the layers
    #adjust_boundaries(bound=bound, regions=cleft_lay.regions)

    #######################################################
    #
    # Formation and grey-scale analysis of columns
    #

    if do_columns_flag:

        # calculate thin layers
        logging.info('Starting thin layers formation and analysis')
        cleft_thin_lay = do_regions(
            mode='thin_layers', image=image, bound=bound, inset=full_inset, 
            reference=image_file_name)

        # calculate columns
        logging.info('Starting column formation and analysis')
        cleft_col = do_regions(
            mode='columns', image=image, bound=bound, inset=full_inset, 
            layers=cleft_thin_lay, reference=image_file_name)

    else:

        # read columns pickle file 
        col_file_name = common.make_file_name(
            directory=col_directory, prefix=col_prefix, suffix=col_suffix,
            insert_root=col_insert_root, reference=image_file_name)
        try:
            cleft_col = pickle.load(
                open(col_file_name, 'rb'), encoding='latin1')
            logging.info('Reading columns')
        except IOError:
            do_layers_on_columns_flag = False        

    #######################################################
    #
    # Formation and grey-scale analysis of layers formed on columns
    #

    if do_layers_on_columns_flag:

        logging.info('Starting layers on columns') 
        for layers_on_col, column_id in intersect_layers_columns(
            layers=cleft_lay, columns=cleft_col):

            # analyze layers 
            cleft_layers_on_col = do_regions(
                mode='layers_on_columns', image=image, bound=bound, 
                regions=layers_on_col, inset=full_inset, 
                reference=image_file_name, in_suffix=column_id)

    ######################################################
    # 
    # Segmentation 
    #

    # segmentation and analysis pickle file name
    sa_file_name = common.make_file_name(
        directory=sa_directory, prefix=sa_prefix, suffix=sa_suffix,
        insert_root=sa_insert_root, reference=image_file_name)

    if do_segmentation_flag:

        # do segmentation and analysis
        logging.info('Starting segmentation')
        sa = do_segmentation(
            image=image, bound=bound, cleft_lay=cleft_lay, 
            sa_file_name=sa_file_name, inset=full_inset)

    elif read_segmentation_flag:

        # read segmentation and analysis
        logging.info('Reading segmentation')
        sa = common.read_pickle(file_name=sa_file_name, 
                                compact=['hierarchy.contacts'])

    else:
        return

    ######################################################
    # 
    # Classification and segment analysis
    #

    if do_classify_flag or do_layers_on_segments_flag:

        logging.info('Starting classification')
        for sa_cls, sa_cls_name, sa_cls_file_name \
                in do_classification(sa=sa, inset=full_inset, 
                                     cleft_lay=cleft_lay):

            ######################################################
            # 
            # Layers on segments
            #

            if do_layers_on_segments_flag:

                logging.info('Starting layers on segments of class ' 
                             + sa_cls_name)
                do_regions(mode='layers', regions=cleft_lay, mask=sa_cls.labels,
                          inset=full_inset, reference=sa_cls_file_name)
                
            else:
                pass


# run if standalone
if __name__ == '__main__':
    main()

