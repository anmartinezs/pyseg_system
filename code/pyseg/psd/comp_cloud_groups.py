"""

    Script for comparing statistics from different groups of clouds

    Input:  - Path to the input SetCloudP pickles

    Output: - Plot graphs with the statistical comparison
            - Store a CloudComparator object in pickle file

"""

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################


ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1'

# Input pickles
groups_pkl = ((ROOT_PATH + '/uli/pre/graph_stat/syn_11_1_bin2_sirt_rot_crop2_c0.01_s1.3_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_2_bin2_sirt_rot_crop2_c0.01_s1.3_att_slices_2.pkl',
              ROOT_PATH + '/uli/pre/graph_stat/syn_11_5_bin2_sirt_rot_crop2_c0.01_s1.3_att_slices_2.pkl'),
              (ROOT_PATH + '/zd/pre/graph_stat/syn_14_9_bin2_sirt_rot_crop2_att_slices_2.pkl',
              ROOT_PATH + '/zd/pre/graph_stat/syn_14_13_bin2_sirt_rot_crop2_att_slices_2.pkl',
              ROOT_PATH + '/zd/pre/graph_stat/syn_14_14_bin2_sirt_rot_crop2_att_slices_2.pkl',
              ROOT_PATH + '/zd/pre/graph_stat/syn_14_15_bin2_sirt_rot_crop2_att_slices_2.pkl'),

              (ROOT_PATH + '/uli/pre/graph_stat/syn_11_1_bin2_sirt_rot_crop2_c0.01_s1.3_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_2_bin2_sirt_rot_crop2_c0.01_s1.3_att_slices_2.pkl',
              ROOT_PATH + '/uli/pre/graph_stat/syn_11_5_bin2_sirt_rot_crop2_c0.01_s1.3_att_slices_2.pkl'),
              (ROOT_PATH + '/zd/pre/graph_stat/syn_14_9_bin2_sirt_rot_crop2_att_slices_2.pkl',
              ROOT_PATH + '/zd/pre/graph_stat/syn_14_13_bin2_sirt_rot_crop2_att_slices_2.pkl',
              ROOT_PATH + '/zd/pre/graph_stat/syn_14_14_bin2_sirt_rot_crop2_att_slices_2.pkl',
              ROOT_PATH + '/zd/pre/graph_stat/syn_14_15_bin2_sirt_rot_crop2_att_slices_2.pkl')

              # (ROOT_PATH + '/uli/pre/graph_stat/syn_11_1_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_2_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_5_bin2_sirt_rot_crop2_att_slices_2.pkl'),
              # (ROOT_PATH + '/zd/pre/graph_stat/syn_14_9_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_13_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_14_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_15_bin2_sirt_rot_crop2_att_slices_2.pkl'),
              #
              # (ROOT_PATH + '/uli/pre/graph_stat/syn_11_1_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_2_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_5_bin2_sirt_rot_crop2_att_slices_2.pkl'),
              # (ROOT_PATH + '/zd/pre/graph_stat/syn_14_9_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_13_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_14_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_15_bin2_sirt_rot_crop2_att_slices_2.pkl'),
              #
              # (ROOT_PATH + '/uli/pre/graph_stat/syn_11_1_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_2_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_5_bin2_sirt_rot_crop2_att_slices_2.pkl'),
              # (ROOT_PATH + '/zd/pre/graph_stat/syn_14_9_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_13_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_14_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_15_bin2_sirt_rot_crop2_att_slices_2.pkl'),
              #
              # (ROOT_PATH + '/uli/pre/graph_stat/syn_11_1_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_2_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/uli/pre/graph_stat/syn_11_5_bin2_sirt_rot_crop2_att_slices_2.pkl'),
              # (ROOT_PATH + '/zd/pre/graph_stat/syn_14_9_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_13_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_14_bin2_sirt_rot_crop2_att_slices_2.pkl',
              # ROOT_PATH + '/zd/pre/graph_stat/syn_14_15_bin2_sirt_rot_crop2_att_slices_2.pkl')
             )
# groups_slice = ('cito', 'cito_l', 'cito_r', 'cito', 'cito_l', 'cito_r')
groups_slice = ('cito', 'cito', 'cleft', 'cleft') # 'cito_r', 'cito_r', 'cleft', 'cleft', 'cleft_l', 'cleft_l', 'cleft_r', 'cleft_r')
# groups_name = ('S_Y', 'S_Y_T', 'S_Y_N', 'R_Y', 'R_Y_T', 'R_Y_N')
groups_name = ('PDBu_R_Y', 'R_Y', 'PDBu_R_C', 'R_C') # , 'PDBu_R_Y_R', 'R_Y_R','PDBu_R_C', 'R_C', 'PDBu_R_C_L', 'R_C_L', 'PDBu_R_C_R', 'R_C_R')

####### Output data

output_dir = ROOT_PATH + '/groups/pre_pdbu_vs_zd'
prefix_name = 'pre_pdbu_vs_zd_c0.01_s1.3_att_slices_2' # Stem for stored files

####### Display options

disp_plots = True
store_plots = True
store_clouds = True # level_2 must be True

####### Analysis options

level_1 = True # Functions F and G
level_2 = True # Ripleys

# Level 1
n_samples = 80
n_sim = 2000
per_h = 2.5 # %

# Level 2
max_d = 60 # nm

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import sys
import pyseg as ps
from pyseg.spatial import GroupClouds

########## Global variables

########## Print initial message

print('Statistics comparison from different profiles.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput file(s): ' + str(groups_pkl))
print('\tGroup slices: ' + str(groups_slice))
print('\tGroup names: ' + str(groups_name))
print('\tOutput directory: ' + str(output_dir))
print('\tOutput prefix: ' + str(prefix_name))
if level_1:
    print('\tAnalysis level 1 activated.')
if level_2:
    print('\tAnalysis level 2 activated.')
if (not level_1) and (not level_2):
    print('\tERROR: no analysis activated.\nFinishing...')
    sys.exit()
if disp_plots:
    print('\tPlots display activated.')
if store_plots:
    print('\tPlots store activated.')
if store_clouds:
    if level_2:
        print('\tClouds store activated.')
    else:
        print('\tERROR: Clouds storing cannot be activated without Level 2 Analysis.\nFinishing...')
        sys.exit()
print('')

######### Process

print('Loading clouds from input pickles...')
groups_cloud = list()
groups_boxes = list()
for (group_pkl, slice_name) in zip(groups_pkl, groups_slice):
    clouds = list()
    boxes = list()
    for pkl in group_pkl:
        cloud_set = ps.factory.unpickle_obj(pkl)
        clouds.append(cloud_set.get_cloud_by_name(slice_name))
        boxes.append(cloud_set.get_box())
    groups_cloud.append(clouds)
    groups_boxes.append(boxes)

print('Comparator initialization...')
comp = GroupClouds(n_samples, n_sim, max_d, per_h)
for (group_cloud, group_boxes, group_name) in zip(groups_cloud, groups_boxes, groups_name):
    comp.insert_group(group_cloud, group_boxes, group_name)

if level_1:
    print('Analysis level 1...')
    comp.analyze_1(verbose=True)
if level_2:
    print('Analysis level 2...')
    comp.analyze_2(verbose=True)

if disp_plots:
    if level_1:
        print('Plotting level 1 results, close all windows to continue...')
        comp.plot_1(block=True)
    if level_2:
        print('Plotting level 2 results, close all windows to continue...')
        comp.plot_2(block=True)

if store_plots:
    if level_1:
        fig_dir = output_dir + '/' + prefix_name + '_1'
        print('Storing results level 1 in directory: ' + fig_dir)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        comp.store_figs_1(fig_dir)
    if level_2:
        fig_dir = output_dir + '/' + prefix_name + '_2'
        print('Storing results level 2 in directory: ' + fig_dir)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        comp.store_figs_2(fig_dir, plt_cl=store_clouds)

file_name = output_dir + '/' + prefix_name + '.pkl'
print('Picking analyzer in file: ')
comp.pickle(file_name)

print('Terminated. (' + time.strftime("%c") + ')')

