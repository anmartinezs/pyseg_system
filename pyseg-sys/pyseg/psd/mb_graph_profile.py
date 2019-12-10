"""

    Script for computing statistics about slices of a membrane (along different penetration values)
    (v0.2) -> Working with sparse spatial analysis

    Input:  - Path to a MbGraphMCF
            - Parameters for thresholding the graph
            - Parameters for setting the statistical analysis

    Output: - Store a UniStat object in pickle file

"""

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################


ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1'

# Input pickle
input_pkl_l = (ROOT_PATH + '/ex/psd_slices/graphs_pst/syn_14_7_bin2_rot_crop2.pkl',
	           ROOT_PATH + '/ex/psd_slices/graphs_pst/syn_14_9_bin2_rot_crop2.pkl',
	           ROOT_PATH + '/ex/psd_slices/graphs_pst/syn_14_13_bin2_rot_crop2.pkl',
	           ROOT_PATH + '/ex/psd_slices/graphs_pst/syn_14_14_bin2_rot_crop2.pkl',
               ROOT_PATH + '/ex/psd_slices/graphs_pst/syn_14_15_bin2_rot_crop2.pkl'
               )
del_coord_l = (0,
	           1,
	           0,
	           0,
	           0
               )
pkl_lbl_l = ('7',
	         '9',
	         '13',
             '14',
             '15'
            )

####### Output data

output_dir = ROOT_PATH + '/ex/psd_slices/slices_data/pst_cito_1'

###### Slices settings file

slices_file = ROOT_PATH + '/ex/psd_slices/slices/pst_cito_pf.xml'

###### Thresholds for graph

store_seg = True

####### Input parameters

plane = False # If True coordinates are projected to a plane (see del_coord_l)

# Clustering

get_cgs = False
plt_clst = True
del_mask = True

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import pyseg as ps
from pyseg.xml_io import SliceSet, ThresSliceSet
from pyseg.spatial import UniStat
from pyseg.spatial.plane import make_plane

########## Global variables

MB_DST_LBL = 'dst'
MB_SEG_LBL = 1
eps_cont = 0.1 # nm

########## Print initial message

print 'Spatial analysis on membrane attached filaments. (v2)'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(output_dir)
print '\tSlices file: ' + slices_file
print ''

######### Process

print '\tLoading XML file with the slices...'
slices = SliceSet(slices_file)

print '\tLoading XML file with the thresholds...'
thresholds = ThresSliceSet(slices_file)

print '\tTomograms loop:'

for (input_pkl, del_coord, pkl_lbl) in zip(input_pkl_l, del_coord_l, pkl_lbl_l):

    print '\t\tLoading the input graph: ' + input_pkl

    print '\t\tUnpicking graph...'
    path, fname = os.path.split(input_pkl)
    stem_name, _ = os.path.splitext(fname)
    graph = ps.factory.unpickle_obj(input_pkl)
    graph.compute_graph_gt()

    print '\t\tSlices loop (' + str(slices.get_num_slices()) + ' slices found):'
    names = list()
    clouds = list()
    clouds_ids = list()
    modes = list()
    for sl in slices.get_slices_list():

        print '\t\tProcessing slice ' + sl.get_name() + ':'
        print '\t\t\t-Euclidean distance: (' + sl.get_eu_dst_sign() + ')[' \
              + str(sl.get_eu_dst_low()) + ', ' + str(sl.get_eu_dst_high()) + '] nm'
        print '\t\t\t-Geodesic distance: (' + sl.get_geo_dst_sign() + ')[' \
              + str(sl.get_geo_dst_low()) + ', ' + str(sl.get_geo_dst_high()) + '] nm'
        print '\t\t\t-Geodesic length: (' + sl.get_geo_len_sign() + ')[' \
              + str(sl.get_geo_len_low()) + ', ' + str(sl.get_geo_len_high()) + '] nm'
        print '\t\t\t-Sinuosity: (' + sl.get_sin_sign() + ')[' \
              + str(sl.get_sin_low()) + ', ' + str(sl.get_sin_high()) + '] nm'
        print '\t\t\t-Cluster number of points: (' + sl.get_cnv_sign() + ')[' \
              + str(sl.get_cnv_low()) + ', ' + str(sl.get_cnv_high()) + ']'
        if sl.get_cont():
            print '\t\t\t-Contact points mode active.'
            cloud, cloud_ids, mask = graph.get_cloud_mb_slice(sl, cont_mode=True)
            print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))
            modes.append(True)
        else:
            modes.append(False)
            if sl.get_fil():
                print '\t\t\t-Filament vertex mode active.'
                cloud, cloud_ids, mask = graph.get_cloud_mb_slice_fils(sl)
                print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))
            else:
                cloud, cloud_ids, mask = graph.get_cloud_mb_slice(sl, cont_mode=False)
                print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))

        print '\t\tLoop for slice thresholding:'
        for thres in thresholds.get_thres_list():
            print '\t\t\tProcessing threshold ' + thres.get_name() + ':'
            print '\t\t\t\t-Property key: ' + thres.get_prop_key()
            print '\t\t\t\t-Range: [' + str(thres.get_value_low()) + ', ' + str(thres.get_value_high()) + ']'
            print '\t\t\t\t-Mode: ' + thres.get_mode()
            print '\t\t\tThresholding ...'
            cloud, cloud_ids = graph.slice_vertex_threshold(cloud, cloud_ids, thres)
            print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))

        if plane:
            print '\t\tProjecting coordinates into a plane...'
            cloud = make_plane(cloud, del_coord)
            mask = (mask.sum(axis=del_coord) > 0)
            print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))

        print '\t\tBuilding statistical analyzer...'
        print '\t\t\tCurrent number of points: ' + str(len(cloud_ids))
        print '\t\t\tPrint mask size (n voxels): ' + str(mask.sum())
        unisp = UniStat(cloud, mask, graph.get_resolution(), name=sl.get_name()+'_'+pkl_lbl)

        print '\t\t\t-Vertices found: ' + str(cloud.shape[0])
        output_sp = stem_name + '_' + sl.get_name() + '_unisp.pkl'
        print '\t\t\t-Storing spatial analyzer as : ' + output_sp
        output_sp = output_dir + '/' + output_sp
        unisp.pickle(output_sp)
        if store_seg:
            output_seg = stem_name + '_' + sl.get_name()
            print '\t\t\t-Storing slices segmentation with name: ' + output_seg
            print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))
            output_seg = output_dir + '/' + output_seg
            if sl.get_side() == MB_SEG_LBL:
                unisp.save_dense(output_seg+'_d.mrc')
                ps.disperse_io.save_vtp(graph.mb_to_vtp(cloud_ids, av_mode=True, edges=True), output_seg+'_v.vtp')
                ps.disperse_io.save_vtp(graph.mb_to_vtp(cloud_ids, av_mode=False, edges=True), output_seg+'_e.vtp')
                ps.disperse_io.save_numpy(graph.print_slice(cloud_ids, th_den=0, slc=False), output_seg+'.vti')
            else:
                ps.disperse_io.save_vtp(graph.slice_to_vtp(cloud_ids), output_seg+'.vtp')
                ps.disperse_io.save_numpy(graph.print_slice(cloud_ids, th_den=0, slc=True), output_seg+'.vti')

print 'Terminated. (' + time.strftime("%c") + ')'
