"""

    Script for applying Random Walk with Restart to a SynGraphMCF

    Input:  - List of paths to SynGraphMCF pickles
            - Path to XML file with sources groups

    Output: - SynGraphMCF with RWR properties added

"""

###### Global variables

import pyseg as ps
__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1'

# Input pickle
in_pkls = (ROOT_PATH + '/ex/syn/graphs_2/syn_14_9_bin2_rot_crop2.pkl',
	       # ROOT_PATH + '/ex/syn/graphs/syn_14_13_bin2_rot_crop2.pkl',
	       # ROOT_PATH + '/ex/syn/graphs/syn_14_14_bin2_rot_crop2.pkl',
           # ROOT_PATH + '/ex/syn/graphs/syn_14_15_bin2_rot_crop2.pkl',
           )

####### Output data

store_lvl = 2
output_dir = ROOT_PATH + '/ex/syn/rwr/mb_1'

###### RWR settings file

in_xml = ROOT_PATH + '/ex/syn/rwr/mb_1.xml'

####### Input parameters

key_w_v = None
key_w_e = None
inv_w_e = False

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import operator
import numpy as np
from pyseg.xml_io import RwrGroupSet

########## Global variables

########## Print initial message

print('Applying RWR synapses.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput graphs: ' + str(in_pkls))
print('\tOutput directory: ' + str(output_dir))
print('\tRWR XML file: ' + in_xml)
if key_w_v is not None:
    print('\tVertex weighting: ' + key_w_v)
if key_w_e is not None:
    print('\tEdge weighting: ' + key_w_e)
    if inv_w_e:
        print('\tEdge weighting inversion activated!')
print('')

######### Process

print('\tLoading XML file with the slices...')
rwr_set = RwrGroupSet(in_xml)

print('\tGraphs loop:')

for in_pkl in in_pkls:

    print('\t\tLoading the input graph: ' + in_pkl)

    print('\t\tUnpicking graph...')
    path, fname = os.path.split(in_pkl)
    f_stem, _ = os.path.splitext(fname)
    graph = ps.factory.unpickle_obj(in_pkl)
    graph.compute_graph_gt()
    graph_gt = ps.graph.GraphGT(graph)

    print('\t\tRWR groups loop (' + str(rwr_set.get_num_groups()) + ' groups found):')
    for group in rwr_set.get_groups_list():

        print('\t\tProcessing group ' + group.get_name() + ':')
        graph.add_prop(group.get_name(), 'float', 1, def_val=1.)

        print('\t\tSlices loop (' + str(group.get_num_slices()) + ' slices found):')
        for sl_i in range(group.get_num_slices()):

            sl = group.get_slice(sl_i)
            sgn = group.get_sign(sl_i)
            rwr_c = group.get_rwr_c(sl_i)
            print('\t\t\tProcessing slice ' + sl.get_name() + ':')
            print('\t\t\tSegmentation label: ' + str(sl.get_seg()))
            print('\t\t\tMembrane: ' +str(sl.get_mb()))
            print('\t\t\t\t-Euclidean distance: (' + sl.get_eu_dst_sign() + ')[' \
                  + str(sl.get_eu_dst_low()) + ', ' + str(sl.get_eu_dst_high()) + '] nm')
            print('\t\t\t\t-Geodesic distance: (' + sl.get_geo_dst_sign() + ')[' \
                  + str(sl.get_geo_dst_low()) + ', ' + str(sl.get_geo_dst_high()) + '] nm')
            print('\t\t\t\t-Geodesic length: (' + sl.get_geo_len_sign() + ')[' \
                  + str(sl.get_geo_len_low()) + ', ' + str(sl.get_geo_len_high()) + '] nm')
            print('\t\t\t\t-Sinuosity: (' + sl.get_sin_sign() + ')[' \
                  + str(sl.get_sin_low()) + ', ' + str(sl.get_sin_high()) + '] nm')
            print('\t\t\t\t-Cluster number of vertices: (' + sl.get_cnv_sign() + ')[' \
                  + str(sl.get_cnv_low()) + ', ' + str(sl.get_cnv_high()) + ']')
            cloud, cloud_ids, mask = graph.get_cloud_mb_slice(sl, cont_mode=False)
            print('\t\t\t\tCurrent number of vertices: ' + str(len(cloud_ids)))

            print('\t\t\tRWR (c=' + str(rwr_c) + ') on slice: ' + sl.get_name())
            rwr_sl_name = 'rwr_' + sl.get_name()
            graph.rwr_sources(cloud_ids, rwr_sl_name, rwr_c, key_w_v, key_w_e, inv_w_e, graph_gt)

            print('\t\t\tUpdating RWR group: ')
            graph.two_props_operator(group.get_name(), rwr_sl_name, group.get_name(), operator.mul)

    print('\tSaving graphs at level ' + str(store_lvl) + '...')
    if store_lvl > 0:
        ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                                output_dir + '/' + f_stem + '_edges.vtp')
    if store_lvl > 1:
        ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                                output_dir + '/' + f_stem + '_edges_2.vtp')
    if store_lvl > 2:
        ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                                output_dir + '/' + f_stem + '_sch.vtp')
    graph.pickle(output_dir + '/' + f_stem + '.pkl')

print('Terminated. (' + time.strftime("%c") + ')')