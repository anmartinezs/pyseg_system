"""

    Script for extracting an analyzing a GraphMCF from Chlamy tomos and analyze a specific membrane attached
    protein according to its dimension
    22.05.16

    Input:  - Input GraphMCF generated with mb_graph_batch.py script

    Output: - Output GraphMCF with the pre-segmentation of the protein specified

"""

###### Global variables

import pyseg as ps
__author__ = 'Antonio Martinez-Sanchez'

MB_SEG_LBL = 'mb_seg'
MB_EU_LEN = 'mb_eu_dst'
MB_GEO_LEN = 'mb_geo_len'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/home/martinez/workspace/disperse/data/chlamy'

# Input pickle
in_pkls = (ROOT_PATH + '/field/T1L1_b4_M1_den.pkl',
           )

# input seg
in_segs = (ROOT_PATH + '/s/T1L1_b4_M1_seg.fits',
         )

# Input offsets
in_offs =((74,282,209),
          )

####### Output data

store_lvl = 2
output_dir = ROOT_PATH + '/analysis/mb_1/str'

# Fill this with final path if subvolumes an particle list will be copied to another location before subtomo alignment
# Default: None
output_dir_2 = '/fs/pool/pool-lucic2/antonio/tomograms/ben/clamy/analysis/mb_1/str'

####### Input parameters

key_w_v = ps.globals.STR_FIELD_VALUE_EQ

###### Protein parameters

p_names = ('ribo',
           'atp',
           'field_value_eq_inv',)
is_tms = (True,
          True,
          False,)
mb_sides = (3,
            3,
            2,)
mb_eu_lens = ((4, 17),
              (4, 10),
              (1, 3),) # nm
mb_geo_lens = (90,
               30,
               3,) # nm
cc_ths = (.25,
          .3,
          .5,)
n_ngs = (2,
         2,
         1,)
mb_mx_angs = (60,
              80,
              180,)
rads = (30,
        15,
        6,) # nm

in_tomo_ref = ROOT_PATH+'/in/Tomo1L1_bin4.mrc'
sv_shapes = ((40, 40, 40),
             (30, 30, 30),
             (32, 32, 32),)

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import copy
import operator
import numpy as np
from pyseg.spatial import UniStat

########## Global variables

########## Print initial message

print 'Getting proteins attached to membranes.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput graphs: ' + str(in_pkls)
print '\tInput segmentations: ' + str(in_segs)
print '\tOutput directory: ' + output_dir
if output_dir_2 is not None:
    print '\tOutput directory for particles list: ' + output_dir_2
if key_w_v is not None:
    print '\tVertex weighting: ' + key_w_v
print 'Protein specifications:'
print '\t-Names: ' + str(p_names)
print '\t-Membrane sides: ' + str(mb_sides)
print '\t-Maximum angles: ' + str(mb_mx_angs)
print '\t-Euclidean ranges: ' + str(mb_eu_lens) + ' nm'
print '\t-Geodesic ranges: ' + str(mb_geo_lens) + ' nm'
print '\t-Field modes: ' + str(is_tms)
print '\t-Thresholds: ' + str(cc_ths)
print '\t-Radius: ' + str(rads) + ' nm'
print '\t-Number of neighs: ' + str(n_ngs)
print '\t-Reference tomograms: ' + in_tomo_ref
print '\t-Cropping offsets: ' + str(in_offs)
print '\t-Subvolume shapes: ' + str(sv_shapes)
print ''

######### Process

print 'Main Routine: '

print '\tGraphs loop:'


for (in_pkl, in_seg, in_off) in zip(in_pkls, in_segs, in_offs):

    print '\tLoading the input graph: ' + in_pkl

    print '\t\tUnpicking graph...'
    path, fname = os.path.split(in_pkl)
    f_stem, _ = os.path.splitext(fname)
    hold_graph = ps.factory.unpickle_obj(in_pkl)
    print '\t\t\t+Current number of vertices ' + str(len(hold_graph.get_vertices_list()))

    print '\t\tLoading mask...'
    seg = ps.disperse_io.load_tomo(in_seg)
    mask = seg == ps.mb.MB_LBL

    print '\t\tProteins loop:'

    for (p_name, is_cc, mb_side, cc_th, mb_mx_ang, mb_eu_len, mb_geo_len, rad, n_ng, sv_sp) in \
        zip(p_names, is_tms, mb_sides, cc_ths, mb_mx_angs, mb_eu_lens, mb_geo_lens, rads, n_ngs, sv_shapes):

        print '\t\t\tGraph copying...'
        graph = copy.deepcopy(hold_graph)

        print '\t\t\tThresholding vertices by region (preserving region ' + str(mb_side) + ')...'
        for lbl in range(1, 4):
            if lbl != mb_side:
                graph.threshold_seg_region(MB_SEG_LBL, lbl, keep_b=False)
        graph.threshold_vertices(MB_EU_LEN, 0, operator.le)
        graph.threshold_edges(MB_EU_LEN, 0, operator.le)
        print '\t\t\t\t+Current number of vertices ' + str(len(graph.get_vertices_list()))

        if mb_eu_len is not None:
            print '\t\t\tThresholding vertices by max euclidean distance to membrane of ' + str(mb_eu_len) + ' nm...'
            graph.threshold_vertices(MB_EU_LEN, mb_eu_len[0], operator.lt)
            graph.threshold_edges(MB_EU_LEN, mb_eu_len[0], operator.lt)
            graph.threshold_vertices(MB_EU_LEN, mb_eu_len[1], operator.gt)
            graph.threshold_edges(MB_EU_LEN, mb_eu_len[1], operator.gt)
            print '\t\t\t\t+Current number of vertices ' + str(len(graph.get_vertices_list()))

        if mb_geo_len is not None:
            print '\t\t\tThresholding vertices by max geodesic distance to membrane of ' + str(mb_geo_len) + ' nm...'
            graph.threshold_vertices(MB_GEO_LEN, mb_geo_len, operator.gt)
            print '\t\t\t\t+Current number of vertices ' + str(len(graph.get_vertices_list()))

        if is_cc:
            p_name_cc = p_name + '_cc'
            p_name_norm = p_name + '_norm'
            p_name_ang = p_name + '_ang'
            p_cont_norm, p_ang_c = graph.angle_vector_norms(p_name_norm, v_ids=None)
        else:
            p_name_cc = p_name

        print '\t\t\tThresholding vertices and edges by CC: ' + str(cc_th)
        graph.threshold_vertices(p_name_cc, cc_th, operator.lt)
        graph.threshold_edges(p_name_cc, cc_th, operator.lt)
        print '\t\t\t\t+Current number of vertices ' + str(len(graph.get_vertices_list()))

        if is_cc and (mb_mx_ang is not None):
            print '\t\t\tThresholding vertices by protein-contact angles: ' + str(mb_mx_ang) + ' deg'
            graph.threshold_vertices(p_ang_c, mb_mx_ang, operator.gt)
            print '\t\t\t\t+Current number of vertices ' + str(len(graph.get_vertices_list()))

        print '\t\tFinding peaks through DBSCAN clustering...'
        graph_gtt = ps.graph.GraphGT(graph)
        cloud_ids = graph_gtt.find_peaks_dbscan(p_name_cc+'_dbscan', ps.globals.STR_VERT_DST, graph,
                                                eps=rad, min_samples=n_ng)
        graph_gtt.add_prop_to_GraphMCF(graph, p_name_cc+'_dbscan', up_index=True)
        cloud = graph.get_vertices_coords(cloud_ids)
        graph.threshold_vertices(p_name_cc+'_dbscan', 0, operator.lt)

        print '\t\tCreating the peaks container...'
        tomo_peaks = ps.mb.TomoPeaks(shape=mask.shape, name=p_name, mask=mask)
        tomo_peaks.add_peaks(cloud)
        tomo_peaks.add_prop(p_name_cc, n_comp=1, vals=graph.get_prop_values(prop_key=p_name_cc, ids=cloud_ids))
        if is_cc:
            tomo_peaks.add_prop(p_name_ang, n_comp=3, vals=graph.get_prop_values(prop_key=p_name_ang, ids=cloud_ids))
            tomo_peaks.add_prop(p_ang_c, n_comp=1, vals=graph.get_prop_values(prop_key=p_ang_c, ids=cloud_ids))
            tomo_peaks.add_prop(p_name_norm, n_comp=3, vals=graph.get_prop_values(prop_key=p_name_norm, ids=cloud_ids))
            tomo_peaks.add_prop(p_cont_norm, n_comp=3, vals=graph.get_prop_values(prop_key=p_cont_norm, ids=cloud_ids))
        tomo_peaks.sort_peaks(p_name_cc, ascend=False)
        print '\t\t\t-Number of peaks found: ' + str(tomo_peaks.get_num_peaks())

        if tomo_peaks.get_num_peaks() == 0:
            print 'WARNING: number of peaks for protein ' + p_name + ' is 0, no further analysis can be applied, protein skipped!'
            continue

        print '\t\tBuilding statistical analyzer...'
        print '\t\t\tCurrent number of points: ' + str(len(cloud_ids))
        print '\t\t\tPrint mask size (n voxels): ' + str(mask.sum())
        unisp = UniStat(cloud, mask, graph.get_resolution(), name=f_stem+'_'+p_name)

        print '\t\t\t-Vertices found: ' + str(cloud.shape[0])
        output_sp = f_stem + '_' + p_name + '_unisp.pkl'
        print '\t\t\t-Storing spatial analyzer as : ' + output_sp
        output_sp = output_dir + '/' + output_sp
        unisp.pickle(output_sp)

        out_vtp = output_dir + '/' + f_stem + '_' + p_name + '_peak.vtp'
        out_csv = output_dir + '/' + f_stem + '_' + p_name + '_peak.csv'
        print '\t\t\t-Storing peaks (vtp and csv): ' + str((out_vtp, out_csv))
        ps.disperse_io.save_vtp(tomo_peaks.to_vtp(), out_vtp)
        tomo_peaks.save_csv(out_csv)
        out_sub = output_dir + '/' + f_stem + '_' + p_name + '_sta'
        out_sub_2 = None
        if output_dir_2 is not None:
            out_sub_2 = output_dir_2 + '/' + f_stem + '_' + p_name + '_sta'
        if in_tomo_ref is not None:
            print '\t\t\t-Storing peaks subvolumes in directory: ' + out_sub
            tomo_ref = ps.disperse_io.load_tomo(in_tomo_ref).astype(np.float32)
            try:
                n_peaks = tomo_peaks.save_particles(tomo_ref, sv_sp, out_sub, stem=p_name, prop_ang=p_name_ang,
                                                    ref_trans=((0, 0, 0), in_off), outdir_2=out_sub_2)
            except KeyError:
                n_peaks = tomo_peaks.save_particles(tomo_ref, sv_sp, out_sub, stem=p_name,
                                                    ref_trans=((0, 0, 0), in_off), outdir_2=out_sub_2)
            del tomo_ref
            print '\t\t\t-Number of particles found: ' + str(n_peaks)

        print '\tSaving graphs at level ' + str(store_lvl) + '...'
        if store_lvl > 0:
            ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                                    output_dir + '/' + f_stem + '_' + p_name + '_edges.vtp')
        if store_lvl > 1:
            ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                                    output_dir + '/' + f_stem + '_' + p_name + '_edges_2.vtp')
        if store_lvl > 2:
            ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                                    output_dir + '/' + f_stem + '_' + p_name + '_sch.vtp')
        graph.pickle(output_dir + '/' + f_stem + '_' + p_name + '.pkl')

        print '\t\tSuppresing peaks detected and their neighborhoods...'
        # graph.threshold_vertices(p_name_cc+'_dbscan', 0, operator.lt)
        # cloud_ids = list()
        # for v in graph.get_vertices_list():
        #     cloud_ids.append(v.get_id())
        # graph.suppress_vertices(cloud_ids)
        hold_graph.suppress_vertices(cloud_ids, rad_n=rad, key_dst=ps.disperse_io.STR_VERT_DST)

print 'Terminated. (' + time.strftime("%c") + ')'