"""

    Script for extracting slices from a SynGraphMCF

    Input:  - List of paths to SynGraphMCF pickles
            - Path to XML file with slices description

    Output: - Slices segmentations
            - UniStat object for every slice

"""

__author__ = 'Antonio Martinez-Sanchez'
import pyseg as ps

########################################################################################
# PARAMETERS
########################################################################################


ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1'

# Input pickle
input_pkl_l = (ROOT_PATH + '/ex/syn/tm/graphs_2/syn_14_9_bin2_rot_crop2.pkl',
	           ROOT_PATH + '/ex/syn/tm/graphs_2/syn_14_13_bin2_rot_crop2.pkl',
	           # ROOT_PATH + '/ex/syn/tm/graphs_2/syn_14_14_bin2_rot_crop2.pkl',
               # ROOT_PATH + '/ex/syn/tm/graphs_2/syn_14_15_bin2_rot_crop2.pkl',
               )
del_coord_l = (1,
	           0,
	           # 0,
	           # 0,
               )
pkl_lbl_l = ('9',
	         '13',
             # '14',
             # '15',
            )

####### Output data

output_dir = ROOT_PATH + '/ex/syn/slices/tm_test/pst'

###### Slices settings file

slices_file = ROOT_PATH + '/ex/syn/slices/tm_test_pst.xml'

###### Thresholds for graph

store_seg = 1 # 1 slice graph is stored as .vtp, 2 slices vertices are also printed

####### Input parameters

plane = False # If True coordinates are projected to a plane (see del_coord_l)
del_v_sl = True # if True vertices are being deleted from graph as they detected in a slice

###### Peaks configuration (if peak_prop is None it is deactivated)

peak_prop = 'ampar_cc'
peak_prop_ang = 'ampar_ang'
peak_prop_norm = 'ampar_norm'
peak_dst = ps.globals.STR_VERT_DST
peak_th = 10 # Percentile %
peak_ns = 10
peak_nn = 1
peak_ang = 180 # degrees

##### Storing subvolumes

in_tomo_refs = (ROOT_PATH+'/in/zd/bin2/syn_14_9_bin2_s1.5.mrc',
                ROOT_PATH+'/in/zd/bin2/syn_14_13_bin2_s1.5.mrc',
	            # ROOT_PATH+'/in/zd/bin2/syn_14_14_bin2.mrc',
                # ROOT_PATH+'/in/zd/bin2/syn_14_15_bin2.mrc',
                )
in_offs = ((447,911,159),
           (767,847,87),
	       # (747,991,107),
           # (731,883,99),
           )
in_rots = ((0,-2,0),
           (0,-67,0),
           # (0,-19,0),
           # (0,44,0),
           )
sv_shape = (52, 52, 52)

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import sys
import time
import errno
import operator
import numpy as np
from pyseg.xml_io import SynSliceSet
from pyseg.spatial import UniStat
from pyseg.spatial.plane import make_plane
from pyseg.sub import TomoPeaks, SetTomoPeaks

########## Global variables

########## Print initial message

print 'Extracting slices from synapses.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(output_dir)
print '\tSlices file: ' + slices_file
if plane:
    print '\tMode 2D (plane projection) activated.'
    if peak_prop is not None:
        peak_prop = None
        print 'WARNING: peaks detection is deactivated because is not compatible with 2D mode.'
else:
    print '\tMode 3D activated.'
if del_v_sl:
    print '\tProgressive vertex deleting active.'
if peak_prop is not None:
    print '\tPeaks detection active, settings:'
    print '\t\t-Vertices property: ' + peak_prop
    print '\t\t\t+Threshold: ' + str(peak_th)
    print '\t\t-Edge property for distance measuring: ' + peak_dst
    print '\t\t\t+Neighbourhood radius: ' + str(peak_ns)
    print '\t\t\t+Number of neighbors: ' + str(peak_nn)
    print '\t\t\t+Maximum angle: ' + str(peak_ang) + ' deg'
print ''

######### Process

print '\tLoading XML file with the slices...'
slices = SynSliceSet(slices_file)

set_tpeaks = SetTomoPeaks()
print '\tTomograms loop:'
for (input_pkl, del_coord, pkl_lbl, in_tomo_ref, in_off, in_rot) in \
        zip(input_pkl_l, del_coord_l, pkl_lbl_l, in_tomo_refs, in_offs, in_rots):

    print '\t\tLoading the input graph: ' + input_pkl

    print '\t\tUnpicking graph...'
    path, fname = os.path.split(input_pkl)
    stem_name, _ = os.path.splitext(fname)
    graph = ps.factory.unpickle_obj(input_pkl)

    if not del_v_sl:
        print '\t\tUpdating GraphGT...'
        graph.compute_graph_gt()
        graph_gt = graph.get_gt(fupdate=True)

    print '\t\tSlices loop (' + str(slices.get_num_slices()) + ' slices found):'
    for sl in slices.get_slices_list():

        if del_v_sl:
            print '\t\tUpdating GraphGT...'
            graph_gtt = ps.graph.GraphGT(graph)
            graph_gt = graph_gtt.get_gt()

        print '\t\tProcessing slice ' + sl.get_name() + ':'
        print '\t\tSegmentation label: ' + str(sl.get_seg())
        print '\t\tMembrane: ' + str(sl.get_mb())
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
        for th in sl.get_list_th():
            print '\t\t\t\tVertices threshold: ' + th.get_name()
            print '\t\t\t\t\t-Property: ' + th.get_prop_key()
            print '\t\t\t\t\t-Mode: ' + th.get_mode()
            print '\t\t\t\t\t-Range: ' + str(th.get_range()) + ' %'
        if sl.get_cont():
            print '\t\t\t-Contact points mode active.'
            cloud, cloud_ids, mask = graph.get_cloud_mb_slice(sl, cont_mode=True, graph_gt=graph_gt)
            print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))
        else:
            cloud, cloud_ids, mask = graph.get_cloud_mb_slice(sl, cont_mode=False, graph_gt=graph_gt)
            print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))

        if peak_prop is not None:
            print '\t\tFiltering points with ' + peak_prop + ' and percentile ' + str(peak_th) + ' %'
            cloud_cc = graph.get_prop_values(peak_prop, cloud_ids)
            hold_cloud, hold_cloud_ids = cloud, cloud_ids
            cloud, cloud_ids = list(), list()
            per_th = np.percentile(cloud_cc, peak_th)
            for (point, cloud_id, c_cc) in zip(hold_cloud, hold_cloud_ids, cloud_cc):
                if c_cc >= per_th:
                    cloud.append(point)
                    cloud_ids.append(cloud_id)
            print '\t\tComputing vertices normal to membrane...'
            prop_norm, prop_ang_c = graph.angle_vector_norms(peak_prop_norm, sl.get_mb(), v_ids=None)
            print '\t\tTemporal copy of current graph...'
            hold_graph = graph.gen_subgraph(cloud_ids)
            graph_gtt = ps.graph.GraphGT(hold_graph)
            print '\t\tFinding peaks through DBSCAN clustering...'
            cloud_ids = graph_gtt.find_peaks_dbscan(peak_prop+'_dbscan', ps.globals.STR_VERT_DST, graph,
                                                    eps=peak_ns, min_samples=peak_nn)
            cloud = graph.get_vertices_coords(cloud_ids)
        elif del_v_sl:
            graph.threshold_vertices_list(cloud_ids, in_mode=True)

        if plane:
            print '\t\tProjecting coordinates into a plane...'
            cloud = make_plane(cloud, del_coord)
            mask = (mask.sum(axis=del_coord) > 0)
            print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))

        output_seg = stem_name + '_' + sl.get_name()
        if peak_prop is not None:
            print '\t\tCreating the peaks container...'
            tomo_peaks = TomoPeaks(shape=mask.shape, name=output_seg, mask=mask)
            tomo_peaks.add_peaks(cloud)
            tomo_peaks.add_prop(peak_prop, n_comp=1,
                                vals=graph.get_prop_values(prop_key=peak_prop, ids=cloud_ids))
            tomo_peaks.add_prop(prop_ang_c, n_comp=1,
                                vals=graph.get_prop_values(prop_key=prop_ang_c, ids=cloud_ids))
            tomo_peaks.add_prop(prop_norm, n_comp=3,
                                vals=graph.get_prop_values(prop_key=prop_norm, ids=cloud_ids))
            tomo_peaks.add_prop(peak_prop_norm, n_comp=3,
                                vals=graph.get_prop_values(prop_key=peak_prop_norm, ids=cloud_ids))
            tomo_peaks.add_prop(peak_prop_ang, n_comp=3,
                                vals=graph.get_prop_values(prop_key=peak_prop_ang, ids=cloud_ids))
            print '\t\t\t-Number of peaks found: ' + str(tomo_peaks.get_num_peaks())
            if tomo_peaks.get_num_peaks() == 0:
                print 'WARNING: number of peaks for this slice is 0 no further analysis can be applied, slice skipped!'
                continue
            set_tpeaks.add_tomo_peaks(tomo_peaks, in_tomo_ref, swap_xy=True)
        elif len(cloud_ids) == 0:
            print 'WARNING: number of points for this slice is 0 no further analysis can be applied, slice skipped!'
            continue

        print '\t\tBuilding statistical analyzer...'
        print '\t\t\tCurrent number of points: ' + str(len(cloud_ids))
        print '\t\t\tPrint mask size (n voxels): ' + str(mask.sum())
        unisp = UniStat(cloud, mask, graph.get_resolution(), name=sl.get_name()+'_'+pkl_lbl)

        print '\t\t\t-Vertices found: ' + str(cloud.shape[0])
        output_sp = stem_name + '_' + sl.get_name() + '_unisp.pkl'
        print '\t\t\t-Storing spatial analyzer as : ' + output_sp
        output_sp = output_dir + '/' + output_sp
        unisp.pickle(output_sp)
        if (store_seg == 1) or (store_seg == 2):
            print '\t\t\t-Storing slices graph with name: ' + output_seg
            print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))
            out_seg = output_dir + '/' + output_seg
            ps.disperse_io.save_vtp(graph.slice_to_vtp(cloud_ids, mb_id=sl.get_mb()), out_seg+'.vtp')
        if store_seg == 2:
            print '\t\t\t-Printing slices vertices with name: ' + output_seg
            ps.disperse_io.save_numpy(graph.print_slice(cloud_ids, th_den=0, slc=True), out_seg+'.vti')

        if peak_prop is not None:
            out_vtp = output_dir + '/' + output_seg + '_peak.vtp'
            print '\t\t\t-Storing peaks (vtp): ' + out_vtp
            ps.disperse_io.save_vtp(tomo_peaks.to_vtp(), out_vtp)
            out_sub = output_dir + '/' + output_seg + '_sta'

            if in_tomo_ref:
                print '\t\t\t-Converting graph coordinate to reference tomogram coordinates...'
                tomo_ref = ps.disperse_io.load_tomo(in_tomo_ref, mmap=True)
                tomo_peaks.peaks_coords_swapxy()
                center = (tomo_ref.shape[0]*.5, tomo_ref.shape[1]*.5, tomo_ref.shape[2]*.5)
                tomo_peaks.peaks_prop_op(ps.sub.PK_COORDS, (in_off[0], in_off[1], in_off[2]), operator.add)
                tomo_peaks.rotate_coords(float(in_rot[0]), float(in_rot[1]), float(in_rot[2]), center=center)

if peak_prop is not None:
    out_pl = output_dir + '/' + peak_prop + '_pl'
    print '\tStoring peaks particle list and subvolumes in directory: ' + out_pl
    try:
        os.makedirs(out_pl)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            print 'ERROR: directory ' + out_pl + ' could not be created'
            print 'Wrong terminated. (' + time.strftime("%c") + ')'
            sys.exit()
    plist = set_tpeaks.gen_plist(out_pl+'/'+peak_prop+'_pl.xml')
    plist.save_subvolumes(sv_shape, purge=True, align=False)
    plist.store()
    plist.save_plain_txt(output_dir+'/'+peak_prop+'_plain.txt')
    print '\t\t-Number of particles found: ' + str(plist.get_num_particles())

print 'Terminated. (' + time.strftime("%c") + ')'