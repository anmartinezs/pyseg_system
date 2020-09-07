"""

    Script for extracting slices from a SynGraphMCF

    Input:  - List of paths to SynGraphMCF pickles
            - Path to XML file with slices description

    Output: - Slices segmentations
            - UniStat object for every slice

"""

__author__ = 'Antonio Martinez-Sanchez'
import pyseg as ps
from pyseg.globals.utils import coords_scale_supression

########################################################################################
# PARAMETERS
########################################################################################


ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an'

# Input STAR file
in_star = ROOT_PATH + '/ex/syn/fils/pst/fil_sources_to_targets_net_hold.star'

####### Output data

output_dir = ROOT_PATH + '/ex/syn/fils/trans/sub'

###### Slices settings file

slices_file = ROOT_PATH + '/ex/syn/fils/pst/sub/pst_cont.xml'

###### Thresholds for graph

store_seg = 1 # 1 slice graph is stored as .vtp, 2 slices vertices are also printed

####### Input parameters

plane = False # If True coordinates are projected to a plane (see del_coord_l)
del_v_sl = True # if True vertices are being deleted from graph as they detected in a slice

###### Peaks configuration (if peak_prop is None it is deactivated)

peak_prop = 'field_value_eq'
peak_prop_pt = 'pt_normal'
peak_prop_norm = 'smb_normal'
peak_prop_rot = 'norm_rot'
peak_dst = ps.globals.STR_VERT_DST
peak_th = 10 # Percentile %
peak_ns = 5 # nn
peak_conn = False
peak_off = 10 # nm

##### Storing subvolumes
sv_shape = (128, 128, 128)

##### STAR file

st_ctf = None # ROOT_PATH + '/ex/syn/slices/neu/ctf_wedge_40_64.mrc'
st_mask = None # ROOT_PATH + '/ex/syn/slices/neu/mask_inv_sph_64_31.mrc'

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import gc
import time
import operator
import numpy as np
from pyseg.xml_io import SynSliceSet
from pyseg.spatial import UniStat
from pyseg.sub import TomoPeaks, SetTomoPeaks

########## Global variables

########## Print initial message

print('Extracting slices from synapses.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tSTAR file with the segmentations: ' + str(in_star))
print('\tOutput directory: ' + str(output_dir))
print('\tSlices file: ' + slices_file)
if plane:
    print('\tMode 2D (plane projection) activated.')
    if peak_prop is not None:
        peak_prop = None
        print('WARNING: peaks detection is deactivated because is not compatible with 2D mode.')
else:
    print('\tMode 3D activated.')
if del_v_sl:
    print('\tProgressive vertex deleting active.')
if peak_prop is not None:
    print('\tPeaks detection active, settings:')
    print('\t\t-Vertices property: ' + peak_prop)
    print('\t\t\t+Threshold (percentile): ' + str(peak_th) + ' %')
    print('\t\t-Edge property for distance measuring: ' + peak_dst)
    print('\t\t\t+Neighbourhood radius: ' + str(peak_ns))
    if peak_conn:
        print('\t\t\t-Peak connectivity active.')
    if peak_off:
        print('\t\t\t-Peak normal offset: ' + str(peak_off) + ' nm')
if st_ctf is not None:
    print('\t\t-STAR file:')
    print('\t\t\t+Ctf correction: ' + str(st_ctf))
    print('\t\t\t+Normalization mask file: ' + str(st_mask))
print('')

######### Process

print('\tParing input STAR files...')
star = ps.sub.Star()
star.load(in_star)
in_tomo_refs, in_seg_l = star.get_column_data('_rlnMicrographName'), star.get_column_data('_psSegImage')
input_pkl_l = star.get_column_data('_psGhMCFPickle')
in_offx_l, in_offy_l, in_offz_l = star.get_column_data('_psSegOffX'), star.get_column_data('_psSegOffY'), \
                                  star.get_column_data('_psSegOffZ')
in_rot_l, in_tilt_l, in_psi_l = star.get_column_data('_psSegRot'), star.get_column_data('_psSegTilt'), \
                                  star.get_column_data('_psSegPsi')
in_offs, in_rots = list(), list()
for( offx, offy, offz, rot, tilt, psi) in zip(in_offx_l, in_offy_l, in_offz_l, in_rot_l, in_tilt_l, in_psi_l):
    in_offs.append((offx, offy, offz))
    in_rots.append((rot, tilt, psi))

print('\tLoading XML file with the slices...')
slices = SynSliceSet(slices_file)

set_tpeaks = SetTomoPeaks()
print('\tTomograms loop:')
for (input_pkl, in_tomo_ref, in_seg, in_off, in_rot) in \
        zip(input_pkl_l, in_tomo_refs, in_seg_l, in_offs, in_rots):

    print('\t\tLoading the input graph: ' + input_pkl)

    print('\t\tUnpicking graph...')
    path, fname = os.path.split(input_pkl)
    stem_name, _ = os.path.splitext(fname)
    graph = ps.factory.unpickle_obj(input_pkl)
    tomo_f = ps.disperse_io.load_tomo(in_seg)

    if not del_v_sl:
        print('\t\tUpdating GraphGT...')
        graph.compute_graph_gt()
        graph_gt = graph.get_gt(fupdate=True)

    print('\t\tSlices loop (' + str(slices.get_num_slices()) + ' slices found):')
    for sl in slices.get_slices_list():

        if del_v_sl:
            print('\t\tUpdating GraphGT...')
            graph_gtt = ps.graph.GraphGT(graph)
            graph_gt = graph_gtt.get_gt()

        print('\t\tProcessing slice ' + sl.get_name() + ':')
        print('\t\tSegmentation label: ' + str(sl.get_seg()))
        print('\t\tMembrane: ' + str(sl.get_mb()))
        print('\t\t\t-Euclidean distance: (' + sl.get_eu_dst_sign() + ')[' \
              + str(sl.get_eu_dst_low()) + ', ' + str(sl.get_eu_dst_high()) + '] nm')
        print('\t\t\t-Geodesic distance: (' + sl.get_geo_dst_sign() + ')[' \
              + str(sl.get_geo_dst_low()) + ', ' + str(sl.get_geo_dst_high()) + '] nm')
        print('\t\t\t-Geodesic length: (' + sl.get_geo_len_sign() + ')[' \
              + str(sl.get_geo_len_low()) + ', ' + str(sl.get_geo_len_high()) + '] nm')
        print('\t\t\t-Sinuosity: (' + sl.get_sin_sign() + ')[' \
              + str(sl.get_sin_low()) + ', ' + str(sl.get_sin_high()) + '] nm')
        print('\t\t\t-Cluster number of points: (' + sl.get_cnv_sign() + ')[' \
              + str(sl.get_cnv_low()) + ', ' + str(sl.get_cnv_high()) + ']')
        for th in sl.get_list_th():
            print('\t\t\t\tVertices threshold: ' + th.get_name())
            print('\t\t\t\t\t-Property: ' + th.get_prop_key())
            print('\t\t\t\t\t-Mode: ' + th.get_mode())
            print('\t\t\t\t\t-Range: ' + str(th.get_range()) + ' %')
	try:
	    if sl.get_cont():
                print('\t\t\t-Contact points mode active.')
                cloud, cloud_ids, mask, cloud_w = graph.get_cloud_mb_slice(sl, cont_mode=True, graph_gt=graph_gt,
                                                                           cont_prop=peak_prop)
                print('\t\t\t\tCurrent number of points: ' + str(len(cloud_ids)))
            else:
                cloud, cloud_ids, mask = graph.get_cloud_mb_slice(sl, cont_mode=False, graph_gt=graph_gt)
                print('\t\t\t\tCurrent number of points: ' + str(len(cloud_ids)))
        except ValueError:
            print('WARNING: no points found in the slice for pickle: ' + input_pkl)
            continue

        if peak_prop is not None:
            print('\t\tFiltering points with ' + peak_prop + ' and percentile ' + str(peak_th) + ' %')
            if sl.get_cont():
                cloud_cc = graph.get_prop_values(peak_prop, cloud_ids)
            else:
                # cloud_cc = cloud_w.tolist()
                cloud_cc = graph.get_prop_values(peak_prop, cloud_ids)
            hold_cloud, hold_cloud_ids, hold_cloud_cc = cloud, cloud_ids, cloud_cc
            cloud, cloud_ids, cloud_cc = list(), list(), list()
            per_th = np.percentile(hold_cloud_cc, peak_th)
            print('\t\t\t-Threshold found: ' + str(per_th))
            for (point, cloud_id, c_cc) in zip(hold_cloud, hold_cloud_ids, hold_cloud_cc):
                if c_cc >= per_th:
                    cloud.append(point)
                    cloud_ids.append(cloud_id)
                    cloud_cc.append(c_cc)
            print('\t\t\t-Peaks thresholded: ' + str(len(cloud)) + ' of ' + str(len(hold_cloud)))
            print('\t\tTemporal copy of current graph...')
            hold_graph = graph.gen_subgraph(cloud_ids)
            graph_gtt = ps.graph.GraphGT(hold_graph)
            print('\t\tScale suppresion...')
            h_cloud_ids, h_cloud, h_cloud_cc = list(), list(), list()
            if not sl.get_cont():
                del_ids = graph_gtt.vertex_scale_supression(peak_ns, peak_prop, peak_conn)
                for coord, cloud_id, c_cc in zip(cloud, cloud_ids, cloud_cc):
                    if not(cloud_id in del_ids):
                        h_cloud_ids.append(cloud_id)
                        h_cloud.append(coord)
                        h_cloud_cc.append(c_cc)
                print('\t\t\t-Peaks thresholded: ' + str(len(h_cloud)) + ' of ' + str(len(cloud)))
                cloud_ids, cloud = np.asarray(h_cloud_ids, dtype=np.int), np.asarray(h_cloud, dtype=np.float32)
            else:
                del_ids = coords_scale_supression(cloud, peak_ns/graph.get_resolution(), weights=cloud_cc)
                h_cloud_ids, h_cloud, h_cloud_cc = list(), list(), list()
                for i, coord in enumerate(cloud):
                    if not(i in del_ids):
                        h_cloud_ids.append(cloud_ids[i])
                        h_cloud.append(coord)
                        h_cloud_cc.append(cloud_cc[i])
                print('\t\t\t-Peaks thresholded: ' + str(len(h_cloud)) + ' of ' + str(len(cloud)))
                cloud_ids, cloud = np.asarray(h_cloud_ids, dtype=np.int), np.asarray(h_cloud, dtype=np.float32)
            cloud_cc = np.asarray(h_cloud_cc, dtype=np.float32)
        elif del_v_sl:
            graph.threshold_vertices_list(cloud_ids, in_mode=True)

        output_seg = stem_name + '_' + sl.get_name()
        if peak_prop is not None:
            print('\t\tCreating the peaks container...')
            tomo_peaks = TomoPeaks(shape=mask.shape, name=output_seg, mask=mask)
            tomo_peaks.add_peaks(cloud)
            tomo_peaks.add_prop(peak_prop, n_comp=1, vals=cloud_cc)
            print('\t\t\t-Number of peaks found: ' + str(tomo_peaks.get_num_peaks()))
            if tomo_peaks.get_num_peaks() == 0:
                print('WARNING: number of peaks for this slice is 0 no further analysis can be applied, slice skipped!')
                continue
            tomo_seg = None
            if sl.get_mb()==1:
                if sl.get_seg() == 3:
                     tomo_seg = tomo_f == 5
                elif sl.get_seg() == 5:
                     tomo_seg = tomo_f == 3
            elif sl.get_mb() == 2:
                if sl.get_seg() == 4:
                     tomo_seg = tomo_f == 5
                elif sl.get_seg() == 5:
                     tomo_seg = tomo_f == 4
            if tomo_seg is not None:
                tomo_peaks.seg_shortest_pt(tomo_seg, peak_prop_pt)
            set_tpeaks.add_tomo_peaks(tomo_peaks, in_tomo_ref, swap_xy=True, ctf=st_ctf)
        elif len(cloud_ids) == 0:
            print('WARNING: number of points for this slice is 0 no further analysis can be applied, slice skipped!')
            continue

        print('\t\tBuilding statistical analyzer...')
        print('\t\t\tCurrent number of points: ' + str(len(cloud_ids)))
        print('\t\t\tPrint mask size (n voxels): ' + str(mask.sum()))
        pkl_lbl = stem_name[:stem_name.rfind('_bin')]
        unisp = UniStat(cloud, mask, graph.get_resolution(), name=sl.get_name()+'_'+pkl_lbl)
        # unisp.save_dense(output_dir+'/hold.mrc')

        print('\t\t\t-Vertices found: ' + str(cloud.shape[0]))
        output_sp = stem_name + '_' + sl.get_name() + '_unisp.pkl'
        print('\t\t\t-Storing spatial analyzer as : ' + output_sp)
        output_sp = output_dir + '/' + output_sp
        unisp.pickle(output_sp)
        if (store_seg == 1) or (store_seg == 2):
            print('\t\t\t-Storing slices graph with name: ' + output_seg)
            print('\t\t\t\tCurrent number of points: ' + str(len(cloud_ids)))
            out_seg = output_dir + '/' + output_seg
            ps.disperse_io.save_vtp(graph.slice_to_vtp(cloud_ids, mb_id=sl.get_mb()), out_seg+'.vtp')
        if store_seg == 2:
            print('\t\t\t-Printing slices vertices with name: ' + output_seg)
            ps.disperse_io.save_numpy(graph.print_slice(cloud_ids, th_den=0, slc=True), out_seg+'.vti')

        if peak_prop is not None:
            out_vtp = output_dir + '/' + output_seg + '_peak.vtp'
            print('\t\t\t-Storing peaks (vtp): ' + out_vtp)
            tomo_peaks.vect_2pts(ps.sub.PK_COORDS, peak_prop_pt, peak_prop_norm)
            ps.disperse_io.save_vtp(tomo_peaks.to_vtp(), out_vtp)
            out_sub = output_dir + '/' + output_seg + '_sta'

            if in_tomo_ref:
                print('\t\t\t-Converting graph coordinate to reference tomogram coordinates...')
                out_vtp = output_dir + '/' + output_seg + '_ref_peak.vtp'
                ps.disperse_io.save_vtp(tomo_peaks.to_vtp(), out_vtp)
                tomo_ref = ps.disperse_io.load_tomo(in_tomo_ref, mmap=True)
                center = (tomo_ref.shape[0]*.5, tomo_ref.shape[1]*.5, tomo_ref.shape[2]*.5)
                tomo_peaks.peaks_prop_op(ps.sub.PK_COORDS, (in_off[0], in_off[1], in_off[2]), operator.add)
                tomo_peaks.rotate_coords(float(in_rot[0]), float(in_rot[1]), float(in_rot[2]), center=center, conv='relion')
                tomo_peaks.peaks_prop_op(peak_prop_pt, (in_off[0], in_off[1], in_off[2]), operator.add)
                tomo_peaks.rotate_coords(float(in_rot[0]), float(in_rot[1]), float(in_rot[2]), center=center,
                                         key=peak_prop_pt, conv='relion')
                tomo_peaks.vect_2pts(ps.sub.PK_COORDS, peak_prop_pt, peak_prop_norm)
                # tomo_peaks.vect_rotation_ref(key_v=peak_prop_norm, key_r=peak_prop_rot, v_ref=(0.,0.,1.), conv='relion',
                #                             key_vo=peak_prop_norm+'_ref')
                tomo_peaks.vect_rotation_zrelion(key_v=peak_prop_norm, key_r=peak_prop_rot)
                if peak_off is not None:
                    tomo_peaks.norm_offset(ps.sub.PK_COORDS, peak_prop_norm, peak_off/graph.get_resolution())
                out_vtp = output_dir + '/' + output_seg + '_ref_unrot_peak.vtp'
                ps.disperse_io.save_vtp(tomo_peaks.to_vtp(), out_vtp)
                tref_stem = os.path.split(in_tomo_ref)[1]
                out_coords = output_dir + '/' + os.path.splitext(tref_stem)[0] + '.coords'
        tomo_peaks.save_coords(out_coords, swap_xy=True, add_prop=peak_prop_rot, fmode='a')
    gc.collect()

if peak_prop is not None:
    out_star = output_dir + '/' + peak_prop + '.star'
    print('\tStoring peaks STAR file and subvolumes in directory: ' + out_star)
    star = set_tpeaks.gen_star(n_key=peak_prop_rot)
    if st_mask is not None:
        st_mask = ps.disperse_io.load_tomo(st_mask)
    star.store(out_star, sv=sv_shape, mask=st_mask, swap_xy=False, del_ang=(0,0,0))
    print('\t\t-Number of particles found: ' + str(star.get_nrows()))

print('Terminated. (' + time.strftime("%c") + ')')
