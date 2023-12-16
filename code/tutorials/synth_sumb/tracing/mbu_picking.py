"""

    Script for extracting particles from a filament network of a unoriented single membrane graph

    Input:  - A STAR file with a list of graphs (MbGraphMCF) as it is returned by
              by mb_fils_network.py script:
            	+ Density map tomogram
            	+ Segmentation tomogram
                + MbGraphMCF object
            - XML file for point selection on filaments
            - Picking settings

    Output: - A STAR file and a list with the coordinates pickled
            - Additional files for visualization

"""

__author__ = 'Antonio Martinez-Sanchez'

import argparse
import operator
import pyseg as ps
from pyseg.globals.utils import coords_scale_supression
import os

########################################################################################
# PARAMETERS
########################################################################################


#ROOT_PATH = '/scratch/users/muth9/simsiam/particle_picking'

## Input STAR file
#in_star = ROOT_PATH + '/data/fils/fil_mb_sources_to_no_mb_targets_net.star'

####### Output data

output_dir = 'pick/out'  # ROOT_PATH + '/data/pick'

####### Slices settings file

#slices_file = ROOT_PATH + '/data/pick/mb_cont_1.xml'

#Sarahs addition
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inStar', help='Input star file.', required=True)
parser.add_argument('--inSlices', help='Input filament slices file.', required=True)
parser.add_argument('--outDir', default=output_dir, help='Output subtomograms directory.')

args = parser.parse_args()

in_star = args.inStar
output_dir = args.outDir
slices_file = args.inSlices  # ROOT_PATH + '/data/pick/mb_cont_1.xml'

###### Peaks configuration

peak_th = 0 # Percentile %
peak_ns = 0.5 # 5 # nm

###### Advanced peaks configuration

peak_th_op = operator.ge
peak_conn = False
peak_prop = 'field_value_inv' # 'field_value_eq_inv'
peak_prop_pt = 'pt_normal'
peak_prop_norm = 'smb_normal'
peak_prop_rot = 'norm_rot'
peak_dst = ps.globals.STR_VERT_DST

del_v_sl = True # if True vertices are being deleted from graph as they detected in a slice

###### Advanced normal configuration

norm_sg = 2 # sigma for gaussian smoothing
norm_th = 0.45 # iso-surface threshold
norm_proj = True # project the picked point on the surface for the normal
norm_ow = True # normals pointing inwards (false) or outwards (true); ribo==True
norm_shift = 0 # (px) amount of coordinate shift along normal (15px a 2.62A => 3px a 10,48A)

del_v_sl = True # if True vertices are being deleted from graph as they detected in a slice

###### Advanced mean shift clustering

ms_bg = None # 10 # nm (if None then Mean Shift is not applied)
ms_clst_all = True

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import gc
import csv
import time
import operator
import numpy as np
from pyseg.xml_io import SliceSet
from pyseg.spatial import UniStat
from pyseg.sub import TomoPeaks, SetTomoPeaks
import sys

########## Global variables

########## Print initial message

print(f'{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Extracting slices from synapses.', file=sys.stdout, flush=True)
print('\tAuthor: ' + __author__, file=sys.stdout, flush=True)
print('\tDate: ' + time.strftime("%c") + '\n', file=sys.stdout, flush=True)
print('Options:', file=sys.stdout, flush=True)
print('\tSTAR file with the segmentations: ' + str(in_star), file=sys.stdout, flush=True)
# print '\tCTF model file path: ' + str(in_ctf)
print('\tOutput directory: ' + str(output_dir), file=sys.stdout, flush=True)
print('\tSlices file: ' + slices_file, file=sys.stdout, flush=True)
if del_v_sl:
    print('\tProgressive vertex deleting active.', file=sys.stdout, flush=True)
if peak_prop is not None:
    print('\tPeaks detection active, settings:', file=sys.stdout, flush=True)
    print('\t\t-Vertices property: ' + peak_prop, file=sys.stdout, flush=True)
    print('\t\t\t+Threshold (percentile): ' + str(peak_th) + ' %', file=sys.stdout, flush=True)
    print('\t\t\t+Operation for the threshold: ' + str(peak_th_op), file=sys.stdout, flush=True)
    print('\t\t-Edge property for distance measuring: ' + peak_dst, file=sys.stdout, flush=True)
    print('\t\t\t+Neighbourhood radius: ' + str(peak_ns), file=sys.stdout, flush=True)
    if peak_conn:
        print('\t\t\t-Peak connectivity active.', file=sys.stdout, flush=True)
print('', file=sys.stdout, flush=True)

######### Process

print(f'\tParing input STAR files...', file=sys.stdout, flush=True)
star = ps.sub.Star()
star.load(in_star)
in_tomo_refs, in_seg_l, in_img_l = star.get_column_data('_rlnMicrographName'), star.get_column_data('_psSegImage'),\
                                   star.get_column_data('_rlnImageName')
input_pkl_l = star.get_column_data('_psGhMCFPickle')
in_offx_l, in_offy_l, in_offz_l = star.get_column_data('_psSegOffX'), star.get_column_data('_psSegOffY'), \
                                  star.get_column_data('_psSegOffZ')
in_rot_l, in_tilt_l, in_psi_l = star.get_column_data('_psSegRot'), star.get_column_data('_psSegTilt'), \
                                  star.get_column_data('_psSegPsi')
in_offs, in_rots = list(), list()
for(offx, offy, offz, rot, tilt, psi) in zip(in_offx_l, in_offy_l, in_offz_l, in_rot_l, in_tilt_l, in_psi_l):
    in_offs.append((offx, offy, offz))
    in_rots.append((rot, tilt, psi))

print(f'Creating IMOD CSV and RELION COORDS files for every tomogram', file=sys.stdout, flush=True)
if not os.path.isdir(output_dir) : os.makedirs(output_dir)
out_dir_files = os.listdir(output_dir)
for out_dir_file in out_dir_files:
    if out_dir_file.endswith('_imod.csv') or out_dir_file.endswith('_rln.coords'):
        os.remove(os.path.join(output_dir, out_dir_file))
for in_mic in in_tomo_refs:
    stem_mic = os.path.splitext(os.path.split(in_mic)[1])[0]
    out_imod_csv = output_dir + '/' + stem_mic + '_imod.csv'
    if not os.path.exists(out_imod_csv):
        print(f'\t-Creating output IMOD CSV file: ' + out_imod_csv, file=sys.stdout, flush=True)
        with open(out_imod_csv, 'w') as imod_csv_file:
            writer = csv.DictWriter(imod_csv_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z'))
    out_rln_coords = output_dir + '/' + stem_mic + '_rln.coords'
    if not os.path.exists(out_rln_coords):
        print(f'\t-Creating output RELION COORDS file: ' + out_rln_coords, file=sys.stdout, flush=True)
        with open(out_rln_coords, 'w') as rln_coords_file:
            writer = csv.DictWriter(rln_coords_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z', 'Rho', 'Tilt', 'Psi'))

print(f'\t-Loading XML file with the slices...', file=sys.stdout, flush=True)
slices = SliceSet(slices_file)

print(f'\tPreparing particles STAR file...', file=sys.stdout, flush=True)
star_parts = ps.sub.Star()
star_parts.add_column('_rlnMicrographName')
star_parts.add_column('_rlnImageName')
star_parts.add_column('_psSegImage')
star_parts.add_column('_rlnCoordinateX')
star_parts.add_column('_rlnCoordinateY')
star_parts.add_column('_rlnCoordinateZ')
star_parts.add_column('_rlnAngleRot')
star_parts.add_column('_rlnAngleTilt')
star_parts.add_column('_rlnAnglePsi')
part_row = 0

set_tpeaks = SetTomoPeaks()
print(f'\tTomograms loop:', file=sys.stdout, flush=True)
for (input_pkl, in_tomo_ref, in_seg, in_img, in_off, in_rot) in \
        zip(input_pkl_l, in_tomo_refs, in_seg_l, in_img_l, in_offs, in_rots):

    print('\t\tLoading the input graph: ' + input_pkl, file=sys.stdout, flush=True)

    print('\t\tUnpicking graph...', file=sys.stdout, flush=True)
    path, fname = os.path.split(input_pkl)
    stem_name, _ = os.path.splitext(fname)
    graph = ps.factory.unpickle_obj(input_pkl)
    tomo_f = ps.disperse_io.load_tomo(in_seg)

    if not del_v_sl:
        print('\t\tUpdating GraphGT...', file=sys.stdout, flush=True)
        graph.compute_graph_gt()
        graph_gt = graph.get_gt(fupdate=True)

    print('\t\tSlices loop (' + str(slices.get_num_slices()) + ' slices found):', file=sys.stdout, flush=True)
    for sl in slices.get_slices_list():

        if del_v_sl:
            print('\t\tUpdating GraphGT...', file=sys.stdout, flush=True)
            graph_gtt = ps.graph.GraphGT(graph)
            graph_gt = graph_gtt.get_gt()

        print('\t\tProcessing slice ' + sl.get_name() + ':', file=sys.stdout, flush=True)
        print('\t\tMembrane side: ' + str(sl.get_side()), file=sys.stdout, flush=True)
        print('\t\t\t-Euclidean distance: (' + sl.get_eu_dst_sign() + ')[' \
              + str(sl.get_eu_dst_low()) + ', ' + str(sl.get_eu_dst_high()) + '] nm', file=sys.stdout, flush=True)
        print('\t\t\t-Geodesic distance: (' + sl.get_geo_dst_sign() + ')[' \
              + str(sl.get_geo_dst_low()) + ', ' + str(sl.get_geo_dst_high()) + '] nm', file=sys.stdout, flush=True)
        print('\t\t\t-Geodesic length: (' + sl.get_geo_len_sign() + ')[' \
              + str(sl.get_geo_len_low()) + ', ' + str(sl.get_geo_len_high()) + '] nm', file=sys.stdout, flush=True)
        print('\t\t\t-Sinuosity: (' + sl.get_sin_sign() + ')[' \
              + str(sl.get_sin_low()) + ', ' + str(sl.get_sin_high()) + '] nm', file=sys.stdout, flush=True)
        print('\t\t\t-Cluster number of points: (' + sl.get_cnv_sign() + ')[' \
              + str(sl.get_cnv_low()) + ', ' + str(sl.get_cnv_high()) + ']', file=sys.stdout, flush=True)
        for th in sl.get_list_th():
            print('\t\t\t\tVertices threshold: ' + th.get_name(), file=sys.stdout, flush=True)
            print('\t\t\t\t\t-Property: ' + th.get_prop_key(), file=sys.stdout, flush=True)
            print('\t\t\t\t\t-Mode: ' + th.get_mode(), file=sys.stdout, flush=True)
            print('\t\t\t\t\t-Range: ' + str(th.get_range()) + ' %', file=sys.stdout, flush=True)
        try:
            if sl.get_cont():
                print('\t\t\t-Contact points mode active.', file=sys.stdout, flush=True)
                cloud, cloud_ids, mask, cloud_w = graph.get_cloud_mb_slice_pick(sl, cont_mode=1, graph_gt=graph_gt,
                                                                           cont_prop=peak_prop)
                print('\t\t\t\tCurrent number of points: ' + str(len(cloud_ids)), file=sys.stdout, flush=True)
            else:
                cloud, cloud_ids, mask = graph.get_cloud_mb_slice_pick(sl, cont_mode=0, graph_gt=graph_gt)
                print('\t\t\t\tCurrent number of points: ' + str(len(cloud_ids)), file=sys.stdout, flush=True)
        except ValueError:
            print('WARNING: no points found in the slice for pickle: ' + input_pkl, file=sys.stdout, flush=True)
            continue

        if peak_prop is not None:
            print('\t\tFiltering points with ' + peak_prop + ' and percentile ' + str(peak_th) + ' %', file=sys.stdout, flush=True)
            if sl.get_cont():
                cloud_cc = graph.get_prop_values(peak_prop, cloud_ids)
            else:
                # cloud_cc = cloud_w.tolist()
                cloud_cc = graph.get_prop_values(peak_prop, cloud_ids)
            hold_cloud, hold_cloud_ids, hold_cloud_cc = cloud, cloud_ids, cloud_cc
            cloud, cloud_ids, cloud_cc = list(), list(), list()
            per_th = np.percentile(hold_cloud_cc, peak_th)
            print('\t\t\t-Threshold found: ' + str(per_th), file=sys.stdout, flush=True)
            for (point, cloud_id, c_cc) in zip(hold_cloud, hold_cloud_ids, hold_cloud_cc):
                if peak_th_op(c_cc, per_th):
                    cloud.append(point)
                    cloud_ids.append(cloud_id)
                    cloud_cc.append(c_cc)
            print('\t\t\t-Peaks thresholded: ' + str(len(cloud)) + ' of ' + str(len(hold_cloud)), file=sys.stdout, flush=True)
            print(f'\t\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Temporal copy of current graph...', file=sys.stdout, flush=True)
            hold_graph = graph.gen_subgraph(cloud_ids)
            graph_gtt = ps.graph.GraphGT(hold_graph)
            print(f'\t\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Scale suppresion...', file=sys.stdout, flush=True)
            h_cloud_ids, h_cloud, h_cloud_cc = list(), list(), list()
            if not sl.get_cont():
                del_ids = graph_gtt.vertex_scale_supression(peak_ns, peak_prop, peak_conn)
                for coord, cloud_id, c_cc in zip(cloud, cloud_ids, cloud_cc):
                    if not(cloud_id in del_ids):
                        h_cloud_ids.append(cloud_id)
                        h_cloud.append(coord)
                        h_cloud_cc.append(c_cc)
                print('\t\t\t-Peaks thresholded: ' + str(len(h_cloud)) + ' of ' + str(len(cloud)), file=sys.stdout, flush=True)
                cloud_ids, cloud = np.asarray(h_cloud_ids, dtype=int), np.asarray(h_cloud, dtype=np.float32)
            else:
                del_ids = coords_scale_supression(cloud, peak_ns/graph.get_resolution(), weights=cloud_cc)
                h_cloud_ids, h_cloud, h_cloud_cc = list(), list(), list()
                for i, coord in enumerate(cloud):
                    if not(i in del_ids):
                        h_cloud_ids.append(cloud_ids[i])
                        h_cloud.append(coord)
                        h_cloud_cc.append(cloud_cc[i])
                print('\t\t\t-Peaks thresholded: ' + str(len(h_cloud)) + ' of ' + str(len(cloud)), file=sys.stdout, flush=True)
                cloud_ids, cloud = np.asarray(h_cloud_ids, dtype=int), np.asarray(h_cloud, dtype=np.float32)
            cloud_cc = np.asarray(h_cloud_cc, dtype=np.float32)
        elif del_v_sl:
            graph.threshold_vertices_list(cloud_ids, in_mode=True)

        output_seg = stem_name + '_' + sl.get_name()
        if peak_prop is not None:
            print(f'\t\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Creating the peaks container...', file=sys.stdout, flush=True)
            tomo_peaks = TomoPeaks(shape=mask.shape, name=output_seg, mask=mask)
            tomo_peaks.add_peaks(cloud)
            tomo_peaks.add_prop(peak_prop, n_comp=1, vals=cloud_cc)
            print('\t\t\t-Number of peaks found: ' + str(tomo_peaks.get_num_peaks()), file=sys.stdout, flush=True)
            if tomo_peaks.get_num_peaks() == 0:
                print('WARNING: number of peaks for this slice is 0 no further analysis can be applied, slice skipped!', file=sys.stdout, flush=True)
                continue
        elif len(cloud_ids) == 0:
            print('WARNING: number of points for this slice is 0 no further analysis can be applied, slice skipped!', file=sys.stdout, flush=True)
            continue

        print('\t\t\t-Applying the Mean Shift...', file=sys.stdout, flush=True)
        if ms_bg is not None:
            tomo_peaks = tomo_peaks.cluster_mean_shift(ms_bg/graph.get_resolution(), ms_clst_all)

        print('\t\t\t-Computing the normals...', file=sys.stdout, flush=True)
        lbl_norm = 1
        tomo_seg = tomo_f == lbl_norm
        if norm_sg > 0:
            surf = ps.disperse_io.tomo_smooth_surf(tomo_seg, norm_sg, norm_th)
            tomo_peaks.get_normals_from_cubes(surf, peak_prop_pt, mode='point', outwards=norm_ow, proj=norm_proj)
            if norm_shift > 0:
                tomo_peaks.shift_coordinates_along_vector(ps.sub.PK_COORDS, norm_shift)
                tomo_peaks.shift_coordinates_along_vector(peak_prop_pt, norm_shift)
            out_seg_surf = output_dir + '/' + output_seg + '_surf.vtp'
            ps.disperse_io.save_vtp(surf, out_seg_surf)
        else:
            tomo_peaks.seg_shortest_pt(tomo_seg, peak_prop_pt)
        set_tpeaks.add_tomo_peaks(tomo_peaks, in_tomo_ref, swap_xy=True)

        print('\t\t\t-Vertices found: ' + str(cloud.shape[0]), file=sys.stdout, flush=True)
        print('\t\t\t-Storing slice graph with name:' + output_seg, file=sys.stdout, flush=True)
        print('\t\t\t\tCurrent number of points: ' + str(len(cloud_ids)), file=sys.stdout, flush=True)
        out_seg = output_dir + '/' + output_seg
        ps.disperse_io.save_vtp(graph.slice_to_vtp(cloud_ids), out_seg+'.vtp')
        tomo_peaks.vect_2pts(ps.sub.PK_COORDS, peak_prop_pt, peak_prop_norm)
        ps.disperse_io.save_vtp(tomo_peaks.to_vtp(), out_seg+'_peak.vtp')

        print(f'\t\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Particles loop..', file=sys.stdout, flush=True)
        part_seg_row = 1
        gcrop_off = in_off[0], in_off[1], in_off[2]
        gcrop_off_rln = np.asarray((gcrop_off[1], gcrop_off[0], gcrop_off[2]), dtype=np.float32)
        coords, coords_pt = tomo_peaks.get_prop_vals(ps.sub.PK_COORDS), tomo_peaks.get_prop_vals(peak_prop_pt)
        for coord, pt_coord in zip(coords, coords_pt):

            # Initialization
            vec_tlist = pt_coord - coord
            rho, tilt, psi = ps.globals.vect_to_zrelion(np.asarray((vec_tlist[1], vec_tlist[0], vec_tlist[2]),
                                                        dtype=np.float32), mode='passive')

            # Coordinate transformation for IMOD
            coord_imod, pt_coord_imod = coord + gcrop_off, pt_coord + gcrop_off
            vec_imod = pt_coord_imod - coord_imod
            stem_mic = os.path.splitext(os.path.split(in_tomo_ref)[1])[0]
            out_imod_csv = output_dir + '/' + stem_mic + '_imod.csv'
            with open(out_imod_csv, 'a') as imod_csv_file:
                writer = csv.DictWriter(imod_csv_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z'))
                writer.writerow({'X':coord_imod[0], 'Y':coord_imod[1], 'Z':coord_imod[2]})

            # Coordinate transformation for RELION
            coord_rln = np.asarray((coord[1], coord[0], coord[2]), dtype=np.float32)
            pt_coord_rln = np.asarray((pt_coord[1], pt_coord[0], pt_coord[2]), dtype=np.float32)
            coord_rln, pt_coord_rln = coord_rln + gcrop_off_rln, pt_coord_rln + gcrop_off_rln
            vec_imod_rln = pt_coord_rln - coord_rln
            rho, tilt, psi = ps.globals.vect_to_zrelion((-1.) * vec_imod_rln)
            out_rln_coords = output_dir + '/' + stem_mic + '_rln.coords'
            with open(out_rln_coords, 'a') as rln_coords_file:
                writer = csv.DictWriter(rln_coords_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z', 'Rho', 'Tilt', 'Psi'))
                writer.writerow({'X':coord_imod[0], 'Y':coord_imod[1], 'Z':coord_imod[2], 'Rho':rho, 'Tilt':tilt, 'Psi':psi})
            star_row = {'_rlnMicrographName':in_tomo_ref, '_rlnImageName':in_img, '_psSegImage':in_seg,
                        '_rlnCoordinateX':coord_imod[0], '_rlnCoordinateY':coord_imod[1], '_rlnCoordinateZ':coord_imod[2],
                        '_rlnAngleRot':rho, '_rlnAngleTilt':tilt, '_rlnAnglePsi':psi}
            star_parts.add_row(**star_row)
            part_row += 1
            part_seg_row += 1

    gc.collect()

out_star = output_dir + '/' + os.path.splitext(os.path.split(in_star)[1])[0] + '_parts.star'
print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Storing particles STAR file in: ' + out_star, file=sys.stdout, flush=True)
star_parts.store(out_star)

print(f'Terminated {os.path.basename(__file__)}, ' + time.strftime("%c"), file=sys.stdout, flush=True)
