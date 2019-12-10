"""

    Script for extracting particles from a filament network in a single membrane graph

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

import operator
import pyseg as ps
from pyseg.globals.utils import coords_scale_supression

########################################################################################
# PARAMETERS
########################################################################################


ROOT_PATH = '/fs/pool/pool-lucic2/in_situ_er'

# Input STAR file
in_star = ROOT_PATH + '/mb_fil/out/fil_mb_sources_to_lumen_targets_net.star'
# in_ctf = ROOT_PATH + '/antonio/johannes/tomograms/stefan/ctf/CTF_model_160.mrc'

####### Output data

output_dir = ROOT_PATH + '/mb_cont/out' # '/pick/cyto'
# out_dir_parts = ROOT_PATH + '/pick_test/recontruct/ps/160614/protein_160'

###### Slices settings file

slices_file = ROOT_PATH + '/mb_cont/in/mb_lumen_cont.xml' # '/in/cyto/mb_cyto_cont.xml'

###### Peaks configuration

peak_th = 0 # Percentile %
peak_ns = 5 # nm

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
norm_ow = True # normals pointing inwards (false) or outwards (true); ribo==True
norm_shift = 0 # (px) amount of coordinate shift along normal (15px a 2.62A => 3px a 10,48A)

del_v_sl = True # if True vertices are being deleted from graph as they detected in a slice


########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import gc
import csv
import time
import operator
import numpy as np
from pyseg.xml_io import SliceSet
from pyseg.spatial import UniStat
from pyseg.sub import TomoPeaks, SetTomoPeaks

########## Global variables

########## Print initial message

print 'Extracting slices from synapses.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tSTAR file with the segmentations: ' + str(in_star)
# print '\tCTF model file path: ' + str(in_ctf)
print '\tOutput directory: ' + str(output_dir)
print '\tSlices file: ' + slices_file
if del_v_sl:
    print '\tProgressive vertex deleting active.'
if peak_prop is not None:
    print '\tPeaks detection active, settings:'
    print '\t\t-Vertices property: ' + peak_prop
    print '\t\t\t+Threshold (percentile): ' + str(peak_th) + ' %'
    print '\t\t\t+Operation for the threshold: ' + str(peak_th_op)
    print '\t\t-Edge property for distance measuring: ' + peak_dst
    print '\t\t\t+Neighbourhood radius: ' + str(peak_ns)
    if peak_conn:
        print '\t\t\t-Peak connectivity active.'
print ''

######### Process

print '\tParing input STAR files...'
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

print 'Creating IMOD CSV and RELION COORDS files for every tomogram'
out_dir_files = os.listdir(output_dir)
for out_dir_file in out_dir_files:
    if out_dir_file.endswith('_imod.csv') or out_dir_file.endswith('_rln.coords'):
        os.remove(os.path.join(output_dir, out_dir_file))
for in_mic in in_tomo_refs:
    stem_mic = os.path.splitext(os.path.split(in_mic)[1])[0]
    out_imod_csv = output_dir + '/' + stem_mic + '_imod.csv'
    if not os.path.exists(out_imod_csv):
        print '\t-Creating output IMOD CSV file: ' + out_imod_csv
        with open(out_imod_csv, 'w') as imod_csv_file:
            writer = csv.DictWriter(imod_csv_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z'))
    out_rln_coords = output_dir + '/' + stem_mic + '_rln.coords'
    if not os.path.exists(out_rln_coords):
        print '\t-Creating output RELION COORDS file: ' + out_rln_coords
        with open(out_rln_coords, 'w') as rln_coords_file:
            writer = csv.DictWriter(rln_coords_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z', 'Rho', 'Tilt', 'Psi'))

print '\tLoading XML file with the slices...'
slices = SliceSet(slices_file)

# print '\tPreparing particles STAR file...'
# star_parts = ps.sub.Star()
# star_parts.add_column('_rlnMicrographName')
# star_parts.add_column('_rlnImageName')
# star_parts.add_column('_rlnCtfImage')
# star_parts.add_column('_rlnCoordinateX')
# star_parts.add_column('_rlnCoordinateY')
# star_parts.add_column('_rlnCoordinateZ')
# star_parts.add_column('_rlnAngleRot')
# star_parts.add_column('_rlnAngleTilt')
# star_parts.add_column('_rlnAnglePsi')
part_row = 0

set_tpeaks = SetTomoPeaks()
print '\tTomograms loop:'
for (input_pkl, in_tomo_ref, in_seg, in_off, in_rot) in \
        zip(input_pkl_l, in_tomo_refs, in_seg_l, in_offs, in_rots):

    print '\t\tLoading the input graph: ' + input_pkl

    print '\t\tUnpicking graph...'
    path, fname = os.path.split(input_pkl)
    stem_name, _ = os.path.splitext(fname)
    graph = ps.factory.unpickle_obj(input_pkl)
    tomo_f = ps.disperse_io.load_tomo(in_seg)

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
        print '\t\tMembrane side: ' + str(sl.get_side())
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
        try:
            if sl.get_cont():
                print '\t\t\t-Contact points mode active.'
                cloud, cloud_ids, mask, cloud_w = graph.get_cloud_mb_slice(sl, cont_mode=True, graph_gt=graph_gt,
                                                                           cont_prop=peak_prop)
                print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))
            else:
                cloud, cloud_ids, mask = graph.get_cloud_mb_slice(sl, cont_mode=False, graph_gt=graph_gt)
                print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))
        except ValueError:
            print 'WARNING: no points found in the slice for pickle: ' + input_pkl
            continue

        if peak_prop is not None:
            print '\t\tFiltering points with ' + peak_prop + ' and percentile ' + str(peak_th) + ' %'
            if sl.get_cont():
                cloud_cc = graph.get_prop_values(peak_prop, cloud_ids)
            else:
                # cloud_cc = cloud_w.tolist()
                cloud_cc = graph.get_prop_values(peak_prop, cloud_ids)
            hold_cloud, hold_cloud_ids, hold_cloud_cc = cloud, cloud_ids, cloud_cc
            cloud, cloud_ids, cloud_cc = list(), list(), list()
            per_th = np.percentile(hold_cloud_cc, peak_th)
            print '\t\t\t-Threshold found: ' + str(per_th)
            for (point, cloud_id, c_cc) in zip(hold_cloud, hold_cloud_ids, hold_cloud_cc):
                if peak_th_op(c_cc, per_th):
                    cloud.append(point)
                    cloud_ids.append(cloud_id)
                    cloud_cc.append(c_cc)
            print '\t\t\t-Peaks thresholded: ' + str(len(cloud)) + ' of ' + str(len(hold_cloud))
            print '\t\tTemporal copy of current graph...'
            hold_graph = graph.gen_subgraph(cloud_ids)
            graph_gtt = ps.graph.GraphGT(hold_graph)
            print '\t\tScale suppresion...'
            h_cloud_ids, h_cloud, h_cloud_cc = list(), list(), list()
            if not sl.get_cont():
                del_ids = graph_gtt.vertex_scale_supression(peak_ns, peak_prop, peak_conn)
                for coord, cloud_id, c_cc in zip(cloud, cloud_ids, cloud_cc):
                    if not(cloud_id in del_ids):
                        h_cloud_ids.append(cloud_id)
                        h_cloud.append(coord)
                        h_cloud_cc.append(c_cc)
                print '\t\t\t-Peaks thresholded: ' + str(len(h_cloud)) + ' of ' + str(len(cloud))
                cloud_ids, cloud = np.asarray(h_cloud_ids, dtype=np.int), np.asarray(h_cloud, dtype=np.float32)
            else:
                del_ids = coords_scale_supression(cloud, peak_ns/graph.get_resolution(), weights=cloud_cc)
                h_cloud_ids, h_cloud, h_cloud_cc = list(), list(), list()
                for i, coord in enumerate(cloud):
                    if not(i in del_ids):
                        h_cloud_ids.append(cloud_ids[i])
                        h_cloud.append(coord)
                        h_cloud_cc.append(cloud_cc[i])
                print '\t\t\t-Peaks thresholded: ' + str(len(h_cloud)) + ' of ' + str(len(cloud))
                cloud_ids, cloud = np.asarray(h_cloud_ids, dtype=np.int), np.asarray(h_cloud, dtype=np.float32)
            cloud_cc = np.asarray(h_cloud_cc, dtype=np.float32)
        elif del_v_sl:
            graph.threshold_vertices_list(cloud_ids, in_mode=True)

        output_seg = stem_name + '_' + sl.get_name()
        if peak_prop is not None:
            print '\t\tCreating the peaks container...'
            tomo_peaks = TomoPeaks(shape=mask.shape, name=output_seg, mask=mask)
            tomo_peaks.add_peaks(cloud)
            tomo_peaks.add_prop(peak_prop, n_comp=1, vals=cloud_cc)
            print '\t\t\t-Number of peaks found: ' + str(tomo_peaks.get_num_peaks())
            if tomo_peaks.get_num_peaks() == 0:
                print 'WARNING: number of peaks for this slice is 0 no further analysis can be applied, slice skipped!'
                continue
            lbl_norm = 1
            # if sl.get_side() == 2:
            #     lbl_norm = 3
            # elif sl.get_side() == 3:
            #     lbl_norm = 2
            tomo_seg = tomo_f == lbl_norm
            if norm_sg > 0:
                surf = ps.disperse_io.tomo_smooth_surf(tomo_seg, norm_sg, norm_th)
                tomo_peaks.get_normals_from_cubes(surf, peak_prop_pt, mode='point', outwards=norm_ow)
                if norm_shift > 0:
                    tomo_peaks.shift_coordinates_along_vector(ps.sub.PK_COORDS, norm_shift)
                    tomo_peaks.shift_coordinates_along_vector(peak_prop_pt, norm_shift)
                out_seg_surf = output_dir + '/' + output_seg + '_surf.vtp'
                ps.disperse_io.save_vtp(surf, out_seg_surf)
            else:
                tomo_peaks.seg_shortest_pt(tomo_seg, peak_prop_pt)
            set_tpeaks.add_tomo_peaks(tomo_peaks, in_tomo_ref, swap_xy=True)
        elif len(cloud_ids) == 0:
            print 'WARNING: number of points for this slice is 0 no further analysis can be applied, slice skipped!'
            continue

        print '\t\t\t-Vertices found: ' + str(cloud.shape[0])
        print '\t\t\t-Storing slice graph with name:' + output_seg
        print '\t\t\t\tCurrent number of points: ' + str(len(cloud_ids))
        out_seg = output_dir + '/' + output_seg
        ps.disperse_io.save_vtp(graph.slice_to_vtp(cloud_ids), out_seg+'.vtp')
        tomo_peaks.vect_2pts(ps.sub.PK_COORDS, peak_prop_pt, peak_prop_norm)
        ps.disperse_io.save_vtp(tomo_peaks.to_vtp(), out_seg+'_peak.vtp')

        print '\t\tParticles loop..'
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
            # star_row = {'_rlnMicrographName':in_tomo_ref, '_rlnImageName':'particle_'+str(part_row), '_rlnCoordinateX':coord_imod[0], '_rlnCoordinateY':coord_imod[1], '_rlnCoordinateZ':coord_imod[2], '_rlnAngleRot':rho, '_rlnAngleTilt':tilt, '_rlnAnglePsi':psi}
            # part_path = out_dir_parts + '/' + stem_mic + '_' + str(part_seg_row) + '.mrc'
            # star_row = {'_rlnMicrographName':in_tomo_ref, '_rlnImageName':part_path, '_rlnCtfImage':in_ctf,
            #         '_rlnCoordinateX':coord_rln[0], '_rlnCoordinateY':coord_rln[1], '_rlnCoordinateZ':coord_rln[2],
            #         '_rlnAngleRot':rho, '_rlnAngleTilt':tilt, '_rlnAnglePsi':psi}
            # star_parts.add_row(**star_row)
            part_row += 1
            part_seg_row += 1

    gc.collect()

# out_star = output_dir + '/' + os.path.splitext(os.path.split(in_star)[1])[0] + '_parts.star'
# print '\tStoring particles STAR file in: ' + out_star
# star_parts.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'