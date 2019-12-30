"""

    Script for extracting particles randomly within a segmented membrane

    Input:  - Directory with the _imod.csv files with reference picked particles for each microsome
            - Directory with offsets for each microsome in a tomogram

    Output: - A STAR file and a list with the coordinates pickled per microsome
            - Additional files for visualization

"""

__author__ = 'Antonio Martinez-Sanchez'

import pyseg as ps

########################################################################################
# GLOBAL VARIABLES
########################################################################################

MB_LBL = 1

########################################################################################
# PARAMETERS
########################################################################################


ROOT_PATH = '/fs/pool/pool-lucic2'

# Input STAR files
in_dir = ROOT_PATH + '/antonio/pick_test/cont/160614' # '/johannes/tomograms/stefan/090614/filaments_extract_protein_clean' # '/johannes/tomograms/stefan/090614/reconstruct/coords'
in_dir_seg = ROOT_PATH + '/johannes/tomograms/stefan/160614/graph_ribo' # '/johannes/tomograms/stefan/090614/graph_ribo'
in_ctf = ROOT_PATH + '/johannes/tomograms/stefan/ctf/CTF_model_160.mrc'

####### Output data

out_dir = ROOT_PATH + '/antonio/pick_test/pick/rnd/160614' # '/antonio/pick_test/pick/rnd/090614'
out_dir_parts = ROOT_PATH + '/antonio/pick_test/recontruct/rnd/160614/protein_160'

###### Advanced peaks configuration

peak_side = 2 # 3
peak_prop_pt = 'pt_normal'
peak_prop_norm = 'smb_normal'
peak_prop_rot = 'norm_rot'

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
import math
import numpy as np
from pyseg.sub import TomoPeaks
from pyorg.surf import points_to_poly

########## Global variables

########## Print initial message

print 'Extracting particles randomly within segmented membranes.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tDirectory for reference picked particles: ' + str(in_dir)
print '\tDirectory for microsomes offset per tomogram: ' + str(in_dir_seg)
print '\tCTF model file path: ' + str(in_ctf)
print '\tOutput directory: ' + str(out_dir)
print ''

######### Process

print '\tSearching for *_imod.csv files in input folder...'
ref_files = list()
for fname in os.listdir(in_dir):
    if fname.endswith('_imod.csv'):
        ref_files.append(os.path.split(fname)[1])
        print '\t\t-File found: ' + ref_files[-1]

print '\tPairing files found with their segmentation offset...'
mic_dic = dict()
for fname in ref_files:
    names = fname.split('_')
    fname_off = in_dir_seg + '/' + names[0] + '_' + names[1] + '_mb_graph.star'
    ves_id = names[3]
    print '\t\t-Tomogram offset found: ' + fname_off
    star = ps.sub.Star()
    star.load(fname_off)
    for row in range(star.get_nrows()):
        path_seg = star.get_element('_psSegImage', row)
        path_ref = star.get_element('_rlnMicrographName', row)
        fname_seg = os.path.split(path_seg)[1]
        names_seg = fname_seg.split('_')
        if ves_id == names_seg[3]:
            offx = star.get_element('_psSegOffX', row)
            offy = star.get_element('_psSegOffY', row)
            offz = star.get_element('_psSegOffZ', row)
            n_parts = sum(1 for line in open(in_dir + '/' + fname))
            mic_dic[fname] = (path_ref, path_seg, offx, offy, offz, n_parts)
            print '\t\t\t-Offset found for microsome ' + fname + ' with ' + str(n_parts) + ' particles.'

print '\tPreparing output particles STAR file...'
star_parts = ps.sub.Star()
star_parts.add_column('_rlnMicrographName')
star_parts.add_column('_rlnImageName')
star_parts.add_column('_rlnCtfImage')
star_parts.add_column('_rlnCoordinateX')
star_parts.add_column('_rlnCoordinateY')
star_parts.add_column('_rlnCoordinateZ')
star_parts.add_column('_rlnAngleRot')
star_parts.add_column('_rlnAngleTilt')
star_parts.add_column('_rlnAnglePsi')
part_row = 0

print '\tPICKING LOOP:'
for in_mic in mic_dic.iterkeys():

    path_seg, path_ref = mic_dic[in_mic][1], mic_dic[in_mic][0]
    print '\t-Loading microsome segmentation file: ' + path_seg
    seg = ps.disperse_io.load_tomo(path_seg, mmap=True)
    mb_mask, seg_mask = seg == MB_LBL, seg == peak_side
    del seg
    n_samp = mic_dic[in_mic][5]
    mb_ids = np.where(mb_mask)
    if len(mb_ids[0]) > 1:
        ids_rnd = np.random.randint(0, len(mb_ids[0])-1, n_samp)
    else:
        ids_rnd = np.asarray((0,), dtype=np.int)

    print '\t-Random sampling with the membrane...'
    coords = list()
    for id_rnd in ids_rnd:
        x_rnd, y_rnd, z_rnd = mb_ids[0][id_rnd], mb_ids[1][id_rnd], mb_ids[2][id_rnd]
        coords.append((x_rnd, y_rnd, z_rnd))

    print '\t-Creating the peaks container...'
    names = (os.path.splitext(os.path.split(path_seg)[1])[0]).split('_')
    stem_mic = names[0] + '_' + names[1] + '_' + names[2] + '_' + names[3]
    out_seg = out_dir + '/' + stem_mic + '_mb.mrc'
    tomo_peaks = TomoPeaks(shape=mb_mask.shape, name=out_seg, mask=mb_mask)
    tomo_peaks.add_peaks(coords)
    if norm_sg > 0:
        surf = ps.disperse_io.tomo_smooth_surf(seg_mask, norm_sg, norm_th)
        tomo_peaks.get_normals_from_cubes(surf, peak_prop_pt, mode='point', outwards=norm_ow)
        if norm_shift > 0:
            tomo_peaks.shift_coordinates_along_vector(ps.sub.PK_COORDS, norm_shift)
            tomo_peaks.shift_coordinates_along_vector(peak_prop_pt, norm_shift)
        out_seg_surf = out_dir + '/' + os.path.splitext(os.path.split(path_seg)[1])[0] + '_surf.vtp'
        ps.disperse_io.save_vtp(surf, out_seg_surf)
    else:
        tomo_peaks.seg_shortest_pt(seg_mask, peak_prop_pt)
    print '\t\t\t-Number of peaks found: ' + str(tomo_peaks.get_num_peaks())

    out_imod_csv = out_dir + '/' + stem_mic + '_rnd_imod.csv'
    if not os.path.exists(out_imod_csv):
        print '\t-Creating output IMOD CSV file: ' + out_imod_csv
        with open(out_imod_csv, 'w') as imod_csv_file:
            writer = csv.DictWriter(imod_csv_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z'))
    out_rln_coords = out_dir + '/' + stem_mic + '_rnd_rln.coords'
    if not os.path.exists(out_rln_coords):
        print '\t-Creating output RELION COORDS file: ' + out_rln_coords
        with open(out_rln_coords, 'w') as rln_coords_file:
            writer = csv.DictWriter(rln_coords_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z', 'Rho', 'Tilt', 'Psi'))

    print '\t\tParticles loop..'
    part_seg_row = 1
    coords_noff, normals_imod = list(), list()
    gcrop_off = mic_dic[in_mic][2], mic_dic[in_mic][3], mic_dic[in_mic][4]
    gcrop_off_rln = np.asarray((gcrop_off[1], gcrop_off[0], gcrop_off[2]), dtype=np.float32)
    coords, coords_pt = tomo_peaks.get_prop_vals(ps.sub.PK_COORDS), tomo_peaks.get_prop_vals(peak_prop_pt)
    for coord, pt_coord in zip(coords, coords_pt):

        # Coordinate transformation for IMOD
        coords_noff.append(coord)
        coord_imod, pt_coord_imod = coord + gcrop_off, pt_coord + gcrop_off
        vec_imod = pt_coord_imod - coord_imod
        hold_norm = math.sqrt((vec_imod * vec_imod).sum())
        if hold_norm <= 0:
            vec_imod = np.asarray((0., 0., 0.))
        else:
            vec_imod /= hold_norm
        normals_imod.append(vec_imod)
        out_imod_csv = out_dir + '/' + stem_mic + '_rnd_imod.csv'
        with open(out_imod_csv, 'a') as imod_csv_file:
            writer = csv.DictWriter(imod_csv_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z'))
            writer.writerow({'X':coord_imod[0], 'Y':coord_imod[1], 'Z':coord_imod[2]})

        # Coordinate transformation for RELION
        coord_rln = np.asarray((coord[1], coord[0], coord[2]), dtype=np.float32)
        pt_coord_rln = np.asarray((pt_coord[1], pt_coord[0], pt_coord[2]), dtype=np.float32)
        coord_rln, pt_coord_rln = coord_rln + gcrop_off_rln, pt_coord_rln + gcrop_off_rln
        vec_rln = pt_coord_rln - coord_rln
        # hold_norm = math.sqrt((vec_rln * vec_rln).sum())
        # if hold_norm <= 0:
        #     vec_rln = np.asarray((0., 0., 0.))
        # else:
        #     vec_rln /= hold_norm
        rho, tilt, psi = ps.globals.vect_to_zrelion(vec_rln)
        out_rln_coords = out_dir + '/' + stem_mic + '_rnd_rln.coords'
        with open(out_rln_coords, 'a') as rln_coords_file:
            writer = csv.DictWriter(rln_coords_file, dialect=csv.excel_tab, fieldnames=('X', 'Y', 'Z', 'Rho', 'Tilt', 'Psi'))
            writer.writerow({'X':coord_rln[0], 'Y':coord_rln[1], 'Z':coord_rln[2], 'Rho':rho, 'Tilt':tilt, 'Psi':psi})
        part_path = out_dir_parts + '/' + stem_mic + '_' + str(part_seg_row) + '.mrc'
        star_row = {'_rlnMicrographName':path_seg, '_rlnImageName':part_path, '_rlnCtfImage':in_ctf,
                    '_rlnCoordinateX':coord_rln[0], '_rlnCoordinateY':coord_rln[1], '_rlnCoordinateZ':coord_rln[2],
                    '_rlnAngleRot':rho, '_rlnAngleTilt':tilt, '_rlnAnglePsi':psi}
        star_parts.add_row(**star_row)
        part_row += 1
        part_seg_row += 1

    out_vtp = out_dir + '/' + stem_mic + '.vtp'
    print '\t\t-Storing the vtp file: ' + out_vtp
    coords_vtp = points_to_poly(coords_noff, normals=normals_imod, n_name='n_normal')
    ps.disperse_io.save_vtp(coords_vtp, out_vtp)

    gc.collect()

print '\tNumber of particles found: ' + str(part_row)
out_star = out_dir + '/particles_160614_rnd.star'
print '\tStoring particles STAR file in: ' + out_star
star_parts.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'