"""

    Clean close particles in a STAR file

    Input:  - STAR file
            - Minimum scale

    Output: - A copy of input STAR file where some rows can be filtered

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import numpy as np
import pyseg as ps

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext'

# Input star file (file to be filtered)
in_star = ROOT_PATH + '/whole/class_nali/run1_c1_ct_it092_data_k2.star'

# Input STAR file with references (if None auto mode activated)
in_ref = None # ROOT_PATH + '/whole/class_nali/run1_c1_ct_it092_data_k2_ssup8.star' # ROOT_PATH + '/ves/0_particles_ves_gap_14_cont.star' # '/ves_ap_10/ltomos_ves_ap_10_premb_mask/sph_rad_5_surf_parts.star'

# Input STAR file with pairing references psSegFile and rlnMicrographName
in_mic_ref = None # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/in/syn_seg_no_l14_gap.star'

# Output filtered star file (it can be the same as the filtered class)
out_star = ROOT_PATH + '/whole/class_nali/run1_c1_ct_it092_data_k2_ssup5.star'

# Scale supresion
min_dst = 2 # nm # 20 nm for vesicles
res = 0.262 # nm voxels

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import math
import sys
import time
import copy
from pyseg import disperse_io
from pyseg.globals import rot_mat_relion

########## Print initial message

print('Filtering STAR particles file with scale suppression.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tFile to filter: ' + in_star)
if in_ref is None:
    print('\tAuto mode activated.')
else:
    print('\tFile with references: ' + in_ref)
    if in_mic_ref is not None:
        print('\tTable with segmentations: ' + in_mic_ref)
print('\tOutput file: ' + out_star)
print('\tScale supresion settings: ')
min_dst = float(min_dst)
min_dst_v = min_dst / res
print('\t\t-Data resolution: ' + str(res) + ' nm/voxel')
print('\t\t-Minimum distance: ' + str(min_dst) + ' nm, ' + str(min_dst_v) + ' voxels')
print('')

######### Process

print('Main Routine: ')

print('\tLoading input STAR files...')
star = ps.sub.Star()
try:
    star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be read because of "' + str(e.msg, e.expr) + '"')
    sys.exit(-1)
star_out = copy.deepcopy(star)

if in_ref is not None:
    print('\tLoading input STAR file with references...')
    star_ref, star_tab = ps.sub.Star(), ps.sub.Star()
    try:
        star_ref.load(in_ref)
        if in_mic_ref is not None:
            star_tab.load(in_mic_ref)
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: input STAR file could not be read because of "' + str(e.msg, e.expr) + '"')
        sys.exit(-1)
    if in_mic_ref is not None:
        print('\tConverting reference coordiantes using input segmentation table...')
        for row in range(star_ref.get_nrows()):
            # Initialization
            seg_str_ref = star_ref.get_element('_rlnMicrographName', row)
            seg_row = star_tab.find_element('_psSegImage', seg_str_ref)
            mic_str_ref = star_tab.get_element('_rlnMicrographName', seg_row)
            mic = disperse_io.load_tomo(mic_str_ref, mmap=True)
            cx, cy, cz = star_ref.get_particle_coords(row, orig=False, rots=False)
            # Add Cropping
            seg_offy, seg_offx, seg_offz = star_tab.get_element('_psSegOffX', seg_row), \
                                           star_tab.get_element('_psSegOffY', seg_row), \
                                           star_tab.get_element('_psSegOffZ', seg_row)
            # cx, cy, cz = cx+seg_offx, cy+seg_offy, cz+seg_offz
            cx, cy, cz = cy+seg_offx, cx+seg_offy, cz+seg_offz
            # Centering
            mic_cx, mic_cy, mic_cz = .5*mic.shape[0], .5*mic.shape[1], .5*mic.shape[2]
            cx, cy, cz = cx-mic_cx, cy-mic_cy, cz-mic_cz
            # Un-rotation
            seg_rot, seg_tilt, seg_psi = star_tab.get_element('_psSegRot', seg_row), \
                                         star_tab.get_element('_psSegTilt', seg_row), \
                                         star_tab.get_element('_psSegPsi', seg_row)
            M = rot_mat_relion(seg_rot, seg_tilt, seg_psi, deg=True)
            hold_coords = np.asarray(M.T * np.asarray((cx, cy, cz), dtype=float).reshape(3, 1)).reshape(3)
            # Un-centering
            cx, cy, cz = hold_coords[0]+mic_cx, hold_coords[1]+mic_cy, hold_coords[2]+mic_cz
            # Update STAR file entry
            star_ref.set_element('_rlnMicrographName', row, mic_str_ref)
            star_ref.set_element('_rlnCoordinateX', row, cx)
            star_ref.set_element('_rlnCoordinateY', row, cy)
            star_ref.set_element('_rlnCoordinateZ', row, cz)

    print('\tGetting particles coordinates by micrograph...')
    mics, mics_ref = star.get_column_data('_rlnMicrographName'), star_ref.get_column_data('_rlnMicrographName')
    mics_set = list(set(mics) & set(mics_ref))
    coords_dic, rows_dic = dict.fromkeys(mics_set), dict.fromkeys(mics_set)
    for key in coords_dic.keys():
        coords_dic[key], rows_dic[key] = list(), list()
    for row in range(star.get_nrows()):
        mic = mics[row]
        if mic in mics_set:
            coords_dic[mic].append(star.get_particle_coords(row, orig=True, rots=False))
            rows_dic[mic].append(row)

hold_ids = list()
if in_ref is None:

    print('\tGetting particles coordinates by micrograph...')
    mics = star.get_column_data('_rlnMicrographName')
    mics_set = list(set(mics))
    coords_dic, rows_dic = dict.fromkeys(mics_set), dict.fromkeys(mics_set)
    for key in coords_dic.keys():
        coords_dic[key], rows_dic[key] = list(), list()
    for row in range(star.get_nrows()):
        mic = mics[row]
        if mic in mics_set:
            coords_dic[mic].append(star.get_particle_coords(row, orig=True, rots=False))
            rows_dic[mic].append(row)

    print('\tScale suppression by micrograph (Auto)...')
    for mic in mics_set:
        print('\t\t-Processing micrograph: ' + str(mic))
        coords, rows = np.asarray(coords_dic[mic], dtype=float), np.asarray(rows_dic[mic], dtype=int)
        ids_lut = np.zeros(shape=len(rows), dtype=bool)
        for i in range(len(rows)):
            coord, row = coords[i, :], rows[i]
            if not ids_lut[i]:
                hold = coord - coords
                dsts = np.sqrt((hold*hold).sum(axis=1))
                ids = np.where(dsts < min_dst_v)[0]
                for idx in ids:
                    if idx != i:
                        ids_lut[idx] = True
        for i in range(len(rows)):
            if ids_lut[i]:
                hold_ids.append(rows[i])
        print('\t\t\t+Rows to delete: ' + str(ids_lut.sum()) + ' of ' + str(len(rows)))
else:
    print('\tScale suppression by micrograph...')
    coords_ref_dic = dict.fromkeys(mics_set)
    for key in coords_ref_dic.keys():
        coords_ref_dic[key] = list()
    for row in range(star_ref.get_nrows()):
        mic = mics_ref[row]
        if mic in mics_set:
            coords_ref_dic[mic].append(star_ref.get_particle_coords(row, orig=True, rots=False))
    for mic in mics_set:
        print('\t\t-Processing micrograph: ' + str(mic))
        count = 0
        coords, rows = np.asarray(coords_dic[mic], dtype=float), np.asarray(rows_dic[mic], dtype=int)
        coords_ref = np.asarray(coords_ref_dic[mic], dtype=float)
        for i in range(len(rows)):
            coord, row = coords[i, :], rows[i]
            hold = coords_ref - coord
            dsts = np.sqrt((hold*hold).sum(axis=1))
            # print dsts.min() * res
            ids = np.where(dsts < min_dst_v)[0]
            if len(ids) > 0:
                hold_ids.append(rows[i])
                count += 1
        print('\t\t\t+Rows to delete: ' + str(count) + ' of ' + str(len(rows)))

print('\tDeleting ' + str(len(hold_ids)) + ' rows of ' + str(star.get_nrows()) + ' ...')
star_out.del_rows(hold_ids)

print('\t\t-Storing the results in: ' + out_star)
star_out.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')