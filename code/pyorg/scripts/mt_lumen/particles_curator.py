"""

    Curates an output STAR file from Relion to work as input for pyseg.pyorg scripts for microtubules

    Input:  - STAR file with the particles to curate
            - STAR file to pair tomograms used for reconstruction with the one segmented used to pick the particles

    Output: - A curated output STAR file

"""

################# Package import

import os
import vtk
import numpy as np
import scipy as sp
import sys
import time
import shutil
#from surf_dst import pexceptions, sub, disperse_io, surf
#from surf_dst import globals as gl
import pyorg
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg import globals as gl

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-plitzko/Saikat/luminal_particle_organization/lattice_break_clustering' # '/fs/pool/pool-plitzko/Saikat/luminal_particle_organization/int_HeLa'

# Input STAR file
in_star = ROOT_PATH + '/in/0_lattice_break_center_points.star' # '/in/0_picking_mrc.star'

# Input STAR for with the sub-volumes segmentations
in_seg = ROOT_PATH + '/in/mts_clines_1_mts_t3_t6_pcorr.star' # '/in/mts_clines_mts_seg_picking_v1_parth_curated.star'

# Output directory
out_star = ROOT_PATH + '/in/0_lattice_break_center_points_curated.star'

p_bin = 1 # since particle coordinates are binned in the relion star file corresponding to picking resolution
p_max_dst = 1000 # 10 # nm
p_res = 1.368 # nm/pixel
p_swapxy = True
p_cp_ptomo = False # Copy column '_rlnMicrographName' values to '_mtParticlesTomo'
p_cp_mt_to_mn = True # if '_mtParticleTomo' does not exist, it is taken from '_rlnMicrographName'

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Curating particle STAR files for Microtubules.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput STAR file of particles: ' + str(in_star))
print('\tInput STAR file for segmentations: ' + str(in_seg))
print('\tOutput STAR file: ' + str(out_star))
print('\tPre-processing settings: ')
print('\t\t-Coordinates binning factor: ' + str(p_bin))
print('\t\t-Maximun distance to a centerline: ' + str(p_max_dst) + ' nm')
print('\t\t-Picking data resolution: ' + str(p_res) + ' nm/vx')
print('\t\t-Swap X and Y.')
if p_cp_ptomo:
    print('\t\t-Copying _mtParticlesTomo -> _rlnMicrographName.')
if p_cp_mt_to_mn:
    print('\t\t-Copying _rlnMicrographName -> _mtParticlesTomo.')
print('')

######### Process

print('Main Routine: ')

print('\tGenerating Micrograph-segmentations dictionary...')
star_seg = sub.Star()
try:
    star_seg.load(in_seg)
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
ct_dic = dict()
for seg_row in range(star_seg.get_nrows()):
    ct_str = star_seg.get_element('_mtCenterLine', seg_row)
    ct_dic[ct_str] = seg_row

print('\tPre-processing the input particles STAR file: ')

surfs = list()
print('\tLoading input STAR file(s)...')
star, star_out = sub.Star(), sub.Star()
p_max_dst_v = float(p_max_dst) / float(p_res)
try:
    star.load(in_star)
    star.add_column('_psSegImage')
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
part_dsts= dict().fromkeys(list(range(star.get_nrows())))
for key in part_dsts.keys():
    part_dsts[key] = np.finfo(np.float).max

if p_bin > 0:
    print('\t\t-Binning the input coordinates: ')
    p_bin_f = float(p_bin)
    for row in range(star.get_nrows()):
        hold_x = star.get_element('_rlnCoordinateX', row)
        hold_x /= p_bin
        star.set_element(key='_rlnCoordinateX', val=hold_x, row=row)
        hold_y = star.get_element('_rlnCoordinateY', row)
        hold_y /= p_bin
        star.set_element(key='_rlnCoordinateY', val=hold_y, row=row)
        hold_z = star.get_element('_rlnCoordinateZ', row)
        hold_z /= p_bin
        star.set_element(key='_rlnCoordinateZ', val=hold_z, row=row)

if p_cp_ptomo:
    print('\t\t-Copying column values mtParticlesTomo to rlnMicrographName...')
    for row in range(star_seg.get_nrows()):
        star_seg.set_element(key='_rlnMicrographName', val=star_seg.get_element('_mtParticlesTomo', row), row=row)

if p_cp_mt_to_mn:
    if not star.has_column('_mtParticlesTomo'):
        print('\t\t-Copying column values mtParticlesTomo to rlnMicrographName...')
        star.add_column('_mtParticlesTomo')
        for row in range(star.get_nrows()):
            star.set_element(key='_mtParticlesTomo', val=star.get_element('_rlnMicrographName', row), row=row)

print('\tLoop for MT: ')
seg_dic = dict()
for row_ct in range(star_seg.get_nrows()):

    ct_str = star_seg.get_element('_mtCenterLine', row_ct)
    print('\t\t-MT to process: ' + ct_str)
    ct_vtp = disperse_io.load_poly(ct_str)
    ct_points = np.zeros(shape=(ct_vtp.GetNumberOfPoints(), 3), dtype=np.float32)
    for i in range(ct_points.shape[0]):
        ct_points[i, :] = ct_vtp.GetPoint(i)
    seg_dic[star_seg.get_element('_psSegImage', row_ct)] = star_seg.get_element('_rlnMicrographName', row_ct)

    print('\tLoop for particles: ')
    for row in range(star.get_nrows()):

        # Loading the input coordiante
        x = star.get_element('_rlnCoordinateX', row)
        y = star.get_element('_rlnCoordinateY', row)
        z = star.get_element('_rlnCoordinateZ', row)
        part_center = np.asarray((x, y, z), dtype=np.float32)

        # Transform to MT space
        # Un-cropping
        seg_offy, seg_offx, seg_offz = star_seg.get_element('_psSegOffX', row_ct), \
                                       star_seg.get_element('_psSegOffY', row_ct), \
                                       star_seg.get_element('_psSegOffZ', row_ct)
        if p_swapxy:
            part_center -= np.asarray((seg_offy, seg_offx, seg_offz), dtype=np.float32)
        else:
            part_center -= np.asarray((seg_offx, seg_offy, seg_offz), dtype=np.float32)

        # Finding the minimum distance
        hold = ct_points - part_center
        hold_min = np.sqrt((hold * hold).sum(axis=1)).min()
        if hold_min < part_dsts[row]:
            hold_seg = star_seg.get_element('_psSegImage', row_ct)
            star.set_element(key='_psSegImage', val=hold_seg, row=row)
            part_dsts[row] = hold_min

print(part_dsts)

print('\tDeleting unidentified particles...')
del_ids = list()
for row in range(star.get_nrows()):
    if part_dsts[row] > p_max_dst_v:
        del_ids.append(row)
    else:
        hold_mic = star.get_element('_rlnMicrographName', row)
        hold_seg = star.get_element('_psSegImage', row)
        if seg_dic[hold_seg] != hold_mic:
            print('Row [' + str(row) + ']: ' + seg_dic[hold_seg] + ', ' + hold_mic)
            del_ids.append(row)
star.del_rows(del_ids)
print('\t\t-Final number of particles: ' + str(star.get_nrows()))

print('\tWriting output STAR file: ' + out_star)
star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')
