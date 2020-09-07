"""

    Generates a ListTomoParticle object from a input STAR file with vesicles segmentations

    Input:  - STAR file for pairing STAR files with vesicles segmentations

    Output: - A STAR file with a ListTomoParticles object pickled

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import vtk
import time
import scipy as sp
import os
import sys
import numpy as np
from pyorg import pexceptions, sub, disperse_io, surf
from surf.utils import vtp_to_vtp_closest_point
try:
    import pickle as pickle
except:
    import pickle

########## Global variables

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an'

# Input STAR file with segmentations
in_star = ROOT_PATH + '/in/syn_seg_ves_apex_lgap_14.star'

# Input vtp path
in_vtp = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre/vtps/sph_rad_5_surf.vtp'

####### Output data

out_dir = ROOT_PATH+'/ex/syn/sub/relion/fils/ves_ap/ltomos_ves_ap_premb_mask'
out_sufix = 'ves_ap_premb_mask'

####### Segmentation pre-processing

sg_lbl = 2 # VOI label
sg_v_lbl = 6 # vesicles labels
sg_dsts = [0, 10] # nm, range of distances to consider, if None not used
sg_sg = 0
sg_dec = 0.9
sg_bc = False
sg_bm = 'box'
sg_pj = True
sg_voi_mask = True
sg_min_vol = 500 # voxels
sg_res = 0.684 # nm/voxels

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print('Adding segmentation to GraphMCF objects.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
# print '\tDisPerSe persistence threshold (nsig): ' + str(nsig)
print('\tSTAR file with the GraphMCF and segmentation pairs: ' + in_star)
print('\tInput vtp file with particle shape: ' + in_vtp)
print('\tOutput directory: ' + out_dir)
print('\t\t-Files sufix: ' + out_sufix)
print('\tVOI processing: ')
print('\t\t-Pixel size: ' + str(sg_res) + ' nm/pixel')
print('\t\t-Segmentation Gaussian smoothing sigma: ' + str(sg_sg))
if sg_dec is not None:
    print('\t\t-Triangle decimation factor: ' + str(sg_dec))
if sg_bc:
    print('\t\t-Checking particles VOI boundary with mode: ' + str(sg_bm))
if sg_pj:
    print('\t\t-Activated particles projecting on surface VOI.')
if sg_voi_mask:
    print('\t\t-Mask VOI mode activated!')
print('\tSegmentation processing: ')
print('\t\t-Segmentation label: ' + str(sg_lbl))
print('\t\t-Range of distances: ' + str(sg_dsts) + ' nm')
print('\t\t-Minimum volume: ' + str(sg_min_vol) + ' voxels')
print('')

print('Loading the input star file...')
star, star_out = sub.Star(), sub.Star()
try:
    star.load(in_star)
    star_out.add_column('_psPickleFile')
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
if not star.has_column('_psSegImage'):
    print('ERROR: input pairs STAR file has no \'psSegImage\' column.')
    print('Un-successfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
if not star.has_column('_suSurfaceVtp'):
    print('ERROR: input pairs STAR file has no \'psSegImage\' column.')
    print('Un-successfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
if not star.has_column('_rlnMicrographName'):
    print('ERROR: input graph STAR file has no \'rlnMicrographName\' column.')
    print('Un-successfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
part_vtp = disperse_io.load_poly(in_vtp)

print('\tLoop for generating tomograms VOIs: ')
vois = dict()
for row in range(star.get_nrows()):
    mic_file, seg_file = star.get_element('_rlnMicrographName', row), \
                         star.get_element('_psSegImage', row)
    print('\t\t-Generating VOI from segmentation: ' + str(mic_file))
    tomo = disperse_io.load_tomo(seg_file, mmap=False)
    if sg_voi_mask:
        voi = tomo == sg_lbl
        vois[seg_file] = voi
        seg_fname = os.path.split(os.path.splitext(seg_file)[0])[1]
        disperse_io.save_numpy(voi, out_dir + '/' + seg_fname + '_mask_voi.mrc')
    else:
        tomo = (tomo == sg_lbl).astype(np.float32)
        if (sg_sg is not None) and (sg_sg > 0):
            tomo = sp.ndimage.filters.gaussian_filter(tomo, sg_sg)
        voi = surf.iso_surface(tomo, 0.5, closed=True, normals='outwards')
        if sg_dec is not None:
            voi = surf.poly_decimate(voi, sg_dec)
        vois[seg_file] = voi

print('\t\tList of tomograms initialization...')
set_lists, list_tomos = surf.SetListTomoParticles(), surf.ListTomoParticles()
for seg_file, voi in zip(iter(vois.keys()), iter(vois.values())):
    list_tomos.add_tomo(surf.TomoParticles(seg_file, sg_lbl, voi=voi))

# Loop for processing the input data
print('Running main loop: ')
for row in range(star.get_nrows()):

    seg_file, mic_file = star.get_element('_psSegImage', row), star.get_element('_rlnMicrographName', row)
    ves_file = star.get_element('_suSurfaceVtp', row)
    print('\tPre-processing segmentation tomogram: ' + seg_file)
    try:
        seg = disperse_io.load_tomo(seg_file).astype(np.uint16)
    except IOError:
        print('WARNING: input tomograms ' + seg_file + ' could not be read!')
        continue
    try:
        ves_seg = disperse_io.load_tomo(ves_file).astype(np.uint16)
    except IOError:
        print('WARNING: input tomograms ' + seg_file + ' could not be read!')
        continue
    if os.path.splitext(ves_file)[1] == '.vti':
        ves_seg = ves_seg.swapaxes(0,2)

    print('\tLabelling vesicles: ')
    ves_lbls, num_lbls = sp.ndimage.measurements.label(ves_seg == sg_v_lbl)
    print('\t\t-Vesicles found: ' + str(num_lbls))
    for ves_id in np.arange(start=1, stop=num_lbls+1):
        print('\t\t\t+Processing vesicle: ' + str(ves_id))
        ves_vol = ves_lbls == ves_id
        ves_vol_sum = ves_vol.sum()
        if ves_vol_sum > sg_min_vol:
            hold_tomo = (ves_lbls == ves_id).astype(np.float32)
            if (sg_sg is not None) and (sg_sg > 0):
                hold_tomo = sp.ndimage.filters.gaussian_filter(hold_tomo, sg_sg)
            ves_voi = surf.iso_surface(hold_tomo, 0.5, closed=True, normals='outwards')
            mb_voi = vois[seg_file]
            if not isinstance(mb_voi, vtk.vtkPolyData):
                hold_tomo = (mb_voi).astype(np.float32)
                if (sg_sg is not None) and (sg_sg > 0):
                    hold_tomo = sp.ndimage.filters.gaussian_filter(hold_tomo, sg_sg)
                mb_voi = surf.iso_surface(hold_tomo, 0.5, closed=True, normals='outwards')
            # disperse_io.save_vtp(ves_voi, out_dir + '/hold_ves.vtp')
            # disperse_io.save_numpy(hold_tomo, out_dir + '/hold_ves.vti')
            # disperse_io.save_vtp(mb_voi, out_dir + '/hold_mb.vtp')
            pt_id, pt_dst = vtp_to_vtp_closest_point(ves_voi, mb_voi)
            pt_dst *= sg_res
            if (pt_dst >= sg_dsts[0]) and (pt_dst <= sg_dsts[1]):
                pt_coords = ves_voi.GetPoint(pt_id)
                print('\t\t\t\t*Inserting particle in coords ' + str(pt_coords) + ', distance ' + str(pt_dst) + ' nm' + \
                    ' and volume ' + str(ves_vol_sum) + ' voxels')
                # Insert the new particle in the proper tomogram
                part = surf.Particle(part_vtp, center=(0., 0., 0.), normal=(0, 0, 1.))
                part.translation(pt_coords[0], pt_coords[1], pt_coords[2])
                try:
                    list_tomos.insert_particle(part, seg_file, check_bounds=sg_bc, mode=sg_bm, voi_pj=sg_pj)
                except pexceptions.PySegInputError as e:
                    print('WARINING: particle in row ' + str(row) + ' could not be inserted in tomogram ' + seg_file + \
                          ' because of "' + e.get_message() + '"')
                    pass
            else:
                print('\t\t\t\t*Particles NOT inserted (distance ' + str(pt_dst) + ' nm)')
        else:
            print('\t\t\t\t*Particles NOT inserted (volume ' + str(ves_vol_sum) + ' voxels)')

star_stem = os.path.splitext(os.path.split(in_vtp)[1])[0]
out_pkl = out_dir + '/0_' + star_stem + '_tpl.pkl'
print('\t\tPickling the list of tomograms in the file: ' + out_pkl)
try:
    list_tomos.pickle(out_pkl)
    kwargs = {'_psPickleFile': out_pkl}
    star_out.add_row(**kwargs)
except pexceptions.PySegInputError as e:
    print('ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

out_parts = out_dir + '/' + star_stem + '_parts.star'
print('\tStoring particles STAR file: ' + out_parts)
star_parts = list_tomos.to_particles_star()
star_parts.store(out_parts)

out_app = out_dir + '/' + star_stem + '_app'
if not os.path.exists(out_app):
    os.makedirs(out_app)
print('\tStoring particles grouped by tomograms: ' + out_app)
for tomo in list_tomos.get_tomo_list():
    if tomo.get_num_particles() > 0:
        tomo_fname = os.path.splitext(os.path.split(tomo.get_tomo_fname())[1])[0]
        disperse_io.save_vtp(tomo.append_particles_vtp(mode='surface'), out_app + '/' + tomo_fname + '.vtp')

# Adding particle to list
set_lists.add_list_tomos(list_tomos, out_pkl)

print('\tStoring list appended by tomograms in: ' + out_dir)
tomos_vtp = set_lists.tomos_to_vtp(mode='surface')
for key, poly in zip(iter(tomos_vtp.keys()), iter(tomos_vtp.values())):
    stem_tomo = os.path.splitext(os.path.split(key)[1])[0]
    disperse_io.save_vtp(poly, out_dir + '/' + stem_tomo + '_lists_app.vtp')

out_star = out_dir + '/' + out_sufix + '_ltomos.star'
print('\tOutput STAR file: ' + out_star)
star_out.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')