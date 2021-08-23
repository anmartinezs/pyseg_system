"""

    Generates a ListTomoParticle object from a input STAR file (light particles version)

    Input:  - STAR file for pairing STAR files with particles and particles shapes

    Output: - A ListTomoPaticles pickled object for every STAR file row

"""

################# Package import

import os
import vtk
import numpy as np
import scipy as sp
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg import globals as gl

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-engel/antonio/ribo'

# Input STAR file
in_star = ROOT_PATH + '/ltomos/in_ltomos_test.star'

# Input STAR for with the sub-volumes segmentations
in_seg = ROOT_PATH + '/ltomos/in_seg_test.star'

# Output directory
out_dir = ROOT_PATH + '/ltomos/testT_mask_l'
out_stem = 'testT_mask_l' # 'pre'

# Segmentation pre-processing
sg_lbl = 1
sg_sg = 0
sg_dec = 0.9
sg_bc = False
sg_bm = 'center'
sg_pj = False
sg_voi_mask = True

# Post-processing
pt_min_parts = 0
pt_keep = None

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Generation of Surface objects from and input STAR file or tomogram (light version).')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tInput STAR file of particles: ' + str(in_star))
print('\t\t-Input STAR file for segmentations: ' + str(in_seg))
print('\tSegmentation pre-processing: ')
print('\t\t-Segmentation label: ' + str(sg_lbl))
print('\t\t-Segmentation Gaussian smoothing sigma: ' + str(sg_sg))
if sg_dec is not None:
    print('\t\t-Triangle decimation factor: ' + str(sg_dec))
if sg_bc:
    print('\t\t-Checking particles VOI boundary with mode: ' + str(sg_bm))
if sg_pj:
    print('\t\t-Activated particles projecting on surface VOI.')
if sg_voi_mask:
    print('\t\t-Mask VOI mode activated!')
print('\tPost-processing: ')
print('\t\t-Keep tomograms the ' + str(pt_keep) + 'th with the highest number of particles.')
print('\t\t-Minimum number of particles: ' + str(pt_min_parts))
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
seg_dic, seg_dic_fname = dict(), dict()
for seg_row in range(star_seg.get_nrows()):
    mic_str, seg_str = star_seg.get_element('_rlnMicrographName', seg_row), \
                       star_seg.get_element('_psSegImage', seg_row)
    seg_dic[mic_str] = seg_row
    seg_dic_fname[os.path.split(mic_str)[1]] = seg_row

print('\tProcessing STAR file rows: ')

surfs = list()
print('\tLoading input STAR file(s)...')
star, star_out = sub.Star(), sub.Star()
try:
    star.load(in_star)
    star_out.add_column('_psPickleFile')
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('\tLoop for generating tomograms VOIs: ')
vois = dict()
for tomo_row in range(star_seg.get_nrows()):
    mic_str, seg_str = star_seg.get_element('_rlnMicrographName', tomo_row), \
                       star_seg.get_element('_psSegImage', tomo_row)
    print('\t\t-Generating VOI from segmentation: ' + str(mic_str))
    tomo = disperse_io.load_tomo(seg_str, mmap=False)
    if sg_voi_mask:
        voi = tomo == sg_lbl
        seg_fname = os.path.split(os.path.splitext(seg_str)[0])[1]
        out_voi = out_dir + '/' + seg_fname + '_mask_voi.mrc'
        disperse_io.save_numpy(voi, out_voi)
        voi = disperse_io.load_tomo(out_voi, mmap=True)
        vois[seg_str] = voi
    else:
        tomo = (tomo == sg_lbl).astype(np.float32)
        if (sg_sg is not None) and (sg_sg > 0):
            tomo = sp.ndimage.filters.gaussian_filter(tomo, sg_sg)
        voi = surf.iso_surface(tomo, 0.5, closed=True, normals='outwards')
        if sg_dec is not None:
            voi = surf.poly_decimate(voi, sg_dec)
        vois[seg_str] = voi

print('\tLoop for tomograms in the list: ')
set_lists = surf.SetListTomoParticles()
for star_row in range(star.get_nrows()):

    print('\t\tNow list of tomograms initialization...')
    list_tomos = surf.ListTomoParticles()
    part_star_str, part_surf_str = star.get_element('_psStarFile', star_row), \
                                   star.get_element('_suSurfaceVtp', star_row)
    for tomo_fname, voi in zip(iter(vois.keys()), iter(vois.values())):
        list_tomos.add_tomo(surf.TomoParticles(tomo_fname, sg_lbl, voi=voi))

    print('\t\tLoading particles STAR file(s):')
    star_part = sub.Star()
    try:
        star_part.load(part_star_str)
    except pexceptions.PySegInputError as e:
        print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
    part_vtp = disperse_io.load_poly(part_surf_str)
    if not isinstance(part_vtp, vtk.vtkPolyData):
        print('ERROR: file ' + part_surf_str + ' is not a vtkPolyData!')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
    if not surf.is_closed_surface(part_vtp):
        print('ERROR: file ' + part_surf_str + ' is not a closed surface!')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    print('\t\tParticles loop..')
    for part_row in range(star_part.get_nrows()):

        # Initialization
        mic_str = star_part.get_element('_rlnMicrographName', part_row)
        try:
            seg_row = seg_dic[mic_str]
        except KeyError:
            seg_row = seg_dic_fname[mic_str]
            mic_str = star_seg.get_element('_rlnMicrographName', seg_row)
        seg_str = star_seg.get_element('_psSegImage', seg_row)
        mic = disperse_io.load_tomo(mic_str, mmap=True)
        (cx, cy, cz), (rho, tilt, psi) = star_part.get_particle_coords(part_row, orig=True, rots=True)
        part_center, part_eu_angs = np.asarray((cx, cy, cz), dtype=np.float32), \
                                    np.asarray((rho, tilt, psi), dtype=np.float32)

        # Segmentation rigid body transformations
        mic_center = np.asarray((.5*mic.shape[0], .5*mic.shape[1], .5*mic.shape[2]), dtype=np.float32)
        # Centering
        part_center -= mic_center
        # Un-rotation
        seg_rot, seg_tilt, seg_psi = star_seg.get_element('_psSegRot', seg_row), \
                                     star_seg.get_element('_psSegTilt', seg_row), \
                                     star_seg.get_element('_psSegPsi', seg_row)
        if (seg_rot != 0) and (seg_tilt != 0) and (seg_psi != 0):
            R1 = gl.rot_mat_relion(part_eu_angs[0], part_eu_angs[1], part_eu_angs[2])
            R2 = gl.rot_mat_relion(seg_rot, seg_tilt, seg_psi)
            Rt = R2 * R1
            part_eu_angs = np.asarray(gl.rot_mat_eu_relion(Rt), dtype=np.float32)
            part_center *= R1
        # Un-centering
        part_center += mic_center
        # Un-cropping
        seg_offy, seg_offx, seg_offz = star_seg.get_element('_psSegOffX', seg_row), \
                                       star_seg.get_element('_psSegOffY', seg_row), \
                                       star_seg.get_element('_psSegOffZ', seg_row)
        part_center -= np.asarray((seg_offx, seg_offy, seg_offz), dtype=np.float32)
        # part.swap_xy()

        # Insert the new particle in the proper tomogram
        try:
            part = surf.ParticleL(part_surf_str, center=part_center, eu_angs=part_eu_angs)
            list_tomos.insert_particle(part, seg_str, check_bounds=sg_bc, mode=sg_bm, voi_pj=sg_pj)
        except pexceptions.PySegInputError as e:
            print('WARINING: particle in row ' + str(part_row) + ' could not be inserted in tomogram ' + tomo_fname + \
                  ' because of "' + e.get_message() + '"')
            pass

    if pt_keep is not None:
        print('\t\tFiltering to keep the ' + str(pt_keep) + 'th more highly populated')
        list_tomos.clean_low_pouplated_tomos(pt_keep)
    if pt_min_parts >= 0:
        print('\t\tFiltering tomograms with less particles than: ' + str(pt_min_parts))
        list_tomos.filter_by_particles_num(pt_min_parts)

    star_stem = os.path.splitext(os.path.split(part_star_str)[1])[0]
    out_pkl = out_dir + '/' + star_stem + '_tpl.pkl'
    print('\t\tPickling the list of tomograms in the file: ' + out_pkl)
    try:
        list_tomos.pickle(out_pkl)
        kwargs = {'_psPickleFile': out_pkl}
        star_out.add_row(**kwargs)
    except pexceptions.PySegInputError as e:
        print('ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    out_app = out_dir + '/' + star_stem + '_app'
    if not os.path.exists(out_app):
        os.makedirs(out_app)
    print('\tStoring particles grouped by tomograms: ' + out_app)
    for tomo in list_tomos.get_tomo_list():
        if tomo.get_num_particles() > 0:
            tomo_fname = os.path.splitext(os.path.split(tomo.get_tomo_fname())[1])[0]
            disperse_io.save_vtp(tomo.append_particles_vtp(mode='surface'), out_app+'/'+tomo_fname+'.vtp')

    # Adding particle to list
    set_lists.add_list_tomos(list_tomos, star_stem)

out_parts = out_dir + '/' + out_stem + '_parts.star'
print('\tStoring the particles STAR file: ' + out_parts)
set_lists.to_particles_star().store(out_parts)

print('\tStoring list appended by tomograms in: ' + out_dir)
tomos_vtp = set_lists.tomos_to_vtp(mode='surface')
for key, poly in zip(iter(tomos_vtp.keys()), iter(tomos_vtp.values())):
    stem_tomo = os.path.splitext(os.path.split(key)[1])[0]
    disperse_io.save_vtp(poly, out_dir+'/'+stem_tomo+'_lists_app.vtp')

out_star = out_dir + '/' + out_stem + '_ltomos.star'
print('\tOutput STAR file: ' + out_star)
star_out.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')
