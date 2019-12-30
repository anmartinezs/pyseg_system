"""

    Generates Surface objects from and input STAR file or tomogram

    Input:  - Either a STAR file of particles and the particle average or a segmented tomogram

    Output: - An output STAR file with the set of Surface objects generated which are stored in disk

"""

################# Package import

import os
import vtk
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf
import scipy as sp

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/fils/ves'

# Input STAR file
in_star = ROOT_PATH + '/sub/field_value_eq.star' # None

# Input Tomogram for particle density (required and only applicable if the input is a list of particles)
in_vol = ROOT_PATH + '/ref_8/mask_20_50_10/klass_8_particles_k6/run1_class001.mrc' # None

# Input STAR for with the sub-volumes segmentations
in_seg = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/in/syn_seg_all_rln.star'
# in_seg = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/in/syn_seg_hold.star'

# Output directory
out_dir = ROOT_PATH + '/surf'
out_stem = 'ves_clft'

# Tomogram pre-processing
pt_sg = 1. # voxels
pt_th = 0.5
pt_dec = 0.9

# Particle density map
pd_th = 0.121 # 0.3
pd_dec = 0.9

# Segmentation pre-processing
sg_lbl = 5
sg_bc = False
sg_bm = 'center'
sg_pj = True

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Generation of Surface objects from and input STAR file or tomogram.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\t\t-Output stem: ' + str(out_stem)
if in_vol is None:
    print '\tInput STAR file of segmentations: ' + str(in_star)
    print '\tTomogram pre-processing: '
    print '\t\t-Binary segmentation smoothing: ' + str(pt_sg)
    print '\t\t-Iso-surface threshold: ' + str(pt_th)
else:
    print '\tInput STAR file of particles: ' + str(in_star)
    print '\tParticle density map: ' + str(in_vol)
    print '\tParticle pre-processing: '
    print '\t\t-Iso-surface threshold: ' + str(pd_th)
    if pd_dec is not None:
        print '\t\t-Triangle decimation factor: ' + str(pd_dec)
print '\tSegmentation pre-processing: '
print '\t\t-Segmentation label: ' + str(sg_lbl)
if pt_dec is not None:
    print '\t\t-Triangle decimation factor: ' + str(pt_dec)
if sg_bc:
    print '\t\t-Checking particles VOI boundary with mode: ' + str(sg_bm)
if sg_pj:
    print '\t\t-Activated particles projecting on surface VOI.'
print '\tGeneral pre-processing: '
if in_seg is not None:
    print '\t\t-Input STAR file for sub-volumes segmentation: ' + str(in_seg)
print ''

######### Process

print 'Main Routine: '

surfs = list()
print '\tLoading input STAR file(s)...'
star = sub.Star()
try:
    star.load(in_star)
    star.add_column('_suPartName')
    if in_seg is not None:
        star_seg = sub.Star()
        star_seg.load(in_seg)
        seg_mics, seg_segs = star_seg.get_column_data('_rlnMicrographName'), \
                             star_seg.get_column_data('_psSegImage')
except pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\tInitializing tomograms containers...'
list_tomos = surf.ListTomoParticles()
if in_seg is None:
    hold_fnames = seg_mics
else:
    hold_fnames = seg_segs
for fname in hold_fnames:
    hold_path = out_dir + '/' + out_stem + '_' + os.path.splitext(os.path.split(fname)[1])[0]
    print '\t\t-Processing tromogram: ' + str(fname)
    try:
        hold_tomo = surf.TomoParticles(fname, sg_lbl, sg=2.)
        if pt_dec is not None:
            hold_tomo.decimate_voi(pt_dec)
        list_tomos.add_tomo(hold_tomo)
    except OSError:
        print 'ERROR: directory ' + hold_path + ' could not be created!'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)

# Tomogram case
if in_vol is None:

    print 'ERROR: un-implemented option!'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

    print '\tParsing the input STAR file...'
    col_keys = star.get_column_keys()
    if '_psSegImage' not in col_keys:
        print 'ERROR: not column psSegImage found in the input STAR file'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    if '_psSegLabel' not in col_keys:
        print 'ERROR: not column psSegLabel found in the input STAR file'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)

    print '\tSegmented tomograms loop:'
    for row in star.get_nrows():

        seg, lbl = star.get_element('_psSegImage', row), star.get_element('_psSegLabel', row)
        print '\t\t-Loading input tomogram: ' + seg
        print '\t\t-Label for binarization: ' + lbl
        try:
            tomo = disperse_io.load_tomo(seg)
        except pexceptions.PySegInputError as e:
            print 'ERROR: input tomogram could not be loaded because of "' + e.get_message() + '"'
            print 'Terminated. (' + time.strftime("%c") + ')'
            sys.exit(-1)

        print '\t\t-Extracting iso-surfaces...'
        bin_tomo = density = sp.ndimage.filters.gaussian_filter(tomo == lbl, pt_sg)
        hold_surf = surf.Surface(surf.iso_surface(bin_tomo, pt_th, dec=pg_dec))

        out_surf = out_dir + '/' + os.path.splitext(os.path.splitext(seg)[1])[0] + '_surf.vtp'
        print '\t\t-Storing generated Surface object in: ' + out_surf
        try:
            hold_surf.store(out_surf)
        except pexceptions.PySegInputError as e:
            print 'ERROR: input tomogram could not be loaded because of "' + e.get_message() + '"'
            print 'Terminated. (' + time.strftime("%c") + ')'
            sys.exit(-1)
        star.set_column_data('_suSurfName', out_surf)

# Input STAR file case
else:

    print '\tGenerating particle iso-surface from file: ' + str(in_vol)
    try:
        tomo = disperse_io.load_tomo(in_vol)
    except pexceptions.PySegInputError as e:
        print 'ERROR: input tomogram could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    part_vtp = surf.iso_surface(tomo, pd_th)
    if pd_dec is not None:
        part_vtp = surf.poly_decimate(part_vtp, pd_dec)
    cx, cy, cz = -.5*float(tomo.shape[0]), -.5*float(tomo.shape[1]), -.5*float(tomo.shape[2])

    print '\tParticles loop:'
    for row in range(star.get_nrows()):

        print '\t\t-Generating the Surface object...'
        hold_surf = surf.Particle(part_vtp, center=(-cx, -cy, -cz), normal=(0, 0, 1.))

        print '\t\t-Applying rigid body transformation for reference tomogram...'
        (shift_x, shift_y, shift_z), (rho, tilt, psi) = star.get_particle_coords(row, orig=True, rots=True)
        # Centering
        hold_surf.translation(cx, cy, cz)
        # Storing reference surface
        if row == 0:
            out_surf_ref = out_dir + '/' + out_stem + '_surf_ref.vtp'
            print '\t\t-Storing surface reference in: ' + out_surf_ref
            disperse_io.save_vtp(hold_surf.get_vtp(mode='surface'), out_surf_ref)
        # Un-rotation
        hold_surf.rotation(rho, tilt, psi, active=False)
        # Translation to tomogram place
        hold_surf.translation(shift_x, shift_y, shift_z)

        mic_name = star.get_element('_rlnMicrographName', row)
        mic = disperse_io.load_tomo(mic_name, mmap=True)
        try:
            seg_row = seg_mics.index(mic_name)
        except ValueError:
            print 'WARNING: Particle ' + str(row) + ' not in tomograms list!'
            continue
        if in_seg is not None:
            print '\t\t-Applying rigid body transformation for segmentation tomogram...'
            mic_cx, mic_cy, mic_cz = .5*mic.shape[0], .5*mic.shape[1], .5*mic.shape[2]
            # Centering
            hold_surf.translation(-mic_cx, -mic_cy, -mic_cz)
            # Un-rotation
            seg_rot, seg_tilt, seg_psi = star_seg.get_element('_psSegRot', seg_row), \
                                         star_seg.get_element('_psSegTilt', seg_row), \
                                         star_seg.get_element('_psSegPsi', seg_row)
            hold_surf.rotation(seg_rot, seg_tilt, seg_psi, active=True)
            # Un-centering
            hold_surf.translation(mic_cx, mic_cy, mic_cz)
            # Un-cropping
            seg_offy, seg_offx, seg_offz = star_seg.get_element('_psSegOffX', seg_row), \
                                           star_seg.get_element('_psSegOffY', seg_row), \
                                           star_seg.get_element('_psSegOffZ', seg_row)
            hold_surf.translation(-seg_offx, -seg_offy, -seg_offz)
            # print str(hold_surf.get_bounds())
            hold_surf.swap_xy()

        # Insert the new particle in the proper tomogram
        try:
            list_tomos.insert_particle(hold_surf, seg_segs[seg_row], check_bounds=sg_bc, mode=sg_bm, voi_pj=sg_pj)
        except pexceptions.PySegInputError as e:
            print 'WARINING: particle in row ' + str(row) + ' could not be inserted ' + \
                  'because of "' + e.get_message() + '"'
            pass

out_pkl = out_dir + '/' + out_stem + '_tpl.pkl'
print '\tPickling the list of tomograms in the stem: ' + out_pkl
try:
    list_tomos.pickle(out_pkl)
    # list_tomos.store_stars(out_stem, out_dir)
    # list_tomos.store_particles(out_dir, mode='surface')
    # list_tomos.store_particles(out_dir, mode='center')
except pexceptions.PySegInputError as e:
    print 'ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print 'Terminated. (' + time.strftime("%c") + ')'
