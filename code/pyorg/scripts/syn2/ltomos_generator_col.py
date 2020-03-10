"""

    Generates a ListTomoParticle object from a input STAR file

    Input:  - STAR file for pairing STAR files with particles and particles shapes

    Output: - A ListTomoPaticles pickled object for every STAR file row

"""

################# Package import

import os
import numpy as np
import scipy as sp
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf

# TODEBUG
import vtk

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

# ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst'
ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn2'

# Input STAR file
in_star = ROOT_PATH + '/org/col/in/all_col_ltomos.star' # '/org/col/in/all_xd5nm_ltomos.star' # '/org/col/in/all_test_ltomos.star'

# Input STAR for with the sub-volumes segmentations
in_seg = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/in/syn_seg_no_l14_gap_zeros_fits.star' # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/in/syn_seg_no_l14_gap.star' # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/in/syn_seg_no_l14_gap.star'

# Output directory
out_dir = ROOT_PATH + '/org/col/ltomos/ltomos_col' # '/org/col/ltomos/ltomos_col_tmin3_min5' # '/org/col/ltomos/ltomos_all_scol_test'
out_stem = 'all_col' # 'all_col_tmin3_min5' # 'all_scol_test' # 'pre'

# Segmentation pre-processing
sg_lbl = 2 # 1
sg_sg = 0
sg_dec = 0.9
sg_bc = False
sg_bm = 'center'
sg_pj = True
sg_voi_mask = True

# Post-processing
pt_min_parts = 0 # {'0': 5, '1': 5, '2': 3, '3': 5, '4': 5, '5': 5}
pt_keep = None
pt_ssup = 7.31 # 5 # voxels

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Generation of Surface objects from and input STAR file or tomogram.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tInput STAR file of particles: ' + str(in_star)
print '\t\t-Input STAR file for segmentations: ' + str(in_seg)
print '\tSegmentation pre-processing: '
print '\t\t-Segmentation label: ' + str(sg_lbl)
print '\t\t-Segmentation Gaussian smoothing sigma: ' + str(sg_sg)
if sg_dec is not None:
    print '\t\t-Triangle decimation factor: ' + str(sg_dec)
if sg_bc:
    print '\t\t-Checking particles VOI boundary with mode: ' + str(sg_bm)
if sg_pj:
    print '\t\t-Activated particles projecting on surface VOI.'
if sg_voi_mask:
    print '\t\t-Mask VOI mode activated!'
print '\tPost-processing: '
print '\t\t-Keep tomograms the ' + str(pt_keep) + 'th with the highest number of particles.'
print '\t\t-Minimum number of particles: ' + str(pt_min_parts)
print '\t\t-Scale suppression: ' + str(pt_ssup) + ' voxels'
print ''

######### Process

print 'Main Routine: '

print '\tGenerating Micrograph-segmentations dictionary...'
star_seg = sub.Star()
try:
    star_seg.load(in_seg)
except pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
seg_dic = dict()
for seg_row in range(star_seg.get_nrows()):
    mic_str, seg_str = star_seg.get_element('_rlnMicrographName', seg_row), \
                       star_seg.get_element('_psSegImage', seg_row)
    seg_dic[os.path.split(mic_str)[1]] = seg_row

print '\tProcessing STAR file rows: '

surfs = list()
print '\tLoading input STAR file(s)...'
star, star_out = sub.Star(), sub.Star()
try:
    star.load(in_star)
    star_out.add_column('_psPickleFile')
except pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\tLoop for generating tomograms VOIs: '
vois = dict()
for tomo_row in range(star_seg.get_nrows()):
    mic_str, seg_str = star_seg.get_element('_rlnMicrographName', tomo_row), \
                       star_seg.get_element('_psSegImage', tomo_row)
    print '\t\t-Generating VOI from segmentation: ' + str(mic_str)
    tomo = disperse_io.load_tomo(seg_str, mmap=False)
    if sg_voi_mask:
        voi = tomo == sg_lbl
        vois[seg_str] = voi
        seg_fname = os.path.split(os.path.splitext(seg_str)[0])[1]
        disperse_io.save_numpy(voi, out_dir + '/' + seg_fname + '_mask_voi.mrc')
    else:
        tomo = (tomo == sg_lbl).astype(np.float32)
        if (sg_sg is not None) and (sg_sg > 0):
            tomo = sp.ndimage.filters.gaussian_filter(tomo, sg_sg)
        voi = surf.iso_surface(tomo, 0.5, closed=True, normals='outwards')
        if sg_dec is not None:
            voi = surf.poly_decimate(voi, sg_dec)
        vois[seg_str] = voi

# TODEBUG
appender = vtk.vtkAppendPolyData()
flt_stem = 'syn_11_2_bin2'

print '\tLoop for tomograms in the list: '
set_lists = surf.SetListTomoParticles()
for star_row in range(star.get_nrows()):

    print '\t\tNow list of tomograms initialization...'
    list_tomos = surf.ListTomoParticles()
    part_star_str, part_surf_str = star.get_element('_psStarFile', star_row), \
                                   star.get_element('_suSurfaceVtp', star_row)
    for tomo_fname, voi in zip(vois.iterkeys(), vois.itervalues()):
        list_tomos.add_tomo(surf.TomoParticles(tomo_fname, sg_lbl, voi=voi))

    print '\t\tLoading particles STAR file(s):'
    star_part = sub.Star()
    try:
        star_part.load(part_star_str)
    except pexceptions.PySegInputError as e:
        print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    part_vtp = disperse_io.load_poly(part_surf_str)

    print '\t\tParticles loop..'
    for part_row in range(star_part.get_nrows()):

        # Initialization
        mic_str = star_part.get_element('_rlnMicrographName', part_row)
        try:
            seg_row = seg_dic[os.path.split(mic_str)[1]]
        except KeyError:
            # if not mic_str.endswith('.fits'):
            print 'WARNING: particle in micrograph ' + mic_str + ' not considered!'
            continue
        # if mic_str.endswith('.fits'):
        #     seg_str = mic_str
        #     (cy, cx, cz), (rho, tilt, psi) = star_part.get_particle_coords(part_row, orig=True, rots=True)
        #     mic = disperse_io.load_tomo(mic_str, mmap=True)
        # else:
        seg_str = star_seg.get_element('_psSegImage', seg_row)
        (cx, cy, cz), (rho, tilt, psi) = star_part.get_particle_coords(part_row, orig=True, rots=True)
        mic = disperse_io.load_tomo(mic_str, mmap=True)
        part = surf.Particle(part_vtp, center=(0., 0., 0.), normal=(0, 0, 1.))

        # Initial rotation
        # ox, oy, oz = star_part.get_element('_rlnOriginX', part_row), \
        #              star_part.get_element('_rlnOriginY', part_row), \
        #              star_part.get_element('_rlnOriginZ', part_row)
        part.rotation(rho, tilt, psi, active=False)
        # part.translation(-ox, -oy, -oz)
        # part.translation(ox, oy, oz)

        # Original localization
        part.translation(cx, cy, cz)

        # Segmentation rigid body transformations
        mic_cx, mic_cy, mic_cz = .5*mic.shape[0], .5*mic.shape[1], .5*mic.shape[2]
        # Centering
        part.translation(-mic_cx, -mic_cy, -mic_cz)
        # Un-rotation
        if mic_str.endswith('.fits'):
            seg_rot, seg_tilt, seg_psi = 0, 0, 0
        else:
            seg_rot, seg_tilt, seg_psi = star_seg.get_element('_psSegRot', seg_row), \
                                         star_seg.get_element('_psSegTilt', seg_row), \
                                         star_seg.get_element('_psSegPsi', seg_row)
        part.rotation(seg_rot, seg_tilt, seg_psi, active=True)
        # Un-centering
        part.translation(mic_cx, mic_cy, mic_cz)
        # Un-cropping
        # if mic_str.endswith('.fits'):
        #     seg_offy, seg_offx, seg_offz = 0, 0, 0
        # else:
        seg_offy, seg_offx, seg_offz = star_seg.get_element('_psSegOffX', seg_row), \
                                       star_seg.get_element('_psSegOffY', seg_row), \
                                       star_seg.get_element('_psSegOffZ', seg_row)
        part.translation(-seg_offx, -seg_offy, -seg_offz)
        part.swap_xy()

        # Insert the new particle in the proper tomogram
        # meta_info = dict()
        # meta_info['_rlnMicrographName'] = mic_str
        # meta_info['_rlnImageName'] = star_part.get_element('_rlnImageName', part_row)
        # meta_info['_rlnCtfImage'] = star_part.get_element('_rlnCtfImage', part_row)
        # meta_info['_rlnCoordinateX'] = star_part.get_element('_rlnCoordinateX', part_row)
        # meta_info['_rlnCoordinateY'] = star_part.get_element('_rlnCoordinateY', part_row)
        # meta_info['_rlnCoordinateZ'] = star_part.get_element('_rlnCoordinateZ', part_row)
        # meta_info['_rlnOriginX'] = star_part.get_element('_rlnOriginX', part_row)
        # meta_info['_rlnOriginY'] = star_part.get_element('_rlnOriginY', part_row)
        # meta_info['_rlnOriginZ'] = star_part.get_element('_rlnOriginZ', part_row)
        # meta_info['_rlnAngleRot'] = star_part.get_element('_rlnAngleRot', part_row)
        # meta_info['_rlnAngleTilt'] = star_part.get_element('_rlnAngleTilt', part_row)
        # meta_info['_rlnAnglePsi'] = star_part.get_element('_rlnAnglePsi', part_row)
        meta_info = None
        try:
            list_tomos.insert_particle(part, seg_str, check_bounds=sg_bc, mode=sg_bm, voi_pj=sg_pj, meta=meta_info)
        except pexceptions.PySegInputError as e:
            print 'WARNING: particle in row ' + str(part_row) + ' could not be inserted in tomogram ' + tomo_fname + \
                  ' because of "' + e.get_message() + '"'
            pass

        # TODEBUG
        if os.path.splitext(os.path.split(mic_str)[1])[0] == flt_stem:
            appender.AddInputData(part.get_vtp())

    if pt_keep is not None:
        print '\t\tFiltering to keep the ' + str(pt_keep) + 'th more highly populated'
        list_tomos.clean_low_pouplated_tomos(pt_keep)
    if not isinstance(pt_min_parts, dict):
        if pt_min_parts >= 0:
            print '\t\tFiltering tomograms with less particles than: ' + str(pt_min_parts)
            list_tomos.filter_by_particles_num(pt_min_parts)
    if pt_ssup is not None:
        list_tomos.scale_suppression(pt_ssup)

    star_stem = os.path.splitext(os.path.split(part_star_str)[1])[0]
    star_stem = star_stem.split('_')[0]
    out_pkl = out_dir + '/' + star_stem + '_tpl.pkl'
    print '\t\tPickling the list of tomograms in the file: ' + out_pkl
    try:
        list_tomos.pickle(out_pkl)
        kwargs = {'_psPickleFile': out_pkl}
        star_out.add_row(**kwargs)
    except pexceptions.PySegInputError as e:
        print 'ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)

    out_app = out_dir + '/' + star_stem + '_app'
    if not os.path.exists(out_app):
        os.makedirs(out_app)
    print '\tStoring particles grouped by tomograms: ' + out_app
    for tomo in list_tomos.get_tomo_list():
        if tomo.get_num_particles() > 0:
            tomo_fname = os.path.splitext(os.path.split(tomo.get_tomo_fname())[1])[0]
            disperse_io.save_vtp(tomo.append_particles_vtp(mode='surface'), out_app+'/'+tomo_fname+'.vtp')

    # Adding particle to list
    set_lists.add_list_tomos(list_tomos, star_stem)

if isinstance(pt_min_parts, dict):
    print '\t\tFiltering tomograms with less particles than: ' + str(pt_min_parts)
    set_lists.filter_by_particles_num_tomos(pt_min_parts)

for star_row in range(star.get_nrows()):

    part_star_str, part_surf_str = star.get_element('_psStarFile', star_row), \
                                   star.get_element('_suSurfaceVtp', star_row)
    star_stem = os.path.splitext(os.path.split(part_star_str)[1])[0]
    star_stem = star_stem.split('_')[0]
    list_tomos = set_lists.get_lists_by_key(star_stem)
    out_pkl = out_dir + '/' + star_stem + '_tpl.pkl'
    print '\t\tPickling the list of tomograms in the file: ' + out_pkl
    try:
        list_tomos.pickle(out_pkl)
        kwargs = {'_psPickleFile': out_pkl}
        star_out.add_row(**kwargs)
    except pexceptions.PySegInputError as e:
        print 'ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)

    out_app = out_dir + '/' + star_stem + '_app'
    if not os.path.exists(out_app):
        os.makedirs(out_app)
    print '\tStoring particles grouped by tomograms: ' + out_app
    for tomo in list_tomos.get_tomo_list():
        if tomo.get_num_particles() > 0:
            tomo_fname = tomo.get_tomo_fname().replace('/', '_')
            tomo_vtp = tomo.append_particles_vtp(mode='surface')
            # tomo_vtp = poly_swapxy(tomo.append_particles_vtp(mode='surface'))
            disperse_io.save_vtp(tomo_vtp, out_app+'/'+tomo_fname+'.vtp')

out_parts = out_dir + '/' + out_stem + '_parts.star'
print '\tStoring the particles STAR file: ' + out_parts
set_lists.to_particles_star().store(out_parts)

print '\tStoring list appended by tomograms in: ' + out_dir
tomos_vtp = set_lists.tomos_to_vtp(mode='surface')
for key, poly in zip(tomos_vtp.iterkeys(), tomos_vtp.itervalues()):
    stem_tomo = os.path.splitext(os.path.split(key)[1])[0]
    disperse_io.save_vtp(poly, out_dir+'/'+stem_tomo+'_lists_app.vtp')

# out_star = out_dir + '/' + out_stem + '_ltomos.star'
# print '\tOutput STAR file: ' + out_star
# star_out.store(out_star)

# # TODEBUG
# appender.Update()
# disperse_io.save_vtp(appender.GetOutput(), out_dir + '/' + flt_stem + '_debug.vtp')

# print '\tStoring list appended by tomograms in: ' + out_dir
# tomos_vtp = set_lists.tomos_to_vtp(mode='surface')
# for key, poly in zip(tomos_vtp.iterkeys(), tomos_vtp.itervalues()):
#     stem_tomo = os.path.splitext(os.path.split(key)[1])[0]
#     disperse_io.save_vtp(poly, out_dir+'/'+stem_tomo+'_lists_app.vtp')

out_star = out_dir + '/' + out_stem + '_ltomos.star'
print '\tOutput STAR file: ' + out_star
star_out.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'
