"""

    Generates a ListTomoParticle object from a input STAR file (translocon project version)

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
import shutil
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg import globals as gl
from pyorg.surf.utils import poly_swapxy

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-engel/antonio/rubisco/org'

# Input STAR file
in_star = ROOT_PATH + '/in/in_parts_test_v2.star' # '/in/in_parts_all_iso1.7.star' # '/in/in_parts_all.star'

# Input STAR for with the sub-volumes segmentations
in_seg = ROOT_PATH + '/in/in_seg_test2.star' # '/in/in_seg_all.star' # '/in/in_seg_all.star'

# Output directory
out_dir = ROOT_PATH + '/ltomos/test_v2' # '/ltomos/all_iso1.7' # '/ltomos/all_pid' # '/stat/ltomos/trans_run2_test_swapxy'
out_stem = 'test_remove_parts' # 'all_iso1.7' # 'pre'

pt_res = 1.368 # nm/vx - resolution

#### Advanced settings

# Segmentation pre-processing
sg_lbl = 1 # segmented label
sg_bc = True # False
sg_bm = 'center' # Embedding checking mode
sg_pj = False # Project particles to VOI's surfaces
sg_origins = (2,) # If not None, subtomoavg shiftings are considered,
                    # then scale factor (one per each pattern in in_star) from picked particle to subtomograms
sg_swap_xy = False # Swap X and Y coordinates of the input particle STAR files
sg_voi_surf = False # It forces to convert VOI to a surface (vtkPolyData object)
sg_sg = 0 # Gaussian filtering for surface conversion
sg_dec = 0.9 # Decimation factor for the triangle mesh

# Post-processing
pt_ssup = 5 # nm - scale suppression for the input particles
pt_ss_ref = None # a tuple with the sorted preference for crossed patterns scaled suppresion, if None deactivated
pt_ss_ref_dst = None # for using different scale-suppression distances for crossed scale suppression
pt_min_parts = 1 # Tomograms with less particles are removed
pt_min_by_tomo = False # If True particle from all patterns are considered
pt_keep = None # To keep just the indicated highly populated tomograms
pt_pparts = False # Print the particles found in an output tomogram
pt_mode_over = False # If True particles of all parttern are printed overlapped in the same scalar field
pt_pparts_sg = 0.8 # To Gaussian filter the printed particles

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Generation of Surface objects from and input STAR file or tomogram (light version).'
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
if sg_origins is not None:
    print '\t\t-Scale factor for subtomo-averaging origins: ' + str(sg_origins)
if sg_voi_surf:
    print '\t\t-Forcing surface VOI mode activated!'
if sg_swap_xy:
    print '\t\t-Swap X and Y particle coordinates.'
print '\tPost-processing: '
print '\t\t-Resolution: ' + str(pt_res) + ' nm/vx'
print '\t\t-Scale suppression: ' + str(pt_ssup) + ' nm'
if (pt_ss_ref is not None) and (pt_ss_ref_dst is not None):
    print '\t\t-Reference key for crossed scale suppresion: ' + str(pt_ss_ref)
    print '\t\t-Crossed scale suppresion: ' + str(pt_ss_ref_dst) + ' nm'
print '\t\t-Keep tomograms the ' + str(pt_keep) + 'th with the highest number of particles.'
print '\t\t-Minimum number of particles: ' + str(pt_min_parts)
if pt_min_by_tomo:
    print '\t\t-Num. particles computation by tomograms.'
if pt_pparts:
    print '\tPrinting particles: '
    if pt_mode_over:
        print '\t\t-Particles are overlapped in the same label field.'
    else:
        print '\t\t-Particles are stored in different label fields.'
    if pt_pparts_sg:
        print '\t\t-Sigma for Gaussian filtering the scalar density field for printed particles: ' + str(pt_pparts_sg) + ' vx'
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
seg_dic, seg_dic_fname = dict(), dict()
for seg_row in range(star_seg.get_nrows()):
    mic_str, seg_str = star_seg.get_element('_rlnMicrographName', seg_row), \
                       star_seg.get_element('_psSegImage', seg_row)
    seg_dic[mic_str] = seg_row
    seg_dic_fname[os.path.split(mic_str)[1]] = seg_row

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
if star.get_nrows() != len(sg_origins):
    print 'ERROR: the length of input factors for binning origins must have the same dimension than input particle Star files.'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\tLoop for generating tomograms VOIs: '
vois, nvois = dict(), dict()
for tomo_row in range(star_seg.get_nrows()):
    mic_str, seg_str = star_seg.get_element('_rlnMicrographName', tomo_row), \
                       star_seg.get_element('_psSegImage', tomo_row)
    print '\t\t-Generating VOI from segmentation: ' + str(mic_str)
    tomo = disperse_io.load_tomo(seg_str, mmap=True).swapaxes(0, 1)
    if sg_voi_surf:
        tomo = (tomo == sg_lbl).astype(np.float32)
        if (sg_sg is not None) and (sg_sg > 0):
            tomo = sp.ndimage.filters.gaussian_filter(tomo, sg_sg)
        voi = surf.iso_surface(tomo, 0.5, closed=True, normals='outwards')
        if sg_dec is not None:
            voi = surf.poly_decimate(voi, sg_dec)
        seg_fname = os.path.splitext(seg_str.replace('/', '_'))[0]
        out_voi = out_dir + '/' + seg_fname + '_surf_voi.vtp'
        vois[seg_str] = out_voi
    else:
        voi = tomo == sg_lbl
        # seg_fname = os.path.split(os.path.splitext(seg_str)[0])[1]
        seg_fname = os.path.splitext(seg_str.replace('/', '_'))[0]
        out_voi = out_dir + '/' + seg_fname + '_mask_voi.mrc'
        # print out_voi
        disperse_io.save_numpy(voi, out_voi)
        # voi = disperse_io.load_tomo(out_voi, mmap=True)
        vois[seg_str] = out_voi
    nvois[seg_str] = 0

print '\tLoop for tomograms in the list: '
parts_inserted = 0
shape_paths, seg_mic_dir = dict(), dict()
set_lists = surf.SetListTomoParticles()
for star_row in range(star.get_nrows()):

    print '\t\tNow list of tomograms initialization...'
    list_tomos = surf.ListTomoParticles()
    part_star_str, part_surf_str = star.get_element('_psStarFile', star_row), \
                                   star.get_element('_suSurfaceVtp', star_row)
    for tomo_fname, voi in zip(vois.iterkeys(), vois.itervalues()):
        if isinstance(voi, str):
            voi_ext = os.path.splitext(voi)[1]
            if voi_ext == '.mrc':
                voi = disperse_io.load_tomo(voi, mmap=True)
            elif voi_ext == '.vtp':
                voi = disperse_io.load_poly(voi)
            else:
                print 'ERROR: ' + voi_ext + ' not recognized extension for VOI ' + voi
                print 'Terminated. (' + time.strftime("%c") + ')'
                sys.exit(-1)
        hold_tomo = surf.TomoParticles(tomo_fname, sg_lbl, voi=voi)
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
    if not isinstance(part_vtp, vtk.vtkPolyData):
        print 'ERROR: file ' + part_surf_str + ' is not a vtkPolyData!'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    if not surf.is_closed_surface(part_vtp):
        print 'ERROR: file ' + part_surf_str + ' is not a closed surface!'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    if not star_part.is_column('_rlnLogLikeliContribution'):
        star_part.add_column('_rlnLogLikeliContribution', val=0)

    print '\t\tParticles loop..'
    part_lut = np.zeros(shape=star_part.get_nrows(), dtype=np.bool)
    part_coords = np.zeros(shape=(star_part.get_nrows(), 3), dtype=np.float32)
    for part_row in range(star_part.get_nrows()):

        # Initialization
        mic_str = star_part.get_element('_rlnMicrographName', part_row)
        try:
            seg_row = seg_dic[mic_str]
        except KeyError:
            print 'WARNING: Micrograph not found: ' + mic_str
            continue
            # seg_row = seg_dic_fname[mic_str]
            # mic_str = star_seg.get_element('_rlnMicrographName', seg_row)
        seg_str = star_seg.get_element('_psSegImage', seg_row)
        seg_mic_dir[seg_str] = mic_str
        mic = disperse_io.load_tomo(mic_str, mmap=True)
        if sg_swap_xy:
            (cy, cx, cz), (rho, tilt, psi) = star_part.get_particle_coords(part_row, orig=sg_origins[star_row], rots=True)
        else:
            (cx, cy, cz), (rho, tilt, psi) = star_part.get_particle_coords(part_row, orig=sg_origins[star_row],
                                                                           rots=True)
        part_center, part_eu_angs = np.asarray((cx, cy, cz), dtype=np.float32), \
                                    np.asarray((rho, tilt, psi), dtype=np.float32)

        # Segmentation rigid body transformations
        if sg_swap_xy:
            mic_center = np.asarray((.5*mic.shape[1], .5*mic.shape[0], .5*mic.shape[2]), dtype=np.float32)
        else:
            mic_center = np.asarray((.5 * mic.shape[0], .5 * mic.shape[1], .5 * mic.shape[2]), dtype=np.float32)

        # Centering
        part_center -= mic_center
        # Un-rotation
        seg_rot, seg_tilt, seg_psi = star_seg.get_element('_psSegRot', seg_row), \
                                     star_seg.get_element('_psSegTilt', seg_row), \
                                     star_seg.get_element('_psSegPsi', seg_row)
        if sg_swap_xy:
            seg_rot, seg_tilt, seg_psi = -1 * seg_rot, -1 * seg_tilt, -1 * seg_psi
        if (seg_rot != 0) or (seg_tilt != 0) or (seg_psi != 0):
            R1 = gl.rot_mat_relion(part_eu_angs[0], part_eu_angs[1], part_eu_angs[2])
            R2 = gl.rot_mat_relion(seg_rot, seg_tilt, seg_psi)
            Rt = R2 * R1
            part_eu_angs = np.asarray(gl.rot_mat_eu_relion(Rt), dtype=np.float32)
            part_center = np.asarray(R2 * np.matrix(part_center).T, dtype=np.float32).reshape(3)

        # Un-centering
        part_center += mic_center
        # Un-cropping
        seg_offy, seg_offx, seg_offz = star_seg.get_element('_psSegOffX', seg_row), \
                                       star_seg.get_element('_psSegOffY', seg_row), \
                                       star_seg.get_element('_psSegOffZ', seg_row)
        if sg_swap_xy:
            part_center -= np.asarray((seg_offy, seg_offx, seg_offz), dtype=np.float32)
        else:
            part_center -= np.asarray((seg_offx, seg_offy, seg_offz), dtype=np.float32)

        # Insert the new particle in the proper tomogram
        try:
            part = surf.ParticleL(part_surf_str, center=part_center, eu_angs=part_eu_angs)
            # part = surf.Particle(part_vtp, center=(0, 0, 0))
            # part.rotation(part_eu_angs[0], part_eu_angs[1], part_eu_angs[2])
            # part.translation(part_center[0], part_center[1], part_center[2])
            list_tomos.insert_particle(part, seg_str, check_bounds=sg_bc, mode=sg_bm, voi_pj=sg_pj)
            parts_inserted += 1
            nvois[seg_str] += 1
        except pexceptions.PySegInputError as e:
            print 'WARNING: particle in row ' + str(part_row) + ' could not be inserted in tomogram ' + tomo_fname + \
                  ' because of "' + e.get_message() + '"'
            pass
        part_lut[part_row] = True
        part_coords[part_row, :] = part_center
    del_l = list(np.where(part_lut == False)[0])
    if pt_ssup is not None:
        pt_ssup_v = pt_ssup / pt_res
        print '\t\tApplying scale suppresion (' + str(pt_ssup_v) + ')...'
        list_tomos.scale_suppression(pt_ssup_v)

        #'Computing tomograms dictionary
        parts_mic = dict()
        for row in range(star_part.get_nrows()):
            mic = star_part.get_element('_rlnMicrographName', row)
            try:
                seg_dic[mic]
                try:
                    parts_mic[mic].append(row)
                except KeyError:
                    parts_mic[mic] = list()
                    parts_mic[mic].append(row)
            except KeyError:
                del_l.append(row)
        # Particle suppression on output STAR file
        for mic, rows in zip(parts_mic.iterkeys(), parts_mic.itervalues()):
            mic_coords = np.zeros(shape=(len(rows), 3), dtype=np.float32)
            mic_lut = np.ones(shape=len(rows), dtype=np.bool)
            for i, row in enumerate(rows):
                mic_coords[i, :] = part_coords[row, :]
            for i, coord in enumerate(mic_coords):
                row = rows[i]
                if mic_lut[i]:
                    hold = mic_coords - coord
                    dsts = np.sqrt((hold * hold).sum(axis=1))
                    ids = np.where((dsts < pt_ssup_v) & mic_lut)[0]
                    logs = np.zeros(shape=len(ids), dtype=np.float32)
                    for j, idx in enumerate(ids):
                        logs[j] = star_part.get_element('_rlnLogLikeliContribution', rows[idx])
                    m_idx = ids[np.argmax(logs)]
                    # Only clean neighbours when we are place at maximum
                    if i == m_idx:
                        for idx in ids:
                            if mic_lut[idx] and (idx != i):
                                mic_lut[idx] = False
                                del_l.append(rows[idx])
        star_part.del_rows(del_l)
    if pt_keep is not None:
        print '\t\tFiltering to keep the ' + str(pt_keep) + 'th more highly populated'
        list_tomos.clean_low_pouplated_tomos(pt_keep)
    if not pt_min_by_tomo:
        if pt_min_parts >= 0:
            print '\t\tFiltering tomograms with less particles than: ' + str(pt_min_parts)
            list_tomos.filter_by_particles_num(pt_min_parts)

    # Adding ListTomograms to Set
    star_stem = os.path.splitext(os.path.split(part_star_str)[1])[0]
    set_lists.add_list_tomos(list_tomos, star_stem)
    if star.is_column('_suPartShape'):
        shape_paths[star_stem] = star.get_element(key='_suPartShape', row=star_row)

    out_post_part = out_dir + '/' + os.path.splitext(os.path.split(part_star_str)[1])[0] + '_post.star'
    print '\t\tStoring processed STAR file in : ' + out_post_part
    star_part.store(out_post_part)

if (pt_ss_ref is not None) and (pt_ss_ref_dst is not None):
    pt_ss_ref_dst_v = pt_ss_ref_dst / pt_res
    for ss_ref in pt_ss_ref:
        print '\t-Applying crossed scale suppression using ' + str(ss_ref) + \
              ' as reference list: ' + str(pt_ss_ref_dst) + ' nm'
        set_lists.scale_suppression(pt_ss_ref_dst_v, ref_list=ss_ref)

if pt_min_by_tomo:
    if pt_min_parts >= 0:
        print '\t-Filtering lists with less particles than: ' + str(pt_min_parts)
        set_lists.filter_by_particles_num(pt_min_parts)

if pt_pparts:
    if len(shape_paths.keys()) > 0:
        out_ppart_dir = out_dir + '/' + out_stem + '_tomo_pparts'
        if os.path.exists(out_ppart_dir):
            shutil.rmtree(out_ppart_dir)
        os.makedirs(out_ppart_dir)
        print '\t-Saving tomograms with particles printed in folder: ' + out_ppart_dir
        if pt_pparts_sg is not None:
            print '\t\t+ Scalar density field Gaussian smoothed with sigma=' + str(pt_pparts_sg) + 'vx'
        set_lists.print_tomo_particles(out_ppart_dir, shape_paths, seg_mic_dir, pt_mode_over, pt_pparts_sg)

for star_row in range(star.get_nrows()):

    part_star_str, part_surf_str = star.get_element('_psStarFile', star_row), \
                                   star.get_element('_suSurfaceVtp', star_row)
    star_stem = os.path.splitext(os.path.split(part_star_str)[1])[0]
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

print '\tTotal number of particles inserted (before post-processing): ' + str(parts_inserted)

hold_sum = 0
for key, val in zip(nvois.iterkeys(), nvois.itervalues()):
    print '\t-Particles in VOI [' + str(key) + ']: ' + str(val)
    hold_sum += val
print '\tSum: ' + str(hold_sum)

out_parts = out_dir + '/' + out_stem + '_parts.star'
print '\tStoring the particles STAR file: ' + out_parts
set_lists.to_particles_star().store(out_parts)

print '\tStoring list appended by tomograms in: ' + out_dir
tomos_vtp = set_lists.tomos_to_vtp(mode='surface')
for key, poly in zip(tomos_vtp.iterkeys(), tomos_vtp.itervalues()):
    stem_tomo = os.path.splitext(os.path.split(key)[1])[0]
    disperse_io.save_vtp(poly, out_dir+'/'+stem_tomo+'_lists_app.vtp')

out_star = out_dir + '/' + out_stem + '_ltomos.star'
print '\tOutput STAR file: ' + out_star
star_out.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'