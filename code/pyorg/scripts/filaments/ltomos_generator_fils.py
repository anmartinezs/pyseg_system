"""

    Generates a ListTomoFilaments object from a input STAR file

    Input:  - STAR file for pairing STAR files with filaments and segmentations

    Output: - A ListTomoFilaments pickled object for every STAR file row

"""

################# Package import

import os
import numpy as np
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-ruben/antonio/filaments'

# Input STAR file
in_star = ROOT_PATH + '/in/in_ltomos_test.star'

# Input STAR for with the sub-volumes segmentations
in_seg = ROOT_PATH + '/in/in_seg_test.star'

# Output directory
out_dir = ROOT_PATH + '/ltomos/test' # '/stat/ltomos/trans_run2_test_swapxy'
out_stem = 'test' # 'pre'

# Segmentation pre-processing
sg_lbl = 1 # segmented label
sg_bc = False

# Post-processing
pt_res = 1.408 # nm/vx - resolution
pt_ssup = 5. # nm - scale suppression for the input particles

#### Advanced settings

# Segmentation pre-processing
sg_swap_xy = False # Swap X and Y coordinates of the input particle STAR files
sg_sg = 1.5 # Gaussian filtering for surface conversion
sg_dec = 0.9 # Decimation factor for the triangle mesh

# Post-processing
pt_ss_ref = None # a tuple with the sorted preference for crossed patterns scaled suppresion, if None deactivated
pt_ss_ref_dst = None # for using different scale-suppression distances for crossed scale suppression
pt_min_parts = 1 # Tomograms with less particles are removed
pt_min_by_tomo = False # If True particle from all patterns are considered
pt_keep = None # To keep just the indicated highly populated tomograms

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Generation of Surface objects from and input STAR file or tomogram for filaments.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tInput STAR file of particles: ' + str(in_star)
print '\t\t-Input STAR file for segmentations: ' + str(in_seg)
print '\tSegmentation pre-processing: '
print '\t\t-Segmentation label: ' + str(sg_lbl)
print '\t\t-Segmentation Gaussian smoothing sigma: ' + str(sg_sg)
print '\t\t-Triangle decimation factor: ' + str(sg_dec)
if sg_swap_xy:
    print '\t\t-Swap X and Y particle coordinates.'
if sg_bc:
    print '\t\t-Checking filaments-VOI overlapping.'
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

print '\tLoop for generating tomograms VOIs: '
vois, nvois = dict(), dict()
for tomo_row in range(star_seg.get_nrows()):
    mic_str, seg_str = star_seg.get_element('_rlnMicrographName', tomo_row), \
                       star_seg.get_element('_psSegImage', tomo_row)
    print '\t\t-Generating VOI from segmentation: ' + str(mic_str)
    tomo = disperse_io.load_tomo(seg_str, mmap=True).swapaxes(0, 1)
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
fils_inserted = 0
shape_paths, seg_mic_dir = dict(), dict()
set_lists = surf.SetListTomoParticles()
for star_row in range(star.get_nrows()):

    print '\t\tNow list of tomograms initialization...'
    list_tomos = surf.ListTomoFilaments()
    fils_xml_str, mic_str = star.get_element('_psXMLFile', star_row), star.get_element('_rlnMicrographName', star_row)
    try:
        seg_row = seg_dic[mic_str]
    except KeyError:
        print 'WARNING: Micrograph not found: ' + mic_str
        continue
    seg_str = star_seg.get_element('_psSegImage', seg_row)
    seg_mic_dir[seg_str] = mic_str
    mic = disperse_io.load_tomo(mic_str, mmap=True)
    for tomo_fname, voi in zip(vois.iterkeys(), vois.itervalues()):
        if isinstance(voi, str):
            voi_ext = os.path.splitext(voi)[1]
            if voi_ext == '.mrc':
                voi = disperse_io.load_tomo(voi, mmap=True)
            else:
                print 'ERROR: ' + voi_ext + ' not recognized extension for VOI ' + voi
                print 'Terminated. (' + time.strftime("%c") + ')'
                sys.exit(-1)
        hold_tomo = surf.TomoFilaments(tomo_fname, sg_lbl, voi=voi, surf_g=sg_sg, surf_dec=sg_dec)
        list_tomos.add_tomo(hold_tomo)

    print '\t\tLoading filaments XML file(s):'
    xml_fils = sub.XMLFilaments()
    try:
        xml_fils.load(fils_xml_str)
    except pexceptions.PySegInputError as e:
        print 'ERROR: input XML file could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)

    print '\t\tParticles loop..'
    fils_lut = np.zeros(shape=xml_fils.get_nfils(), dtype=np.bool)
    fils_coords = np.zeros(shape=(xml_fils.get_nfils(), 3), dtype=np.float32)
    for fils_row in range(xml_fils.get_nfils()):

        # Initialization
        hold_fil = surf.Filament(xml_fils.get_fil_coords(fils_row))

        # Segmentation rigid body transformations
        if sg_swap_xy:
            mic_center = np.asarray((.5*mic.shape[1], .5*mic.shape[0], .5*mic.shape[2]), dtype=np.float32)
        else:
            mic_center = np.asarray((.5 * mic.shape[0], .5 * mic.shape[1], .5 * mic.shape[2]), dtype=np.float32)

        # Centering
        hold_fil.translate((-1.) * mic_center)
        # Un-rotation
        seg_rot, seg_tilt, seg_psi = star_seg.get_element('_psSegRot', seg_row), \
                                     star_seg.get_element('_psSegTilt', seg_row), \
                                     star_seg.get_element('_psSegPsi', seg_row)
        if sg_swap_xy:
            seg_rot, seg_tilt, seg_psi = -1 * seg_rot, -1 * seg_tilt, -1 * seg_psi
        if (seg_rot != 0) or (seg_tilt != 0) or (seg_psi != 0):
            hold_fil.rotate(seg_rot, seg_tilt, seg_psi, mode='relion')

        # Un-centering
        hold_fil.translate(mic_center)
        # Un-cropping
        seg_offy, seg_offx, seg_offz = star_seg.get_element('_psSegOffX', seg_row), \
                                       star_seg.get_element('_psSegOffY', seg_row), \
                                       star_seg.get_element('_psSegOffZ', seg_row)
        if sg_swap_xy:
            hold_fil.translate(seg_offy, seg_offx, seg_offz)
        else:
            hold_fil.translate(seg_offx, seg_offy, seg_offz)

        # Insert the new particle in the proper tomogram
        try:
            list_tomos.insert_filament(hold_fil, seg_str, check_bounds=sg_bc)
            fils_inserted += 1
            nvois[seg_str] += 1
        except pexceptions.PySegInputError as e:
            print 'WARNING: particle in row ' + str(fils_row) + ' could not be inserted in tomogram ' + tomo_fname + \
                  ' because of "' + e.get_message() + '"'
            pass
        fils_lut[fils_row] = True

    del_l = list(np.where(fils_lut == False)[0])
    if pt_ssup is not None:
        pt_ssup_v = pt_ssup / pt_res
        print '\t\tApplying scale suppresion (' + str(pt_ssup_v) + ')...'
        list_tomos.scale_suppression(pt_ssup_v)
    if pt_keep is not None:
        print '\t\tFiltering to keep the ' + str(pt_keep) + 'th more highly populated'
        list_tomos.clean_low_pouplated_tomos(pt_keep)
    if not pt_min_by_tomo:
        if pt_min_parts >= 0:
            print '\t\tFiltering tomograms with less particles than: ' + str(pt_min_parts)
            list_tomos.filter_by_particles_num(pt_min_parts)

    # Adding ListTomograms to Set
    xml_stem = os.path.splitext(os.path.split(fils_xml_str)[1])[0]
    set_lists.add_list_tomos(list_tomos, xml_stem)

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

for star_row in range(star.get_nrows()):

    fils_xml_str = star.get_element('_psXMLFile', star_row)
    xml_stem = os.path.splitext(os.path.split(fils_xml_str)[1])[0]
    list_tomos = set_lists.get_lists_by_key(xml_stem)
    out_pkl = out_dir + '/' + xml_stem + '_tpl.pkl'
    print '\t\tPickling the list of tomograms in the file: ' + out_pkl
    try:
        list_tomos.pickle(out_pkl)
        kwargs = {'_psPickleFile': out_pkl}
        star_out.add_row(**kwargs)
    except pexceptions.PySegInputError as e:
        print 'ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)

    out_app = out_dir + '/' + xml_stem + '_app'
    if not os.path.exists(out_app):
        os.makedirs(out_app)
    print '\tStoring filaments grouped by tomograms: ' + out_app
    for tomo in list_tomos.get_tomo_list():
        if tomo.get_num_particles() > 0:
            tomo_fname = tomo.get_tomo_fname().replace('/', '_')
            tomo_vtp = tomo.append_filaments_vtp(mode='surface')
            disperse_io.save_vtp(tomo_vtp, out_app+'/'+tomo_fname+'.vtp')

print '\tTotal number of filaments inserted (before post-processing): ' + str(fils_inserted)

hold_sum = 0
for key, val in zip(nvois.iterkeys(), nvois.itervalues()):
    print '\t-Particles in VOI [' + str(key) + ']: ' + str(val)
    hold_sum += val
print '\tSum: ' + str(hold_sum)

out_star = out_dir + '/' + out_stem + '_ltomos.star'
print '\tOutput STAR file: ' + out_star
star_out.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'
