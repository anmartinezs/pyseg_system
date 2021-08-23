"""

    Generates a ListTomoOMSegmentations object from a input STAR file

        Input:  - STAR file with the list of segmentations, each list pairs membrane and lumen segmentations

    Output: - A ListTomoOMSegmentations pickled object for every STAR file row

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
in_star = ROOT_PATH + '/in/in_ltomos_omsegs.star' # '/in/in_ltomos_fils_new_omsegs.star' # '/in/in_ltomos_fil_ctrl_omsegs.star'

# Output directory
out_dir = ROOT_PATH + '/ltomos_omsegs/omsegs_all' # '/ltomos_omsegs/omsegs_fil_new' # '/ltomos_omsegs/omsegs_fil_ctrl' # '/ltomos_omsegs/omsegs_1' # '/stat/ltomos/trans_run2_test_swapxy'
out_stem = 'all' # 'fil_ctrl' # 'omsegs' # 'pre'

# Segmentation pre-processing
sg_lbl = 1 # segmented label
sg_max_rad = 10 # nm

#### Advanced settings

# Post-processing
pt_min_segs = 1 # Tomograms with less segmentations are removed
pt_min_by_tomo = False # If True particle from all patterns are considered
pt_keep = None # To keep just the indicated highly populated tomograms

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Generation of Surface objects from and input STAR file or tomogram for segmentations.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tInput STAR file of segmentations: ' + str(in_star))
print('\tSegmentation pre-processing: ')
print('\t\t-Segmentation label: ' + str(sg_lbl))
print('\t\t-Keep tomograms the ' + str(pt_keep) + 'th with the highest number of segmentations.')
print('\t\t-Minimum number of segmentations: ' + str(pt_min_segs))
if pt_min_by_tomo:
    print('\t\t-Num. segmentations computation by tomograms.')
print('')

######### Process

print('Main Routine: ')

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

print('\tLoop for tomograms in the list: ')
segs_inserted = 0
set_lists = surf.SetListTomoOMSegmentations()
for star_row in range(star.get_nrows()):

    print('\t\tNow list of tomograms initialization...')
    list_tomos = surf.ListTomoOMSegmentations()
    seg_star_str = star.get_element('_psStarFile', star_row)

    print('\t\tLoading segmentations STAR file(s):')
    star_seg = sub.Star()
    try:
        star_seg.load(seg_star_str)
    except pexceptions.PySegInputError as e:
        print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    print('\t\tFilaments loop..')
    for seg_row in range(star_seg.get_nrows()):

        segs_lm_str = star_seg.get_element('_omLmSegmentation', seg_row)
        print('\t\t\tLoading lumen segmentations file: ' + segs_lm_str)
        try:
            seg_lm = disperse_io.load_tomo(segs_lm_str) == sg_lbl
        except pexceptions.PySegInputError as e:
            print('ERROR: input lumen segmentations file could not be loaded because of "' + e.get_message() + '"')
            print('Terminated. (' + time.strftime("%c") + ')')
            sys.exit(-1)
        if star_seg.has_column('_omMbSegmentation'):
            segs_mb_str = star_seg.get_element('_omMbSegmentation', seg_row)
            print('\t\t\tLoading membrane segmentations file: ' + segs_mb_str)
            try:
                seg_mb = disperse_io.load_tomo(segs_mb_str) == sg_lbl
            except pexceptions.PySegInputError as e:
                print('ERROR: input membrane segmentations file could not be loaded because of "' + e.get_message() + '"')
                print('Terminated. (' + time.strftime("%c") + ')')
                sys.exit(-1)
        mic_str = star_seg.get_element('_rlnMicrographName', seg_row)
        mic = disperse_io.load_tomo(mic_str, mmap=True)
        sg_res = star_seg.get_element('_psPixelSize', seg_row)
        if sg_res <= 0:
            print('WARNING: Pixel size for a tomogram must be greater than zero:' + mic_str)
            continue
        sg_max_rad_v = sg_max_rad / sg_res
        list_tomos.set_resolution(sg_res)

        # Insert the new particle in the proper tomogram
        tomo_fname = os.path.splitext(os.path.split(mic_str)[1])[0]
        try:
            list_tomos.insert_tomo(tomo_fname, voi_mb=seg_mb, voi_lm=seg_lm, max_dst=sg_max_rad_v)
            segs_inserted += 1
        except pexceptions.PySegInputError as e:
            print('WARNING: segmentations in row ' + str(seg_row) + ' could not be inserted in tomogram ' + \
                  tomo_fname + ' because of "' + e.get_message() + '"')

    if pt_keep is not None:
        print('\t\tFiltering to keep the ' + str(pt_keep) + 'th more highly populated')
        list_tomos.clean_low_pouplated_tomos(pt_keep)
    if not pt_min_by_tomo:
        if pt_min_segs >= 0:
            print('\t\tFiltering tomograms with less segmentations than: ' + str(pt_min_segs))
            list_tomos.filter_by_segmentations_num(pt_min_segs)

    # Adding ListTomograms to Set
    seg_stem = os.path.splitext(os.path.split(seg_star_str)[1])[0]
    set_lists.add_list_tomos(list_tomos, seg_stem)

for star_row in range(star.get_nrows()):

    fil_star_str = star.get_element('_psStarFile', star_row)
    lfil_stem = os.path.splitext(os.path.split(fil_star_str)[1])[0]
    list_tomos = set_lists.get_lists_by_key(lfil_stem)
    out_pkl = out_dir + '/' + lfil_stem + '_tpl.pkl'
    print('\t\tPickling the list of tomograms in the file: ' + out_pkl)
    try:
        list_tomos.pickle(out_pkl)
        kwargs = {'_psPickleFile': out_pkl}
        star_out.add_row(**kwargs)
    except pexceptions.PySegInputError as e:
        print('ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    out_app = out_dir + '/' + lfil_stem + '_app'
    if not os.path.exists(out_app):
        os.makedirs(out_app)
    print('\tStoring segmentations grouped by tomograms: ' + out_app)
    for tomo in list_tomos.get_tomo_list():
        if tomo.get_num_segmentations() > 0:
            tomo_fname = tomo.get_tomo_name().replace('/', '_')
            hold_tomo = tomo.get_lbl_voi(mode='mb')
            disperse_io.save_numpy(hold_tomo, out_app+'/'+tomo_fname+'_mb.mrc')
            hold_tomo = tomo.get_lbl_voi(mode='lm')
            disperse_io.save_numpy(hold_tomo, out_app + '/' + tomo_fname + '_lm.mrc')

print('\tTotal number of segmentations inserted (before post-processing): ' + str(segs_inserted))

out_star = out_dir + '/' + out_stem + '_ltomos.star'
print('\tOutput STAR file: ' + out_star)
star_out.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')
