"""

    Add the particles in a ListTomoParticle to another

    Input:  - ListTomoParticle STAR file where particles are added
            - ListTomoParticle STAR file from where particles are read

    Output: - A ListTomoPaticles with the particles of the two input lists

"""

################# Package import

import os
import vtk
import numpy as np
import scipy as sp
import sys
import time
import math
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg import globals as gl
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils' # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre' # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/ves_40'

# Input STAR files
in_star_1 = ROOT_PATH + '/pre/ref_nomb_1_clean/ltomos_clst_flt_high/pre_gather_clst_flt_high_ltomos.star' # '/ltomos_premb_mask/premb_mask_ltomos.star'
in_star_2 = ROOT_PATH + '/ves_40/ltomos_premb_mask/premb_mask_ltomos.star'

# Output directory
out_dir = ROOT_PATH + '/ves_40/ltomos_add_pre_nolap' # '/ref_nomb_1_clean/ltomos_clst_flt_high_lap' # '/ltomos_lap'
out_stem = 'pre_ves_40_add_nolap' # 'clst_flt_high_lap' # 'lap'

# Segmentation pre-processing
sg_bc = False
sg_bm = 'box'
sg_pj = True
sg_voi_mask = True

# Post-processing
pt_res = 0.684
pt_ss = 5 # nm
pt_min_parts = 0
pt_keep = None


########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Gathering two ListTomoParticles')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tInput STAR file with the ListTomoParticle 1: ' + str(in_star_1))
print('\tInput STAR file with the ListTomoParticle 2: ' + str(in_star_2))
print('\tSegmentation pre-processing: ')
if sg_bc:
    print('\t\t-Checking particles VOI boundary with mode: ' + str(sg_bm))
if sg_pj:
    print('\t\t-Activated particles projecting on surface VOI.')
if sg_voi_mask:
    print('\t\t-Mask VOI mode activated!')
print('\tPost-processing: ')
if pt_ss is not None:
    pt_ss_v = pt_ss / pt_res
    print('\t\t-Scale suppression: ' + str(pt_ss) + ' nm (' + str(pt_ss_v) + ' voxels)')
print('\t\t-Keep tomograms the ' + str(pt_keep) + 'th with the highest number of particles.')
print('\t\t-Minimum number of particles: ' + str(pt_min_parts))
print('')

######### Process

print('Main Routine: ')

print('\tLoading input ListTomoParticles...')
set_lists = surf.SetListTomoParticles()
star_1, star_2, star_out = sub.Star(), sub.Star(), sub.Star()
try:
    star_1.load(in_star_1)
    star_2.load(in_star_2)
    star_out.add_column('_psPickleFile')
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('\tGeting short keys for list 2:')
lists_2_hash = dict()
for list_pkl_2 in star_2.get_column_data('_psPickleFile'):
    print('\t\t-Processing list 2: ' + list_pkl_2)
    short_key_idx = list_pkl_2.index('_')
    short_key = list_pkl_2[:short_key_idx]
    print('\t\t\t+Short key found: ' + short_key)
    lists_2_hash[short_key] = list_pkl_2


print('\tLOOP FOR LISTS: ')
list_pkl_1 = star_2.get_column_data('_psPickleFile')
set_lists = surf.SetListTomoParticles()
for list_pkl_1 in star_1.get_column_data('_psPickleFile'):

    print('\t\t-Processing list 1: ' + list_pkl_1)
    ltomos_1 = gl.unpickle_obj(list_pkl_1)
    short_key_idx = list_pkl_1.index('_')
    short_key = list_pkl_1[:short_key_idx]
    print('\t\t\t+Short key found: ' + short_key)

    try:
        ltomos_2 = gl.unpickle_obj(lists_2_hash[short_key])
    except KeyError:
        print('\tWARNING: list 2 does not have short key list ' + short_key)
        continue
    tomos_fname_2 = ltomos_2.get_tomo_fname_list()

    for tomo_1 in ltomos_1.get_tomo_list():

        tomo_fname = tomo_1.get_tomo_fname()
        print('\t\tProcessing tomogram: ' + str(tomo_fname))
        try:
            tomo_2 = ltomos_2.get_tomo_by_key(tomo_fname)
        except KeyError:
            print('\tWARNING: tomo ' + tomo_fname + ' is not present in list 2')
            continue
        print('\t\t\t-Inserting particle from tomo counterpart in list 2...')
        for part_2 in tomo_2.get_particles():
            try:
                if isinstance(part_2, surf.Particle):
                    part_1 = surf.Particle(part_2.get_vtp(), center=part_2.get_center(), normal=part_2.get_normal())
                else:
                    part_1 = surf.ParticleL(part_2.get_vtp_fname(), center=part_2.get_center(), eu_angs=part_2.get_eu_angs())
                ltomos_1.insert_particle(part_1, tomo_fname, check_bounds=sg_bc, mode=sg_bm, voi_pj=sg_pj)
            except pexceptions.PySegInputError as e:
                print('WARINING: particle ' + str(part_2.get_center()) + ' could not be inserted in tomogram ' + tomo_fname + \
                        ' because of "' + e.get_message() + '"')
                pass

    if pt_keep is not None:
        print('\t\tFiltering to keep the ' + str(pt_keep) + 'th more highly populated')
        ltomos_1.clean_low_pouplated_tomos(pt_keep)
    if pt_min_parts >= 0:
        print('\t\tFiltering tomograms with less particles than: ' + str(pt_min_parts))
        ltomos_2.filter_by_particles_num(pt_min_parts)

    star_stem = os.path.splitext(os.path.split(list_pkl_1)[1])[0]
    out_pkl = out_dir + '/' + star_stem + '_tpl.pkl'
    print('\t\tPickling the list of tomograms in the file: ' + out_pkl)
    try:
        ltomos_1.pickle(out_pkl)
        kwargs = {'_psPickleFile': out_pkl}
        star_out.add_row(**kwargs)
    except pexceptions.PySegInputError as e:
        print('ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    # Adding particle to list
    set_lists.add_list_tomos(ltomos_1, star_stem)

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
