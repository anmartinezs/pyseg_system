"""


    Avoid double picking by forcing particles within the same tomogram to stay in the same half set

    Input:  - Source STAR file

    Output: - A STAR file where particles within the same tomogram are forced to go to the same half set,

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

################# Package import

import os
import sys
import time
import copy
import pyto
import pyseg as ps
import numpy as np

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn2' # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst' # '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext_repick/ribo'

# Input STAR
in_star = ROOT_PATH + '/rln/class2_model_inv/run1_c2_HA_mb_it035_data.star' # '/ampar_vs_nmdar/ampar/loc_ref_4uqj_swapxy/run2_c1_r12_data.star' # '/rln_class2_trans/run4_it035_data_ribo_trap.star'

# Output STAR file
out_star = ROOT_PATH + '/rln/class2_model_inv/run1_c2_HA_mb_it035_data_c1.star' # '/ampar_vs_nmdar/ampar/loc_ref_4uqj_swapxy/run2_c1_r12_data_noset.star' # '/rln_class2_trans/run4_it035_data_ribo_trap_dpick.star'

# Input parameters
del_subset = True
set_groups = False
set_halves = False # True
set_one_group = None # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/in/rln/tomos/syn_11_2_bin2/syn_11_2_bin2.mrc'
keep_classes = [1, ] # None #

########################################################################################
# Global functions
########################################################################################

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Fix random halves by tomograms.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tSource STAR file: ' + str(in_star))
print('\tOutput STAR file: ' + str(out_star))
print('\tOptions:')
if del_subset:
    if del_subset:
        print('\t\t-Delete random subset column.')
    if set_groups:
        print('\t\t-Set all particles to group 1.')
    if set_halves:
        print('\t\t-Separate halves by tomograms')
    if keep_classes is not None:
        print('\t\t-Keep the next classes: ' + str(keep_classes))
print('')

######### Process

print('Main Routine: ')

print('\tLoading input STAR files...')
star = ps.sub.Star()
try:
    star.load(in_star)
    print('\t\t-Number of particles: ' + str(star.get_nrows()))
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input list of STAR files could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

if del_subset:
    print('\tDeleting random subset column...')
    if star.has_column('_rlnRandomSubset'):
        star.del_column('_rlnRandomSubset')

if set_groups:
    if set_halves:
        if not star.has_column('_rlnGroupNumber'):
            star.add_column('_rlnGroupNumber')
        star.set_column_data('_rlnGroupNumber', np.ones(shape=star.get_nrows(), dtype=np.int))
    else:
        if star.has_column('_rlnGroupNumber'):
            star.del_column('_rlnGroupNumber')

if set_halves:
    if not star.has_column('_rlnRandomSubset'):
        star.add_column('_rlnRandomSubset')
    print('\tComputing tomograms dictionary...')
    parts_mic = dict()
    mics = star.get_column_data('_rlnMicrographName')
    mic_keys = set(mics)
    for mic in mic_keys:
        parts_mic[mic] = mics.count(mic)
    mic_keys_sorted = sorted(parts_mic.keys())[::-1]

    print('\tSorted alternated half assignment...')
    half = 1
    parts_half = dict().fromkeys(mic_keys, 0)
    for mic in mic_keys_sorted:
        for row in range(star.get_nrows()):
            if star.get_element('_rlnMicrographName', row) == mic:
                star.set_element('_rlnRandomSubset', row, half)
        if half == 1:
            half = 2
        else:
            half = 1

if set_one_group is not None:
    for row in range(star.get_nrows()):
        star.set_element('_rlnMicrographName', row, set_one_group)
    star.del_column('_rlnGroupNumber')

if (keep_classes is not None) and (star.has_column('_rlnClassNumber')):
    row_ids = list()
    for row in range(star.get_nrows()):
        if not(star.get_element('_rlnClassNumber', row) in keep_classes):
            row_ids.append(row)
    star.del_rows(row_ids)

print('Storing output STAR file in: ' + out_star)
star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')
