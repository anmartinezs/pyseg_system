"""

    Gather several STAR files with into one updating the class number

    Input:  - A STAR file with the list STAR files to gather

    Output: - A a new STAR file where the class number take into account the different input files

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

################# Package import

import os
import sys
import time
import pyseg as ps
import numpy as np

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/radu'

# Input STAR file with the list of files to gather or a directory
in_star = ROOT_PATH + '/class/noali_all_2/merge_big_gap'

# Input STAR file with segmentation information to focus the masks
in_seg = None # ROOT_PATH + '/two_segmentations.star'

# Output file
out_star = ROOT_PATH + '/class/noali_all_2/class_1_ap_r_ali_mbclean_merge_big_gap.star'

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('STAR files gather.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
in_star_ext = os.path.splitext(in_star)[1]
if in_star_ext == '':
    print('\tInput directory to look for STAR files: ' + str(in_star))
elif in_star_ext == '.star':
    print('\tInput with the list of STAR files: ' + str(in_star))
else:
    print('ERROR: unrecognized input, it must be a STAR file or a directory.')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
print('\tOutput STAR file: ' + str(out_star))
print('')

######### Process

print('Main Routine: ')

star = ps.sub.Star()

if in_star_ext == '':
    print('\tLooking STAR files in the input directory...' + str(in_star))
    star_l = [f for f in os.listdir(in_star) if (os.path.isfile(os.path.join(in_star, f)) and f.endswith('.star'))]
    nrows = len(star_l)
else:
    print('\tLoading input STAR file list...')
    star_l = ps.sub.Star()
    try:
        star_l.load(in_star)
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: input list of STAR files could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
    nrows = star_l.get_nrows()

print('\tLoop for input STAR files:')
for row in range(nrows):

    if in_star_ext == '':
        hold_fname = in_star + '/' + star_l[row]
    else:
        hold_fname = star_l.get_element('_psSTARFile', row)
    print('\t\t-Reading STAR file: ' + hold_fname)
    hold_star = ps.sub.Star()
    try:
        hold_star.load(hold_fname)
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: the STAR file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    if in_star_ext == '':
        klass = (row+1) * np.ones(shape=hold_star.get_nrows(), dtype=np.int)
    else:
        klass = star_l.get_element('_rlnClassNumber', row)
    print('\t\t-Setting class number...')
    hold_star.set_column_data('_rlnClassNumber', klass)

    print('\t\t-Appending (' + str(hold_star.get_nrows()) + ') rows...')
    if row == 0:
        for key in hold_star.get_column_keys():
            star.add_column(key)
    for hold_row in range(hold_star.get_nrows()):
        kwargs = dict()
        for key in hold_star.get_column_keys():
            kwargs[key] = hold_star.get_element(key, hold_row)
        star.add_row(**kwargs)

print('Final number of rows: ' + str(star.get_nrows()))

print('Storing output STAR file in: ' + out_star)
star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')
