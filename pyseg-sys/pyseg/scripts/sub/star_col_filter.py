"""

    Filter rows in a STAR file by column

    Input:  - STAR file
            - Column filter setings

    Output: - A copy of input STAR file where some rows can be filtered

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import pyseg as ps

key_col, key_img = '_rlnClassNumber', '_rlnImageName'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/in_situ_er' # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils'

# Input star file (file to be filtered)
in_star = ROOT_PATH + '/mb_klass/klass_1_sp4_3d_dmh.star' # '/rln/nali_sup_klass/mb_unoriented/run1_c10_it006_data.star' # '/az/ref_1/ref_1_20_50_12.star'

# Classes star file (it can be the same as the filtered class)
out_star = ROOT_PATH + '/mb_klass/klass_1_sp4_3d_dmh_dose_flt.star' # '/de_novo_klass_3d/mb_unorionted_h_nali_dose_flt.star' # '/de_novo_klass_3d/mb_unrionted_h_nali_no_dose_flt.star' # '/az/ref_1/ref_1_20_50_12_gap_14_cont.star'

# Filter definition: colum, mode, list of values
flt_op = ('_rlnMicrographName', 'end-in', '-dose_filt.rec')

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import sys
import time
import copy

########## Print initial message

print 'Filtering STAR file by column values.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tFile to filter: ' + in_star
print '\tOutput file: ' + out_star
print '\tFiltering settings: '
print '\t\t-Column: ' + flt_op[0]
print '\t\t-Mode: ' + flt_op[1]
print '\t\t-Values: ' + str(flt_op[2])
print ''

######### Process

print 'Main Routine: '

print '\tLoading input STAR files...'
star = ps.sub.Star()
try:
    star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be read because of "' + str(e.msg, e.expr) + '"'
    sys.exit(-1)
star_out = copy.deepcopy(star)

print '\tLoop for processing the row...'
hold_ids = list()
if (flt_op[1] == 'end-out') or (flt_op[1] == 'end-in'):
    idx_s = len(flt_op[2])
    for row in xrange(star.get_nrows()):
        hold_val = star.get_element(flt_op[0], row)
        short_val = hold_val[len(hold_val)-idx_s:]
        if flt_op[1] == 'end-in':
            if short_val != flt_op[2]:
                hold_ids.append(row)
        elif flt_op[1] == 'end-out':
            if short_val == flt_op[2]:
                hold_ids.append(row)
else:
    for row in xrange(star.get_nrows()):
        if star.get_element(flt_op[0], row) in flt_op[2]:
            if flt_op[1] == 'out':
                hold_ids.append(row)
        else:
            if flt_op[1] == 'in':
                hold_ids.append(row)

print '\tDeleting ' + str(len(hold_ids)) + ' rows of ' + str(star.get_nrows()) + ' ...'
star_out.del_rows(hold_ids)

print '\t\t-Storing the results in: ' + out_star
star_out.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'