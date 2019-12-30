"""

    Select randomly a set of rows of an input STAR file

    Input:  - Input STAR file
            - Number of rows for the output STAR file
            - A list with the classes to consider

    Output: - Output STAR file

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

################# Package import

import os
import sys
import time
import pyseg as ps
from scipy.misc import imsave
import numpy as np
import scipy as sp

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/pick_test'

# Input STAR file
in_star = ROOT_PATH + '/klass/class_nali_ps_k2/run2_it050_data.star' # '/klass/class_nali_ps/run4_new_it100_data.star' #

# Output STAR file
out_star = ROOT_PATH + '/klass/class_nali_ps_k2/run2_it050_data_rnd_k1_20.star' # '/klass/class_nali_ps/run4_new_it100_data_rnd_k1345_180.star' #

# Settings
out_rows = 20 # 180 # 1800 # 200
classes = {1,} # {1,3,4,5} # None


########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Select randomly a set of rows of an input STAR file.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput STAR file: ' + in_star
print '\tOutput STAR file: ' + out_star
print '\tSettings:'
print '\t\t-Number of rows: ' + str(out_rows)
if classes is not None:
    print '\t\t-Classes to consider: ' + str(classes)
print ''

######### Process

print 'Main Routine: '

print '\tLoading input STAR file...'
star = ps.sub.Star()
try:
    star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: input list of STAR files could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

if (classes is not None) and star.has_column('_rlnClassNumber'):
    print '\tFiltering rows not in list: ' + str(classes)
    del_ids = list()
    for row in range(star.get_nrows()):
        if not(star.get_element('_rlnClassNumber', row) in classes):
            del_ids.append(row)
    star.del_rows(del_ids)
n_rows = star.get_nrows()
print '\t\t-Number of rows found: ' + str(n_rows)
if out_rows > n_rows:
    print 'ERROR: ' + str(out_rows) + ' rows cannot be returned from a file with ' + str(n_rows) + ' rows!'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\t\t-Getting randomly ' + str(out_rows) + ' rows...'
rnd_ids = np.arange(out_rows)
np.random.shuffle(rnd_ids)
o_star = ps.sub.Star()
for key in star.get_column_keys():
    o_star.add_column(key)
for rnd_id in rnd_ids:
    hold_dic = dict()
    for key in star.get_column_keys():
        hold_dic[key] = star.get_element(key, rnd_id)
    o_star.add_row(**hold_dic)

print '\t\tStoring output STAR file in: ' + out_star
o_star.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'