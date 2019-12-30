"""

    Binning a particles STAR file

    Input:  - The particles STAR file
            - Binning parameters

    Output: - A new STAR file with binned particle subvolumes

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

################# Package import

import os
import sys
import time
import pyseg as ps
from scipy.misc import imsave

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/pick_test/klass'   # '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext'

# Input STAR file
in_star = ROOT_PATH + '/klass_ps/klass_5_k4_split.star' # '/whole/class_nali/run1_c1_ct_it092_data_k2_high.star' # '/klass_1/klass_1_k142_split.star' # '/ost_ribo/class_nali/run1_c1_it050_data_c8.star'

# Output STAR file
out_star = ROOT_PATH + '/klass_ps_cc/klass_5_k4_split_bin4.star' # '/stm/test3/run1_c1_ct_it092_data_k2_high_3000_bin4_lp2.5.star' # '/stm/test2/klass_1_k142_split_bin4_lp6.star' # '/ost_ribo/class_nali/run1_c1_it050_data_c8_bin4.star'
out_svol_dir = ROOT_PATH +'/klass_ps_cc/klass_5_k4_bin4'  # '/stm/test3/svols_3000_high_bin4_lp2.5' # '/stm/test2/svols_k142_bin4_lp6' # '/stm/test/svols_ost_ribo_k8/' #

# Binning factor
b_res = 0.262 # nm/vx
b_factor = 4
b_cutoff = 2.5 # nm
b_npr = 6 # None # Auto


########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Binning a particles STAR file.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput STAR file: ' + in_star
print '\tOutput STAR file: ' + out_star
print '\tOutput directory for subvolumes: ' + out_svol_dir
print '\tBinning parameters:'
print '\t\t-Subvolume original resolution: ' + str(b_res) + ' nm/vx'
print '\t\t-Binning factor: ' + str(b_factor)
print '\t\t-Low pass filter cutoff: ' + str(b_cutoff) + ' nm'
print '\t\t-Number of processes: ' + str(b_npr)
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

print '\tBinning particle subvolumes...'
try:
    star_out = star.gen_binned_copy(out_svol_dir, bin=b_factor, res=b_res, cutoff=b_cutoff, npr=b_npr)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: binning procedure failed because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\t\tStoring output STAR file in: ' + out_star
star_out.store(out_star)


print 'Terminated. (' + time.strftime("%c") + ')'