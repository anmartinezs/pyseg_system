"""

    Template matching on a particles STAR file

    Input:  - The particles STAR file
            - Template matching settings

    Output: - A new column with the template matching scores (_psCCScores) is added to the STAR file

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

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/pick_test' # '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext'

# Input STAR file
in_star = ROOT_PATH + '/klass/klass_ps_cc/klass_5_k4_split_bin4.star' # '/stm/test3/run1_c1_ct_it092_data_k2_low_3000_bin4_lp2.5.star' # '/stm/test2/klass_1_k54_split_bin4_lp6.star' # '/ost_ribo/class_nali/run1_c1_it050_data_c8_bin4.star'

# Output STAR file
out_star = ROOT_PATH + '/klass/klass_ps_cc/klass_5_k4_split_bin4_tm.star' # '/stm/test3/run1_c1_ct_it092_data_k2_low_3000_bin4_lp2.5_tm.star' # '/stm/test2/klass_1_k54_split_bin4_lp6_tm.star' # '/ost_ribo/class_nali/run1_c1_it050_data_c8_bin4_tm_za.star'

# Template matching settings
tm_model = ROOT_PATH + '/nali/translocon_model_shift_bin4.mrc' #  '/ost_ribo/ost_trap_low_resampled_bin4.mrc'
tm_mask = ROOT_PATH + '/masks/mask_cyl_160_55_130_35_r_bin4.mrc' # '/masks/mask_cyl_40_21_34_9.mrc' #'/masks/mask_cyl_40_20_31_10_r.mrc' # '/masks/mask_cyl_80_28_70_20_r.mrc' # '/masks/mask_cyl_80_40_62_20_r.mrc'
tm_res = 1.048 # nm/vx
tm_mshift = 3 # nm
tm_npr = None
tm_mode = '3d' # 'za' #
# 3D mode
tm_rots = range(0, 360, 15)
tm_tilts = np.asarray((0,)) # range(-15, 15, 5)
tm_psis = np.asarray((0,)) # range(-15, 15, 5)
# ZA mode settings

# Model adjusting settings
md_cutoff = 6 # 2.5 # 6 # nm


########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Template matching on a particles STAR file.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput STAR file: ' + in_star
print '\tOutput STAR file: ' + out_star
print '\tTemplate matching settings:'
print '\t\t-Input model: ' + tm_model
print '\t\t-Input mask: ' + tm_mask
print '\t\t-Resolution: ' + str(tm_res) + ' nm/vx'
print '\t\t-Maximum shift: ' + str(tm_mshift) + ' nm'
print '\t\t-Number of processes: ' + str(tm_npr)
if tm_mode == 'za':
    print '\t\t-Z-axis averaged mode selected.'
elif tm_mode == '3d':
    print '\t\t-3D mode selected: '
    print '\t\t\t+Rotation angles to search: ' + str(tm_rots)
    print '\t\t\t+Tilt angles to search: ' + str(tm_tilts)
    print '\t\t\t+Psi angles to search: ' + str(tm_psis)
else:
    print 'ERROR: invalid input mode: ' + str(tm_mode)
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
print '\tModel preprocessing settings:'
print '\t\t-Low pass filtering cutoff: ' + str(md_cutoff) + ' nm'
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
model = ps.disperse_io.load_tomo(tm_model, mmap=False)
if tm_mask is None:
    mask = None
else:
    mask = ps.disperse_io.load_tomo(tm_mask, mmap=False)

if md_cutoff is not None:
    print '\tLow pass filtering input model...'
    nyquist = 2. * tm_res
    rad = (nyquist/8) * np.min(np.asarray(model.shape, dtype=np.float32)) * .5
    lpf = ps.globals.low_pass_fourier_3d(model.shape, tm_res, cutoff=md_cutoff)
    model = np.real(sp.fftpack.ifftn(sp.fftpack.fftn(model) * sp.fftpack.fftshift(lpf)))

print '\tTemplate matching by particles...'
try:
    if tm_mode is 'za':
        star.particles_template_matching_za(model, tm_res, max_shift=tm_mshift, mask=mask, npr=tm_npr)
    else:
        star.particles_template_matching(model, tm_res, max_shift=tm_mshift, mask=mask,
                                         rots=tm_rots, tilts=tm_tilts, psis=tm_psis, npr=tm_npr)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: binning procedure failed because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\t\tStoring output STAR file in: ' + out_star
star.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'