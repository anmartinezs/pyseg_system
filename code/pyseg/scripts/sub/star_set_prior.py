"""

    In a Relion's STAR file sets prior columns values as a copy of the rotation angles.
    VERY IMPORTANT NOTES: Relion reconstructs move particles to reference, in contrast refinement procedures,
                          so euler angles must ve swapped and inverted for generating the prior information.
                          This also applies to randomization.

    Input:  - STAR file
            - rotation angles to set prior

    Output: - A copy of input STAR file but now prior column values are copied form their for ration angles
              counterparts

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import pyseg as ps

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn2'

# Input star file
# in_star = ROOT_PATH + '/particles_ves_cont.star'
in_star = ROOT_PATH + '/rln/ref_model/run3_c1_LB_mb_data.star' # '/rln/pst/class_blob_ha_v6/run21_c1_nomb_it015_data_c3.star'

# Oput star file stem
# out_star = ROOT_PATH+'/nali/particles_ves_cont_prior_rnd_rot.star'
out_star = ROOT_PATH + '/rln/ref_model/run3_c1_LB_mb_data_prior.star' # '/rln/pst/class_blob_ha_v6/run21_c1_nomb_it015_data_c3_prior_clean.star'

###### Parameters_

do_inv = False
num_copies = 1

# Randomization before priors
do_rnd_rot = False
do_rnd_tilt = False
do_rnd_psi = False

# Priors
do_rot = True
do_tilt = True
do_psi = True

# Angle deletion

del_rot = False
del_tilt = False
del_psi = False

# Cleaning

do_clean = True # Clean some columns introduced during refinement

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import sys
import copy
import numpy as np

########## Helper fucntions

def inv_vals(vals):
    vals_arr = np.asarray(vals, dtype=np.float)
    return (-1.*vals_arr).tolist()

########## Print initial message

print('Setting prior in a STAR file.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput file: ' + in_star)
print('\tOuput file: ' + out_star)
print('\tNumber of instances :' + str(num_copies))
if do_inv:
    print('\t\t-Invert rotations')
print('\tAngles to randomize: ')
if do_rnd_rot:
    print('\t\t-Rotation')
if do_rnd_tilt:
    print('\t\t-Tilt')
if do_rnd_psi:
    print('\t\t-Psi')
print('\tAngles to set prior: ')
if do_rot:
    print('\t\t-Rotation')
if do_tilt:
    print('\t\t-Tilt')
if do_psi:
    print('\t\t-Psi')
print('\tAngles to delete: ')
if del_rot:
    print('\t\t-Rotation')
if del_tilt:
    print('\t\t-Tilt')
if del_psi:
    print('\t\t-Psi')
if do_clean:
    print('\tClean refinement columns.')
print('')

######### Process

print('Main Routine: ')

print('\tLoading input STAR file...')
star = ps.sub.Star()
try:
    star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be read because of "' + str(e.msg) + ', ' + str(e.expr) + '"')
    sys.exit(-1)

if do_inv:
    print('\tInverting rotations...')
    hold_rot = np.array(star.get_column_data('_rlnAngleRot'), dtype=np.float)
    hold_tilt = np.array(star.get_column_data('_rlnAngleTilt'), dtype=np.float)
    hold_psi = np.array(star.get_column_data('_rlnAnglePsi'), dtype=np.float)
    star.add_column('_rlnAngleRot', -1.*hold_psi)
    star.add_column('_rlnAngleTilt', -1.*hold_tilt)
    star.add_column('_rlnAnglePsi', -1.*hold_rot)

if do_rnd_rot:
    print('\tRandomize Rotation angles...')
    try:
        star.rnd_data_column('_rlnAngleRot', -180., 180)
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: copying data for Rotation prior because of "' + str(e.msg) + ', ' + str(e.expr) + '"')
        sys.exit(-2)
if do_rnd_tilt:
    print('\tRandomize Tilt angles...')
    try:
        star.rnd_data_column('_rlnAngleTilt', -180., 180)
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: randomization for Tilt failed because of "' + str(e.msg) + ', ' + str(e.expr) + '"')
        sys.exit(-2)
if do_rnd_psi:
    print('\tRandomize Psi angles...')
    try:
        star.rnd_data_column('_rlnAnglePsi', -180., 180)
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: randomization for Psi failed because of "' + str(e.msg) + ', ' + str(e.expr) + '"')
        sys.exit(-2)

if do_rot:
    print('\tSetting prior for Rotation...')
    try:
        star.copy_data_columns('_rlnAngleRot', '_rlnAngleRotPrior')
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: copying data for Rotation prior failed because of "' + str(e.msg) + ', ' + str(e.expr) + '"')
        sys.exit(-2)
# else:
#     print '\tDeleting previous prior information for Rotation...'
#     star.del_column('_rlnAngleRotPrior')
if do_tilt:
    print('\tSetting prior for Tilt...')
    try:
        star.copy_data_columns('_rlnAngleTilt', '_rlnAngleTiltPrior')
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: copying data for Tilt prior failed because of "' +str(e.msg) + ', ' + str(e.expr) + '"')
        sys.exit(-2)
# else:
#     print '\tDeleting previous prior information for Tilt...'
#     star.del_column('_rlnAngleTiltPrior')
if do_psi:
    print('\tSetting prior for Psi...')
    try:
        star.copy_data_columns('_rlnAnglePsi', '_rlnAnglePsiPrior')
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: copying data for Psi prior failed because of "' + str(e.msg) + ', ' + str(e.expr) + '"')
        sys.exit(-2)
# else:
#     print '\tDeleting previous prior information for Psi...'
#     star.del_column('_rlnAnglePsiPrior')

if del_rot:
    print('\tDeleting Rotation angles...')
    try:
        star.del_column('_rlnAngleRot')
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: deleting data for Rotation because of "' + str(e.msg) + ', ' + str(e.expr) + '"')
        sys.exit(-2)
if del_tilt:
    print('\tDeleting Tilt angles...')
    try:
        star.del_column('_rlnAngleTilt')
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: deleting data for Tilt because of "' + str(e.msg) + ', ' + str(e.expr) + '"')
        sys.exit(-2)
if del_psi:
    print('\tDeleting Psi angles...')
    try:
        star.del_column('_rlnAnglePsi')
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: deleting data for Psi because of "' + str(e.msg) + ', ' + str(e.expr) + '"')
        sys.exit(-2)

if do_clean:
    print('\tCleaning refinement columns: ')
    if star.has_column('_rlnNormCorrection'):
        print('\t\t-Column rlnNormCorrection found...')
        star.del_column('_rlnNormCorrection')
    if star.has_column('_rlnRandomSubset'):
        print('\t\t-Column rlnRandomSubset found...')
        star.del_column('_rlnRandomSubset')
    if star.has_column('_rlnLogLikeliContribution'):
        print('\t\t-Column rlnLogLikeliContribution found...')
        star.del_column('_rlnLogLikeliContribution')
    if star.has_column('_rlnMaxValueProbDistribution'):
        print('\t\t-Column rlnMaxValueProbDistribution found...')
        star.del_column('_rlnMaxValueProbDistribution')
    if star.has_column('_rlnNrOfSignificantSamples'):
        print('\t\t-Column rlnNrOfSignificantSamples found...')
        star.del_column('_rlnNrOfSignificantSamples')

print('\tStoring the results in: ' + out_star)
if num_copies <= 1:
    star.store(out_star)
else:
    hold_star_name = os.path.splitext(out_star)[0]
    print('\tStoring the results in bacth mode with stem: ' + hold_star_name)
    for i in range(num_copies):
        hold_star = copy.deepcopy(star)
        star.rnd_data_column('_rlnAngleRot', -180., 180)
        star.store(hold_star_name+'_'+str(i)+'.star')

print('Terminated. (' + time.strftime("%c") + ')')
