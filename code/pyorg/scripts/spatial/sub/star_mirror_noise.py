"""

    From an input particles STAR file, makes a copy that indexes the subvolumes with noise added on background
    (Original subvolumes are not modified)

    Input:  - A STAR with particles
            - Mask to set foreground (True) and background (False) in the reference space

    Output: - A mirrores STAR file which indexes a copy to input subvolumenes with noise added

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

################# Package import

import os
import sys
import time
import pyto
import pyseg as ps
import numpy as np

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = ''

# Input STAR
in_star = ROOT_PATH + ''

# Input mask
in_mask = ROOT_PATH + ''

# Output STAR file
out_star = ROOT_PATH + ''

# Path to store output subvolumes with noise
out_svols_dir = ROOT_PATH + ''

# Randomization settings
rd_mode = 'white' # 'swapping'
rd_per = 10 # percentile for background, valid only if rd_mode='white'

########################################################################################
# Global functions
########################################################################################

# Function to randomize voxel density value in masked volumes
# vol: volume with the density map
# mask: volume with the binary mask
# Returns: a copy of vol but with the pixel in region where mask is False moved randomly
def randomize_voxel_mask(vol, mask):
    # Initialization
    o_vol = np.copy(vol)

    # Randomization
    ids = np.where(mask == False)
    l_ids = len(ids[0])
    rnd_ids = np.random.randint(0, l_ids, size=l_ids)
    for i in range(len(ids[0])):
        rnd_idx = rnd_ids[i]
        x, y, z = ids[0][i], ids[1][i], ids[2][i]
        rnd_x, rnd_y, rnd_z = ids[0][rnd_idx], ids[1][rnd_idx], ids[2][rnd_idx]
        o_vol[x, y, z] = vol[rnd_x, rnd_y, rnd_z]

    return o_vol


# Function to add white noise to background (mask equal to False)
# noise and std are taken from foreground (mask equal to True)
# vol: volume with the density map
# mask: volume with the binary mask
# per: fg percentifle to set noise mean
# Returns: a copy of vol but with the pixel in region where mask is False moved randomly
def add_white_noise(vol, mask, per):
    # Initialization
    o_vol = np.copy(vol)
    mask_inv = mask == False

    # Creating the noise
    vol_per, vol_std = np.percentile(vol[mask], per), np.std(vol[mask])
    noise = np.random.normal(vol_per, vol_std, size=mask_inv.sum())

    # Adding the noise
    o_vol[mask_inv] = noise

    return o_vol

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Create a mirrored particles STAR with noise.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput STAR file: ' + str(in_star))
print('\tInput mask file: ' + str(in_star))
print('\tOutput STAR file: ' + str(out_star))
print('\tOutput directory for subvolumens with noise added: ' + str(out_svols_dir))
print('\tRandomization settings:')
if rd_mode == 'swapping':
    print('\tMode: swapping')
elif rd_mode == 'white':
    print('\tWhite noise with percentile: ' + rd_per)
else:
    print('ERROR: mode for randomization ' + rd_mode + ' not recognized!')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
print('')

######### Process

print('Main Routine: ')

print('\tLoading input mask...')
try:
    mask = ps.disperse_io.load_tomo(in_mask) > 0
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input mask file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
svol_sp = np.asarray(mask.shape, dtype=int)
svol_sp2 = int(.5 * svol_sp[0])
svol_cent = np.asarray((svol_sp2, svol_sp2, svol_sp2), dtype=np.float32)

print('\tLoading input STAR file...')
star = ps.sub.Star()
try:
    star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input list of STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
star_out = np.copy(star)

if not os.path.exists(out_svols_dir):
    os.makedirs(out_svols_dir)

print('\tLoop for input STAR files:')
for row in range(star.get_nrows()):

    svol_fname = star.get_element('_rlnImageName', row)
    print('\t\t-Processing subvolume: ' + svol_fname)

    print('\t\t\t+Loading...')
    svol = ps.disperse_io.load_tomo(svol_fname)
    if svol.shape != mask.shape:
        print('ERROR: particle and mask dimension does not agree!')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    print('\t\t\t+Transforming to mask space...')
    try:
        shift_x, shift_y, shift_z = star.get_element('_rlnOriginX', row), star.get_element('_rlnOriginY', row), \
                                    star.get_element('_rlnOriginZ', row)
        svol = ps.globals.tomo_shift(svol, (shift_y, shift_x, shift_z))
    except KeyError:
        shift_x, shift_y, shift_z = None, None, None
    try:
        angs = np.asarray(star.get_element('_rlnAngleRot', row), star.get_element('_rlnAngleTilt', row), \
                          star.get_element('_rlnAnglePsi', row))
        r3d = pyto.geometry.Rigid3D()
        r3d.q = r3d.make_r_euler(angles=np.radians(angs), mode='zyz_in_active')
        svol = r3d.transformArray(svol, origin=svol_cent, order=3, prefilter=True)
    except KeyError:
        angs = None

    print('\t\t\t+Adding noise in background...')
    if rd_mode == 'swapping':
        svol_noise = randomize_voxel_mask(svol, mask)
    elif rd_mode == 'white':
        svol_noise = add_white_noise(svol, mask, rd_per)

    print('\t\t\t+Transforming back to original space...')
    if angs is not None:
        r3d_inv = pyto.geometry.Rigid3D()
        r3d_inv.q = r3d_inv.make_r_euler(angles=np.radians(angs), mode='zyz_in_pasive')
        svol_noise = r3d_inv.transformArray(svol_noise, origin=svol_cent, order=3, prefilter=True)
    if (shift_x is not None) and (shift_y is not None) and (shift_z is not None):
        svol_noise = ps.globals.tomo_shift(svol_noise, (-shift_y, -shift_x, -shift_z))

    svol_fname_out = out_svols_dir + '/' + os.path.splitext(os.path.split(svol_fname)[1])[0] + '_noise.mrc'
    print('\t\t\t+Storing output: ' + svol_fname_out)
    ps.disperse_io.save_numpy(svol_noise, svol_fname_out)

    print('\t\t\t+Adding entry to mirror STAR file...')
    star.set_element('_rlnImageName', row, svol_fname_out)

print('Storing output STAR file in: ' + out_star)
star_out.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')