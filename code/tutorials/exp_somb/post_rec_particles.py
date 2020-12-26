"""

    Script for post-processing already reconstructed particles

    Input:  - STAR file with next columns:
                '_rlnMicrographName': tomogram that will be used for reconstruction
                '_rlnImageName': tomograms used for picking
                '_rlnCtfImage': (optional) CTF model subvolume
                '_psSegImage': (optional) mask for particles within the reconstructed tomograms
                '_rlnCoordinate{X,Y,Z}': {X,Y,Z} coordinates in Relion format
                '_rlnAngle{Rot,Tilt,Psi}': (optional) {Rot,Tilt,Psi} angles
            - Output paths
            - Post-processing settings

    Output: - Particles sub-volumes are post-processed and restored
            - A new STAR file

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import gc
import os
import sys
import time
import copy
import pyto
import random
import pyseg as ps
import numpy as np
import multiprocessing as mp
from pyseg.globals import tomo_shift, get_sub_copy, lin_map, gaussian_low, gen_gaussian_low

from pyseg import sub, pexceptions

########## Global variables

ANGLE_NAMES = ['Rot', 'Tilt', 'Psi']

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/fs/pool/pool-ruben/antonio/nuc_mito'

# Input STAR file
in_star = ROOT_PATH + '/rln/mito/ref1/run1_blob_it017_data_nopost.star' # '/class/mito_bin0_ali2_blob2/class_ap_r_1_nopost_k0_split.star' # '/rec/nali/run1_bin0_it006_data.star' # '/class/mb/mb_merge.star' # '/rec/particles_rln.star' #
in_mask = ROOT_PATH + '/masks/mask_cyl_256_95_168_35_r.mrc' # '/masks/mask_cyl_100_37_85_25.mrc'
in_mask_mb = None # ROOT_PATH + '/masks/mask_cyl_256_95_130_50_mt.mrc' # '/masks/mask_cyl_100_37_53_25_r.mrc'

####### Output data

out_part_dir = ROOT_PATH + '/rec/particles_bin0_post_gauss_cres4_ca0.01' # '/rec/particles_bin0_post_mt0.2' # '/rec/particles_ali_post_mb'
out_star = ROOT_PATH + '/rec/particles_bin0_post_gauss_cres4_ca0.01.star' # '/rec/particles_bin0_post_mt0.2.star' # '/rec/particles_ali_post_mb.star'

####### Particles pre-processing settings

do_ang_prior = ['Tilt', 'Psi'] # ['Rot', 'Tilt', 'Psi']
do_ang_rnd = ['Rot']
do_mb_sf = None # 0.2 # 0.5

####### Gaussian low pass filtering (optional)

do_glp = True
glp_res = 0.352 # nm/vx
glp_cres = 4 # Cut-off resolution in nm
glp_ca = 0.01 # Amplitude at cut-off resolution (0, 1]
glp_ctf = True # The filter is applied to the CTF an not to the subvolume

####### Multiprocessing settings

mp_npr = 10 # 1

########################################################################################
# Local functions
########################################################################################


class Settings(object):
    out_part_dir = None
    out_star = None
    do_ang_prior = None
    do_ang_rnd = None
    do_mb_sf = None
    do_glp = None
    in_mask = None
    in_mask_mb = None


def pr_worker(pr_id, star, sh_star, rows, settings, qu):
    """
    Function which implements the functionality for the paralled workers.
    Each worker process a pre-splited set of rows of Star object
    :param pr_id: process ID
    :param star: Star object with input information
    :param rln_star: shared output Star object
    :param rows: list with Star rows to process for the worker
    :param settings: object with the settings
    :param qu: queue to store the output Star object
    :return: stored the reconstructed tomograms and insert the corresponding entries in the
             input Star object
    """

    # Initialization
    out_part_dir = settings.out_part_dir
    do_ang_prior = settings.do_ang_prior
    do_ang_rnd = settings.do_ang_rnd
    in_mask = settings.in_mask
    in_mask_mb = settings.in_mask_mb
    do_mb_sf = settings.do_mb_sf
    rln_star = copy.deepcopy(sh_star)
    if in_mask is not None:
        mask = ps.disperse_io.load_tomo(in_mask, mmap=False)
    if in_mask_mb is not None:
        mask_mb = ps.disperse_io.load_tomo(in_mask_mb, mmap=False)
        mask_mb = lin_map(mask_mb, lb=1, ub=do_mb_sf)
    if settings.do_glp is None:
        glp_params = None
    else:
        glp_params = settings.do_glp

    # print '\tLoop for particles: '
    count, n_rows = 0, len(rows)
    for row in rows:

        # print '\t\t\t+Reading the entry...'
        in_pick_tomo = star.get_element('_rlnImageName', row)
        in_rec_tomo = star.get_element('_rlnMicrographName', row)
        in_ctf = star.get_element('_rlnCtfImage', row)
        x_pick = star.get_element('_rlnCoordinateX', row)
        y_pick = star.get_element('_rlnCoordinateY', row)
        z_pick = star.get_element('_rlnCoordinateZ', row)
        try:
            shift_x = star.get_element('_rlnOriginX', row)
        except KeyError:
            shift_x = 0
        try:
            shift_y = star.get_element('_rlnOriginY', row)
        except KeyError:
            shift_y = 0
        try:
            shift_z = star.get_element('_rlnOriginZ', row)
        except KeyError:
            shift_z = 0
        rot = star.get_element('_rlnAngleRot', row)
        tilt = star.get_element('_rlnAngleTilt', row)
        psi = star.get_element('_rlnAnglePsi', row)
        rot_prior, tilt_prior, psi_prior = None, None, None
        if ANGLE_NAMES[0] in do_ang_prior:
            rot_prior = rot
        if ANGLE_NAMES[0] in do_ang_rnd:
            rot = 180. * random.random()
        if ANGLE_NAMES[1] in do_ang_prior:
            tilt_prior = tilt
        if ANGLE_NAMES[1] in do_ang_rnd:
            tilt = 180. * random.random()
        if ANGLE_NAMES[2] in do_ang_prior:
            psi_prior = psi
        if ANGLE_NAMES[2] in do_ang_rnd:
            psi = 180. * random.random()
        angs = np.asarray((rot, tilt, psi), dtype=np.float)

        # Sub-volumes post-processing
        # Forward transformation
        svol = ps.disperse_io.load_tomo(in_pick_tomo, mmap=False)
        if (in_mask is not None) or (in_mask_mb is not None):
            r3d = pyto.geometry.Rigid3D()
            r3d.q = r3d.make_r_euler(angles=np.radians(angs), mode='zyz_in_active')
            if (shift_x != 0) or (shift_y != 0) or (shift_z != 0):
                svol = tomo_shift(svol, (shift_y, shift_x, shift_z))
            svol_sp = np.asarray(svol.shape, dtype=np.int)
            svol_cent = np.asarray((int(.5 * svol_sp[0]), int(.5 * svol_sp[1]), int(.5 * svol_sp[2])), dtype=np.float32)
            svol = r3d.transformArray(svol, center=svol_cent, order=3, prefilter=True)
            # Membrane suppression
            if in_mask_mb is not None:
                svol *= mask_mb
            # Adding noise
            if in_mask is not None:
                svol *= mask
                stat_vol = svol[mask > 0]
                mn, st = stat_vol.mean(), stat_vol.std()
                if st > 0:
                    svol = (svol - mn) / st
                svol = ps.globals.randomize_voxel_mask(svol, mask, ref='fg')
            # Backward tranformation
            r3d_inv = pyto.geometry.Rigid3D()
            r3d_inv.q = r3d.make_r_euler(angles=np.radians(angs), mode='zyz_in_passive')
            svol = r3d_inv.transformArray(svol, center=svol_cent, order=3, prefilter=True)
            if (shift_x != 0) or (shift_y != 0) or (shift_z != 0):
                svol = tomo_shift(svol, (-shift_y, -shift_x, -shift_z))

        # Gaussian low-pass filtering
        if glp_params is not None:
            if glp_params[3]:
                ctf = ps.disperse_io.load_tomo(in_ctf, mmap=False)
                g_ctf = gen_gaussian_low(ctf, c_res=glp_params[1], c=glp_params[2], vx_size=glp_params[0])
                ctf *= g_ctf
                in_ctf = out_part_dir + '/' + os.path.splitext(os.path.split(in_ctf)[1])[0] + '_gauss.mrc'
                ps.disperse_io.save_numpy(ctf, in_ctf)
                svol = np.real(np.fft.ifftn(np.fft.fftn(svol) * np.fft.fftshift(ctf)))
                # svol = gaussian_low(svol, c_res=glp_params[1], c=glp_params[2], vx_size=glp_params[0])

        # Adding entry to particles STAR file
        out_part = out_part_dir + '/' + os.path.splitext(os.path.split(in_pick_tomo)[1])[0] + '.mrc'
        ps.disperse_io.save_numpy(svol, out_part)

        # Writing in the shared object
        print('\t\t-Process[' + str(pr_id) + '], Particle [' + str(count) + '/' + str(n_rows) + ']: ' + out_part)
        part_row = {'_rlnMicrographName': in_rec_tomo,
                    '_rlnCtfImage': in_ctf,
                    '_rlnImageName': out_part,
                    '_rlnCoordinateX': x_pick,
                    '_rlnCoordinateY': y_pick,
                    '_rlnCoordinateZ': z_pick,
                    '_rlnOriginX': shift_x,
                    '_rlnOriginY': shift_y,
                    '_rlnOriginZ': shift_z}
        part_row['_rlnAngleRot'] = rot
        part_row['_rlnAngleTilt'] = tilt
        part_row['_rlnAnglePsi'] = psi
        if ANGLE_NAMES[0] in do_ang_prior:
            part_row['_rlnAngleRotPrior'] = rot_prior
        if ANGLE_NAMES[1] in do_ang_prior:
            part_row['_rlnAngleTiltPrior'] = tilt_prior
        if ANGLE_NAMES[2] in do_ang_prior:
            part_row['_rlnAnglePsiPrior'] = psi_prior
        rln_star.add_row(**part_row)

        count += 1

    # Finishing the process
    qu.put(rln_star)
    sys.exit(pr_id)

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print('Extracting transmembrane features.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput STAR file: ' + in_star)
print('\tInput mask: ' + str(in_mask))
if in_mask_mb:
    print('\tInput mask for membrane supression: ')
    print('\t\t-Membrane suppression factor: ' + str(do_mb_sf))
print('\tOutput directory for reconstructed particles: ' + out_part_dir)
print('\tOutput STAR file: ' + out_star)
print('\tParticles pre-processing settings: ')
if len(do_ang_prior) > 0:
    for ang_prior in do_ang_prior:
        if ang_prior not in ['Rot', 'Tilt', 'Psi']:
            print('ERROR: unrecognized angle: ' + ang_prior)
            print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
            sys.exit(-1)
    print('\t\t-Adding prior for angles: ' + ang_prior)
if len(do_ang_rnd) > 0:
    for ang_rnd in do_ang_rnd:
        if ang_rnd not in ['Rot', 'Tilt', 'Psi']:
            print('ERROR: unrecognized angle: ' + ang_rnd)
            print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
            sys.exit(-1)
    print('\t\t-Setting random values for angles: ' + ang_rnd)
if do_glp:
    print('\tGaussian low-pass filter for particles:')
    print('\t\t-Resolution (voxel size): ' + str(glp_res) + ' nm/vx')
    print('\t\t-Cut-off resolution: ' + str(glp_cres) + ' nm')
    print('\t\t-Amplitude at cut-off: ' + str(glp_ca))
    if glp_ctf:
        print('\t\tApply to the CTF.')
print('\tMultiprocessing settings: ')
print('\t\t-Number processes: ' + str(mp_npr))
print('')


print('Loading input STAR file...')
star, rln_star = sub.Star(), sub.Star()
try:
    star.load(in_star)
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
if not os.path.exists(out_part_dir):
    os.makedirs(out_part_dir)

print('\tInitializing output Relion STAR file: ')
rln_star.add_column(key='_rlnMicrographName')
rln_star.add_column(key='_rlnCtfImage')
rln_star.add_column(key='_rlnImageName')
rln_star.add_column(key='_rlnCoordinateX')
rln_star.add_column(key='_rlnCoordinateY')
rln_star.add_column(key='_rlnCoordinateZ')
rln_star.add_column(key='_rlnOriginX')
rln_star.add_column(key='_rlnOriginY')
rln_star.add_column(key='_rlnOriginZ')
if star.has_column('_rlnAngleRot'):
    rln_star.add_column('_rlnAngleRot')
if star.has_column('_rlnAngleTilt'):
    rln_star.add_column('_rlnAngleTilt')
if star.has_column('_rlnAnglePsi'):
    rln_star.add_column('_rlnAnglePsi')
if ANGLE_NAMES[0] in do_ang_prior:
    if rln_star.has_column(key='_rlnAngleRot'):
        rln_star.add_column(key='_rlnAngleRot')
        rln_star.add_column(key='_rlnAngleRotPrior')
    else:
        print('ERROR: Prior Rot angle cannot be added since not Rot angle in the input tomogram.')
        print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
if ANGLE_NAMES[1] in do_ang_prior:
    if rln_star.has_column(key='_rlnAngleTilt'):
        rln_star.add_column(key='_rlnAngleTilt')
        rln_star.add_column(key='_rlnAngleTiltPrior')
    else:
        print('ERROR: Prior Tilt angle cannot be added since not Tilt angle in the input tomogram.')
        print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
if ANGLE_NAMES[2] in do_ang_prior:
    if rln_star.has_column(key='_rlnAnglePsi'):
        rln_star.add_column(key='_rlnAnglePsi')
        rln_star.add_column(key='_rlnAnglePsiPrior')
    else:
        print('ERROR: Prior Psi angle cannot be added since not Psi angle in the input tomogram.')
        print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
if ANGLE_NAMES[0] in do_ang_rnd:
    if not rln_star.has_column(key='_rlnAngleRot'):
        rln_star.add_column(key='_rlnAngleRot')
if ANGLE_NAMES[1] in do_ang_rnd:
    if not rln_star.has_column(key='_rlnAngleTilt'):
        rln_star.add_column(key='_rlnAngleTilt')
if ANGLE_NAMES[2] in do_ang_rnd:
    if not rln_star.has_column(key='_rlnAnglePsi'):
        rln_star.add_column(key='_rlnAnglePsi')

print('\tInitializing multiprocessing with ' + str(mp_npr) + ' processes: ')
settings = Settings()
settings.out_part_dir = out_part_dir
settings.out_star = out_star
settings.in_mask = in_mask
settings.in_mask_mb = in_mask_mb
settings.do_mb_sf = do_mb_sf
settings.do_ang_prior = do_ang_prior
settings.do_ang_rnd = do_ang_rnd
if do_glp:
    settings.do_glp = (glp_res, glp_cres, glp_ca, glp_ctf)
stars, qu = list(), mp.Queue()
if mp_npr <= 1:
    pr_worker(-1, star, rln_star, list(range(star.get_nrows())), settings, qu)
    stars.append(qu.get())
else:
    processes = list()
    spl_ids = np.array_split(list(range(star.get_nrows())), mp_npr)
    # Starting the processes
    for pr_id in range(mp_npr):
        pr = mp.Process(target=pr_worker, args=(pr_id, star, rln_star, spl_ids[pr_id], settings, qu))
        pr.start()
        processes.append(pr)
    # Getting processes results
    pr_results = list()
    for pr in processes:
        stars.append(qu.get())
    for pr_id, pr in enumerate(processes):
        pr.join()
        pr_results.append(pr.exitcode)
        if pr_id != pr_results[pr_id]:
            print('ERROR: Process ' + str(pr_id) + ' ended incorrectly.')
            print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
            sys.exit(-1)
    gc.collect()

# Merging output STAR files
rln_merged_star = sub.Star()
keys = stars[0].get_column_keys()
for key in keys:
    rln_merged_star.add_column(key)
for star in stars:
    for row in range(star.get_nrows()):
        hold_row = dict()
        for key in keys:
            hold_row[key] = star.get_element(key, row)
        rln_merged_star.add_row(**hold_row)

print('\tStoring output STAR file in: ' + out_star)
rln_merged_star.store(out_star)
print('Successfully terminated. (' + time.strftime("%c") + ')')
