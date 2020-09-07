"""

    Script for reconstructing particles RELION compatible from full reconstructed tomograms

    Input:  - STAR file with next columns:
                '_rlnMicrographName': tomogram that will be used for reconstruction
                '_rlnImageName': tomograms used for picking
                '_rlnCtfImage': (optional) CTF model subvolume
                '_psSegImage': (optional) mask for particles within the reconstructed tomograms
                '_rlnCoordinate{X,Y,Z}': {X,Y,Z} coordinates in Relion format
                '_rlnAngle{Rot,Tilt,Psi}': (optional) {Rot,Tilt,Psi} angles
            - Path to CTF subvolume, required in case not already included
            - Particles gray-value pre-processing settings

    Output: - The reconstructed particles
            - An output particles STAR files

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import gc
import sys
import time
import copy
import random
import pyseg as ps
import numpy as np
import multiprocessing as mp

from pyseg import sub, pexceptions

########## Global variables

ANGLE_NAMES = ['Rot', 'Tilt', 'Psi']

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '../../../..'

# Input STAR file
in_star = ROOT_PATH + '/data/tutorials/synth_sumb/pick/out/fil_mb_sources_to_no_mb_targets_net_parts.star'
in_ctf = ROOT_PATH + '/data/tutorials/synth_sumb/rec/wedge_130_60.mrc'
# Required if '_psSegImage' not in input STAR
in_mask_norm = ROOT_PATH + '/data/tutorials/synth_sumb/rec/mask_sph_130_60.mrc'

####### Output data

out_part_dir = ROOT_PATH + '/data/tutorials/synth_sumb/rec/particles'
out_star = ROOT_PATH + '/data/tutorials/synth_sumb/rec/particles_rln.star'

####### Particles pre-processing settings

do_bin = 4 # 1
do_ang_prior = ['Tilt', 'Psi'] # ['Rot', 'Tilt', 'Psi']
do_ang_rnd = ['Rot']
do_noise = False
do_use_fg = True
do_norm = True
do_inv = True

####### Multiprocessing settings

mp_npr = 5 # 10

########################################################################################
# Local functions
########################################################################################


class Settings(object):
    out_part_dir = None
    out_star = None
    do_bin = None
    do_inv = False
    do_ang_prior = None
    do_ang_rnd = None
    do_noise = None
    do_use_fg = None
    do_norm = None
    in_mask_norm = None
    in_ctf = None


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

    # Mapping settings
    out_part_dir = settings.out_part_dir
    do_ang_prior = settings.do_ang_prior
    do_ang_rnd = settings.do_ang_rnd
    do_noise = settings.do_noise
    do_use_fg = settings.do_use_fg
    do_norm = settings.do_norm
    do_inv = settings.do_inv
    in_mask_norm = settings.in_mask_norm
    hold_ctf = settings.in_ctf
    tomo_bin = settings.do_bin

    # Making a copy of the shared object
    rln_star = copy.deepcopy(sh_star)

    # print '\tLoop for particles: '
    count, n_rows = 0, len(rows)
    for row in rows:

        # print '\t\t\t+Reading the entry...'
        # in_pick_tomo = star.get_element('_rlnImageName', row)
        in_rec_tomo = star.get_element('_rlnMicrographName', row)
        if hold_ctf is not None:
            in_ctf = hold_ctf
        else:
            in_ctf = star.get_element('_rlnCtfImage', row)
        x_pick = star.get_element('_rlnCoordinateX', row)
        y_pick = star.get_element('_rlnCoordinateY', row)
        z_pick = star.get_element('_rlnCoordinateZ', row)
        rot, tilt, psi = None, None, None
        rot_prior, tilt_prior, psi_prior = None, None, None
        if star.has_column('_rlnAngleRot'):
            rot = star.get_element('_rlnAngleRot', row)
            if ANGLE_NAMES[0] in do_ang_prior:
                rot_prior = rot
            if ANGLE_NAMES[0] in do_ang_rnd:
                rot = 180. * random.random()
        if star.has_column('_rlnAngleTilt'):
            tilt = star.get_element('_rlnAngleTilt', row)
            if ANGLE_NAMES[1] in do_ang_prior:
                tilt_prior = tilt
            if ANGLE_NAMES[1] in do_ang_rnd:
                tilt = 180. * random.random()
        if star.has_column('_rlnAnglePsi'):
            psi = star.get_element('_rlnAnglePsi', row)
            if ANGLE_NAMES[2] in do_ang_prior:
                psi_prior = psi
            if ANGLE_NAMES[2] in do_ang_rnd:
                psi = 180. * random.random()

        # print '\t\t\t+Pre-processing input subvolume models...'
        ctf_svol = ps.disperse_io.load_tomo(in_ctf, mmap=True)
        sv_size, seg_svol = ctf_svol.shape, None
        if do_noise and star.has_column('_psSegImage'):
            in_seg = star.get_element('_psSegImage', row)
            seg_svol = ps.disperse_io.load_tomo(in_seg, mmap=False) > 0
            if sv_size != seg_svol.shape:
                print('ERROR: CTF model subvolume "' + in_ctf + '" and segmentation "' + \
                      in_seg + ' sizes does not fit.')
                print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
                sys.exit(-1)
            noise_mode = 'bg'
            if do_use_fg: noise_mode = 'fg'

        # print '\t\t\t+Reconstructing particle subvolume...'
        rec_tomo = ps.disperse_io.load_tomo(in_rec_tomo, mmap=True)
        # pick_tomo = ps.disperse_io.load_tomo(in_pick_tomo, mmap=True)
        # tomo_bin = max(rec_tomo.shape) / max(pick_tomo.shape)
        # x_rln, y_rln, z_rln = y_pick * tomo_bin, x_pick * tomo_bin, z_pick * tomo_bin
        x_rln, y_rln, z_rln = x_pick * tomo_bin, y_pick * tomo_bin, z_pick * tomo_bin
        try:
            part_svol = ps.globals.get_sub_copy(rec_tomo, (x_rln, y_rln, z_rln), sv_size).astype(np.float32)
        except ValueError:
            print('\t\t\tWARNING: This particle was not reconstructed properly, Value Error raised!')
            continue
        if part_svol.shape != sv_size:
            print('\t\t\tWARNING: This particle was not reconstructed properly ' + \
                  '(usually because is close to tomogram border), skipping to next...')
            continue

        if seg_svol is not None:
            # print '\t\t\t+Padding BG with noise...'
            part_svol = ps.globals.randomize_voxel_mask(part_svol, seg_svol, noise_mode)

        if do_norm:
            # print '\t\t\t+Gray-values normalization...'
            if in_mask_norm is None:
                part_svol = ps.sub.relion_norm(part_svol, seg_svol, inv=do_inv)
            else:
                part_svol = ps.sub.relion_norm(part_svol, seg_svol, inv=do_inv)

        # Adding entry to particles STAR file
        out_part = out_part_dir + '/particle_rln_' + str(row) + '.mrc'
        ps.disperse_io.save_numpy(part_svol, out_part)

        # Writing in the shared object
        # print '\t\t-Process[' + str(pr_id) + '], Particle [' + str(count) + '/' + str(n_rows) + ']: ' + out_part
        part_row = {'_rlnMicrographName': in_rec_tomo,
                    '_rlnCtfImage': in_ctf,
                    '_rlnImageName': out_part,
                    '_rlnCoordinateX': x_rln,
                    '_rlnCoordinateY': y_rln,
                    '_rlnCoordinateZ': z_rln}
        if rln_star.has_column('_rlnAngleRot'):
            part_row['_rlnAngleRot'] = rot
        if rln_star.has_column('_rlnAngleTilt'):
            part_row['_rlnAngleTilt'] = tilt
        if rln_star.has_column('_rlnAnglePsi'):
            part_row['_rlnAnglePsi'] = psi
        if rln_star.has_column('_rlnAngleRotPrior'):
            part_row['_rlnAngleRotPrior'] = rot_prior
        if rln_star.has_column('_rlnAngleTiltPrior'):
            part_row['_rlnAngleTiltPrior'] = tilt_prior
        if rln_star.has_column('_rlnAnglePsiPrior'):
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
if in_ctf is not None:
    print('\tInput CTF file: ' + in_ctf)
print('\tOutput directory for reconstructed particles: ' + out_part_dir)
print('\tOutput STAR file: ' + out_star)
print('\tParticles pre-processing settings: ')
if do_bin > 0:
    print('\t\t-Particles picked with binning: ' + str(do_bin))
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
if do_norm:
    print('\t\t-Applying relion normalization: ')
    if in_mask_norm is not None:
        print('\t\t\t-Tomogram for FG: ' + in_mask_norm)
        mask_norm = ps.disperse_io.load_tomo(in_mask_norm)
    if do_inv:
        print('\t\t-Invert density values.')
if do_noise:
    print('\t\t-Set gray-values in background (BG) randomly.')
    if do_use_fg:
        print('\t\t\t+Take FG values as reference.')
    else:
        print('\t\t\t+Take BG values as reference.')
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
if (in_ctf is None) and (not star.has_column('_rlnCtfImage')):
    print(' ERROR: No CTF specified and the input STAR file does not conta rlnCtfImage column')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('\tInitializing output relion STAR file: ')
rln_star.add_column(key='_rlnMicrographName')
rln_star.add_column(key='_rlnCtfImage')
rln_star.add_column(key='_rlnImageName')
rln_star.add_column(key='_rlnCoordinateX')
rln_star.add_column(key='_rlnCoordinateY')
rln_star.add_column(key='_rlnCoordinateZ')
if ANGLE_NAMES[0] in do_ang_prior:
    if star.has_column(key='_rlnAngleRot'):
        rln_star.add_column(key='_rlnAngleRot')
        rln_star.add_column(key='_rlnAngleRotPrior')
    else:
        print('ERROR: Prior Rot angle cannot be added since not Rot angle in the input tomogram.')
        print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
if ANGLE_NAMES[1] in do_ang_prior:
    if star.has_column(key='_rlnAngleTilt'):
        rln_star.add_column(key='_rlnAngleTilt')
        rln_star.add_column(key='_rlnAngleTiltPrior')
    else:
        print('ERROR: Prior Tilt angle cannot be added since not Tilt angle in the input tomogram.')
        print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
if ANGLE_NAMES[2] in do_ang_prior:
    if star.has_column(key='_rlnAnglePsi'):
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
if do_norm and (not star.has_column('_psSegImage')) and (in_mask_norm is None):
    print('ERROR: Unable to do gray-value normalization: ')
    print('\tNeither \'_psSegImage\' column in input STAR file nor \'in_mask_norm\' input set.')
    print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('\tInitializing multiprocessing with ' + str(mp_npr) + ' processes: ')
settings = Settings()
settings.out_part_dir = out_part_dir
settings.out_star = out_star
settings.do_bin = do_bin
settings.do_inv = do_inv
settings.do_ang_prior = do_ang_prior
settings.do_ang_rnd = do_ang_rnd
settings.do_noise = do_noise
settings.do_use_fg = do_use_fg
settings.do_norm = do_norm
settings.in_mask_norm = in_mask_norm
settings.in_ctf = in_ctf
processes = list()
qu = mp.Queue()
spl_ids = np.array_split(list(range(star.get_nrows())), mp_npr)
# Starting the processes
for pr_id in range(mp_npr):
    pr = mp.Process(target=pr_worker, args=(pr_id, star, rln_star, spl_ids[pr_id], settings, qu))
    pr.start()
    processes.append(pr)
# Getting processes results
pr_results, stars = list(), list()
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
