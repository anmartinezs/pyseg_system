"""

    Script for resizing (upsizing) already reconstructed particles, the resizing can be selective so its allow to mix
    particles with different pixel sizes.

    Input:  - STAR file with next columns:
                '_rlnMicrographName': tomogram that will be used for reconstruction
                '_rlnImageName': tomograms used for picking
                '_rlnCtfImage': (optional) CTF model subvolume
                '_psSegImage': (optional) mask for particles within the reconstructed tomograms
                '_rlnCoordinate{X,Y,Z}': {X,Y,Z} coordinates in Relion format
                '_rlnAngle{Rot,Tilt,Psi}': (optional) {Rot,Tilt,Psi} angles
            - STAR file to only resize the reconstruction from some specific tomograms

    Output: - A new column is added to the input STAR file with ListTomoParticles generated
            - Intermediate information

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import gc
import os
import sys
import time
import copy
import random
import pyseg as ps
import numpy as np
import scipy as sp
import skimage as ski
import multiprocessing as mp

from pyseg import sub, pexceptions

########## Global variables

ANGLE_NAMES = ['Rot', 'Tilt', 'Psi']

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

####### Input data

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex'

# Input STAR file
in_star = ROOT_PATH + '/syn2/rec/pre/particles_pre.star' # '/syn2/rec/pre/root_v5_rot_rnd_prior_post.star' # '/syn2/rec/pst/root_v5_rot_rnd_prior_post.star' # '/pick/out/fil_mb_sources_to_no_mb_targets_net_parts.star'

# Input STAR file
in_star_tomos = ROOT_PATH + '/syn3/rec/syn_scale_tomos.star'

####### Output data

out_part_dir = ROOT_PATH + '/syn3/rec/pre/particles_pre_scaled' # '/syn3/rec/pre/particles_pre_root_v6_scaled' # '/syn3/rec/pst/particles_pst_root_v6_scaled'
out_star = ROOT_PATH + '/syn3/rec/pre/particles_pre_scaled.star' # '/syn3/rec/pre/particles_pre_root_v6_scaled.star' # '/syn3/rec/pst/particles_pst_root_v6_scaled.star'

####### Particles pre-processing settings

do_scale = 8.42/6.84 # 1

####### Multiprocessing settings

mp_npr = 1 # 5

########################################################################################
# Local functions
########################################################################################


class Settings(object):
    out_part_dir = None
    out_star = None
    do_scale = None


def pr_worker(pr_id, star, star_seg, sh_star, rows, settings, qu):
    """
    Function which implements the functionality for the paralled workers.
    Each worker process a pre-splited set of rows of Star object
    :param pr_id: process ID
    :param star: Star object with input information
    :param star_seg: Star object with the tomograms to scale (if None then every particle is processed)
    :param rln_star: shared output Star object
    :param rows: list with Star rows to process for the worker
    :param settings: object with the settings
    :param qu: queue to store the output Star object
    :return: stored the reconstructed tomograms and insert the corresponding entries in the
             input Star object
    """

    # Mapping settings
    out_part_dir = settings.out_part_dir
    scale = settings.do_scale
    if star_seg is not None:
        tomos_l = star_seg.get_column_data_set('_rlnMicrographName')
    else:
        tomos_l = star.get_column_data_set('_rlnMicrographName')
    star_col_keys = star.get_column_keys()

    # Making a copy of the shared object
    rln_star = copy.deepcopy(sh_star)

    # print '\tLoop for particles: '
    count, n_rows = 0, len(rows)
    for row in rows:

        # Loading particle
        in_img = star.get_element('_rlnImageName', row)
        img = ps.disperse_io.load_tomo(in_img)

        # Checking if the particle has to be scaled
        if star.get_element('_rlnMicrographName', row) in tomos_l:

            # Scaling particle
            out_shape = np.round(scale * np.asarray(img.shape, dtype=np.float)).astype(np.int)
            img_scaled = sp.ndimage.zoom(img, scale, order=1)
            img = ps.globals.get_sub_copy(img_scaled, out_shape/2, img.shape).astype(np.float32)

        # Storing particle and updating STAR file row
        out_part = out_part_dir + '/' + os.path.splitext(os.path.split(in_img)[1])[0] + '.mrc'
        ps.disperse_io.save_numpy(img, out_part)
        out_img = out_part

        # Adding the row entry to the input STAR file
        part_row = dict()
        for key in star_col_keys:
            if key == '_rlnImageName':
                part_row['_rlnImageName'] = out_img
            else:
                part_row[key] = star.get_element(key=key, row=row)
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
if in_star_tomos is not None:
    print('\tInput STAR files with the tomograms with particles to resize: ' + in_star_tomos)
print('\tOutput directory for reconstructed particles: ' + out_part_dir)
print('\tOutput STAR file: ' + out_star)
print('\tParticles pre-processing settings: ')
if do_scale < 1:
    print('ERROR: scaling must be higher than 1: ' + str(do_scale))
    print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
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
star_seg = None
if in_star_tomos is not None:
    star_seg = sub.Star()
    try:
        star_seg.load(in_star_tomos)
    except pexceptions.PySegInputError as e:
        print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

print('\tInitializing output relion STAR file: ')
for key in star.get_column_keys():
    rln_star.add_column(key=key)

print('\tInitializing multiprocessing with ' + str(mp_npr) + ' processes: ')
settings = Settings()
settings.out_part_dir = out_part_dir
settings.out_star = out_star
settings.do_scale = do_scale
processes = list()
qu = mp.Queue()
spl_ids = np.array_split(list(range(star.get_nrows())), mp_npr)
# Starting the processes
for pr_id in range(mp_npr):
    pr = mp.Process(target=pr_worker, args=(pr_id, star, star_seg, rln_star, spl_ids[pr_id], settings, qu))
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
