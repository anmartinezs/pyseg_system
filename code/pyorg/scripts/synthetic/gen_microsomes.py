"""

    Generate a synthetic set of microsomes with different membrane bound proteins

    Input:  - Data set parameters:
                + Number of tomograms (one microsome each)
                + Tomogram size
                + Resolution (pixel size)
                + SNR range
                + Missing wedge (semi-angle in degrees or input file)
                + binning factor for segmentation an particle picking
            - Microsome parameters:
                + Membrane thickness (in nm)
                + Density model for each protein type inserted
                + Averaged number of particles per microsome and per model
                + Model for protein insertion, available: CSRV, SRPV, 2CCSRV

    Output: - Tomograms generated with one microsome each:
                + Full resolution
                + Binned counterpart
            - A STAR file pairing the originals and the binned tomograms

"""

################# Package import

import os
import sys
import time
import numpy as np
import scipy as sp
import multiprocessing as mp
from pyorg import disperse_io, sub, spatial
from pyorg.globals import *

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = ''

# Output directory
out_dir = ROOT_PATH + ''
out_stem = ''

# Tomogram settings
tm_nt = 10 # tomograms
tm_size = (800, 800, 400) # pixels
tm_res = 0.262 # nm/pixel
tm_snr_rg = (0.01, 0.05)
tm_wedge = 30 # semi-angle in degrees or input file
tm_bin = 2

# Microsome parameters
mc_mbt = 5 # nm
mc_mbs = 1.5 # nm
mc_ip_min_dst = 15 # nm
# By default 1st model has clusters randomly distributed of radius mc_1st_crad and 2nd clusters of size 2*mc_3rd_crad,
# 3rd is CSRV distributed, 4th particles are 2CCSRV placed at an averaged distance of mc_4th_dst,
mc_1st_crad = 50 # nm
mc_c_jump_prob = 0.1
mc_4th_dst = 20 # nm
mc_in_models = ('', '', '', '', '', '', '', '')
mc_avg_nparts = (20, 20, 20, 50, 50, 50, 30, 30)
mc_3sg_nparts = 5
mc_zh = 0

########################################################################################
# ADDITIONAL ROUTINES
########################################################################################


def gen_mask_msome(shape, rad1, rad2):
    """
    Generates a microsome mask
    :param shape: 3-tuple for the output tomogram
    :param rad1: radius 1 for the microsome
    :param rad2: radius 2 for the microsome
    :return: the generated microme
    """
    dx, dy, dz = float(shape[0]), float(shape[1]), float(shape[2])
    dx2, dy2, dz2 = math.floor(.5 * dx), math.floor(.5 * dy), math.floor(.5 * dz)
    x_l, y_l, z_l = -dx2, -dy2, -dz2
    x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
    X, Y, Z = np.meshgrid(np.arange(x_l, x_h), np.arange(y_l, y_h), np.arange(z_l, z_h), indexing='xy')
    R = X*X + Y*Y + Z*Z
    return (R >= (rad1*rad1)) & (R <= (rad2*rad2))


def add_dmb_msome(tomo, rad, res, mb_t, mb_s):
    """
    Add a double Gaussian layered membrane of a microsome to a tomogram
    :param tomo: tomogram where the membrane is added
    :param rad: microsome radius
    :param res: tomograms resolution (nm/px)
    :param mb_t: membrane thickness in nm
    :param mb_s: Gaussian sigma for each layer in nm
    :return: None
    """

    # Input parsing
    t_v, s_v, rad_v = .5 * (mb_t / res), mb_s / res, rad / res
    rad1, rad2 = rad_v - t_v, rad_v + t_v
    g_cte = 2 * s_v * s_v
    s_v_2 = .5 * s_v
    g_cte_2 = 2 * s_v_2 * s_v_2

    #  Getting membrane gray intensity from input tomogram
    tomo_bin = tomo > 0
    tomo_vals = tomo(tomo_bin)
    tomo_mn = tomo_vals.mean()

    # Generating the bilayer
    dx, dy, dz = float(tomo.shape[0]), float(tomo.shape[1]), float(tomo.shape[2])
    dx2, dy2, dz2 = math.floor(.5 * dx), math.floor(.5 * dy), math.floor(.5 * dz)
    x_l, y_l, z_l = -dx2, -dy2, -dz2
    x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
    X, Y, Z = np.meshgrid(np.arange(x_l, x_h), np.arange(y_l, y_h), np.arange(z_l, z_h), indexing='xy')
    R = np.sqrt(X * X + Y * Y + Z * Z)
    G_u = tomo_mn * np.exp(-(R-rad1)**2 / g_cte)
    G_l = tomo_mn * np.exp(-(R-rad2)**2 / g_cte)

    # Creating the softmaks for the model structure
    BW = sp.ndimage.morphology.distance_transform_edt(np.invert(tomo_bin))
    Ss = 1. - np.exp(-(BW * BW) / g_cte_2)
    G_u = G_u * Ss
    G_l = G_l * Ss

    # Merge the structure and the membrane
    tomo += (G_u + G_l)


def gen_ccsrv_msome(shape, n_parts, mic_rad, c_rad, min_ip_dst, c_jump_prob):
    """
    Generates a list of 3D coordinates and rotations distributed in cluster randomly placed on a microsome
    :param shape: tomogram shape
    :param n_parts: number of particles to try to generate
    :param mic_rad: microsome radius
    :param c_rad: cluster radius
    :param min_ip_dst: minimum interparticle distance
    :param c_jump_prob: probabilty to create a new cluster evaluated each time a particle is addded [0, 1]
    :return: two lists; coordinates and rotations
    """

    # Initialization
    count = 0
    min_ip_dst_2 = float(min_ip_dst)**2
    locs, rots = list(), list()
    mic_cent = .5 * np.asarray(shape, dtype=float)
    mic_rad_f = float(mic_rad)
    max_n_tries = np.prod(np.asarray(shape, dtype=int))

    # Generate all possible the uniformly distributed particle seeds at 2x pixel precision
    n_seeds = 8. * np.pi * mic_rad * mic_rad
    if n_seeds < npart:
        n_seeds = npart
    hold_seeds = np.random.randn(n_seeds, 3)
    norm = mic_rad_f / np.linalg.norm(hold_seeds)
    hold_seeds *= norm
    hold_seeds += mic_cent
    seeds = list()
    for p_cent in hold_seeds:
        # Check that the particle is within the tomogram
        if (p_cent[0] >= 0) and (p_cent[0] < shape[0]) \
                and (p_cent[1] >= 0) and (p_cent[1] < shape[1]) \
                and (p_cent[2] >= 0) and (p_cent[2] < shape[2]):
            seeds.append(p_cent)

    # Loop for clusters
    mic_end, n_try = False, 0
    while not mic_end:

        # Generating the cluster center (random sampling on S^2)
        c_cent = np.random.randn(1, 3)[0]
        norm = mic_rad_f / np.linalg.norm(c_cent)
        c_cent *= norm
        c_cent += mic_cent

        # Sort seeds by distance to cluster center
        dsts = geo_point_dst_sphere(c_cent, seeds, mic_cent, mic_rad)

        # Loop for particles likely within the cluster radius
        c_end = False
        while not c_end:
            rnd_val = np.random.random()
            if (rnd_val > c_jump_prob) and (n_try <= max_n_tries):
                rnd_dst = math.fabs(np.random.normal(0, c_rad))
                idx = (np.abs(dsts - rnd_dst)).argmin()
                c_part = seeds[idx]
                # Check that the new particle does not overlap with other already inserted
                hold_dst = c_part - np.asarray(locs, dtype=float)
                if np.sum(hold_dst*hold_dst, axis=1) >= min_ip_dst_2:
                    locs.append(c_part)
                    tilt, psi = vect_to_zrelion(c_part - mic_cent, mode='active')[1:]
                    rots.append((360.*np.random.rand()-180., tilt, psi))
                    count += 1
            else:
                c_end = True

        # Ensure termination
        n_try += 1
        if (n_try > max_n_tries) or (count >= n_parts):
            mic_end = True

    return locs, rots


def gen_2ccsrv_msome(locs, rots, shape, n_parts, mic_rad, avg_dst, min_ip_dst):
    """
    Generates a list of 3D coordinates and rotations distributed in correlation with some reference inputs
    :param locs, rots: referece input lists with localizations and rotation
    :param shape: tomogram shape
    :param n_parts: number of particles to try to generate
    :param mic_rad: microsome radius
    :param avg_dst: averaged distance to the reference pattern nearest particle
    :param min_ip_dst: minimum interparticle distance
    :param c_jump_prob: probabilty to create a new cluster evaluated each time a particle is addded [0, 1]
    :return: two output lists; coordinates and rotations
    """

    # Initialization
    count = 0
    min_ip_dst_2 = float(min_ip_dst) ** 2
    locs_out, rots_out = list(), list()
    mic_cent = .5 * np.asarray(shape, dtype=float)
    mic_rad_f = float(mic_rad)
    max_n_tries = np.prod(np.asarray(shape, dtype=int))
    locs_a, rots_a = np.asarray(locs, dtype=float), np.asarray(rots, dtype=float)

    # Generate all possible the uniformly distributed particle seeds at 2x pixel precision
    n_seeds = 8. * np.pi * mic_rad * mic_rad
    if n_seeds < npart:
        n_seeds = npart
    hold_seeds = np.random.randn(n_seeds, 3)
    norm = mic_rad_f / np.linalg.norm(hold_seeds)
    hold_seeds *= norm
    hold_seeds += mic_cent
    seeds = list()
    for p_cent in hold_seeds:
        # Check that the particle is within the tomogram
        if (p_cent[0] >= 0) and (p_cent[0] < shape[0]) \
                and (p_cent[1] >= 0) and (p_cent[1] < shape[1]) \
                and (p_cent[2] >= 0) and (p_cent[2] < shape[2]):
            seeds.append(p_cent)

    # Loop for particles
    mic_end, n_try = False, 0
    p_end = False
    while not p_end:
        rnd_idx = np.random.randint(0, len(seeds))
        c_cent = seeds[rnd_idx]
        rnd_dst = math.fabs(np.random.normal(0, avg_dst))
        dsts = geo_point_dst_sphere(c_cent, seeds, mic_cent, mic_rad)
        idx = (np.abs(dsts - rnd_dst)).argmin()
        p_cent = seeds[idx]
        # Check that the new particle does not overlap with other already inserted
        hold_dst = p_cent - locs_a
        if np.sum(hold_dst * hold_dst, axis=1) >= min_ip_dst_2:
            locs_out.append(p_cent)
            tilt, psi = vect_to_zrelion(p_cent - mic_cent, mode='active')[1:]
            rots_out.append((360. * np.random.rand() - 180., tilt, psi))
            count += 1

        # Ensure termination
        n_try += 1
        if (n_try > max_n_tries) or (count >= n_parts):
            p_end = True

    return locs_out, rots_out


def gen_csrv_msome(shape, n_parts, mic_rad, min_ip_dst):
    """
    Generates a list of 3D coordinates and rotations a CSRV pattern
    :param shape: tomogram shape
    :param n_parts: number of particles to try to generate
    :param mic_rad: microsome radius
    :param min_ip_dst: minimum interparticle distance
    :param c_jump_prob: probabilty to create a new cluster evaluated each time a particle is addded [0, 1]
    :return: two output lists; coordinates and rotations
    """

    # Initialization
    count = 0
    min_ip_dst_2 = float(min_ip_dst) ** 2
    locs, rots = list(), list()
    mic_cent = .5 * np.asarray(shape, dtype=float)
    mic_rad_f = float(mic_rad)
    max_n_tries = np.prod(np.asarray(shape, dtype=int))

    # Loop for particles
    mic_end, n_try = False, 0
    p_end = False
    while not p_end:
        p_cent = np.random.randn(1, 3)[0]
        norm = mic_rad_f / np.linalg.norm(p_cent)
        p_cent *= norm
        p_cent += mic_cent
        # Check that the particle is within the tomogram
        if (p_cent[0] >= 0) and (p_cent[0] < shape[0]) \
                and (p_cent[1] >= 0) and (p_cent[1] < shape[1]) \
                and (p_cent[2] >= 0) and (p_cent[2] < shape[2]):
            if len(locs) > 0:
                # Check that the new particle does not overlap with other already inserted
                hold_dst = p_cent - np.asarray(locs, dtype=float)
                if np.sum(hold_dst * hold_dst, axis=1) >= min_ip_dst_2:
                    locs.append(p_cent)
                    tilt, psi = vect_to_zrelion(p_cent - mic_cent, mode='active')[1:]
                    rots.append((360. * np.random.rand() - 180., tilt, psi))
                    count += 1
            else:
                locs.append(p_cent)
                tilt, psi = vect_to_zrelion(p_cent - mic_cent, mode='active')[1:]
                rots.append((360. * np.random.rand() - 180., tilt, psi))
                count += 1

        # Ensure termination
        n_try += 1
        if (n_try > max_n_tries) or (count >= n_parts):
            p_end = True

    return locs, rots

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Printing the initial message

print('Generate synthetic microsomes.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tTomograms settings:')
print('\t\t-Number of tomograms: ' + str(tm_nt))
print('\t\t-Tomogram size: ' + str(tm_size) + ' px')
print('\t\t-Resolution: ' + str(tm_res) + ' nm/px')
print('\t\t-SNR range: ' + str(tm_snr_rg))
print('\t\t-Missing wedge semi-angle: ' + str(tm_wedge) + ' deg')
print('\t\t-Binning factor: ' + str(tm_bin))
print('\tMicrosome settings:')
print('\t\t-Membrane thickness: ' + str(mc_mbt) + ' nm')
print('\t\t-Membrane layer sigma: ' + str(mc_mbs) + ' nm')
print('\t\t-Minimum iter-particles distance: ' + str(mc_ip_min_dst) + ' nm')
print('\t\t-Clusters radius for 1st pattern: ' + str(mc_1st_crad) + ' nm')
print('\t\t-Clusters radius for 2nd pattern: ' + str(2*mc_1st_crad) + ' nm')
print('\t\t-Averaged ditance to the 3rd pattern particles for the 4th pattern particles: ' + str(mc_4th_dst) + ' nm')
print('\t\t-Input files with the density models: ' + str(mc_in_models))
print('\t\t-Averaged (normal distribution) number of particles per model and microsome: ' + str(mc_avg_nparts))
print('\t\t-Maximum deviation (3sg for Gaussian): ' + str(mc_3sg_nparts))
print('\t\t-Subvolumes center height: ' + str(mc_zh))
print('')

########### Input parsing

######### Process

# Initialization of global variables
max_svol_dim = 0
svol_refs = dict()
nparts, snrs = np.zeros(shape=(tm_nt, len(mc_avg_nparts)), dtype=int), np.zeros(shape=tm_nt, dtype=float)
for i in range(tm_nt):
    snrs[i] = (tm_snr_rg[1] - tm_snr_rg[0]) * np.random.random() + tm_snr_rg[0]
    for j, avg_parts in enumerate(mc_avg_nparts):
        hold_nparts = np.random.normal(avg_parts, mc_3sg_nparts)
        if hold_nparts < 0:
            hold_nparts = 0
        nparts[i, j] = hold_nparts
hold_ref_0 = disperse_io.load_tomo(mc_in_models[0])
for in_model in mc_in_models:
    hold_ref = disperse_io.load_tomo(in_model)
    if hold_ref_0.shape != hold_ref.shape:
        print('ERROR: All input density subvolume models must have the same size.')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
    svol_refs[mc_in_models] = hold_ref
max_tm_size, max_svol_dim = max(tm_size[0:1]), max(hold_ref_0.shape[0:1])
mic_rad = .75 * max_tm_size
if (mic_rad + max_svol_dim) > max_tm_size:
    mic_rad = max_tm_size - max_svol_dim
cent_v = .5*np.asarray(hold_ref_0.shape, dtype=float) - np.asarray((0, 0, mc_zh), dtype=float)
mc_ip_min_dst_v, mc_1st_crad_v, mc_4th_dst_v = float(mc_ip_min_dst)/tm_res, float(mc_1st_crad)/tm_res, \
                                               float(mc_4th_dst)/tm_res

print('Main routine: ')

print('\tLoop for microsomes: ')
cols = list()
for i in range(tm_nt):

    snr = snrs[i]
    print('\t\t-Generating microsome ' + str(i) + ' with SNR=' + snr + ':')
    tomo = np.zeros(shape=tm_size, dtype=np.float16)

    print('\t\t\t+Adding the particles: ')
    hold_locs, hold_angs = None, None
    for j, key in enumerate(svol_refs.keys()):
        svol_ref = svol_refs[key]
        npart = nparts[i, j]
        if j == 0:
            locs, angs = gen_ccsrv_msome(tm_size, npart, mic_rad, mc_1st_crad_v, mc_ip_min_dst_v, mc_c_jump_prob)
        elif j == 1:
            locs, angs = gen_ccsrv_msome(tm_size, npart, mic_rad, 2*mc_1st_crad_v, mc_ip_min_dst_v, mc_c_jump_prob)
        elif j == 3:
            locs, angs = gen_2ccsrv_msome(hold_locs, hold_angs, npart, mic_rad, hold_locs, hold_angs)
        else:
            locs, angs = gen_csrv_msome(npart, mic_rad)
        for loc, ang in zip(locs, angs):
            svol = tomo_rot(svol_ref, ang, conv='relion', active=True, deg=True, center=(0, 0, mc_zh))
            point = loc + cent_v
            tomo = spatial.over_sub_tomo(tomo, point, svol, np.max)
        if j == 2:
            hold_locs, hold_angs = locs, angs
        print('\t\t\t\t* Particles inserted: ' + str(len(locs)) + ' of ' + str(npart))

    print('\t\t\t+Adding the membrane...')
    add_dmb_msome(tomo, mic_rad, tm_res, mc_mbt, mc_mbs)

    print('\t\t\t+Adding the distortions...')
    mc_rad1, mc_rad2 = mic_rad - mc_zh, mic_rad + max_svol_dim - mc_zh
    mask = gen_mask_msome(tm_size, mc_rad1, mc_rad2)
    mn = tomo[mask].mean()
    sg_bg = mn / snr
    tomo += np.random.normal(mn, sg_bg, size=tm_size)
    tomo = lin_map(tomo, tomo.max(), tomo.min())
    tomo = add_mw(tomo, 0, tm_wedge)

    out_tomo = out_dir + '/' + out_stem + '_ + tomo_mic_' + str(i) + '.mrc'
    print('\t\t\t+Saving the microsome as: ' + out_tomo)
    disperse_io.save_numpy(tomo, out_tomo)
    out_tomo_bin = out_dir + '/' + out_stem + '_ + tomo_mic_bin_' + str(i) + '.mrc'
    print('\t\t\t+Saving the ' + str(tm_bin) + ' binned microsome as: ' + out_tomo_bin)
    tomo_bin = tomo
    for i in range(bin):
        tomo_bin = tomo_binning(tomo_bin)
    disperse_io.save_numpy(tomo_bin, out_tomo)
    cols.append((out_tomo, out_tomo_bin))

out_star = out_dir + '/' + out_stem + '.star'
print('\tStoring output STAR file in: ' + out_star)
star_tomos = sub.Star()
star_tomos.add_column('_rlnMicrographName')
star_tomos.add_column('_rlnImageName')
for col in cols:
    row = {'_rlnMicrographName':col[0], '_rlnImageName':col[1]}
    star_tomos.add_row(**row)
star_tomos.store(out_star)

print('Successfully terminated. (' + time.strftime("%c") + ')')
