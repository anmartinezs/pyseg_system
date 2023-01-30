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
import copy
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

ROOT_PATH = '../../../..' # '/fs/home/martinez/workspace/pyseg_system'

# Output directory
out_dir = ROOT_PATH + '/data/tutorials/synth_sumb/mics'
out_stem = 'test_1'

# Multiprocessing settings
mp_npr = 3 # number of parallel processes

# Tomogram settings
tm_nt = 20 # tomograms
tm_size = (1200, 1200, 640) # pixels
tm_res = 0.262 # nm/pixel
tm_snr_rg = (0.03, 0.04) # (0.01, 0.05)
tm_wedge = 60 # angle in degrees or input file
tm_wedge_rot = 13 # rotation for wedge axis in degrees
tm_bin = 2
tm_rsz = [130, 130, 130] # pixels

# Microsome parameters
mc_mbt = 3 # nm
mc_mbs = .5 # nm
mc_ip_min_dst = 20 # 15 # nm
# By default 1st model has clusters randomly distributed of radius mc_1st_crad and 2nd clusters of size 2*mc_3rd_crad,
# 3rd is CSRV distributed, 4th particles are 2CCSRV placed at an averaged distance of mc_4th_dst,
mc_1st_crad = 50 # nm
mc_c_jump_prob = 0.01
mc_4th_dst = 30 # nm
mc_in_models = (ROOT_PATH + '/data/tutorials/synth_sumb/models/4uqj_r2.62_90_nostd.mrc',
                ROOT_PATH + '/data/tutorials/synth_sumb/models/5gjv_r2.62_90_nostd.mrc',
                ROOT_PATH + '/data/tutorials/synth_sumb/models/5vai_r2.62_90_nostd.mrc',
                ROOT_PATH + '/data/tutorials/synth_sumb/models/5kxi_r2.62_90_nostd.mrc',
                ROOT_PATH + '/data/tutorials/synth_sumb/models/4pe5_r2.62_90_nostd.mrc')
# mc_in_models = (ROOT_PATH + '/data/tutorials/synth_sumb/models/4uqj_r2.62_90_nostd.mrc',
#                 ROOT_PATH + '/data/tutorials/synth_sumb/models/5kxi_r2.62_90_nostd.mrc',
#                 ROOT_PATH + '/data/tutorials/synth_sumb/models/4pe5_r2.62_90_nostd.mrc',
#                 ROOT_PATH + '/data/tutorials/synth_sumb/models/5vai_r2.62_90_nostd.mrc',
#                 ROOT_PATH + '/data/tutorials/synth_sumb/models/5ide_r2.62_90_nostd.mrc',
#                 ROOT_PATH + '/data/tutorials/synth_sumb/models/5gjv_r2.62_90_nostd.mrc',
#                 ROOT_PATH + '/data/tutorials/synth_sumb/models/5tj6_r2.62_90_nostd.mrc',
#                 ROOT_PATH + '/data/tutorials/synth_sumb/models/5tqq_r2.62_90_nostd.mrc')
mc_avg_nparts = (15, 30, 20, 30, 20)
# mc_avg_nparts = (20, 10, 30, 20, 30, 20, 25, 10)
mc_3sg_nparts = 5
mc_zh = 30
mc_halo_z = 80

########################################################################################
# ADDITIONAL ROUTINES AND STRUCTURES
########################################################################################

class Settings(object):
    snrs = None
    svol_refs = None
    nparts = None
    cent_p = None
    cent_v = None
    mic_rad = None
    tm_nt = None
    tm_size = None
    tm_res = None
    tm_snr_rg = None
    tm_wedge = None
    tm_wedge_rot = None
    tm_bin = None
    tm_rsz = None
    mc_mbt = None
    mc_mbs = None
    mc_ip_min_dst = None
    mc_1st_crad = None
    mc_c_jump_prob = None
    mc_4th_dst = None
    mc_in_models = None
    mc_avg_nparts = None
    mc_3sg_nparts = None
    mc_zh = None
    mc_halo_z = None

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


def add_dmb_msome(tomo, mb_val, rad, res, mb_t, mb_s):
    """
    Add a double Gaussian layered membrane of a microsome to a tomogram
    :param tomo: tomogram where the membrane is added
    :param mb_val: membrane maximum value (if None this is automatedly computed)
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
    if mb_val is None:
        tomo_bin = tomo > (.1 * tomo.max())
        tomo_vals = tomo[tomo_bin]
        mb_val = tomo_vals.mean()

    # Generating the bilayer
    dx, dy, dz = float(tomo.shape[0]), float(tomo.shape[1]), float(tomo.shape[2])
    dx2, dy2, dz2 = math.floor(.5 * dx), math.floor(.5 * dy), math.floor(.5 * dz)
    x_l, y_l, z_l = -dx2, -dy2, -dz2
    x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
    X, Y, Z = np.meshgrid(np.arange(x_l, x_h), np.arange(y_l, y_h), np.arange(z_l, z_h), indexing='xy')
    # X, Y, Z = X.astype(np.float16), Y.astype(np.float16), X.astype(np.float16)
    R = np.sqrt(X * X + Y * Y + Z * Z)
    G_u = mb_val * np.exp(-(R-rad1)**2 / g_cte)
    G_l = mb_val * np.exp(-(R-rad2)**2 / g_cte)

    # Creating the softmaks for the model structure
    BW = sp.ndimage.morphology.distance_transform_edt(tomo == 0)
    Ss = 1. - np.exp(-(BW * BW) / g_cte_2)
    G_u = G_u * Ss
    G_l = G_l * Ss

    # Merge the structure and the membrane
    return tomo + G_u.astype(np.float16) + G_l.astype(np.float16)


def gen_ccsrv_msome(tot_locs, shape, n_parts, mic_rad, c_rad, min_ip_dst, c_jump_prob):
    """
    Generates a list of 3D coordinates and rotations distributed in cluster randomly placed on a microsome
    :param tot_locs: already taken localizations
    :param shape: tomogram shape
    :param n_parts: number of particles to try to generate
    :param mic_rad: microsome radius
    :param c_rad: cluster radius
    :param min_ip_dst: minimum interparticle distance
    :param c_jump_prob: probability to create a new cluster evaluated each time a particle is addded [0, 1]
    :return: two lists; coordinates and rotations
    """

    # Initialization
    count = 0
    min_ip_dst_2 = float(min_ip_dst)**2
    locs_out, rots_out = list(), list()
    mic_cent = .5 * np.asarray(shape, dtype=float)
    mic_rad_f = float(mic_rad)
    max_n_tries = np.prod(np.asarray(shape, dtype=int))
    locs_a = np.asarray(tot_locs, dtype=float)

    # Loop for clusters
    mic_end, n_try = False, 0
    while not mic_end:

        # Generating the cluster center (random sampling on S^2)
        c_cent = np.random.randn(3) # np.asarray((1., 1., 0.))
        norm = mic_rad_f / np.linalg.norm(c_cent)
        c_cent *= norm
        c_cent += mic_cent
        # Check that the cluster center is within the tomogram
        if not((c_cent[0] >= 0) and (c_cent[0] < shape[0])
            and (c_cent[1] >= 0) and (c_cent[1] < shape[1])
            and (c_cent[2] >= 0) and (c_cent[2] < shape[2])):
            n_try += 1
            if (n_try > max_n_tries) or (count >= n_parts):
                mic_end = True
            continue

        # Loop for particles likely within the cluster radius
        c_end = False
        while not c_end:
            rnd_val = np.random.random()
            if (rnd_val > c_jump_prob) and (n_try <= max_n_tries):
                c_part = mic_cent + gen_urnd_sphere_cap(mic_rad, c_rad, a_cap=c_cent-mic_cent)
                # Check that the particle is within the tomogram
                if (c_part[0] >= 0) and (c_part[0] < shape[0]) \
                        and (c_part[1] >= 0) and (c_part[1] < shape[1]) \
                        and (c_part[2] >= 0) and (c_part[2] < shape[2]):
                    # Check that the new particle does not overlap with other already inserted
                    # from other pattern
                    nover_p = True
                    if len(locs_a) > 0:
                        hold_dst = c_part - locs_a
                        nover_p = np.min(np.sum(hold_dst * hold_dst, axis=1)) >= min_ip_dst_2
                    if nover_p:
                        if len(locs_out) > 0:
                            # Check that the new particle does not overlap with other already inserted
                            hold_dst_2 = c_part - np.asarray(locs_out, dtype=float)
                            if np.min(np.sum(hold_dst_2 * hold_dst_2, axis=1)) >= min_ip_dst_2:
                                locs_out.append(c_part)
                                tilt, psi = vect_to_zrelion(c_part - mic_cent, mode='active')[1:]
                                rots_out.append((360.*np.random.rand()-180., tilt, psi))
                                count += 1
                        else:
                            locs_out.append(c_part)
                            tilt, psi = vect_to_zrelion(c_part - mic_cent, mode='active')[1:]
                            rots_out.append((360. * np.random.rand() - 180., tilt, psi))
                            count += 1
                    n_try += 1
                    if (n_try > max_n_tries) or (count >= n_parts):
                        c_end, mic_end = True, True
            else:
                c_end = True
            n_try += 1
            if (n_try > max_n_tries) or (count >= n_parts):
                mic_end = True

        # Ensure termination
        n_try += 1
        if (n_try > max_n_tries) or (count >= n_parts):
            mic_end = True

    return locs_out, rots_out


def gen_2ccsrv_msome(tot_locs, locs, shape, n_parts, mic_rad, avg_dst, min_ip_dst):
    """
    Generates a list of 3D coordinates and rotations distributed in correlation with some reference inputs
    :param tot_locs: already taken localizations
    :param locs: referece input lists with localizations
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
    max_n_tries = np.prod(np.asarray(shape, dtype=int))
    locs_a = np.asarray(tot_locs + locs, dtype=float)

    # Loop for particles
    mic_end, n_try = False, 0
    p_end = False
    while not p_end:
        rnd_idx = np.random.randint(0, len(locs))
        rnd_rad = math.fabs(np.random.normal(avg_dst, 3./min_ip_dst))
        c_cent = locs[rnd_idx]
        p_cent = mic_cent + gen_urnd_sphere_cap(mic_rad, rnd_rad, a_cap=c_cent-mic_cent)
        # Check that the particle is within the tomogram
        if (p_cent[0] >= 0) and (p_cent[0] < shape[0]) \
                and (p_cent[1] >= 0) and (p_cent[1] < shape[1]) \
                and (p_cent[2] >= 0) and (p_cent[2] < shape[2]):
            # Check that the new particle does not overlap with other already inserted
            nover_p = False
            if len(locs_a) > 0:
                hold_dst = p_cent - locs_a
                nover_p = np.min(np.sum(hold_dst * hold_dst, axis=1)) >= min_ip_dst_2
            if nover_p:
                if len(locs_out) > 0:
                    hold_dst_2 = p_cent - np.asarray(locs_out, dtype=float)
                    if np.min(np.sum(hold_dst_2 * hold_dst_2, axis=1)) >= min_ip_dst_2:
                        locs_out.append(p_cent)
                        tilt, psi = vect_to_zrelion(p_cent - mic_cent, mode='active')[1:]
                        rots_out.append((360. * np.random.rand() - 180., tilt, psi))
                        count += 1
                else:
                    locs_out.append(p_cent)
                    tilt, psi = vect_to_zrelion(p_cent - mic_cent, mode='active')[1:]
                    rots_out.append((360. * np.random.rand() - 180., tilt, psi))
                    count += 1

        # Ensure termination
        n_try += 1
        if (n_try > max_n_tries) or (count >= n_parts):
            p_end = True

    return locs_out, rots_out


def gen_csrv_msome(tot_locs, shape, n_parts, mic_rad, min_ip_dst):
    """
    Generates a list of 3D coordinates and rotations a CSRV pattern
    :param tot_locs: already taken localizations
    :param shape: tomogram shape
    :param n_parts: number of particles to try to generate
    :param mic_rad: microsome radius
    :param min_ip_dst: minimum interparticle distance
    :param c_jump_prob: probabilty to create a new cluster evaluated each time a particle is added [0, 1]
    :return: two output lists; coordinates and rotations
    """

    # Initialization
    count = 0
    min_ip_dst_2 = float(min_ip_dst) ** 2
    locs_out, rots_out = list(), list()
    mic_cent = .5 * np.asarray(shape, dtype=float)
    mic_rad_f = float(mic_rad)
    max_n_tries = np.prod(np.asarray(shape, dtype=int))
    locs_a = np.asarray(tot_locs, dtype=float)

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
            # Check that the new particle does not overlap with other from another pattern
            nover_p = True
            if len(locs_a) > 0:
                hold_dst = p_cent - locs_a
                nover_p = np.min(np.sum(hold_dst * hold_dst, axis=1)) >= min_ip_dst_2
            if nover_p:
                if len(locs_out) > 0:
                    # Check that the new particle does not overlap with other already inserted
                    hold_dst_2 = p_cent - np.asarray(locs_out, dtype=float)
                    if np.min(np.sum(hold_dst_2 * hold_dst_2, axis=1)) >= min_ip_dst_2:
                        locs_out.append(p_cent)
                        tilt, psi = vect_to_zrelion(p_cent - mic_cent, mode='active')[1:]
                        rots_out.append((360. * np.random.rand() - 180., tilt, psi))
                        count += 1
                else:
                    locs_out.append(p_cent)
                    tilt, psi = vect_to_zrelion(p_cent - mic_cent, mode='active')[1:]
                    rots_out.append((360. * np.random.rand() - 180., tilt, psi))
                    count += 1

        # Ensure termination
        n_try += 1
        if (n_try > max_n_tries) or (count >= n_parts):
            p_end = True

    return locs_out, rots_out


def insert_particles(tomo, model, cent, cent_v, locs, angs):
    """
    Insert instances of a model in tomogram
    :param tomo: tomogram where the particles are inserted
    :param model: particle model subvolume
    :param cent, cent_v: particle centers
    :param locs: particles localizations
    :param angs: particles rotation angles (Relion)
    :return: the updated tomogram with the particle instances inserted
    """
    for loc, ang in zip(locs, angs):
        svol = tomo_rot(model, ang, conv='relion', active=True, deg=True, center=cent)
        point = loc + cent_v
        spatial.over_sub_tomo(tomo, np.round(point).astype(int), svol, np.maximum)
    return tomo


def pr_routine(pr_id, tomo_ids, settings):

    for i in tomo_ids:

        snr = settings.snrs[i]
        print('\t\t-M[' + str(pr_id) + '/' + str(i) + '] Generating microsome ' + str(i) + ' with SNR=' + str(snr) + ':')
        tm_size_hz = np.asarray(settings.tm_size, dtype=int)
        tm_size_hz[2] -= (2 * settings.mc_halo_z)
        tomo = np.zeros(shape=settings.tm_size, dtype=np.float16)

        print('\t\t\t+M[' + str(pr_id) + '/' + str(i) + '] Adding the particles: ')
        hold_locs, tot_locs = None, list()
        for j, key in enumerate(settings.mc_in_models):
            svol_ref = settings.svol_refs[key]
            npart = settings.nparts[i, j]
            if j == len(settings.mc_in_models) - 1:
                locs, angs = gen_ccsrv_msome(tot_locs, tm_size_hz, npart, settings.mic_rad, settings.mc_1st_crad,
                                             settings.mc_ip_min_dst, settings.mc_c_jump_prob)
                pat_str = 'CCSRV'
            elif j == len(settings.mc_in_models) - 2:
                locs, angs = gen_ccsrv_msome(tot_locs, tm_size_hz, npart, settings.mic_rad,
                                             1.5*settings.mc_1st_crad, settings.mc_ip_min_dst, mc_c_jump_prob)
                pat_str = 'CCSRV'
            elif j == len(settings.mc_in_models) - 3:
                locs, angs = gen_2ccsrv_msome(tot_locs, hold_locs, tm_size_hz, npart, settings.mic_rad,
                                              settings.mc_4th_dst, settings.mc_ip_min_dst)
                pat_str = '2CCSRV'
            else:
                locs, angs = gen_csrv_msome(tot_locs, tm_size_hz, npart, settings.mic_rad,
                                            settings.mc_ip_min_dst)
                pat_str = 'CSRV'
            if j == 0:
                hold_locs = locs
                # pass
            for loc in locs:
                loc[2] += settings.mc_halo_z
            insert_particles(tomo, svol_ref, settings.cent_p, settings.cent_v, locs, angs)
            tot_locs += locs
            try:
                print('\t\t\t\t*M[' + str(pr_id) + '/' + str(i) + '] Particles inserted for ' + pat_str + ' pattern ' + str(j) + \
                      ' (' + os.path.split(key)[1] + ') : ' + str(len(locs)) + ' of ' + str(npart))
            except NameError:
                pass

        print('\t\t\t+M[' + str(pr_id) + '/' + str(i) + '] Adding the membrane...')
        tomo = add_dmb_msome(tomo, None, settings.mic_rad*settings.tm_res, settings.tm_res, settings.mc_mbt,
                             settings.mc_mbs)
        mc_halo_z_f = int(round(0.75 * settings.mc_halo_z))
        tomo[:, :, :mc_halo_z_f] = 0
        tomo[:, :, settings.tm_size[2]-mc_halo_z_f:] = 0
        out_tomo_bin_nodist = out_dir + '/' + out_stem + '_tomo_mic_' + str(i) + '_nodist_bin_' + str(tm_bin) + '.mrc'
        print('\t\t\t-M[' + str(pr_id) + '/' + str(i) + '] Saving the ' + str(tm_bin) + ' binned microsome without distortions as: ' + out_tomo_bin_nodist)
        tomo_bin_nodist = tomo_binning(tomo, settings.tm_bin)
        disperse_io.save_numpy(tomo_bin_nodist, out_tomo_bin_nodist)

        print('\t\t\t+M[' + str(pr_id) + '/' + str(i) + '] Adding the distortions...')
        mask = tomo > 0
        mn = tomo[mask].mean()
        sg_bg = mn / snr
        tomo += np.random.normal(mn, sg_bg, size=settings.tm_size)
        tomo = lin_map(tomo, tomo.max(), tomo.min())
        tomo = add_mw(tomo, settings.tm_wedge_rot, settings.tm_wedge)
        tomo = tomo.astype(np.float16)

        out_tomo = out_dir + '/' + out_stem + '_tomo_mic_' + str(i) + '.mrc'
        print('\t\t\t-M[' + str(pr_id) + '/' + str(i) + '] Saving the microsome as: ' + out_tomo)
        disperse_io.save_numpy(tomo.astype(np.float16), out_tomo)
        out_tomo_bin = out_dir + '/' + out_stem + '_tomo_mic_' + str(i) + '_bin_' + str(tm_bin) + '.mrc'
        print('\t\t\t-M[' + str(pr_id) + '/' + str(i) + '] Saving the ' + str(tm_bin) + ' binned microsome as: ' + out_tomo_bin)
        tomo_bin = tomo_binning(tomo, tm_bin)
        disperse_io.save_numpy(tomo_bin, out_tomo_bin)
        cols.append((out_tomo, out_tomo_bin))

    print('\tProcess ' + str(pr_id) + ' has finished.')
    sys.exit(pr_id)

########################################################################################
# MAIN ROUTINE
########################################################################################

if mp_npr > tm_nt:
    mp_npr = tm_nt

########## Printing the initial message

print('Generate synthetic microsomes.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tMultiprocessing settings:')
print('\t\t-Number of parallel processes: ' + str(mp_npr))
print('\tTomograms settings:')
print('\t\t-Number of tomograms: ' + str(tm_nt))
print('\t\t-Tomogram size: ' + str(tm_size) + ' px')
print('\t\t-Resolution: ' + str(tm_res) + ' nm/px')
print('\t\t-SNR range: ' + str(tm_snr_rg))
print('\t\t-Missing wedge semi-angle: ' + str(tm_wedge) + ' deg')
print('\t\t-Rotation for wedge axis: ' + str(tm_wedge_rot) + ' deg')
print('\t\t-Binning factor: ' + str(tm_bin))
print('\t\t-Sub-volume size: ' + str(tm_size))
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
print('\t\t-Subvolumes center height: ' + str(mc_zh) + ' px')
print('\t\t-Slices to discard (to set zero) on top and bottom of the microsomes: ' + str(mc_halo_z) + ' px')
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
    hold_ref_sz = np.asarray(hold_ref.shape, dtype=int)
    temp = np.zeros(shape=tm_rsz, dtype=hold_ref.dtype)
    off = np.asarray((np.asarray(tm_rsz, dtype=int) / 2) - (hold_ref_sz / 2), dtype=int)
    temp[off[0]:off[0] + hold_ref_sz[0],
         off[1]:off[1] + hold_ref_sz[1],
         off[2]:off[2] + hold_ref_sz[2]] = hold_ref
    svol_refs[in_model] = temp
hold_ref_0, mc_zh = svol_refs[mc_in_models[0]], off[2] + mc_zh
max_tm_size, max_svol_dim = max(tm_size[0:1]), max(hold_ref_0.shape[0:1])
mic_rad = .75 * (max_tm_size/2)
if (mic_rad + max_svol_dim) > max_tm_size:
    mic_rad = max_tm_size - max_svol_dim
cent_p = np.asarray((.5*hold_ref_0.shape[0], .5*hold_ref_0.shape[1], mc_zh), dtype=float)
cent_v = .5*np.asarray(hold_ref_0.shape, dtype=float) - cent_p
mc_ip_min_dst_v, mc_1st_crad_v, mc_4th_dst_v = float(mc_ip_min_dst)/tm_res, float(mc_1st_crad)/tm_res, \
                                               float(mc_4th_dst)/tm_res

print('Main routine: ')

print('\tParallel loop for microsomes: ')
settings = Settings()
settings.snrs = snrs
settings.svol_refs = svol_refs
settings.nparts = nparts
settings.cent_p = cent_p
settings.cent_v = cent_v
settings.mic_rad = mic_rad
settings.tm_nt = tm_nt
settings.tm_size = tm_size
settings.tm_res = tm_res
settings.tm_snr_rg = tm_snr_rg
settings.tm_wedge = tm_wedge
settings.tm_wedge_rot = tm_wedge_rot
settings.tm_bin = tm_bin
settings.tm_rsz = tm_rsz
settings.mc_mbt = mc_mbt
settings.mc_mbs = mc_mbs
settings.mc_ip_min_dst = mc_ip_min_dst_v
settings.mc_1st_crad = mc_1st_crad_v
settings.mc_c_jump_prob = mc_c_jump_prob
settings.mc_4th_dst = mc_4th_dst_v
settings.mc_in_models = mc_in_models
settings.mc_avg_nparts = mc_avg_nparts
settings.mc_3sg_nparts = mc_3sg_nparts
settings.mc_zh = mc_zh
settings.mc_halo_z = mc_halo_z
cols = list()
if mp_npr <= 1:
    pr_routine(0, list(range(tm_nt)), settings)
else:
    processes, pr_results = dict(), dict()
    mp_tomos = np.array_split(list(range(tm_nt)), mp_npr)
    for pr_id in range(mp_npr):
        pr = mp.Process(target=pr_routine, args=(pr_id, mp_tomos[pr_id], settings))
        pr.start()
        processes[pr_id] = pr
    for pr_id, pr in zip(iter(processes.keys()), iter(processes.values())):
        pr.join()
        if pr_id != pr.exitcode:
            print('ERROR: the process ' + str(pr_id) + ' ended unsuccessfully [' + str(pr.exitcode) + ']')
            print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')

out_star = out_dir + '/' + out_stem + '.star'
print('\tStoring output STAR file in: ' + out_star)
star_tomos = sub.Star()
star_tomos.add_column('_rlnMicrographName')
star_tomos.add_column('_rlnImageName')
for i in range(tm_nt):
    out_tomo = out_dir + '/' + out_stem + '_tomo_mic_' + str(i) + '.mrc'
    out_tomo_bin = out_dir + '/' + out_stem + '_tomo_mic_' + str(i) + '_bin_' + str(tm_bin) + '.mrc'
    if os.path.exists(out_tomo) and os.path.exists(out_tomo_bin):
        row = {'_rlnMicrographName': out_tomo, '_rlnImageName': out_tomo_bin}
        star_tomos.add_row(**row)
star_tomos.store(out_star)

print('Successfully terminated. (' + time.strftime("%c") + ')')
