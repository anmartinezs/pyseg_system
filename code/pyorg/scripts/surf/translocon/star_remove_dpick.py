# -*- coding: utf-8 -*-
"""Script for removig double picked partiles in STAR file."""


import os
import math
import logging
import time
import csv
import itertools
import operator
import copy as cp
import scipy as sp
import numpy as np

from datetime import timedelta

# import pyseg as ps
from pyorg import sub, disperse_io

# filepaths
# ROOT_PATH = '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext_repick/FTR/stat'
ROOT_PATH = '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext_repick/ribo'

# Input STAR file
in_star = ROOT_PATH + '/ref_rln2/run1_ribo_data.star' # '/../FTR/stat/in/2_class_T.star' # '/class_ABC/run1_C_it050_data.star'

# To select a specific microsome
in_mic = None # '/fs/pool/pool-lucic2/johannes/tomograms/stefan/160614/tomo22_z120_ves_1.mrc' # '/fs/pool/pool-lucic2/johannes/tomograms/stefan/160614/oriented_ribo_bin2/tomo22_z120_ves_1.mrc'

# Output STAR file
out_star = ROOT_PATH + '/ref_rln2/0_run1_ribo_dpick.star' # '/ref_rln2/2_class_T_dpick.star' # '/class_ABC/run1_C_it050_data_dp8_v2.star'

# Parameters
pre_bin = None # .5
res = 1.048 # nm/vx
ssup = 15 # 8 # 20 # 8 # nm
s_ubin = 2. # 4.
s_pbin = 0.5 # 1.0
seg_lbl = 1
err_dst = 15 # 5 # 15 # nm
err_ptg = 60 # %
max_shift = 30 # 10 # nm
log_crit = True

def main():

    start_time = time.time()

    print('Loading star file: {}.'.format(in_star))
    star, star_ves = sub.Star(), sub.Star()
    star.load(in_star)
    star_ves = cp.deepcopy(star)

    print('Pre-processing input star file...')

    print('\tRemoving rlnRandomSubset column...')
    if star.has_column('_rlnRandomSubset'):
        star.del_column('_rlnRandomSubset')
        star_ves.del_column('_rlnRandomSubset')

    if in_mic is not None:
        print('\tChoosing micrograph: ' + in_mic)
        del_l = list()
        for row in range(star.get_nrows()):
            if star.get_element('_rlnMicrographName', row) != in_mic:
                del_l.append(row)
        print('\t\t-Deleting ' + str(len(del_l)) + ' of ' + str(star.get_nrows()) + ' particles.')
        star.del_rows(del_l)
        star_ves.del_rows(del_l)

    print('\tParticles pre-processing...')
    for row in range(star.get_nrows()):
        x = star.get_element('_rlnCoordinateX', row) * s_pbin
        y = star.get_element('_rlnCoordinateY', row) * s_pbin
        z = star.get_element('_rlnCoordinateZ', row) * s_pbin
        star.set_element('_rlnCoordinateX', row, x)
        star.set_element('_rlnCoordinateY', row, y)
        star.set_element('_rlnCoordinateZ', row, z)
        star_ves.set_element('_rlnCoordinateX', row, x * 2.)
        star_ves.set_element('_rlnCoordinateY', row, y * 2.)
        star_ves.set_element('_rlnCoordinateZ', row, z * 2.)

    print('\tRemoving highly shifted particles (' + str(max_shift) + ' nm)')
    del_l = list()
    max_shift_v = max_shift / (res / float(s_ubin))
    for row in range(star.get_nrows()):
        # Getting shifting coordinates
        sx = star.get_element('_rlnOriginX', row)
        sy = star.get_element('_rlnOriginY', row)
        sz = star.get_element('_rlnOriginZ', row)
        shifts = np.asarray((sx, sy, sz), dtype=float)
        dst = math.sqrt((shifts * shifts).sum())
        if dst > max_shift_v:
            del_l.append(row)
    print('\t\t-Deleting ' + str(len(del_l)) + ' of ' + str(star.get_nrows()) + ' particles.')
    star.del_rows(del_l)
    star_ves.del_rows(del_l)

    print('\tRemoving badly segmented microsomes...')
    err_dst_v = err_dst / res
    pt_ssup_v = ssup / res
    s_bin = 1. / s_ubin
    parts_mic, del_l = dict(), list()
    part_coords = np.zeros(shape=(star.get_nrows(), 3), dtype=np.float32)
    for row in range(star.get_nrows()):
        # Getting particle coordinates
        x = star.get_element('_rlnCoordinateX', row)
        y = star.get_element('_rlnCoordinateY', row)
        z = star.get_element('_rlnCoordinateZ', row)
        sx = star.get_element('_rlnOriginX', row)
        sy = star.get_element('_rlnOriginY', row)
        sz = star.get_element('_rlnOriginZ', row)
        part_coords[row, :] = x - sy * s_bin, y - sx * s_bin, z - sz * s_bin
        # part_coords[row, :] = x - sx * s_bin, y - sy * s_bin, z - sz * s_bin
        # Adding partcle to micrographs dictionary
        mic = star.get_element('_rlnMicrographName', row)
        mic_seg = os.path.split(mic)[0] + '/' + os.path.splitext(os.path.split(mic)[1])[0] + '_seg.mrc'
        path_mic, stem_mic = os.path.split(mic_seg)
        path_mic = path_mic.replace('/oriented_ribo_bin2', '')
        name = stem_mic.split('_')
        seg_star = sub.Star()
        seg_star.load(path_mic + '/graph_ribo/' + name[0] + '_' + name[1] + '_mb_graph.star')
        try:
            row_seg = seg_star.find_element('_psSegImage', mic_seg)
        except ValueError:
            hold_path, hold_stem = os.path.split(mic_seg)
            mic_seg = hold_path + '/oriented_ribo_bin2/' + hold_stem
            row_seg = seg_star.find_element('_psSegImage', mic_seg)
        part_coords[row, 0] -= seg_star.get_element('_psSegOffY', row_seg)
        part_coords[row, 1] -= seg_star.get_element('_psSegOffX', row_seg)
        part_coords[row, 2] -= seg_star.get_element('_psSegOffZ', row_seg)
        # Finding segmentation offset
        try:
            parts_mic[mic_seg].append(row)
        except KeyError:
            parts_mic[mic_seg] = list()
            parts_mic[mic_seg].append(row)
    mics_del, tot_dst = 0, 0
    for mic_seg, rows in zip(iter(parts_mic.keys()), iter(parts_mic.values())):
        try:
            mic = disperse_io.load_tomo(mic_seg)
        except IOError:
            hold_path, hold_stem = os.path.split(mic_seg)
            mic_seg = hold_path + '/oriented_ribo_bin2/' + hold_stem
            mic = disperse_io.load_tomo(mic_seg)
        mic_dst = sp.ndimage.morphology.distance_transform_edt(mic!=seg_lbl)
        num_err = 0
        for row in rows:
            coord = np.round(part_coords[row, :]).astype(int)
            hold_dst = mic_dst[coord[1], coord[0], coord[2]]
            tot_dst += hold_dst
            if hold_dst > err_dst_v:
                del_l.append(row)
                num_err += 1
        ptg = 100. * (num_err / float(len(rows)))
        print('\t\tProcessing microsome: ' + mic_seg)
        print('\t\t\t-Number of particles: ' + str(len(rows)))
        print('\t\t\t-Number of bad particles: ' + str(num_err) + ' (' + str(ptg) + '%)')
        if ptg > err_ptg:
            print('\t\t\t-BAD MICROSOME DELETING ALL PARTICLES!')
            for row in rows:
                try:
                    del_l.index(row)
                except ValueError:
                    del_l.append(row)
            mics_del += 1
    print('\tTotal distance measured + ' + str(tot_dst) + ' vx')
    print('\tDeleted microsomes + ' + str(mics_del) + ' of ' + str(len(list(parts_mic.keys()))))
    print('\t\t-Deleted ' + str(len(del_l)) + ' of ' + str(star.get_nrows()) + ' particles.')
    star.del_rows(del_l)
    star_ves.del_rows(del_l)

    print('\tFrom microsomes path indexing to tomograms path indexing...')
    for row in range(star.get_nrows()):
        hold_mic = star.get_element('_rlnMicrographName', row)
        new_mic = hold_mic.replace('/oriented_ribo_bin2', '')
        tomo_path, fname = os.path.split(new_mic)
        hold_stem = fname.split('_ves_')[0]
        star.set_element(key='_rlnMicrographName', val=tomo_path + '/' + hold_stem + '_bin_4.em', row=row)

    print('\t\tApplying scale suppression (' + str(pt_ssup_v) + ' vx)...')
    # 'Computing tomograms dictionary
    parts_mic, del_l = dict(), list()
    part_coords = np.zeros(shape=(star.get_nrows(), 3), dtype=np.float32)
    for row in range(star.get_nrows()):
        # Getting particle coordinates
        x = star.get_element('_rlnCoordinateX', row)
        y = star.get_element('_rlnCoordinateY', row)
        z = star.get_element('_rlnCoordinateZ', row)
        sx = star.get_element('_rlnOriginX', row)
        sy = star.get_element('_rlnOriginY', row)
        sz = star.get_element('_rlnOriginZ', row)
        part_coords[row, :] = x - sy*s_bin, y - sx*s_bin, z - sz*s_bin
        # part_coords[row, :] = x - sx * s_bin, y - sy * s_bin, z - sz * s_bin
        # Adding partcle to micrographs dictionary
        mic = star.get_element('_rlnMicrographName', row)
        try:
            parts_mic[mic].append(row)
        except KeyError:
            parts_mic[mic] = list()
            parts_mic[mic].append(row)
    # Particle suppression on output STAR file (maximum likelihood criterium)
    if log_crit:
        for mic, rows in zip(iter(parts_mic.keys()), iter(parts_mic.values())):
            mic_coords = np.zeros(shape=(len(rows), 3), dtype=np.float32)
            mic_lut = np.ones(shape=len(rows), dtype=bool)
            mic_logs = np.zeros(shape=len(rows), dtype=bool)
            for i, row in enumerate(rows):
                mic_coords[i, :] = part_coords[row, :]
                mic_logs[i] = star.get_element('_rlnLogLikeliContribution', row)
            log_ids = np.argsort(mic_logs)[::-1]
            for i in log_ids:
                if mic_lut[i]:
                    hold = mic_coords - mic_coords[i, :]
                    dsts = np.sqrt((hold * hold).sum(axis=1))
                    ids = np.where((dsts < pt_ssup_v) & mic_lut)[0]
                    # Only clean neighbours when we are place at maximum
                    for idx in ids:
                        if mic_lut[idx] and (idx != i):
                            mic_lut[idx] = False
                            del_l.append(rows[idx])
    else:
        # Particle suppression on output STAR file (first found criterium)
        for mic, rows in zip(iter(parts_mic.keys()), iter(parts_mic.values())):
            mic_coords = np.zeros(shape=(len(rows), 3), dtype=np.float32)
            mic_lut = np.ones(shape=len(rows), dtype=bool)
            for i, row in enumerate(rows):
                mic_coords[i, :] = part_coords[row, :]
            for i, coord in enumerate(mic_coords):
                if mic_lut[i]:
                    hold = mic_coords - coord
                    dsts = np.sqrt((hold * hold).sum(axis=1))
                    ids = np.where(dsts < pt_ssup_v)[0]
                    for idx in ids:
                        if mic_lut[idx] and (idx != i):
                            mic_lut[idx] = False
                            del_l.append(rows[idx])
    print('\t\t-Deleted ' + str(len(del_l)) + ' of ' + str(star.get_nrows()) + ' particles.')
    star.del_rows(del_l)
    star_ves.del_rows(del_l)

    print('\tChecking removing procedure...')
    parts_mic = dict()
    part_coords = np.zeros(shape=(star.get_nrows(), 3), dtype=np.float32)
    for row in range(star.get_nrows()):
        # Getting particle coordinates
        x = star.get_element('_rlnCoordinateX', row)
        y = star.get_element('_rlnCoordinateY', row)
        z = star.get_element('_rlnCoordinateZ', row)
        sx = star.get_element('_rlnOriginX', row)
        sy = star.get_element('_rlnOriginY', row)
        sz = star.get_element('_rlnOriginZ', row)
        part_coords[row, :] = x - sy * s_bin, y - sx * s_bin, z - sz * s_bin
        # part_coords[row, :] = x - sx * s_bin, y - sy * s_bin, z - sz * s_bin
        # Adding partcle to micrographs dictionary
        mic = star.get_element('_rlnMicrographName', row)
        try:
            parts_mic[mic].append(row)
        except KeyError:
            parts_mic[mic] = list()
            parts_mic[mic].append(row)
    for mic, rows in zip(iter(parts_mic.keys()), iter(parts_mic.values())):
        if len(rows) <= 1:
            continue
        mic_coords = np.zeros(shape=(len(rows), 3), dtype=np.float32)
        for i, row in enumerate(rows):
            mic_coords[i, :] = part_coords[row, :]
        for i, coord in enumerate(mic_coords):
            hold = mic_coords - coord
            dsts = np.sqrt((hold * hold).sum(axis=1))
            dsts_min = np.sort(dsts)[1]
            if dsts_min <= pt_ssup_v:
                print('\t-WARNING: particle in row ' + str(rows[i]) + ' with minimum distance ' + str(dsts_min*res) + 'nm not suppressed!')
    imgs = star.get_column_data('_rlnImageName')
    for row, img in enumerate(imgs):
        if imgs.count(img) != 1:
            print('\t-WARNING: not a single entry for particle in row ' + str(row) + ' with image name ' + imgs)

    # Store the Star file
    print('Storing output Star file in: ' + out_star)
    star.store(out_star)

    out_star_stem = os.path.splitext(out_star)[0]
    out_star_ves = out_star_stem + '_ves.star'
    print('Storing output Star with vesicles file in: ' + out_star_ves)
    star_ves.store(out_star_ves)

    out_path, out_stem = os.path.split(out_star)
    out_star_shift = out_path + '/' + os.path.splitext(out_stem)[0] + '_shift.star'
    print('\tGenerating the shifted version: ' + out_star_shift)
    for row in range(star.get_nrows()):
        # Getting particle coordinates
        x = star.get_element('_rlnCoordinateX', row)
        y = star.get_element('_rlnCoordinateY', row)
        z = star.get_element('_rlnCoordinateZ', row)
        sx = star.get_element('_rlnOriginX', row)
        sy = star.get_element('_rlnOriginY', row)
        sz = star.get_element('_rlnOriginZ', row)
        star.set_element('_rlnCoordinateX', row, x - sy*s_bin)
        star.set_element('_rlnCoordinateY', row, y - sx*s_bin)
        star.set_element('_rlnCoordinateZ', row, z - sz*s_bin)
        star.set_element('_rlnOriginX', row, 0)
        star.set_element('_rlnOriginY', row, 0)
        star.set_element('_rlnOriginZ', row, 0)
    star.store(out_star_shift)

    print('Finished. Runtime {}.'.format(str(timedelta(seconds=time.time()-start_time))))

if __name__ == "__main__":
    main()