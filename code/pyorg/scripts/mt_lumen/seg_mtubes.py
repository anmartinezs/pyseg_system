"""

    Generates subvolumes and a segmentation mask for lumminal and shell regions for
    every microtubule (the difference with previous versions is that this one allows the
    further direct particle reconstruction within pymtubes package)

    Input:  - STAR file with the next columns:
                + CSV (file from Amira with microtuble ID and centerline sampled) and
                + Tomogram at picking resolution
                + (optional) Tomogram at particle reconstruction resolution
                + (optional) CTF model as subvolume MRC

    Output: - A subvolume for every microtubule
            - A mask for the luminal and shell part of every microtubule
            - A STAR file with the list of segmentations

"""

################# Package import

import os
import sys
import math
import time
import pyseg as ps
from pyorg import sub
from pyorg import pexceptions
from pyorg.seg import *
from pyorg.surf.utils import points_to_poly
import scipy as sp

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

LUM_LBL, SHELL_LBL = 1, 2
GLEVEL_PERCENTIL = 0.001 # %

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-plitzko/Saikat/luminal_density_analysis_all/primary_neurons'

# Input STAR file
in_star = ROOT_PATH + '/in/mts_clines_2.star'

# Output directory
out_dir = ROOT_PATH + '/seg_2'

# CSV file pre-processing
cv_coords_cools = (1, 2, 3)
cv_id_col = 4

# Segmentation settings
sg_res = 1.792 # 1.368 # nm/voxel
sg_rad_lum = 6 # nm
sg_max_shell = 15 # nm
sg_off_voxels = 5 # vox

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Generation segmentation sub-volumes for a micro-tubules network in a tomogram.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tInput STAR file: ' + str(in_star))
print('\tCSV pre-processing: ')
print('\t\t-Columns for samples coordinates (X, Y, Z): ' + str(cv_coords_cools))
print('\t\t-Column for microtubule ID: ' + str(cv_id_col))
print('\tSegmentation settings: ')
print('\t\t-Offset voxels: ' + str(sg_off_voxels))
print('\t\t-Data resolution: ' + str(sg_res) + ' nm/voxel')
print('\t\t-Microtube luminal radius: ' + str(sg_rad_lum) + ' nm')
print('\t\t-Microtube radius: ' + str(sg_max_shell) + ' nm')
print('')

######### Process

print('Parsing input parameters...')
sg_res, sg_rad_lum, sg_max_shell, sg_off_voxels = float(sg_res), float(sg_rad_lum), float(sg_max_shell), \
                                                  int(sg_off_voxels)
svol_off = sg_off_voxels + int(math.ceil(sg_max_shell/sg_res))
out_stem = os.path.splitext(os.path.split(in_star)[1])[0]

print('Loading input STAR file...')
gl_star = sub.Star()
try:
    gl_star.load(in_star)
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
star = sub.Star()
star.add_column(key='_rlnMicrographName')
star.add_column(key='_rlnImageName')
star.add_column(key='_psSegImage')
star.add_column(key='_mtCenterLine')
if gl_star.has_column('_mtParticlesTomo'):
    star.add_column(key='_mtParticlesTomo')
if gl_star.has_column('_rlnCtfImage'):
    star.add_column(key='_rlnCtfImage')
star.add_column(key='_psSegRot')
star.add_column(key='_psSegTilt')
star.add_column(key='_psSegPsi')
star.add_column(key='_psSegOffX')
star.add_column(key='_psSegOffY')
star.add_column(key='_psSegOffZ')

print('Main Routine: tomograms loop')
for row in range(gl_star.get_nrows()):

    in_ref = gl_star.get_element('_rlnMicrographName', row)
    print('\tLoading input tomogram: ' + in_ref)
    out_ref_stem = os.path.splitext(os.path.split(in_ref)[1])[0]
    tomo_ref = ps.disperse_io.load_tomo(in_ref, mmap=True)

    in_csv = gl_star.get_element('_mtMtubesCsv', row)
    print('\tReading input CSV file: ' + in_csv)
    mt_dic = read_csv_mts(in_csv, cv_coords_cools, cv_id_col)

    if gl_star.has_column('_mtParticlesTomo'):
        mt_part = gl_star.get_element('_mtParticlesTomo', row)
        print('\t\t+ Tomogram for particle reconstruction: ' + str(mt_part))
    if gl_star.has_column('_rlnCtfImage'):
        mt_ctf = gl_star.get_element('_rlnCtfImage', row)
        print('\t\t+ CTF Image: ' + str(mt_ctf))

    print('\tMicrotubules loop: ')
    for mt_id, mt_samps in zip(iter(mt_dic.keys()), iter(mt_dic.values())):

        print('\t\t-Processing microtube with ID: ' + str(mt_id))
        mt_points = np.asarray(mt_samps, dtype=np.float32) * (1./sg_res)

        print('\t\t\t+Cropping microtube subvolume...')
        x_min, x_max = int(math.floor(mt_points[:, 0].min())), int(math.ceil(mt_points[:, 0].max()))
        y_min, y_max = int(math.floor(mt_points[:, 1].min())), int(math.ceil(mt_points[:, 1].max()))
        z_min, z_max = int(math.floor(mt_points[:, 2].min())), int(math.ceil(mt_points[:, 2].max()))
        x_min, x_max = x_min-svol_off, x_max+svol_off
        y_min, y_max = y_min-svol_off, y_max+svol_off
        z_min, z_max = z_min-svol_off, z_max+svol_off
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if z_min < 0:
            z_min = 0
        if x_max >= tomo_ref.shape[0]:
            x_max = tomo_ref.shape[0] - 1
        if y_max >= tomo_ref.shape[1]:
            y_max = tomo_ref.shape[1] - 1
        if z_max >= tomo_ref.shape[2]:
            z_max = tomo_ref.shape[2] - 1
        sub_vol = np.asarray(tomo_ref[x_min:x_max, y_min:y_max, z_min:z_max], dtype=np.float32)
        mt_points -= np.asarray((x_min, y_min, z_min), dtype=np.float32)

        print('\t\t-Computing distance map to microtube centerline...')
        max_dst_vox = math.ceil(sg_max_shell / sg_res)
        ## FAST VERSION
        points_mask = points_to_mask(mt_points, sub_vol.shape, inv=True)
        sub_dmap = sp.ndimage.morphology.distance_transform_edt(points_mask, sampling=sg_res,
                                                                return_indices=False)

        print('\t\t\t+Generating microtube segmentation (1-lumen, 2-shell and 0-backgroud)...')
        lum_mask, shell_mask = sub_dmap <= sg_rad_lum, sub_dmap <= sg_max_shell
        mt_mask = lum_mask | shell_mask
        sub_seg = np.zeros(shape=sub_dmap.shape, dtype=np.int8)
        sub_seg[shell_mask] = SHELL_LBL
        sub_seg[lum_mask] = LUM_LBL

        print('\t\t+Subvolume density normalization...')
        mt_density = sub_vol[mt_mask]
        try:
            sub_vol = (sub_vol-mt_density.mean()) / mt_density.std()
        except ZeroDivisionError:
            print('WARNING: std=0 for microtube density, this tomogram cannot be processed, skipping...')
            continue

        # Crop lower and higher density values
        lb, ub = np.percentile(sub_vol, GLEVEL_PERCENTIL), \
                 np.percentile(sub_vol, 100-GLEVEL_PERCENTIL)
        sub_vol[sub_vol < lb] = lb
        sub_vol[sub_vol > ub] = ub
        sub_vol = ps.globals.lin_map(sub_vol, lb=0, ub=1)
        sub_vol[sub_vol < 0] = 0
        sub_vol[sub_vol > 1] = 1

        print('\t\t-Storing sub-volumes...')
        ct_vtp = points_to_poly(mt_points, normals=None)
        out_ct = out_dir + '/' + out_ref_stem + '_mt' + str(mt_id) + '_ct.vtp'
        mt_vol_fname = out_dir+'/'+out_ref_stem+'_mt_'+str(mt_id)+'.mrc'
        mt_seg_fname = out_dir+'/'+out_ref_stem+'_mt_'+str(mt_id)+'_seg.mrc'
        ps.disperse_io.save_numpy(sub_vol, mt_vol_fname)
        ps.disperse_io.save_numpy(sub_seg, mt_seg_fname)
        ps.disperse_io.save_vtp(ct_vtp, out_ct)
        kwargs = {'_rlnMicrographName': in_ref, '_rlnImageName': mt_vol_fname,
                  '_psSegImage': mt_seg_fname, '_mtCenterLine' : out_ct,
                  '_mtParticlesTomo': mt_part, '_rlnCtfImage': mt_ctf,
                  '_psSegRot': 0., '_psSegTilt': 0., '_psSegPsi': 0.,
                  '_psSegOffX': x_min, '_psSegOffY': y_min, '_psSegOffZ': z_min}
        star.add_row(**kwargs)

out_star = out_dir + '/' + out_stem + '_mts.star'
print('\tStoring output STAR file in: ' + out_star)
star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')