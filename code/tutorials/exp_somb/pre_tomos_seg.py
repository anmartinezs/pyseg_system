"""

    Pre-processing for mb_graph_batch.py of oriented membranes from TomoSegMemTV output

    Input:  - STAR file with 3 columns:
                + _rlnMicrographName: tomogram original
                + _rlnImageName: TomoSegMemTV density map output
                + _psSegLabel: (optional) label for membrane segmentation
                + _psSegImage: (optional) binary mask to focus the segmentation analysis
                + _mtMtubesCsv: (optional) a .csv file with microtubule center lines
            - Setting for segmenting the membranes from TomoSegMemTV density map:
                + Density threshold: (optional) required if _psSegLabel not defined
                + Size threshold: (optional) required if _psSegLabel not defined
            - Sub-volume splitting settings

    Output: - A STAR file with 3 columns:
                + _rlnMicrographName: tomogram original
                + _rlnImageName: sub-volumes
                + _psSegImage: Un-oriented membrane segmentations for each subvolume
                + Columns for localizing the sub-volumes within each original tomogram

"""

################# Package import
import argparse
import gc
import os
import sys
import math
import time
import pyseg as ps
import scipy as sp
import skimage as sk
import numpy as np
from pyseg.globals import signed_distance_2d

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

MB_LBL, MB_NEIGH = 1, 2
MB_NEIGH_INT, MB_NEIGH_EXT = 2, 3

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-ruben/antonio/shiwei'

# Input STAR file
in_star = ROOT_PATH + '/pre/in/mb_seg_single_oriented.star'

# Output directory
out_dir = ROOT_PATH + '/pre/mbo_nosplit'

# Subvolume splitting settings
sp_split = None # (2, 2, 1)
sp_off_voxels = 30 # vox

# Membrane segmentation
sg_res = 0.52 # nm/voxel
sg_th = None # 8
sg_sz = None # 3e3
sg_mb_thick = 4 # nm
sg_mb_neigh = 15 # nm

# CSV file pre-processing
cv_coords_cools = (1, 2, 3)
cv_id_col = 4

# Microtubule settings
mt_rad = 30 # nm
mt_swap_xy = False


########################################################################################
# MAIN ROUTINE
########################################################################################

# Get them from the command line if they were passed through it
parser = argparse.ArgumentParser()
parser.add_argument('--inStar', default=in_star, help='Input star file.')
parser.add_argument('--outDir', default=out_dir, help='Output directory.')
parser.add_argument('--spSplit', nargs='+', type=int, default=sp_split, help='Number of splits (X, Y, Z).')
parser.add_argument('--spOffVoxels', type=int, default=sp_off_voxels, help='Offset voxels.')
parser.add_argument('--sgVoxelSize', default=sg_res, type=float, help='Voxel size (nm/voxel).')
parser.add_argument('--sgThreshold', type=int, default=sg_th, help='Density threshold.')
parser.add_argument('--sgSizeThreshold', type=int, default=sg_sz, help='Size threshold (voxels).')
parser.add_argument('--sgMembThk', default=sg_mb_thick, type=float, help='Segmented membrane thickness (nm)')
parser.add_argument('--sgMembNeigh', default=sg_mb_neigh, type=float, help='Segmented membrane neighbours (nm)')

args = parser.parse_args()
in_star = args.inStar
out_dir = args.outDir
sp_split = None if args.spSplit == [-1] else args.spSplit
sp_off_voxels = args.spOffVoxels
sg_res = args.sgVoxelSize
sg_th = None if args.sgThreshold == -1 else args.sgThreshold
sg_sz = None if args.sgSizeThreshold == -1 else args.sgSizeThreshold
sg_mb_thick = args.sgMembThk
sg_mb_neigh = args.sgMembNeigh

########## Print initial message

print('Pre-processing for SEG analysis of un-oriented membranes from TomoSegMemTV output.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tInput STAR file: ' + str(in_star))
print('\tData resolution: ' + str(sg_res) + ' nm/vx')
if sg_th is not None:
    print('\tSegmentation settings: ')
    print('\t\t-Density threshold: ' + str(sg_th))
    print('\t\t-Size threshold: ' + str(sg_sz) + ' vx')
print('\tSub-volume splitting settings: ')
print('\t\t-Number of splits (X, Y, Z): ' + str(sp_split))
print('\t\t-Offset voxels: ' + str(sp_off_voxels))
print('\tMicrotubule settings:')
print('\t\t-Microtube luminal radius: ' + str(mt_rad) + ' nm')
print('\tCSV pre-processing: ')
print('\t\t-Columns for samples coordinates (X, Y, Z): ' + str(cv_coords_cools))
print('\t\t-Column for microtubule ID: ' + str(cv_id_col))
print('')

######### Process

print('Parsing input parameters...')
sp_res, mt_rad, sp_off_voxels = float(sg_res), float(mt_rad), int(sp_off_voxels)
out_stem = os.path.splitext(os.path.split(in_star)[1])[0]
conn_mask = np.ones(shape=(3,3,3))
out_seg_dir = out_dir + '/segs'
if not os.path.isdir(out_seg_dir):
    os.makedirs(out_seg_dir)

print('Loading input STAR file...')
gl_star = ps.sub.Star()
try:
    gl_star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
star = ps.sub.Star()
star.add_column(key='_rlnMicrographName')
star.add_column(key='_rlnImageName')
star.add_column(key='_psSegImage')
star.add_column(key='_psSegRot')
star.add_column(key='_psSegTilt')
star.add_column(key='_psSegPsi')
star.add_column(key='_psSegOffX')
star.add_column(key='_psSegOffY')
star.add_column(key='_psSegOffZ')

mode_oriented = False
if gl_star.has_column('_rlnOriginX') and gl_star.has_column('_rlnOriginY') and gl_star.has_column('_rlnOriginZ'):
    print('\t-Segmentation origin found, oriented membrane segmentation activated!')
    mode_oriented = True

print('Main Routine: tomograms loop')
tomo_id = 0
for row in range(gl_star.get_nrows()):

    in_ref = gl_star.get_element('_rlnMicrographName', row)
    print('\tProcessing tomogram: ' + in_ref)
    out_ref_stem = os.path.splitext(os.path.split(in_ref)[1])[0]
    in_mb = gl_star.get_element('_rlnImageName', row)
    print('\t\t-Loading membrane segmentation: ' + in_mb)
    tomo_mb = ps.disperse_io.load_tomo(in_mb)
    tomo_ref = ps.disperse_io.load_tomo(in_ref, mmap=True)
    off_mask_min_x, off_mask_max_x = 0, tomo_ref.shape[0]
    off_mask_min_y, off_mask_max_y = 0, tomo_ref.shape[1]
    off_mask_min_z, off_mask_max_z = 0, tomo_ref.shape[2]
    wide_x = off_mask_max_x - off_mask_min_x
    wide_y = off_mask_max_y - off_mask_min_y
    wide_z = off_mask_max_z - off_mask_min_z

    mt_mask = None
    if gl_star.has_column('_mtMtubesCsv'):
        in_csv = gl_star.get_element('_mtMtubesCsv', row)
        print('\tReading input CSV file: ' + in_csv)
        mt_dic = ps.globals.read_csv_mts(in_csv, cv_coords_cools, cv_id_col, swap_xy=mt_swap_xy)
        mts_points = list()
        for mt_id, mt_samps in zip(iter(mt_dic.keys()), iter(mt_dic.values())):
            mts_points += mt_samps
        mts_points = np.asarray(mts_points, dtype=np.float32) * (1./sg_res)

        print('\tSegmenting the microtubules...')
        mt_mask = ps.globals.points_to_mask(mts_points, tomo_mb.shape, inv=True)
        mt_mask = sp.ndimage.morphology.distance_transform_edt(mt_mask, sampling=sg_res, return_indices=False)
        mt_mask = mt_mask > mt_rad

    mb_lbl = 0
    if sg_th is None:
        if gl_star.has_column('_psSegLabel'):
            mb_lbl = gl_star.get_element('_psSegLabel', row)
            print('\t\t\t+Segmenting membranes with label: ' + str(mb_lbl))
            if mb_lbl > 0:
                tomo_mb = tomo_mb == mb_lbl
            else:
                tomo_mb = tomo_mb > 0
        else:
            tomo_mb = tomo_mb > 0
    else:
        tomo_mb = tomo_mb >= sg_th
    if gl_star.has_column('_mtMtubesCsv'):
        tomo_mb *= mt_mask
        del mt_mask
    if gl_star.has_column('_psSegImage'):
        print('\tApplying the mask...')
        hold_mask = ps.disperse_io.load_tomo(gl_star.get_element('_psSegImage', row))
        if mb_lbl > 0:
            hold_mask = hold_mask == mb_lbl
        else:
            hold_mask = hold_mask > 0
        tomo_mb *= hold_mask
        ids_mask = np.where(hold_mask)
        off_mask_min_x, off_mask_max_x = ids_mask[0].min()-sp_off_voxels, ids_mask[0].max()+sp_off_voxels
        if off_mask_min_x < 0:
            off_mask_min_x = 0
        if off_mask_max_x > hold_mask.shape[0]:
            off_mask_max_x = hold_mask.shape[0]
        off_mask_min_y, off_mask_max_y = ids_mask[1].min()-sp_off_voxels, ids_mask[1].max()+sp_off_voxels
        if off_mask_min_y < 0:
            off_mask_min_y = 0
        if off_mask_max_y > hold_mask.shape[1]:
            off_mask_max_y = hold_mask.shape[1]
        off_mask_min_z, off_mask_max_z = ids_mask[2].min()-sp_off_voxels, ids_mask[2].max()+sp_off_voxels
        if off_mask_min_z < 0:
            off_mask_min_z = 0
        if off_mask_max_z > hold_mask.shape[2]:
            off_mask_max_z = hold_mask.shape[2]
        del hold_mask
        del ids_mask
    # ps.disperse_io.save_numpy(tomo_mb, out_dir + '/hold.mrc')
    if sg_th is not None:
        print('\tMembrane thresholding...')
        tomo_sz = ps.globals.global_analysis(tomo_mb, 0.5, c=26)
        tomo_mb = tomo_sz > sg_sz
        del tomo_sz

    seg_center = None
    if mode_oriented:
        seg_center = np.asarray((gl_star.get_element('_rlnOriginX', row),
                                 gl_star.get_element('_rlnOriginY', row),
                                 gl_star.get_element('_rlnOriginZ', row)))
        seg_center[0] -= off_mask_min_x
        seg_center[1] -= off_mask_min_y
        seg_center[2] -= off_mask_min_z

    print('\tSegmenting the membranes...')
    if sp_split is None:
        svol_mb = tomo_mb[off_mask_min_x:off_mask_max_x, off_mask_min_y:off_mask_max_y, off_mask_min_z:off_mask_max_z]
        svol = tomo_ref[off_mask_min_x:off_mask_max_x, off_mask_min_y:off_mask_max_y, off_mask_min_z:off_mask_max_z]
        svol_dst = sp.ndimage.morphology.distance_transform_edt(np.invert(svol_mb), sampling=sg_res,
                                                                return_indices=False)
        svol_seg = np.zeros(shape=svol.shape, dtype=np.float32)
        if not mode_oriented:
            svol_seg[svol_dst < sg_mb_neigh + sg_mb_thick] = MB_NEIGH
            svol_seg[svol_dst < sg_mb_thick] = MB_LBL
        else:
            svol_dst = signed_distance_2d(svol_mb, res=1, del_b=True, mode_2d=True, set_point=seg_center)
            svol_seg[(svol_dst > 0) & (svol_dst < sg_mb_neigh + sg_mb_thick)] = MB_NEIGH_INT
            svol_seg[(svol_dst < 0) & (svol_dst > -1. * (sg_mb_neigh + sg_mb_thick))] = MB_NEIGH_EXT
            svol_seg[np.absolute(svol_dst) < sg_mb_thick] = MB_LBL
            svol_seg[svol_dst == 0] = 0
            svol_seg[svol_mb > 0] = MB_LBL
        out_svol = out_seg_dir + '/' + out_ref_stem + '_tid_' + str(tomo_id) + '.mrc'
        out_seg = out_seg_dir + '/' + out_ref_stem + '_tid_' + str(tomo_id) + '_seg.mrc'
        ps.disperse_io.save_numpy(svol, out_svol)
        ps.disperse_io.save_numpy(svol_seg, out_seg)
        del svol_seg
        del svol_dst
        row_dic = dict()
        row_dic['_rlnMicrographName'] = in_ref
        row_dic['_rlnImageName'] = out_svol
        row_dic['_psSegImage'] = out_seg
        row_dic['_psSegRot'] = 0
        row_dic['_psSegTilt'] = 0
        row_dic['_psSegPsi'] = 0
        row_dic['_psSegOffX'] = off_mask_min_x # 0
        row_dic['_psSegOffY'] = off_mask_min_y # 0
        row_dic['_psSegOffZ'] = off_mask_min_z
        star.add_row(**row_dic)
    else:
        print('\tSplitting into subvolumes:')
        if sp_split[0] > 1:
            hold_wide = int(math.ceil(wide_x / sp_split[0]))
            hold_pad = int(math.ceil((off_mask_max_x - off_mask_min_x) / sp_split[0]))
            hold_split = int(sp_split[0] * math.ceil(float(hold_pad)/hold_wide))
            offs_x = list()
            pad_x = off_mask_min_x + int(math.ceil((off_mask_max_x-off_mask_min_x) / hold_split))
            offs_x.append((off_mask_min_x, pad_x+sp_off_voxels))
            lock = False
            while not lock:
                hold = offs_x[-1][1] + pad_x
                if hold >= off_mask_max_x:
                    offs_x.append((offs_x[-1][1] - sp_off_voxels, off_mask_max_x))
                    lock = True
                else:
                    offs_x.append((offs_x[-1][1]-sp_off_voxels, offs_x[-1][1]+pad_x+sp_off_voxels))
        else:
            offs_x = [(off_mask_min_x, off_mask_max_x),]
        if sp_split[1] > 1:
            hold_wide = int(math.ceil(wide_y / sp_split[1]))
            hold_pad = int(math.ceil((off_mask_max_y - off_mask_min_y) / sp_split[1]))
            hold_split = int(sp_split[1] * math.ceil(float(hold_pad) / hold_wide))
            offs_y = list()
            pad_y = off_mask_min_y + int(math.ceil((off_mask_max_y-off_mask_min_y) / hold_split))
            offs_y.append((off_mask_min_x, pad_y + sp_off_voxels))
            lock = False
            while not lock:
                hold = offs_y[-1][1] + pad_y
                if hold >= off_mask_max_y:
                    offs_y.append((offs_y[-1][1] - sp_off_voxels, off_mask_max_y))
                    lock = True
                else:
                    offs_y.append((offs_y[-1][1] - sp_off_voxels, offs_y[-1][1] + pad_y + sp_off_voxels))
        else:
            offs_y = [(off_mask_min_y, off_mask_max_y),]
        if sp_split[2] > 1:
            hold_wide = int(math.ceil(wide_z / sp_split[2]))
            hold_pad = int(math.ceil((off_mask_max_z - off_mask_min_z) / sp_split[2]))
            hold_split = int(sp_split[2] * math.ceil(float(hold_pad) / hold_wide))
            offs_z = list()
            pad_z = off_mask_min_z + int(math.ceil((off_mask_max_z-off_mask_min_z) / hold_split))
            offs_z.append((off_mask_min_z, pad_z + sp_off_voxels))
            lock = False
            while not lock:
                hold = offs_z[-1][1] + pad_z
                if hold >= off_mask_max_z:
                    offs_z.append((offs_z[-1][1] - sp_off_voxels, off_mask_max_z))
                    lock = True
                else:
                    offs_z.append((offs_z[-1][1] - sp_off_voxels, offs_z[-1][1] + pad_z + sp_off_voxels))
        else:
            offs_z = [(off_mask_min_z, off_mask_max_z),]
        split_id = 1
        for off_x in offs_x:
            for off_y in offs_y:
                for off_z in offs_z:
                    print('\t\t-Splitting subvolume: [' + str(off_x) + ', ' + str(off_y) + ', ' + str(off_z) +']')
                    svol_mb = tomo_mb[off_x[0]:off_x[1], off_y[0]:off_y[1], off_z[0]:off_z[1]]
                    svol = tomo_ref[off_x[0]:off_x[1], off_y[0]:off_y[1], off_z[0]:off_z[1]]
                    svol_seg = np.zeros(shape=svol.shape, dtype=np.float32)
                    if not mode_oriented:
                        svol_dst = sp.ndimage.morphology.distance_transform_edt(np.invert(svol_mb), sampling=sg_res,
                                                                                return_indices=False)
                        svol_seg[svol_dst < sg_mb_neigh + sg_mb_thick] = MB_NEIGH
                        svol_seg[svol_dst < sg_mb_thick] = MB_LBL
                    else:
                        seg_off_center = seg_center - np.asarray((off_x[0], off_y[0], off_z[0]))
                        svol_dst = signed_distance_2d(svol_mb, res=1, del_b=True, mode_2d=True,
                                                      set_point=seg_off_center)
                        svol_seg[(svol_dst > 0) & (svol_dst < sg_mb_neigh + sg_mb_thick)] = MB_NEIGH_INT
                        svol_seg[(svol_dst < 0) & (svol_dst > -1. * (sg_mb_neigh + sg_mb_thick))] = MB_NEIGH_EXT
                        svol_seg[np.absolute(svol_dst) < sg_mb_thick] = MB_LBL
                        svol_seg[svol_dst == 0] = 0
                        svol_seg[svol_mb > 0] = MB_LBL
                    out_svol = out_seg_dir + '/' + out_ref_stem + '_id_' + str(tomo_id) + '_split_' + str(split_id) + '.mrc'
                    out_seg = out_seg_dir + '/' + out_ref_stem + '_id_' + str(tomo_id) + '_split_' + str(split_id) + '_mb.mrc'
                    ps.disperse_io.save_numpy(svol, out_svol)
                    ps.disperse_io.save_numpy(svol_seg, out_seg)
                    del svol_seg
                    del svol_dst
                    split_id += 1
                    row_dic = dict()
                    row_dic['_rlnMicrographName'] = in_ref
                    row_dic['_rlnImageName'] = out_svol
                    row_dic['_psSegImage'] = out_seg
                    row_dic['_psSegRot'] = 0
                    row_dic['_psSegTilt'] = 0
                    row_dic['_psSegPsi'] = 0
                    row_dic['_psSegOffX'] = off_x[0]
                    row_dic['_psSegOffY'] = off_y[0]
                    row_dic['_psSegOffZ'] = off_z[0]
                    star.add_row(**row_dic)

    # Prepare next iteration
    gc.collect()
    tomo_id += 1

out_star = out_dir + '/' + out_stem + '_pre.star'
print('\tStoring output STAR file in: ' + out_star)
star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')