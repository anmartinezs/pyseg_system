"""

    Pre-processing for mb2_graph_batch.py of two opposed membranes from TomoSegMemTV output

    Input:  - STAR file with 3 columns:
                + _rlnMicrographName: tomogram original
                + _rlnImageName: TomoSegMemTV density map output
                + _psSegImage: segmentation for the two membranes: 1-mb1, 2-mb2, 3-gap, 4-lumen1, 5-lumen2
            - Sub-volume splitting settings

    Output: - A STAR file with 3 columns:
                + _rlnMicrographName: tomogram original
                + _rlnImageName: sub-volumes
                + _psSegImage: two membranes segmentations for each subvolume
                + Columns for localizing the sub-volumes within each original tomogram

"""

################# Package import

import gc
import os
import sys
import math
import time
import pyseg as ps
import scipy as sp
import skimage as sk
import numpy as np

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/in_situ_mitoo'

# Input STAR file
in_star = ROOT_PATH + '/mbo_seg/mb_seg_mitoo.star'

# Output directory
out_dir = ROOT_PATH + '/mbo_seg/pre'

# Subvolume splitting settings
sp_split = (2, 2, 1)
sp_off_voxels = 5 # vox


########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Pre-processing for SEG analysis of un-oriented membranes from TomoSegMemTV output.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tInput STAR file: ' + str(in_star))
print('\tSub-volume splitting settings: ')
print('\t\t-Number of splits (X, Y, Z): ' + str(sp_split))
print('\t\t-Offset voxels: ' + str(sp_off_voxels))
print('')

######### Process

print('Parsing input parameters...')
sp_off_voxels = int(sp_off_voxels)
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

print('Main Routine: tomograms loop')
tomo_id = 0
for row in range(gl_star.get_nrows()):

    in_ref = gl_star.get_element('_rlnMicrographName', row)
    print('\tProcessing tomogram: ' + in_ref)
    out_ref_stem = os.path.splitext(os.path.split(in_ref)[1])[0]
    tomo_ref = ps.disperse_io.load_tomo(in_ref, mmap=True)
    off_mask_min_x, off_mask_max_x = 0, tomo_ref.shape[0]
    off_mask_min_y, off_mask_max_y = 0, tomo_ref.shape[1]
    off_mask_min_z, off_mask_max_z = 0, tomo_ref.shape[2]

    if gl_star.has_column('_psSegImage'):
        seg_file = gl_star.get_element('_psSegImage', row)
        print('\t\t-Loading segmantion file: ' + seg_file)
        tomo_seg = ps.disperse_io.load_tomo(seg_file).astype(np.int8)
        ids_mask = np.where(tomo_seg)
        off_mask_min_x, off_mask_max_x = ids_mask[0].min()-sp_off_voxels, ids_mask[0].max()+sp_off_voxels
        if off_mask_min_x < 0:
            off_mask_min_x = 0
        if off_mask_max_x > tomo_seg.shape[0]:
            off_mask_max_x = tomo_seg.shape[0]
        off_mask_min_y, off_mask_max_y = ids_mask[1].min()-sp_off_voxels, ids_mask[1].max()+sp_off_voxels
        if off_mask_min_y < 0:
            off_mask_min_y = 0
        if off_mask_max_y > tomo_seg.shape[1]:
            off_mask_max_y = tomo_seg.shape[1]
        off_mask_min_z, off_mask_max_z = ids_mask[2].min()-sp_off_voxels, ids_mask[2].max()+sp_off_voxels
        if off_mask_min_z < 0:
            off_mask_min_z = 0
        if off_mask_max_z > tomo_seg.shape[2]:
            off_mask_max_z = tomo_seg.shape[2]
    else:
        print('ERROR: input star file ' + str(in_star) + ' does not contain _psSegImage column!')
        sys.exit(-1)

    if sp_split is None:
        print('\t\t-Cropping to fit the segmentation...')
        svol_seg = tomo_seg[off_mask_min_x:off_mask_max_x, off_mask_min_y:off_mask_max_y, off_mask_min_z:off_mask_max_z]
        svol = tomo_ref[off_mask_min_x:off_mask_max_x, off_mask_min_y:off_mask_max_y, off_mask_min_z:off_mask_max_z]
        out_svol = out_seg_dir + '/' + out_ref_stem + '_tid_' + str(tomo_id) + '_split_' + str(split_id) + '.mrc'
        out_seg = out_seg_dir + '/' + out_ref_stem + '_tid_' + str(tomo_id) + '_split_' + str(split_id) + '_seg.mrc'
        ps.disperse_io.save_numpy(svol, out_svol)
        ps.disperse_io.save_numpy(svol_seg, out_seg)
        del svol_seg
        split_id += 1
        row_dic = dict()
        row_dic['_rlnMicrographName'] = in_ref
        row_dic['_rlnImageName'] = out_svol
        row_dic['_psSegImage'] = out_seg
        row_dic['_psSegRot'] = 0
        row_dic['_psSegTilt'] = 0
        row_dic['_psSegPsi'] = 0
        row_dic['_psSegOffX'] = off_mask_min_x
        row_dic['_psSegOffY'] = off_mask_min_y
        row_dic['_psSegOffZ'] = off_mask_min_z
        star.add_row(**row_dic)
    else:
        print('\t\t-Splitting into subvolumes:')
        if sp_split[0] > 1:
            offs_x = list()
            pad_x = int(math.ceil((off_mask_max_x-off_mask_min_x) / sp_split[0]))
            offs_x.append((off_mask_min_x, pad_x+sp_off_voxels))
            lock = False
            while not lock:
                hold = offs_x[-1][1] - sp_off_voxels + pad_x
                if hold >= off_mask_max_x:
                    offs_x.append((offs_x[-1][1] - sp_off_voxels, off_mask_max_x))
                    lock = True
                else:
                    offs_x.append((offs_x[-1][1]-sp_off_voxels, offs_x[-1][1]+pad_x+sp_off_voxels))
        else:
            offs_x = [(off_mask_min_x, off_mask_max_x),]
        if sp_split[1] > 1:
            offs_y = list()
            pad_y = int(math.ceil((off_mask_max_y-off_mask_min_y) / sp_split[1]))
            offs_y.append((off_mask_min_x, pad_y + sp_off_voxels))
            lock = False
            while not lock:
                hold = offs_y[-1][1] - sp_off_voxels + pad_y
                if hold >= off_mask_max_y:
                    offs_y.append((offs_y[-1][1] - sp_off_voxels, off_mask_max_y))
                    lock = True
                else:
                    offs_y.append((offs_y[-1][1] - sp_off_voxels, offs_y[-1][1] + pad_y + sp_off_voxels))
        else:
            offs_y = [(off_mask_min_x, off_mask_max_x),]
        if sp_split[2] > 1:
            offs_z = list()
            pad_z = int(math.ceil((off_mask_max_z-off_mask_min_z) / sp_split[2]))
            offs_z.append((off_mask_min_z, pad_z + sp_off_voxels))
            lock = False
            while not lock:
                hold = offs_z[-1][1] - sp_off_voxels + pad_z
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
                    print('\t\t\t+Splitting subvolume: [' + str(off_x) + ', ' + str(off_y) + ', ' + str(off_z) +']')
                    svol_seg = tomo_seg[off_x[0]:off_x[1], off_y[0]:off_y[1], off_z[0]:off_z[1]]
                    svol = tomo_ref[off_x[0]:off_x[1], off_y[0]:off_y[1], off_z[0]:off_z[1]]
                    out_svol = out_seg_dir + '/' + out_ref_stem + '_id_' + str(tomo_id) + '_split_' + str(split_id) + '.mrc'
                    out_seg = out_seg_dir + '/' + out_ref_stem + '_id_' + str(tomo_id) + '_split_' + str(split_id) + '_mb.mrc'
                    ps.disperse_io.save_numpy(svol, out_svol)
                    ps.disperse_io.save_numpy(svol_seg, out_seg)
                    del svol_seg
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