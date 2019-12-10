"""

    Pre-process two inputs of segmented membranes to generate pairs.
    Each pair of membranes consisted in two membranes, labeled as 1 and 2, lummen or gap space, 
    label 3, and citoplasmic sides, one on each membrane labeled as 4 and 5 respectively

    Input:  - Reference tomogram
            - Segmentation tomogram for membranes labeled as 1
            - Segmentation tomogram for membranes labeled as 2

    Output: - A set of subvolumes with the original density map, segmentation and mask for PySeg
            - A STAR file with the list membrane pairs sub-volumes and segmentations

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

LBL_MB1, LBL_MB2, LBL_CT1, LBL_CT2, LBL_LUM = 1, 2, 3, 4, 5

################# Package import

import gc
import os
import sys
import time
import pyto
import math
import pyseg as ps
import scipy as sp
import numpy as np

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/home/martinez/pool/pub/4Antonio/spacer'

# Input reference tomogram
in_ref = ROOT_PATH + '/in/DW_tomo9.rec'

# Input segmentations
in_seg_1 = ROOT_PATH + '/in/tomo9_odd_spacer_membranes.rec'
in_seg_2 = ROOT_PATH + '/in/tomo9_even_spacer_membranes.rec'

# Output directory for the resutls
out_dir = ROOT_PATH + '/seg'
out_stem = 'test'

# Segmentation parameters
sg_res = 1.048 # nm/voxel
sg_mbp_rg = [13, 25] # nm

# Cropping parameters
cp_dst = 30 # nm
cp_mb_thick = 3 # nm
cp_mode_2d = True
cp_border = True

########################################################################################
# AUXILIARY FUNCTIONALITY
########################################################################################

# From a membrane segmented, find the closed labeled membrane in another tomogram
# Closest membrane criterion: smallest median closes distance
# seg_mb: segmented (binary) reference membrane
# lbl_mb: labelled membrane
# lbls: possible labels in lbl_mb
# dsts_rg: distances valid range
def find_lbl_pair(seg_mb, lbl_mb, lbls, dsts_rg):

    # Initialization
    dst_mb = sp.ndimage.morphology.distance_transform_edt(seg_mb==False)
    min_dst, min_lbl = np.finfo('float32').max, None

    for i, lbl in enumerate(lbls):
        mask = lbl_mb == lbl
        if mask.max():
            hold_dst = np.median(dst_mb[mask])
            if (hold_dst > dsts_rg[0]) and (hold_dst < dsts_rg[1]) and (hold_dst < min_dst):
                min_dst = hold_dst
                min_lbl = lbl

    return min_lbl, min_dst

# From two membrane segmentaions labels a pair (1-mb1, 2-mb2, 3-lumen, 4-cyto1 and 5-cyto2)
# seg_mb1|2: membrane segmentation 1 and 2
# res: pixel size in nm/pixel
# mb_thick: membrane thickness in nm
# max_dst: maximum distance in nm
# mode_2d: if False (Default) analysis in 3D, otherwise 2D
# border: if True (default False) then border segmentation is suppressed
# Returns: two tomograms: volumen labelled and masked in DisPerSe mode
def label_mb_pair(seg_mb1, seg_mb2, res, max_dst, mode_2d=False, border=False):

    # Computing signed distances
    tomo_s1, ref_pt = ps.globals.signed_distance_2d(seg_mb1, 1, res, del_b=border, mode_2d=mode_2d, get_point=True)
    tomo_s2 = ps.globals.signed_distance_2d(seg_mb2, 1, res, del_b=border, mode_2d=mode_2d, set_point=ref_pt)
    mask_s1, mask_s2 = (tomo_s1!=0) & (np.abs(tomo_s1)<=max_dst), (tomo_s2!=0) & (np.abs(tomo_s2)<=max_dst)
    tomo_ss1, tomo_ss2 = np.sign(tomo_s1), np.sign(tomo_s2)
    
    # Labelling the regions
    tomo_s = np.zeros(shape=tomo_s1.shape, dtype=np.int16)
    tomo_neq = (tomo_ss1!=tomo_ss2) & (tomo_ss1!=0) & (tomo_ss2!=0)
    sgn_az = (-1) * tomo_ss2[tomo_neq].max()
    tomo_eq = np.invert(tomo_neq)
    tomo_s[mask_s1 & tomo_eq] = LBL_CT1
    tomo_s[mask_s2 & tomo_eq & (tomo_ss2==sgn_az)] = LBL_CT2
    tomo_s[mask_s1 & mask_s2 & tomo_neq] = LBL_LUM
    del tomo_eq, tomo_ss1, tomo_ss2
    tomo_s[seg_mb1] = LBL_MB1
    tomo_s[seg_mb2] = LBL_MB2
    # ps.disperse_io.save_numpy(tomo_s, out_dir+'/hold_1.mrc')
    # ps.disperse_io.save_numpy(mask, out_dir+'/hold_2.mrc')
    del tomo_s1, tomo_s2, mask_s1, mask_s2
    mask = tomo_s == 0

    return tomo_s, mask

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Pre-processing membrane pairs segmentation.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput reference tomogram: ' + str(in_ref)
print '\tInput segmentation tomogram 1: ' + str(in_seg_1)
print '\tInput segmentation tomogram 2: ' + str(in_seg_2)
print '\tOutput directory: ' + str(out_dir)
print '\tOutput stem: ' + str(out_stem)
print '\tSegmentation parameters: '
print '\t\t-Tomograms resolution: ' + str(sg_res) + ' nm/voxel'
print '\t\t-Intermembrane valid range of distances: ' + str(sg_mbp_rg)
print '\tCropping parameters: '
print '\t\t-Membrane thickness: ' + str(cp_mb_thick) + ' nm'
if cp_mode_2d:
    print '\t\t\t+Mode 2D for segmentation activated.'
print '\t\t-Intermembranes gap maximum length: ' + str(cp_dst) + ' nm'
if cp_border:
    print '\t\t-Do not segment borders.'
print ''

######### Process

print 'Main Routine: '

sg_mbp_rg_v = [sg_mbp_rg[0] / float(sg_res), sg_mbp_rg[1] / float(sg_res)]
cp_dst_v, cp_mb_thick_v = math.ceil(float(cp_dst)/float(sg_res)), math.ceil(float(cp_mb_thick)/float(sg_res))

print '\tLoading input tomograms...'
tomo_ref = ps.disperse_io.load_tomo(in_ref, mmap=True)
star = ps.sub.Star()
star.add_column(key='_rlnMicrographName')
star.add_column(key='_psSegImage')
star.add_column(key='_psSegRot')
star.add_column(key='_psSegTilt')
star.add_column(key='_psSegPsi')
star.add_column(key='_psSegOffX')
star.add_column(key='_psSegOffY')
star.add_column(key='_psSegOffZ')

print '\tFinding membranes in input segmentation 1:'
seg_1, seg_2 = ps.disperse_io.load_tomo(in_seg_1) > 0, ps.disperse_io.load_tomo(in_seg_2) > 0
labelled_1, num_lbls_1 = sp.ndimage.measurements.label(seg_1)
labelled_2, num_lbls_2 = sp.ndimage.measurements.label(seg_2)
mb_lbls_1, mb_lbls_2 = np.arange(1, num_lbls_1), np.arange(1, num_lbls_2)
print '\t\t-Number of membranes found: ' + str(num_lbls_1)

print '\tVesicles loop: '
for mb_lbl in mb_lbls_1:

    print '\t\t-Processing membrane with label: ' + str(mb_lbl)

    print '\t\t\t+Cropping membrane subvolume...'
    cp_ids = np.where(labelled_1 == mb_lbl)
    x_min, x_max = cp_ids[0].min() - 2*cp_dst_v, cp_ids[0].max() + 2*cp_dst_v
    y_min, y_max = cp_ids[1].min() - 2*cp_dst_v, cp_ids[1].max() + 2*cp_dst_v
    z_min, z_max = cp_ids[2].min() - 2*cp_dst_v, cp_ids[2].max() + 2*cp_dst_v
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
    sub_mb1 = labelled_1[x_min:x_max, y_min:y_max, z_min:z_max] == mb_lbl
    sub_lbl_mb2 = labelled_2[x_min:x_max, y_min:y_max, z_min:z_max]

    print '\t\t\t+Finding pair membrane label: '
    mb_lbl_pair, dst_pair = find_lbl_pair(sub_mb1, sub_lbl_mb2, mb_lbls_2, sg_mbp_rg_v)
    if mb_lbl_pair is None:
        print 'WARING: No pair label found!'
        continue
    print '\t\t\t\t*Label found ' + str(mb_lbl_pair) + ' at distance ' + str(dst_pair/sg_res) + ' nm'
    sub_mb2 = sub_lbl_mb2 == mb_lbl_pair

    if cp_mb_thick > 0:
        print '\t\t\t+Growing membranes by factor: ' + str(cp_mb_thick) + ' nm'
        sub_mb1 = ps.disperse_io.seg_dist_trans(sub_mb1) <= cp_mb_thick_v
        sub_mb2 = ps.disperse_io.seg_dist_trans(sub_mb2) <= cp_mb_thick_v

    print '\t\t\t+Generating membranes pair segmentation (1-mb_1, 2-mb_2, 3-lummen, 4-cyto_1, 5-cyto_2)...'
    try:
        sub_pair, sub_mask = label_mb_pair(sub_mb1, sub_mb2, sg_res, cp_dst, mode_2d=cp_mode_2d, border=cp_border)
    except ValueError:
        print 'WARING: Segmentation failed for membrane labelled as: ' + str(mb_lbl)
        continue
    sub_ref = tomo_ref[x_min:x_max, y_min:y_max, z_min:z_max].astype(np.float32)

    print '\t\t-Storing sub-volumes...'
    mbp_fname = out_dir+'/'+out_stem+'_mbp_'+str(mb_lbl)+'.mrc'
    mask_fname = out_dir+'/'+out_stem+'_mbp_'+str(mb_lbl)+'.fits'
    mbp_seg_fname = out_dir+'/'+out_stem+'_mbp_'+str(mb_lbl)+'_seg.mrc'
    ps.disperse_io.save_numpy(sub_ref, mbp_fname)
    ps.disperse_io.save_numpy(sub_mask.astype(np.float32), mask_fname)
    ps.disperse_io.save_numpy(sub_pair, mbp_seg_fname)
    # ps.disperse_io.save_numpy((tomo_seg_ves==1)*tomo_ref_cp, out_dir+'/hold.mrc')
    kwargs = {'_rlnMicrographName':in_ref, '_psSegImage':mbp_seg_fname, '_psSegRot':0., '_psSegTilt':0., '_psSegPsi':0.,
              '_psSegOffX':x_min, '_psSegOffY':y_min, '_psSegOffZ':z_min}
    star.add_row(**kwargs)
    gc.collect()

out_star = out_dir + '/' + out_stem + '.star'
print '\tStoring output STAR file in: ' + out_star
star.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'