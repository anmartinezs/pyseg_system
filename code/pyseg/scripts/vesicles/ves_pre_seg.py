"""

    Pre-process PyTo segmentation of the pre-synaptic vesicle.
    Each vesicle is cropped in a sub-volume, also membrane segmentations for PySeg application are generated

    Input:  - Reference tomogram
            - Segmentation tomogram

    Output: - A set of subvolumes with the original density map, segmentation and mask for PySeg
            - A STAR file with the list vesicles sub-volumes and segmentations

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

################# Package import

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

ROOT_PATH = '/home/martinez/pool/EMpub/4Antonio/fromQiang/spheres'

# Input reference tomogram
in_ref = ROOT_PATH + '/S3-1_T52.rec'

# Input segmentation
in_seg = ROOT_PATH + '/plot_T52.mrc'

# Output directory for the results
out_dir = ROOT_PATH + '/seg' # '/det_class'  #
out_stem = 't52'

# Segmentation parameters
sg_res = 1.048 # nm/voxel
sg_cyto_lbl = 0 # 3
sg_ves_lbls = None # range(9,88)

# Cropping parameters
cp_dst = 30 # nm
cp_mb_thick = 5 # nm

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Pre-processing vesicles segmentation.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput reference tomogram: ' + str(in_ref))
print('\tInput segmentation tomogram: ' + str(in_ref))
print('\tOutput directory: ' + str(out_dir))
print('\tOutput stem: ' + str(out_stem))
print('\tSegmentation parameters: ')
print('\t\t-Tomograms resolution: ' + str(sg_res) + ' nm/voxel')
if sg_cyto_lbl is not None:
    print('\t\t-Cytoplasm labels: ' + str(sg_cyto_lbl))
if sg_ves_lbls is not None:
    print('\t\t-Minimum label for vesicles: ' + str(sg_ves_lbls))
print('\tCropping parameters: ')
print('\t\t-Vesicles membrane thickness: ' + str(cp_mb_thick) + ' nm')
print('\t\t-Vesicle radial halo length: ' + str(cp_dst) + ' nm')
print('')

######### Process

print('Main Routine: ')

print('\tLoading input tomograms...')
tomo_ref = ps.disperse_io.load_tomo(in_ref, mmap=True)
if os.path.splitext(in_seg)[1] == '.dat':
    seg_img = pyto.io.ImageIO()
    seg_img.readRaw(file=in_seg, dataType='uint8', shape=tomo_ref.shape)
    tomo_seg = seg_img.data
else:
    tomo_seg = ps.disperse_io.load_tomo(in_seg)
star = ps.sub.Star()
star.add_column(key='_rlnMicrographName')
star.add_column(key='_psSegImage')
star.add_column(key='_psSegRot')
star.add_column(key='_psSegTilt')
star.add_column(key='_psSegPsi')
star.add_column(key='_psSegOffX')
star.add_column(key='_psSegOffY')
star.add_column(key='_psSegOffZ')

if sg_ves_lbls is None:
    print('\tFinding vesicles labels...')
    tomo_seg, num_lbls_1 = sp.ndimage.measurements.label(tomo_seg)
    print('\t\t-Number of vesicles found: ' + str(num_lbls_1))
    sg_ves_lbls = 1 + np.arange(0, num_lbls_1)

print('\tVesicles loop: ')
for ves_lbl in sg_ves_lbls:

    print('\t\t-Processing vesicle: ' + str(ves_lbl))

    print('\t\t\t+Cropping vesicle subvolume...')
    cp_dst_v, cp_mb_thick_v = math.ceil(float(cp_dst)/float(sg_res)), math.ceil(float(cp_mb_thick)/float(sg_res))
    cp_ids = np.where(tomo_seg == ves_lbl)
    x_min, x_max = cp_ids[0].min()-cp_dst_v, cp_ids[0].max()+cp_dst_v
    y_min, y_max = cp_ids[1].min()-cp_dst_v, cp_ids[1].max()+cp_dst_v
    z_min, z_max = cp_ids[2].min()-cp_dst_v, cp_ids[2].max()+cp_dst_v
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
    sub_seg = tomo_seg[x_min:x_max, y_min:y_max, z_min:z_max]

    print('\t\t\t+Pre-growing the segmentation...')
    sub_dst = sp.ndimage.morphology.distance_transform_edt(sub_seg == ves_lbl)

    print('\t\t\t+Generating vesicle segmentation (1-mb, 2-lumen and 3-pre-cyto)...')
    sub_dst = sp.ndimage.morphology.distance_transform_edt(sub_seg != ves_lbl)
    tomo_seg_ves = np.zeros(shape=sub_dst.shape, dtype=int)
    dst_mask, cyto_mask, ves_mask = sub_dst<=cp_dst_v, sub_seg==sg_cyto_lbl, sub_seg==ves_lbl
    tomo_seg_ves[dst_mask * cyto_mask] = 3
    tomo_seg_ves[ves_mask] = 1
    sub_dst = sp.ndimage.morphology.distance_transform_edt(sub_seg == ves_lbl)
    inn_mask = sub_dst > cp_mb_thick_v
    tomo_seg_ves[inn_mask * ves_mask] = 2
    tomo_ref_cp = tomo_ref[x_min:x_max, y_min:y_max, z_min:z_max]

    print('\t\t-Storing sub-volumes...')
    ves_fname = out_dir+'/'+out_stem+'_ves_'+str(ves_lbl)+'.mrc'
    ves_seg_fname = out_dir+'/'+out_stem+'_ves_'+str(ves_lbl)+'_seg.mrc'
    ps.disperse_io.save_numpy(tomo_ref_cp, ves_fname)
    ps.disperse_io.save_numpy(tomo_seg_ves, ves_seg_fname)
    ps.disperse_io.save_numpy(sub_seg,  out_dir+'/'+out_stem+'_ves_'+str(ves_lbl)+'_seg_orig.mrc')
    # ps.disperse_io.save_numpy((tomo_seg_ves==1)*tomo_ref_cp, out_dir+'/hold.mrc')
    kwargs = {'_rlnMicrographName':in_ref, '_psSegImage':ves_seg_fname, '_psSegRot':0., '_psSegTilt':0., '_psSegPsi':0.,
              '_psSegOffX':x_min, '_psSegOffY':y_min, '_psSegOffZ':z_min}
    star.add_row(**kwargs)

out_star = out_dir + '/' + out_stem + '.star'
print('\tStoring output STAR file in: ' + out_star)
star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')
