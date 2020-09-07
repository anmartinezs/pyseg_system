"""

    Script for computing signed distance for membrane segmentations and generating cropped tomograms

    Input:  - Density map tomogram
            - Membrane segmentations

    Output: - Cropped density maps for segmentations
            - Signed distance map
            - Signed segmentation (mb=1, inside=2 and outside=3)
"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import os
import time
import scipy
import pyseg as ps
import numpy as np
try:
    import pickle as pickle
except:
    import pickle

########## Global variables

########## Helping functionality

def compute_roi(tomo_dst, max_dst):

   # Initialization
   mask = (tomo_dst != 0) & (np.abs(tomo_dst) < max_dst)
   # ps.disperse_io.save_numpy(mask, './hold.mrc')

   # Computation for every axis
   roi = np.zeros(shape=6, dtype=np.int16)
   mask_s = mask.sum(axis=2).sum(axis=1)
   roi[0] = np.argmax(mask_s > 0)
   roi[1] = mask.shape[0] - np.argmax(mask_s[::-1]>0)
   mask_s = mask.sum(axis=2).sum(axis=0)
   roi[2] = np.argmax(mask_s > 0)
   roi[3] = mask.shape[1] - np.argmax(mask_s[::-1]>0)
   mask_s = mask.sum(axis=0).sum(axis=0)
   roi[4] = np.argmax(mask_s > 0)
   roi[5] = mask.shape[2] - np.argmax(mask_s[::-1]>0)
      
   return roi

def write_crop_info(output_dir, f_name_s, f_name, roi):
    with open(output_dir+'/'+f_name_s+'_crop.txt', 'w') as text_file:
        text_file.write('Input density: ' + f_name + '\n')
        text_file.write('Cropping dimensions (x, X, y, Y, z, Z): ' + str(roi) + '\n')

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/home/martinez/workspace/disperse/data/chlamy'

# Original density map
in_tomo = ROOT_PATH+'/in/T1L1_b4.mrc'

# Membrane segmentation: 0-bg, otherwise-mb
in_seg_l = (ROOT_PATH+'/in/T1L1_b4_M1.mrc',
	    # ROOT_PATH+'/in/T1L1_b4_M2.mrc',
	    # ROOT_PATH+'/in/T1L1_b4_M3.mrc',
	    # ROOT_PATH+'/in/T1L1_b4_M4.mrc'
            )

# If not '-' change membrane orientations
in_or_l = ('+',
           # '-',
           # '+',
           # '-'
           )

####### Output data

output_dir = ROOT_PATH+'/s'
out_fmt = '.fits'

###### Signed distance computation

mode_2d = True
del_b = True

###### Cropping parameters

res = 1.368 # nm
max_dst = 30 # nm
mb_thick = 5 # nm 

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print('Membrane cropping and signed distance computation.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput density map: ' + in_tomo)
print('\tInput membrane segmentations: ' + str(in_seg_l)) 
print('\tOuput directory: ' + output_dir)
print('\tDistance computation: ')
if mode_2d:
    print('\t\t-Mode 2D active')
else:
    print('\t\t-Mode 3D active')
if del_b:
    print('\t\t-Deleting borders with no reliable distance computation active')
print('\tCropping parameters: ')
print('\t\t-Resolution: ' + str(res) + ' nm/pix')
print('\t\t-Maximum distance: ' + str(max_dst) + ' nm')
print('\t\t-Membrane thickness: ' + str(mb_thick) + 'nm')
print('')

f_path, f_name = os.path.split(in_tomo)
print('Loading density map: ' + f_name)
density = ps.disperse_io.load_tomo(in_tomo) 

# Loop for processing the input data
print('Running main loop: ')
for (in_seg, in_or) in zip(in_seg_l, in_or_l):

    f_path_s, f_name_s = os.path.split(in_seg)
    f_stem_s, f_ext_s = os.path.splitext(f_name_s)
    print('\tLoading membrane segmentation: ' + f_name_s)
    mb_seg = ps.disperse_io.load_tomo(in_seg)

    print('\tComputing signed distance...')
    tomo_s = ps.globals.signed_distance_2d(mb_seg, None, res, del_b=del_b, mode_2d=mode_2d)
    msk = (tomo_s==0) & (mb_seg==0)

    print('\tComputing cropping dimensions...')
    roi = compute_roi(tomo_s, max_dst)
    roi_den = density[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    roi_dst = tomo_s[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    roi_seg = np.zeros(shape=roi_dst.shape, dtype=np.int16)
    roi_msk = msk[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    if in_or == '+':
        roi_seg[roi_dst < 0] = 2
        roi_seg[roi_dst > 0] = 3
    else:
        roi_seg[roi_dst > 0] = 3
        roi_seg[roi_dst < 0] = 2
    mb_thick_2 = mb_thick * .5
    roi_dst_abs = np.abs(roi_dst)
    roi_seg[roi_dst_abs<=mb_thick_2] = 1
    roi_seg[roi_msk] = 4

    print('\tStoring results...')
    if out_fmt == '.fits':
        ps.disperse_io.save_numpy(roi_den.transpose().astype(np.float32), output_dir + '/' + f_stem_s + '_den' + out_fmt)
        ps.disperse_io.save_numpy(roi_dst.transpose().astype(np.float32), output_dir + '/' + f_stem_s + '_dst' + out_fmt)
        ps.disperse_io.save_numpy(roi_seg.transpose().astype(np.float32), output_dir + '/' + f_stem_s + '_seg' + out_fmt)
    else:
        ps.disperse_io.save_numpy(roi_den.astype(np.float32), output_dir + '/' + f_stem_s + '_den' + out_fmt)
        ps.disperse_io.save_numpy(roi_dst.astype(np.float32), output_dir + '/' + f_stem_s + '_dst' + out_fmt)
        ps.disperse_io.save_numpy(roi_seg.astype(np.float32), output_dir + '/' + f_stem_s + '_seg' + out_fmt)
    write_crop_info(output_dir, f_stem_s, f_name, roi)

print('Terminated. (' + time.strftime("%c") + ')')
