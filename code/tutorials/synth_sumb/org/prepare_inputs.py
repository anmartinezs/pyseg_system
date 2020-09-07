"""

    Prepare the input for PyOrg analysis package (Tutorial for unsigned membranes)

    Input:  - STAR files for the particles
            - Surface models
            - Surface models for simulations
            - models resolution
            - Segmentations:
                + Segmentation paths
                + Segmentation suffixes
            - Segmentation resolution

    Output: - The modified STAR files for particles with the VOIs segmentations entries
            - The VOIs segmentations STAR file
            - STAR file for ltomos_generator.py input

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

# Input files
in_parts = (ROOT_PATH + '/data/tutorials/synth_sumb/rln/4uqj/ref2/run1_data.star',
            ROOT_PATH + '/data/tutorials/synth_sumb/rln/4pe5/ref2/run1_data.star',
            ROOT_PATH + '/data/tutorials/synth_sumb/rln/5gjv/ref2/run1_data.star',
            ROOT_PATH + '/data/tutorials/synth_sumb/rln/5kxi/ref2/run1_data.star',
            ROOT_PATH + '/data/tutorials/synth_sumb/rln/small_blob/ref/run2_post_data.star')
in_models = (ROOT_PATH + '/data/tutorials/synth_sumb/models/4uqj_fit_s3_cent_dec0.6_bin4.vtp',
             ROOT_PATH + '/data/tutorials/synth_sumb/models/4pe5_fit_s3_cent_dec0.6_bin4.vtp',
             ROOT_PATH + '/data/tutorials/synth_sumb/models/5gjv_fit_s3_cent_dec0.6_bin4.vtp',
             ROOT_PATH + '/data/tutorials/synth_sumb/models/5kxi_fit_s3_cent_dec0.6_bin4.vtp',
             ROOT_PATH + '/data/tutorials/synth_sumb/models/5vai_fit_s3_cent_dec0.6_bin4.vtp')
in_models_sim = (ROOT_PATH + '/data/tutorials/synth_sumb/models/sph_r15_bin4.vtp',
                 ROOT_PATH + '/data/tutorials/synth_sumb/models/sph_r15_bin4.vtp',
                 ROOT_PATH + '/data/tutorials/synth_sumb/models/sph_r15_bin4.vtp',
                 ROOT_PATH + '/data/tutorials/synth_sumb/models/sph_r15_bin4.vtp',
                 ROOT_PATH + '/data/tutorials/synth_sumb/models/sph_r15_bin4.vtp')

# Output directory
out_dir = ROOT_PATH + '/data/tutorials/synth_sumb/org/in'
out_stem = 'test_1'

# Segmentation
sg_md_res = 0.262 # nm/px
sg_res = 1.048 # nm/px
sg_dir = ROOT_PATH + '/data/tutorials/synth_sumb/segs'
sg_suf = '_bin_2_mbu'

########################################################################################
# ADDITIONAL ROUTINES AND STRUCTURES
########################################################################################


########################################################################################
# MAIN ROUTINE
########################################################################################

########## Printing the initial message

print('Prepare inputs for PyOrg.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput particles STAR files: ' + str(in_parts))
print('\tInput models: ' + str(in_models))
print('\tInput models for simulations: ' + str(in_models_sim))
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tSegmentation: ')
print('\t\t-Subvolume resolution: ' + str(sg_md_res) + ' nm/px')
print('\t\t-Segmentation resolution: ' + str(sg_res) + ' nm/px')
print('\t\t-Segmentations directory: ' + str(sg_dir))
print('\t\t-Segmentation suffix: ' + str(sg_suf))
print('')

########### Input parsing

######### Process

count = 0
mics, segs, mods, mod_sims, out_parts = list(), list(), list(), list(), list()
sg_bin = sg_res / sg_md_res

print('\tMain loop:')
for in_part, in_model, in_model_sim in zip(in_parts, in_models, in_models_sim):

    print('\t\t-Processing the STAR file: ' + str(in_part))
    star = sub.Star()
    star.load(in_part)

    for row in range(star.get_nrows()):
        x = star.get_element('_rlnCoordinateX', row) / sg_bin
        y = star.get_element('_rlnCoordinateY', row) / sg_bin
        z = star.get_element('_rlnCoordinateZ', row) / sg_bin
        star.set_element(key='_rlnCoordinateX', row=row, val=x)
        star.set_element(key='_rlnCoordinateY', row=row, val=y)
        star.set_element(key='_rlnCoordinateZ', row=row, val=z)
        mic = star.get_element('_rlnMicrographName', row)
        if not(mic in mics):
            seg = sg_dir + '/' + os.path.splitext(os.path.split(mic)[1])[0] + sg_suf + '.mrc'
            mics.append(mic)
            segs.append(seg)
            mods.append(in_model)
            mod_sims.append(in_models_sim)

    out_part = out_dir + '/' + str(count) + '_' + os.path.splitext(os.path.split(in_model)[1])[0].split('_')[0] + '_' \
               + os.path.splitext(os.path.split(in_part)[1])[0] + '_parts.star'
    print('\t\t+Storing the processed STAR file in: ' + str(out_part))
    star.store(out_part)
    out_parts.append(out_part)
    count += 1

out_seg_star = out_dir + '/' + out_stem + '_seg.star'
print('\tStoring STAR segmentation file in: ' + out_seg_star)
star_seg_out = sub.Star()
star_seg_out.add_column('_rlnMicrographName')
star_seg_out.add_column('_psSegImage')
star_seg_out.add_column('_psSegOffX')
star_seg_out.add_column('_psSegOffY')
star_seg_out.add_column('_psSegOffZ')
star_seg_out.add_column('_psSegRot')
star_seg_out.add_column('_psSegTilt')
star_seg_out.add_column('_psSegPsi')
for mic, seg in zip(mics, segs):
    part_row = {'_rlnMicrographName': mic,
                '_psSegImage': seg,
                '_psSegOffX': 0,
                '_psSegOffY': 0,
                '_psSegOffZ': 0,
                '_psSegRot': 0,
                '_psSegTilt': 0,
                '_psSegPsi': 0}
    star_seg_out.add_row(**part_row)
star_seg_out.store(out_seg_star)

out_star = out_dir + '/' + out_stem + '_ltomos.star'
print('\tStoring output STAR file in: ' + out_star)
star_out = sub.Star()
star_out.add_column('_psStarFile')
star_out.add_column('_suSurfaceVtp')
star_out.add_column('_suSurfaceVtpSim')
star_out.add_column('_psPixelSize')
star_out.add_column('_psPixelSizeSvol')
for out_part, out_model, out_model_sim in zip(out_parts, in_models, in_models_sim):
    part_row = {'_psStarFile': out_part,
                '_suSurfaceVtp': out_model,
                '_suSurfaceVtpSim': out_model_sim,
                '_psPixelSize': sg_res,
                '_psPixelSizeSvol':sg_md_res}
    star_out.add_row(**part_row)
star_out.store(out_star)

print('Successfully terminated. (' + time.strftime("%c") + ')')
