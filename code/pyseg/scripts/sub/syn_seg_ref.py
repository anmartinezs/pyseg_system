"""

    Express a cropped segmentation in the reference system

    Input:  - The STAR file with the info for the cropped segmentation

    Output: - The segmentation in the reference tomogram context
            -  The STAR with the new segmentation info

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
import numpy as np

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/home/martinez/workspace/disperse/data/klass/test_1'

## Input
# STAR file with the segmentation info
in_star = ROOT_PATH + '/two_segmentations.star'

# Output
# Path for the output STAR file
out_star = ROOT_PATH + '/two_segmentations_ref.star'
# Suffix for the output segmented files
out_suffix = 'ref_syn_seg'

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Express segmentations in their reference context.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput'
print '\t\t-Input STAR file: ' + str(in_star)
print '\tOuput:'
print '\tOutput STAR file: ' + str(out_star)
print '\tOutput suffix: ' + str(out_suffix)
print ''

######### Process

print 'Main Routine: '

print '\tLoading STAR file...'
star, star_out = ps.sub.Star(), ps.sub.Star()
try:
    star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
star_out.add_column('_psSegImage')
star_out.add_column('_rlnMicrographName')

print '\tLoop for transforming the segmentations...'
for row in range(star.get_nrows()):

    ## Read segmentation mask
    seg_fname, ref_fname = star.get_element('_psSegImage', row), star.get_element('_rlnMicrographName', row)
    if (seg_fname != '') and (ref_fname != ''):
        try:
            tomo_seg = ps.disperse_io.load_tomo(seg_fname, mmap=False)
            tomo_ref = ps.disperse_io.load_tomo(ref_fname, mmap=False)
            hold_tomo = np.zeros(shape=tomo_ref.shape, dtype=np.int16)
            # Uncropping
            seg_offx, seg_offy, seg_offz = star.get_element('_psSegOffX', row), star.get_element('_psSegOffY', row), \
                                           star.get_element('_psSegOffZ', row)
            seg_offx, seg_offy, seg_offz = int(math.floor(seg_offx)), int(math.floor(seg_offy)), int(math.floor(seg_offz))
            hold_tomo[seg_offx:seg_offx + tomo_seg.shape[0],
                      seg_offy:seg_offy + tomo_seg.shape[1],
                      seg_offz:seg_offz + tomo_seg.shape[2]] = tomo_seg
            # Rotation and scaling
            seg_rot, seg_tilt, seg_psi = star.get_element('_psSegRot', row), star.get_element('_psSegTilt', row), \
                                         star.get_element('_psSegPsi', row)
            angs = np.asarray((seg_rot, seg_tilt, seg_psi), dtype=np.float32)
            seg_scale = star.get_element('_psSegScale', row)
            r3d = pyto.geometry.Rigid3D()
            r3d.q = r3d.make_r_euler(angles=np.radians(angs), mode='zyz_in_active')
            r3d.s_scalar = 1
            ref_cent = np.round(.5 * np.asarray(tomo_ref.shape, dtype=np.float32))
            hold_tomo = r3d.transformArray(hold_tomo, origin=ref_cent, order=1, prefilter=False)
            # Insert an entry in the output STAR file
            out_seg = os.path.split(seg_fname)[0] + '/' + os.path.splitext(os.path.split(seg_fname)[1])[0] + out_suffix + '.mrc'
            ps.disperse_io.save_numpy(hold_tomo, out_seg)
            star_out.set_element('_psSegImage', row, out_seg)
            star_out.set_element('_rlnMicrographName', row, ref_fname)
        except ValueError:
            print 'WARNING : segmentation file ' + str(seg_fname) + ' did not fit the reference tomogram ' + str(ref_fname)
        except IOError:
            print 'WARNING : segmentation file ' + str(seg_fname) + ' or reference tomogram ' + str(ref_fname) + ' could not be opened'

print '\tStoring the output STAR file...'
try:
    star_out.store(out_star)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Result could not be stored because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print 'Terminated. (' + time.strftime("%c") + ')'
