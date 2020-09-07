"""

    Script for cleaning a particle list an generating the required files for rendering it in 3D with ParaView

    Input:  - XML file with a particle list
            - Pre-processing parameters for generating the particle list
            - Membrane segmentation (optional)

    Output: - Filtered particle list
            - VTP file with the distribution and orientation of the particles
            - Tomographic reconstruction if particles Templates are provided

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import pyseg as ps

MB_SEG_LBL = 'mb_seg'
MB_EU_LEN = 'mb_eu_dst'
MB_GEO_LEN = 'mb_geo_len'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1/ex/syn'

# Particles list
in_plist_xml = ROOT_PATH + '/slices/tm_test/pst/ampar_cc_ptc/ampar_cc_plst_hold.xml'
# in_plist_xml = '/home/martinez/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/em/sz52/pst_s1.5/em_out/ParticleList-1.xml'

# Reference tomogram
in_ref = ROOT_PATH+'/../../in/zd/bin2/syn_14_9_bin2.mrc'

###### Transformation data (from particle list to holding tomogram)

# Rotation angles
in_rot_angs = (0, -2, 0)
# in_rot_angs = (0, -67, 0)

# Offsets
in_off = (447,911,159)
# in_off = (767,847,87)

# Binning
in_bin = 1.

# Normal vector
in_norm = (0, 0, 1)

####### Sub-volumes

sv_shape = (52, 52, 52)

####### Membrane miss-alignment

an_th = 20 # Degrees
in_mb_seg = ROOT_PATH + '/../../in/zd/bin2/syn_14_9_bin2_crop2_pst_seg.fits'
in_mb_lbl = 1
in_mask = '/home/martinez/pool/pool-lucic2/antonio/template_matching/ampar/templates/eman/3kg2_40_6.84_mask_sph.mrc'

####### Output data

stem = 'ampar'
output_dir = ROOT_PATH + '/slices/tm_test/pst/ampar_cc_ptc'

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import sys
import time
import errno
import copy
import operator
import numpy as np
from pyseg.sub import ParticleList

########## Global variables

########## Print initial message

print('Showing particles in a tomogram.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput particles list: ' + in_plist_xml)
print('\tReference tomogram name: ' + str(in_ref))
print('\tOutput directory: ' + output_dir)
print('\tTransformation properties (from reference to holding tomogram):')
print('\t\t-Offset: ' + str(in_off) + ' voxels')
print('\t\t-Rotation angles (phi, psi, the): ' + str(in_rot_angs) + ' degs')
print('\t\t-Binning: ' + str(in_bin))
print('\tSub-volumes properties: ')
print('\t\t-Shape: ' + str(sv_shape))
if in_mb_seg is not None:
    print('\tMembrane Miss-alignment deletion:')
    print('\t\t-Angle threshold: ' + str(an_th) + ' deg')
    print('\t\t-Membrane Segmentation tomogram: ' + in_mb_seg)
    print('\t\t-Memebrane label: ' + str(in_mb_lbl))
    print('\t\t-Mask file: ' + in_mask)
print('')

######### Process

print('Main Routine: ')

print('\tParsing particle list...')
pl_path, pl_fname = os.path.split(in_plist_xml)
plist = ParticleList(pl_path + '/sub')
plist.load(in_plist_xml)
print('\t\t-Number of particles found: ' + str(plist.get_num_particles()))

_, in_ref_name = os.path.split(in_ref)
ref_stem, ref_ext = os.path.splitext(in_ref_name)
print('\tDeleting particles which does not belong to a reference tomogram (' + str(ref_stem) + ')...')
plist.filter_particle_fname(ref_stem, keep=True, cmp='stem')
flt_plist = copy.deepcopy(plist)
print('\t\t-Number of particles found: ' + str(plist.get_num_particles()))
if plist.get_num_particles() <= 0:
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit()

print('\tGenerates peaks container for reference tomogram...')
in_ref_map = ps.disperse_io.load_tomo(in_ref, mmap=True)
in_ref_shape = in_ref_map.shape
swap_xy = False
if (ref_ext == '.mrc') or (ref_ext == '.em'):
    swap_xy = True
tpeaks = plist.gen_TomoPeaks(in_ref_shape, ref_stem, swap_xy=swap_xy)

out_sv = output_dir + '/sub_alg'
print('\tExtracting aligned subvolumes in directory: ' + out_sv)
mask = None
if in_mask is not None:
    mask = ps.disperse_io.load_tomo(in_mask)
nsvs, sv_avg = tpeaks.save_subvolumes(in_ref_map, sv_shape, out_sv, stem=stem, mask=mask,
                                      key_eu='Rotation', key_sf='Shift', swap_xy=swap_xy, av_ref=True)
ps.disperse_io.save_numpy(sv_avg, output_dir + '/' + stem + '_alg_avg.mrc')
print('\t\t-Number of subvolumes stored: ' + str(nsvs))

print('\tRotate reference coordinates...')
center = (in_ref_shape[0]*.5, in_ref_shape[1]*.5, in_ref_shape[2]*.5)
tpeaks.rotate_coords(in_rot_angs[0], in_rot_angs[1], in_rot_angs[2], center=center)

print('\tCrop reference coordinates...')
hold_off = (in_off[1], in_off[0], in_off[2])
tpeaks.peaks_prop_op(ps.sub.PK_COORDS, hold_off, operator.sub)

print('\tInverse rotation of normal...')
tpeaks.add_prop('Normal', 3, vals=in_norm, dtype=np.float32)
tpeaks.rotate_vect(key_vect='Normal', key_eu='Rotation', inv=True)

print('\tUn rotate normals...')
tpeaks.add_prop('Unrotation', 3, vals=in_rot_angs, dtype=np.float32)
tpeaks.rotate_vect(key_vect='Normal', key_eu='Unrotation')

if in_mb_lbl:
    print('\tComputing angle to membranes...')
    tomo_seg = ps.disperse_io.load_tomo(in_mb_seg) == in_mb_lbl
    tpeaks.seg_shortest_normal(tomo_seg, 'Mb_normal')
    tpeaks.vects_angle(key_v1='Normal', key_v2='Mb_normal', key_a='Angle')
    tpeaks.filter_prop_scalar(key_s='Angle', cte=an_th, op=operator.gt)
    print('\tDeleting miss-aligned peaks...')
    print('\t\tNumber of peaks: ' + str(tpeaks.get_num_peaks()))
    if plist.get_num_particles() > 0:
        tp_fnames = tpeaks.get_prop_vals('Filename', dtype=str)
        flt_plist.filter_particle_nolist(tp_fnames)
        tpeaks_msa = flt_plist.gen_TomoPeaks(in_ref_shape, ref_stem, swap_xy=swap_xy)
        out_dir_msa = output_dir + '/' + ref_stem + '_msa'
        try:
            os.makedirs(out_dir_msa)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                print('ERROR: directory ' + out_dir_msa + ' could not be created')
                print('Wrong terminated. (' + time.strftime("%c") + ')')
                sys.exit()
        print('\t\tStoring corrected subvolumes in : ' + out_dir_msa)
        nsvs, sv_avg = tpeaks_msa.save_particles(in_ref_map, sv_shape, out_dir_msa, stem=stem, mask=mask,
                                                 swap_xy=swap_xy, av_ref=True)
        ps.disperse_io.save_numpy(sv_avg, out_dir_msa + '/' + stem + '_alg_avg.mrc')
        pl_stem, _ = os.path.splitext(pl_fname)
        ps.disperse_io.save_vtp(tpeaks_msa.to_vtp(), out_dir_msa + '/' + ref_stem + '_plist.vtp')

print('\tGenerating VTP file...')
poly = tpeaks.to_vtp()

print('\tStoring the results in: ' + output_dir)
ps.disperse_io.save_vtp(poly, output_dir + '/' + ref_stem + '_plist.vtp')

print('Terminated. (' + time.strftime("%c") + ')')