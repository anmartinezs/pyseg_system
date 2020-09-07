"""

    Deterministic classification of particles in a STAR file (Heuristic approximation for large dataset)

    Input:  - The STAR file with the particles to classify
            - Classification parameters

    Output: - A set of STAR files with the new classes

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

################# Package import

import os
import sys
import time
import glob, shutil
import pyseg as ps

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-plitzko/Matthias/Tomography/4Antonio/relion_big'

# Input STAR file to classify
in_star = ROOT_PATH + '/nali/run2_c1_it004_data.star' # '/nali/run1_data.star' #  # '/ref_7/ref_7_20_50_12.star' # '/run1_c10_data.star' # '/run1_c8_data.star' #
# Root directory for particles in the STAR file
in_root_dir = ROOT_PATH
in_ref_dir = None # '/media/martinez/DATAPART1/syn/in'

# Output directory for the star files
out_dir = ROOT_PATH + '/klass_msafe' # '/det_class'  #
out_stem = 'klass_test_shift_neg_dir'

# Particles pre-processing
pp_mask = ROOT_PATH + '/masks/mask_60_15_45_30.mrc'
pp_low_sg = 3 # voxels
pp_rln_norm = False
pp_2d_norm = True
pp_npr = 1 # None # if None then auto
pp_3d = False
pp_direct = True
pp_n_sset = None #

# Affinity Propagation clustering parameters
ap_damp = 0.9
ap_pref = None # -6 # -10 # -3 # -10.
ap_max_iter = 2000
ap_conv_iter = 40
ap_ref = 'exemplar' # 'average'
ap_ref_per = 33 # %

# Heuristic settings
ss_set = 30
ss_iter = 3

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Deterministic classification of a STAR file (Heuristic approximation for large datase).')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
in_ext = os.path.splitext(os.path.split(in_star)[1])[1]
if in_ext == '.pkl':
    print('\tInput pickle file: ' + str(in_star))
elif in_ext == '.star':
    print('\tInput STAR file: ' + str(in_star))
    if in_root_dir is not None:
        print('\t\t-Root directory for particles: ' + str(in_root_dir))
    if in_ref_dir is not None:
        print('\t\t-Root directory for reference tomograms: ' + str(in_ref_dir))
    print('\tOutput directory: ' + str(out_dir))
    print('\tOutput stem: ' + str(out_stem))
    print('\tParticles pre-processing:')
    print('\t\t-Mask: ' + str(pp_mask))
    print('\t\t-Low pass Gaussian filter sigma: ' + str(pp_low_sg) + ' voxels')
    if pp_rln_norm:
        print('\t\t-Normalize particles according relion convention.')
    if pp_2d_norm:
        print('\t\t-Renormalize particles after the radial averaging.')
    if pp_3d:
        print('\t\t-Radial compensation for 3D.')
    if pp_npr is None:
        print('\t\t-Number of processes: Auto')
    else:
        print('\t\t-Number of processes: ' + str(pp_npr))
    if pp_direct:
        print('\t\t-Direct particles loading activated.')
    if pp_n_sset:
        print('\t\t-Taking a random subset of: ' + str(pp_n_sset) + ' particles')
else:
    print('ERROR: unrecognized extension for the input file, valid: .star, .pkl')
    print('Terminated. (' + time.strftime("%c") + ')')
print('\t\tAffinity Propagation classification settings: ')
print('\t\t-Damping: ' + str(ap_damp))
if ap_pref is not None:
    print('\t\t-Affinity propagation preference: ' + str(ap_pref))
print('\t\t-Maximum number of iterations: ' + str(ap_max_iter))
print('\t\t-Iterations for convergence: ' + str(ap_conv_iter))
print('\t\t-Reference for statistics: ' + str(ap_ref))
print('\t\t-Percentile for statistics: ' + str(ap_ref_per) + ' %')
print('\t\tHeuristic settings: ')
print('\t\t\t-Subset size: ' + str(ss_set))
print('\t\t\t-Subset iterations: ' + str(ss_iter))
print('')

######### Process

print('Main Routine: ')

out_stem_dir = out_dir + '/' + out_stem + '*'
print('\tCleaning output stem directory: ' + str(out_stem_dir))
for token in glob.glob(out_stem_dir):
    if os.path.isdir(token):
        shutil.rmtree(token)
    else:
        os.remove(token)

if in_ext == '.pkl':
    print('\tLoading input pickle file...')
    star_class = ps.factory.unpickle_obj(in_star)
else:
    print('\tLoading STAR file...')
    star = ps.sub.Star()
    try:
        star.load(in_star)
        if pp_n_sset:
            print('\t\tCurrent STAR file has ' + str(star.get_nrows()) + ' particles')
            print('\t\tGetting a random subset of ' + str(pp_n_sset) + ' particles')
            star = star.gen_random_subset(pp_n_sset)
        star_class = ps.sub.ClassStar(star)
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
    if in_root_dir is not None:
        star.set_root_dir(in_root_dir)
    print('\tLoading input mask..')
    mask = ps.disperse_io.load_tomo(pp_mask)

print('\tClassification...')
try:
    dir_imgs = out_dir + '/' + out_stem + '_all_particles'
    if not os.path.exists(dir_imgs):
        os.makedirs(dir_imgs)
    star_class.affinity_propagation_memsafe(mask, sset_size=ss_set, sset_iter=ss_iter,
                                            damping=ap_damp, preference=ap_pref, max_iter=ap_max_iter,
                                            convergence_iter=ap_conv_iter, metric='cc',
                                            low_sg=pp_low_sg, rln_norm=pp_rln_norm, avg_norm=pp_2d_norm, rad_3D=pp_3d,
                                            direct_rec=pp_direct, ref_dir=in_ref_dir, imgs_dir=dir_imgs,
                                            npr=pp_npr, verbose=True)
    star_class.compute_ccap_stat_dir(mask, dir_imgs, reference=ap_ref)
    star_class.print_ccap_stat(percentile=ap_ref_per)
except ps.pexceptions.PySegInputError as e:
    print('ERROR: Classification failed because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
star_class.update_relion_classes()

out_pkl = out_dir + '/' + out_stem + '_class_star.pkl'
print('\tPickling classification object in: ' + out_pkl)
star_class.pickle(out_pkl)

print('\tStoring the results...')
try:
    star_class.save_star(out_dir, out_stem, parse_rln=True, mode='gather')
    star_class.save_star(out_dir, out_stem, parse_rln=True, mode='split')
    star_class.save_class_dir(dir_imgs, out_dir, out_stem, mode='classes')
    star_class.save_class_dir(dir_imgs, out_dir, out_stem, mode='exemplars')
    star_class.save_class_dir(dir_imgs, out_dir, out_stem, mode='averages')
except ps.pexceptions.PySegInputError as e:
    print('ERROR: Result could not be stored because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('Terminated. (' + time.strftime("%c") + ')')