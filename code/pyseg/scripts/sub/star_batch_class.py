"""

    Deterministic classification of particles in a STAR file

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
import pyseg as ps
from scipy.misc import imsave

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-plitzko/Matthias/Tomography/4Antonio/relion_big'

# Input STAR file to classify
in_star = ROOT_PATH + '/ref_class_rln_c1/k1/run6_data.star' # '/nali/run1_data.star' #  # '/ref_7/ref_7_20_50_12.star' # '/run1_c10_data.star' # '/run1_c8_data.star' #
# Root directory for particles in the STAR file
in_root_dir = ROOT_PATH
in_ref_dir = None # '/media/martinez/DATAPART1/syn/in'

# Input STAR file with segmentation information to focus the masks
in_seg = None # ROOT_PATH + '/two_segmentations.star'

# Output directory for the star files
out_dir = ROOT_PATH + '/klass_k1' # '/det_class'  #
out_stem = 'klass_k1_test'
out_debug_dir = None # ROOT_PATH + '/test_whole/debug' # '/det_class/debug'  #

# Particles pre-processing
pp_mask = ROOT_PATH + '/masks/mask_60_15_45_15.mrc'
pp_low_sg = 2 # voxels
pp_rln_norm = False
pp_2d_norm = True
pp_npr = None # None # if None then auto
pp_3d = False
pp_direct = True
pp_n_sset = None # 3000

# CC 2d radial matrix computation parameters
cc_metric = 'cc' # 'cc' or 'similarity'
cc_npy = None # ROOT_PATH + '/test_whole/test_1_cc.npy'
cc_npr = None # None # 1 # None # if None then auto

# Image moments
mo_mode = 'raw' # 'spatial', 'central', 'normalized', 'raw'
mo_pca_nfeat = 3 # None
mo_npr = None # None # 1 # bNone # if None then auto

## Clustering
cu_alg = 'AP' # 'AP', 'AG'
cu_mode = 'ncc_2dz' # 'moments' # 'ncc_2dz'

# Affinity Propagation clustering parameters
ap_damp = 0.9
ap_pref = -10 # -6 # -10 # -3 # -10.
ap_max_iter = 2000
ap_conv_iter = 40
ap_ref = 'exemplar' # 'average'
ap_ref_per = 33 # %

# Agglomerative Clustering
ag_n_clusters = 12
ag_linkage = 'ward'

# Classification post processing
cp_min_cz = None # 16
cp_min_ccap = None # 0.6 # 0.6 #

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Deterministic classification of a STAR file.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
in_ext = os.path.splitext(os.path.split(in_star)[1])[1]
if in_ext == '.pkl':
    print '\tInput pickle file: ' + str(in_star)
elif in_ext == '.star':
    print '\tInput STAR file: ' + str(in_star)
    if in_root_dir is not None:
        print '\t\t-Root directory for particles: ' + str(in_root_dir)
    if in_ref_dir is not None:
        print '\t\t-Root directory for reference tomograms: ' + str(in_ref_dir)
    if in_seg is not None:
        print '\tInput segmentation STAR file: ' + str(in_seg)
    print '\tOutput directory: ' + str(out_dir)
    print '\tOutput stem: ' + str(out_stem)
    if out_debug_dir is not None:
        print '\tDebugging directory: ' + str(out_debug_dir)
    if cc_npy is not None:
        print '\tCC matrix already computed in file: ' + cc_npy
    else:
        print '\tParticles pre-processing:'
        print '\t\t-Mask: ' + str(pp_mask)
        print '\t\t-Low pass Gaussian filter sigma: ' + str(pp_low_sg) + ' voxels'
        if pp_rln_norm:
            print '\t\t-Normalize particles according relion convention.'
        if pp_2d_norm:
            print '\t\t-Renormalize particles after the radial averaging.'
        if pp_3d:
            print '\t\t-Radial compensation for 3D.'
        if pp_npr is None:
            print '\t\t-Number of processes: Auto'
        else:
            print '\t\t-Number of processes: ' + str(pp_npr)
        if pp_direct:
            print '\t\t-Direct particles loading activated.'
        if pp_n_sset:
            print '\t\t-Taking a random subset of: ' + str(pp_n_sset) + ' particles'
        print '\tCC Z-axis radially averages matrix parameters: '
        print '\t\t-Metric: ' + str(cc_metric)
        if cc_npr is None:
            print '\t\t-Number of processes: Auto'
        else:
            print '\t\t-Number of processes: ' + str(cc_npr)
        print '\tImage moments: '
        print '\t\t-Mode : ' + str(mo_mode)
        if mo_pca_nfeat is not None:
            print '\t\t-PCA dimensionality reduction to: ' + str(mo_pca_nfeat)
        if mo_npr is None:
            print '\t\t-Number of processes: Auto'
        else:
            print '\t\t-Number of processes: ' + str(mo_npr)
else:
    print 'ERROR: unrecognized extension for the input file, valid: .star, .pkl'
    print 'Terminated. (' + time.strftime("%c") + ')'
print '\tClustering: '
print '\t\t-Mode: ' + str(cu_mode)
if cu_alg == 'AP':
    print '\t\tAffinity Propagation classification settings: '
    print '\t\t\t-Damping: ' + str(ap_damp)
    if ap_pref is not None:
        print '\t\t\t-Affinity propagation preference: ' + str(ap_pref)
    print '\t\t\t-Maximum number of iterations: ' + str(ap_max_iter)
    print '\t\t\t-Iterations for convergence: ' + str(ap_conv_iter)
    print '\t\t\t-Reference for statistics: ' + str(ap_ref)
    print '\t\t\t-Percentile for statistics: ' + str(ap_ref_per) + ' %'
elif cu_alg == 'AG':
    print '\t\tAgglomerative clustering classificiation settings: '
    print '\t\t\t-Number of clusters to find: ' + str(ag_n_clusters)
    print '\t\t\t-Linkage: ' + str(ag_linkage)
else:
    print 'ERROR: invalid input mode for classification, valid: AP or AG'
    print 'Terminated. (' + time.strftime("%c") + ')'
print '\tClassification post-processing: '
if cp_min_ccap is not None:
    print '\t\t-Purge purge particles with CCAP against reference lower than: ' + str(cp_min_ccap)
if cp_min_cz is not None:
    print '\t\t-Purge classes with less than ' + str(cp_min_cz) + ' particles'
print ''

######### Process

print 'Main Routine: '

if in_ext == '.pkl':
    print '\tLoading input pickle file...'
    star_class = ps.factory.unpickle_obj(in_star)
else:
    print '\tLoading STAR file...'
    star = ps.sub.Star()
    try:
        star.load(in_star)
        if pp_n_sset:
            print '\t\tCurrent STAR file has ' + str(star.get_nrows()) + ' particles'
            print '\t\tGetting a random subset of ' + str(pp_n_sset) + ' particles'
            star = star.gen_random_subset(pp_n_sset)
        star_class = ps.sub.ClassStar(star)
    except ps.pexceptions.PySegInputError as e:
        print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    if in_root_dir is not None:
        star.set_root_dir(in_root_dir)

    if in_seg is not None:
        seg_star = ps.sub.Star()
        try:
            seg_star.load(in_seg)
            star_class.add_segmentation(seg_star)
        except ps.pexceptions.PySegInputError as e:
            print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
            print 'Terminated. (' + time.strftime("%c") + ')'
            sys.exit(-1)

    if cc_npy is not None:
        print '\tLoading NCC matrix from: ' + cc_npy
        star_class.load_cc(cc_npy)
    else:
        print '\tLoading and pre-processing the particles...'
        try:
            mask = ps.disperse_io.load_tomo(pp_mask)
            star_class.load_particles(mask, low_sg=pp_low_sg, avg_norm=pp_2d_norm, rln_norm=pp_rln_norm, rad_3D=pp_3d,
                                      npr=pp_npr, debug_dir=out_debug_dir, ref_dir=in_ref_dir, direct_rec=pp_direct)
            star_class.save_particles(out_dir+'/all_particles', out_stem, masks=True, stack=True)
            imsave(out_dir+'/all_particles/global_mask.png', star_class.get_global_mask())
        except ps.pexceptions.PySegInputError as e:
            print 'ERROR: Particles could not be loaded because of "' + e.get_message() + '"'
            print 'Terminated. (' + time.strftime("%c") + ')'
            sys.exit(-1)

        print '\tBuilding the NCC matrix...'
        try:
            star_class.build_ncc_z2d(metric=cc_metric, npr=cc_npr)
            star_class.save_cc(out_dir + '/' + out_stem + '_cc.npy')
        except ps.pexceptions.PySegInputError as e:
            print 'ERROR: The NCC matrix could not be created because of "' + e.get_message() + '"'
            print 'Terminated. (' + time.strftime("%c") + ')'
            sys.exit(-1)

        print '\tComputing the image moments...'
        try:
            star_class.build_moments(mode=mo_mode, npr=mo_npr)
            if mo_pca_nfeat is not None:
                if cu_mode == 'ncc_2dz':
                    print '\t\t-WARNING: Dimensionality reduction is not used for ncc_2dz classification mode!'
                star_class.moments_dim_reduction(n_comp=mo_pca_nfeat, method='ltsa')
        except ps.pexceptions.PySegInputError as e:
            print 'ERROR: The moments could not be ccomputed because of "' + e.get_message() + '"'
            print 'Terminated. (' + time.strftime("%c") + ')'
            sys.exit(-1)

print '\tClassification...'
try:
    if cu_alg == 'AP':
        star_class.affinity_propagation(mode_in=cu_mode, damping=ap_damp, preference=ap_pref,
                                        max_iter=ap_max_iter, convergence_iter=ap_conv_iter,
                                        verbose=True)
        star_class.compute_ccap_stat(reference=ap_ref)
        star_class.print_ccap_stat(percentile=ap_ref_per)
    elif cu_alg == 'AG':
        star_class.agglomerative_clustering(mode_in=cu_mode, n_clusters=ag_n_clusters, linkage=ag_linkage,
                                            knn=3,
                                            verbose=True)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Classification failed because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
star_class.update_relion_classes()

print '\tClassification post-processing...'
try:
    if cp_min_ccap is not None:
        print '\t\t-Purging purging classes with CC against reference lower than: ' + str(cp_min_ccap)
        purged_klasses = star_class.purge_low_ccap_particles(cp_min_ccap)
        print '\t\t\t+Purged output classes: '
        for klass, nk_parts in purged_klasses.iteritems():
            print '\t\t\t\t-Number of particles in class ' + str(klass) + ': ' + str(nk_parts)
    if cp_min_cz is not None:
        print '\t\t-Purging classes smaller than: ' + str(cp_min_cz)
        purged_klasses = star_class.purge_small_classes(cp_min_cz)
        print '\t\t\t+Purged output classes: '
        for klass, nk_parts in purged_klasses.iteritems():
            print '\t\t\t\t-Number of particles in class ' + str(klass) + ': ' + str(nk_parts)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Post-processing failed because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

out_pkl = out_dir + '/' + out_stem + '_class_star.pkl'
print '\tPickling classification object in: ' + out_pkl
star_class.pickle(out_pkl)

print '\tStoring the results...'
try:
    star_class.save_star(out_dir, out_stem, parse_rln=True, mode='gather')
    star_class.save_star(out_dir, out_stem, parse_rln=True, mode='split')
    if cc_npy is None:
        star_class.save_star(out_dir, out_stem, mode='particle')
        star_class.save_class(out_dir, out_stem, purge_k=16, mode='exemplars')
        star_class.save_class(out_dir, out_stem, purge_k=16, mode='averages')
    if star_class.get_moments_nfeatures() <= 3:
        ps.disperse_io.save_vtp(star_class.moments_to_vtp(), out_dir+'/'+out_stem+'_moments.vtp')
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Result could not be stored because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print 'Terminated. (' + time.strftime("%c") + ')'