"""

    Unsupervised and deterministic classification of membrane-bound particles

    Input:  - The STAR file with the particles to classify
            - Classification parameters

    Output: - A set of STAR files with the new classes
            - 2D rotational averaged around membrane normal exemplars and inter-particles averages per class

"""

__author__ = 'Antonio Martinez-Sanchez'

import os
import sys
import time
import numpy as np
import pyseg as ps
from matplotlib import pyplot as plt
import argparse

# Global variables

CDF_TH = 0.95 # Threshold for accumulated correlation

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '../../../..'

# Input STAR file to classify
in_star = ROOT_PATH + '/data/tutorials/synth_sumb/class/mb_align_all/class_ap_r_ali_mbf.star' # '/data/tutorials/synth_sumb/rec/nali/run1_data.star'

# Output directory for the star files
out_dir = ROOT_PATH + '/data/tutorials/synth_sumb/class/mbf_align_all/' # '/data/tutorials/synth_sumb/class/mb_align_all/'
out_stem = 'class_ap_r_ali' # 'class_ap_r'

# Particles pre-processing
pp_mask = ROOT_PATH + '/data/tutorials/synth_sumb/class/mask_cyl_130_30_110_30_nomb.mrc' # '/data/tutorials/synth_sumb/class/mask_cyl_130_30_110_30.mrc'
pp_low_sg = 4  # voxels
pp_3d = True

# Affinity propagation settings
ap_pref = -6  # -10 # -3

# Multiprocessing settings
mp_npr = 20  # Number of parallel processors if None then auto

###### Advanced settings

in_root_dir = ROOT_PATH
in_ref_dir = None # '/media/martinez/DATAPART1/syn/in'
# Input STAR file with segmentation information to focus the masks
in_seg = None # ROOT_PATH + '/two_segmentations.star'

out_debug_dir = None
out_level = 3 # Processing level: 1-particles flattening, 2-CC matrix / Feature matrix, 3-Classification

# Particles pre-processing
pp_rln_norm = False
pp_2d_norm = True
pp_direct = True
pp_n_sset = None # 3000
pp_bin = None

# CC 2d radial matrix computation parameters
cc_metric = 'cc' # 'cc' or 'similarity'
cc_npy = None # ROOT_PATH + '/test_whole/test_1_cc.npy'

## Clustering
cu_alg = 'AP' # 'AG' # 'Kmeans'
cu_mode = 'ncc_2dz' # 'ncc_2dz' # (only valid for 'AP') # 'vectors' #
# PCA dimensionality reduction (required for HAC and Kmeans)
cu_n_comp = 20

# Affinity Propagation clustering parameters
ap_damp = 0.9
ap_max_iter = 2000
ap_conv_iter = 40
ap_ref = 'exemplar' # 'average'
ap_ref_per = 33 # %

# Agglomerative Clustering (HAC)
ag_n_clusters = 50
ag_linkage = 'ward'

# Kmeans clustering
km_n_clusters = 50

# Classification post processing (requires AP)
cp_min_cz = None # 16
cp_min_ccap = 0.4 # 0.6 # 0.6 #

########################################################################################
# MAIN ROUTINE
########################################################################################

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--inStar', default=in_star, help='Input star file.')
parser.add_argument('--inMask', default=pp_mask, help='Input mask file.')
parser.add_argument('--outDir', default=out_dir, help='Output subtomograms directory.')
parser.add_argument('--filterSize', default=pp_low_sg, help='Filter size (voxels).')
parser.add_argument('--procLevel', default=out_level, help='Processing level: 1-particles flattening, 2-CC matrix '
                                                           '/ Feature matrix, 3-Classification.')
parser.add_argument('--doCC3d', default=pp_3d, help='Do 3D radial compensation.')
parser.add_argument('--ccMetric', default=cc_metric, help='Cross correlation metric: cc, similarity or cc_full.')
parser.add_argument('--clusteringAlg', default=cu_alg, help='Clustering algorithm: AP, AG or Kmeans.')
parser.add_argument('--distanceMetric', default=cu_mode, help='Distance metric: ncc_2dz or vectors (ncc_2dz only valid '
                                                              'for AP).')
parser.add_argument('--pcaComps', default=cu_n_comp, help='Number of components for PCA dimensionality reduction.')
parser.add_argument('--aggNClusters', default=ag_n_clusters, help='Number of clusters to find.')
parser.add_argument('--apPref', default=ap_pref, help='Affinity propagation preference (-inf, inf).')
parser.add_argument('--apDumping', default=ap_damp, help='AP dumping [0.5, 1).')
parser.add_argument('--apMaxIter', default=ap_max_iter, help='AP maximum number of iterations.')
parser.add_argument('--apConvIter', default=ap_conv_iter, help='AP iterations for fitting the convergence criteria.')
parser.add_argument('--apReference', default=ap_ref, help='AP reference 2D image used for classes: exemplar or'
                                                          ' average.')
parser.add_argument('--apPartSizeFilter', default=cp_min_cz, help='AP post-processing: minimum number of particles '
                                                                  'per class.')
parser.add_argument('--apCCRefFilter', default=cp_min_ccap, help='AP post-processing: Purge classes with the cross '
                                                                 'correlation against the reference lower than the '
                                                                 'specified value..')

args = parser.parse_args()
in_star = args.inStar
pp_mask = args.inMask
out_dir = args.outDir
pp_low_sg = args.filterSize
out_level = args.procLevel
pp_3d = args.doCC3d
cc_metric = args.ccMetric
cu_alg = args.clusteringAlg
cu_mode = args.distanceMetric
cu_n_comp = args.pcaComps
ag_n_clusters = args.aggNClusters
ap_pref = args.apPref
ap_damp = args.apDumping
ap_max_iter = args.apMaxIter
ap_conv_iter = args.apConvIter
ap_ref = args.apReference
cp_min_cz = args.apPartSizeFilter
cp_min_ccap = args.apDumping

########## Print initial message

print('Deterministic classification of a STAR file.')
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
    if in_seg is not None:
        print('\tInput segmentation STAR file: ' + str(in_seg))
    print('\tOutput directory: ' + str(out_dir))
    print('\tOutput stem: ' + str(out_stem))
    if out_debug_dir is not None:
        print('\tDebugging directory: ' + str(out_debug_dir))
    if out_level == 1:
        print('\tProcessing level ' + str(out_level) + ': particle flattening')
    elif out_level == 2:
        print('\tProcessing level ' + str(out_level) + ': CC matrix / Feature vectors')
    else:
        print('\tProcessing level ' + str(out_level) + ': Full classification')
    if cc_npy is not None:
        print('\tCC matrix already computed in file: ' + cc_npy)
    else:
        print('\tParticles pre-processing:')
        print('\t\t-Mask: ' + str(pp_mask))
        print('\t\t-Low pass Gaussian filter sigma: ' + str(pp_low_sg) + ' voxels')
        if pp_bin:
            print('\t\t-Binning particles by factor: ' + str(pp_bin))
        if pp_rln_norm:
            print('\t\t-Normalize particles according relion convention.')
        if pp_2d_norm:
            print('\t\t-Renormalize particles after the radial averaging.')
        if pp_3d:
            print('\t\t-Radial compensation for 3D.')
        if mp_npr is None:
            print('\t\t-Number of processes: Auto')
        else:
            print('\t\t-Number of processes: ' + str(mp_npr))
        if pp_direct:
            print('\t\t-Direct particles loading activated.')
        if pp_n_sset:
            print('\t\t-Taking a random subset of: ' + str(pp_n_sset) + ' particles')
        print('\tCC Z-axis radially averages matrix parameters: ')
        print('\t\t-Metric: ' + str(cc_metric))
else:
    print('ERROR: unrecognized extension for the input file, valid: .star, .pkl')
    print('Terminated. (' + time.strftime("%c") + ')')
print('\tClustering: ')
if cu_mode == 'ncc_2dz':
    print('\t\t-CC used as distance.')
    if cu_alg != 'AP':
        print('ERROR: ncc_2dz is a non-metric space only valid for AP clustering.')
        print('Terminated. (' + time.strftime("%c") + ')')
elif cu_mode == 'vectors':
    print('\t\t-Euclidean distance among image vectors used as distance metric.')
    print('\t\t-Number of components for PCA dimensionality reduction: ' + str(cu_n_comp))
else:
    print('ERROR: invalid input mode for clustering distance, valid: ncc_2dz or vectors')
    print('Terminated. (' + time.strftime("%c") + ')')
print('\t\t-Mode: ' + str(cu_mode))
if cu_alg == 'AP':
    print('\t\tAffinity Propagation clustering settings: ')
    print('\t\t\t-Damping: ' + str(ap_damp))
    if ap_pref is not None:
        print('\t\t\t-Affinity propagation preference: ' + str(ap_pref))
    print('\t\t\t-Maximum number of iterations: ' + str(ap_max_iter))
    print('\t\t\t-Iterations for convergence: ' + str(ap_conv_iter))
    print('\t\t\t-Reference for statistics: ' + str(ap_ref))
    print('\t\t\t-Percentile for statistics: ' + str(ap_ref_per) + ' %')
elif cu_alg == 'AG':
    print('\t\tHierarchical Ascending Clustering (or Agglomerative Clustering) settings: ')
    print('\t\t\t-Number of clusters to find: ' + str(ag_n_clusters))
    print('\t\t\t-Linkage: ' + str(ag_linkage))
elif (cu_alg == 'Kmeans') and (cu_mode == 'vectors') and (cc_npy is None):
    print('\t\tKmeans clustering settings: ')
    print('\t\t\t-Number of clusters to find: ' + str(km_n_clusters))
else:
    print('ERROR: invalid input mode for classification, valid: AP, AG or Kmeans (only with cu_mode==vectors)')
    print('Terminated. (' + time.strftime("%c") + ')')
if cu_alg == 'AP':
    print('\tClassification post-processing: ')
    if cp_min_ccap is not None:
        print('\t\t-Purge purge particles with CCAP against reference lower than: ' + str(cp_min_ccap))
    if cp_min_cz is not None:
        print('\t\t-Purge classes with less than ' + str(cp_min_cz) + ' particles')
print('')

######### Process

print('Main Routine: ')

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

    if in_seg is not None:
        seg_star = ps.sub.Star()
        try:
            seg_star.load(in_seg)
            star_class.add_segmentation(seg_star)
        except ps.pexceptions.PySegInputError as e:
            print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
            print('Terminated. (' + time.strftime("%c") + ')')
            sys.exit(-1)

    if cc_npy is not None:
        print('\tLoading NCC matrix from: ' + cc_npy)
        star_class.load_cc(cc_npy)
    else:
        print('\tLoading and pre-processing the particles...')
        try:
            mask = ps.disperse_io.load_tomo(pp_mask)
            star_class.load_particles(mask, low_sg=pp_low_sg, avg_norm=pp_2d_norm, rln_norm=pp_rln_norm, rad_3D=pp_3d,
                                      npr=mp_npr, debug_dir=out_debug_dir, ref_dir=in_ref_dir, direct_rec=pp_direct,
                                      bin=pp_bin)
            star_class.save_particles(out_dir+'/all_particles', out_stem, masks=True, stack=True)
            # imsave(out_dir+'/all_particles/global_mask.png', star_class.get_global_mask())
        except ps.pexceptions.PySegInputError as e:
            print('ERROR: Particles could not be loaded because of "' + e.get_message() + '"')
            print('Terminated. (' + time.strftime("%c") + ')')
            sys.exit(-1)

        if out_level == 1:
            print('Terminated up to level 1 (Particles flattening). (' + time.strftime("%c") + ')')
            sys.exit(1)

        if cu_mode == 'ncc_2dz':
            print('\tBuilding the NCC matrix...')
            try:
                star_class.build_ncc_z2d(metric=cc_metric, npr=mp_npr)
                star_class.save_cc(out_dir + '/' + out_stem + '_cc.npy')
            except ps.pexceptions.PySegInputError as e:
                print('ERROR: The NCC matrix could not be created because of "' + e.get_message() + '"')
                print('Terminated. (' + time.strftime("%c") + ')')
                sys.exit(-1)
        elif cu_mode == 'vectors':
            print('\tBuilding vectors...')
            try:
                star_class.build_vectors()
            except ps.pexceptions.PySegInputError as e:
                print('ERROR: The NCC matrix could not be created because of "' + e.get_message() + '"')
                print('Terminated. (' + time.strftime("%c") + ')')
                sys.exit(-1)
            if cu_n_comp is not None:
                print('\tPCA dimensionality reduction...')
                try:
                    evs = star_class.vectors_dim_reduction(n_comp=cu_n_comp, method='pca')
                    ids_evs_sorted = np.argsort(evs)[::-1]
                except ps.pexceptions.PySegInputError as e:
                    print('ERROR: Classification failed because of "' + e.get_message() + '"')
                    print('Terminated. (' + time.strftime("%c") + ')')
                    sys.exit(-1)
                plt.figure()
                plt.bar(np.arange(1, len(evs) + 1) - 0.25, evs[ids_evs_sorted], width=0.5, linewidth=2)
                plt.xlim(0, len(evs) + 1)
                plt.xticks(list(range(1, len(evs) + 1)))
                plt.xlabel('#eigenvalue')
                plt.ylabel('fraction of total correlation')
                plt.tight_layout()
                plt.savefig(out_dir + '/' + out_stem + '_evs.png', dpi=300)
                cdf_evs, evs_sorted = np.zeros(shape=len(evs), dtype=np.float32), np.sort(evs)[::-1]
                cdf_evs[0] = evs_sorted[0]
                th_x = None
                for i in range(len(evs_sorted)):
                    cdf_evs[i] = evs_sorted[:i + 1].sum()
                    if (cdf_evs[i] >= CDF_TH) and (th_x is None):
                        th_x = i + 1
                plt.figure()
                plt.bar(np.arange(1, len(evs) + 1) - 0.25, cdf_evs, width=0.5, linewidth=2)
                plt.xlim(0, len(evs) + 1)
                plt.ylim(0, 1)
                if th_x is not None:
                    plt.plot((th_x + 0.5, th_x + 0.5), (0, 1), color='k', linewidth=2, linestyle='--')
                plt.xticks(list(range(1, len(evs) + 1)))
                plt.xlabel('#eigenvalue')
                plt.ylabel('Accumulated fraction of total correlation')
                plt.tight_layout()
                plt.savefig(out_dir + '/' + out_stem + '_cdf_evs.png', dpi=300)

        if out_level == 2:
            print('Terminated up to level 1 (CC Matrix / Feature vectors). (' + time.strftime("%c") + ')')
            sys.exit(1)

print('\tClassification...')
try:
    if cu_alg == 'AP':
        star_class.affinity_propagation(mode_in=cu_mode, damping=ap_damp, preference=ap_pref,
                                        max_iter=ap_max_iter, convergence_iter=ap_conv_iter,
                                        verbose=True)
        star_class.compute_ccap_stat(reference=ap_ref)
        star_class.print_ccap_stat(percentile=ap_ref_per)
    elif cu_alg == 'AG':
        star_class.agglomerative_clustering(mode_in=cu_mode, n_clusters=ag_n_clusters, linkage=ag_linkage, knn=None,
                                            verbose=True)
    elif (cu_alg == 'Kmeans') and (cu_mode == 'vectors') and (cc_npy is None):
        star_class.kmeans_clustering(n_clusters=km_n_clusters, verbose=True)
    else:
        raise ps.pexceptions.PySegInputError('ERROR', 'No valid classification settings!')
except ps.pexceptions.PySegInputError as e:
    print('ERROR: Classification failed because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
star_class.update_relion_classes()

if cu_alg == 'AP':
    print('\tClassification post-processing...')
    try:
        if cp_min_ccap is not None:
            print('\t\t-Purging purging classes with CC against reference lower than: ' + str(cp_min_ccap))
            purged_klasses = star_class.purge_low_ccap_particles(cp_min_ccap)
            print('\t\t\t+Purged output classes: ')
            for klass, nk_parts in purged_klasses.items():
                print('\t\t\t\t-Number of particles in class ' + str(klass) + ': ' + str(nk_parts))
        if cp_min_cz is not None:
            print('\t\t-Purging classes smaller than: ' + str(cp_min_cz))
            purged_klasses = star_class.purge_small_classes(cp_min_cz)
            print('\t\t\t+Purged output classes: ')
            for klass, nk_parts in purged_klasses.items():
                print('\t\t\t\t-Number of particles in class ' + str(klass) + ': ' + str(nk_parts))
    except ps.pexceptions.PySegInputError as e:
        print('ERROR: Post-processing failed because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

out_pkl = out_dir + '/' + out_stem + '_class_star.pkl'
print('\tPickling classification object in: ' + out_pkl)
star_class.pickle(out_pkl)

print('\tStoring the results...')
try:
    star_class.save_star(out_dir, out_stem, parse_rln=True, mode='gather')
    star_class.save_star(out_dir, out_stem, parse_rln=True, mode='split')
    if cc_npy is None:
        star_class.save_star(out_dir, out_stem, mode='particle')
        star_class.save_class(out_dir, out_stem, purge_k=16, mode='averages')
        if cu_alg == 'AP':
            star_class.save_class(out_dir, out_stem, purge_k=16, mode='exemplars')
except ps.pexceptions.PySegInputError as e:
    print('ERROR: Result could not be stored because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('Terminated. (' + time.strftime("%c") + ')')
