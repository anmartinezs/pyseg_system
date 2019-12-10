"""

    Unsupervised and deterministic classification test for PCA-KMEAS algorithm

    Input:  - The STAR file with the particles to classify
            - Ground truth mask
            - Classification parameters

    Output: - A set of STAR files with the new classes
            - 2D rotational averaged around membrane normal exemplars and inter-particles averages per class
            - Sensitivity (TPR) and fall-out (FPR)

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

CDF_TH = 0.95 # Threshold for accumulated correlation

################# Package import

import os
import sys
import time
import pyseg as ps
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/ribo_johannes' # '/fs/pool/pool-lucic2/antonio/pick_test'

# Input STAR file to classify
in_star = '/fs/pool/pool-lucic2/johannes/tomograms/stefan/centering_protein/protein_160_rescaled_alignmenttransfered.star' # ROOT_PATH + '/klass/ps_2000_trans_200.star'

# Output directory for the star files
out_dir = ROOT_PATH + '/lum/klass_pca' # '/klass/test_2000/pca_nk_15_sg4'
out_stem = 'klass_1'

# Ground truth mask
gt_mask = np.concatenate((np.ones(200), np.zeros(1800))) # np.concatenate((np.ones(20), np.zeros(180)))

# Particles pre-processing
pp_mask = ROOT_PATH + '/lum/masks/mask_cyl_160_44_128_35.mrc' # '/masks/mask_cyl_160_55_130_30.mrc' # '/masks/mask_cyl_160_81_128_20.mrc' #
pp_low_sg = 4 # voxels
pp_npr = 20 # Number of parallel processors if None then auto

###### Advanced settings

# Particles pre-processing
pp_3d = False
pp_rln_norm = False
pp_2d_norm = True
pp_direct = True
pp_n_sset = None # 3000

# CC 2d radial matrix computation parameters
cc_metric = 'cc' # 'cc' or 'similarity'
cc_npr = 20 # None # 1 # None # if None then auto

# PCA+Kmeans classification
pk_ncomp = 18
pk_n_clusters = 135 # 38 # 22 # 67

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Test for deterministic classification of a STAR file (PCA+Kmeans).'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
in_ext = os.path.splitext(os.path.split(in_star)[1])[1]
if in_ext == '.pkl':
    print '\tInput pickle file: ' + str(in_star)
elif in_ext == '.star':
    print '\tInput STAR file: ' + str(in_star)
    print '\tOutput directory: ' + str(out_dir)
    print '\tOutput stem for PCA+Kmeans: ' + str(out_stem)
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
print '\tClustering: '
print '\t\t-Number of components after dimensionality reduction with PCA: ' + str(pk_ncomp)
print '\t\t-Number of clusters for Kmeans: ' + str(pk_n_clusters)
print '\tClassification post-processing: '
print ''

######### Process

print 'Main Routine: '

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

print '\tLoading and pre-processing the particles...'
try:
    mask = ps.disperse_io.load_tomo(pp_mask)
    star_class.load_particles(mask, low_sg=pp_low_sg, avg_norm=pp_2d_norm, rln_norm=pp_rln_norm, rad_3D=pp_3d,
                              npr=pp_npr, debug_dir=None, ref_dir=None, direct_rec=pp_direct)
    star_class.save_particles(out_dir+'/all_particles', out_stem, masks=True, stack=True)
    imsave(out_dir+'/all_particles/global_mask.png', star_class.get_global_mask())
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Particles could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\tBuilding the vectors...'
try:
    star_class.build_vectors()
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: The NCC matrix could not be created because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\tPreparing the ground truth...'
gt_imgs = list()
for i in range(gt_mask.shape[0]):
    if gt_mask[i]:
        gt_imgs.append(star.get_element(key='_rlnImageName', row=i))

print '\tPCA dimensionality reduction...'
try:
    evs = star_class.vectors_dim_reduction(n_comp=pk_ncomp, method='pca')
    ids_evs_sorted = np.argsort(evs)[::-1]
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Classification failed because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
plt.figure()
plt.bar(np.arange(1,len(evs)+1)-0.25, evs[ids_evs_sorted], width=0.5, linewidth=2)
plt.xlim(0, len(evs)+1)
plt.xticks(range(1,len(evs)+1))
plt.xlabel('#eigenvalue')
plt.ylabel('fraction of total correlation')
plt.tight_layout()
plt.savefig(out_dir + '/' + out_stem + '_evs.png', dpi=300)
cdf_evs, evs_sorted = np.zeros(shape=len(evs), dtype=np.float32), np.sort(evs)[::-1]
cdf_evs[0] = evs_sorted[0]
th_x = None
for i in range(len(evs_sorted)):
    cdf_evs[i] = evs_sorted[:i+1].sum()
    if (cdf_evs[i] >= CDF_TH) and (th_x is None):
        th_x = i + 1
plt.figure()
plt.bar(np.arange(1,len(evs)+1)-0.25, cdf_evs, width=0.5, linewidth=2)
plt.xlim(0, len(evs)+1)
plt.ylim(0, 1)
if th_x is not None:
    plt.plot((th_x+0.5, th_x+0.5), (0, 1), color='k', linewidth=2, linestyle='--')
plt.xticks(range(1,len(evs)+1))
plt.xlabel('#eigenvalue')
plt.ylabel('Accumulated fraction of total correlation')
plt.tight_layout()
plt.savefig(out_dir + '/' + out_stem + '_cdf_evs.png', dpi=300)

print '\tKmeans clustering...'
try:
        star_class.kmeans_clustering(n_clusters=pk_n_clusters, verbose=True)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Classification failed because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
star_class.update_relion_classes()
split_stars = star_class.get_split_stars()

print '\t\tEvaluating the classification:'
n_pos, n_tot = float(gt_mask.sum()), float(gt_mask.shape[0])
tp_lut, np_lut = np.zeros(shape=len(split_stars), dtype=np.float32), np.zeros(shape=len(split_stars), dtype=np.float32)
fp_lut = np.zeros(shape=len(split_stars), dtype=np.float32)
for k_id, split_star in enumerate(split_stars):
    np_lut[k_id] = split_star.get_nrows()
    for row in range(split_star.get_nrows()):
        if split_star.get_element('_rlnImageName', row) in gt_imgs:
            tp_lut[k_id] += 1
k_max_id = np.argmax(tp_lut)
fp = np_lut[k_max_id] - tp_lut[k_max_id]
for k_id in range(tp_lut.shape[0]):
    if k_id != k_max_id:
        fp += tp_lut[k_id]
fpr = fp / float(n_tot-n_pos)
tp = tp_lut[k_max_id]
tpr = tp / float(n_pos)
print '\t\t-Principal class found ' + str(k_max_id) + ' P=' + str(np_lut[k_max_id]) + ': [TP=' + str(int(tp)) + \
          ', TPR=' + str(tpr) + ']' + '/ [FP=' + str(int(fp)) + ', FPR=' + str(fpr) + ']'
for k_id in np.argsort(tp_lut)[::-1]:
    print '\t\t\t-Class ' + str(k_id) + ' NP=' + str(np_lut[k_id]) + ' TP=' + str(tp_lut[k_id]) + \
          ' Pk=' + str(float(tp_lut[k_id]) / np_lut[k_id])

print '\t\tStoring the results...'
try:
    star_class.save_star(out_dir, out_stem, parse_rln=True, mode='gather')
    star_class.save_star(out_dir, out_stem, parse_rln=True, mode='split')
    star_class.save_star(out_dir, out_stem, mode='particle')
    star_class.save_class(out_dir, out_stem, purge_k=0, mode='exemplars')
    star_class.save_class(out_dir, out_stem, purge_k=0, mode='averages')
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Result could not be stored because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print 'Terminated. (' + time.strftime("%c") + ')'