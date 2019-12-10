"""

    Unsupervised and deterministic classification of membrane-bound particles

    Input:  - The STAR file with the particles to classify
            - Classification parameters

    Output: - A set of STAR files with the new classes
            - 2D rotational averaged around membrane normal exemplars and inter-particles averages per class

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

################# Package import

import os
import sys
import time
import pyseg as ps
import numpy as np
from scipy.misc import imsave

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/pick_test'

# Input STAR file to classify
in_star = ROOT_PATH + '/klass/ps_2000_trans_200.star'

# Output directory for the star files
out_dir = ROOT_PATH + '/klass/test_2000/ap_pca_pref_5000'
out_stem_ap = 'klass_ap'
out_stem_ag = 'klass_ag'

# Ground truth mask
gt_mask = np.concatenate((np.ones(200), np.zeros(1800))) # np.concatenate((np.ones(20), np.zeros(180)))

# Particles pre-processing
pp_mask = ROOT_PATH + '/masks/mask_cyl_160_55_130_30.mrc' # '/masks/mask_cyl_160_81_128_20.mrc' #
pp_low_sg = 6 # voxels
pp_npr = 20 # Number of parallel processors if None then auto
ap_pref = -5000 # 0 # -3 # -2 # -1 # 0 # -3 # None # -40 # -10 # -3 # -6

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

## Clustering
cu_mode = 'vectors' # 'ncc_2dz' # 'moments' # 'ncc_2dz'
cu_n_comp = 18 # None

# Affinity Propagation clustering parameters
ap_damp = 0.9
ap_max_iter = 2000
ap_conv_iter = 40
ap_ref = 'exemplar' # 'average'
ap_ref_per = 33 # %

# Agglomerative Clustering
ag_n_clusters = 15
ag_linkage = 'average' # 'complete'

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Test for deterministic classification of a STAR file.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput STAR file: ' + str(in_star)
print '\tOutput directory: ' + str(out_dir)
print '\tOutput stem for AP: ' + str(out_stem_ap)
print '\tOutput stem for AG: ' + str(out_stem_ag)
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
if cu_mode == 'ncc_2dz':
    print '\tCC Z-axis radially averages matrix parameters: '
    print '\t\t-Metric: ' + str(cc_metric)
    if cc_npr is None:
        print '\t\t-Number of processes: Auto'
    else:
        print '\t\t-Number of processes: ' + str(cc_npr)
print '\tClustering: '
print '\t\t-Mode: ' + str(cu_mode)
if cu_mode != 'ncc_2dz':
    print '\t\t-Number of components: ' + str(cu_n_comp)
print '\t\tAffinity Propagation classification settings: '
print '\t\t\t-Damping: ' + str(ap_damp)
if ap_pref is not None:
    print '\t\t\t-Affinity propagation preference: ' + str(ap_pref)
print '\t\t\t-Maximum number of iterations: ' + str(ap_max_iter)
print '\t\t\t-Iterations for convergence: ' + str(ap_conv_iter)
print '\t\t\t-Reference for statistics: ' + str(ap_ref)
print '\t\t\t-Percentile for statistics: ' + str(ap_ref_per) + ' %'
print '\t\tAgglomerative clustering classificiation settings: '
print '\t\t\t-Number of clusters to find: ' + str(ag_n_clusters)
print '\t\t\t-Linkage: ' + str(ag_linkage)
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
    star_class.save_particles(out_dir+'/all_particles', out_stem_ap, masks=True, stack=True)
    imsave(out_dir+'/all_particles/global_mask.png', star_class.get_global_mask())
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Particles could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

if cu_mode == 'ncc_2dz':
    print '\tBuilding the NCC matrix...'
    try:
        star_class.build_ncc_z2d(metric=cc_metric, npr=cc_npr)
    except ps.pexceptions.PySegInputError as e:
        print 'ERROR: The NCC matrix could not be created because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
elif cu_mode == 'vectors':
    print '\tBuilding vectors...'
    try:
        star_class.build_vectors()
        if cu_n_comp is not None:
            star_class.vectors_dim_reduction(n_comp=cu_n_comp, method='pca')
    except ps.pexceptions.PySegInputError as e:
        print 'ERROR: The NCC matrix could not be created because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)

print '\tPreparing the ground truth...'
gt_imgs = list()
for i in range(gt_mask.shape[0]):
    if gt_mask[i]:
        gt_imgs.append(star.get_element(key='_rlnImageName', row=i))

print '\tAffinity Propagation classification...'
try:
        star_class.affinity_propagation(mode_in=cu_mode, damping=ap_damp, preference=ap_pref,
                                        max_iter=ap_max_iter, convergence_iter=ap_conv_iter,
                                        verbose=True)
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
    print '\t\t\t-Class ' + str(k_id) + ' NP=' + str(np_lut[k_id]) + ' Pk=' + str(float(tp_lut[k_id])/np_lut[k_id])

print '\t\tStoring the results...'
try:
    star_class.save_star(out_dir, out_stem_ap, parse_rln=True, mode='gather')
    star_class.save_star(out_dir, out_stem_ap, parse_rln=True, mode='split')
    star_class.save_star(out_dir, out_stem_ap, mode='particle')
    star_class.save_class(out_dir, out_stem_ap, purge_k=0, mode='exemplars')
    star_class.save_class(out_dir, out_stem_ap, purge_k=0, mode='averages')
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Result could not be stored because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

if cu_mode == 'ncc_2dz':

    print '\tAgglomerative clustering classification...'
    try:
            star_class.agglomerative_clustering(mode_in=cu_mode, n_clusters=ag_n_clusters, linkage=ag_linkage,knn=3,
                                            verbose=True)
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
        print '\t\t\t-Class ' + str(k_id) + ' NP=' + str(np_lut[k_id]) + ' Pk=' + str(float(tp_lut[k_id]) / np_lut[k_id])

print '\t\tStoring the results...'
try:
    star_class.save_star(out_dir, out_stem_ag, parse_rln=True, mode='gather')
    star_class.save_star(out_dir, out_stem_ag, parse_rln=True, mode='split')
    star_class.save_star(out_dir, out_stem_ag, mode='particle')
    star_class.save_class(out_dir, out_stem_ag, purge_k=16, mode='exemplars')
    star_class.save_class(out_dir, out_stem_ag, purge_k=16, mode='averages')
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: Result could not be stored because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print 'Terminated. (' + time.strftime("%c") + ')'