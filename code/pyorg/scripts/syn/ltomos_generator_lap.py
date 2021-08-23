"""

    Generates ListTomoParticle objects from a previous one file (light particles version) and AP clustering

    Input:  - ListTomoParticle STAR file
            - Particle vtkPolyData
            - AP clustering settings

    Output: - A ListTomoPaticles where previous particles have been substituted by culsters centers

"""

################# Package import

import os
import vtk
import numpy as np
import scipy as sp
import sys
import time
import math
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg import globals as gl
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils' # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre' # '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/ves_40'

# Input STAR file
in_star = ROOT_PATH + '/ves_40/ltomos_tether/ves_40_cleft_premb_mask_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_ABC/pre_premb_ABC_ltomos.star' # '/pst/nrt/ltomos_k4_gather_premb_mask/k4_gather_premb_mask_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_gather_mask/pre_premb_gather_mask_ltomos.star'

# Input particle vtkPolyData
in_vtp = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre/vtps/sph_rad_5_surf.vtp'

# Output directory
out_dir = ROOT_PATH + '/ves_40/ltomos_tether_lap' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_ABC_lap' # '/pst/nrt/ltomos_pst_premb_mask_lap' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_mask_lap' # '/ref_nomb_1_clean/ltomos_clst_flt_high_lap' # '/ltomos_lap'
out_stem = 'ves_40_cleft_premb_mask_lap_ltomos.star' # 'pst_premb_ABC_lap' # 'pre_premb_mask_lap'

# Segmentation pre-processing
sg_bc = False
sg_bm = 'box'
sg_pj = True
sg_voi_mask = True

# AP clustering settings
ap_damp = 0.5
ap_max_iter = 20000
ap_conv_iter = 15
ap_pref = -1000

# Post-processing
pt_res = 0.684
pt_ss = 5 # nm
pt_min_parts = 0
pt_keep = None

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Clustering a ListTomoParticles.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tInput STAR file with the ListTomoParticle: ' + str(in_star))
print('\tInput particle vtkPolyData: ' + str(in_vtp))
print('\tSegmentation pre-processing: ')
if sg_bc:
    print('\t\t-Checking particles VOI boundary with mode: ' + str(sg_bm))
if sg_pj:
    print('\t\t-Activated particles projecting on surface VOI.')
if sg_voi_mask:
    print('\t\t-Mask VOI mode activated!')
print('\tAffinity propagation settings: ')
print('\t\t-Damping: ' + str(ap_damp))
print('\t\t-Maximum iterations: ' + str(ap_damp))
print('\t\t-Convergence iterations: ' + str(ap_damp))
print('\t\t-Preference: ' + str(ap_damp))
print('\tPost-processing: ')
if pt_ss is not None:
    pt_ss_v = pt_ss / pt_res
    print('\t\t-Scale suppression: ' + str(pt_ss) + ' nm (' + str(pt_ss_v) + ' voxels)')
print('\t\t-Keep tomograms the ' + str(pt_keep) + 'th with the highest number of particles.')
print('\t\t-Minimum number of particles: ' + str(pt_min_parts))
print('')

######### Process

print('Main Routine: ')

print('\tChecking particle vtkPolyData...')
part_vtp = disperse_io.load_poly(in_vtp)
if not isinstance(part_vtp, vtk.vtkPolyData):
    print('ERROR: input file ' + in_vtp + ' is not a vtkPolyData object!')
    print('Unsuccesfully terminated. (' + time.strftime("%c") + ')')
    print(sys.exit(-1))
if not surf.is_closed_surface(part_vtp):
    print('ERROR: input file ' + in_vtp + ' is not a closed surface!')
    print('Unsuccesfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
out_vtp_str = out_dir + '/' + out_stem + '_part_surf.vtp'
disperse_io.save_vtp(part_vtp, out_vtp_str)

print('\tLoading input ListTomoParticles...')
set_lists = surf.SetListTomoParticles()
star, star_ap = sub.Star(), sub.Star()
try:
    star.load(in_star)
    star_ap.add_column('_psPickleFile')
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('\tLOOP FOR TOMOGRAMS: ')
clsts_size, clsts_np = list(), list()
set_lists = surf.SetListTomoParticles()
for list_pkl in star.get_column_data('_psPickleFile'):

    print('\t\t-Processing list: ' + list_pkl)
    ltomos = gl.unpickle_obj(list_pkl)
    list_ap = surf.ListTomoParticles()
    if pt_ss is not None:
        print('\tApplying scale suppression...')
        ltomos.scale_suppression(pt_ss_v)

    for tomo in ltomos.get_tomo_list():

        tomo_fname = tomo.get_tomo_fname()
        print('\tProcessing tomogram: ' + str(tomo_fname))
        list_ap.add_tomo(surf.TomoParticles(tomo_fname, 1, voi=tomo.get_voi()))

        print('\tGetting tomogram points...')
        coords = tomo.get_particle_coords()
        if coords is None:
            print('WARNING: no coordinates found, skipping...')
            continue

        print('\tAffinity propagation...')
        aff = AffinityPropagation(damping=ap_damp,
                                  convergence_iter=ap_conv_iter,
                                  max_iter=ap_max_iter,
                                  preference=ap_pref)
        aff.fit(coords)

        print('\tGetting the clusters...')
        clsts_cg = list()
        if aff.cluster_centers_indices_ is not None:
            n_clst = len(aff.cluster_centers_indices_)
            for lbl in aff.labels_:
                clst_coords = list()
                ids = np.where(aff.labels_ ==  lbl)[0]
                for idx in ids:
                    clst_coords.append(coords[idx, :])
                clst_coords = np.asarray(clst_coords, dtype=np.float32)
                clsts_cg.append(clst_coords.mean(axis=0))
                clsts_np.append(clst_coords.shape[0])
                # Compute the cluster size (the largest distance)
                dst = 0
                for i in range(clst_coords.shape[0]):
                    hold = clst_coords - clst_coords[i, :]
                    hold = math.sqrt((hold * hold).sum(axis=1).max())
                    if hold > dst:
                        dst = hold
                clsts_size.append(dst)
            print('\t\t-Number of clusters found: ' + str(n_clst))

            print('\tInserting cluster centers to tomogram...')
            for cg in clsts_cg:
                try:
                    part = surf.ParticleL(out_vtp_str, center=cg, eu_angs=(0, 0, 0))
                    list_ap.insert_particle(part, tomo_fname, check_bounds=sg_bc, mode=sg_bm, voi_pj=sg_pj)
                except pexceptions.PySegInputError as e:
                    print('WARINING: particle ' + str(cg) + ' could not be inserted in tomogram ' + tomo_fname + \
                          ' because of "' + e.get_message() + '"')
                    pass
        else:
            print('WARNING: affinity propagation did not converge, skipping...')
            continue

    if pt_keep is not None:
        print('\t\tFiltering to keep the ' + str(pt_keep) + 'th more highly populated')
        list_ap.clean_low_pouplated_tomos(pt_keep)
    if pt_min_parts >= 0:
        print('\t\tFiltering tomograms with less particles than: ' + str(pt_min_parts))
        list_ap.filter_by_particles_num(pt_min_parts)

    star_stem = os.path.splitext(os.path.split(list_pkl)[1])[0]
    out_pkl = out_dir + '/' + star_stem + '_tpl.pkl'
    print('\t\tPickling the list of tomograms in the file: ' + out_pkl)
    try:
        list_ap.pickle(out_pkl)
        kwargs = {'_psPickleFile': out_pkl}
        star_ap.add_row(**kwargs)
    except pexceptions.PySegInputError as e:
        print('ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    # Adding particle to list
    set_lists.add_list_tomos(list_ap, star_stem)

out_fig_np = out_dir + '/' + out_stem + '_np.png'
plt.figure()
plt.title('Cluster number of particles histogram')
plt.ylabel('Frequency')
plt.xlabel('Number of particles')
plt.hist(clsts_np, bins=10)
plt.tight_layout()
plt.savefig(out_fig_np)
plt.close()
out_fig_size = out_dir + '/' + out_stem + '_sz.png'
plt.figure()
plt.title('Cluster size histogram')
plt.ylabel('Frequency')
plt.xlabel('Size (nm)')
plt.hist(clsts_size, bins=10)
plt.tight_layout()
plt.savefig(out_fig_size)
plt.close()

out_parts = out_dir + '/' + out_stem + '_parts.star'
print('\tStoring the particles STAR file: ' + out_parts)
set_lists.to_particles_star().store(out_parts)

print('\tStoring list appended by tomograms in: ' + out_dir)
tomos_vtp = set_lists.tomos_to_vtp(mode='surface')
for key, poly in zip(iter(tomos_vtp.keys()), iter(tomos_vtp.values())):
    stem_tomo = os.path.splitext(os.path.split(key)[1])[0]
    disperse_io.save_vtp(poly, out_dir+'/'+stem_tomo+'_lists_app.vtp')

out_star = out_dir + '/' + out_stem + '_ltomos.star'
print('\tOutput STAR file: ' + out_star)
star_ap.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')
