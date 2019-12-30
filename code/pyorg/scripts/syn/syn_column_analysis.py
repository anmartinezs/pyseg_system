"""

    Find columns on synapse tomograms (ListTomograms) and measure their statistics.
    Files for columns visualization in VTK format.

    Input:  - 3 STAR files (layers) with the ListTomoParticles pickles

    Output: - Plots by tomograms and globaly:
                + Number of columns
                + Columns density
            - Visualization file:
                + vtkPolyData objects for each synapse with the columns

"""

################# Package import

import os
import copy
import csv
import numpy as np
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg.globals import unpickle_obj, clean_dir
from pyorg.surf.model import ModelCSRV
from pyorg.surf import ColumnsFinder
from matplotlib import pyplot as plt, rcParams
import seaborn as sns
import pandas as pd
try:
    import cPickle as pickle
except ImportError:
    import pickle

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

BAR_WIDTH = .40

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['patch.linewidth'] = 2

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils'

# Input STAR file
in_star_1 = ROOT_PATH + '/ves_40/ltomos_tether/ves_40_cleft_premb_mask_ltomos.star' # '/ves_40/ltomos_tether_lap/ves_40_cleft_premb_mask_lap_ltomos.star_ltomos.star' # '/ves_40/ltomos_lap/lap_ltomos.star' # '/ves_40/ltomos_premb_mask/premb_mask_ltomos.star'
key_1 = 'tether'
in_star_2 = ROOT_PATH + '/pre/ref_nomb_1_clean/ltomos_pre_premb_gather_mask/pre_premb_gather_mask_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_clst_flt_high_lap/clst_flt_high_lap_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_ABC/pre_premb_ABC_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_mask_lap/pre_premb_mask_lap_ltomos.star'
key_2 = 'pre'
in_star_3 = ROOT_PATH + '/pst/nrt/ltomos_k4_gather_premb_mask/k4_gather_premb_mask_ltomos.star' # '/pst/nrt/ltomos_clst_flt_high_lap/clst_flt_high_lap_ltomos.star' # '/pst/ampar_vs_nmdar/org/ltomos/ltomos_ampar_nmdar_premb_mask/ampar_nmdar_premb_gather_mask_ltomos.star' # '/pst/nrt/k2_ABC/ltomos_k2_premb_ABC/k2_premb_ABC_ltomos.star' # '/pst/nrt/ltomos_k2_premb_mask/k2_premb_mask_ltomos.star' # '/pst/nrt/ltomos_pst_premb_mask_lap/pst_premb_mask_lap_ltomos.star'
key_3 = 'pst'
in_tethers_csv = ROOT_PATH + '/pre/ref_nomb_1_clean/py_scripts/syn_num_tethers.csv'
in_wspace = None # ROOT_PATH + '/pst/ampar_vs_nmdar/org/col/col_gather/col_2_sim_20_maxd_10_nn2_1_nn3_1_wspace.pkl'

# Output directory
out_dir = ROOT_PATH + '/pst/ampar_vs_nmdar/org/col/col_gather'
out_stem = 'col_3_sim_20_maxd_10_nn2_1_nn3_1' # ''uni_sph_4_60_5'

# Pre-processing variables
pre_ssup = 5 #nm
pre_min_parts = 1

# Analysis variables
ana_res = 0.684 # nm/voxel
ana_max_dist = 10 # nm
ana_nn_2 = 1
ana_nn_3 = 1

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = 20 # 200
p_per = 5 # %

# AP clustering settings
do_ap_1, do_ap_2, do_ap_3 = False, False, False
ap_damp = 0.5
ap_max_iter = 20000
ap_conv_iter = 15
ap_pref = -1000
ap_c_rad = 5

# Figure saving options
fig_fmt = '.png' # if None they showed instead
fig_weight_tet = True # In categories plotting tomograms are weigthed by the number of tethers

# Plotting options

# Ctrl vs Stim tomograms
ctrl_stems = ('11_2', '11_5', '11_6', '11_9', '14_9', '14_17', '14_18', '14_19', '14_20', '14_22', '14_24', '14_25')
stim_stems = ('13_1', '13_3', '14_14', '14_15', '14_26', '14_27', '14_28', '14_32', '14_33', '15_7', '15_8', '15_12')

########################################################################################
# MAIN ROUTINE
########################################################################################

###### Additional functionality

# Computes IC from a matrix of measurements (n_arrays, array_samples)
def compute_ic(per, sims):
    if len(sims.shape) == 1:
        return sims, sims, sims
    ic_low = np.percentile(sims, per, axis=0, interpolation='linear')
    ic_med = np.percentile(sims, 50, axis=0, interpolation='linear')
    ic_high = np.percentile(sims, 100 - per, axis=0, interpolation='linear')
    return ic_low, ic_med, ic_high

# Compute the p-value for a single scalar in an array with the distribution
def compute_pval(val, sim, slope='high'):
    if slope == 'high':
        return float((val >= sim).sum()) / float(len(sim))
    else:
        return float((val >= sim).sum()) / float(len(sim))

# Computes pvalue from a matrix of simulations (n_arrays, array_samples)
def compute_pvals(exp_med, sims):
    n_sims = float(sims.shape[0])
    p_vals = np.zeros(shape=exp_med.shape, dtype=np.float32)
    for i, exp in enumerate(exp_med):
        sim_slice = sims[:, i]
        p_vals[i] = float((exp >= sim_slice).sum()) / n_sims
    return p_vals

# Check if a set of particles (rows) are aligned from the number of neighbours
def are_aligned(neighs, mask):
    aligns = np.zeros(shape=len(neighs), dtype=np.bool)
    for i, p_neigh in enumerate(neighs):
        if ((p_neigh[mask]>0).sum()) > 0:
            aligns[i] = True
    return aligns

# Units conversion
ana_max_dist_v = ana_max_dist / ana_res

########## Print initial message

print 'Second order analysis for colocalization to  ListTomoParticles by tomograms.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
print '\tInput analysis STAR file 1: ' + str(in_star_1)
print '\t\t-Key: ' + key_1
print '\tInput analysis STAR file 2: ' + str(in_star_2)
print '\t\t-Key: ' + key_2
print '\tInput analysis STAR file 3: ' + str(in_star_3)
print '\t\t-Key: ' + key_3
if in_wspace is not None:
    print '\tLoad workspace from: ' + in_wspace
else:
    print '\tPre-processing: '
    if pre_ssup is not None:
        print '\t\t-Scale supression: ' + str(pre_ssup) + ' nm'
print '\tOrganization analysis settings: '
print '\t\t-Data resolution: ' + str(ana_res) + ' nm/vx '
print '\t\t-Column radius: ' + str(ana_max_dist) + ' nm'
print '\t\t-Column radius: ' + str(ana_max_dist_v) + ' vx'
print '\t\t-Number of aligned particles in layer 2: ' + str(ana_nn_2)
print '\t\t-Number of aligned particles in layer 3: ' + str(ana_nn_3)
if do_ap_1 or do_ap_2 or do_ap_3:
    print '\tAffinity propagation settings: '
    print '\t\t-Damping: ' + str(ap_damp)
    print '\t\t-Maximum iterations: ' + str(ap_damp)
    print '\t\t-Convergence iterations: ' + str(ap_damp)
    print '\t\t-Preference: ' + str(ap_damp)
    print '\t\t-Centroid radius: ' + str(ap_c_rad)
    print '\t\t-Processing layers: '
    if do_ap_1:
        print '\t\t\t+Layer 1'
    if do_ap_2:
        print '\t\t\t+Layer 2'
    if do_ap_3:
        print '\t\t\t+Layer 3'
print '\tP-Value computation setting:'
print '\t\t-Percentile: ' + str(p_per) + ' %'
print '\t\t-Number of instances for simulations: ' + str(p_nsims)
if fig_fmt is not None:
    print '\tStoring figures:'
    print '\t\t-Format: ' + str(fig_fmt)
else:
    print '\tPlotting settings: '
print ''

######### Process

print 'Main Routine: '
mats_lists, gl_lists = None, None

out_stem_dir = out_dir + '/' + out_stem
print '\tCleaning the output dir: ' + out_stem
if os.path.exists(out_stem_dir):
    clean_dir(out_stem_dir)
else:
    os.makedirs(out_stem_dir)

print '\tLoading input data (only the first entry is loaded)...'
star_1, star_2, star_3 = sub.Star(), sub.Star(), sub.Star()
try:
    star_1.load(in_star_1)
    star_2.load(in_star_2)
    star_3.load(in_star_3)
except pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
ltomos_pkl = star_1.get_element('_psPickleFile', 0)
list_1 = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_2.get_element('_psPickleFile', 0)
list_2 = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_3.get_element('_psPickleFile', 0)
list_3 = unpickle_obj(ltomos_pkl)
set_lists = surf.SetListTomoParticles()
set_lists.add_list_tomos(list_1, key_1)
set_lists.add_list_tomos(list_2, key_2)
set_lists.add_list_tomos(list_3, key_3)

print '\tSet pre-processing...'
if pre_ssup is not None:
    pre_ssup_v = pre_ssup / ana_res
    set_lists.scale_suppression(pre_ssup_v)
if pre_min_parts > 0:
    set_lists.filter_by_particles_num_tomos(pre_min_parts)
    list_1 = set_lists.get_lists_by_key(key_1)
    list_2 = set_lists.get_lists_by_key(key_2)
    list_3 = set_lists.get_lists_by_key(key_3)

print '\tBuilding the dictionaries...'
short_tkeys_dic = dict()
tomos_np_1, tomos_np_2, tomos_np_3 = dict(), dict(), dict()
tomos_nc, tomos_den, tomos_denv, tomos_dent = dict(), dict(), dict(), dict()
tomos_nc12, tomos_den12, tomos_denv12, tomos_dent12 = dict(), dict(), dict(), dict()
tomos_nc13, tomos_den13, tomos_denv13, tomos_dent13 = dict(), dict(), dict(), dict()
tomos_nc_sims, tomos_den_sims, tomos_denv_sims, tomos_dent_sims = dict(), dict(), dict(), dict()
tomos_nc_sims12, tomos_den_sims12, tomos_denv_sims12, tomos_dent_sims12 = dict(), dict(), dict(), dict()
tomos_nc_sims13, tomos_den_sims13, tomos_denv_sims13, tomos_dent_sims13 = dict(), dict(), dict(), dict()
tomos_nc_sims2, tomos_den_sims2, tomos_denv_sims2, tomos_dent_sims2 = dict(), dict(), dict(), dict()
tomos_nc_sims212, tomos_den_sims212, tomos_denv_sims212, tomos_dent_sims212 = dict(), dict(), dict(), dict()
tomos_nc_sims213, tomos_den_sims213, tomos_denv_sims213, tomos_dent_sims213 = dict(), dict(), dict(), dict()
for tkey, ltomo in zip(list_1.get_tomos().iterkeys(), list_1.get_tomos().itervalues()):
    try:
        ltomo_1, ltomo_2 = list_2.get_tomo_by_key(tkey), list_2.get_tomo_by_key(tkey)
    except KeyError:
        print 'WARNING: tomogram in layer 1 not in lists for layers 2 or 3!'
        continue
    tomos_np_1[tkey], tomos_np_2[tkey], tomos_np_3[tkey] = 0, 0, 0
    tomos_nc[tkey], tomos_den[tkey], tomos_denv[tkey], tot_dent = 0, 0, 0, 0
    tomos_nc12[tkey], tomos_den12[tkey], tomos_denv12[tkey], tot_dent12 = 0, 0, 0, 0
    tomos_nc13[tkey], tomos_den13[tkey], tomos_denv13[tkey], tot_dent13 = 0, 0, 0, 0
    tomos_nc_sims[tkey], tomos_den_sims[tkey], tomos_denv_sims[tkey], tomos_dent_sims[tkey] = \
        list(), list(), list(), list()
    tomos_nc_sims12[tkey], tomos_den_sims12[tkey], tomos_denv_sims12[tkey], tomos_dent_sims12[tkey] = \
        list(), list(), list(), list()
    tomos_nc_sims13[tkey], tomos_den_sims13[tkey], tomos_denv_sims13[tkey], tomos_dent_sims13[tkey] = \
        list(), list(), list(), list()
    tomos_nc_sims2[tkey], tomos_den_sims2[tkey], tomos_denv_sims2[tkey], tomos_dent_sims2[tkey] = \
        list(), list(), list(), list()
    tomos_nc_sims212[tkey], tomos_den_sims212[tkey], tomos_denv_sims212[tkey], tomos_dent_sims212[tkey] = \
        list(), list(), list(), list()
    tomos_nc_sims213[tkey], tomos_den_sims213[tkey], tomos_denv_sims213[tkey], tomos_dent_sims213[tkey] = \
        list(), list(), list(), list()
    short_tkey = os.path.splitext(os.path.split(tkey)[1])[0]
    short_tkeys_dic[short_tkey] = tkey

print '\tComputing reference properties...'
vols = list_1.get_volumes_dict()
with open(in_tethers_csv, mode='r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    vesicles = dict()
    for row in reader:
        try:
            vesicles[short_tkeys_dic[row[0]]] = float(row[1])
        except KeyError:
            continue

out_tomos_dir = out_stem_dir + '/tomos'
os.makedirs(out_tomos_dir)
if in_wspace is None:

    tomo_count = 0
    print '\t\t-Tomograms computing loop:'
    for tkey, ltomo_1 in zip(list_1.get_tomos().iterkeys(), list_1.get_tomos().itervalues()):

        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        print '\t\t\t+Processing tomogram (' + str(tomo_count + 1) + \
              ' of ' + str(len(tomos_nc.keys())) + ') : ' + os.path.split(tkey)[1]
        tomo_count += 1
        ltomo_2, ltomo_3 = list_2.get_tomo_by_key(tkey), list_3.get_tomo_by_key(tkey)

        cfinder_raw = ColumnsFinder(ltomo_1, ltomo_2, ltomo_3)
        if do_ap_1 or do_ap_2 or do_ap_3:
            hold_dir = out_tomos_dir + '/' + tkey_short
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
        if do_ap_1:
            ltomo_1_ap = ltomo_1.gen_newtomo_ap(ap_pref, ap_damp, ap_max_iter, ap_conv_iter,
                                                c_rad=ap_c_rad, pj=True)
            print '\t\t\t\t-Particles->APc for layer 1: ' + str(ltomo_1.get_num_particles()) \
                  + '->' + str(ltomo_1_ap.get_num_particles())
            disperse_io.save_vtp(ltomo_1_ap.gen_particles_vtp(), hold_dir + '/' + tkey_short + '_cap1.vtp')
            ltomo_1 = ltomo_1_ap
        if do_ap_2:
            ltomo_2_ap = ltomo_2.gen_newtomo_ap(ap_pref, ap_damp, ap_max_iter, ap_conv_iter,
                                                c_rad=ap_c_rad, pj=True)
            print '\t\t\t\t-Particles->APc for layer 2: ' + str(ltomo_2.get_num_particles()) \
                  + '->' + str(ltomo_2_ap.get_num_particles())
            disperse_io.save_vtp(ltomo_2_ap.gen_particles_vtp(), hold_dir + '/' + tkey_short + '_cap2.vtp')
            ltomo_2 = ltomo_2_ap
        if do_ap_3:
            ltomo_3_ap = ltomo_3.gen_newtomo_ap(ap_pref, ap_damp, ap_max_iter, ap_conv_iter,
                                                c_rad=ap_c_rad, pj=True)
            print '\t\t\t\t-Particles->APc for layer 3: ' + str(ltomo_3.get_num_particles()) \
                    + '->' + str(ltomo_3_ap.get_num_particles())
            disperse_io.save_vtp(ltomo_3_ap.gen_particles_vtp(), hold_dir + '/' + tkey_short + '_cap3.vtp')
            ltomo_3 = ltomo_3_ap

        print '\t\t\t\t-Count the number of particles...'
        tomos_np_1[tkey] = ltomo_1.get_num_particles()
        tomos_np_2[tkey] = ltomo_2.get_num_particles()
        tomos_np_3[tkey] = ltomo_3.get_num_particles()

        print '\t\t\t\t-Computing columns...'
        cfinder = ColumnsFinder(ltomo_1, ltomo_2, ltomo_3)
        cfinder.find_columns(ana_max_dist_v, ana_nn_2, ana_nn_3)
        cfinder.find_aln12(ana_max_dist_v, ana_nn_2)
        cfinder.find_aln13(ana_max_dist_v, ana_nn_3)
        tomos_nc[tkey] = cfinder.get_num_columns()
        tomos_nc12[tkey] = cfinder.get_num_aln12()
        tomos_nc13[tkey] = cfinder.get_num_aln13()
        tomos_den[tkey] = cfinder.get_den_columns()
        tomos_den12[tkey] = cfinder.get_den_aln12()
        tomos_den13[tkey] = cfinder.get_den_aln13()
        tomos_dent[tkey] = float(cfinder.get_num_columns()) / float(ltomo_1.get_num_particles())
        tomos_dent12[tkey] = float(cfinder.get_num_aln12()) / float(ltomo_1.get_num_particles())
        tomos_dent13[tkey] = float(cfinder.get_num_aln13()) / float(ltomo_1.get_num_particles())
        if vesicles[tkey] > 0:
            tomos_denv[tkey] = float(cfinder.get_num_columns()) / float(vesicles[tkey])
            tomos_denv12[tkey] = float(cfinder.get_num_aln12()) / float(vesicles[tkey])
            tomos_denv13[tkey] = float(cfinder.get_num_aln13()) / float(vesicles[tkey])

        print '\t\t\t\t-Simulating columns (v1):'
        for i in range(p_nsims):
            print '\t\t\t\t\t+Simulating instance ' + str(i) + ' of ' + str(p_nsims) + '...'
            sim_lyr_2 = cfinder_raw.gen_layer_model(2, ModelCSRV, mode_emb='center')
            sim_lyr_3 = cfinder_raw.gen_layer_model(3, ModelCSRV, mode_emb='center')
            if do_ap_2:
                hold = sim_lyr_2.get_num_particles()
                sim_lyr_2 = sim_lyr_2.gen_newtomo_ap(ap_pref, ap_damp, ap_max_iter, ap_conv_iter,
                                                     c_rad=ap_c_rad, pj=True)
                print '\t\t\t\t-Particles->APc for layer 2: ' + str(hold) + '->' + str(sim_lyr_2.get_num_particles())
            if do_ap_3:
                hold = sim_lyr_3.get_num_particles()
                sim_lyr_3 = sim_lyr_3.gen_newtomo_ap(ap_pref, ap_damp, ap_max_iter, ap_conv_iter,
                                                     c_rad=ap_c_rad, pj=True)
                print '\t\t\t\t-Particles->APc for layer 3: ' + str(hold) + '->' + str(sim_lyr_3.get_num_particles())
            sim_cfinder = ColumnsFinder(ltomo_1, sim_lyr_2, sim_lyr_3)
            sim_cfinder.find_columns(ana_max_dist_v, ana_nn_2, ana_nn_3)
            sim_cfinder.find_aln12(ana_max_dist_v, ana_nn_2)
            sim_cfinder.find_aln13(ana_max_dist_v, ana_nn_3)
            tomos_nc_sims[tkey].append(sim_cfinder.get_num_columns())
            tomos_nc_sims12[tkey].append(sim_cfinder.get_num_aln12())
            tomos_nc_sims13[tkey].append(sim_cfinder.get_num_aln13())
            tomos_den_sims[tkey].append(sim_cfinder.get_den_columns())
            tomos_den_sims12[tkey].append(sim_cfinder.get_den_aln12())
            tomos_den_sims13[tkey].append(sim_cfinder.get_den_aln13())
            tomos_dent_sims[tkey].append(float(sim_cfinder.get_num_columns()) / float(ltomo_1.get_num_particles()))
            tomos_dent_sims12[tkey].append(float(sim_cfinder.get_num_aln12()) / float(ltomo_1.get_num_particles()))
            tomos_dent_sims13[tkey].append(float(sim_cfinder.get_num_aln13()) / float(ltomo_1.get_num_particles()))
            if vesicles[tkey] > 0:
                tomos_denv_sims[tkey].append(float(sim_cfinder.get_num_columns()) / float(vesicles[tkey]))
                tomos_denv_sims12[tkey].append(float(sim_cfinder.get_num_aln12()) / float(vesicles[tkey]))
                tomos_denv_sims13[tkey].append(float(sim_cfinder.get_num_aln13()) / float(vesicles[tkey]))

        print '\t\t\t\t-Simulating columns (v2):'
        for i in range(p_nsims):
            print '\t\t\t\t\t+Simulating instance ' + str(i) + ' of ' + str(p_nsims) + '...'
            sim_lyr_1 = cfinder_raw.gen_layer_model(1, ModelCSRV, mode_emb='center')
            if do_ap_1:
                hold = sim_lyr_1.get_num_particles()
                sim_lyr_1 = sim_lyr_1.gen_newtomo_ap(ap_pref, ap_damp, ap_max_iter, ap_conv_iter,
                                                     c_rad=ap_c_rad, pj=True)
                print '\t\t\t\t-Particles->APc for layer 1: ' + str(hold) + '->' + str(sim_lyr_1.get_num_particles())
            sim_cfinder2 = ColumnsFinder(sim_lyr_1, ltomo_2, ltomo_3)
            sim_cfinder2.find_columns(ana_max_dist_v, ana_nn_2, ana_nn_3)
            sim_cfinder2.find_aln12(ana_max_dist_v, ana_nn_2)
            sim_cfinder2.find_aln13(ana_max_dist_v, ana_nn_3)
            tomos_nc_sims2[tkey].append(sim_cfinder2.get_num_columns())
            tomos_nc_sims212[tkey].append(sim_cfinder2.get_num_aln12())
            tomos_nc_sims213[tkey].append(sim_cfinder2.get_num_aln13())
            tomos_den_sims2[tkey].append(sim_cfinder2.get_den_columns())
            tomos_den_sims212[tkey].append(sim_cfinder2.get_den_aln12())
            tomos_den_sims213[tkey].append(sim_cfinder2.get_den_aln13())
            tomos_dent_sims2[tkey].append(float(sim_cfinder2.get_num_columns()) / float(sim_lyr_1.get_num_particles()))
            tomos_dent_sims212[tkey].append(float(sim_cfinder2.get_num_aln12()) / float(sim_lyr_1.get_num_particles()))
            tomos_dent_sims213[tkey].append(float(sim_cfinder2.get_num_aln13()) / float(sim_lyr_1.get_num_particles()))
            if vesicles[tkey] > 0:
                tomos_denv_sims2[tkey].append(float(sim_cfinder2.get_num_columns()) / float(vesicles[tkey]))
                tomos_denv_sims212[tkey].append(float(sim_cfinder2.get_num_aln12()) / float(vesicles[tkey]))
                tomos_denv_sims213[tkey].append(float(sim_cfinder2.get_num_aln13()) / float(vesicles[tkey]))

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print '\tPickling computation workspace in: ' + out_wspace
    wspace = (tomos_nc, tomos_den, tomos_denv, tomos_dent,
              tomos_nc12, tomos_den12, tomos_denv12, tomos_dent12,
              tomos_nc13, tomos_den13, tomos_denv13, tomos_dent13,
              tomos_nc_sims, tomos_den_sims, tomos_denv_sims, tomos_dent_sims,
              tomos_nc_sims12, tomos_den_sims12, tomos_denv_sims12, tomos_dent_sims12,
              tomos_nc_sims13, tomos_den_sims13, tomos_denv_sims13, tomos_dent_sims13,
              tomos_nc_sims2, tomos_den_sims2, tomos_denv_sims2, tomos_dent_sims2,
              tomos_nc_sims212, tomos_den_sims212, tomos_denv_sims212, tomos_dent_sims212,
              tomos_nc_sims213, tomos_den_sims213, tomos_denv_sims213, tomos_dent_sims213,
              vesicles, vols,
              tomos_np_1, tomos_np_2, tomos_np_3)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:

    print '\tLoading the workspace: ' + in_wspace
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    tomos_nc, tomos_den, tomos_denv, tomos_dent = wspace[0], wspace[1], wspace[2], wspace[3]
    tomos_nc12, tomos_den12, tomos_denv12, tomos_dent12 = wspace[4], wspace[5], wspace[6], wspace[7]
    tomos_nc13, tomos_den13, tomos_denv13, tomos_dent13 = wspace[8], wspace[9], wspace[10], wspace[11]
    tomos_nc_sims, tomos_den_sims, tomos_denv_sims, tomos_dent_sims = wspace[12], wspace[13], \
                                                                      wspace[14], wspace[15]
    tomos_nc_sims12, tomos_den_sims12, tomos_denv_sims12, tomos_dent_sims12 = wspace[16], wspace[17], \
                                                                              wspace[18], wspace[19]
    tomos_nc_sims13, tomos_den_sims13, tomos_denv_sims13, tomos_dent_sims13 = wspace[20], wspace[21], \
                                                                              wspace[22], wspace[23]
    tomos_nc_sims2, tomos_den_sims2, tomos_denv_sims2, tomos_dent_sims2 = wspace[24], wspace[25], \
                                                                          wspace[26], wspace[27]
    tomos_nc_sims212, tomos_den_sims212, tomos_denv_sims212, tomos_dent_sims212 = wspace[28], wspace[29], \
                                                                                  wspace[30], wspace[31]
    tomos_nc_sims213, tomos_den_sims213, tomos_denv_sims213, tomos_dent_sims213 = wspace[32], wspace[33], \
                                                                                  wspace[34], wspace[35]
    vesicles, vols = wspace[36], wspace[37]
    tomos_np_1, tomos_np_2, tomos_np_3 = wspace[38], wspace[39], wspace[40]


print '\tTOMOGRAMS PLOTTING LOOP: '

print '\t\t-Plotting the number of columns...'
for tkey, nc in zip(tomos_nc.iterkeys(), tomos_nc.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('Number of columns')
    plt.bar(0.8, nc, BAR_WIDTH, color='blue', linewidth=2)
    ic_low = np.percentile(tomos_nc_sims[tkey], p_per)
    ic_med = np.percentile(tomos_nc_sims[tkey], 50)
    ic_high = np.percentile(tomos_nc_sims[tkey], 100-p_per)
    plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low2 = np.percentile(tomos_nc_sims2[tkey], p_per)
    ic_med2 = np.percentile(tomos_nc_sims2[tkey], 50)
    ic_high2 = np.percentile(tomos_nc_sims2[tkey], 100 - p_per)
    plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/nc.png')
    plt.close()

for tkey, nc in zip(tomos_nc12.iterkeys(), tomos_nc12.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('Number of alignments 1->2')
    plt.bar(0.8, nc, BAR_WIDTH, color='blue', linewidth=2)
    ic_low = np.percentile(tomos_nc_sims12[tkey], p_per)
    ic_med = np.percentile(tomos_nc_sims12[tkey], 50)
    ic_high = np.percentile(tomos_nc_sims12[tkey], 100-p_per)
    plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low2 = np.percentile(tomos_nc_sims212[tkey], p_per)
    ic_med2 = np.percentile(tomos_nc_sims212[tkey], 50)
    ic_high2 = np.percentile(tomos_nc_sims212[tkey], 100 - p_per)
    plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/aln12/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/nc.png')
    plt.close()

for tkey, nc in zip(tomos_nc13.iterkeys(), tomos_nc13.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('Number of alignments 1->3')
    plt.bar(0.8, nc, BAR_WIDTH, color='blue', linewidth=2)
    ic_low = np.percentile(tomos_nc_sims13[tkey], p_per)
    ic_med = np.percentile(tomos_nc_sims13[tkey], 50)
    ic_high = np.percentile(tomos_nc_sims13[tkey], 100-p_per)
    plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low2 = np.percentile(tomos_nc_sims213[tkey], p_per)
    ic_med2 = np.percentile(tomos_nc_sims213[tkey], 50)
    ic_high2 = np.percentile(tomos_nc_sims213[tkey], 100 - p_per)
    plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/aln13/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/nc.png')
    plt.close()

print '\t\t-Plotting columns density...'
for tkey, den in zip(tomos_den.iterkeys(), tomos_den.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('Columns density [column/nm$^3$]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.bar(0.8, den, BAR_WIDTH, color='blue', linewidth=2)
    den_sims = np.asarray(tomos_den_sims[tkey])
    ic_low = np.percentile(den_sims, p_per)
    ic_med = np.percentile(den_sims, 50)
    ic_high = np.percentile(den_sims, 100-p_per)
    plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    den_sims2 = np.asarray(tomos_den_sims2[tkey])
    ic_low2 = np.percentile(den_sims2, p_per)
    ic_med2 = np.percentile(den_sims2, 50)
    ic_high2 = np.percentile(den_sims2, 100 - p_per)
    plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/den.png')
    plt.close()

for tkey, den in zip(tomos_den12.iterkeys(), tomos_den12.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('Alignments 1->2 density [column/nm$^3$]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.bar(0.8, den, BAR_WIDTH, color='blue', linewidth=2)
    den_sims = np.asarray(tomos_den_sims12[tkey])
    ic_low = np.percentile(den_sims, p_per)
    ic_med = np.percentile(den_sims, 50)
    ic_high = np.percentile(den_sims, 100-p_per)
    plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    den_sims2 = np.asarray(tomos_den_sims212[tkey])
    ic_low2 = np.percentile(den_sims2, p_per)
    ic_med2 = np.percentile(den_sims2, 50)
    ic_high2 = np.percentile(den_sims2, 100 - p_per)
    plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/aln12/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/den.png')
    plt.close()

for tkey, den in zip(tomos_den13.iterkeys(), tomos_den13.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('Alignments 1->3 density [column/nm$^3$]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.bar(0.8, den, BAR_WIDTH, color='blue', linewidth=2)
    den_sims = np.asarray(tomos_den_sims13[tkey])
    ic_low = np.percentile(den_sims, p_per)
    ic_med = np.percentile(den_sims, 50)
    ic_high = np.percentile(den_sims, 100-p_per)
    plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    den_sims2 = np.asarray(tomos_den_sims213[tkey])
    ic_low2 = np.percentile(den_sims2, p_per)
    ic_med2 = np.percentile(den_sims2, 50)
    ic_high2 = np.percentile(den_sims2, 100 - p_per)
    plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
    plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/aln13/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/den.png')
    plt.close()

print '\t\t-Plotting columns density per synaptic vesicle...'
for tkey, denv in zip(tomos_denv.iterkeys(), tomos_denv.itervalues()):
    if len(tomos_denv_sims[tkey]) > 0:
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.ylabel('Columns density [column/sv]')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.bar(0.8, denv, BAR_WIDTH, color='blue', linewidth=2)
        denv_sims = np.asarray(tomos_denv_sims[tkey])
        ic_low = np.percentile(denv_sims, p_per)
        ic_med = np.percentile(denv_sims, 50)
        ic_high = np.percentile(denv_sims, 100-p_per)
        plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        denv_sims2 = np.asarray(tomos_denv_sims2[tkey])
        ic_low2 = np.percentile(denv_sims2, p_per)
        ic_med2 = np.percentile(denv_sims2, 50)
        ic_high2 = np.percentile(denv_sims2, 100 - p_per)
        plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
        plt.xlim(0.5, 3.5)
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            plt.savefig(hold_dir + '/denv.png')
        plt.close()

for tkey, denv in zip(tomos_denv12.iterkeys(), tomos_denv12.itervalues()):
    if len(tomos_denv_sims12[tkey]) > 0:
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.ylabel('Aligment 1->2 density [column/sv]')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.bar(0.8, denv, BAR_WIDTH, color='blue', linewidth=2)
        denv_sims = np.asarray(tomos_denv_sims12[tkey])
        ic_low = np.percentile(denv_sims, p_per)
        ic_med = np.percentile(denv_sims, 50)
        ic_high = np.percentile(denv_sims, 100-p_per)
        plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        denv_sims2 = np.asarray(tomos_denv_sims212[tkey])
        ic_low2 = np.percentile(denv_sims2, p_per)
        ic_med2 = np.percentile(denv_sims2, 50)
        ic_high2 = np.percentile(denv_sims2, 100 - p_per)
        plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
        plt.xlim(0.5, 3.5)
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            hold_dir = out_tomos_dir + '/' + tkey_short + '/aln12/'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            plt.savefig(hold_dir + '/denv.png')
        plt.close()

for tkey, denv in zip(tomos_denv13.iterkeys(), tomos_denv13.itervalues()):
    if len(tomos_denv_sims13[tkey]) > 0:
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.ylabel('Alignment 1->3 density [column/sv]')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.bar(0.8, denv, BAR_WIDTH, color='blue', linewidth=2)
        denv_sims = np.asarray(tomos_denv_sims13[tkey])
        ic_low = np.percentile(denv_sims, p_per)
        ic_med = np.percentile(denv_sims, 50)
        ic_high = np.percentile(denv_sims, 100-p_per)
        plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        denv_sims2 = np.asarray(tomos_denv_sims213[tkey])
        ic_low2 = np.percentile(denv_sims2, p_per)
        ic_med2 = np.percentile(denv_sims2, 50)
        ic_high2 = np.percentile(denv_sims2, 100 - p_per)
        plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
        plt.xlim(0.5, 3.5)
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            hold_dir = out_tomos_dir + '/' + tkey_short + '/aln13/'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            plt.savefig(hold_dir + '/denv.png')
        plt.close()

print '\t\t-Plotting columns density per tether...'
high_pvals, high_pvals2 = dict(), dict()
for tkey, dent in zip(tomos_dent.iterkeys(), tomos_dent.itervalues()):
    if len(tomos_dent_sims[tkey]) > 0:
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.ylabel('Columns density [column/tether]')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.bar(0.8, dent, BAR_WIDTH, color='blue', linewidth=2)
        dent_sims = np.asarray(tomos_dent_sims[tkey])
        ic_low = np.percentile(dent_sims, p_per)
        ic_med = np.percentile(dent_sims, 50)
        ic_high = np.percentile(dent_sims, 100-p_per)
        plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        dent_sims2 = np.asarray(tomos_dent_sims2[tkey])
        ic_low2 = np.percentile(dent_sims2, p_per)
        ic_med2 = np.percentile(dent_sims2, 50)
        ic_high2 = np.percentile(dent_sims2, 100 - p_per)
        plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
        plt.xlim(0.5, 3.5)
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            plt.savefig(hold_dir + '/dent.png')
        plt.close()
        high_pvals[tkey] = compute_pval(dent, dent_sims)
        high_pvals2[tkey] = compute_pval(dent, dent_sims2)

high_pvals12, high_pvals212 = dict(), dict()
for tkey, dent in zip(tomos_dent12.iterkeys(), tomos_dent12.itervalues()):
    if len(tomos_dent_sims12[tkey]) > 0:
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.ylabel('Alignment 1->2 density [column/tether]')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.bar(0.8, dent, BAR_WIDTH, color='blue', linewidth=2)
        dent_sims = np.asarray(tomos_dent_sims12[tkey])
        ic_low = np.percentile(dent_sims, p_per)
        ic_med = np.percentile(dent_sims, 50)
        ic_high = np.percentile(dent_sims, 100-p_per)
        plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        dent_sims2 = np.asarray(tomos_dent_sims212[tkey])
        ic_low2 = np.percentile(dent_sims2, p_per)
        ic_med2 = np.percentile(dent_sims2, 50)
        ic_high2 = np.percentile(dent_sims2, 100 - p_per)
        plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
        plt.xlim(0.5, 3.5)
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            hold_dir = out_tomos_dir + '/' + tkey_short + '/aln12/'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            plt.savefig(hold_dir + '/dent.png')
        plt.close()
        high_pvals12[tkey] = compute_pval(dent, dent_sims)
        high_pvals212[tkey] = compute_pval(dent, dent_sims2)

high_pvals13, high_pvals213 = dict(), dict()
for tkey, dent in zip(tomos_dent13.iterkeys(), tomos_dent13.itervalues()):
    if len(tomos_dent_sims13[tkey]) > 0:
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.ylabel('Alignment 1->3 density [column/tether]')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.bar(0.8, dent, BAR_WIDTH, color='blue', linewidth=2)
        dent_sims = np.asarray(tomos_dent_sims13[tkey])
        ic_low = np.percentile(dent_sims, p_per)
        ic_med = np.percentile(dent_sims, 50)
        ic_high = np.percentile(dent_sims, 100-p_per)
        plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        dent_sims2 = np.asarray(tomos_dent_sims213[tkey])
        ic_low2 = np.percentile(dent_sims2, p_per)
        ic_med2 = np.percentile(dent_sims2, 50)
        ic_high2 = np.percentile(dent_sims2, 100 - p_per)
        plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
        plt.xlim(0.5, 3.5)
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            hold_dir = out_tomos_dir + '/' + tkey_short + '/aln13/'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            plt.savefig(hold_dir + '/dent.png')
        plt.close()
        high_pvals13[tkey] = compute_pval(dent, dent_sims)
        high_pvals213[tkey] = compute_pval(dent, dent_sims2)

print '\t\t-Plotting column density by categories...'
lst, cols = list(), ['side', 'p-value']
plt.figure()
pd_pvals = None
plt.title('Columns density [Col/Tether]')
for hold_side, hold_pvals in zip(('SIMULATED', 'SIMULATED\''), (high_pvals, high_pvals2)):
    for tkey, hold_pval in zip(hold_pvals.iterkeys(), hold_pvals.itervalues()):
        if (hold_pval is not None) and np.isfinite(hold_pval):
            if fig_weight_tet:
                for i in range(tomos_np_1[tkey]):
                    lst.append([hold_side, hold_pval])
            else:
                lst.append([hold_side, hold_pval])
pd_pvals = pd.DataFrame(lst, columns=cols)
sns.boxplot(x='side', y='p-value', data=pd_pvals, color='white')
flatui = ['k', 'gray']
sns.set_palette(flatui)
sns.swarmplot(x='side', y='p-value', data=pd_pvals, size=7)
plt.ylim((0.5, 1.1))
x_line = np.linspace(-0.5, 2.5, 10)
plt.plot(x_line, .95*np.ones(shape=len(x_line)), 'k--', linewidth=2)
plt.grid(True, alpha=0.5)
plt.ylabel('p-value')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/coloc.png', dpi=300)
plt.close()

lst, cols = list(), ['side', 'p-value']
plt.figure()
pd_pvals = None
plt.title('Alignments 1->2 density [Aln/Tether]')
for hold_side, hold_pvals in zip(('SIMULATED', 'SIMULATED\''), (high_pvals12, high_pvals212)):
    for tkey, hold_pval in zip(hold_pvals.iterkeys(), hold_pvals.itervalues()):
        if (hold_pval is not None) and np.isfinite(hold_pval):
            if fig_weight_tet:
                for i in range(tomos_np_1[tkey]):
                    lst.append([hold_side, hold_pval])
            else:
                lst.append([hold_side, hold_pval])
pd_pvals = pd.DataFrame(lst, columns=cols)
sns.boxplot(x='side', y='p-value', data=pd_pvals, color='white')
flatui = ['k', 'gray']
sns.set_palette(flatui)
sns.swarmplot(x='side', y='p-value', data=pd_pvals, size=7)
plt.ylim((0.5, 1.1))
x_line = np.linspace(-0.5, 2.5, 10)
plt.plot(x_line, .95*np.ones(shape=len(x_line)), 'k--', linewidth=2)
plt.grid(True, alpha=0.5)
plt.ylabel('p-value')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/aln12.png', dpi=300)
plt.close()

lst, cols = list(), ['side', 'p-value']
plt.figure()
pd_pvals = None
plt.title('Alignments 1->3 density [Aln/Tether]')
for hold_side, hold_pvals in zip(('SIMULATED', 'SIMULATED\''), (high_pvals13, high_pvals213)):
    for tkey, hold_pval in zip(hold_pvals.iterkeys(), hold_pvals.itervalues()):
        if (hold_pval is not None) and np.isfinite(hold_pval):
            if fig_weight_tet:
                for i in range(tomos_np_1[tkey]):
                    lst.append([hold_side, hold_pval])
            else:
                lst.append([hold_side, hold_pval])
pd_pvals = pd.DataFrame(lst, columns=cols)
sns.boxplot(x='side', y='p-value', data=pd_pvals, color='white')
flatui = ['k', 'gray']
sns.set_palette(flatui)
sns.swarmplot(x='side', y='p-value', data=pd_pvals, size=7)
plt.ylim((0.5, 1.1))
x_line = np.linspace(-0.5, 2.5, 10)
plt.plot(x_line, .95*np.ones(shape=len(x_line)), 'k--', linewidth=2)
plt.grid(True, alpha=0.5)
plt.ylabel('p-value')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/aln13.png', dpi=300)
plt.close()

print '\t\t-Plotting column density by categories (CTRL vs STIM)...'
lst, cols = list(), ['side', 'p-value']
plt.figure()
pd_pvals = None
plt.title('Columns density [Col/Tether]')
for hold_side, hold_pvals in zip(('SIMULATED', 'SIMULATED\''), (high_pvals, high_pvals2)):
    for tkey, hold_pval in zip(hold_pvals.iterkeys(), hold_pvals.itervalues()):
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        if (hold_pval is not None) and np.isfinite(hold_pval):
            if tkey_stem in stim_stems:
                if fig_weight_tet:
                    for i in range(tomos_np_1[tkey]):
                        lst.append([hold_side+'+', hold_pval])
                else:
                    lst.append([hold_side+'+', hold_pval])
            else:
                if fig_weight_tet:
                    for i in range(tomos_np_1[tkey]):
                        lst.append([hold_side, hold_pval])
                else:
                    lst.append([hold_side, hold_pval])
pd_pvals = pd.DataFrame(lst, columns=cols)
sns.boxplot(x='side', y='p-value', data=pd_pvals, color='white')
flatui = ['k', 'gray', 'k', 'gray']
sns.set_palette(flatui)
sns.swarmplot(x='side', y='p-value', data=pd_pvals, size=7)
plt.ylim((0.5, 1.1))
x_line = np.linspace(-0.5, 4.5, 10)
plt.plot(x_line, .95*np.ones(shape=len(x_line)), 'k--', linewidth=2)
plt.grid(True, alpha=0.5)
plt.ylabel('p-value')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/coloc_ctrl_vs_stim.png', dpi=300)
plt.close()

print '\t\t-Plotting alignment 1->2 density by categories (CTRL vs STIM)...'
lst, cols = list(), ['side', 'p-value']
plt.figure()
pd_pvals = None
plt.title('Alignment 1->2 density [Aln/Tether]')
for hold_side, hold_pvals in zip(('SIMULATED', 'SIMULATED\''), (high_pvals12, high_pvals212)):
    for tkey, hold_pval in zip(hold_pvals.iterkeys(), hold_pvals.itervalues()):
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        if (hold_pval is not None) and np.isfinite(hold_pval):
            if tkey_stem in stim_stems:
                if fig_weight_tet:
                    for i in range(tomos_np_1[tkey]):
                        lst.append([hold_side+'+', hold_pval])
                else:
                    lst.append([hold_side+'+', hold_pval])
            else:
                if fig_weight_tet:
                    for i in range(tomos_np_1[tkey]):
                        lst.append([hold_side, hold_pval])
                else:
                    lst.append([hold_side, hold_pval])
pd_pvals = pd.DataFrame(lst, columns=cols)
sns.boxplot(x='side', y='p-value', data=pd_pvals, color='white')
flatui = ['k', 'gray', 'k', 'gray']
sns.set_palette(flatui)
sns.swarmplot(x='side', y='p-value', data=pd_pvals, size=7)
plt.ylim((0.5, 1.1))
x_line = np.linspace(-0.5, 4.5, 10)
plt.plot(x_line, .95*np.ones(shape=len(x_line)), 'k--', linewidth=2)
plt.grid(True, alpha=0.5)
plt.ylabel('p-value')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/aln12_ctrl_vs_stim.png', dpi=300)
plt.close()

print '\t\t-Plotting alignment 1->3 density by categories (CTRL vs STIM)...'
lst, cols = list(), ['side', 'p-value']
plt.figure()
pd_pvals = None
plt.title('Alignment 1->3 density [Aln/Tether]')
for hold_side, hold_pvals in zip(('SIMULATED', 'SIMULATED\''), (high_pvals13, high_pvals213)):
    for tkey, hold_pval in zip(hold_pvals.iterkeys(), hold_pvals.itervalues()):
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        if (hold_pval is not None) and np.isfinite(hold_pval):
            if tkey_stem in stim_stems:
                if fig_weight_tet:
                    for i in range(tomos_np_1[tkey]):
                        lst.append([hold_side+'+', hold_pval])
                else:
                    lst.append([hold_side+'+', hold_pval])
            else:
                if fig_weight_tet:
                    for i in range(tomos_np_1[tkey]):
                        lst.append([hold_side, hold_pval])
                else:
                    lst.append([hold_side, hold_pval])
pd_pvals = pd.DataFrame(lst, columns=cols)
sns.boxplot(x='side', y='p-value', data=pd_pvals, color='white')
flatui = ['k', 'gray', 'k', 'gray']
sns.set_palette(flatui)
sns.swarmplot(x='side', y='p-value', data=pd_pvals, size=7)
plt.ylim((0.5, 1.1))
x_line = np.linspace(-0.5, 4.5, 10)
plt.plot(x_line, .95*np.ones(shape=len(x_line)), 'k--', linewidth=2)
plt.grid(True, alpha=0.5)
plt.ylabel('p-value')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/aln13_ctrl_vs_stim.png', dpi=300)
plt.close()

print '\tTOTAL PLOTTING: '

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

print '\t\t-Gathering tomogram simulations: '
tot_nc, tot_nc12, tot_nc13, tot_vol, tot_ves, tot_teth = 0, 0, 0, 0, 0, 0
ncs_sims, ncs_sims2 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
ncs_sims12, ncs_sims212 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
ncs_sims13, ncs_sims213 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_sims, den_sims2 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_sims12, den_sims212 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_sims13, den_sims213 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_sims, denv_sims2 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_sims12, denv_sims212 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_sims13, denv_sims213 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_sims, dent_sims2 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_sims12, dent_sims212 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_sims13, dent_sims213 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
for tkey, nc_sim in zip(tomos_nc_sims.iterkeys(), tomos_nc_sims.itervalues()):
    for i in range(p_nsims):
        ncs_sims[i] += nc_sim[i]
        ncs_sims2[i] += tomos_nc_sims2[tkey][i]
        ncs_sims12[i] += tomos_nc_sims12[tkey][i]
        ncs_sims212[i] += tomos_nc_sims212[tkey][i]
        ncs_sims13[i] += tomos_nc_sims13[tkey][i]
        ncs_sims213[i] += tomos_nc_sims213[tkey][i]
    tot_vol += vols[tkey]
    tot_ves += vesicles[tkey]
    tot_teth += tomos_np_1[tkey]
    tot_nc += tomos_nc[tkey]
    tot_nc12 += tomos_nc12[tkey]
    tot_nc13 += tomos_nc13[tkey]
for i in range(p_nsims):
    if tot_vol > 0:
        den_sims[i], den_sims2[i] = ncs_sims[i] / tot_vol, ncs_sims2[i] / tot_vol
        den_sims12[i], den_sims212[i] = ncs_sims12[i] / tot_vol, ncs_sims212[i] / tot_vol
        den_sims13[i], den_sims213[i] = ncs_sims13[i] / tot_vol, ncs_sims213[i] / tot_vol
    if tot_ves > 0:
        denv_sims[i], denv_sims2[i] = ncs_sims[i] / tot_ves, ncs_sims2[i] / tot_ves
        denv_sims12[i], denv_sims212[i] = ncs_sims12[i] / tot_ves, ncs_sims212[i] / tot_ves
        denv_sims13[i], denv_sims213[i] = ncs_sims13[i] / tot_ves, ncs_sims213[i] / tot_ves
    if tot_teth > 0:
        dent_sims[i], dent_sims2[i] = ncs_sims[i] / tot_teth, ncs_sims2[i] / tot_teth
        dent_sims12[i], dent_sims212[i] = ncs_sims12[i] / tot_teth, ncs_sims212[i] / tot_teth
        dent_sims13[i], dent_sims213[i] = ncs_sims13[i] / tot_teth, ncs_sims213[i] / tot_teth

print '\t\t-Plotting the number of columns...'
plt.figure()
plt.ylabel('Number of columns')
# plt.xlabel('Total columns in the dataset')
plt.bar(0.8, tot_nc, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_sims, p_per)
ic_med = np.percentile(ncs_sims, 50)
ic_high = np.percentile(ncs_sims, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(ncs_sims2, p_per)
ic_med2 = np.percentile(ncs_sims2, 50)
ic_high2 = np.percentile(ncs_sims2, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/nc.png')
plt.close()

plt.figure()
plt.ylabel('Number of alignments 1->2')
# plt.xlabel('Total columns in the dataset')
plt.bar(0.8, tot_nc12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_sims12, p_per)
ic_med = np.percentile(ncs_sims12, 50)
ic_high = np.percentile(ncs_sims12, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(ncs_sims212, p_per)
ic_med2 = np.percentile(ncs_sims212, 50)
ic_high2 = np.percentile(ncs_sims212, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/aln12'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/nc.png')
plt.close()

plt.figure()
plt.ylabel('Number of alignments 1->3')
# plt.xlabel('Total columns in the dataset')
plt.bar(0.8, tot_nc13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_sims13, p_per)
ic_med = np.percentile(ncs_sims13, 50)
ic_high = np.percentile(ncs_sims13, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(ncs_sims213, p_per)
ic_med2 = np.percentile(ncs_sims213, 50)
ic_high2 = np.percentile(ncs_sims213, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/aln13'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/nc.png')
plt.close()

print '\t\t-Plotting columns density..'
tot_den = tot_nc / tot_vol
plt.figure()
plt.ylabel('Column density [Col/nm$^3$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.xlabel('Columns density by pre-syanptic membrane volume')
plt.bar(0.8, tot_den, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_sims, p_per)
ic_med = np.percentile(den_sims, 50)
ic_high = np.percentile(den_sims, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(den_sims2, p_per)
ic_med2 = np.percentile(den_sims2, 50)
ic_high2 = np.percentile(den_sims2, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/den.png')
plt.close()

tot_den12 = tot_nc12 / tot_vol
plt.figure()
plt.ylabel('Alignment 1->2 density [Col/nm$^3$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.xlabel('Columns density by pre-syanptic membrane volume')
plt.bar(0.8, tot_den12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_sims12, p_per)
ic_med = np.percentile(den_sims12, 50)
ic_high = np.percentile(den_sims12, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(den_sims212, p_per)
ic_med2 = np.percentile(den_sims212, 50)
ic_high2 = np.percentile(den_sims212, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/aln12'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/den.png')
plt.close()

tot_den13 = tot_nc13 / tot_vol
plt.figure()
plt.ylabel('Alignment 1->3 density [Col/nm$^3$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.xlabel('Columns density by pre-syanptic membrane volume')
plt.bar(0.8, tot_den13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_sims13, p_per)
ic_med = np.percentile(den_sims13, 50)
ic_high = np.percentile(den_sims13, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(den_sims213, p_per)
ic_med2 = np.percentile(den_sims213, 50)
ic_high2 = np.percentile(den_sims213, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/aln13'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/den.png')
plt.close()

print '\t\t-Plotting columns density by synaptic vesicles..'
tot_denv = tot_nc / tot_ves
plt.figure()
plt.ylabel('Column density [Col/SV]')
# plt.xlabel('Column probability per synaptic vesicle')
plt.bar(0.8, tot_denv, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_sims, p_per)
ic_med = np.percentile(denv_sims, 50)
ic_high = np.percentile(denv_sims, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(denv_sims2, p_per)
ic_med2 = np.percentile(denv_sims2, 50)
ic_high2 = np.percentile(denv_sims2, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/denv.png')
plt.close()

tot_denv12 = tot_nc12 / tot_ves
plt.figure()
plt.ylabel('Alignment 1->2 density [Col/SV]')
# plt.xlabel('Column probability per synaptic vesicle')
plt.bar(0.8, tot_denv12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_sims12, p_per)
ic_med = np.percentile(denv_sims12, 50)
ic_high = np.percentile(denv_sims12, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(denv_sims212, p_per)
ic_med2 = np.percentile(denv_sims212, 50)
ic_high2 = np.percentile(denv_sims212, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/aln12'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/denv.png')
plt.close()

tot_denv13 = tot_nc13 / tot_ves
plt.figure()
plt.ylabel('Alignment 1->3 density [Col/SV]')
# plt.xlabel('Column probability per synaptic vesicle')
plt.bar(0.8, tot_denv13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_sims13, p_per)
ic_med = np.percentile(denv_sims13, 50)
ic_high = np.percentile(denv_sims13, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(denv_sims213, p_per)
ic_med2 = np.percentile(denv_sims213, 50)
ic_high2 = np.percentile(denv_sims213, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/aln13'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/denv.png')
plt.close()

print '\t\t-Plotting columns density by tethers..'
tot_dent = float(tot_nc) / tot_teth
plt.figure()
plt.ylabel('Column density [Col/Tether]')
# plt.xlabel('Column probability per tether')
plt.bar(0.8, tot_dent, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_sims, p_per)
ic_med = np.percentile(dent_sims, 50)
ic_high = np.percentile(dent_sims, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(dent_sims2, p_per)
ic_med2 = np.percentile(dent_sims2, 50)
ic_high2 = np.percentile(dent_sims2, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/dent.png')
plt.close()

tot_dent12 = float(tot_nc12) / tot_teth
plt.figure()
plt.ylabel('Alignment 1->2 density [Col/Tether]')
# plt.xlabel('Column probability per tether')
plt.bar(0.8, tot_dent12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_sims12, p_per)
ic_med = np.percentile(dent_sims12, 50)
ic_high = np.percentile(dent_sims12, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(dent_sims212, p_per)
ic_med2 = np.percentile(dent_sims212, 50)
ic_high2 = np.percentile(dent_sims212, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/aln12'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/dent.png')
plt.close()

tot_dent13 = float(tot_nc13) / tot_teth
plt.figure()
plt.ylabel('Alignment 1->3 density [Col/Tether]')
# plt.xlabel('Column probability per tether')
plt.bar(0.8, tot_dent13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_sims13, p_per)
ic_med = np.percentile(dent_sims13, 50)
ic_high = np.percentile(dent_sims13, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(dent_sims213, p_per)
ic_med2 = np.percentile(dent_sims213, 50)
ic_high2 = np.percentile(dent_sims213, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
plt.xlim(0.5, 3.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/aln13'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/dent.png')
plt.close()

print '\tCTRL vs STIM PLOTTING: '

out_cs_dir = out_stem_dir + '/ctrl_vs_stim'
os.makedirs(out_cs_dir)

print '\t\t-Gathering tomogram simulations: '
tot_ctrl_nc, tot_ctrl_nc12, tot_ctrl_nc13, tot_ctrl_vol, tot_ctrl_ves, tot_ctrl_teth = 0, 0, 0, 0, 0, 0
tot_stim_nc, tot_stim_nc12, tot_stim_nc13, tot_stim_vol, tot_stim_ves, tot_stim_teth = 0, 0, 0, 0, 0, 0
ncs_ctrl_sims, ncs_stim_sims = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
ncs_ctrl_sims12, ncs_stim_sims12 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
ncs_ctrl_sims13, ncs_stim_sims13 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_ctrl_sims, den_stim_sims = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_ctrl_sims12, den_stim_sims12 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_ctrl_sims13, den_stim_sims13 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_ctrl_sims, denv_stim_sims = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_ctrl_sims12, denv_stim_sims12 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_ctrl_sims13, denv_stim_sims13 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_ctrl_sims, dent_stim_sims = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_ctrl_sims12, dent_stim_sims12 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_ctrl_sims13, dent_stim_sims13 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
ncs_ctrl_sims2, ncs_stim_sims2 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
ncs_ctrl_sims212, ncs_stim_sims212 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
ncs_ctrl_sims213, ncs_stim_sims213 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_ctrl_sims2, den_stim_sims2 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_ctrl_sims212, den_stim_sims212 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_ctrl_sims213, den_stim_sims213 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_ctrl_sims2, denv_stim_sims2 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_ctrl_sims212, denv_stim_sims212 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_ctrl_sims213, denv_stim_sims213 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_ctrl_sims2, dent_stim_sims2 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_ctrl_sims212, dent_stim_sims212 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_ctrl_sims213, dent_stim_sims213 = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
for tkey, nc_sim in zip(tomos_nc_sims.iterkeys(), tomos_nc_sims.itervalues()):
    tkey_hold = os.path.split(tkey)[1].split('_')
    tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
    if tkey_stem in ctrl_stems:
        for i in range(p_nsims):
            ncs_ctrl_sims[i] += nc_sim[i]
            ncs_ctrl_sims12[i] += tomos_nc_sims12[tkey][i]
            ncs_ctrl_sims13[i] += tomos_nc_sims13[tkey][i]
            ncs_ctrl_sims2[i] += tomos_nc_sims2[tkey][i]
            ncs_ctrl_sims212[i] += tomos_nc_sims212[tkey][i]
            ncs_ctrl_sims213[i] += tomos_nc_sims213[tkey][i]
        tot_ctrl_nc += tomos_nc[tkey]
        tot_ctrl_nc12 += tomos_nc12[tkey]
        tot_ctrl_nc13 += tomos_nc13[tkey]
        tot_ctrl_vol += vols[tkey]
        tot_ctrl_ves += vesicles[tkey]
        tot_ctrl_teth += tomos_np_1[tkey]
    elif tkey_stem in stim_stems:
        for i in range(p_nsims):
            ncs_stim_sims[i] += nc_sim[i]
            ncs_stim_sims12[i] += tomos_nc_sims12[tkey][i]
            ncs_stim_sims13[i] += tomos_nc_sims13[tkey][i]
            ncs_stim_sims2[i] += tomos_nc_sims2[tkey][i]
            ncs_stim_sims212[i] += tomos_nc_sims212[tkey][i]
            ncs_stim_sims213[i] += tomos_nc_sims213[tkey][i]
        tot_stim_nc += tomos_nc[tkey]
        tot_stim_nc12 += tomos_nc12[tkey]
        tot_stim_nc13 += tomos_nc13[tkey]
        tot_stim_vol += vols[tkey]
        tot_stim_ves += vesicles[tkey]
        tot_stim_teth += tomos_np_1[tkey]
for i in range(p_nsims):
    if tot_ctrl_vol > 0:
        den_ctrl_sims[i] = ncs_ctrl_sims[i] / tot_ctrl_vol
        den_ctrl_sims12[i] = ncs_ctrl_sims12[i] / tot_ctrl_vol
        den_ctrl_sims13[i] = ncs_ctrl_sims13[i] / tot_ctrl_vol
        den_ctrl_sims2[i] = ncs_ctrl_sims2[i] / tot_ctrl_vol
        den_ctrl_sims212[i] = ncs_ctrl_sims212[i] / tot_ctrl_vol
        den_ctrl_sims213[i] = ncs_ctrl_sims213[i] / tot_ctrl_vol
    if tot_ctrl_ves > 0:
        denv_ctrl_sims[i] = ncs_ctrl_sims[i] / tot_ctrl_ves
        denv_ctrl_sims12[i] = ncs_ctrl_sims12[i] / tot_ctrl_ves
        denv_ctrl_sims2[i] = ncs_ctrl_sims2[i] / tot_ctrl_ves
        denv_ctrl_sims212[i] = ncs_ctrl_sims212[i] / tot_ctrl_ves
        denv_ctrl_sims213[i] = ncs_ctrl_sims213[i] / tot_ctrl_ves
    if tot_ctrl_teth > 0:
        dent_ctrl_sims[i] = ncs_ctrl_sims[i] / tot_ctrl_teth
        dent_ctrl_sims12[i] = ncs_ctrl_sims12[i] / tot_ctrl_teth
        dent_ctrl_sims13[i] = ncs_ctrl_sims13[i] / tot_ctrl_teth
        dent_ctrl_sims2[i] = ncs_ctrl_sims2[i] / tot_ctrl_teth
        dent_ctrl_sims212[i] = ncs_ctrl_sims212[i] / tot_ctrl_teth
        dent_ctrl_sims213[i] = ncs_ctrl_sims213[i] / tot_ctrl_teth
    if tot_stim_vol > 0:
        den_stim_sims[i] = ncs_stim_sims[i] / tot_stim_vol
        den_stim_sims12[i] = ncs_stim_sims12[i] / tot_stim_vol
        den_stim_sims13[i] = ncs_stim_sims13[i] / tot_stim_vol
        den_stim_sims2[i] = ncs_stim_sims2[i] / tot_stim_vol
        den_stim_sims212[i] = ncs_stim_sims212[i] / tot_stim_vol
        den_stim_sims213[i] = ncs_stim_sims213[i] / tot_stim_vol
    if tot_stim_ves > 0:
        denv_stim_sims[i] = ncs_stim_sims[i] / tot_stim_ves
        denv_stim_sims12[i] = ncs_stim_sims12[i] / tot_stim_ves
        denv_stim_sims13[i] = ncs_stim_sims13[i] / tot_stim_ves
        denv_stim_sims2[i] = ncs_stim_sims2[i] / tot_stim_ves
        denv_stim_sims212[i] = ncs_stim_sims212[i] / tot_stim_ves
        denv_stim_sims213[i] = ncs_stim_sims213[i] / tot_stim_ves
    if tot_stim_teth > 0:
        dent_stim_sims[i] = ncs_stim_sims[i] / tot_stim_teth
        dent_stim_sims12[i] = ncs_stim_sims12[i] / tot_stim_teth
        dent_stim_sims13[i] = ncs_stim_sims13[i] / tot_stim_teth
        dent_stim_sims2[i] = ncs_stim_sims2[i] / tot_stim_teth
        dent_stim_sims212[i] = ncs_stim_sims212[i] / tot_stim_teth
        dent_stim_sims213[i] = ncs_stim_sims213[i] / tot_stim_teth

print '\t\t-Plotting the number of columns...'
plt.figure()
plt.ylabel('Number of columns')
plt.bar(0.8, tot_ctrl_nc, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_ctrl_sims, p_per)
ic_med = np.percentile(ncs_ctrl_sims, 50)
ic_high = np.percentile(ncs_ctrl_sims, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(ncs_ctrl_sims2, p_per)
ic_med2 = np.percentile(ncs_ctrl_sims2, 50)
ic_high2 = np.percentile(ncs_ctrl_sims2, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_nc, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_stim_sims, p_per)
ic_med = np.percentile(ncs_stim_sims, 50)
ic_high = np.percentile(ncs_stim_sims, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(ncs_stim_sims2, p_per)
ic_med2 = np.percentile(ncs_stim_sims2, 50)
ic_high2 = np.percentile(ncs_stim_sims2, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/nc.png')
plt.close()

plt.figure()
plt.ylabel('Number of alignments 1->2')
plt.bar(0.8, tot_ctrl_nc12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_ctrl_sims12, p_per)
ic_med = np.percentile(ncs_ctrl_sims12, 50)
ic_high = np.percentile(ncs_ctrl_sims12, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(ncs_ctrl_sims212, p_per)
ic_med2 = np.percentile(ncs_ctrl_sims212, 50)
ic_high2 = np.percentile(ncs_ctrl_sims212, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_nc12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_stim_sims12, p_per)
ic_med = np.percentile(ncs_stim_sims12, 50)
ic_high = np.percentile(ncs_stim_sims12, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(ncs_stim_sims212, p_per)
ic_med2 = np.percentile(ncs_stim_sims212, 50)
ic_high2 = np.percentile(ncs_stim_sims212, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/aln12'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/nc.png')
plt.close()

plt.figure()
plt.ylabel('Number of alignments 1->3')
plt.bar(0.8, tot_ctrl_nc13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_ctrl_sims13, p_per)
ic_med = np.percentile(ncs_ctrl_sims13, 50)
ic_high = np.percentile(ncs_ctrl_sims13, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(ncs_ctrl_sims213, p_per)
ic_med2 = np.percentile(ncs_ctrl_sims213, 50)
ic_high2 = np.percentile(ncs_ctrl_sims213, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_nc13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_stim_sims13, p_per)
ic_med = np.percentile(ncs_stim_sims13, 50)
ic_high = np.percentile(ncs_stim_sims13, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(ncs_stim_sims213, p_per)
ic_med2 = np.percentile(ncs_stim_sims213, 50)
ic_high2 = np.percentile(ncs_stim_sims213, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/aln13'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/nc.png')
plt.close()

print '\t\t-Plotting columns density..'
tot_ctrl_den, tot_stim_den = float(tot_ctrl_nc)/tot_ctrl_vol, float(tot_stim_nc)/tot_stim_vol
plt.figure()
plt.ylabel('Column density [Col/nm$^3$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.bar(0.8, tot_ctrl_den, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_ctrl_sims, p_per)
ic_med = np.percentile(den_ctrl_sims, 50)
ic_high = np.percentile(den_ctrl_sims, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(den_ctrl_sims2, p_per)
ic_med2 = np.percentile(den_ctrl_sims2, 50)
ic_high2 = np.percentile(den_ctrl_sims2, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_den, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_stim_sims, p_per)
ic_med = np.percentile(den_stim_sims, 50)
ic_high = np.percentile(den_stim_sims, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(den_stim_sims2, p_per)
ic_med2 = np.percentile(den_stim_sims2, 50)
ic_high2 = np.percentile(den_stim_sims2, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/den.png')
plt.close()

tot_ctrl_den12, tot_stim_den12 = float(tot_ctrl_nc12)/tot_ctrl_vol, float(tot_stim_nc12)/tot_stim_vol
plt.figure()
plt.ylabel('Alignment 1->2 density [Col/nm$^3$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.bar(0.8, tot_ctrl_den12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_ctrl_sims12, p_per)
ic_med = np.percentile(den_ctrl_sims12, 50)
ic_high = np.percentile(den_ctrl_sims12, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(den_ctrl_sims212, p_per)
ic_med2 = np.percentile(den_ctrl_sims212, 50)
ic_high2 = np.percentile(den_ctrl_sims212, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_den12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_stim_sims12, p_per)
ic_med = np.percentile(den_stim_sims12, 50)
ic_high = np.percentile(den_stim_sims12, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(den_stim_sims212, p_per)
ic_med2 = np.percentile(den_stim_sims212, 50)
ic_high2 = np.percentile(den_stim_sims212, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/aln12'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/den.png')
plt.close()

tot_ctrl_den13, tot_stim_den13 = float(tot_ctrl_nc13)/tot_ctrl_vol, float(tot_stim_nc13)/tot_stim_vol
plt.figure()
plt.ylabel('Alignment 1->3 density [Col/nm$^3$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.bar(0.8, tot_ctrl_den13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_ctrl_sims13, p_per)
ic_med = np.percentile(den_ctrl_sims13, 50)
ic_high = np.percentile(den_ctrl_sims13, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(den_ctrl_sims213, p_per)
ic_med2 = np.percentile(den_ctrl_sims213, 50)
ic_high2 = np.percentile(den_ctrl_sims213, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_den13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_stim_sims13, p_per)
ic_med = np.percentile(den_stim_sims13, 50)
ic_high = np.percentile(den_stim_sims13, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(den_stim_sims213, p_per)
ic_med2 = np.percentile(den_stim_sims213, 50)
ic_high2 = np.percentile(den_stim_sims213, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/aln13'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/den.png')
plt.close()

print '\t\t-Plotting columns density by synaptic vesicles..'
tot_ctrl_denv, tot_stim_denv = float(tot_ctrl_nc)/tot_ctrl_ves, float(tot_stim_nc)/tot_stim_ves
plt.figure()
plt.ylabel('Column density [Col/SV]')
plt.bar(0.8, tot_ctrl_denv, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_ctrl_sims, p_per)
ic_med = np.percentile(denv_ctrl_sims, 50)
ic_high = np.percentile(denv_ctrl_sims, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(denv_ctrl_sims2, p_per)
ic_med2 = np.percentile(denv_ctrl_sims2, 50)
ic_high2 = np.percentile(denv_ctrl_sims2, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_denv, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_stim_sims, p_per)
ic_med = np.percentile(denv_stim_sims, 50)
ic_high = np.percentile(denv_stim_sims, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(denv_stim_sims2, p_per)
ic_med2 = np.percentile(denv_stim_sims2, 50)
ic_high2 = np.percentile(denv_stim_sims2, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/denv.png')
plt.close()

tot_ctrl_denv12, tot_stim_denv12 = float(tot_ctrl_nc12)/tot_ctrl_ves, float(tot_stim_nc12)/tot_stim_ves
plt.figure()
plt.ylabel('Alignments 1->2 density [Col/SV]')
plt.bar(0.8, tot_ctrl_denv12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_ctrl_sims12, p_per)
ic_med = np.percentile(denv_ctrl_sims12, 50)
ic_high = np.percentile(denv_ctrl_sims12, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(denv_ctrl_sims212, p_per)
ic_med2 = np.percentile(denv_ctrl_sims212, 50)
ic_high2 = np.percentile(denv_ctrl_sims212, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_denv12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_stim_sims12, p_per)
ic_med = np.percentile(denv_stim_sims12, 50)
ic_high = np.percentile(denv_stim_sims12, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(denv_stim_sims212, p_per)
ic_med2 = np.percentile(denv_stim_sims212, 50)
ic_high2 = np.percentile(denv_stim_sims212, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/aln12'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/denv.png')
plt.close()

tot_ctrl_denv13, tot_stim_denv13 = float(tot_ctrl_nc13)/tot_ctrl_ves, float(tot_stim_nc13)/tot_stim_ves
plt.figure()
plt.ylabel('Alignments 1->3 density [Col/SV]')
plt.bar(0.8, tot_ctrl_denv13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_ctrl_sims13, p_per)
ic_med = np.percentile(denv_ctrl_sims13, 50)
ic_high = np.percentile(denv_ctrl_sims13, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(denv_ctrl_sims213, p_per)
ic_med2 = np.percentile(denv_ctrl_sims213, 50)
ic_high2 = np.percentile(denv_ctrl_sims213, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_denv13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_stim_sims13, p_per)
ic_med = np.percentile(denv_stim_sims13, 50)
ic_high = np.percentile(denv_stim_sims13, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(denv_stim_sims213, p_per)
ic_med2 = np.percentile(denv_stim_sims213, 50)
ic_high2 = np.percentile(denv_stim_sims213, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/aln13'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/denv.png')
plt.close()

print '\t\t-Plotting columns density by tethers..'
tot_ctrl_dent, tot_stim_dent = float(tot_ctrl_nc)/tot_ctrl_teth, float(tot_stim_nc)/tot_stim_teth
plt.figure()
plt.ylabel('Column density [Col/Tether]')
plt.bar(0.8, tot_ctrl_dent, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_ctrl_sims, p_per)
ic_med = np.percentile(dent_ctrl_sims, 50)
ic_high = np.percentile(dent_ctrl_sims, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(dent_ctrl_sims2, p_per)
ic_med2 = np.percentile(dent_ctrl_sims2, 50)
ic_high2 = np.percentile(dent_ctrl_sims2, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_dent, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_stim_sims, p_per)
ic_med = np.percentile(dent_stim_sims, 50)
ic_high = np.percentile(dent_stim_sims, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(dent_stim_sims2, p_per)
ic_med2 = np.percentile(dent_stim_sims2, 50)
ic_high2 = np.percentile(dent_stim_sims2, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/dent.png')
plt.close()

tot_ctrl_dent12, tot_stim_dent12 = float(tot_ctrl_nc12)/tot_ctrl_teth, float(tot_stim_nc12)/tot_stim_teth
plt.figure()
plt.ylabel('Alignments 1->2 density [Col/Tether]')
plt.bar(0.8, tot_ctrl_dent12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_ctrl_sims12, p_per)
ic_med = np.percentile(dent_ctrl_sims12, 50)
ic_high = np.percentile(dent_ctrl_sims12, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(dent_ctrl_sims212, p_per)
ic_med2 = np.percentile(dent_ctrl_sims212, 50)
ic_high2 = np.percentile(dent_ctrl_sims212, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_dent12, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_stim_sims12, p_per)
ic_med = np.percentile(dent_stim_sims12, 50)
ic_high = np.percentile(dent_stim_sims12, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(dent_stim_sims212, p_per)
ic_med2 = np.percentile(dent_stim_sims212, 50)
ic_high2 = np.percentile(dent_stim_sims212, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/aln12'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/dent.png')
plt.close()

tot_ctrl_dent13, tot_stim_dent13 = float(tot_ctrl_nc13)/tot_ctrl_teth, float(tot_stim_nc13)/tot_stim_teth
plt.figure()
plt.ylabel('Alignments 1->3 density [Col/Tether]')
plt.bar(0.8, tot_ctrl_dent13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_ctrl_sims13, p_per)
ic_med = np.percentile(dent_ctrl_sims13, 50)
ic_high = np.percentile(dent_ctrl_sims13, 100-p_per)
plt.bar(1.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(dent_ctrl_sims213, p_per)
ic_med2 = np.percentile(dent_ctrl_sims213, 50)
ic_high2 = np.percentile(dent_ctrl_sims213, 100-p_per)
plt.bar(2.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(3, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4.8, tot_stim_dent13, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_stim_sims13, p_per)
ic_med = np.percentile(dent_stim_sims13, 50)
ic_high = np.percentile(dent_stim_sims13, 100-p_per)
plt.bar(5.8, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low2 = np.percentile(dent_stim_sims213, p_per)
ic_med2 = np.percentile(dent_stim_sims213, 50)
ic_high2 = np.percentile(dent_stim_sims213, 100-p_per)
plt.bar(6.8, ic_med2, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(7, ic_med2, yerr=np.asarray([[ic_med2 - ic_low2, ic_high2 - ic_med2], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3, 5, 6, 7), ('EXP', 'SIM', 'SIM\'', 'EXP+', 'SIM+', 'SIM\'+'))
plt.xlim(0.5, 7.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/aln13'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/dent.png')
plt.close()

print '\t\t-Printing results: '
print '\n'
print '\t\t\t+Number of columns found: ' + str(tot_nc)
print '\t\t\t+Column density per pre-synaptic membrane volume: ' + str(tot_den)
print '\t\t\t+Column density per synaptic vesicle: ' + str(tot_denv)
print '\t\t\t+Column density per tether: ' + str(tot_dent)
print '\t\t\t\t*P-value (SIMULATED): ' + str(compute_pval(tot_dent, dent_sims))
print '\t\t\t\t*P-value (SIMULATED\'): ' + str(compute_pval(tot_dent, dent_sims2))
print '\t\t\t+Tether per synaptic vesicle: ' + str(float(tot_teth) / float(tot_ves))
print ''
print '\t\t\t+Number of columns found for CTRL: ' + str(tot_ctrl_nc)
print '\t\t\t+Column density per pre-synaptic membrane volume: ' + str(tot_ctrl_den)
print '\t\t\t+Column density per synaptic vesicle: ' + str(tot_ctrl_denv)
print '\t\t\t+Column density per tether: ' + str(tot_ctrl_dent)
print '\t\t\t\t*P-value (SIMULATED): ' + str(compute_pval(tot_ctrl_dent, dent_ctrl_sims))
print '\t\t\t\t*P-value (SIMULATED\'): ' + str(compute_pval(tot_ctrl_dent, dent_ctrl_sims2))
print '\t\t\t+Tether per synaptic vesicle: ' + str(float(tot_ctrl_teth) / float(tot_ctrl_ves))
print ''
print '\t\t\t+Number of columns found for STIM: ' + str(tot_stim_nc)
print '\t\t\t+Column density per pre-synaptic membrane volume: ' + str(tot_stim_den)
print '\t\t\t+Column density per synaptic vesicle: ' + str(tot_stim_denv)
print '\t\t\t+Column density per tether: ' + str(tot_stim_dent)
print '\t\t\t\t*P-value (SIMULATED): ' + str(compute_pval(tot_stim_dent, dent_stim_sims))
print '\t\t\t\t*P-value (SIMULATED\'): ' + str(compute_pval(tot_stim_dent, dent_stim_sims2))
print '\t\t\t+Tether per synaptic vesicle: ' + str(float(tot_stim_teth) / float(tot_stim_ves))
print '\n'
print '\t\t\t+Number of alignments 1->2 found: ' + str(tot_nc12)
print '\t\t\t+Alignment 1->2 density per pre-synaptic membrane volume: ' + str(tot_den12)
print '\t\t\t+Alignment 1->2 density per synaptic vesicle: ' + str(tot_denv12)
print '\t\t\t+Column density per tether: ' + str(tot_dent12)
print '\t\t\t\t*P-value (SIMULATED): ' + str(compute_pval(tot_dent12, dent_sims12))
print '\t\t\t\t*P-value (SIMULATED\'): ' + str(compute_pval(tot_dent12, dent_sims212))
print ''
print '\t\t\t+Number of alignments 1->2 found for CTRL: ' + str(tot_ctrl_nc12)
print '\t\t\t+Alignment 1->2 density per pre-synaptic membrane volume: ' + str(tot_ctrl_den12)
print '\t\t\t+Alignment 1->2 density per synaptic vesicle: ' + str(tot_ctrl_denv12)
print '\t\t\t+Alignment 1->2 density per tether: ' + str(tot_ctrl_dent12)
print '\t\t\t\t*P-value (SIMULATED): ' + str(compute_pval(tot_ctrl_dent12, dent_ctrl_sims12))
print '\t\t\t\t*P-value (SIMULATED\'): ' + str(compute_pval(tot_ctrl_dent12, dent_ctrl_sims212))
print ''
print '\t\t\t+Number of alignments 1->2 found for STIM: ' + str(tot_stim_nc12)
print '\t\t\t+Alignment 1->2 density per pre-synaptic membrane volume: ' + str(tot_stim_den12)
print '\t\t\t+Alignment 1->2 density per synaptic vesicle: ' + str(tot_stim_denv12)
print '\t\t\t+Alignment 1->2 density per tether: ' + str(tot_stim_dent12)
print '\t\t\t\t*P-value (SIMULATED): ' + str(compute_pval(tot_stim_dent12, dent_stim_sims12))
print '\t\t\t\t*P-value (SIMULATED\'): ' + str(compute_pval(tot_stim_dent12, dent_stim_sims212))
print '\n'
print '\t\t\t+Number of alignments 1->3 found: ' + str(tot_nc13)
print '\t\t\t+Alignment 1->3 density per pre-synaptic membrane volume: ' + str(tot_den13)
print '\t\t\t+Alignment 1->3 density per synaptic vesicle: ' + str(tot_denv13)
print '\t\t\t+Column density per tether: ' + str(tot_dent13)
print '\t\t\t\t*P-value (SIMULATED): ' + str(compute_pval(tot_dent13, dent_sims13))
print '\t\t\t\t*P-value (SIMULATED\'): ' + str(compute_pval(tot_dent13, dent_sims213))
print ''
print '\t\t\t+Number of alignments 1->3 found for CTRL: ' + str(tot_ctrl_nc13)
print '\t\t\t+Alignment 1->3 density per pre-synaptic membrane volume: ' + str(tot_ctrl_den13)
print '\t\t\t+Alignment 1->3 density per synaptic vesicle: ' + str(tot_ctrl_denv13)
print '\t\t\t+Alignment 1->3 density per tether: ' + str(tot_ctrl_dent13)
print '\t\t\t\t*P-value (SIMULATED): ' + str(compute_pval(tot_ctrl_dent13, dent_ctrl_sims13))
print '\t\t\t\t*P-value (SIMULATED\'): ' + str(compute_pval(tot_ctrl_dent13, dent_ctrl_sims213))
print ''
print '\t\t\t+Number of alignments 1->3 found for STIM: ' + str(tot_stim_nc13)
print '\t\t\t+Alignment 1->3 density per pre-synaptic membrane volume: ' + str(tot_stim_den13)
print '\t\t\t+Alignment 1->3 density per synaptic vesicle: ' + str(tot_stim_denv13)
print '\t\t\t+Alignment 1->3 density per tether: ' + str(tot_stim_dent13)
print '\t\t\t\t*P-value (SIMULATED): ' + str(compute_pval(tot_stim_dent13, dent_stim_sims13))
print '\t\t\t\t*P-value (SIMULATED\'): ' + str(compute_pval(tot_stim_dent13, dent_stim_sims213))
print ''
print 'Successfully terminated. (' + time.strftime("%c") + ')'
