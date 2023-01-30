"""

    Performs Univariate 2nd order analysis to a SetListTomograms object by tomograms

    Input:  - A STAR file with a set of ListTomoParticles pickles (SetListTomograms object input)

    Output: - Plots by tomograms:
                + Number of particles by list
                + Particles densities by list
                + Plots by list:
                    * Univariate 2nd order analysis against random simulation
            - Global plots:
                + Global univariate 2nd order analysis against random simulation by list

"""

################# Package import

import os
import pickle
import csv
import numpy as np
import scipy as sp
import copy
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg.globals import unpickle_obj, clean_dir
from pyorg.surf.model import ModelCSRV
from pyorg.surf.utils import list_tomoparticles_pvalues
from matplotlib import pyplot as plt, rcParams
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
try:
    import pickle as pickle
except ImportError:
    import pickle

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

BAR_WIDTH = .4
BAR_WIDTH_THIN = 0.1
MB_THICK = 5 # nm

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/col'

# Input STAR file
in_star_pre = ROOT_PATH + '/ltomos_method_1/3_pre_tether_pst_ltomos.star' # '/ltomos_nn/ltomos_2_2_2/az_pst_all_premb_2_2_2_ltomos.star' # '/ltomos_ves_40_lap_gather_gather/ves_40_lap_gather_gather_premb_ltomos.star'
key_pre = 'COL'
in_star_pre_a = ROOT_PATH + '/ltomos_method_1/3_pre_tether_pst_ltomos.star' # '/ltomos_nn/ltomos_2_2_2/az_pst_all_premb_2_2_2_ltomos.star' # '/ltomos_ves_40_lap_gather_gather/ves_40_lap_gather_gather_premb_ltomos.star'
key_pre_a = 'PRE' # 'PST_AP'
in_star_pre_b = ROOT_PATH + '/ltomos_method_1/3_pre_tether_pst_ltomos.star' # '/ltomos_nn/ltomos_2_2_2/az_pst_all_premb_2_2_2_ltomos.star' # '/ltomos_ves_40_lap_gather_gather/ves_40_lap_gather_gather_premb_ltomos.star'
key_pre_b = 'TETHER' # 'PRE'
in_star_pre_c = ROOT_PATH + '/ltomos_method_1/3_pre_tether_pst_ltomos.star' # '/ltomos_nn/ltomos_2_2_2/az_pst_all_premb_2_2_2_ltomos.star' # '/ltomos_ves_40_lap_gather_gather/ves_40_lap_gather_gather_premb_ltomos.star'
key_pre_c = 'PST' # 'PRE_AP'
in_tethers_csv = ROOT_PATH + '/../pre/ref_nomb_1_clean/py_scripts/syn_num_tethers.csv'
in_wspace = None # ROOT_PATH + '/uni_col/uni_shell_method_1/uni_1_102_2_10_sim_20_local_ssup_5_wspace.pkl' # (Insert a path to recover a pickled workspace instead of doing a new computation)

# Output directory
out_dir = ROOT_PATH + '/uni_col/uni_sph_method_1'
out_stem = 'uni_1_102_2_10_sim_20_local_ssup_5_ref' # ''uni_sph_4_60_5'

# Pre-processing variables
pre_ssup = 5 #nm
pre_min_parts = None # 5

# Analysis variables
ana_res = 0.684 # nm/voxel
ana_rg = np.arange(1, 102, 2) # np.arange(4, 100, 2) # in nm
ana_global = False # if False all tomograms are treated separatedly for computing the results by lists
ana_shell_thick = None # 10 # None # 5 # nm
ana_border = True
ana_conv_iter = None # 100
ana_max_iter = None # 100000
ana_fmm = True
ana_npr = 5 # None means Auto
ana_npr_model = 1 # In general equal to ana_npr unless you have memory problems

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = 20
p_per = 5 # %
# p_max_den = 6*0.000025
# Particle surface
p_vtp = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre/vtps/sph_rad_0.5_surf.vtp'

# Figure saving options
fig_fmt = '.png' # if None they showed instead

# Plotting options
pt_sim_v = True
pt_cmap = plt.get_cmap('gist_rainbow')
pt_xmin, pt_xmax = 1, 100
pt_sg_flt_ncoefs = None # 7 # Number of coeficients for Stravisky-Golay filter (when functions are noisy)

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

# Computes pvalue from a matrix of simulations (n_arrays, array_samples)
def compute_pvals(exp_med, sims):
    n_sims = float(sims.shape[0])
    p_vals = np.zeros(shape=exp_med.shape, dtype=np.float32)
    for i, exp in enumerate(exp_med):
        sim_slice = sims[:, i]
        p_vals[i] = float((exp >= sim_slice).sum()) / n_sims
    return p_vals

# Units conversion
ana_rg_v = ana_rg / ana_res
ana_shell_thick_v = None
if ana_shell_thick is not None:
    ana_shell_thick_v = float(ana_shell_thick) / ana_res

########## Print initial message

print('Univariate second order analysis for a ListTomoParticles by tomograms.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tInput STAR files: ')
print('\t\t-Key ' + str(key_pre) + ': ' + str(in_star_pre))
print('\t\t-Key ' + str(key_pre_a) + ': ' + str(in_star_pre_a))
print('\t\t-Key ' + str(key_pre_b) + ': ' + str(in_star_pre_b))
print('\t\t-Key ' + str(key_pre_c) + ': ' + str(in_star_pre_c))
if in_wspace is not None:
    print('\tLoad workspace from: ' + in_wspace)
else:
    print('\tPre-processing: ')
    if pre_ssup is not None:
        print('\t\t-Scale supression: ' + str(pre_ssup) + ' nm')
    print('\t\t-Minimum number of particles in tomogram by list: ' + str(pre_min_parts))
print('\tOrganization analysis settings: ')
print('\t\t-Range of radius: ' + str(ana_rg) + ' nm')
print('\t\t-Range of radius: ' + str(ana_rg_v) + ' voxels')
if ana_shell_thick is None:
    print('\t\t-Spherical neighborhood')
else:
    print('\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick) + ' nm')
    print('\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick_v) + ' voxels')
if ana_global:
    print('\t\t-Global computation mode activated.')
if in_wspace is None:
    if (ana_conv_iter is not None) and (ana_max_iter is not None):
        print('\t\t-Convergence number of samples for stochastic volume estimations: ' + str(ana_conv_iter))
        print('\t\t-Maximum number of samples for stochastic volume estimations: ' + str(ana_max_iter))
    else:
        if ana_fmm:
            print('\t\t-FMM for distance metric computation.')
        else:
            print('\t\t-DT for distance transform computation.')
    if ana_npr is None:
        print('\t\t-Number of processors: Auto')
    else:
        print('\t\t-Number of processors: ' + str(ana_npr))
    if ana_npr_model:
        print('\t\t-Number of processors for models simulation: Auto')
    else:
        print('\t\t-Number of processors for models simulation: ' + str(ana_npr))
print('\tP-Value computation setting:')
print('\t\t-Percentile: ' + str(p_per) + ' %')
print('\t\t-Number of instances for simulations: ' + str(p_nsims))
print('\t\t-Particle surface: ' + p_vtp)
if fig_fmt is not None:
    print('\tStoring figures:')
    print('\t\t-Format: ' + str(fig_fmt))
else:
    print('\tPlotting settings: ')
print('\t\t-Colormap: ' + str(pt_cmap))
if pt_sim_v:
    print('\t\t-Verbose simulation activated!')
if pt_sg_flt_ncoefs is not None:
    print('\t\t-Number coefficients for Stravinsky Golay filtering: ' + str(pt_sg_flt_ncoefs))
print('')

# Ctrl vs Stim tomograms
ctrl_stems = ('11_2', '11_5', '11_6', '11_9', '14_9', '14_17', '14_18', '14_19', '14_20', '14_22', '14_24', '14_25')
stim_stems = ('13_1', '13_3', '14_14', '14_15', '14_26', '14_27', '14_28', '14_32', '14_33', '15_7', '15_8', '15_12')

######### Process

print('Main Routine: ')
mats_lists, gl_lists = None, None

out_stem_dir = out_dir + '/' + out_stem
print('\tCleaning the output dir: ' + out_stem)
if os.path.exists(out_stem_dir):
    clean_dir(out_stem_dir)
else:
    os.makedirs(out_stem_dir)

if in_wspace is None:

    print('\tLoading input data...')
    star_pre, star_pre_a = sub.Star(), sub.Star()
    star_pre_b, star_pre_c = sub.Star(), sub.Star()
    try:
        star_pre.load(in_star_pre)
        star_pre_a.load(in_star_pre_a)
        star_pre_b.load(in_star_pre_b)
        star_pre_c.load(in_star_pre_c)
    except pexceptions.PySegInputError as e:
        print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
    ltomos_pkl = star_pre.get_element('_psPickleFile', 4)
    list_pre = unpickle_obj(ltomos_pkl)
    ltomos_pkl = star_pre_a.get_element('_psPickleFile', 5)
    list_pre_a = unpickle_obj(ltomos_pkl)
    ltomos_pkl = star_pre_b.get_element('_psPickleFile', 6)
    list_pre_b = unpickle_obj(ltomos_pkl)
    ltomos_pkl = star_pre_c.get_element('_psPickleFile', 7)
    list_pre_c = unpickle_obj(ltomos_pkl)
    set_lists = surf.SetListTomoParticles()
    set_lists.add_list_tomos(list_pre, key_pre)
    set_lists.add_list_tomos(list_pre_a, key_pre_a)
    set_lists.add_list_tomos(list_pre_b, key_pre_b)
    set_lists.add_list_tomos(list_pre_c, key_pre_c)
    try:
        part_vtp = disperse_io.load_poly(p_vtp)
    except pexceptions.PySegInputError as e:
        print('ERROR: reference particle surface file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    print('\tSet pre-processing...')
    if pre_ssup is not None:
        pre_ssup_v = pre_ssup / ana_res
        set_lists.scale_suppression(pre_ssup_v)
    if pre_min_parts > 0:
        set_lists.filter_by_particles_num_tomos(pre_min_parts)

    print('\tBuilding the dictionaries...')
    lists_count, tomos_count = 0, 0
    lists_dic = dict()
    lists_hash, tomos_hash = dict(), dict()
    tomos_np, tomos_den, tomos_exp, tomos_sim = dict(), dict(), dict(), dict()
    lists_np, lists_den, lists_gden, lists_exp, lists_sim, lists_color = dict(), dict(), dict(), dict(), dict(), dict()
    tmp_sim_folder = out_dir + '/tmp_gen_list_' + out_stem
    set_lists_dic = set_lists.get_lists()
    for lkey, llist in zip(iter(set_lists_dic.keys()), iter(set_lists_dic.values())):
        print('\t\t-Processing list: ' + lkey)
        short_key = lkey
        print('\t\t\t+Short key found: ' + short_key)
        try:
            lists_dic[short_key]
        except KeyError:
            lists_dic[short_key] = llist
            lists_hash[lists_count] = short_key
            lists_np[short_key], lists_den[short_key], lists_gden[short_key] = dict(), dict(), dict()
            lists_exp[short_key] = list()
            if ana_global:
                lists_sim[short_key] = dict()
            else:
                lists_sim[short_key] = list()
            lists_count += 1
    for lkey, llist in zip(iter(set_lists_dic.keys()), iter(set_lists_dic.values())):
        llist_tomos_dic = llist.get_tomos()
        print('\t\t-Processing list: ' + lkey)
        short_key = lkey
        for tkey, ltomo in zip(iter(llist_tomos_dic.keys()), iter(llist_tomos_dic.values())):
            try:
                tomos_np[tkey]
            except KeyError:
                tomos_hash[tkey] = tomos_count
                tomos_np[tkey], tomos_den[tkey], tomos_exp[tkey], tomos_sim[tkey] = dict(), dict(), dict(), dict()
                tomos_count += 1
            tomos_exp[tkey][short_key] = 0
            tomos_sim[tkey][short_key] = list()
            tomos_np[tkey][short_key] = 0
            tomos_den[tkey][short_key] = 0
            lists_np[short_key][tkey] = 0
            lists_den[short_key][tkey] = 0
            if ana_global:
                lists_sim[short_key][tkey] = list()

    print('\tComputing reference properties...')
    vols = lists_dic[key_pre].get_volumes_dict()
    with open(in_tethers_csv, mode='r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        vesicles = dict()
        for row in reader:
            vesicles[row[0]] = float(row[1])

    print('\tLIST COMPUTING LOOP:')
    for li, lkey in enumerate(lists_hash.values()):

        llist = lists_dic[lkey]
        print('\t\t-Processing list: ' + lkey)
        print('\t\t\t+Computing global density...')
        lists_gden[lkey] = llist.compute_global_density()
        print('\t\t\t+Tomograms computing loop:')
        for lt, tkey in enumerate(tomos_hash.keys()):

            print('\t\t\t\t*Processing tomogram (list ' + str(li+1) + ' of ' + str(len(list(lists_hash.values()))) + ', tomo ' + str(lt+1) + ' of ' + str(len(list(tomos_hash.keys()))) + ') : ' + os.path.split(tkey)[1])
            try:
                ltomo = llist.get_tomo_by_key(tkey)
            except KeyError:
                print('\t\t\t\t(WARNING) Tomogram ' + tkey + ' not in list ' + lkey)
                continue

            print('\t\t\t\t\t-Computing the number of particles...')
            hold_np = ltomo.get_num_particles()
            tomos_np[tkey][lkey] = hold_np
            lists_np[lkey][tkey] = hold_np

            print('\t\t\t\t\t-Computing density by tomogram...')
            hold_den = ltomo.compute_global_density()
            tomos_den[tkey][lkey] = hold_den
            lists_den[lkey][tkey] = hold_den

            print('\t\t\t\t\t-Computing univariate second order metrics...')
            if ana_global:
                hold_arr_1, hold_arr_2 = ltomo.compute_uni_2nd_order(ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                                     conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                                     out_sep=2, fmm=ana_fmm, npr=ana_npr)
                if (hold_arr_1 is not None) and (hold_arr_2 is not None):
                    tomos_exp[tkey][lkey] = (hold_arr_1, hold_arr_2)
                    for npi in range(tomos_np[tkey][lkey]):
                        lists_exp[lkey].append((hold_arr_1, hold_arr_2))
            else:
                hold_arr = ltomo.compute_uni_2nd_order(ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                       conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                       out_sep=0, fmm=ana_fmm, npr=ana_npr)
                if hold_arr is not None:
                    tomos_exp[tkey][lkey] = hold_arr
                    for npi in range(tomos_np[tkey][lkey]):
                        lists_exp[lkey].append(hold_arr)

            if ana_global:
                print('\t\t\t\t\t-Simulating univariate second order metrics...')
                hold_arr_1, hold_arr_2 = ltomo.simulate_uni_2nd_order(p_nsims, ModelCSRV(), part_vtp, 'center',
                                                                      ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                                      conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                                      fmm=ana_fmm, npr=ana_npr)
                for npi in range(tomos_np[tkey][lkey]):
                    if (hold_arr_1 is not None) and (hold_arr_2 is not None):
                        for arr_1, arr_2 in zip(hold_arr_1, hold_arr_2):
                            tomos_sim[tkey][lkey].append((arr_1, arr_2))
                            lists_sim[lkey][tkey].append((arr_1, arr_2))
            else:
                print('\t\t\t\t\t-Simulating univariate second order metrics...')
                hold_arr = ltomo.simulate_uni_2nd_order(p_nsims, ModelCSRV(), part_vtp, 'center',
                                                        ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                        conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                        fmm=ana_fmm, npr=ana_npr)
                if hold_arr is not None:
                    tomos_sim[tkey][lkey] = hold_arr
                    for npi in range(tomos_np[tkey][lkey]):
                        for arr in hold_arr:
                            lists_sim[lkey].append(arr)

    if ana_global:
        print('\tGlobal computations by tomos...')
        hold_tomos_exp, hold_tomos_sim = copy.deepcopy(tomos_exp), copy.deepcopy(tomos_sim)
        del tomos_exp
        del tomos_sim
        tomos_exp, tomos_sim = dict(), dict()
        for tkey in hold_tomos_exp.keys():
            tomos_exp[tkey], tomos_sim[tkey] = dict(), dict()
            dens = tomos_den[tkey]
            for lkey, mat in zip(iter(hold_tomos_exp[tkey].keys()), iter(hold_tomos_exp[tkey].values())):
                arr_1, arr_2 = mat[0], mat[1]
                if ana_shell_thick is None:
                    gl_arr = ana_rg * (np.cbrt((1. / dens[lkey]) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))) - 1.)
                else:
                    gl_arr = (1. / dens[lkey]) * (arr_1.sum(axis=0) / arr_2.sum(axis=0)) - 1.
                tomos_exp[tkey][lkey] = gl_arr
            for lkey, mat in zip(iter(hold_tomos_sim[tkey].keys()), iter(hold_tomos_sim[tkey].values())):
                for n_sim in range(p_nsims):
                    mat = hold_tomos_sim[tkey][lkey]
                    arr_1, arr_2 = mat[n_sim][0], mat[n_sim][1]
                    if ana_shell_thick is None:
                        gl_arr = ana_rg * (np.cbrt((1. / dens[lkey]) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))) - 1.)
                    else:
                        gl_arr = (1. / dens[lkey]) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))  - 1.
                    try:
                        tomos_sim[tkey][lkey].append(gl_arr)
                    except KeyError:
                        tomos_sim[tkey][lkey] = list()
                        tomos_sim[tkey][lkey].append(gl_arr)
        print('\tGlobal computations by lists...')
        hold_lists_exp, hold_lists_sim = copy.deepcopy(lists_exp), copy.deepcopy(lists_sim)
        del lists_exp
        del lists_sim
        lists_exp, lists_sim = dict(), dict()
        for lkey in hold_lists_exp.keys():
            lists_exp[lkey], lists_sim[lkey] = list(), list()
            dens, mat = lists_gden[lkey], hold_lists_exp[lkey]
            arr_1, arr_2 = list(), list()
            for hold_mat in mat:
                for hold_mat_1, hold_mat_2 in zip(hold_mat[0], hold_mat[1]):
                    arr_1.append(hold_mat_1)
                    arr_2.append(hold_mat_2)
            arr_1, arr_2 = arr_1, arr_2 = np.asarray(arr_1), np.asarray(arr_2)
            if ana_shell_thick is None:
                gl_arr = ana_rg * (np.cbrt((1. / dens) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))) - 1.)
            else:
                gl_arr = (1. / dens) * (arr_1.sum(axis=0) / arr_2.sum(axis=0)) - 1.
            lists_exp[lkey] = gl_arr
        for lkey in hold_lists_sim.keys():
            dens = lists_gden[lkey]
            for n_sim in range(p_nsims):
                arr_1, arr_2 = list(), list()
                for mat in hold_lists_sim[lkey].values():
                    for hold_mat_1, hold_mat_2 in zip(mat[n_sim][0], mat[n_sim][1]):
                        arr_1.append(hold_mat_1)
                        arr_2.append(hold_mat_2)
                arr_1, arr_2 = np.asarray(arr_1), np.asarray(arr_2)
                if ana_shell_thick is None:
                    gl_arr = ana_rg * (np.cbrt((1. / dens) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))) - 1.)
                else:
                    gl_arr = (1. / dens) * (arr_1.sum(axis=0) / arr_2.sum(axis=0)) - 1.
                try:
                    lists_sim[lkey].append(gl_arr)
                except KeyError:
                    lists_sim[lkey] = list()
                    lists_sim[lkey].append(gl_arr)

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print('\tPickling computation workspace in: ' + out_wspace)
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash,
              tomos_np, tomos_den, tomos_exp, tomos_sim,
              lists_np, lists_den, lists_gden, lists_exp, lists_sim, lists_color,
              vesicles, vols)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:
    print('\tLoading the workspace: ' + in_wspace)
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    lists_count, tomos_count = wspace[0], wspace[1]
    lists_hash, tomos_hash = wspace[2], wspace[3]
    tomos_np, tomos_den, tomos_exp, tomos_sim = wspace[4], wspace[5], wspace[6], wspace[7]
    lists_np, lists_den, lists_gden, lists_exp, lists_sim, lists_color = wspace[8], wspace[9], wspace[10], wspace[11], wspace[12], wspace[13]
    vesicles, vols = wspace[14], wspace[15]

print('\tPrinting lists hash: ')
for id, lkey in zip(iter(lists_hash.keys()), iter(lists_hash.values())):
    print('\t\t-[' + str(id) + '] -> [' + lkey + ']')
print('\tPrinting tomograms hash: ')
for tkey, val in zip(iter(tomos_hash.keys()), iter(tomos_hash.values())):
    print('\t\t-[' + tkey + '] -> [' + str(val) + ']')

# Getting the lists colormap
n_lists = len(list(lists_hash.keys()))
for i, lkey in zip(iter(lists_hash.keys()), iter(lists_hash.values())):
    lists_color[lkey] = pt_cmap(1.*i/n_lists)

print('\tTOMOGRAMS PLOTTING LOOP: ')

out_tomos_dir = out_stem_dir + '/tomos'
os.makedirs(out_tomos_dir)

print('\t\t-Plotting densities by tethered vesicles...')
plt.figure()
# plt.title('Colocalization respect ' + key_ref)
plt.ylabel('Number of particles')
plt.xlabel('Number of tethered vesicles')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
l_ves, l_num_pre, l_num_pre_a, l_num_pre_b, l_num_pre_c = list(), list(), list(), list(), list()
l_wnum_pre, l_wnum_pre_a, l_wnum_pre_b, l_wnum_pre_c = list(), list(), list(), list()
p_max_np = 0
for tkey in tomos_den.keys():
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    # if vesicles[tkey_short] > 0:
    if np.asarray(list(tomos_np[tkey].values())).sum() > 0:
        l_ves.append(vesicles[tkey_short])
        l_num_pre.append(tomos_np[tkey][key_pre])
        l_wnum_pre.append(1.)
        l_num_pre_a.append(tomos_np[tkey][key_pre_a])
        l_wnum_pre_a.append(1.)
        l_num_pre_b.append(tomos_np[tkey][key_pre_b])
        l_wnum_pre_b.append(1.)
        l_num_pre_c.append(tomos_np[tkey][key_pre_c])
        l_wnum_pre_c.append(1.)
        if l_num_pre[-1] > p_max_np:
            p_max_np = l_num_pre[-1]
l_ves = np.asarray(l_ves, dtype=float).reshape(-1, 1)
l_num_pre = np.asarray(l_num_pre, dtype=float).reshape(-1, 1)
l_num_pre_a = np.asarray(l_num_pre_a, dtype=float).reshape(-1, 1)
l_num_pre_b = np.asarray(l_num_pre_b, dtype=float).reshape(-1, 1)
l_num_pre_c = np.asarray(l_num_pre_c, dtype=float).reshape(-1, 1)
l_wnum_pre = np.asarray(l_wnum_pre, dtype=float)
l_wnum_pre_a = np.asarray(l_wnum_pre_a, dtype=float)
l_wnum_pre_b = np.asarray(l_wnum_pre_b, dtype=float)
l_wnum_pre_c = np.asarray(l_wnum_pre_c, dtype=float)
l_wnum_pre /= l_wnum_pre.sum()
l_wnum_pre_a /= l_wnum_pre_a.sum()
l_wnum_pre_b /= l_wnum_pre_b.sum()
l_wnum_pre_c /= l_wnum_pre_c.sum()
regr_pre = linear_model.LinearRegression()
regr_pre_a = linear_model.LinearRegression()
regr_pre_b = linear_model.LinearRegression()
regr_pre_c = linear_model.LinearRegression()
regr_pre.fit(l_ves, l_num_pre, sample_weight=l_wnum_pre)
regr_pre_a.fit(l_ves, l_num_pre_a, sample_weight=l_wnum_pre_a)
regr_pre_b.fit(l_ves, l_num_pre_b, sample_weight=l_wnum_pre_b)
regr_pre_c.fit(l_ves, l_num_pre_c, sample_weight=l_wnum_pre_c)
l_num_pre_r = regr_pre.predict(l_ves)
l_num_pre_a_r = regr_pre_a.predict(l_ves)
l_num_pre_b_r = regr_pre_b.predict(l_ves)
l_num_pre_c_r = regr_pre_c.predict(l_ves)
plt.plot(l_ves, l_num_pre, color='k', marker='s', markersize=10, label='pre', linestyle='')
plt.plot(l_ves, l_num_pre_a, color='b', marker='*', markersize=10, label='pre_A', linestyle='')
plt.plot(l_ves, l_num_pre_b, color='c', marker='^', markersize=10, label='pre_B', linestyle='')
plt.plot(l_ves, l_num_pre_c, color='darkcyan', marker='o', markersize=10, label='pre_C', linestyle='')
plt.plot(l_ves, l_num_pre_r, color='k', label='PRE-LR', linestyle='-', linewidth=2.0)
plt.plot(l_ves, l_num_pre_a_r, color='b', label='PRE_A-LR', linestyle='-', linewidth=2.0)
plt.plot(l_ves, l_num_pre_b_r, color='c', label='PRE_B-LR', linestyle='--', linewidth=2.0)
plt.plot(l_ves, l_num_pre_c_r, color='darkcyan', label='PRE_C-LR', linestyle='-.', linewidth=2.0)
plt.xlim((-3., l_ves.max()*1.1))
plt.ylim((0, p_max_np*1.1))
plt.xticks((0, 2, 4, 6, 8, 10, 12))
plt.grid(True, alpha=0.5)
plt.legend(loc=3)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/den_by_ntet.png', dpi=600)
plt.close()
print('\t\t\t+Linear regression:')
r2_pre = r2_score(l_num_pre, l_num_pre_r)
r2_pre_a = r2_score(l_num_pre_a, l_num_pre_a_r)
r2_pre_b = r2_score(l_num_pre_b, l_num_pre_b_r)
r2_pre_c = r2_score(l_num_pre_c, l_num_pre_c_r)
print('\t\t\t\t-Coefficient of determination PRE: ' + str(r2_pre))
print('\t\t\t\t-Coefficient of determination PRE_A: ' + str(r2_pre_a))
print('\t\t\t\t-Coefficient of determination PRE_B: ' + str(r2_pre_b))
print('\t\t\t\t-Coefficient of determination PRE_C: ' + str(r2_pre_c))
[pc_pre, pcv_pre] = sp.stats.pearsonr(l_num_pre, l_num_pre_r)
pc_pre, pcv_pre = pc_pre[0], pcv_pre[0]
[pc_pre_a, pcv_pre_a] = sp.stats.pearsonr(l_num_pre_a, l_num_pre_a_r)
pc_pre_a, pcv_pre_a = pc_pre_a[0], pcv_pre_a[0]
[pc_pre_b, pcv_pre_b] = sp.stats.pearsonr(l_num_pre_b, l_num_pre_b_r)
pc_pre_b, pcv_pre_b = pc_pre_b[0], pcv_pre_b[0]
[pc_pre_c, pcv_pre_c] = sp.stats.pearsonr(l_num_pre_c, l_num_pre_c_r)
pc_pre_c, pcv_pre_c = pc_pre_c[0], pcv_pre_c[0]
print('\t\t\t\t-Pearson coefficient PRE [p, t]: ' + str([pc_pre, pcv_pre]))
print('\t\t\t\t-Pearson coefficient PRE_A [p, t]: ' + str([pc_pre_a, pcv_pre_a]))
print('\t\t\t\t-Pearson coefficient PRE_B [p, t]: ' + str([pc_pre_b, pcv_pre_b]))
print('\t\t\t\t-Pearson coefficient PRE_C [p, t]: ' + str([pc_pre_c, pcv_pre_c]))

print('\t\t-Plotting densities by pre-synaptic membrane volume...')
plt.figure()
# plt.title('Colocalization respect ' + key_ref)
plt.ylabel('Number of particles')
plt.xlabel('Membrane Area [nm]')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
l_areas, l_vols, l_num_pre, l_num_pre_a, l_num_pre_b, l_num_pre_c = list(), list(), list(), list(), list(), list()
l_wnum_pre, l_wnum_pre_a, l_wnum_pre_b, l_wnum_pre_c = list(), list(), list(), list()
for tkey in tomos_den.keys():
    vol = vols[tkey] * (1. / (ana_res*ana_res*ana_res))
    area = (vol / MB_THICK)
    if np.asarray(list(tomos_np[tkey].values())).sum() > 0:
        l_vols.append(vol)
        l_areas.append(area)
        l_num_pre.append(tomos_np[tkey]['PRE'])
        l_wnum_pre.append(1.)
        l_num_pre_a.append(tomos_np[tkey][key_pre_a])
        l_wnum_pre_a.append(1.)
        l_num_pre_b.append(tomos_np[tkey][key_pre_b])
        l_wnum_pre_b.append(1.)
        l_num_pre_c.append(tomos_np[tkey][key_pre_c])
        l_wnum_pre_c.append(1.)
l_ves = np.asarray(l_ves, dtype=float).reshape(-1, 1)
l_areas = np.sqrt(np.asarray(l_areas, dtype=float)).reshape(-1, 1)
l_num_pre = np.asarray(l_num_pre, dtype=float).reshape(-1, 1)
l_num_pre_a = np.asarray(l_num_pre_a, dtype=float).reshape(-1, 1)
l_num_pre_b = np.asarray(l_num_pre_b, dtype=float).reshape(-1, 1)
l_num_pre_c = np.asarray(l_num_pre_c, dtype=float).reshape(-1, 1)
l_wnum_pre = np.asarray(l_wnum_pre, dtype=float)
l_wnum_pre_a = np.asarray(l_wnum_pre_a, dtype=float)
l_wnum_pre_b = np.asarray(l_wnum_pre_b, dtype=float)
l_wnum_pre_c = np.asarray(l_wnum_pre_c, dtype=float)
l_wnum_pre /= l_wnum_pre.sum()
l_wnum_pre_a /= l_wnum_pre_a.sum()
l_wnum_pre_b /= l_wnum_pre_b.sum()
l_wnum_pre_c /= l_wnum_pre_c.sum()
regr_pre = linear_model.LinearRegression()
regr_pre_a = linear_model.LinearRegression()
regr_pre_b = linear_model.LinearRegression()
regr_pre_c = linear_model.LinearRegression()
regr_pre.fit(l_areas, l_num_pre, sample_weight=l_wnum_pre)
regr_pre_a.fit(l_areas, l_num_pre_a, sample_weight=l_wnum_pre_a)
regr_pre_b.fit(l_areas, l_num_pre_b, sample_weight=l_wnum_pre_b)
regr_pre_c.fit(l_areas, l_num_pre_c, sample_weight=l_wnum_pre_c)
l_num_pre_r = regr_pre.predict(l_areas)
l_num_pre_a_r = regr_pre_a.predict(l_areas)
l_num_pre_b_r = regr_pre_b.predict(l_areas)
l_num_pre_c_r = regr_pre_c.predict(l_areas)
plt.plot(l_areas, l_num_pre, color='k', marker='s', markersize=10, label='pre', linestyle='')
plt.plot(l_areas, l_num_pre_a, color='b', marker='*', markersize=10, label='pre_A', linestyle='')
plt.plot(l_areas, l_num_pre_b, color='cyan', marker='^', markersize=10, label='pre_B', linestyle='')
plt.plot(l_areas, l_num_pre_c, color='darkcyan', marker='o', markersize=10, label='pre_C', linestyle='')
plt.plot(l_areas, l_num_pre_r, color='k', label='PRE-LR', linestyle='-', linewidth=2.0)
plt.plot(l_areas, l_num_pre_a_r, color='b', label='PRE_A-LR', linestyle='-', linewidth=2.0)
plt.plot(l_areas, l_num_pre_b_r, color='cyan', label='PRE_B-LR', linestyle='--', linewidth=2.0)
plt.plot(l_areas, l_num_pre_c_r, color='darkcyan', label='PRE_C-LR', linestyle='-.', linewidth=2.0)
plt.xlim((-0.1, l_areas.max()*1.1))
plt.ylim((0, p_max_np*1.1))
plt.grid(True, alpha=0.5)
plt.legend(loc=3)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/den_by_vol.png', dpi=600)
plt.close()
r2_pre_v = r2_score(l_num_pre, l_num_pre_r)
r2_pre_a_v = r2_score(l_num_pre_a, l_num_pre_a_r)
r2_pre_b_v = r2_score(l_num_pre_b, l_num_pre_b_r)
r2_pre_c_v = r2_score(l_num_pre_c, l_num_pre_c_r)
print('\t\t\t\t-Coefficient of determination PRE: ' + str(r2_pre_v))
print('\t\t\t\t-Coefficient of determination PRE_A: ' + str(r2_pre_a_v))
print('\t\t\t\t-Coefficient of determination PRE_B: ' + str(r2_pre_b_v))
print('\t\t\t\t-Coefficient of determination PRE_C: ' + str(r2_pre_c_v))
[pc_pre_v, pcv_pre_v] = sp.stats.pearsonr(l_num_pre, l_num_pre_r)
pc_pre_v, pcv_pre_v = pc_pre_v[0], pcv_pre_v[0]
[pc_pre_a_v, pcv_pre_a_v] = sp.stats.pearsonr(l_num_pre_a, l_num_pre_a_r)
pc_pre_a_v, pcv_pre_a_v = pc_pre_a_v[0], pcv_pre_a_v[0]
[pc_pre_b_v, pcv_pre_b_v] = sp.stats.pearsonr(l_num_pre_b, l_num_pre_b_r)
pc_pre_b_v, pcv_pre_b_v = pc_pre_b_v[0], pcv_pre_b_v[0]
[pc_pre_c_v, pcv_pre_c_v] = sp.stats.pearsonr(l_num_pre_c, l_num_pre_c_r)
pc_pre_c_v, pcv_pre_c_v = pc_pre_c_v[0], pcv_pre_c_v[0]
print('\t\t\t\t-Pearson coefficient PRE [p, t]: ' + str([pc_pre_v, pcv_pre_v]))
print('\t\t\t\t-Pearson coefficient PRE_A [p, t]: ' + str([pc_pre_a_v, pcv_pre_a_v]))
print('\t\t\t\t-Pearson coefficient PRE_B [p, t]: ' + str([pc_pre_b_v, pcv_pre_b_v]))
print('\t\t\t\t-Pearson coefficient PRE_C [p, t]: ' + str([pc_pre_c_v, pcv_pre_c_v]))

print('\t\t$R^2$ coefficient of determination...')
plt.figure()
plt.ylabel(r'$R^2$ coefficient of determination')
plt.bar(0.8, r2_pre, width=BAR_WIDTH, color='b')
plt.bar(1.8, r2_pre_v, width=BAR_WIDTH, color='k')
plt.bar(3.8, r2_pre_a, width=BAR_WIDTH, color='b')
plt.bar(4.8, r2_pre_a_v, width=BAR_WIDTH, color='k')
plt.bar(6.8, r2_pre_b, width=BAR_WIDTH, color='b')
plt.bar(7.8, r2_pre_b_v, width=BAR_WIDTH, color='k')
plt.bar(9.8, r2_pre_c, width=BAR_WIDTH, color='b')
plt.bar(10.8, r2_pre_c_v, width=BAR_WIDTH, color='k')
plt.plot((0, 12), (0, 0), 'k--', linewidth=2)
plt.xticks((1, 2, 4, 5, 7, 8, 10, 11),
           (key_pre, 'PRE*', key_pre_a, 'PRE_A*', key_pre_b, 'PRE_B*', key_pre_c, 'PRE_C*'),
           fontsize=10)
plt.ylim((-1, 1))
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/R2_coeffs_of_determination.png')
plt.close()

print('\t\tPearson coefficients...')
plt.subplots()
plt.ylabel(r'$\rho$ Pearson coefficient')
plt.bar(0.8, pc_pre, width=BAR_WIDTH, color='b')
plt.bar(1.8, pc_pre_v, width=BAR_WIDTH, color='k')
plt.bar(3.8, pc_pre_a, width=BAR_WIDTH, color='b')
plt.bar(4.8, pc_pre_a_v, width=BAR_WIDTH, color='k')
plt.bar(6.8, pc_pre_b, width=BAR_WIDTH, color='b')
plt.bar(7.8, pc_pre_b_v, width=BAR_WIDTH, color='k')
plt.bar(9.8, pc_pre_c, width=BAR_WIDTH, color='b')
plt.bar(10.8, pc_pre_c_v, width=BAR_WIDTH, color='k')
plt.plot((0, 12), (0, 0), 'k--', linewidth=2)
plt.xticks((1, 2, 4, 5, 7, 8, 10, 11),
           (key_pre, 'PRE*', key_pre_a, 'PRE_A*', key_pre_b, 'PRE_B*', key_pre_c, 'PRE_C*'),
           fontsize=10)
plt.ylim((-1, 1))
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/pearson_coeffs.png')
plt.close()

print('\t\tPearson confidences...')
plt.subplots()
plt.ylabel(r'$(1-p_{\rho})$ Confidence value')
plt.bar(0.8, 1-pcv_pre, width=BAR_WIDTH, color='b')
plt.bar(1.8, 1-pcv_pre_v, width=BAR_WIDTH, color='k')
plt.bar(3.8, 1-pcv_pre_a, width=BAR_WIDTH, color='b')
plt.bar(4.8, 1-pcv_pre_a_v, width=BAR_WIDTH, color='k')
plt.bar(6.8, 1-pcv_pre_b, width=BAR_WIDTH, color='b')
plt.bar(7.8, 1-pcv_pre_b_v, width=BAR_WIDTH, color='k')
plt.bar(9.8, 1-pcv_pre_c, width=BAR_WIDTH, color='b')
plt.bar(10.8, 1-pcv_pre_c_v, width=BAR_WIDTH, color='k')
plt.plot((0, 12), (.95, .95), 'k--', linewidth=2)
plt.xticks((1, 2, 4, 5, 7, 8, 10, 11),
           (key_pre, 'PRE*', key_pre_a, 'PRE_A*', key_pre_b, 'PRE_B*', key_pre_c, 'PRE_C*'),
           fontsize=10)
plt.ylim((0.9, 1))
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/pearson_conf.png')
plt.close()

print('\t\t-Plotting the number of particles...')
for tkey, ltomo in zip(iter(tomos_np.keys()), iter(tomos_np.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.title('Num. particles for ' + tkey_short)
    plt.ylabel('Num. particles')
    plt.xlabel('Classes')
    for lkey, nparts in zip(iter(ltomo.keys()), iter(ltomo.values())):
        if lkey == key_pre:
            i_lkey = 0
        elif lkey == key_pre_a:
            i_lkey = 1
        elif lkey == key_pre_b:
            i_lkey = 2
        elif lkey == key_pre_c:
            i_lkey = 3
        plt.bar(i_lkey, nparts, width=0.75, color=lists_color[lkey], label=lkey)
    plt.xticks(BAR_WIDTH + np.arange(n_lists), np.arange(n_lists))
    plt.legend(loc=1)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/np.png')
    plt.close()

print('\t\t-Plotting densities...')
for tkey, ltomo in zip(iter(tomos_den.keys()), iter(tomos_den.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.title('Density for ' + tkey_short)
    plt.ylabel('Density (np/vol)')
    plt.xlabel('Classes')
    for lkey, den in zip(iter(ltomo.keys()), iter(ltomo.values())):
        if lkey == key_pre:
            i_lkey = 0
        elif lkey == key_pre_a:
            i_lkey = 1
        elif lkey == key_pre_b:
            i_lkey = 2
        elif lkey == key_pre_c:
            i_lkey = 3
        plt.bar(i_lkey, den, width=0.75, color=lists_color[lkey], label=lkey)
    plt.xticks(BAR_WIDTH + np.arange(n_lists), np.arange(n_lists))
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/den.png')
    plt.close()

print('\t\t-Plotting univariate 2nd order analysis...')
high_pvals = dict()
for lkey in (key_pre, key_pre_a, key_pre_b, key_pre_c):
    high_pvals[lkey] = dict.fromkeys(list(tomos_exp.keys()))
for tkey, tomo_exp in zip(iter(tomos_exp.keys()), iter(tomos_exp.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey in (key_pre, key_pre_a, key_pre_b, key_pre_c):
        plt.figure()
        plt.title('Univariate 2nd order for ' + lkey)
        if ana_shell_thick is None:
            plt.ylabel('Ripley\'s L')
        else:
            plt.ylabel('Ripley\'s O')
        plt.xlabel('Distance [nm]')
        ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(tomos_sim[tkey][lkey]))
        try:
            plt.plot(ana_rg, ic_med, color='gray', linewidth=2, linestyle='-', label='RANDOM')
        except ValueError:
            continue
        plt.fill_between(ana_rg, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        if not ana_global:
            arr_exp = np.asarray(tomo_exp[lkey])
        if lkey == key_pre:
            plt.plot(ana_rg, arr_exp, color='k', linewidth=2, linestyle='-', label=key_pre)
        elif lkey == key_pre_a:
            plt.plot(ana_rg, arr_exp, color='blue', linewidth=2, linestyle='-', label=key_pre_a)
        if lkey == key_pre_b:
            plt.plot(ana_rg, arr_exp, color='blue', linewidth=2, linestyle='--', label=key_pre_b)
        elif lkey == key_pre_c:
            plt.plot(ana_rg, arr_exp, color='blue', linewidth=2, linestyle='-.', label=key_pre_c)
        high_pvals[lkey][tkey] = compute_pvals(tomo_exp[lkey], np.asarray(tomos_sim[tkey][lkey])).max()
        plt.xlim(pt_xmin, pt_xmax)
        # plt.legend(loc=4)
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            hold_dir = out_tomos_dir + '/' + tkey_short
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            plt.savefig(hold_dir + '/uni_' + lkey + '.png')
        plt.close()

print('\t\t-Plotting p-values...')
lst, cols = list(), ['side', 'p-value']
plt.figure()
pd_pvals = None
if ana_shell_thick_v is None:
    plt.title('Ripley\'s L')
else:
    plt.title('Ripley\'s O')
for i, lkey in enumerate((key_pre, key_pre_a, key_pre_b, key_pre_c)):
    for tkey in high_pvals[lkey].keys():
        hold_pval = high_pvals[lkey][tkey]
        if (hold_pval is not None) and np.isfinite(hold_pval):
            lst.append([lkey, hold_pval])
pd_pvals = pd.DataFrame(lst, columns=cols)
sns.boxplot(x='side', y='p-value', data=pd_pvals, color='white')
flatui = ['k', 'gray', 'k', 'gray']
sns.set_palette(flatui)
sns.swarmplot(x='side', y='p-value', data=pd_pvals, size=7)
plt.ylim((0.5, 1.1))
x_line = np.linspace(-0.5, 3.5, 10)
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

print('\t\t-Plotting p-values for control vs stimulated synaptosomes...')
lst, cols = list(), ['side', 'p-value']
plt.figure()
pd_pvals = None
if ana_shell_thick_v is None:
    plt.title('Ripley\'s L')
else:
    plt.title('Ripley\'s O')
for i, lkey in enumerate((key_pre, key_pre_a, key_pre_b, key_pre_c)):
    for tkey in high_pvals[lkey].keys():
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        hold_pval = high_pvals[lkey][tkey]
        if (hold_pval is not None) and np.isfinite(hold_pval):
            if tkey_stem in ctrl_stems:
                lst.append([lkey, hold_pval])
            elif tkey_stem in stim_stems:
                lst.append([lkey+'+', hold_pval])
pd_pvals = pd.DataFrame(lst, columns=cols)
sns.boxplot(x='side', y='p-value', data=pd_pvals, color='white')
flatui = ['k', 'gray', 'k', 'gray', 'k', 'gray', 'k', 'gray']
sns.set_palette(flatui)
sns.swarmplot(x='side', y='p-value', data=pd_pvals, size=7)
plt.ylim((0.5, 1.1))
x_line = np.linspace(-0.5, 7.5, 10)
plt.plot(x_line, .95*np.ones(shape=len(x_line)), 'k--', linewidth=2)
plt.grid(True, alpha=0.5)
plt.ylabel('p-value [Ripley\'s L]')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/coloc_ctrl_stim.png', dpi=300)
plt.close()

print('\t\t-Plotting densities by cagegories...')
lst, cols = list(), ['side', 'density']
plt.figure(figsize=[12.6, 4.8])
pd_pvals = None
# if ana_shell_thick_v is None:
#     plt.title('Ripley\'s L')
# else:
#     plt.title('Ripley\'s O')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
for lkey in (key_pre, key_pre_a, key_pre_b, key_pre_c):
    for tkey in tomos_den.keys():
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        den = tomos_den[tkey][lkey]
        if (den is not None) and np.isfinite(den):
            if tkey_stem in ctrl_stems:
                lst.append([lkey, den])
            elif tkey_stem in stim_stems:
                lst.append([lkey+'+', den])
pd_pvals = pd.DataFrame(lst, columns=cols)
sns.boxplot(x='side', y='density', data=pd_pvals, color='white')
flatui = ['k', 'gray', 'k', 'gray', 'k', 'gray', 'k', 'gray']
sns.set_palette(flatui)
sns.swarmplot(x='side', y='density', data=pd_pvals, size=7)
plt.grid(True, alpha=0.5)
plt.ylabel('Density [#/nm$^3$]')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/den_ctrl_stim.png', dpi=300)
plt.close()

print('\tLISTS PLOTTING LOOP: ')

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

print('\t\t-Plotting the number of particles...')
n_tomos = len(list(tomos_hash.keys()))
for lkey, tlist in zip(iter(lists_np.keys()), iter(lists_np.values())):
    plt.figure()
    plt.title('Num. particles for ' + lkey)
    plt.ylabel('Num. particles')
    plt.xlabel('Tomograms')
    for i, nparts in enumerate(tlist.values()):
        plt.bar(i, nparts, width=0.75)
    plt.xticks(BAR_WIDTH + np.arange(n_tomos), np.arange(n_tomos))
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_lists_dir + '/' + lkey
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/np.png')
    plt.close()

print('\t\t-Plotting densities...')
for lkey, tlist in zip(iter(lists_den.keys()), iter(lists_den.values())):
    plt.figure()
    plt.title('Densities for ' + lkey)
    plt.ylabel('Density [np/vol]')
    plt.xlabel('Tomograms')
    for i, den in enumerate(tlist.values()):
        plt.bar(i, den, width=0.75)
    plt.xticks(BAR_WIDTH + np.arange(n_tomos), np.arange(n_tomos))
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_lists_dir + '/' + lkey
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/den.png')
    plt.close()

print('\t\t-Plotting 2nd order metric...')
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    plt.figure()
    # plt.title('Univariate 2nd order for ' + lkey)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Distance [nm]')
    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(lists_sim[lkey]))
    plt.plot(ana_rg, ic_med, color='gray', linewidth=2, linestyle='-', label='RANDOM')
    plt.fill_between(ana_rg, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    if not ana_global:
        tlist = np.asarray(tlist).mean(axis=0)
    if lkey == key_pre:
        plt.plot(ana_rg, tlist, color='k', linewidth=2, linestyle='-', label=key_pre)
    elif lkey == key_pre_a:
        plt.plot(ana_rg, tlist, color='blue', linewidth=2, linestyle='-', label=key_pre_a)
    if lkey == key_pre_b:
        plt.plot(ana_rg, tlist, color='blue', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, tlist, color='blue', linewidth=2, linestyle='-.', label=key_pre_c)
    plt.xlim(pt_xmin, pt_xmax)
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/uni_list_' + lkey + '.png')
    plt.close()

print('\t\t-Plotting clustering p-value...')
flt_pvals = dict()
plt.figure()
# plt.title('Clustering p-value')
plt.ylabel('p-value [Ripley\'s L]')
plt.xlabel('Distance [nm]')
plt.plot(ana_rg, np.ones(shape=ana_rg.shape), color='black', linewidth=1, linestyle='--')
plt.plot(ana_rg, .95*np.ones(shape=ana_rg.shape), color='black', linewidth=1, linestyle='--')
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    sims = np.asarray(lists_sim[lkey])
    if ana_global:
        exp_med = tlist
    else:
        exp_med = np.asarray(tlist).mean(axis=0)
    p_values = compute_pvals(exp_med, sims)
    if (pt_sg_flt_ncoefs is not None) and (pt_sg_flt_ncoefs > 0):
        if pt_sg_flt_ncoefs <= 1:
            p_values = sp.signal.savgol_filter(p_values, pt_sg_flt_ncoefs, 1, mode='interp')
        else:
            p_values = sp.signal.savgol_filter(p_values, pt_sg_flt_ncoefs, 2, mode='interp')
        p_values[p_values > 1] = 1.
        p_values[p_values < 0] = 0.
    flt_pvals[lkey] = p_values
    if lkey == key_pre:
        plt.plot(ana_rg, p_values, color='k', linewidth=2, linestyle='-', label=key_pre)
    elif lkey == key_pre_a:
        plt.plot(ana_rg, p_values, color='blue', linewidth=2, linestyle='-', label=key_pre_a)
    if lkey == key_pre_b:
        plt.plot(ana_rg, p_values, color='blue', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, p_values, color='blue', linewidth=2, linestyle='-.', label=key_pre_c)
plt.xlim(pt_xmin, pt_xmax)
plt.ylim(0, 1.1)
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/pvals_lists.png', dpi=300)
plt.close()

print('\t\t-Plotting clustering filtered p-value...')
plt.figure()
# plt.title('Clustering p-value')
plt.ylabel('p-value [Ripley\'s L]')
plt.xlabel('Distance [nm]')
plt.plot(ana_rg, np.ones(shape=ana_rg.shape), color='black', linewidth=1, linestyle='--')
plt.plot(ana_rg, .95*np.ones(shape=ana_rg.shape), color='black', linewidth=1, linestyle='--')
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    p_values = flt_pvals[lkey]
    if lkey == key_pre:
        plt.plot(ana_rg, p_values, color='k', linewidth=2, linestyle='-', label=key_pre)
    elif lkey == key_pre_a:
        plt.plot(ana_rg, p_values, color='blue', linewidth=2, linestyle='-', label=key_pre_a)
    if lkey == key_pre_b:
        plt.plot(ana_rg, p_values, color='blue', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, p_values, color='blue', linewidth=2, linestyle='-.', label=key_pre_c)
plt.xlim(pt_xmin, pt_xmax)
plt.ylim(0, 1.1)
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/pvals_lists_flt.png', dpi=300)
plt.close()

print('\t\t-Plotting gathered and filtered 2nd order metric...')
plt.figure()
# plt.title('Univariate 2nd order for ' + lkey)
if ana_shell_thick is None:
    plt.ylabel('Ripley\'s L')
else:
    plt.ylabel('Ripley\'s O')
plt.xlabel('Distance [nm]')
flt_unis = dict()
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    if not ana_global:
        tlist = np.asarray(tlist).mean(axis=0)
    if (pt_sg_flt_ncoefs is not None) and (pt_sg_flt_ncoefs > 0):
        if pt_sg_flt_ncoefs <= 1:
            tlist = sp.signal.savgol_filter(tlist, pt_sg_flt_ncoefs, 1, mode='interp')
        else:
            tlist = sp.signal.savgol_filter(tlist, pt_sg_flt_ncoefs, 2, mode='interp')
    flt_unis[lkey] = tlist
    if lkey == key_pre:
        plt.plot(ana_rg, tlist, color='k', linewidth=2, linestyle='-', label=key_pre)
    elif lkey == key_pre_a:
        plt.plot(ana_rg, tlist, color='blue', linewidth=2, linestyle='-', label=key_pre_a)
        x_max, y_max = ana_rg[tlist.argmax()], tlist.max()
    if lkey == key_pre_b:
        plt.plot(ana_rg, tlist, color='blue', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, tlist, color='blue', linewidth=2, linestyle='-.', label=key_pre_c)
# plt.plot((x_max, x_max), (-22, y_max), linewidth=2, marker='o', linestyle='--', color='k')
# plt.plot((pt_xmin, x_max), (y_max, y_max), linewidth=2, marker='o', linestyle='--', color='k')
# plt.xticks((0, 20, 40, x_max, 80))
# plt.ylim(-22, 15)
plt.xlim(pt_xmin, pt_xmax)
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/uni_list_gather_flt.png', dpi=300)
plt.close()

print('\t\t-Plotting gathered 2nd order metric derivative...')
plt.figure()
if ana_shell_thick is None:
    plt.ylabel('Ripley\'s L\'')
else:
    plt.ylabel('Ripley\'s O\'')
plt.xlabel('Distance [nm]')
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    if not ana_global:
        tlist = np.asarray(tlist).mean(axis=0)
    if (pt_sg_flt_ncoefs is not None) and (pt_sg_flt_ncoefs > 0):
        if pt_sg_flt_ncoefs <= 1:
            tlist = sp.signal.savgol_filter(tlist, pt_sg_flt_ncoefs, 1, mode='interp')
        else:
            tlist = sp.signal.savgol_filter(tlist, pt_sg_flt_ncoefs, 2, mode='interp')
    tlist = flt_unis[lkey]
    dtlist = np.gradient(tlist, ana_rg[1]-ana_rg[0])
    if lkey == key_pre:
        plt.plot(ana_rg, dtlist, color='k', linewidth=2, linestyle='-', label=key_pre)
    elif lkey == key_pre_a:
        plt.plot(ana_rg, dtlist, color='blue', linewidth=2, linestyle='-', label=key_pre_a)
    if lkey == key_pre_b:
        plt.plot(ana_rg, dtlist, color='blue', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, dtlist, color='blue', linewidth=2, linestyle='-.', label=key_pre_c)
# plt.ylim(-0.7, 1.5)
plt.xlim(pt_xmin, pt_xmax)
plt.legend(loc=1)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/uni_list_gather_diff.png', dpi=300)
plt.close()

print('\tLISTS (CTRL vs STIM) PLOTTING LOOP: ')

out_cs_dir = out_stem_dir + '/lists_ctrl_vs_stim'
os.makedirs(out_cs_dir)

lists_exp_ctrl, lists_exp_stim = dict(), dict()
lists_sim_ctrl, lists_sim_stim = dict(), dict()
for tkey, tlist in zip(iter(tomos_exp.keys()), iter(tomos_sim.values())):
    for lkey in (key_pre, key_pre_a, key_pre_b, key_pre_c):
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        arr = tomos_exp[tkey][lkey]
        if tkey_stem in ctrl_stems:
            try:
                lists_exp_ctrl[lkey].append(arr)
            except KeyError:
                lists_exp_ctrl[lkey] = list()
                lists_exp_ctrl[lkey].append(arr)
                lists_sim_ctrl[lkey] = list()
            for hold_arr in tomos_sim[tkey][lkey]:
                lists_sim_ctrl[lkey].append(hold_arr)
        elif tkey_stem in stim_stems:
            try:
                lists_exp_stim[lkey].append(arr)
            except KeyError:
                lists_exp_stim[lkey] = list()
                lists_exp_stim[lkey].append(arr)
                lists_sim_stim[lkey] = list()
            for hold_arr in tomos_sim[tkey][lkey]:
                lists_sim_stim[lkey].append(hold_arr)

print('\t\t-Plotting 2nd order metric for control...')
for lkey, tlist in zip(iter(lists_exp_ctrl.keys()), iter(lists_exp_ctrl.values())):
    plt.figure()
    # plt.title('Univariate 2nd order for ' + lkey)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Distance [nm]')
    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(lists_sim_ctrl[lkey]))
    plt.plot(ana_rg, ic_med, color='gray', linewidth=2, linestyle='-', label='RANDOM')
    plt.fill_between(ana_rg, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    if not ana_global:
        tlist = np.asarray(tlist).mean(axis=0)
    if lkey == key_pre:
        plt.plot(ana_rg, tlist, color='k', linewidth=2, linestyle='-', label=key_pre)
    elif lkey == key_pre_a:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='-', label=key_pre_a)
    if lkey == key_pre_b:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='-.', label=key_pre_c)
    plt.xlim(pt_xmin, pt_xmax)
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_cs_dir + '/uni_list_ctrl_' + lkey + '.png', dpi=300)
    plt.close()

print('\t\t-Plotting 2nd order metric for stim...')
for lkey, tlist in zip(iter(lists_exp_stim.keys()), iter(lists_exp_stim.values())):
    plt.figure()
    # plt.title('Univariate 2nd order for ' + lkey)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Distance [nm]')
    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(lists_sim_stim[lkey]))
    plt.plot(ana_rg, ic_med, color='gray', linewidth=2, linestyle='-', label='RANDOM')
    plt.fill_between(ana_rg, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    if not ana_global:
        tlist = np.asarray(tlist).mean(axis=0)
    if lkey == key_pre:
        plt.plot(ana_rg, tlist, color='k', linewidth=2, linestyle='-', label='PST')
    elif lkey == key_pre_a:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='-', label=key_pre_a)
    if lkey == key_pre_b:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='-.', label=key_pre_c)
    plt.xlim(pt_xmin, pt_xmax)
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_cs_dir + '/uni_list_stim_' + lkey + '.png', dpi=300)
    plt.close()

print('\t\t-Plotting clustering p-value for control (filtered)...')
plt.figure()
# plt.title('Clustering p-value')
plt.ylabel('p-value [Ripley\'s L]')
plt.xlabel('Distance [nm]')
plt.plot(ana_rg, np.ones(shape=ana_rg.shape), color='black', linewidth=1, linestyle='--')
plt.plot(ana_rg, .95*np.ones(shape=ana_rg.shape), color='black', linewidth=1, linestyle='--')
for lkey, tlist in zip(iter(lists_exp_ctrl.keys()), iter(lists_exp_ctrl.values())):
    sims = np.asarray(lists_sim_ctrl[lkey])
    if ana_global:
        exp_med = tlist
    else:
        exp_med = np.asarray(tlist).mean(axis=0)
    p_values = compute_pvals(exp_med, sims)
    if (pt_sg_flt_ncoefs is not None) and (pt_sg_flt_ncoefs > 0):
        if pt_sg_flt_ncoefs <= 1:
            p_values = sp.signal.savgol_filter(p_values, pt_sg_flt_ncoefs, 1, mode='interp')
        else:
            p_values = sp.signal.savgol_filter(p_values, pt_sg_flt_ncoefs, 2, mode='interp')
        p_values[p_values > 1] = 1.
        p_values[p_values < 0] = 0.
    if lkey == key_pre:
        plt.plot(ana_rg, p_values, color='k', linewidth=2, linestyle='-', label=key_pre)
    elif lkey == key_pre_a:
        plt.plot(ana_rg, p_values, color='b', linewidth=2, linestyle='-', label=key_pre_a)
    if lkey == key_pre_b:
        plt.plot(ana_rg, p_values, color='b', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, p_values, color='b', linewidth=2, linestyle='-.', label=key_pre_c)
plt.xlim(pt_xmin, pt_xmax)
plt.ylim(0, 1.1)
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_cs_dir + '/pvals_lists_ctrl.png', dpi=300)
plt.close()

print('\t\t-Plotting clustering p-value for stim (filtered)...')
plt.figure()
# plt.title('Clustering p-value')
plt.ylabel('p-value [Ripley\'s L]')
plt.xlabel('Distance [nm]')
plt.plot(ana_rg, np.ones(shape=ana_rg.shape), color='black', linewidth=1, linestyle='--')
plt.plot(ana_rg, .95*np.ones(shape=ana_rg.shape), color='black', linewidth=1, linestyle='--')
for lkey, tlist in zip(iter(lists_exp_stim.keys()), iter(lists_exp_stim.values())):
    sims = np.asarray(lists_sim_stim[lkey])
    if ana_global:
        exp_med = tlist
    else:
        exp_med = np.asarray(tlist).mean(axis=0)
    p_values = compute_pvals(exp_med, sims)
    if (pt_sg_flt_ncoefs is not None) and (pt_sg_flt_ncoefs > 0):
        if pt_sg_flt_ncoefs <= 1:
            p_values = sp.signal.savgol_filter(p_values, pt_sg_flt_ncoefs, 1, mode='interp')
        else:
            p_values = sp.signal.savgol_filter(p_values, pt_sg_flt_ncoefs, 2, mode='interp')
        p_values[p_values > 1] = 1.
        p_values[p_values < 0] = 0.
    if lkey == key_pre:
        plt.plot(ana_rg, p_values, color='k', linewidth=2, linestyle='-', label='PST')
    elif lkey == key_pre_a:
        plt.plot(ana_rg, p_values, color='b', linewidth=2, linestyle='-', label=key_pre_a)
    if lkey == key_pre_b:
        plt.plot(ana_rg, p_values, color='b', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, p_values, color='b', linewidth=2, linestyle='-.', label=key_pre_c)
plt.xlim(pt_xmin, pt_xmax)
plt.ylim(0, 1.1)
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_cs_dir + '/pvals_lists_stim.png', dpi=300)
plt.close()

print('\t\t-Plotting gathered and filtered 2nd order metric for control...')
plt.figure()
# plt.title('Univariate 2nd order for ' + lkey)
if ana_shell_thick is None:
    plt.ylabel('Ripley\'s L')
else:
    plt.ylabel('Ripley\'s O')
plt.xlabel('Distance [nm]')
flt_unis = dict()
for lkey, tlist in zip(iter(lists_exp_ctrl.keys()), iter(lists_exp_ctrl.values())):
    if not ana_global:
        tlist = np.asarray(tlist).mean(axis=0)
    if (pt_sg_flt_ncoefs is not None) and (pt_sg_flt_ncoefs > 0):
        if pt_sg_flt_ncoefs <= 1:
            tlist = sp.signal.savgol_filter(tlist, pt_sg_flt_ncoefs, 1, mode='interp')
        else:
            tlist = sp.signal.savgol_filter(tlist, pt_sg_flt_ncoefs, 2, mode='interp')
    flt_unis[lkey] = tlist
    if lkey == key_pre:
        plt.plot(ana_rg, tlist, color='k', linewidth=2, linestyle='-', label='PST')
    elif lkey == key_pre_a:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='-', label=key_pre_a)
        x_max, y_max = ana_rg[tlist.argmax()], tlist.max()
    if lkey == key_pre_b:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='-.', label=key_pre_c)
# plt.plot((x_max, x_max), (-22, y_max), linewidth=2, marker='o', linestyle='--', color='k')
# plt.plot((pt_xmin, x_max), (y_max, y_max), linewidth=2, marker='o', linestyle='--', color='k')
# plt.xticks((0, 20, 40, x_max, 80))
# plt.ylim(-22, 15)
plt.xlim(pt_xmin, pt_xmax)
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_cs_dir + '/uni_list_gather_ctrl_flt.png', dpi=300)
plt.close()

print('\t\t-Plotting gathered and filtered 2nd order metric for stim...')
plt.figure()
# plt.title('Univariate 2nd order for ' + lkey)
if ana_shell_thick is None:
    plt.ylabel('Ripley\'s L')
else:
    plt.ylabel('Ripley\'s O')
plt.xlabel('Distance [nm]')
flt_unis = dict()
for lkey, tlist in zip(iter(lists_exp_stim.keys()), iter(lists_exp_stim.values())):
    if not ana_global:
        tlist = np.asarray(tlist).mean(axis=0)
    if (pt_sg_flt_ncoefs is not None) and (pt_sg_flt_ncoefs > 0):
        if pt_sg_flt_ncoefs <= 1:
            tlist = sp.signal.savgol_filter(tlist, pt_sg_flt_ncoefs, 1, mode='interp')
        else:
            tlist = sp.signal.savgol_filter(tlist, pt_sg_flt_ncoefs, 2, mode='interp')
    flt_unis[lkey] = tlist
    if lkey == key_pre:
        plt.plot(ana_rg, tlist, color='k', linewidth=2, linestyle='-', label=key_pre)
    elif lkey == key_pre_a:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='-', label=key_pre_a)
        x_max, y_max = ana_rg[tlist.argmax()], tlist.max()
    if lkey == key_pre_b:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='--', label=key_pre_b)
    elif lkey == key_pre_c:
        plt.plot(ana_rg, tlist, color='b', linewidth=2, linestyle='-.', label=key_pre_c)
# plt.plot((x_max, x_max), (-22, y_max), linewidth=2, marker='o', linestyle='--', color='k')
# plt.plot((pt_xmin, x_max), (y_max, y_max), linewidth=2, marker='o', linestyle='--', color='k')
# plt.xticks((0, 20, 40, x_max, 80))
# plt.ylim(-22, 15)
plt.xlim(pt_xmin, pt_xmax)
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_cs_dir + '/uni_list_gather_stim_flt.png', dpi=300)
plt.close()

print('Successfully terminated. (' + time.strftime("%c") + ')')
