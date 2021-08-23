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
try:
    import pickle as pickle
except ImportError:
    import pickle

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

BAR_WIDTH = .35

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst/ampar_vs_nmdar/org'

# Input STAR file
in_star = ROOT_PATH + '/ltomos/ltomos_ampar_nmdar_premb_mask_test/ampar_nmdar_premb_gather_mask_test_ltomos.star'
in_tethers_csv = ROOT_PATH + '/../../../pre/ref_nomb_1_clean/py_scripts/syn_num_tethers.csv'
in_wspace = ROOT_PATH + '/uni_sph_test/uni_5_200_5_sim_5_local_min_5_wspace.pkl' # (Insert a path to recover a pickled workspace instead of doing a new computation)
in_wspace_bi = ROOT_PATH + '/uni_sph_test/uni_5_200_5_sim_5_local_min_5_wspace_bi.pkl'

# Output directory
out_dir = ROOT_PATH + '/uni_sph_test'
out_stem = 'uni_5_200_5_sim_5_local_min_5' # ''uni_sph_4_60_5'

# Pre-processing variables
pre_ssup = 8 #nm
pre_min_parts = 5

# Analysis variables
ana_res = 0.684 # nm/voxel
ana_rg = np.arange(5, 200, 5) # np.arange(4, 100, 2) # in nm
ana_global = False # if False all tomograms are treated separatedly for computing the results by lists
ana_shell_thick = None #
ana_border = True
ana_conv_iter = 100
ana_max_iter = 100000
ana_npr = 1 # None means Auto
ana_npr_model = 1 # In general equal to ana_npr unless you have memory problems

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = 5
p_per = 5 # %
p_max_den = 0.000025
# Particle surface
p_vtp = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre/vtps/sph_rad_5_surf.vtp'

# Figure saving options
fig_fmt = '.png' # if None they showed instead

# Plotting options
pt_sim_v = True
pt_cmap = plt.get_cmap('gist_rainbow')
pt_xmin, pt_xmax = 5, 200

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
print('\tInput STAR file: ' + str(in_star))
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
    print('\t\t-Convergence number of samples for stochastic volume estimations: ' + str(ana_conv_iter))
    print('\t\t-Maximum number of samples for stochastic volume estimations: ' + str(ana_max_iter))
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
print('')

######### Process

print('Main Routine: ')
mats_lists, gl_lists = None, None

out_stem_dir = out_dir + '/' + out_stem
print('\tCleaning the output dir: ' + out_stem)
if os.path.exists(out_stem_dir):
    clean_dir(out_stem_dir)
else:
    os.makedirs(out_stem_dir)

print('\tLoading input data...')
star = sub.Star()
try:
    star.load(in_star)
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
set_lists = surf.SetListTomoParticles()
for row in range(star.get_nrows()):
    ltomos_pkl = star.get_element('_psPickleFile', row)
    ltomos = unpickle_obj(ltomos_pkl)
    set_lists.add_list_tomos(ltomos, ltomos_pkl)
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
    fkey = os.path.split(lkey)[1]
    print('\t\t-Processing list: ' + fkey)
    short_key_idx = fkey.index('_')
    short_key = fkey[:short_key_idx]
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
    fkey = os.path.split(lkey)[1]
    print('\t\t-Processing list: ' + fkey)
    short_key_idx = fkey.index('_')
    short_key = fkey[:short_key_idx]
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
vols = lists_dic['0'].get_volumes_dict()
with open(in_tethers_csv, mode='r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    vesicles = dict()
    for row in reader:
        vesicles[row[0]] = float(row[1])

if in_wspace is None:

    print('\tLIST COMPUTING LOOP:')
    for li, lkey in enumerate(lists_hash.values()):

        llist = lists_dic[lkey]
        print('\t\t-Processing list: ' + lkey)
        print('\t\t\t+Computing global density...')
        lists_gden[lkey] = llist.compute_global_density()
        print('\t\t\t+Tomograms computing loop:')
        for lt, tkey in enumerate(tomos_hash.keys()):

            print('\t\t\t\t*Processing tomogram (list ' + str(li+1) + ' of ' + str(len(list(lists_hash.values()))) + ', tomo ' + str(lt+1) + ' of ' + str(len(list(tomos_hash.keys()))) + ') : ' + os.path.split(tkey)[1])
            ltomo = llist.get_tomo_by_key(tkey)

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
                                                                     out_sep=2, npr=ana_npr, verbose=False)
                if (hold_arr_1 is not None) and (hold_arr_2 is not None):
                    tomos_exp[tkey][lkey] = (hold_arr_1, hold_arr_2)
                    for npi in range(tomos_np[tkey][lkey]):
                        lists_exp[lkey].append((hold_arr_1, hold_arr_2))
            else:
                hold_arr = ltomo.compute_uni_2nd_order(ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                       conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                       out_sep=0, npr=ana_npr, verbose=False)
                if hold_arr is not None:
                    tomos_exp[tkey][lkey] = hold_arr
                    for npi in range(tomos_np[tkey][lkey]):
                        lists_exp[lkey].append(hold_arr)

            if ana_global:
                print('\t\t\t\t\t-Simulating univariate second order metrics...')
                hold_arr_1, hold_arr_2 = ltomo.simulate_uni_2nd_order(p_nsims, ModelCSRV, part_vtp,
                                                                      ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                                      conv_iter=ana_conv_iter, max_iter=ana_max_iter, out_sep=2,
                                                                      npr=ana_npr, npr_model=ana_npr_model, tmp_folder=tmp_sim_folder,
                                                                      verbose=pt_sim_v)
                for npi in range(tomos_np[tkey][lkey]):
                    if (hold_arr_1 is not None) and (hold_arr_2 is not None):
                        for arr_1, arr_2 in zip(hold_arr_1, hold_arr_2):
                            tomos_sim[tkey][lkey].append((arr_1, arr_2))
                            lists_sim[lkey][tkey].append((arr_1, arr_2))
            else:
                print('\t\t\t\t\t-Simulating univariate second order metrics...')
                hold_arr = ltomo.simulate_uni_2nd_order(p_nsims, ModelCSRV, part_vtp,
                                                        ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                        conv_iter=ana_conv_iter, max_iter=ana_max_iter, out_sep=0,
                                                        npr=ana_npr, npr_model=ana_npr_model, tmp_folder=tmp_sim_folder,
                                                        verbose=pt_sim_v)
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
              lists_np, lists_den, lists_gden, lists_exp, lists_sim, lists_color)
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

lkey_0, lkey_1 = lists_hash[0], lists_hash[1]
if in_wspace_bi is None:

    print('\tBI-VARIATE FOR THE FIRST TWO LISTS LOOP:')

    list_exp_0, list_exp_1 = dict(), dict()
    llist_0, llist_1 = lists_dic[lkey_0], lists_dic[lkey_1]
    print('\t\t+Tomograms computing loop:')
    for lt, tkey in enumerate(tomos_hash.keys()):

        print('\t\t\t\t*Processing tomogram (tomo ' + str(lt + 1) + ' of ' + \
              str(len(list(tomos_hash.keys()))) + ') : ' + os.path.split(tkey)[1])
        ltomo_0, ltomo_1 = llist_0.get_tomo_by_key(tkey), llist_1.get_tomo_by_key(tkey)
        list_exp_0[tkey], list_exp_1[tkey] = list(), list()

        if ana_global:
            hold_arr_1, hold_arr_2 = ltomo_0.compute_bi_2nd_order(ltomo_1, ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                                  conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                                  out_sep=2, npr=ana_npr, verbose=False)
            for npi in range(tomos_np[tkey][lkey_0]):
                if (hold_arr_1 is not None) and (hold_arr_2 is not None):
                    list_exp_0[tkey].append((hold_arr_1, hold_arr_2))
        else:
            hold_arr = ltomo_0.compute_bi_2nd_order(ltomo_1, ana_rg_v, thick=ana_shell_thick_v,
                                                    border=ana_border,
                                                    conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                    out_sep=0, npr=ana_npr, verbose=False)
            for npi in range(tomos_np[tkey][lkey_0]):
                if hold_arr is not None:
                    list_exp_0[tkey].append(hold_arr)

        if ana_global:
            hold_arr_1, hold_arr_2 = ltomo_1.compute_bi_2nd_order(ltomo_0, ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                                  conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                                  out_sep=2, npr=ana_npr, verbose=False)
            for npi in range(tomos_np[tkey][lkey_1]):
                if (hold_arr_1 is not None) and (hold_arr_2 is not None):
                    list_exp_1[tkey].append((hold_arr_1, hold_arr_2))
        else:
            hold_arr = ltomo_1.compute_bi_2nd_order(ltomo_0, ana_rg_v, thick=ana_shell_thick_v,
                                                    border=ana_border,
                                                    conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                    out_sep=0, npr=ana_npr, verbose=False)
            for npi in range(tomos_np[tkey][lkey_1]):
                if hold_arr is not None:
                    list_exp_1[tkey].append(hold_arr)

    print('\tGlobal computations by tomos...')
    hold_list_exp_0, hold_list_exp_1 = copy.deepcopy(list_exp_0), copy.deepcopy(list_exp_1)
    list_exp_0, list_exp_1 = dict(), dict()
    dens_0, dens_1 = lists_gden[lkey_0], lists_gden[lkey_1]
    for tkey in hold_list_exp_0.keys():
        if ana_global:
            dens_0 = lists_gden[lkey_0]
            arr_1, arr_2 = list(), list()
            for hold_arr in hold_list_exp_0[tkey]:
                arr_1.append(hold_arr[0])
                arr_2.append(hold_arr[1])
            arr_1, arr_2 = np.asarray(arr_1), np.asarray(arr_2)
            if ana_shell_thick is None:
                gl_arr = ana_rg * (np.cbrt((1. / dens_0) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))) - 1.)
            else:
                gl_arr = (1. / dens_0) * (arr_1.sum(axis=0) / arr_2.sum(axis=0)) - 1.
        else:
            dens_0 = tomos_den[tkey][lkey_0]
            gl_arr = hold_list_exp_0[tkey]
        list_exp_0[tkey] = gl_arr
    for tkey in hold_list_exp_1.keys():
        if ana_global:
            dens_1 = lists_gden[lkey_1]
            arr_1, arr_2 = list(), list()
            for hold_arr in hold_list_exp_1[tkey]:
                arr_1.append(hold_arr[0])
                arr_2.append(hold_arr[1])
            arr_1, arr_2 = np.asarray(arr_1), np.asarray(arr_2)
            if ana_shell_thick is None:
                gl_arr = ana_rg * (np.cbrt((1. / dens_1) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))) - 1.)
            else:
                gl_arr = (1. / dens_1) * (arr_1.sum(axis=0) / arr_2.sum(axis=0)) - 1.
        else:
            dens_1 = tomos_den[tkey][lkey_1]
            gl_arr = hold_list_exp_1[tkey]
        list_exp_1[tkey] = gl_arr

    out_wspace_bi = out_dir + '/' + out_stem + '_wspace_bi.pkl'
    print('\tPickling computation workspace in: ' + out_wspace_bi)
    wspace_bi = (list_exp_0, list_exp_1)
    with open(out_wspace_bi, "wb") as fl:
        pickle.dump(wspace_bi, fl)
        fl.close()

else:
    print('\tLoading the bi-variate workspace: ' + in_wspace_bi)
    with open(in_wspace_bi, 'r') as pkl:
        wspace_bi = pickle.load(pkl)
    list_exp_0, list_exp_1 = wspace_bi[0], wspace_bi[1]

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

print('\t\t-Plotting the number of particles...')
for tkey, ltomo in zip(iter(tomos_np.keys()), iter(tomos_np.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.title('Num. particles for ' + tkey_short)
    plt.ylabel('Num. particles')
    plt.xlabel('Classes')
    for lkey, nparts in zip(iter(ltomo.keys()), iter(ltomo.values())):
        plt.bar(int(lkey), nparts, width=0.75, color=lists_color[lkey], label=lkey)
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
        plt.bar(int(lkey), den, width=0.75, color=lists_color[lkey], label=lkey)
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

print('\t\t-Plotting densities by tethered vesicles...')
plt.figure()
# plt.title('Colocalization respect ' + key_ref)
plt.ylabel('Density [AMPAR/nm$^3$]')
plt.xlabel('Number of tethered vesicles')
l_ves, l_den_0, l_den_1, l_wden_0, l_wden_1 = list(), list(), list(), list(), list()
for tkey in tomos_den.keys():
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    if vesicles[tkey_short] > 0:
        l_ves.append(vesicles[tkey_short])
        l_den_0.append(tomos_den[tkey]['0'])
        if tomos_den[tkey]['0'] < p_max_den:
            l_wden_0.append(1.)
        else:
            l_wden_0.append(0.)
        l_den_1.append(tomos_den[tkey]['1'])
        if tomos_den[tkey]['1'] < p_max_den:
            l_wden_1.append(1.)
        else:
            l_wden_1.append(0.)
l_ves, l_den_0, l_den_1, l_wden_0, l_wden_1 = np.asarray(l_ves, dtype=np.float).reshape(-1, 1), \
                                              np.asarray(l_den_0, dtype=np.float).reshape(-1, 1), \
                                              np.asarray(l_den_1, dtype=np.float).reshape(-1, 1), \
                                              np.asarray(l_wden_0, dtype=np.float), \
                                              np.asarray(l_wden_1, dtype=np.float)
l_wden_0 /= l_wden_0.sum()
l_wden_1 /= l_wden_1.sum()
regr_0, regr_1 = linear_model.LinearRegression(), linear_model.LinearRegression()
regr_0.fit(l_ves, l_den_0, sample_weight=l_wden_0), regr_1.fit(l_ves, l_den_1, sample_weight=l_wden_1)
l_den_0_r, l_den_1_r = regr_0.predict(l_ves), regr_1.predict(l_ves)
plt.plot(l_ves, l_den_0, color='black', marker='s', markersize=10, label='AMPAR', linestyle='')
plt.plot(l_ves, l_den_1, color='red', marker='*', markersize=10, label='NMDAR', linestyle='')
plt.plot(l_ves, l_den_0_r, color='black', label='AMPAR-LR', linestyle='-', linewidth=2.0)
plt.plot(l_ves, l_den_1_r, color='red', label='NMDAR-LR', linestyle='-', linewidth=2.0)
plt.xlim((0, l_ves.max()*1.1))
plt.ylim((0, p_max_den*1.1))
plt.grid(True, alpha=0.5)
plt.legend(loc=3)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/den_by_ntet.png', dpi=600)
plt.close()
print('\t\t\t+Linear regression:')
print('\t\t\t\t-Variance score AMPAR: ' + str(r2_score(l_den_0, l_den_0_r)))
print('\t\t\t\t-Variance score NMDAR: ' + str(r2_score(l_den_1, l_den_1_r)))

print('\t\t-Plotting densities by pre-syanptic membrane volume...')
plt.figure()
# plt.title('Colocalization respect ' + key_ref)
plt.ylabel('Density [#/nm$^3$]')
plt.xlabel('Pre-synaptic membrane volume [nm$^3$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
l_vols, l_np_0, l_np_1, l_wden_0, l_wden_1 = list(), list(), list(), list(), list()
for tkey in tomos_den.keys():
    l_vols.append(vols[tkey])
    l_np_0.append(tomos_den[tkey]['0'])
    if tomos_den[tkey]['0'] < p_max_den:
        l_wden_0.append(1.)
    else:
        l_wden_0.append(0.)
    l_np_1.append(tomos_den[tkey]['1'])
    if tomos_den[tkey]['1'] < p_max_den:
        l_wden_1.append(1.)
    else:
        l_wden_1.append(0.)
l_vols, l_np_0, l_np_1, l_wden_0, l_wden_1 = np.asarray(l_vols, dtype=np.float).reshape(-1, 1), \
                                             np.asarray(l_np_0, dtype=np.float).reshape(-1, 1), \
                                             np.asarray(l_np_1, dtype=np.float).reshape(-1, 1), \
                                             np.asarray(l_wden_0, dtype=np.float), \
                                             np.asarray(l_wden_1, dtype=np.float)
l_wden_0 /= l_wden_0.sum()
l_wden_1 /= l_wden_1.sum()
regr_0, regr_1 = linear_model.LinearRegression(), linear_model.LinearRegression()
regr_0.fit(l_vols, l_np_0, sample_weight=l_wden_0), regr_1.fit(l_vols, l_np_1, sample_weight=l_wden_1)
l_np_0_r, l_np_1_r = regr_0.predict(l_vols), regr_1.predict(l_vols)
plt.plot(l_vols, l_np_0, color='black', marker='s', markersize=10, label='AMPAR', linestyle='')
plt.plot(l_vols, l_np_1, color='red', marker='*', markersize=10, label='NMDAR', linestyle='')
plt.plot(l_vols, l_np_0_r, color='black', label='AMPAR-LR', linestyle='-', linewidth=2.0)
plt.plot(l_vols, l_np_1_r, color='red', label='NMDAR-LR', linestyle='-', linewidth=2.0)
plt.xlim((-0.1, l_vols.max()*1.1))
plt.ylim((0, p_max_den*1.1))
plt.grid(True, alpha=0.5)
plt.legend(loc=3)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/den_by_vol.png', dpi=600)
plt.close()
print('\t\t\t+Linear regression:')
print('\t\t\t\t-Variance score AMPAR: ' + str(r2_score(l_np_0, l_np_0_r)))
print('\t\t\t\t-Variance score NMDAR: ' + str(r2_score(l_np_1, l_np_1_r)))

print('\t\t-Plotting 2nd order metric...')
for tkey, ltomo in zip(iter(tomos_exp.keys()), iter(tomos_exp.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.title('Univariate 2nd order for ' + tkey_short)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Scale (nm)')
    for lkey, arr in zip(iter(tomos_exp[tkey].keys()), iter(tomos_exp[tkey].values())):
        if lkey == lkey_0:
            plt.plot(ana_rg, arr, color='black', linewidth=2, label='AMPAR$\Rightarrow$AMPAR')
            plt.plot(ana_rg, np.asarray(list_exp_1[tkey]).mean(axis=0), color='black', linewidth=2, linestyle='--',
                     label='AMPAR$\Rightarrow$NMDAR')
        elif lkey == lkey_1:
            plt.plot(ana_rg, arr, color='red', linewidth=2, label='NMDAR$\Rightarrow$NMDAR')
            plt.plot(ana_rg, np.asarray(list_exp_0[tkey]).mean(axis=0), color='red', linewidth=2, linestyle='--',
                     label='NMDAR$\Rightarrow$AMPAR')
    #     if lkey == lkey_0:
    #         plt.plot(ana_rg, np.asarray(list_exp_0[tkey]).mean(axis=0), color='black', label=lkey, linestyle='--')
    #     elif lkey == lkey_1:
    #         plt.plot(ana_rg, np.asarray(list_exp_1[tkey]).mean(axis=0), color='red', label=lkey, linestyle='--')
    plt.legend(loc=4)
    try:
        hold_sim = tomos_sim[tkey][lkey]
        if len(hold_sim) > 0:
            ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(hold_sim))
            # plt.plot(ana_rg, ic_low, 'k--')
            plt.fill_between(ana_rg, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
            plt.plot(ana_rg, ic_med, 'gray', linewidth=2)
            # plt.plot(ana_rg, ic_high, 'k--')
    except IndexError:
        print('\t\t\t+WARNING: no simulations for tomogram: ' + tkey)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/uni.png')
    plt.close()

    plt.figure()
    plt.title('Clustering p-value for ' + tkey_short)
    plt.ylabel('p-value')
    plt.xlabel('Scale (nm)')
    try:
        hold_sim = tomos_sim[tkey][lkey]
        if len(hold_sim) > 0:
            for lkey, arr in zip(iter(tomos_exp[tkey].keys()), iter(tomos_exp[tkey].values())):
                p_values = compute_pvals(arr, np.asarray(hold_sim))
                plt.plot(ana_rg, p_values, color=lists_color[lkey], linestyle='-', label=lkey)
    except IndexError:
        print('\t\t\t+WARNING: no simulations for tomogram: ' + tkey)
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/pvals.png')
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
sims = list()
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    plt.figure()
    # plt.title('Univariate 2nd order for ' + lkey)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Distance [nm]')
    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(lists_sim[lkey]))
    plt.plot(ana_rg, ic_med, color='gray', linewidth=2, linestyle='-', label='RANDOM$\Rightarrow$RANDOM')
    plt.fill_between(ana_rg, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    if ana_global:
        if lkey == lkey_0:
            plt.plot(ana_rg, tlist, color='black', linewidth=2, linestyle='-', label='AMPAR$\Rightarrow$AMPAR')
        elif lkey == lkey_1:
            plt.plot(ana_rg, tlist, color='red', linewidth=2, linestyle='-', label='NMDAR$\Rightarrow$NMDAR')
    else:
        if lkey == lkey_0:
            plt.plot(ana_rg, np.asarray(tlist).mean(axis=0), color='black', linestyle='-', linewidth=2, label='AMPAR$\Rightarrow$AMPAR')
        if lkey == lkey_1:
            plt.plot(ana_rg, np.asarray(tlist).mean(axis=0), color='red', linestyle='-', linewidth=2, label='NMDAR$\Rightarrow$NMDAR')
    if ana_global:
        arr = list()
        if lkey == lkey_0:
            for mat in list_exp_1.values():
                for hold_arr in mat:
                    for npi in range(tomos_np[tkey][lkey_0]):
                        arr.append(hold_arr)
            plt.plot(ana_rg, np.asarray(arr).mean(axis=0), color='black', linewidth=2, linestyle='--', label='AMPAR$\Rightarrow$NMDAR')
        elif lkey == lkey_1:
            for mat in list_exp_0.values():
                for hold_arr in mat:
                    for npi in range(tomos_np[tkey][lkey_1]):
                        arr.append(hold_arr)
            plt.plot(ana_rg, np.asarray(arr).mean(axis=0), color='red', linewidth=2, linestyle='--', label='NMDAR$\Rightarrow$AMPAR')
    else:
        if lkey == lkey_0:
            arr = list()
            for mat in list_exp_1.values():
                for hold_arr in mat:
                    for npi in range(tomos_np[tkey][lkey_0]):
                        arr.append(hold_arr)
            plt.plot(ana_rg, np.asarray(arr).mean(axis=0), color='black', linewidth=2, linestyle='--',
                             label='AMPAR$\Rightarrow$NMDAR')
        elif lkey == lkey_1:
            arr = list()
            for mat in list_exp_0.values():
                for hold_arr in mat:
                    for npi in range(tomos_np[tkey][lkey_1]):
                        arr.append(hold_arr)
            plt.plot(ana_rg, np.asarray(arr).mean(axis=0), color='red', linewidth=2, linestyle='--',
                             label='NMDAR$\Rightarrow$AMPAR')
    plt.xlim(pt_xmin, pt_xmax)
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/uni_list_' + lkey + '.png')
    plt.close()
sims = np.asarray(hold_sim)

print('\t\t-Plotting clustering p-value...')
plt.figure()
# plt.title('Clustering p-value')
plt.ylabel('p-value')
plt.xlabel('Distance [nm]')
plt.plot(ana_rg, np.ones(shape=ana_rg.shape), color='black', linewidth=1, linestyle='--')
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    sims = np.asarray(lists_sim[lkey])
    if ana_global:
        exp_med = tlist
    else:
        exp_med = np.asarray(tlist).mean(axis=0)
    p_values = compute_pvals(exp_med, sims)
    if lkey == lkey_0:
        plt.plot(ana_rg, p_values, color='black', linewidth=2, linestyle='-', label='AMPAR$\Rightarrow$RANDOM')
    elif lkey == lkey_1:
        plt.plot(ana_rg, p_values, color='red', linewidth=2, linestyle='-', label='NMDAR$\Rightarrow$RANDOM')
plt.xlim(pt_xmin, pt_xmax)
plt.ylim(0, 1.1)
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/pvals_lists.png')
plt.close()

print('Successfully terminated. (' + time.strftime("%c") + ')')
