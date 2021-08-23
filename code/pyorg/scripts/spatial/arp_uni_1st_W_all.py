"""

    Performs Univariate 1st order analysis to a SetListTomograms object by tomograms

    Input:  - A STAR file with a set of ListTomoParticles pickles (SetListTomograms object input)

    Output: - Plots by tomograms:
                + Plots by list:
                    * Univariate 1st order analysis against random simulation
            - Global plots:
                + Global univariate 1st order analysis against random simulation by list

"""

################# Package import

import os
import pickle
import numpy as np
import scipy as sp
import sys
import time
import multiprocessing as mp
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg.globals import unpickle_obj, clean_dir
from pyorg.surf.model import ModelCSRV, gen_tlist
from pyorg.spatial.sparse import compute_cdf, compute_hist, compute_J
from matplotlib import pyplot as plt, rcParams
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

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/tomograms/marion/Arp23complex'

# Input STAR file
in_star = ROOT_PATH + '/ltomos_all/W_all_mask/W_all_mask_ltomos.star'
in_wspace = ROOT_PATH + '/uni_sph_1st_all_W/W_all_sim20_wspace.pkl' # (Insert a path to recover a pickled workspace instead of doing a new computation)

# Output directory
out_dir = ROOT_PATH + '/uni_sph_1st_all_W'
out_stem = 'W_all_sim20_plot' # ''uni_sph_4_60_5'

# Analysis variables
ana_res = 1.684 # 0.684 # nm/voxel
ana_nbins = 8 # 20 # 100
ana_rmax = 160 # 60 # nm
ana_f_npoints = 1000

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = 20
p_per = 5 # %

#### Advanced settings

ana_npr_model = 1 # Parallel version doesn't work properly for 1st order analysis yet!!!

# List pre-processing options
pr_ss = None # 10 # nm

p_jhigh = 0.98 # 0.999 # 0.98

# Figure saving options
fig_fmt = '.png' # if None they showed instead

# Plotting options
pt_sim_v = True
pt_cmap = plt.get_cmap('gist_rainbow')

# Particle surface
p_vtp = ROOT_PATH + '/in/iter19_apix_16.84_center.vtp'

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
    p_vals_low, p_vals_high = np.zeros(shape=exp_med.shape, dtype=np.float32), \
                              np.zeros(shape=exp_med.shape, dtype=np.float32)
    for i, exp in enumerate(exp_med):
        sim_slice = sims[:, i]
        p_vals_high[i] = float((exp > sim_slice).sum()) / n_sims
        p_vals_low[i] = float((exp < sim_slice).sum()) / n_sims
    return p_vals_low, p_vals_high

########## Print initial message

print('Univariate first order analysis for a ListTomoParticles by tomograms.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tInput STAR file: ' + str(in_star))
if in_wspace is not None:
    print('\tLoad workspace from: ' + in_wspace)
print('\tList pre-processing options: ')
if pr_ss is not None:
    pr_ss_v = pr_ss / ana_res
    print('\t\t-Scale suppression: ' + str(pr_ss) + ' nm (' + str(pr_ss_v) + ' voxels)')
print('\tOrganization analysis settings: ')
print('\t\t-Number of bins: ' + str(ana_nbins))
ana_rmax_v = float(ana_rmax) / ana_res
print('\t\t-Maximum radius: ' + str(ana_rmax) + ' nm (' + str(ana_rmax_v) + ' voxels)')
print('\t\t-Number of CSR points for F-function: ' + str(ana_f_npoints))
if ana_npr_model is None:
    ana_npr_model = mp.cpu_count()
elif ana_npr_model > p_nsims:
    ana_npr_model = p_nsims
print('\t\t-Number of processors for models simulation: ' + str(ana_npr_model))
print('\tP-Value computation setting:')
print('\t\t-Percentile: ' + str(p_per) + ' %')
print('\t\t-Highest value for Funciton-J: ' + str(p_jhigh))
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

if in_wspace is None:

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

    if pr_ss is not None:
        print('\tApplying scale suppression...')
        set_lists.scale_suppression(pr_ss_v)

    print('\tBuilding the dictionaries...')
    lists_count, tomos_count = 0, 0
    lists_dic = dict()
    lists_hash, tomos_hash = dict(), dict()
    tomos_exp_dsts, tomos_sim_dsts, tomos_exp_fdsts, tomos_sim_fdsts = dict(), dict(), dict(), dict()
    lists_exp_dsts, lists_sim_dsts, lists_exp_fdsts, lists_sim_fdsts, lists_color = dict(), dict(), dict(), dict(), dict()
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
            lists_exp_dsts[short_key], lists_sim_dsts[short_key] = list(), list()
            lists_exp_fdsts[short_key], lists_sim_fdsts[short_key] = list(), list()
            lists_count += 1
    for lkey, llist in zip(iter(set_lists_dic.keys()), iter(set_lists_dic.values())):
        llist_tomos_dic = llist.get_tomos()
        for tkey, ltomo in zip(iter(llist_tomos_dic.keys()), iter(llist_tomos_dic.values())):
            try:
                tomos_exp_dsts[tkey]
            except KeyError:
                tomos_hash[tkey] = tomos_count
                tomos_exp_dsts[tkey], tomos_sim_dsts[tkey] = dict.fromkeys(list(lists_dic.keys())), dict.fromkeys(list(lists_dic.keys()))
                tomos_exp_fdsts[tkey], tomos_sim_fdsts[tkey] = dict.fromkeys(list(lists_dic.keys())), dict.fromkeys(list(lists_dic.keys()))
                tomos_count += 1
    for tkey in tomos_exp_dsts.keys():
        for lkey in lists_dic.keys():
            tomos_exp_dsts[tkey][lkey], tomos_exp_fdsts[tkey][lkey] = list(), list()
            tomos_sim_dsts[tkey][lkey], tomos_sim_fdsts[tkey][lkey] = list(), list()

    print('\tLIST COMPUTING LOOP:')
    sim_obj_set = surf.SetListSimulations()
    for lkey in lists_hash.values():

        llist = lists_dic[lkey]
        sim_obj_list = surf.ListSimulations()
        print('\t\t-Processing list: ' + lkey)
        print('\t\t\t+Tomograms computing loop:')
        for tkey in tomos_hash.keys():

            print('\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1])
            try:
                ltomo = llist.get_tomo_by_key(tkey)
            except KeyError:
                print('\t\t\t\t\t-Tomogram with key ' + tkey + ' is not in the list ' + lkey + ' , continuing...')
                continue
            if ltomo.get_num_particles() <= 0:
                print('\t\t\t\t\t-WARNING: no particles to process, continuing...')
                continue
            if not ltomo.is_active():
                ltomo.set_active(on=True)

            print('\t\t\t\t\t-Computing nearest inter particle distances...')
            hold_arr_dsts = ltomo.compute_uni_1st_order_dsts()
            if hold_arr_dsts is not None:
                lists_exp_dsts[lkey].append(hold_arr_dsts)
                tomos_exp_dsts[tkey][lkey].append(hold_arr_dsts)

            print('\t\t\t\t\t-Computing CSR and particles nearest distances...')
            hold_arr_fdsts = ltomo.compute_uni_1st_order_dsts(ana_f_npoints)
            if hold_arr_fdsts is not None:
                lists_exp_fdsts[lkey].append(hold_arr_fdsts)
                tomos_exp_fdsts[tkey][lkey].append(hold_arr_fdsts)

            print('\t\t\t\t\t-Generating the simulated instances...')
            temp_model = ModelCSRV(ltomo.get_voi(), part_vtp)
            sim_tomos = gen_tlist(p_nsims, ltomo.get_num_particles(), temp_model, mode_emb='center', npr=ana_npr_model,
                                  tmp_folder=tmp_sim_folder)
            hold_mat_dsts, hold_mat_fdsts = list(), list()
            for sim_tomo in sim_tomos.get_tomo_list():
                if hold_arr_dsts is not None:
                    hold_arr_dsts = sim_tomo.compute_uni_1st_order_dsts()
                    tomos_sim_dsts[tkey][lkey].append(hold_arr_dsts)
                    for i in range(ltomo.get_num_particles()):
                        lists_sim_dsts[lkey].append(hold_arr_dsts)
                if hold_arr_fdsts is not None:
                    hold_arr_fdsts = sim_tomo.compute_uni_1st_order_dsts(ana_f_npoints)
                    tomos_sim_fdsts[tkey][lkey].append(hold_arr_fdsts)
                    for i in range(ltomo.get_num_particles()):
                        lists_sim_fdsts[lkey].append(hold_arr_fdsts)

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print('\tPickling computation workspace in: ' + out_wspace)
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash,
              tomos_exp_dsts, tomos_sim_dsts, tomos_exp_fdsts, tomos_sim_fdsts,
              lists_exp_dsts, lists_sim_dsts, lists_exp_fdsts, lists_sim_fdsts, lists_color)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:
    print('\tLoading the workspace: ' + in_wspace)
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    lists_count, tomos_count = wspace[0], wspace[1]
    lists_hash, tomos_hash = wspace[2], wspace[3]
    tomos_exp_dsts, tomos_sim_dsts, tomos_exp_fdsts, tomos_sim_fdsts = wspace[4], wspace[5], wspace[6], wspace[7]
    lists_exp_dsts, lists_sim_dsts, lists_exp_fdsts, lists_sim_fdsts, lists_color = wspace[8], wspace[9], wspace[10], \
                                                                                    wspace[11], wspace[12]

print('\tPrinting lists hash: ')
for id, lkey in zip(iter(lists_hash.keys()), iter(lists_hash.values())):
    print('\t\t-[' + str(id) + '] -> [' + lkey + ']')
print('\tPrinting tomograms hash: ')
for tkey, val in zip(iter(tomos_hash.keys()), iter(tomos_hash.values())):
    print('\t\t-[' + tkey + '] -> [' + str(val) + ']')

# Getting the lists colormap
n_lists = len(list(lists_hash.keys()))
# for i, lkey in zip(lists_hash.iterkeys(), lists_hash.itervalues()):
#     lists_color[lkey] = pt_cmap(1.*i/n_lists)
for i, lkey in zip(iter(lists_hash.keys()), iter(lists_hash.values())):
    if lkey == '0':
        lists_color[lkey] = 'red'
    elif lkey == '1':
        lists_color[lkey] = 'yellow'
    elif lkey == '2':
        lists_color[lkey] = 'cyan'
    else:
        lists_color[lkey] = pt_cmap(1. * i / n_lists)

print('\tTOMOGRAMS PLOTTING LOOP: ')

out_tomos_dir = out_stem_dir + '/tomos'
os.makedirs(out_tomos_dir)

print('\t\t-Plotting Histogram...')
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(iter(tomos_exp_dsts.keys()), iter(tomos_exp_dsts.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(iter(tomos_exp_dsts[tkey].keys()), iter(tomos_exp_dsts[tkey].values())):
        try:
            hist_bins, hist_vals = compute_hist(arr, ana_nbins, ana_rmax)
            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            sims_hist_vals = list()
            for sim_dsts in tomo_sim_dsts:
                sims_hist_vals.append(compute_hist(sim_dsts, ana_nbins, ana_rmax)[1])
            if len(sims_hist_vals) > 0:
                ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
            else:
                raise ValueError
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            continue
        plt.figure()
        # plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Frequency')
        plt.xlabel('Distance [nm]')
        plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(hist_bins, ic_low, 'k--')
        plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
        # plt.plot(hist_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='grey', edgecolor='w')
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey_short
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/H_' + lkey + '.png')
        plt.close()

print('\t\t-Plotting Function-G...')
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(iter(tomos_exp_dsts.keys()), iter(tomos_exp_dsts.values())):
    for lkey, arr in zip(iter(tomos_exp_dsts[tkey].keys()), iter(tomos_exp_dsts[tkey].values())):
        try:
            cdf_bins, cdf_vals = compute_cdf(arr, ana_nbins, ana_rmax)
            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            sims_cdf_vals = list()
            for sim_dsts in tomo_sim_dsts:
                sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1])
            if len(sims_cdf_vals) > 0:
                ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
                pvals_glow[tkey], pvals_ghigh[tkey] = compute_pvals(cdf_vals, np.asarray(sims_cdf_vals))
            else:
                raise ValueError
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            continue
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        # plt.title('Univariate 1st order for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Nearest neighbor distribution')
        plt.xlabel('Distance [nm]')
        plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], label=lkey, linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(cdf_bins, ic_low, 'k--')
        plt.plot(cdf_bins, ic_med, 'k', linewidth=2.0)
        # plt.plot(cdf_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='grey', edgecolor='w')
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey_short
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/G_' + lkey + '.png')
        plt.close()

print('\t\t-Plotting Function-G p-values (tomo[s_low, s_high]= (min_pval, max_pval)):')
min_pvals, max_pvals = list(), list()
for tkey, pvals_low, pvals_high in zip(iter(pvals_glow.keys()), iter(pvals_glow.values()), iter(pvals_ghigh.values())):
    pvals_hmin, pvals_hmax = np.copy(pvals_low), np.copy(pvals_high)
    pvals_hmin[cdf_bins <= pr_ss] = -1
    pvals_hmax[cdf_bins <= pr_ss] = -1
    idx_min, idx_max = np.argmax(pvals_hmin), np.argmax(pvals_hmax)
    min_pval, max_pval = pvals_hmin[idx_min], pvals_hmax[idx_max]
    print('\t\t\t+' + tkey + '[' + str(cdf_bins[idx_min]) + ', ' + str(cdf_bins[idx_max]) + ']= (' \
          + str(min_pval) + ', ' + str(max_pval) + ')')
    min_pvals.append(pvals_hmin[idx_min])
    max_pvals.append(pvals_hmax[idx_max])
plt.figure()
plt.title('Function-G p-values box-plot')
plt.ylabel('Function-G (p-values)')
plt.boxplot([min_pvals, max_pvals], labels=['Low', 'High'])
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/G_pvals.png')
plt.close()

print('\t\t-Plotting Function-F...')
pvals_flow, pvals_fhigh = dict(), dict()
for tkey, ltomo in zip(iter(tomos_exp_fdsts.keys()), iter(tomos_exp_fdsts.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(iter(tomos_exp_fdsts[tkey].keys()), iter(tomos_exp_fdsts[tkey].values())):
        try:
            cdf_bins, cdf_vals = compute_cdf(arr, ana_nbins, ana_rmax)
            tomo_sim_fdsts = tomos_sim_fdsts[tkey][lkey]
            sims_cdf_fvals = list()
            for sim_fdsts in tomo_sim_fdsts:
                sims_cdf_fvals.append(compute_cdf(sim_fdsts, ana_nbins, ana_rmax)[1])
            if len(sims_cdf_fvals) > 0:
                ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_fvals))
                pvals_flow[tkey], pvals_fhigh[tkey] = compute_pvals(cdf_vals, np.asarray(sims_cdf_fvals))
            else:
                raise ValueError
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            continue
        plt.figure()
        # plt.title('Univariate 1st order for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Contact function distribution')
        plt.xlabel('Distance [nm]')
        plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], label=lkey, linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(cdf_bins, ic_low, 'k--')
        plt.plot(cdf_bins, ic_med, 'k', linewidth=2.0)
        # plt.plot(cdf_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='grey', edgecolor='w')
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey_short
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/F_' + lkey + '.png')
        plt.close()

print('\t\t-Plotting Function-F p-values (tomo[s_low, s_high]= (min_pval, max_pval)):')
min_pvals, max_pvals = list(), list()
for tkey, pvals_low, pvals_high in zip(iter(pvals_flow.keys()), iter(pvals_flow.values()), iter(pvals_fhigh.values())):
    pvals_hmin, pvals_hmax = np.copy(pvals_low), np.copy(pvals_high)
    pvals_hmin[cdf_bins <= pr_ss] = -1
    pvals_hmax[cdf_bins <= pr_ss] = -1
    idx_min, idx_max = np.argmax(pvals_hmin), np.argmax(pvals_hmax)
    min_pval, max_pval = pvals_hmin[idx_min], pvals_hmax[idx_max]
    print('\t\t\t+' + tkey + '[' + str(cdf_bins[idx_min]) + ', ' + str(cdf_bins[idx_max]) + ']= (' + \
          str(min_pval) + ', ' + str(max_pval) + ')')
    min_pvals.append(pvals_hmin[idx_min])
    max_pvals.append(pvals_hmax[idx_max])
plt.figure()
plt.title('Function-F p-values box-plot')
plt.ylabel('Function-F (p-values)')
plt.boxplot([min_pvals, max_pvals], labels=['Low', 'High'])
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/F_pvals.png')
plt.close()

print('\t\t-Plotting Function-J...')
for tkey, ltomo in zip(iter(tomos_exp_fdsts.keys()), iter(tomos_exp_fdsts.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, farr in zip(iter(tomos_exp_fdsts[tkey].keys()), iter(tomos_exp_fdsts[tkey].values())):
        arr = tomos_exp_dsts[tkey][lkey]
        try:
            cdf_bins, cdf_vals = compute_cdf(arr, ana_nbins, ana_rmax)
            cdf_fvals = compute_cdf(farr, ana_nbins, ana_rmax)[1]
            cdf_jvals = compute_J(cdf_vals, cdf_fvals, high=p_jhigh)
            tomo_sim_fdsts = tomos_sim_fdsts[tkey][lkey]
            sims_cdf_vals, sims_cdf_fvals = list(), list()
            for sim_dsts, sim_fdsts in zip(tomos_sim_dsts[tkey][lkey], tomo_sim_fdsts):
                sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1])
                sims_cdf_fvals.append(compute_cdf(sim_fdsts, ana_nbins, ana_rmax)[1])
            if (len(sims_cdf_vals) > 0) and (len(sims_cdf_fvals) > 0):
                ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
                icf_low, icf_med, icf_high = compute_ic(p_per, np.asarray(sims_cdf_fvals))
                cdf_sim_jlow = compute_J(ic_low, icf_high, high=p_jhigh)
                cdf_sim_jmed = compute_J(ic_med, icf_med, high=p_jhigh)
                cdf_sim_jhigh = compute_J(ic_high, icf_low, high=p_jhigh)
            else:
                raise ValueError
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            continue
        plt.figure()
        # plt.title('Univariate 1st order for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('J-function')
        plt.xlabel('Distance [nm]')
        plt.plot(cdf_bins[:len(cdf_jvals)], cdf_jvals, color=lists_color[lkey], label=lkey, linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(cdf_bins[:len(cdf_sim_jlow)], cdf_sim_jlow, 'k--')
        plt.plot(cdf_bins[:len(cdf_sim_jmed)], cdf_sim_jmed, 'k', linewidth=2.0)
        # plt.plot(cdf_bins[:len(cdf_sim_jhigh)], cdf_sim_jhigh, 'k--')
        if len(cdf_sim_jlow) < len(cdf_sim_jhigh):
            plt.fill_between(cdf_bins[:len(cdf_sim_jlow)], cdf_sim_jlow[:len(cdf_sim_jlow)], cdf_sim_jhigh[:len(cdf_sim_jlow)], alpha=0.5, color=lists_color[lkey], edgecolor='w')
        else:
            plt.fill_between(cdf_bins[:len(cdf_sim_jhigh)], cdf_sim_jlow[:len(cdf_sim_jhigh)], cdf_sim_jhigh[:len(cdf_sim_jhigh)], alpha=0.5, color=lists_color[lkey], edgecolor='w')
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey_short
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/J_' + lkey + '.png')
        plt.close()

print('\tLISTS PLOTTING LOOP: ')

crf = ana_res / 0.684

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

print('\t\t-Plotting Histogram...')
for lkey, ltomo in zip(iter(lists_exp_dsts.keys()), iter(lists_exp_dsts.values())):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    try:
        hist_bins, hist_vals = compute_hist(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
        list_sim_dsts = lists_sim_dsts[lkey]
        sims_hist_vals = list()
        for sim_dsts in list_sim_dsts:
            sims_hist_vals.append(compute_hist(sim_dsts, ana_nbins, ana_rmax)[1])
        if len(sims_hist_vals) > 0:
            ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
        else:
            raise ValueError
    except ValueError or IndexError:
        print(np.concatenate(np.asarray(ltomo)))
        print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
        continue
    plt.figure()
    # plt.title('Nearest distance histogram for ' + lkey_short)
    plt.ylabel('Frequency')
    plt.xlabel('Distance [nm]')
    plt.plot(hist_bins*crf, hist_vals, color=lists_color[lkey], linewidth=2.0)
    # plt.plot(hist_bins, ic_low, 'k--')
    plt.plot(hist_bins*crf, ic_med, 'k', linewidth=2.0)
    # plt.plot(hist_bins, ic_high, 'k--')
    plt.fill_between(hist_bins*crf, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/H_' + lkey_short + '.png', dpi=600)
    plt.close()

print('\t\t-Plotting Function-G...')
for lkey, ltomo in zip(iter(lists_exp_dsts.keys()), iter(lists_exp_dsts.values())):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    try:
        cdf_bins, cdf_vals = compute_cdf(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
        list_sim_dsts = lists_sim_dsts[lkey]
        sims_cdf_vals = list()
        for sim_dsts in list_sim_dsts:
            sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1])
        if len(sims_cdf_vals) > 0:
            ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
        else:
            raise ValueError
    except ValueError or IndexError:
        print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
        continue
    plt.figure()
    # plt.title('Univariate 1st order for ' + lkey_short)
    plt.ylabel('Nearest neighbor distribution')
    plt.xlabel('Distance [nm]')
    plt.plot(cdf_bins*crf, cdf_vals, color=lists_color[lkey], linewidth=2.0)
    # plt.plot(cdf_bins, ic_low, 'k--')
    plt.plot(cdf_bins*crf, ic_med, 'k', linewidth=2.0)
    # plt.plot(cdf_bins, ic_high, 'k--')
    plt.fill_between(cdf_bins*crf, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/G_' + lkey_short + '.png', dpi=600)
    plt.close()

print('\t\t-Plotting Function-F...')
for lkey, ltomo in zip(iter(lists_exp_fdsts.keys()), iter(lists_exp_fdsts.values())):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    try:
        cdf_bins, cdf_vals = compute_cdf(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
        list_sim_fdsts = lists_sim_fdsts[lkey]
        sims_cdf_vals = list()
        for sim_fdsts in list_sim_fdsts:
            sims_cdf_vals.append(compute_cdf(sim_fdsts, ana_nbins, ana_rmax)[1])
        if len(sims_cdf_vals) > 0:
            ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
        else:
            raise ValueError
    except ValueError or IndexError:
        print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
        continue
    plt.figure()
    # plt.title('Univariate 1st order for ' + lkey_short)
    plt.ylabel('Contact function distribution')
    plt.xlabel('Distance [nm]')
    plt.plot(cdf_bins*crf, cdf_vals, color=lists_color[lkey], linewidth=2.0)
    # plt.plot(cdf_bins, ic_low, 'k--')
    plt.plot(cdf_bins*crf, ic_med, color='k', linewidth=2.0)
    # plt.plot(cdf_bins, ic_high, 'k--')
    plt.fill_between(cdf_bins*crf, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/F_' + lkey_short + '.png', dpi=600)
    plt.close()

print('\t\t-Plotting Function-J...')
for lkey, ltomo in zip(iter(lists_exp_fdsts.keys()), iter(lists_exp_fdsts.values())):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    try:
        cdf_bins, cdf_vals = compute_cdf(np.concatenate(np.asarray(lists_exp_dsts[lkey])), ana_nbins, ana_rmax)
        cdf_fvals = compute_cdf(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)[1]
        cdf_jvals = compute_J(cdf_vals, cdf_fvals, high=p_jhigh)
        list_sim_fdsts = lists_sim_fdsts[lkey]
        sims_cdf_vals, sims_cdf_fvals = list(), list()
        for sim_dsts, sim_fdsts in zip(lists_sim_dsts[lkey], list_sim_fdsts):
            sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1])
            sims_cdf_fvals.append(compute_cdf(sim_fdsts, ana_nbins, ana_rmax)[1])
        if (len(sims_cdf_vals) > 0) and (len(sims_cdf_fvals) > 0):
            ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
            icf_low, icf_med, icf_high = compute_ic(p_per, np.asarray(sims_cdf_fvals))
            cdf_sim_jlow = compute_J(ic_low, icf_high, high=p_jhigh)
            cdf_sim_jmed = compute_J(ic_med, icf_med, high=p_jhigh)
            cdf_sim_jhigh = compute_J(ic_high, icf_low, high=p_jhigh)
        else:
            raise ValueError
    except ValueError or IndexError:
        print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
        continue
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('Univariate 1st order for ' + lkey_short)
    plt.ylabel('J-function')
    plt.xlabel('Distance [nm]')
    plt.plot(cdf_bins[:len(cdf_jvals)]*crf, cdf_jvals, color=lists_color[lkey], linewidth=2.0)
    # plt.plot(cdf_bins[:len(cdf_sim_jlow)], cdf_sim_jlow, 'k--')
    # plt.plot(cdf_bins[:len(cdf_sim_jmed)]*crf, cdf_sim_jmed, 'k', linewidth=2.0)
    # plt.plot(cdf_bins[:len(cdf_sim_jhigh)], cdf_sim_jhigh, 'k--')
    if len(cdf_sim_jlow) < len(cdf_sim_jhigh):
        plt.fill_between(cdf_bins[:len(cdf_sim_jlow)]*crf, cdf_sim_jlow[:len(cdf_sim_jlow)], cdf_sim_jhigh[:len(cdf_sim_jlow)], alpha=0.5, color='grey', edgecolor='w')
        x_lim_max = cdf_bins[len(cdf_sim_jlow)]
    else:
        plt.fill_between(cdf_bins[:len(cdf_sim_jhigh)] * crf, cdf_sim_jlow[:len(cdf_sim_jhigh)],
                         cdf_sim_jhigh[:len(cdf_sim_jhigh)], alpha=0.5, color='grey', edgecolor='w')
        x_lim_max = cdf_bins[len(cdf_sim_jhigh)]
    ax.set_xlim((0, x_lim_max))
    ax.set_xlim((0, 35))
    ax.set_ylim((0, 10))
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/J_' + lkey_short + '.png', dpi=600)
    plt.close()

print('Successfully terminated. (' + time.strftime("%c") + ')')
