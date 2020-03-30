"""

    Analyzes filaments to membrane distances on list of tomograms

    Input:  - A STAR file with a set of ListTomoFilaments pickles (SetListFilaments object input)
            - Settings for the measurements

    Output: - Plots by tomograms
            - Global plots

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
from pyorg.surf.model import ModelCSRV, ModelRR, gen_tlist
from pyorg.spatial.sparse import compute_cdf, compute_hist, compute_J
from matplotlib import pyplot as plt, rcParams
try:
    import cPickle as pickle
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

ROOT_PATH = '/fs/pool/pool-ruben/antonio/filaments'

# Input STAR files
in_star = ROOT_PATH + '/ltomos/test/test_ltomos.star'
in_wspace = None

# Output directory
out_dir = ROOT_PATH + '/ana/fil_dsts/test/'
out_stem = 'test_100_200'

# Analysis variables
ana_res = 1.408 # nm/voxel
ana_nbins = 100
ana_rmax = 200 # nm

# Figure saving options
fig_fmt = '.png' # if None they showed instead

# Plotting options
pt_sim_v = True
pt_cmap = plt.get_cmap('gist_rainbow')

########################################################################################
# MAIN ROUTINE
########################################################################################

###### Additional functionality



########## Print initial message

print 'Univariate first order analysis for a ListTomoFilaments by tomograms.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
print '\tInput STAR file for filaments: ' + str(in_star)
if in_wspace is not None:
    print '\tLoad workspace from: ' + in_wspace
print '\tOrganization analysis settings: '
print '\t\t-Number of bins: ' + str(ana_nbins)
ana_rmax_v = float(ana_rmax) / ana_res
print '\t\t-Maximum radius: ' + str(ana_rmax) + ' nm (' + str(ana_rmax_v) + ' vx)'
if fig_fmt is not None:
    print '\tStoring figures:'
    print '\t\t-Format: ' + str(fig_fmt)
else:
    print '\tPlotting settings: '
print '\t\t-Colormap: ' + str(pt_cmap)
if pt_sim_v:
    print '\t\t-Verbose simulation activated!'
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

if in_wspace is None:

    print '\tLoading input data...'
    star = sub.Star()
    try:
        star.load(in_star)
    except pexceptions.PySegInputError as e:
        print 'ERROR: input STAR file for filaments could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    set_lists, lists_dic_rows = surf.SetListTomoFilaments(), dict()
    for row in range(star.get_nrows()):
        ltomos_pkl = star.get_element('_psPickleFile', row)
        ltomos = unpickle_obj(ltomos_pkl)
        set_lists.add_list_tomos(ltomos, ltomos_pkl)
        fkey = os.path.split(ltomos_pkl)[1]
        short_key_idx = fkey.index('_')
        short_key = fkey[:short_key_idx]
        lists_dic_rows[short_key] = row

    print '\tBuilding the dictionaries...'
    lists_count, tomos_count = 0, 0
    lists_dic = dict()
    lists_hash, tomos_hash = dict(), dict()
    tomos_exp_dsts = dict()
    lists_exp_dsts = dict()
    set_lists_dic = set_lists.get_lists()
    for lkey, llist in zip(set_lists_dic.iterkeys(), set_lists_dic.itervalues()):
        fkey = os.path.split(lkey)[1]
        print '\t\t-Processing list: ' + fkey
        short_key_idx = fkey.index('_')
        short_key = fkey[:short_key_idx]
        print '\t\t\t+Short key found: ' + short_key
        try:
            lists_dic[short_key]
        except KeyError:
            lists_dic[short_key] = llist
            lists_hash[lists_count] = short_key
            lists_exp_dsts[short_key] = list()
            lists_count += 1
    for lkey, llist in zip(set_lists_dic.iterkeys(), set_lists_dic.itervalues()):
        llist_tomos_dic = llist.get_tomos()
        for tkey, ltomo in zip(llist_tomos_dic.iterkeys(), llist_tomos_dic.itervalues()):
            try:
                tomos_exp_dsts[tkey]
            except KeyError:
                tomos_hash[tkey] = tomos_count
                tomos_exp_dsts[tkey] = dict.fromkeys(lists_dic.keys())
                tomos_count += 1
    for tkey in tomos_exp_dsts.iterkeys():
        for lkey in lists_dic.iterkeys():
            tomos_exp_dsts[tkey][lkey] = list()

    print '\tLIST COMPUTING LOOP:'
    for lkey in lists_hash.itervalues():

        llist = lists_dic[lkey]
        print '\t\t-Processing list: ' + lkey
        hold_row = lists_dic_rows[lkey]

        print '\t\t\t+Tomograms computing loop:'
        for tkey in tomos_hash.iterkeys():

            print '\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1]
            try:
                ltomo = llist.get_tomo_by_key(tkey)
            except KeyError:
                print '\t\t\t\t\t-Tomogram with key ' + tkey + ' is not in the list ' + lkey + ' , continuing...'
                continue
            if ltomo.get_num_filaments() <= 0:
                print '\t\t\t\t\t-WARNING: no filaments to process, continuing...'
                continue

            print '\t\t\t\t\t-Computing filament to membrane nearest distances...'
            hold_arr_dsts = ltomo.compute_fils_seg_dsts()
            if hold_arr_dsts is not None:
                lists_exp_dsts[lkey].append(hold_arr_dsts)
                tomos_exp_dsts[tkey][lkey].append(hold_arr_dsts)
            out_fils = out_dir + '/' + tkey.replace('/', '_') + '_fils.vtp'
            disperse_io.save_vtp(ltomo.gen_filaments_vtp(), out_fils)

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print '\tPickling computation workspace in: ' + out_wspace
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash,
              tomos_exp_dsts, lists_exp_dsts)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:
    print '\tLoading the workspace: ' + in_wspace
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    lists_count, tomos_count = wspace[0], wspace[1]
    lists_hash, tomos_hash = wspace[2], wspace[3]
    tomos_exp_dsts, lists_exp_dsts = wspace[4], wspace[5]

print '\tPrinting lists hash: '
for id, lkey in zip(lists_hash.iterkeys(), lists_hash.itervalues()):
    print '\t\t-[' + str(id) + '] -> [' + lkey + ']'
print '\tPrinting tomograms hash: '
for tkey, val in zip(tomos_hash.iterkeys(), tomos_hash.itervalues()):
    print '\t\t-[' + tkey + '] -> [' + str(val) + ']'

# Getting the lists colormap
n_lists, lists_color = len(lists_hash.keys()), dict()
for i, lkey in zip(lists_hash.iterkeys(), lists_hash.itervalues()):
    lists_color[lkey] = pt_cmap(1.*i/n_lists)

print '\tTOMOGRAMS PLOTTING LOOP: '

out_tomos_dir = out_stem_dir + '/tomos'
os.makedirs(out_tomos_dir)

print '\t\t-Plotting Histogram...'
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        try:
            hist_bins, hist_vals = compute_hist(arr, ana_nbins, ana_rmax)
            # tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            # sims_hist_vals = list()
            # for sim_dsts in tomo_sim_dsts:
            #     sims_hist_vals.append(compute_hist(sim_dsts, ana_nbins, ana_rmax)[1])
            # if len(sims_hist_vals) > 0:
            #     ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
            # else:
            #     raise ValueError
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            # continue
            pass
        plt.figure()
        # plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Probability Density')
        plt.xlabel('Filament-Membrane Nearest Distance [nm]')
        plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(hist_bins, ic_low, 'k--')
        # plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
        # plt.plot(hist_bins, ic_high, 'k--')
        # plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        plt.tight_layout()
        plt.grid(True)
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/H_' + lkey + '.png')
        plt.close()

print '\t\t-Plotting Function-G...'
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        try:
            cdf_bins, cdf_vals = compute_cdf(arr, ana_nbins, ana_rmax)
            # tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            # sims_cdf_vals = list()
            # for sim_dsts in tomo_sim_dsts:
            #     sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1])
            # if len(sims_cdf_vals) > 0:
            #     ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
            #     pvals_glow[tkey], pvals_ghigh[tkey] = compute_pvals(cdf_vals, np.asarray(sims_cdf_vals))
            # else:
            #     raise ValueError
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            continue
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        # plt.title('Univariate 1st order for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Cumulative Density')
        plt.xlabel('Filament-Membrane Nearest Distance [nm]')
        plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], label=lkey, linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(cdf_bins, ic_low, 'k--')
        # plt.plot(cdf_bins, ic_med, 'k', linewidth=2.0)
        # plt.plot(cdf_bins, ic_high, 'k--')
        # plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        plt.tight_layout()
        plt.grid(True)
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/G_' + lkey + '.png')
        plt.close()


print '\tLISTS PLOTTING LOOP: '

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

print '\t\t-Plotting Histogram...'
for lkey, ltomo in zip(lists_exp_dsts.iterkeys(), lists_exp_dsts.itervalues()):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    try:
        hist_bins, hist_vals = compute_hist(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
        # list_sim_dsts = lists_sim_dsts[lkey]
        # sims_hist_vals = list()
        # for sim_dsts in list_sim_dsts:
        #     sims_hist_vals.append(compute_hist(sim_dsts, ana_nbins, ana_rmax)[1])
        # if len(sims_hist_vals) > 0:
        #     ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
        # else:
        #     raise ValueError
    except ValueError or IndexError:
        print np.concatenate(np.asarray(ltomo))
        print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
        continue
    plt.figure()
    # plt.title('Nearest distance histogram for ' + lkey_short)
    plt.ylabel('Probability Density')
    plt.xlabel('Filament-Membrane Nearest Distance [nm]')
    plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=2.0)
    # plt.plot(hist_bins, ic_low, 'k--')
    # plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
    # plt.plot(hist_bins, ic_high, 'k--')
    # plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/H_' + lkey_short + '.png', dpi=600)
    plt.close()

print '\t\t-Plotting Function-G...'
for lkey, ltomo in zip(lists_exp_dsts.iterkeys(), lists_exp_dsts.itervalues()):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    try:
        cdf_bins, cdf_vals = compute_cdf(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
        # list_sim_dsts = lists_sim_dsts[lkey]
        # sims_cdf_vals = list()
        # for sim_dsts in list_sim_dsts:
        #     sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1])
        # if len(sims_cdf_vals) > 0:
        #     ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
        # else:
        #     raise ValueError
    except ValueError or IndexError:
        print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
        continue
    plt.figure()
    # plt.title('Univariate 1st order for ' + lkey_short)
    plt.ylabel('Cumulative Density')
    plt.xlabel('Filament-Membrane Nearest Distance [nm]')
    plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], linewidth=2.0)
    # plt.plot(cdf_bins, ic_low, 'k--')
    # plt.plot(cdf_bins, ic_med, 'k', linewidth=2.0)
    # plt.plot(cdf_bins, ic_high, 'k--')
    # plt.fill_between(cdf_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/G_' + lkey_short + '.png', dpi=600)
    plt.close()

print 'Successfully terminated. (' + time.strftime("%c") + ')'

