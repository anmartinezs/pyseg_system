"""

    Analyzes oriented membrane segmentations inter-distances from lists of tomograms

    Input:  - A STAR file with a set of ListTomoOMSegmentations pickles (SetListOMSegmentations object input)
            - Settings for the measurements

    Output: - Plots by tomograms
            - Global plots

"""

################# Package import

import os
import csv
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
in_star = ROOT_PATH + '/ltomos_omsegs/omsegs_all/all_ltomos.star' # '/ltomos_omsegs/omsegs_fil_new/fil_new_ltomos.star' # '/ltomos_omsegs/omsegs_fil_ctrl/fil_ctrl_ltomos.star' # '/ltomos_omsegs/omsegs_1/omsegs_ltomos.star'
in_wspace = ROOT_PATH + '/ana/oms_dsts/oms_all/oms_fil_ctrl_25_250_3_svg_v3_wspace.pkl' # None

# Output directory
out_dir = ROOT_PATH + '/ana/oms_dsts/oms_all' # '/ana/oms_dsts/oms_fil_ctrl' # '/ana/oms_dsts/oms_vols/'
out_stem = 'oms_fil_ctrl_25_250_3_svg_v4' # 'oms_fil_ctrl_25_250_3_png_2'

# Analysis variables
pt_per = 95 # %
ana_nbins = 25
ana_rmax = 250 # nm

# Figure saving options
fig_fmt = '.svg' # '.png' # if None they showed instead

# Plotting options
pt_sim_v = True
pt_cmap = plt.get_cmap('gist_rainbow')

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

########## Print initial message

print 'Univariate first order analysis for a ListTomoOMSegmentations by tomograms.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
print '\tInput STAR file for oriented membrane segmentations: ' + str(in_star)
if in_wspace is not None:
    print '\tLoad workspace from: ' + in_wspace
print '\tOrganization analysis settings: '
print '\t\t-Number of bins: ' + str(ana_nbins)
print '\t\t-Maximum radius: ' + str(ana_rmax) + ' nm'
if fig_fmt is not None:
    print '\tStoring figures:'
    print '\t\t-Format: ' + str(fig_fmt)
else:
    print '\tPlotting settings: '
print '\t\t-Percentile for IC: ' + str(pt_per)
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
    set_lists, lists_dic_rows = surf.SetListTomoOMSegmentations(), dict()
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
    tomos_exp_dsts, tomos_exp_vols, tomos_exp_occ = dict(), dict(), dict()
    lists_exp_dsts, lists_exp_vols, lists_exp_occ = dict(), dict(), dict()
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
            lists_exp_dsts[short_key], lists_exp_vols[short_key], lists_exp_occ[short_key] = list(), list(), list()
            lists_count += 1
    for lkey, llist in zip(set_lists_dic.iterkeys(), set_lists_dic.itervalues()):
        llist_tomos_dic = llist.get_tomos()
        for tkey, ltomo in zip(llist_tomos_dic.iterkeys(), llist_tomos_dic.itervalues()):
            try:
                tomos_exp_dsts[tkey]
            except KeyError:
                tomos_hash[tkey] = tomos_count
                tomos_exp_dsts[tkey], tomos_exp_vols[tkey], tomos_exp_occ[tkey] = dict.fromkeys(lists_dic.keys()), \
                                                                                  dict.fromkeys(lists_dic.keys()), \
                                                                                  dict.fromkeys(lists_dic.keys())
                tomos_count += 1
    for tkey in tomos_exp_dsts.iterkeys():
        for lkey in lists_dic.iterkeys():
            tomos_exp_dsts[tkey][lkey], tomos_exp_vols[tkey][lkey], tomos_exp_occ[tkey][lkey] = list(), list(), list()

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
            if ltomo.get_num_segmentations() <= 0:
                print '\t\t\t\t\t-WARNING: no filaments to process, continuing...'
                continue

            print '\t\t\t\t\t-Computing organelle volumes...'
            hold_arr_vols = ltomo.compute_seg_volumes(mode='lm')
            if hold_arr_vols is not None:
                lists_exp_vols[lkey].append(hold_arr_vols)
                tomos_exp_vols[tkey][lkey].append(hold_arr_vols)
                hold_arr_occ = hold_arr_vols.sum() / ltomo.get_tomo_vol()
                lists_exp_occ[lkey].append(hold_arr_occ)
                tomos_exp_occ[tkey][lkey].append(hold_arr_occ)

            print '\t\t\t\t\t-Computing membrane to membrane nearest distances...'
            hold_dsts_field = ltomo.compute_om_seg_dsts()
            hold_arr_dsts = hold_dsts_field[hold_dsts_field > 0]
            if hold_arr_dsts is not None:
                lists_exp_dsts[lkey].append(hold_arr_dsts)
                tomos_exp_dsts[tkey][lkey].append(hold_arr_dsts)
            out_fils = out_stem_dir + '/' + tkey.replace('/', '_') + '_mb_dsts.mrc'
            disperse_io.save_numpy(hold_dsts_field, out_fils)
            out_fils = out_stem_dir + '/' + tkey.replace('/', '_') + '_mb_lbl.mrc'
            disperse_io.save_numpy(ltomo.get_lbl_voi(mode='mb'), out_fils)
            out_fils = out_stem_dir + '/' + tkey.replace('/', '_') + '_lm_lbl.mrc'
            disperse_io.save_numpy(ltomo.get_lbl_voi(mode='lm'), out_fils)

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print '\tPickling computation workspace in: ' + out_wspace
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash,
              tomos_exp_dsts, lists_exp_dsts, tomos_exp_vols, lists_exp_vols,
              tomos_exp_occ, lists_exp_occ)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:
    print '\tLoading the workspace: ' + in_wspace
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    lists_count, tomos_count = wspace[0], wspace[1]
    lists_hash, tomos_hash = wspace[2], wspace[3]
    tomos_exp_dsts, lists_exp_dsts, tomos_exp_vols, lists_exp_vols = wspace[4], wspace[5], wspace[6], wspace[7]
    tomos_exp_occ, lists_exp_occ = wspace[8], wspace[9]

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
        if len(arr) <= 0:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            continue
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
        plt.xlabel('Inter-membranes Nearest Distance [nm]')
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
            plt.savefig(out_fig_dir + '/H_' + lkey + fig_fmt)
        plt.close()

print '\t\t-Plotting Function-G...'
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        if len(arr) <= 0:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            continue
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
        plt.xlabel('Inter-membranes Nearest Distance [nm]')
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
            plt.savefig(out_fig_dir + '/G_' + lkey + fig_fmt)
        plt.close()

print '\t\t-Plotting box-plots for volumes...'
for tkey, ltomo in zip(tomos_exp_vols.iterkeys(), tomos_exp_vols.itervalues()):
    for lkey, arr in zip(tomos_exp_vols[tkey].iterkeys(), tomos_exp_vols[tkey].itervalues()):
        if len(arr) <= 0:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            continue
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.boxplot(arr, showfliers=False, notch=True, vert=True, patch_artist=True,
                   labels=[lkey], boxprops={'linewidth': 2}, medianprops={'linewidth': 2, 'color':'black'},
                   whiskerprops={'linewidth': 2})
        # plt.xticks([1,], [lkey,])
        plt.ylabel('Organelle volume [$\mu m^3$]')
        plt.tight_layout()
        plt.grid(True)
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/Box_' + lkey + fig_fmt)
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
    plt.xlabel('Inter-membranes Nearest Distance [nm]')
    plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=2.0)
    # plt.plot(hist_bins, ic_low, 'k--')
    # plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
    # plt.plot(hist_bins, ic_high, 'k--')
    # plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/H_' + lkey_short + fig_fmt, dpi=600)
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
    plt.xlabel('Inter-membranes Nearest Distance [nm]')
    plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], linewidth=2.0)
    # plt.plot(cdf_bins, ic_low, 'k--')
    # plt.plot(cdf_bins, ic_med, 'k', linewidth=2.0)
    # plt.plot(cdf_bins, ic_high, 'k--')
    # plt.fill_between(cdf_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/G_' + lkey_short + fig_fmt, dpi=600)
    plt.close()

print '\t\t-Plotting Histogram with tomos IC...'
plt.figure()
# plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
plt.ylabel('Probability Density')
plt.xlabel('Inter-Membrane Nearest Distance [nm]')
exp_hist_dic = dict().fromkeys(lists_exp_dsts.keys())
for lkey in exp_hist_dic.iterkeys():
    exp_hist_dic[lkey] = list()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    exps_hist_vals = list()
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        if len(arr) <= 0:
            continue
        try:
            hist_bins, hist_vals = compute_hist(arr, ana_nbins, ana_rmax)
            exp_hist_dic[lkey].append(hist_vals)
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            # continue
            pass
fieldnames, rows = list(), dict()
for i, lkey in enumerate(exp_hist_dic.iterkeys()):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    if lkey_short == '0':
        lkey_short = 'WT'
    elif lkey_short == '1':
        lkey_short = 'MSA'
    elif lkey_short == '2':
        lkey_short = 'PFF'
    elif lkey_short == '3':
        lkey_short = 'CTRL'
    elif lkey_short == '4':
        lkey_short = 'FIL'
    ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(exp_hist_dic[lkey]))
    plt.plot(hist_bins, ic_med, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
    # plt.plot(hist_bins, ic_low, 'k--')
    # plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
    # plt.plot(hist_bins, ic_high, 'k--')
    plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color=lists_color[lkey], edgecolor='w')
    if i == 0:
        fieldnames.append('distance')
        rows['distance'] = hist_bins
    fieldnames.append(lkey_short + '_low')
    fieldnames.append(lkey_short + '_med')
    fieldnames.append(lkey_short + '_high')
    rows[lkey_short + '_low'] = ic_low
    rows[lkey_short + '_med'] = ic_med
    rows[lkey_short + '_high'] = ic_high
plt.legend(loc=1)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/H_lists' + fig_fmt, dpi=600)
plt.close()

out_csv_file = out_lists_dir + '/H_lists.csv'
with open(out_csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='excel')
    writer.writeheader()
    for i in range(len(rows['distance'])):
        row = dict()
        for key in rows.iterkeys():
            row[key] = rows[key][i]
        writer.writerow(row)

print '\t\t-Plotting the CDF with tomos IC...'
plt.figure()
# plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
plt.ylabel('Cumulative Density')
plt.xlabel('Inter-Membrane Nearest Distance [nm]')
exp_cdf_dic = dict().fromkeys(lists_exp_dsts.keys())
for lkey in exp_cdf_dic.iterkeys():
    exp_cdf_dic[lkey] = list()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    exps_hist_vals = list()
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        if len(arr) <= 0:
            continue
        try:
            cdf_bins, cdf_vals = compute_cdf(arr, ana_nbins, ana_rmax)
            exp_cdf_dic[lkey].append(cdf_vals)
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            # continue
            pass
for lkey in exp_hist_dic.iterkeys():
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    if lkey_short == '0':
        lkey_short = 'WT'
    elif lkey_short == '1':
        lkey_short = 'MSA'
    elif lkey_short == '2':
        lkey_short = 'PFF'
    elif lkey_short == '3':
        lkey_short = 'CTRL'
    elif lkey_short == '4':
        lkey_short = 'FIL'
    ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(exp_cdf_dic[lkey]))
    plt.plot(hist_bins, ic_med, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
    # plt.plot(hist_bins, ic_low, 'k--')
    # plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
    # plt.plot(hist_bins, ic_high, 'k--')
    plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color=lists_color[lkey], edgecolor='w')
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/CDF_lists' + fig_fmt, dpi=600)
plt.close()

print '\t\t-Boxplot for oganelle volumes...'
plt.figure()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
exp_vols_dic = dict().fromkeys(lists_exp_dsts.keys())
for lkey in exp_cdf_dic.iterkeys():
    exp_cdf_dic[lkey] = list()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    exps_hist_vals = list()
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        if len(arr) <= 0:
            continue
        try:
            exp_cdf_dic[lkey] += arr
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            # continue
            pass
n_lists, exp_vols_dic = 0, dict()
for lkey in lists_exp_vols:
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    if lkey_short == '0':
        lkey_short = 'WT'
    elif lkey_short == '1':
        lkey_short = 'MSA'
    elif lkey_short == '2':
        lkey_short = 'PFF'
    elif lkey_short == '3':
        lkey_short = 'CTRL'
    elif lkey_short == '4':
        lkey_short = 'FIL'
    exp_vols_dic[lkey_short] = list()
colors = list()
for lkey in lists_exp_vols.iterkeys():
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    if lkey_short == '0':
        lkey_short = 'WT'
        colors.append('yellow')
    elif lkey_short == '1':
        lkey_short = 'MSA'
        colors.append('blue')
    elif lkey_short == '2':
        lkey_short = 'PFF'
        colors.append('green')
    elif lkey_short == '3':
        lkey_short = 'CTRL'
        colors.append('red')
    elif lkey_short == '4':
        lkey_short = 'FIL'
        colors.append('blue')
    for arr in lists_exp_vols[lkey]:
        exp_vols_dic[lkey_short] += arr.tolist()
    n_lists += 1
boxp = plt.boxplot(exp_vols_dic.values(), showfliers=False, notch=True, vert=True, patch_artist=True,
                   labels=exp_vols_dic.keys(), boxprops={'linewidth': 2}, medianprops={'linewidth': 2, 'color':'black'},
                   whiskerprops={'linewidth': 2})
for patch, color in zip(boxp['boxes'], colors):
        patch.set_facecolor(color)
plt.ylabel('Organelle volume [$\mu m^3$]')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/Box_lists' + fig_fmt, dpi=600)
plt.close()

print '\t\t-Bar-plotting of organelle occupancy...'
plt.figure()
plt.ylabel('Organelle Occupancy [%]')
x_ticks, x_lbls = list(), list()
for idx, lkey in zip(np.arange(1, 3*len(lists_exp_occ.keys())), lists_exp_occ.iterkeys()):
    ic_low = 100. * np.percentile(lists_exp_occ[lkey], pt_per)
    ic_med = 100. * np.percentile(lists_exp_occ[lkey], 50)
    ic_high = 100. * np.percentile(lists_exp_occ[lkey], 100-pt_per)
    plt.bar(idx, ic_med, BAR_WIDTH, color=lists_color[lkey], linewidth=2, edgecolor='k')
    plt.errorbar(idx, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    x_ticks.append(idx)
    if lkey == '0':
        x_lbls.append('WT')
    elif lkey == '1':
        x_lbls.append('MSA')
    elif lkey == '2':
        x_lbls.append('PFF')
    elif lkey == '3':
        x_lbls.append('CTRL')
    elif lkey == '4':
        x_lbls.append('FIL')
    else:
        x_lbls.append(lkey)
plt.xticks(x_ticks, x_lbls)
plt.xlim(0.5, max(x_ticks) + .5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/Bar_lists' + fig_fmt, dpi=600)
plt.close()

if len(lists_exp_occ.values()) >= 2:
    print '\t\t-Statistical test for organelle occupancy for the first two list: '
    print '\t\t\t\t-K-S 2 samples [100*(1-p)]: ' + str(100. * (1 - sp.stats.ks_2samp(lists_exp_occ.values()[0],
                                                                                 lists_exp_occ.values()[1])[1])) + '%'
    print '\t\t\t\t-K-S 2 samples p: ' + str(sp.stats.ks_2samp(lists_exp_occ.values()[0], lists_exp_occ.values()[1])[1])

print 'Plotting the number of membrane voxels analyzed per condition: '
for lkey, vals in zip(lists_exp_dsts.iterkeys(), lists_exp_dsts.itervalues()):
    if lkey == '0':
        hkey = 'WT'
    elif lkey == '1':
        hkey = 'MSA'
    elif lkey == '2':
        hkey = 'PFF'
    elif lkey == '3':
        hkey = 'CTRL'
    elif lkey == '4':
        hkey = 'FIL'
    else:
        hkey = lkey
    count = 0
    for arr in lists_exp_dsts[lkey]:
        count += len(arr)
    print '- ' + str(hkey) + ': ' + str(count) + ' vx'

print 'Successfully terminated. (' + time.strftime("%c") + ')'

