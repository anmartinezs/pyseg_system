
"""

    Performs Bivariate 1st order analysis to a SetListTomograms object by tomograms with ListTomograms as reference

    Input:  - A STAR file with a ListTomoParticles pickle (SetListTomograms object input) used as reference
            - A STAR file with a set of ListTomoParticles pickles (SetListTomograms object input)

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
from pyorg.spatial.sparse import compute_cdf, compute_hist
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except ImportError:
    import pickle

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

BAR_WIDTH = .35

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext_repick/FTR'

# Input STAR file
in_star_ref = ROOT_PATH + '/stat/ltomos/try2_R/try2_ssup_20_min_5_ltomos.star' # '/ves_40/ltomos_premb_mask/premb_mask_ltomos.star'
in_ref_short_key = '0'
in_star = ROOT_PATH + '/stat/ltomos/try2_FT/try2_ssup_8_min_50_ltomos.star' # '/pst/nrt/ltomos_clst_flt_high/k4_gather_clst_flt_high_ltomos.star'
in_wspace = None # ROOT_PATH + '/pre/ref_nomb_1_clean/bi_1st_ves_40_pre_high_tomos/bi_sim_20_wspace.pkl' # (Insert a path to recover a pickled workspace instead of doing a new computation)

# Output directory
out_dir = ROOT_PATH + '/stat/uni/try2_FT/bi_1st_ssup_20_min_5'
out_stem = '50_60_10_sim_1' # ''uni_sph_4_60_5'

# Analysis variables
ana_res = 1.048 # nm/voxel
ana_nbins = 50
ana_rmax = 60 # nm
ana_f_npoints = 1000
ana_npr_model = 10

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = 1
p_per = 5 # %
# Particle surface
p_vtp = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre/vtps/sph_rad_5_surf.vtp'

# Figure saving options
fig_fmt = '.png' # if None they showed instead

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

print 'Univariate 1:N first order analysis for a ListTomoParticles by tomograms.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
print '\tInput reference STAR file: ' + str(in_star_ref)
print '\tInput reference short key: ' + str(in_ref_short_key)
print '\tInput STAR file: ' + str(in_star)
if in_wspace is not None:
    print '\tLoad workspace from: ' + in_wspace
print '\tOrganization analysis settings: '
print '\t\t-Number of bins: ' + str(ana_nbins)
ana_rmax_v = float(ana_rmax) / ana_res
print '\t\t-Maximum radius: ' + str(ana_rmax) + ' nm (' + str(ana_rmax_v) + ' voxels)'
print '\t\t-Number of CSR points for F-function: ' + str(ana_f_npoints)
if ana_npr_model is None:
    ana_npr_model = mp.cpu_count()
elif ana_npr_model > p_nsims:
    ana_npr_model = p_nsims
print '\t\t-Number of processors for models simulation: ' + str(ana_npr_model)
print '\tP-Value computation setting:'
print '\t\t-Percentile: ' + str(p_per) + ' %'
print '\t\t-Number of instances for simulations: ' + str(p_nsims)
print '\t\t-Particle surface: ' + p_vtp
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
    star, star_ref = sub.Star(), sub.Star()
    try:
        star.load(in_star)
        star_ref.load(in_star_ref)
    except pexceptions.PySegInputError as e:
        print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    set_lists = surf.SetListTomoParticles()
    for row in range(star.get_nrows()):
        ltomos_pkl = star.get_element('_psPickleFile', row)
        ltomos = unpickle_obj(ltomos_pkl)
        set_lists.add_list_tomos(ltomos, ltomos_pkl)
    try:
        part_vtp = disperse_io.load_poly(p_vtp)
    except pexceptions.PySegInputError as e:
        print 'ERROR: reference particle surface file could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    ref_list = None
    for row in range(star_ref.get_nrows()):
        ltomos_pkl = star_ref.get_element('_psPickleFile', row)
        fname_pkl = os.path.split(ltomos_pkl)[1]
        try:
            idx = fname_pkl.index('_')
        except ValueError:
            continue
        if fname_pkl[:idx] == in_ref_short_key:
            ref_list = unpickle_obj(ltomos_pkl)
    if ref_list is None:
        print 'ERROR: reference ListTomoParticles with short key ' + in_ref_short_key + ' was not found!'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)

    print '\tBuilding the dictionaries...'
    lists_count, tomos_count = 0, 0
    lists_dic = dict()
    lists_hash, tomos_hash = dict(), dict()
    tomos_exp_dsts, tomos_sim_dsts = dict(), dict()
    lists_exp_dsts, lists_sim_dsts, lists_color = dict(), dict(), dict()
    tmp_sim_folder = out_dir + '/tmp_gen_list_' + out_stem
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
            lists_exp_dsts[short_key], lists_sim_dsts[short_key] = list(), list()
            lists_count += 1
    for lkey, llist in zip(set_lists_dic.iterkeys(), set_lists_dic.itervalues()):
        llist_tomos_dic = llist.get_tomos()
        for tkey, ltomo in zip(llist_tomos_dic.iterkeys(), llist_tomos_dic.itervalues()):
            try:
                tomos_exp_dsts[tkey]
            except KeyError:
                tomos_hash[tkey] = tomos_count
                tomos_exp_dsts[tkey], tomos_sim_dsts[tkey] = dict.fromkeys(lists_dic.keys()), dict.fromkeys(lists_dic.keys())
                tomos_count += 1
    for tkey in tomos_exp_dsts.iterkeys():
        for lkey in lists_dic.iterkeys():
            tomos_exp_dsts[tkey][lkey], tomos_sim_dsts[tkey][lkey] = list(), list()

    print '\tLIST COMPUTING LOOP:'
    sim_obj_set = surf.SetListSimulations()
    for lkey in lists_hash.itervalues():

        llist = lists_dic[lkey]
        sim_obj_list = surf.ListSimulations()
        print '\t\t-Processing list: ' + lkey
        print '\t\t\t+Tomograms computing loop:'
        for tkey in tomos_hash.iterkeys():

            print '\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1]
            try:
                ltomo, ref_tomo = llist.get_tomo_by_key(tkey), ref_list.get_tomo_by_key(tkey)
            except KeyError:
                print '\t\t\t\t\t-Tomogram with key ' + tkey + ' is not in the reference list, continuing...'
                continue
            if (ltomo.get_num_particles() <= 0) or (ref_tomo.get_num_particles() <= 0):
                print '\t\t\t\t\t-WARNING: no particles to process, continuing...'
                continue

            print '\t\t\t\t\t-Computing nearest inter particle distances...'
            # hold_arr_dsts = ltomo.compute_bi_1st_order_dsts(ref_tomo.get_particle_coords())
            hold_arr_dsts = ref_tomo.compute_bi_1st_order_dsts(ltomo.get_particle_coords())
            if tomos_exp_dsts is not None:
                tomos_exp_dsts[tkey][lkey].append(hold_arr_dsts)
                for i in range(ref_tomo.get_num_particles()):
                    lists_exp_dsts[lkey].append(hold_arr_dsts)

            print '\t\t\t\t\t-Generating the simulated instances...'
            temp_model = ModelCSRV(ltomo.get_voi(), part_vtp)
            sim_tomos = gen_tlist(p_nsims, ltomo.get_num_particles(), temp_model, mode_emb='center', npr=ana_npr_model,
                                  tmp_folder=tmp_sim_folder)
            for sim_tomo in sim_tomos.get_tomo_list():
                # hold_arr_dsts = sim_tomo.compute_bi_1st_order_dsts(ref_tomo.get_particle_coords())
                hold_arr_dsts = ref_tomo.compute_bi_1st_order_dsts(sim_tomo.get_particle_coords())
                if hold_arr_dsts is not None:
                    tomos_sim_dsts[tkey][lkey].append(hold_arr_dsts)
                    for i in range(ref_tomo.get_num_particles()):
                        lists_sim_dsts[lkey].append(hold_arr_dsts)

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print '\tPickling computation workspace in: ' + out_wspace
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash,
              tomos_exp_dsts, tomos_sim_dsts,
              lists_exp_dsts, lists_sim_dsts, lists_color)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:
    print '\tLoading the workspace: ' + in_wspace
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    lists_count, tomos_count = wspace[0], wspace[1]
    lists_hash, tomos_hash = wspace[2], wspace[3]
    tomos_exp_dsts, tomos_sim_dsts, = wspace[4], wspace[5],
    lists_exp_dsts, lists_sim_dsts, lists_color = wspace[6], wspace[7], wspace[8]

print '\tPrinting lists hash: '
for id, lkey in zip(lists_hash.iterkeys(), lists_hash.itervalues()):
    print '\t\t-[' + str(id) + '] -> [' + lkey + ']'
print '\tPrinting tomograms hash: '
for tkey, val in zip(tomos_hash.iterkeys(), tomos_hash.itervalues()):
    print '\t\t-[' + tkey + '] -> [' + str(val) + ']'

# Getting the lists colormap
n_lists = len(lists_hash.keys())
for i, lkey in zip(lists_hash.iterkeys(), lists_hash.itervalues()):
    lists_color[lkey] = pt_cmap(1.*i/n_lists)

print '\tTOMOGRAMS PLOTTING LOOP: '

out_tomos_dir = out_stem_dir + '/tomos'
os.makedirs(out_tomos_dir)

print '\t\t-Plotting Function-H...'
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        try:
            hist_bins, hist_vals = compute_hist(arr, ana_nbins, ana_rmax)
            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            if len(tomo_sim_dsts) > 0:
                sims_hist_vals = list()
                for sim_dsts in tomo_sim_dsts:
                    sims_hist_vals.append(compute_hist(sim_dsts, ana_nbins, ana_rmax)[1])
                if len(sims_hist_vals) > 0:
                    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
                else:
                    raise ValueError
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            continue
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.title('Histogram 1st order for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Frequency')
        plt.xlabel('Scale (nm)')
        plt.plot(hist_bins, hist_vals, color=lists_color[lkey], label=lkey)
        plt.legend(loc=4)
        plt.plot(hist_bins, ic_low, 'k--')
        plt.plot(hist_bins, ic_med, 'k')
        plt.plot(hist_bins, ic_high, 'k--')
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey_short
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
            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            if len(tomo_sim_dsts) > 0:
                sims_cdf_vals = list()
                for sim_dsts in tomo_sim_dsts:
                    sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1])
                if len(sims_cdf_vals) > 0:
                    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
                    pvals_glow[tkey], pvals_ghigh[tkey] = compute_pvals(cdf_vals, np.asarray(sims_cdf_vals))
                else:
                    raise ValueError
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            continue
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.title('Univariate 1st order for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Function-G')
        plt.xlabel('Scale (nm)')
        plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], label=lkey)
        plt.legend(loc=4)
        plt.plot(cdf_bins, ic_low, 'k--')
        plt.plot(cdf_bins, ic_med, 'k')
        plt.plot(cdf_bins, ic_high, 'k--')
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey_short
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/G_' + lkey + '.png')
        plt.close()

print '\t\t-Plotting Function-G p-values (tomo[s_low, s_high]= (min_pval, max_pval)):'
min_pvals, max_pvals = list(), list()
for tkey, pvals_low, pvals_high in zip(pvals_glow.iterkeys(), pvals_glow.itervalues(), pvals_ghigh.itervalues()):
    pvals_hmin, pvals_hmax = np.copy(pvals_low), np.copy(pvals_high)
    idx_min, idx_max = np.argmax(pvals_hmin), np.argmax(pvals_hmax)
    min_pval, max_pval = pvals_hmin[idx_min], pvals_hmax[idx_max]
    print '\t\t\t+' + tkey + '[' + str(cdf_bins[idx_min]) + ', ' + str(cdf_bins[idx_max]) + ']= (' + \
          str(min_pval) + ', ' + str(max_pval) + ')'
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

print '\tLISTS PLOTTING LOOP: '

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

print '\t\t-Plotting Function-G...'
for lkey, ltomo in zip(lists_exp_dsts.iterkeys(), lists_exp_dsts.itervalues()):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    plt.figure()
    plt.title('Univariate 1st order for ' + lkey_short)
    plt.ylabel('Function-G')
    plt.xlabel('Scale (nm)')
    cdf_bins, cdf_vals = compute_cdf(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
    plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey])
    try:
        list_sim_dsts = lists_sim_dsts[lkey]
        if len(list_sim_dsts) > 0:
            sims_cdf_vals = list()
            for sim_dsts in list_sim_dsts:
                hold_vals = compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1]
                if np.isnan(hold_vals).sum() == 0:
                    sims_cdf_vals.append(hold_vals)
            ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
            plt.plot(cdf_bins, ic_low, 'k--')
            plt.plot(cdf_bins, ic_med, 'k')
            plt.plot(cdf_bins, ic_high, 'k--')
    except IndexError:
        print '\t\t\t+WARNING: no simulations for tomogram: ' + lkey
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/G_' + lkey_short + '.png')
    plt.close()

print '\t\t-Plotting Function-H...'
for lkey, ltomo in zip(lists_exp_dsts.iterkeys(), lists_exp_dsts.itervalues()):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    plt.figure()
    plt.title('Histogram 1st order for ' + lkey_short)
    plt.ylabel('Frequency')
    plt.xlabel('Scale (nm)')
    hist_bins, hist_vals = compute_hist(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
    plt.plot(hist_bins, hist_vals, color=lists_color[lkey])
    try:
        list_sim_dsts = lists_sim_dsts[lkey]
        if len(list_sim_dsts) > 0:
            sims_hist_vals = list()
            for sim_dsts in list_sim_dsts:
                hold_vals = compute_hist(sim_dsts, ana_nbins, ana_rmax)[1]
                if np.isnan(hold_vals).sum() == 0:
                    sims_hist_vals.append(hold_vals)
            ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
            plt.plot(cdf_bins, ic_low, 'k--')
            plt.plot(cdf_bins, ic_med, 'k')
            plt.plot(cdf_bins, ic_high, 'k--')
    except IndexError:
        print '\t\t\t+WARNING: no simulations for tomogram: ' + lkey
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/H_' + lkey_short + '.png')
    plt.close()

print 'Successfully terminated. (' + time.strftime("%c") + ')'

