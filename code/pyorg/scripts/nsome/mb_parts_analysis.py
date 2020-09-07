"""

    Mesaures metrics to analyze membrane particles relationships

    Input:  - A STAR file with a set of ListTomoParticles pickles (SetListTomograms object input)
                + Each TomoParticles object must contain the membrane segmentation as meta-information indexes by the
                  key omMbSegmentation

    Output: - Plots by tomograms:
                + Plots by list:
                    * Particle-membrane nearest distance
                    * TODO: metric to determine if particles are aligned with membrane normal
            - Global plots:
                + Global plots by list

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

ROOT_PATH = '/fs/pool/pool-plitzko/Peng_Xu/Antonio/test_dst'

# Input STAR files
in_star = ROOT_PATH + '/ltomos/test_nointer/test_1_ltomos.star'
in_wspace = ROOT_PATH + '/dsts/tests/test_4_sim200_wspace.pkl' # (Insert a path to recover a pickled workspace instead of doing a new computation)

# Output directory
out_dir = ROOT_PATH + '/dsts/tests'
out_stem = 'test_4_sim200_2'

# Analysis variables
ana_res = 0.896 # nm/vx - resolution
ana_nbins = 30
ana_nbins_dsts = 10
ana_dmax = 60 # nm
ana_normal_v = (1., .0, .0)
ana_axis = True # True

# Simulation settings
sm_ns = 200 # 10
sm_npr = 10 # 10

# Figure saving options
fig_fmt = '.png' # if None they showed instead

# Plotting options
pt_per = 95 # %
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

print('Membrane-particles relationhips analysis for a ListTomoParticles by tomograms.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tInput STAR file: ' + str(in_star))
if in_wspace is not None:
    print('\tLoad workspace from: ' + in_wspace)
print('\tList pre-processing options: ')
print('\tOrganization analysis settings: ')
print('\t\t-Pixel size: ' + str(ana_res) + ' nm/px')
print('\t\t-Number of bins: ' + str(ana_nbins))
print('\t\t-Number of bins for Angle-Distance plots: ' + str(ana_nbins_dsts))
print('\t\t-Maximum distances: ' + str(ana_dmax) + ' nm')
print('\t\t-Analysis reference vector: ' + str(ana_normal_v))
ang_max = 180
if ana_axis:
    print('\t\t-Mode axis activated.')
    ang_max = 90
print('\t\t-Number of processors for models simulation: ' + str(sm_ns))
print('\tP-Value computation setting:')
print('\t\t-Percentile: ' + str(pt_per) + ' %')
print('\t\t-Number of instances for simulations: ' + str(sm_ns))
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
sim_ids = np.array_split(list(range(sm_ns)), np.ceil(float(sm_ns)/sm_npr))

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
    set_lists, lists_dic_rows = surf.SetListTomoParticles(), dict()
    for row in range(star.get_nrows()):
        ltomos_pkl = star.get_element('_psPickleFile', row)
        ltomos = unpickle_obj(ltomos_pkl)
        set_lists.add_list_tomos(ltomos, ltomos_pkl)
        fkey = os.path.split(ltomos_pkl)[1]
        short_key_idx = fkey.index('_')
        short_key = fkey[:short_key_idx]
        lists_dic_rows[short_key] = row

    print('\tBuilding the dictionaries...')
    lists_count, tomos_count = 0, 0
    lists_dic = dict()
    lists_hash, tomos_hash = dict(), dict()
    tomos_exp_dsts, tomos_sim_dsts = dict(), dict()
    tomos_exp_angs, tomos_sim_angs = dict(), dict()
    lists_exp_dsts, lists_sim_dsts, lists_color = dict(), dict(), dict()
    lists_exp_angs, lists_sim_angs = dict(), dict()
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
            lists_exp_angs[short_key], lists_sim_angs[short_key] = list(), list()
            lists_count += 1
    for lkey, llist in zip(iter(set_lists_dic.keys()), iter(set_lists_dic.values())):
        llist_tomos_dic = llist.get_tomos()
        for tkey, ltomo in zip(iter(llist_tomos_dic.keys()), iter(llist_tomos_dic.values())):
            try:
                tomos_exp_dsts[tkey]
            except KeyError:
                tomos_hash[tkey] = tomos_count
                tomos_exp_dsts[tkey], tomos_sim_dsts[tkey] = dict.fromkeys(list(lists_dic.keys())), dict.fromkeys(list(lists_dic.keys()))
                tomos_exp_angs[tkey], tomos_sim_angs[tkey] = dict.fromkeys(list(lists_dic.keys())), dict.fromkeys(list(lists_dic.keys()))
                tomos_count += 1
    for tkey in tomos_exp_dsts.keys():
        for lkey in lists_dic.keys():
            tomos_exp_dsts[tkey][lkey], tomos_exp_angs[tkey][lkey] = list(), list()
            tomos_sim_dsts[tkey][lkey], tomos_sim_angs[tkey][lkey] = list(), list()

    print('\tLIST COMPUTING LOOP:')
    sim_obj_set = surf.SetListSimulations()
    for lkey in lists_hash.values():

        llist = lists_dic[lkey]
        sim_obj_list = surf.ListSimulations()
        print('\t\t-Processing list: ' + lkey)
        hold_row = lists_dic_rows[lkey]
        try:
            p_vtp = star.get_element('_suSurfaceVtp', hold_row)
        except KeyError:
            print('ERROR: _suSurfaceVtp is required in the input STAR file "' + in_star + '"')
            print('Terminated. (' + time.strftime("%c") + ')')
            sys.exit(-1)
        surf_p_vtp = disperse_io.load_poly(p_vtp)
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
            in_mb_seg = ltomo.get_meta_info('_omMbSegmentation')
            if in_mb_seg is None:
                print('\t\t\t\t\t-WARNING: no membrane segmentation path for this tomogram...')
                continue
            try:
                mb_seg = disperse_io.load_tomo(in_mb_seg)
            except pexceptions.PySegError:
                print('ERROR: the input file "' + in_mb_seg + '" could not be loaded.')
                print('Terminated. (' + time.strftime("%c") + ')')
                sys.exit(-1)

            print('\t\t\t\t\t-Computing nearest membrane-particles distance...')
            hold_mb_dsts, hold_mb_angs = ltomo.compute_mb_seg_distances(mb_seg, do_ang=('normal_v', (.0, .0, .1),
                                                                                        ana_axis))
            hold_mb_dsts *= ana_res
            if hold_mb_dsts is not None:
                lists_exp_dsts[lkey].append(hold_mb_dsts)
                tomos_exp_dsts[tkey][lkey].append(hold_mb_dsts)
                lists_exp_angs[lkey].append(hold_mb_angs)
                tomos_exp_angs[tkey][lkey].append(hold_mb_angs)
            out_fils = out_dir + '/' + out_stem + '/' + str(lkey) + '_' + tkey.replace('/', '_') + '_parts.vtp'
            disperse_io.save_vtp(ltomo.gen_particles_vtp(), out_fils)

            print('\t\t\t\t\t-Generating the simulated instances...')
            for ids in sim_ids:
                temp_model = ModelCSRV(vect=ana_normal_v)
                # temp_model2 = ModelRR()
                temp_model.set_ParticleL_ref(p_vtp)
                # temp_model2.set_ParticleL_ref(p_vtp)
                sim_tomos = gen_tlist(len(ids), ltomo.get_num_particles(), temp_model, ltomo.get_voi(), p_vtp,
                                      mode_emb='center', npr=sm_npr)
                # sim_tomos2 = gen_tlist(len(ids), ltomo.get_num_particles(), temp_model2, ltomo.get_voi(), p_vtp,
                #                        mode_emb='center', npr=ana_npr_model, in_coords=ltomo.get_particle_coords())
                hold_mat_dsts, hold_mat_fdsts = list(), list()
                for sim_i, sim_tomo in enumerate(sim_tomos.get_tomo_list()):
                    hold_mb_dsts, hold_mb_angs = sim_tomo.compute_mb_seg_distances(mb_seg,
                                                                                   do_ang=('normal_v', (.0, .0, .1),
                                                                                           ana_axis))
                    hold_mb_dsts *= ana_res
                    tomos_sim_dsts[tkey][lkey].append(hold_mb_dsts)
                    tomos_sim_angs[tkey][lkey].append(hold_mb_angs)
                    for i in range(ltomo.get_num_particles()):
                        lists_sim_dsts[lkey].append(hold_mb_dsts)
                        lists_sim_angs[lkey].append(hold_mb_angs)
                    if sim_i == 0:
                        out_fils = out_dir + '/' + out_stem + '/' + str(lkey) + '_' + \
                                   tkey.replace('/', '_') + '_parts_sim.vtp'
                        disperse_io.save_vtp(sim_tomo.gen_particles_vtp(), out_fils)

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print('\tPickling computation workspace in: ' + out_wspace)
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash, lists_color,
              tomos_exp_dsts, tomos_sim_dsts,
              lists_exp_dsts, lists_sim_dsts,
              tomos_exp_angs, tomos_sim_angs,
              lists_exp_angs, lists_sim_angs)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:
    print('\tLoading the workspace: ' + in_wspace)
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    lists_count, tomos_count = wspace[0], wspace[1]
    lists_hash, tomos_hash, lists_color = wspace[2], wspace[3], wspace[4]
    tomos_exp_dsts, tomos_sim_dsts = wspace[5], wspace[6]
    lists_exp_dsts, lists_sim_dsts = wspace[7], wspace[8]
    tomos_exp_angs, tomos_sim_angs = wspace[9], wspace[10]
    lists_exp_angs, lists_sim_angs = wspace[11], wspace[12]

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

print('\t\t-Plotting Protein-Membrane distances histogram...')
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(iter(tomos_exp_dsts.keys()), iter(tomos_exp_dsts.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(iter(tomos_exp_dsts[tkey].keys()), iter(tomos_exp_dsts[tkey].values())):
        try:
            hist_bins, hist_vals = compute_hist(arr, ana_nbins, ana_dmax)
            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            sims_hist_vals = list()
            for sim_dsts in tomo_sim_dsts:
                sims_hist_vals.append(compute_hist(sim_dsts, ana_nbins, ana_dmax)[1])
            if len(sims_hist_vals) > 0:
                ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(sims_hist_vals))
            else:
                raise ValueError
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            # continue
            pass
        lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
        if lkey_short == '0':
            lkey_short = '2k'
        elif lkey_short == '0':
            lkey_short = '3k'
        elif lkey_short == '0':
            lkey_short = '6k'
        plt.figure()
        # plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Probability Density')
        plt.xlabel('Filament-Membrane Nearest Distance [nm]')
        plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
        plt.legend(loc=4)
        # plt.plot(hist_bins, ic_low, 'k--')
        plt.plot(hist_bins, ic_med, 'k', linewidth=2.0, label='SIM')
        # plt.plot(hist_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        plt.legend(loc=1)
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

print('\t\t-Plotting Cumulative Protein-Membrane distance...')
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(iter(tomos_exp_dsts.keys()), iter(tomos_exp_dsts.values())):
    for lkey, arr in zip(iter(tomos_exp_dsts[tkey].keys()), iter(tomos_exp_dsts[tkey].values())):
        try:
            cdf_bins, cdf_vals = compute_cdf(arr, ana_nbins, ana_dmax)
            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            sims_cdf_vals = list()
            for sim_dsts in tomo_sim_dsts:
                sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_dmax)[1])
            if len(sims_cdf_vals) > 0:
                ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(sims_cdf_vals))
            else:
                raise ValueError
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            continue
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
        if lkey_short == '0':
            lkey_short = '2k'
        elif lkey_short == '0':
            lkey_short = '3k'
        elif lkey_short == '0':
            lkey_short = '6k'
        plt.figure()
        # plt.title('Univariate 1st order for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Cumulative Density')
        plt.xlabel('Nucleosomes-Membrane Nearest Distance [nm]')
        plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], label=lkey_short, linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(cdf_bins, ic_low, 'k--')
        plt.plot(cdf_bins, ic_med, 'k', linewidth=2.0, label='SIM')
        # plt.plot(cdf_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        plt.legend(loc=4)
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

print('\t\t-Plotting Axis-Angles Histogram...')
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(iter(tomos_exp_angs.keys()), iter(tomos_exp_angs.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(iter(tomos_exp_angs[tkey].keys()), iter(tomos_exp_angs[tkey].values())):
        try:
            hist_bins, hist_vals = compute_hist(arr, ana_nbins, ang_max)
            tomo_sim_angs = tomos_sim_angs[tkey][lkey]
            sims_hist_vals = list()
            for sim_dsts in tomo_sim_angs:
                sims_hist_vals.append(compute_hist(sim_dsts, ana_nbins, ang_max)[1])
            if len(sims_hist_vals) > 0:
                ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(sims_hist_vals))
            else:
                raise ValueError
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            # continue
            pass
        lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
        if lkey_short == '0':
            lkey_short = '2k'
        elif lkey_short == '0':
            lkey_short = '3k'
        elif lkey_short == '0':
            lkey_short = '6k'
        plt.figure()
        # plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Probability Density')
        plt.xlabel('Z Axis-Particle angle [deg]')
        plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
        plt.legend(loc=4)
        # plt.plot(hist_bins, ic_low, 'k--')
        plt.plot(hist_bins, ic_med, 'k', linewidth=2.0, label='SIM')
        # plt.plot(hist_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        plt.xlim(0, ang_max)
        plt.legend(loc=1)
        plt.tight_layout()
        plt.grid(True)
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/H_angs_' + lkey + fig_fmt)
        plt.close()

print('\t\t-Plotting Axis-Angles by distances...')
for tkey, ltomo in zip(iter(tomos_exp_dsts.keys()), iter(tomos_exp_dsts.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(iter(tomos_exp_dsts[tkey].keys()), iter(tomos_exp_dsts[tkey].values())):
        arr = arr[0]
        hist_vals_dsts, hist_bins_dsts = np.histogram(arr, bins=ana_nbins_dsts) #, range=[arr.min(), ana_dmax])
        tomo_exp_angs, tomo_sim_angs = tomos_exp_angs[tkey][lkey], tomos_sim_angs[tkey][lkey]
        dsts_vals, bin_l = list(), arr.min()
        ic_exp_low, ic_exp_med, ic_exp_high = list(), list(), list()
        ic_sim_low, ic_sim_med, ic_sim_high = list(), list(), list()
        for bin_h in hist_bins_dsts[1:]:
            hold_ids = np.where((arr >= bin_l) & (arr < bin_h))[0]
            hold_vals = list()
            for idx in hold_ids:
                hold_vals.append(tomos_exp_angs[tkey][lkey][0][idx])
            if len(hold_vals) == 0:
                ic_exp_low.append(0)
                ic_exp_med.append(0)
                ic_exp_high.append(0)
            else:
                ic_exp_low.append(np.percentile(hold_vals, 100-pt_per))
                ic_exp_med.append(np.percentile(hold_vals, 50))
                ic_exp_high.append(np.percentile(hold_vals, pt_per))
            hold_vals = list()
            for sim_dsts in tomo_sim_angs:
                for idx in hold_ids:
                    hold_vals.append(sim_dsts[idx])
            if len(hold_vals) == 0:
                ic_sim_low.append(0)
                ic_sim_med.append(0)
                ic_sim_high.append(0)
            else:
                ic_sim_low.append(np.percentile(hold_vals, 100-pt_per))
                ic_sim_med.append(np.percentile(hold_vals, 50))
                ic_sim_high.append(np.percentile(hold_vals, pt_per))
            dsts_vals.append(.5 * (bin_l + bin_h))
            bin_l = bin_h
        lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
        if lkey_short == '0':
            lkey_short = '2k'
        elif lkey_short == '0':
            lkey_short = '3k'
        elif lkey_short == '0':
            lkey_short = '6k'
        plt.figure()
        # plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Z Axis-Particle angle [deg]')
        plt.xlabel('Distance to membrane [nm]')
        plt.plot(dsts_vals, ic_exp_med, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
        plt.fill_between(dsts_vals, ic_exp_low, ic_exp_high, alpha=0.5, color=lists_color[lkey], edgecolor='w')
        plt.legend(loc=4)
        # plt.plot(hist_bins, ic_low, 'k--')
        plt.plot(dsts_vals, ic_sim_med, 'k', linewidth=2.0, label='SIM')
        # plt.plot(hist_bins, ic_high, 'k--')
        plt.fill_between(dsts_vals, ic_sim_low, ic_sim_high, alpha=0.5, color='gray', edgecolor='w')
        plt.ylim(0, ang_max)
        plt.legend(loc=1)
        plt.tight_layout()
        plt.grid(True)
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/H_angs_dsts_' + lkey + fig_fmt)
        plt.close()


print('\tLISTS PLOTTING LOOP: ')

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

print('\t\t-Plotting Histogram...')
for lkey, ltomo in zip(iter(lists_exp_dsts.keys()), iter(lists_exp_dsts.values())):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    try:
        hist_bins, hist_vals = compute_hist(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_dmax)
        list_sim_dsts = lists_sim_dsts[lkey]
        sims_hist_vals = list()
        for sim_dsts in list_sim_dsts:
            sims_hist_vals.append(compute_hist(sim_dsts, ana_nbins, ana_dmax)[1])
        if len(sims_hist_vals) > 0:
            ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(sims_hist_vals))
        else:
            raise ValueError
    except ValueError or IndexError:
        print(np.concatenate(np.asarray(ltomo)))
        print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
        continue
    if lkey_short == '0':
        lkey_short = '2k'
    elif lkey_short == '0':
        lkey_short = '3k'
    elif lkey_short == '0':
        lkey_short = '6k'
    plt.figure()
    # plt.title('Nearest distance histogram for ' + lkey_short)
    plt.ylabel('Probability Density')
    plt.xlabel('Nucleasome-Membrane Nearest Distance [nm]')
    plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
    # plt.plot(hist_bins, ic_low, 'k--')
    plt.plot(hist_bins, ic_med, 'k', linewidth=2.0, label='SIM')
    # plt.plot(hist_bins, ic_high, 'k--')
    plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.legend(loc=1)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/H_' + lkey_short + '_all' + fig_fmt, dpi=600)
    plt.close()

print('\t\t-Plotting the CDF...')
for lkey, ltomo in zip(iter(lists_exp_dsts.keys()), iter(lists_exp_dsts.values())):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    try:
        cdf_bins, cdf_vals = compute_cdf(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_dmax)
        list_sim_dsts = lists_sim_dsts[lkey]
        sims_cdf_vals = list()
        for sim_dsts in list_sim_dsts:
            sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_dmax)[1])
        if len(sims_cdf_vals) > 0:
            ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(sims_cdf_vals))
        else:
            raise ValueError
    except ValueError or IndexError:
        print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
        continue
    if lkey_short == '0':
        lkey_short = '2k'
    elif lkey_short == '0':
        lkey_short = '3k'
    elif lkey_short == '0':
        lkey_short = '6k'
    plt.figure()
    # plt.title('Univariate 1st order for ' + lkey_short)
    plt.ylabel('Cumulative Density')
    plt.xlabel('Nucleasome-Membrane Nearest Distance [nm]')
    plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
    # plt.plot(cdf_bins, ic_low, 'k--')
    plt.plot(cdf_bins, ic_med, 'k', linewidth=2.0, label='SIM')
    # plt.plot(cdf_bins, ic_high, 'k--')
    plt.fill_between(cdf_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/CDF_' + lkey_short + '_all' + fig_fmt, dpi=600)
    plt.close()

print('\t\t-Plotting Histogram with tomos IC...')
plt.figure()
# plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
plt.ylabel('Probability Density')
plt.xlabel('Nucleasome-Membrane Nearest Distance [nm]')
sim_hist_all = list()
exp_hist_dic, sim_hist_dic = dict().fromkeys(list(lists_exp_dsts.keys())), dict().fromkeys(list(lists_exp_dsts.keys()))
for lkey in exp_hist_dic.keys():
    exp_hist_dic[lkey], sim_hist_dic[lkey] = list(), list()
exp_cdf_dic, sim_cdf_dic = dict().fromkeys(list(lists_exp_dsts.keys())), dict().fromkeys(list(lists_exp_dsts.keys()))
exp_acc_dic, sim_acc_dic = dict().fromkeys(list(lists_exp_dsts.keys())), dict().fromkeys(list(lists_exp_dsts.keys()))
for lkey in exp_cdf_dic.keys():
    exp_cdf_dic[lkey], sim_cdf_dic[lkey], exp_acc_dic[lkey], sim_acc_dic[lkey] = list(), list(), list(), list()
for tkey, ltomo in zip(iter(tomos_exp_dsts.keys()), iter(tomos_exp_dsts.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    exps_hist_vals = list()
    for lkey, arr in zip(iter(tomos_exp_dsts[tkey].keys()), iter(tomos_exp_dsts[tkey].values())):
        if len(arr) <= 0:
            continue
        try:
            hist_bins, hist_vals = compute_hist(arr[0], ana_nbins, ana_dmax)
            exp_hist_dic[lkey].append(hist_vals)
            for hold_sim in tomos_sim_dsts[tkey][lkey]:
                hist_bins, hist_vals = compute_hist(hold_sim, ana_nbins, ana_dmax)
                sim_hist_dic[lkey].append(hist_vals)
                sim_hist_all.append(hist_vals)
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            # continue
            pass
for lkey in exp_hist_dic.keys():
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    if lkey_short == '0':
        lkey_short = '2k'
    elif lkey_short == '1':
        lkey_short = '3k'
    elif lkey_short == '2':
        lkey_short = '6k'
    ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(exp_hist_dic[lkey]))
    plt.plot(hist_bins, ic_med, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
    # plt.plot(hist_bins, ic_low, 'k--')
    # plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
    # plt.plot(hist_bins, ic_high, 'k--')
    plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color=lists_color[lkey], edgecolor='w')
ic_low_sim, ic_med_sim, ic_high_sim = compute_ic(pt_per, np.asarray(sim_hist_all))
plt.plot(hist_bins, ic_med_sim, color='black', linewidth=2.0, label='SIM')
plt.fill_between(hist_bins, ic_low_sim, ic_high_sim, alpha=0.5, color='gray', edgecolor='w')
plt.legend(loc=1)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/H_lists' + fig_fmt, dpi=600)
plt.close()

print('\t\t-Plotting the CDF with tomos IC...')
sim_cdf_all = list()
plt.figure()
# plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
plt.ylabel('Cumulative Density')
plt.xlabel('Nucleasome-Membrane Nearest Distance [nm]')
for tkey, ltomo in zip(iter(tomos_exp_dsts.keys()), iter(tomos_exp_dsts.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    exps_hist_vals = list()
    for lkey, arr in zip(iter(tomos_exp_dsts[tkey].keys()), iter(tomos_exp_dsts[tkey].values())):
        if len(arr) <= 0:
            continue
        try:
            cdf_bins, cdf_vals = compute_cdf(arr, ana_nbins, ana_dmax)
            exp_cdf_dic[lkey].append(cdf_vals)
            for hold_sim in tomos_sim_dsts[tkey][lkey]:
                cdf_bins, cdf_vals = compute_cdf(hold_sim, ana_nbins, ana_dmax)
                sim_cdf_dic[lkey].append(cdf_vals)
                sim_cdf_all.append(cdf_vals)
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            # continue
            pass
for lkey in exp_hist_dic.keys():
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    if lkey_short == '0':
        lkey_short = '2k'
    elif lkey_short == '1':
       lkey_short = '3k'
    elif lkey_short == '2':
       lkey_short = '6k'
    ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(exp_cdf_dic[lkey]))
    plt.plot(hist_bins, ic_med, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
    # plt.plot(hist_bins, ic_low, 'k--')
    # plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
    # plt.plot(hist_bins, ic_high, 'k--')
    plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color=lists_color[lkey], edgecolor='w')
ic_low_sim, ic_med_sim, ic_high_sim = compute_ic(pt_per, np.asarray(sim_cdf_all))
plt.plot(hist_bins, ic_med_sim, color='black', linewidth=2.0, label='SIM')
plt.fill_between(hist_bins, ic_low_sim, ic_high_sim, alpha=0.5, color='gray', edgecolor='w')
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/CDF_lists' + fig_fmt, dpi=600)
plt.close()

print('\t\t-Plotting Axis-Angle Histogram with tomos IC...')
plt.figure()
# plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
plt.ylabel('Probability Density')
plt.xlabel('Z Axis-Particle angle [deg]')
sim_hist_all = list()
exp_hist_dic, sim_hist_dic = dict().fromkeys(list(lists_exp_angs.keys())), dict().fromkeys(list(lists_exp_angs.keys()))
for lkey in exp_hist_dic.keys():
    exp_hist_dic[lkey], sim_hist_dic[lkey] = list(), list()
for tkey, ltomo in zip(iter(tomos_exp_angs.keys()), iter(tomos_exp_angs.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    exps_hist_vals = list()
    for lkey, arr in zip(iter(tomos_exp_angs[tkey].keys()), iter(tomos_exp_angs[tkey].values())):
        if len(arr) <= 0:
            continue
        try:
            hist_bins, hist_vals = compute_hist(arr[0], ana_nbins, ang_max)
            exp_hist_dic[lkey].append(hist_vals)
            for hold_sim in tomos_sim_angs[tkey][lkey]:
                hist_bins, hist_vals = compute_hist(hold_sim, ana_nbins, ang_max)
                sim_hist_dic[lkey].append(hist_vals)
                sim_hist_all.append(hist_vals)
        except ValueError or IndexError:
            print('\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey)
            # continue
            pass
for lkey in exp_hist_dic.keys():
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    if lkey_short == '0':
        lkey_short = '2k'
    elif lkey_short == '1':
        lkey_short = '3k'
    elif lkey_short == '2':
        lkey_short = '6k'
    ic_low, ic_med, ic_high = compute_ic(pt_per, np.asarray(exp_hist_dic[lkey]))
    plt.plot(hist_bins, ic_med, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
    # plt.plot(hist_bins, ic_low, 'k--')
    # plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
    # plt.plot(hist_bins, ic_high, 'k--')
    plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color=lists_color[lkey], edgecolor='w')
ic_low_sim, ic_med_sim, ic_high_sim = compute_ic(pt_per, np.asarray(sim_hist_all))
plt.plot(hist_bins, ic_med_sim, color='black', linewidth=2.0, label='SIM')
plt.fill_between(hist_bins, ic_low_sim, ic_high_sim, alpha=0.5, color='gray', edgecolor='w')
plt.xlim(0, ang_max)
plt.legend(loc=1)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/H_angs_lists' + fig_fmt, dpi=600)
plt.close()

print('\t\t-Plotting Axis-Angles by distances...')
for lkey, arr in zip(iter(lists_exp_dsts.keys()), iter(lists_exp_dsts.values())):
    arr = arr[0]
    hist_vals_dsts, hist_bins_dsts = np.histogram(arr, bins=ana_nbins_dsts) #, range=[arr.min(), ana_dmax])
    exp_angs, sim_angs = lists_exp_angs[lkey], lists_sim_angs[lkey]
    dsts_vals, bin_l = list(), arr.min()
    ic_exp_low, ic_exp_med, ic_exp_high = list(), list(), list()
    ic_sim_low, ic_sim_med, ic_sim_high = list(), list(), list()
    for bin_h in hist_bins_dsts[1:]:
        hold_ids = np.where((arr >= bin_l) & (arr < bin_h))[0]
        hold_vals = list()
        for idx in hold_ids:
            hold_vals.append(exp_angs[0][idx])
        if len(hold_vals) == 0:
            ic_exp_low.append(0)
            ic_exp_med.append(0)
            ic_exp_high.append(0)
        else:
            ic_exp_low.append(np.percentile(hold_vals, 100-pt_per))
            ic_exp_med.append(np.percentile(hold_vals, 50))
            ic_exp_high.append(np.percentile(hold_vals, pt_per))
        hold_vals = list()
        for sim_dsts in sim_angs:
            for idx in hold_ids:
                hold_vals.append(sim_dsts[idx])
        if len(hold_vals) == 0:
            ic_sim_low.append(0)
            ic_sim_med.append(0)
            ic_sim_high.append(0)
        else:
            ic_sim_low.append(np.percentile(hold_vals, 100-pt_per))
            ic_sim_med.append(np.percentile(hold_vals, 50))
            ic_sim_high.append(np.percentile(hold_vals, pt_per))
        dsts_vals.append(.5 * (bin_l + bin_h))
        bin_l = bin_h
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    if lkey_short == '0':
        lkey_short = '2k'
    elif lkey_short == '0':
        lkey_short = '3k'
    elif lkey_short == '0':
        lkey_short = '6k'
    plt.figure()
    # plt.title('Histogram Nearest distances for ' + tkey_short + ' and ' + lkey)
    plt.ylabel('Z Axis-Particle angle [deg]')
    plt.xlabel('Distance to membrane [nm]')
    plt.plot(dsts_vals, ic_exp_med, color=lists_color[lkey], linewidth=2.0, label=lkey_short)
    plt.fill_between(dsts_vals, ic_exp_low, ic_exp_high, alpha=0.5, color=lists_color[lkey], edgecolor='w')
    plt.legend(loc=4)
    # plt.plot(hist_bins, ic_low, 'k--')
    plt.plot(dsts_vals, ic_sim_med, 'k', linewidth=2.0, label='SIM')
    # plt.plot(hist_bins, ic_high, 'k--')
    plt.fill_between(dsts_vals, ic_sim_low, ic_sim_high, alpha=0.5, color='gray', edgecolor='w')
    plt.ylim(0, ang_max)
    plt.legend(loc=1)
    plt.tight_layout()
    # plt.grid(True)
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/H_angs_dsts_lists' + lkey + fig_fmt)
    plt.close()

print('Successfully terminated. (' + time.strftime("%c") + ')')