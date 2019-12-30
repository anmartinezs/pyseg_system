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

ROOT_PATH = '/fs/pool/pool-engel/antonio/rubisco/org'

# Input STAR files
in_star = ROOT_PATH + '/ltomos/all_v2/all_v2_ltomos_surf.star' # '/ltomos/L2Tomo8/L2Tomo8_ltomos_surf.star' # '/ltomos/all_pid/all_pid_ltomos_surf.star'
in_wspace = None # ROOT_PATH + '/uni_1st/L2Tomo8/L2Tomo8_align_100_30_mxcon_1_rg_min_neg_dst_0_low_tol_neg3_wspace.pkl'  # None # (Insert a path to recover a pickled workspace instead of doing a new computation)

# Output directory
out_dir = ROOT_PATH + '/uni_1st/all_v2/' # '/uni_1st/L2Tomo8/'
out_stem = 'all_v2_align_100_30_mxcon_1_rg_min_neg_dst_0_low_tol_abs'

# List pre-processing options
pr_ss = None  # 10 # nm

# Analysis variables
ana_res = 1.368  # 0.684 # nm/voxel
ana_nbins = 100
ana_rmax = 30  # nm
ana_f_npoints = 50000  # 1000
ana_npr_model = 1
# Required for function-NNS computation, only if columns '_psStartSurfIds' and '_psEndSurfIds' are present
# in the input STAR file
ana_ndst_rg = [0, 25]  # nm
ana_ssize = None  # 5 # nm
ana_mx_conn = 1  # None
ana_del_border = 12 # nm
ana_min_neg_dst = 0 # 2 # nm

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = None # 1 # 3
p_per = 5  # %
p_jhigh = 0.98
# Particle surface
p_vtp = None  # Only required in not include in the input STAR file # '/fs/pool/pool-engel/antonio/ribo/in/yeast_80S/yeast_80S_21A_mirr_centered.vtp'

# Figure saving options
fig_fmt = '.png'  # if None they showed instead

# Plotting options
pt_sim_v = True
pt_cmap = plt.get_cmap('gist_rainbow')


########################################################################################
# ADDITIONAL ROUTINES
########################################################################################

# Compute Histogram Function from a one-dimensional array of random samples
# var: array of stochastic samples
# n:
# mx:
# Returns: cdf values and samples respectively
def compute_hist_nns(var, n, mn, mx):
    """
    Compute Histogram Function from a one-dimensional array of random samples giving the frequency value (function-NNS)
    :param var: array of stochastic samples
    :param n: number of samples for cdf, if n is a sequence it defines the bin edges, including rightmost edge
    :param mn: minimum value fot the histogram
    :param mx: maximum value for the histogram
    :return:
    """
    bins = np.linspace(mn, mx, n + 1)
    hist, _ = np.histogram(var, bins=bins, density=True)
    hist[np.isnan(hist)] = 0
    return .5 * (bins[1]-bins[0]) + bins[:-1], hist


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


########################################################################################
# MAIN ROUTINE
########################################################################################


########## Print initial message

print 'Univariate first order analysis for a ListTomoParticles by tomograms.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
print '\tInput STAR file: ' + str(in_star)
if in_wspace is not None:
    print '\tLoad workspace from: ' + in_wspace
print '\tList pre-processing options: '
if pr_ss is not None:
    pr_ss_v = pr_ss / ana_res
    print '\t\t-Scale suppression: ' + str(pr_ss) + ' nm (' + str(pr_ss_v) + ' vx)'
print '\tOrganization analysis settings: '
print '\t\t-Number of bins: ' + str(ana_nbins)
ana_rmax_v = float(ana_rmax) / ana_res
print '\t\t-Maximum radius: ' + str(ana_rmax) + ' nm (' + str(ana_rmax_v) + ' vx)'
print '\t\t-Number of CSR points for F-function: ' + str(ana_f_npoints)
print '\t\t-Settings for Funcion-NNS: '
ana_ndst_rg_v = [ana_ndst_rg[0] / ana_res, ana_ndst_rg[1] / ana_res]
print '\t\t\t+Distances range: ' + str(ana_ndst_rg) + ' nm (' + str(ana_ndst_rg_v) + ' vx)'
if ana_ssize is not None:
    ana_ssize_v = ana_ssize / ana_res
    print '\t\t\t+Inter-surfaces distance step: ' + str(ana_ssize) + ' nm (' + str(ana_ssize_v) + ' vx)'
if ana_mx_conn is not None:
    print '\t\t\t+Maximum connections for every input point: ' + str(ana_mx_conn)
if ana_del_border > 0:
    print '\t\t\t+Point nearest to ' + str(ana_del_border) + ' nm to border are not considered.'
ana_del_border_v = ana_del_border / ana_res
if ana_min_neg_dst > 0:
    print '\t\t\t+Threshold to consider a negative distance ' + str(ana_min_neg_dst) + ' nm.'
ana_min_neg_dst_v = ana_min_neg_dst / ana_res
if p_nsims is not None:
    if ana_npr_model is None:
        ana_npr_model = mp.cpu_count()
    elif ana_npr_model > p_nsims:
        ana_npr_model = p_nsims
    sim_ids = np.array_split(range(p_nsims), np.ceil(float(p_nsims) / ana_npr_model))
    print '\t\t-Number of processors for models simulation: ' + str(ana_npr_model)
    print '\tP-Value computation setting:'
    print '\t\t-Percentile: ' + str(p_per) + ' %'
    print '\t\t-Highest value for Funciton-J: ' + str(p_jhigh)
    print '\t\t-Number of instances for simulations: ' + str(p_nsims)
if p_vtp is not None:
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
    star = sub.Star()
    try:
        star.load(in_star)
    except pexceptions.PySegInputError as e:
        print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
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
    if p_vtp is not None:
        try:
            part_vtp = disperse_io.load_poly(p_vtp)
        except pexceptions.PySegInputError as e:
            print 'ERROR: reference particle surface file could not be loaded because of "' + e.get_message() + '"'
            print 'Terminated. (' + time.strftime("%c") + ')'
            sys.exit(-1)

    if pr_ss is not None:
        print '\tApplying scale suppression...'
        set_lists.scale_suppression(pr_ss_v)

    print '\tBuilding the dictionaries...'
    lists_count, tomos_count = 0, 0
    save_sim, save_sim2 = True, True
    lists_dic = dict()
    lists_hash, tomos_hash = dict(), dict()
    tomos_exp_dsts, tomos_sim_dsts, tomos_exp_fdsts, tomos_sim_fdsts = dict(), dict(), dict(), dict()
    lists_exp_dsts, lists_sim_dsts, lists_exp_fdsts, lists_sim_fdsts, lists_color = dict(), dict(), dict(), dict(), dict()
    tomos_exp_mnnd, tomos_sim_mnnd, lists_exp_mnnd, lists_sim_mnnd = dict(), dict(), dict(), dict()
    tomos_exp_codes, tomos_sim_codes, lists_exp_codes, lists_sim_codes = dict(), dict(), dict(), dict()
    tomos_sim2_mnnd, lists_sim2_mnnd = dict(), dict()
    tomos_sim2_codes, lists_sim2_codes = dict(), dict()
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
            lists_exp_fdsts[short_key], lists_sim_fdsts[short_key] = list(), list()
            lists_exp_mnnd[short_key], lists_sim_mnnd[short_key], lists_sim2_mnnd[short_key] = list(), list(), list()
            lists_exp_codes[short_key], lists_sim_codes[short_key], lists_sim2_codes[short_key] = list(), list(), list()
            lists_count += 1
    for lkey, llist in zip(set_lists_dic.iterkeys(), set_lists_dic.itervalues()):
        llist_tomos_dic = llist.get_tomos()
        for tkey, ltomo in zip(llist_tomos_dic.iterkeys(), llist_tomos_dic.itervalues()):
            try:
                tomos_exp_dsts[tkey]
            except KeyError:
                tomos_hash[tkey] = tomos_count
                tomos_exp_dsts[tkey], tomos_sim_dsts[tkey] = dict.fromkeys(lists_dic.keys()), dict.fromkeys(
                    lists_dic.keys())
                tomos_exp_fdsts[tkey], tomos_sim_fdsts[tkey] = dict.fromkeys(lists_dic.keys()), dict.fromkeys(
                    lists_dic.keys())
                tomos_exp_mnnd[tkey], tomos_sim_mnnd[tkey] = dict.fromkeys(lists_dic.keys()), dict.fromkeys(
                    lists_dic.keys())
                tomos_exp_codes[tkey], tomos_sim_codes[tkey] = dict.fromkeys(lists_dic.keys()), dict.fromkeys(
                    lists_dic.keys())
                tomos_sim2_mnnd[tkey] = dict.fromkeys(lists_dic.keys())
                tomos_sim2_codes[tkey] = dict.fromkeys(lists_dic.keys())
                tomos_count += 1
    for tkey in tomos_exp_dsts.iterkeys():
        for lkey in lists_dic.iterkeys():
            tomos_exp_dsts[tkey][lkey], tomos_exp_fdsts[tkey][lkey] = list(), list()
            tomos_sim_dsts[tkey][lkey], tomos_sim_fdsts[tkey][lkey] = list(), list()
            tomos_exp_mnnd[tkey][lkey], tomos_sim_mnnd[tkey][lkey], tomos_sim2_mnnd[tkey][lkey] = None, list(), list()
            tomos_exp_codes[tkey][lkey], tomos_sim_codes[tkey][lkey], tomos_sim2_codes[tkey][lkey] = None, list(), list()

    print '\tLIST COMPUTING LOOP:'
    sim_obj_set = surf.SetListSimulations()
    for lkey in lists_hash.itervalues():

        llist = lists_dic[lkey]
        sim_obj_list = surf.ListSimulations()
        print '\t\t-Processing list: ' + lkey
        hold_row = lists_dic_rows[lkey]
        if p_vtp is None:
            try:
                p_vtp = star.get_element('_suSurfaceVtp', hold_row)
            except KeyError:
                print 'ERROR: _suSurfaceVtp is required in the input STAR file "' + in_star + '"'
                print 'Terminated. (' + time.strftime("%c") + ')'
                sys.exit(-1)
        surf_p_vtp = disperse_io.load_poly(p_vtp)
        pids_start, pids_end = list(), list()
        if star.has_column('_psStartSurfIds') and star.has_column('_psEndSurfIds'):
            print '\t\t\t+STAR files with contacts found.'
            in_pids_start = star.get_element('_psStartSurfIds', hold_row)
            in_pids_end = star.get_element('_psEndSurfIds', hold_row)
            star_pids_start, star_pids_end = sub.Star(), sub.Star()
            star_pids_start.load(in_pids_start)
            for hold_row in range(star_pids_start.get_nrows()):
                pids_start.append(star_pids_start.get_element('_psSurfacePID', hold_row))
            star_pids_end.load(in_pids_end)
            for hold_row in range(star_pids_end.get_nrows()):
                pids_end.append(star_pids_end.get_element('_psSurfacePID', hold_row))
        print '\t\t\t+Tomograms computing loop:'
        for tkey in tomos_hash.iterkeys():

            print '\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1]
            try:
                ltomo = llist.get_tomo_by_key(tkey)
            except KeyError:
                print '\t\t\t\t\t-Tomogram with key ' + tkey + ' is not in the list ' + lkey + ' , continuing...'
                continue
            if ltomo.get_num_particles() <= 0:
                print '\t\t\t\t\t-WARNING: no particles to process, continuing...'
                continue

            print '\t\t\t\t\t-Computing nearest inter particle distances...'
            hold_arr_dsts = ltomo.compute_uni_1st_order_dsts()
            if hold_arr_dsts is not None:
                lists_exp_dsts[lkey].append(hold_arr_dsts)
                tomos_exp_dsts[tkey][lkey].append(hold_arr_dsts)

            print '\t\t\t\t\t-Computing CSR and particles nearest distances...'
            hold_arr_fdsts = ltomo.compute_uni_1st_order_dsts(ana_f_npoints)
            if hold_arr_fdsts is not None:
                lists_exp_fdsts[lkey].append(hold_arr_fdsts)
                tomos_exp_fdsts[tkey][lkey].append(hold_arr_fdsts)

            if (len(pids_start) > 0) and (len(pids_end) > 0):
                print '\t\t\t\t\t-Computing nearest intersurfaces distances...'
                if ana_ssize is None:
                    hold_mat_ndsts, hold_mat_codes, nn_points, nn_lines = ltomo.compute_shortest_distances_matrix(
                                                                                                  pids_start,
                                                                                                  pids_end,
                                                                                                  rg_dsts=ana_ndst_rg_v,
                                                                                                  inter_ssize=ana_ssize,
                                                                                                  max_conn=ana_mx_conn,
                                                                                                  del_border=ana_del_border_v,
                                                                                                  min_neg_dst=ana_min_neg_dst_v)
                else:
                    hold_mat_ndsts, hold_mat_codes, nn_points, nn_lines, nn_parts = ltomo.compute_shortest_distances_matrix(
                                                                                                            pids_start,
                                                                                                            pids_end,
                                                                                                            rg_dsts=ana_ndst_rg_v,
                                                                                                            inter_ssize=ana_ssize,
                                                                                                            max_conn=ana_mx_conn,
                                                                                                            del_border=ana_del_border_v,
                                                                                                            min_neg_dst=ana_min_neg_dst_v)
                out_nn_mat_dir = out_dir + '/nn_mats'
                if not os.path.exists(out_nn_mat_dir):
                    os.makedirs(out_nn_mat_dir)
                out_nn_mat = out_nn_mat_dir + '/' + str(lkey) + str(
                    os.path.splitext(tkey)[0].replace('/', '_')) + '.npy'
                print '\t\t\t\t\t\t+Storing NN distances matrix: ' + out_nn_mat
                np.save(out_nn_mat, hold_mat_ndsts)
                out_nn_ptvtp = out_nn_mat_dir + '/' + str(lkey) + str(os.path.splitext(tkey)[0].replace('/', '_')) + \
                               '_points.vtp'
                disperse_io.save_vtp(nn_points, out_nn_ptvtp)
                out_nn_vtp = out_nn_mat_dir + '/' + str(lkey) + str(os.path.splitext(tkey)[0].replace('/', '_')) + \
                             '_lines.vtp'
                disperse_io.save_vtp(nn_lines, out_nn_vtp)
                out_nn_pvtp = out_nn_mat_dir + '/' + str(lkey) + str(os.path.splitext(tkey)[0].replace('/', '_')) + \
                              '_parts.vtp'
                if ana_ssize is not None:
                    disperse_io.save_vtp(nn_parts, out_nn_pvtp)
                else:
                    disperse_io.save_vtp(ltomo.gen_particles_vtp(), out_nn_pvtp)
                # hold_mnnd = hold_mat_ndsts.min(axis=1)
                lists_exp_mnnd[lkey].append(hold_mat_ndsts)
                tomos_exp_mnnd[tkey][lkey] = hold_mat_ndsts
                lists_exp_codes[lkey].append(hold_mat_codes)
                tomos_exp_codes[tkey][lkey] = hold_mat_codes

            print '\t\t\t\t\t-Generating the simulated instances...'
            if (p_nsims is None) or (p_nsims > 0):
                continue
            for ids in sim_ids:
                temp_model, temp_model2 = ModelCSRV(), ModelRR()
                temp_model.set_ParticleL_ref(p_vtp)
                temp_model2.set_ParticleL_ref(p_vtp)
                sim_tomos = gen_tlist(len(ids), ltomo.get_num_particles(), temp_model, ltomo.get_voi(), p_vtp,
                                      mode_emb='center', npr=ana_npr_model)
                if (len(pids_start) > 0) and (len(pids_end) > 0):
                    sim_tomos2 = gen_tlist(len(ids), ltomo.get_num_particles(), temp_model2, ltomo.get_voi(), p_vtp,
                                           mode_emb='center', npr=ana_npr_model, in_coords=ltomo.get_particle_coords())
                hold_mat_dsts, hold_mat_fdsts = list(), list()
                for sim_i, sim_tomo in enumerate(sim_tomos.get_tomo_list()):
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
                    if (len(pids_start) > 0) and (len(pids_end) > 0):
                        out_rnd_pvtp = out_nn_mat_dir + '/' + str(lkey) + str(
                            os.path.splitext(tkey)[0].replace('/', '_')) + \
                                       '_parts_csrv.vtp'
                        disperse_io.save_vtp(sim_tomo.gen_particles_vtp(), out_rnd_pvtp)
                        hold_mat_ndsts, hold_mat_codes, nn_points, nn_lines = sim_tomo.compute_shortest_distances_matrix(pids_start,
                                                                                                         pids_end,
                                                                                                         rg_dsts=ana_ndst_rg_v,
                                                                                                         inter_ssize=ana_ssize,
                                                                                                         part_surf_fname=p_vtp,
                                                                                                         max_conn=ana_mx_conn,
                                                                                                         del_border = ana_del_border_v,
                                                                                                         min_neg_dst=ana_min_neg_dst_v)[:4]
                        lists_sim_mnnd[lkey].append(hold_mat_ndsts)
                        tomos_sim_mnnd[tkey][lkey].append(hold_mat_ndsts)
                        lists_sim_codes[lkey].append(hold_mat_codes)
                        tomos_sim_codes[tkey][lkey].append(hold_mat_codes)
                        if save_sim:
                            out_rr_pvtp = out_nn_mat_dir + '/' + str(lkey) + str(
                                os.path.splitext(tkey)[0].replace('/', '_')) + \
                                          '_parts_csrv.vtp'
                            disperse_io.save_vtp(sim_tomo.gen_particles_vtp(), out_rr_pvtp)
                            out_nn_ptvtp = out_nn_mat_dir + '/' + str(lkey) + str(
                                os.path.splitext(tkey)[0].replace('/', '_')) + \
                                           '_points_csrv.vtp'
                            disperse_io.save_vtp(nn_points, out_nn_ptvtp)
                            out_nn_vtp = out_nn_mat_dir + '/' + str(lkey) + str(
                                os.path.splitext(tkey)[0].replace('/', '_')) + \
                                         '_lines_csrv.vtp'
                            disperse_io.save_vtp(nn_lines, out_nn_vtp)
                            save_sim = False
                        sim_tomo2 = sim_tomos2.get_tomo_list()[sim_i]
                        hold_mat_ndsts, hold_mat_codes, nn_points, nn_lines = sim_tomo2.compute_shortest_distances_matrix(pids_start,
                                                                                                          pids_end,
                                                                                                          rg_dsts=ana_ndst_rg_v,
                                                                                                          inter_ssize=ana_ssize,
                                                                                                          part_surf_fname=p_vtp,
                                                                                                          max_conn=ana_mx_conn,
                                                                                                          del_border = ana_del_border_v,
                                                                                                          min_neg_dst=ana_min_neg_dst_v)[:4]
                        lists_sim2_mnnd[lkey].append(hold_mat_ndsts)
                        tomos_sim2_mnnd[tkey][lkey].append(hold_mat_ndsts)
                        lists_sim2_codes[lkey].append(hold_mat_codes)
                        tomos_sim2_codes[tkey][lkey].append(hold_mat_codes)
                        if save_sim2:
                            out_rr_pvtp = out_nn_mat_dir + '/' + str(lkey) + str(
                                os.path.splitext(tkey)[0].replace('/', '_')) + \
                                          '_parts_rr.vtp'
                            disperse_io.save_vtp(sim_tomo2.gen_particles_vtp(), out_rr_pvtp)
                            out_nn_ptvtp = out_nn_mat_dir + '/' + str(lkey) + str(
                                os.path.splitext(tkey)[0].replace('/', '_')) + \
                                           '_points_rr.vtp'
                            disperse_io.save_vtp(nn_points, out_nn_ptvtp)
                            out_nn_vtp = out_nn_mat_dir + '/' + str(lkey) + str(
                                os.path.splitext(tkey)[0].replace('/', '_')) + \
                                         '_lines_rr.vtp'
                            disperse_io.save_vtp(nn_lines, out_nn_vtp)
                            save_sim2 = False

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print '\tPickling computation workspace in: ' + out_wspace
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash,
              tomos_exp_dsts, tomos_sim_dsts, tomos_exp_fdsts, tomos_sim_fdsts,
              lists_exp_dsts, lists_sim_dsts, lists_exp_fdsts, lists_sim_fdsts, lists_color,
              tomos_exp_mnnd, tomos_sim_mnnd, lists_exp_mnnd, lists_sim_mnnd,
              tomos_sim2_mnnd, lists_sim2_mnnd,
              tomos_exp_codes, tomos_sim_codes, lists_exp_codes, lists_sim_codes,
              tomos_sim2_codes, lists_sim2_codes)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:
    print '\tLoading the workspace: ' + in_wspace
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    lists_count, tomos_count = wspace[0], wspace[1]
    lists_hash, tomos_hash = wspace[2], wspace[3]
    tomos_exp_dsts, tomos_sim_dsts, tomos_exp_fdsts, tomos_sim_fdsts = wspace[4], wspace[5], wspace[6], wspace[7]
    lists_exp_dsts, lists_sim_dsts, lists_exp_fdsts, lists_sim_fdsts, lists_color = wspace[8], wspace[9], wspace[10], \
                                                                                    wspace[11], wspace[12]
    tomos_exp_mnnd, tomos_sim_mnnd, lists_exp_mnnd, lists_sim_mnnd = wspace[13], wspace[14], wspace[15], wspace[16]
    tomos_sim2_mnnd, lists_sim2_mnnd = wspace[17], wspace[18]
    tomos_exp_codes, tomos_sim_codes, lists_exp_codes, lists_sim_codes = wspace[19], wspace[20], wspace[21], wspace[22]
    tomos_sim2_codes, lists_sim2_codes = wspace[23], wspace[24]

print '\tPrinting lists hash: '
for id, lkey in zip(lists_hash.iterkeys(), lists_hash.itervalues()):
    print '\t\t-[' + str(id) + '] -> [' + lkey + ']'
print '\tPrinting tomograms hash: '
for tkey, val in zip(tomos_hash.iterkeys(), tomos_hash.itervalues()):
    print '\t\t-[' + tkey + '] -> [' + str(val) + ']'

# Getting the lists colormap
n_lists = len(lists_hash.keys())
for i, lkey in zip(lists_hash.iterkeys(), lists_hash.itervalues()):
    lists_color[lkey] = pt_cmap(1. * i / n_lists)

print '\tTOMOGRAMS PLOTTING LOOP: '

out_tomos_dir = out_stem_dir + '/tomos'
os.makedirs(out_tomos_dir)

print '\t\t-Plotting Histogram...'
pvals_glow, pvals_ghigh = dict(), dict()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        try:
            hist_bins, hist_vals = compute_hist(np.asarray(arr, dtype=float) * ana_res, ana_nbins, ana_rmax)
            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            sims_hist_vals = list()
            for sim_dsts in tomo_sim_dsts:
                sims_hist_vals.append(compute_hist(np.asarray(sim_dsts, dtype=float) * ana_res, ana_nbins, ana_rmax)[1])
            if len(sims_hist_vals) > 0:
                ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
            else:
                raise ValueError
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
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
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
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
            cdf_bins, cdf_vals = compute_cdf(np.asarray(arr, dtype=float) * ana_res, ana_nbins, ana_rmax)
            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            sims_cdf_vals = list()
            for sim_dsts in tomo_sim_dsts:
                sims_cdf_vals.append(compute_cdf(np.asarray(sim_dsts, dtype=float) * ana_res, ana_nbins, ana_rmax)[1])
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
        # plt.title('Univariate 1st order for ' + tkey_short + ' and ' + lkey)
        plt.ylabel('Nearest neighbor distribution')
        plt.xlabel('Distance [nm]')
        plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], label=lkey, linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(cdf_bins, ic_low, 'k--')
        plt.plot(cdf_bins, ic_med, 'k', linewidth=2.0)
        # plt.plot(cdf_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
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

print '\t\t-Plotting Function-G p-values (tomo[s_low, s_high]= (min_pval, max_pval)):'
min_pvals, max_pvals = list(), list()
for tkey, pvals_low, pvals_high in zip(pvals_glow.iterkeys(), pvals_glow.itervalues(), pvals_ghigh.itervalues()):
    pvals_hmin, pvals_hmax = np.copy(pvals_low), np.copy(pvals_high)
    pvals_hmin[cdf_bins <= pr_ss] = -1
    pvals_hmax[cdf_bins <= pr_ss] = -1
    idx_min, idx_max = np.argmax(pvals_hmin), np.argmax(pvals_hmax)
    min_pval, max_pval = pvals_hmin[idx_min], pvals_hmax[idx_max]
    print '\t\t\t+' + tkey + '[' + str(cdf_bins[idx_min]) + ', ' + str(cdf_bins[idx_max]) + ']= (' \
          + str(min_pval) + ', ' + str(max_pval) + ')'
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

print '\t\t-Plotting Funtion-G for all simulations:'
plt.figure()
plt.title('Function-G (1st order) for all simulations')
plt.ylabel('Function-G')
plt.xlabel('Distance [nm]')
pvals_glow, pvals_ghigh = dict(), dict()
hold_cdf_vals = dict()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        try:

            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            for sim_dsts in tomo_sim_dsts:
                cdf_bins, cdf_vals = compute_cdf(sim_dsts * ana_res, ana_nbins, ana_rmax)
                plt.plot(cdf_bins, cdf_vals, color=lists_color[lkey], linewidth=.5, alpha=.5)
                try:
                    hold_cdf_vals[lkey].append(cdf_vals)
                except KeyError:
                    hold_cdf_vals[lkey] = list()
                    hold_cdf_vals[lkey].append(cdf_vals)
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            continue
for lkey, cdf_vals in zip(hold_cdf_vals.iterkeys(), hold_cdf_vals.itervalues()):
    cdf_mat = np.asarray(cdf_vals, dtype=np.float32)
    lbl = lkey
    if lkey == '0':
        lbl = 'CONTROL'
    elif lkey == '1':
        lbl = 'RAPA'
    plt.plot(cdf_bins, np.median(cdf_mat, axis=0), color=lists_color[lkey], label=lbl, linewidth=3.0, linestyle='--')
plt.legend(loc=4)
plt.tight_layout()
plt.grid(True)
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/sims_G.png')
plt.close()

print '\t\t-Plotting histogram for all simulations:'
plt.figure()
plt.title('Univariate 1st order for all simulations')
plt.ylabel('Frequency')
plt.xlabel('Distance [nm]')
pvals_glow, pvals_ghigh = dict(), dict()
hold_hist_vals = dict()
for tkey, ltomo in zip(tomos_exp_dsts.iterkeys(), tomos_exp_dsts.itervalues()):
    for lkey, arr in zip(tomos_exp_dsts[tkey].iterkeys(), tomos_exp_dsts[tkey].itervalues()):
        try:
            tomo_sim_dsts = tomos_sim_dsts[tkey][lkey]
            for sim_dsts in tomo_sim_dsts:
                hist_bins, hist_vals = compute_hist(sim_dsts * ana_res, ana_nbins, ana_rmax)
                plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=.5, alpha=.3)
                try:
                    hold_hist_vals[lkey].append(hist_vals)
                except KeyError:
                    hold_hist_vals[lkey] = list()
                    hold_hist_vals[lkey].append(hist_vals)
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            continue
for lkey, hist_vals in zip(hold_hist_vals.iterkeys(), hold_hist_vals.itervalues()):
    hist_mat = np.asarray(hist_vals, dtype=np.float32)
    lbl = lkey
    if lkey == '0':
        lbl = 'CONTROL'
    elif lkey == '1':
        lbl = 'RAPA'
    plt.plot(hist_bins, np.median(hist_mat, axis=0), color=lists_color[lkey], label=lbl, linewidth=3.0, linestyle='--')
plt.legend(loc=0)
plt.tight_layout()
plt.grid(True)
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/sims_H.png')
plt.close()

print '\t\t-Plotting Function-F...'
pvals_flow, pvals_fhigh = dict(), dict()
for tkey, ltomo in zip(tomos_exp_fdsts.iterkeys(), tomos_exp_fdsts.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(tomos_exp_fdsts[tkey].iterkeys(), tomos_exp_fdsts[tkey].itervalues()):
        try:
            cdf_bins, cdf_vals = compute_cdf(np.asarray(arr, dtype=float) * ana_res, ana_nbins, ana_rmax)
            tomo_sim_fdsts = tomos_sim_fdsts[tkey][lkey]
            sims_cdf_fvals = list()
            for sim_fdsts in tomo_sim_fdsts:
                sims_cdf_fvals.append(compute_cdf(np.asarray(sim_fdsts, dtype=float) * ana_res, ana_nbins, ana_rmax)[1])
            if len(sims_cdf_fvals) > 0:
                ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_fvals))
                pvals_flow[tkey], pvals_fhigh[tkey] = compute_pvals(cdf_vals, np.asarray(sims_cdf_fvals))
            else:
                raise ValueError
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
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
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        plt.tight_layout()
        plt.grid(True)
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/F_' + lkey + '.png')
        plt.close()

print '\t\t-Plotting Function-F p-values (tomo[s_low, s_high]= (min_pval, max_pval)):'
min_pvals, max_pvals = list(), list()
for tkey, pvals_low, pvals_high in zip(pvals_flow.iterkeys(), pvals_flow.itervalues(), pvals_fhigh.itervalues()):
    pvals_hmin, pvals_hmax = np.copy(pvals_low), np.copy(pvals_high)
    pvals_hmin[cdf_bins <= pr_ss] = -1
    pvals_hmax[cdf_bins <= pr_ss] = -1
    idx_min, idx_max = np.argmax(pvals_hmin), np.argmax(pvals_hmax)
    min_pval, max_pval = pvals_hmin[idx_min], pvals_hmax[idx_max]
    print '\t\t\t+' + tkey + '[' + str(cdf_bins[idx_min]) + ', ' + str(cdf_bins[idx_max]) + ']= (' + \
          str(min_pval) + ', ' + str(max_pval) + ')'
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

print '\t\t-Plotting Function-J...'
for tkey, ltomo in zip(tomos_exp_fdsts.iterkeys(), tomos_exp_fdsts.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, farr in zip(tomos_exp_fdsts[tkey].iterkeys(), tomos_exp_fdsts[tkey].itervalues()):
        arr = tomos_exp_dsts[tkey][lkey]
        try:
            cdf_bins, cdf_vals = compute_cdf(np.asarray(arr, dtype=float) * ana_res, ana_nbins, ana_rmax)
            cdf_fvals = compute_cdf(np.asarray(farr, dtype=float) * ana_res, ana_nbins, ana_rmax)[1]
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
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
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
        cdf_sim_len = len(cdf_sim_jhigh)
        if len(cdf_sim_jlow) < len(cdf_sim_jhigh):
            cdf_sim_len = len(cdf_sim_jlow)
        plt.fill_between(cdf_bins[:cdf_sim_len], cdf_sim_jlow[:cdf_sim_len], cdf_sim_jhigh[:cdf_sim_len],
                         alpha=0.5, color='gray', edgecolor='w')
        plt.tight_layout()
        plt.grid(True)
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/J_' + lkey + '.png')
        plt.close()

# if star.has_column('_psStartSurfIds') and star.has_column('_psEndSurfIds'):
print '\t\t-Plotting Function-NNS...'
pvals_glow, pvals_ghigh = dict(), dict()
arr_exp_, arr_sim_, arr_sim2_ = list(), list(), list()
for tkey, ltomo in zip(tomos_exp_mnnd.iterkeys(), tomos_exp_mnnd.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(tomos_exp_mnnd[tkey].iterkeys(), tomos_exp_mnnd[tkey].itervalues()):
        hold_mnnd, hold_codes = tomos_exp_mnnd[tkey][lkey], tomos_exp_codes[tkey][lkey]
        if hold_mnnd is None:
            continue
        arr_exp, arr_sim, arr_sim2 = list(), dict(), dict()
        for p_id in range(hold_mnnd.shape[1]):
            arr_sim[p_id] = list()
            arr_sim2[p_id] = list()
        for p_id in range(hold_mnnd.shape[1]):
            try:
                arr = list()
                for pr_id in range(hold_mnnd.shape[0]):
                    for hold, code in zip(hold_mnnd[pr_id, p_id], hold_codes[pr_id, p_id]):
                        if code == 1:
                            arr.append(hold)
                arr_exp += arr
                arr = np.asarray(arr, dtype=np.float32) * ana_res
                arr_exp_.append(arr)
                hist_bins, hist_vals = compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])
                hold_sim_mnnd, hold_sim2_mnnd = tomos_sim_mnnd[tkey][lkey], tomos_sim2_mnnd[tkey][lkey]
                hold_sim_codes, hold_sim2_codes = tomos_sim_codes[tkey][lkey], tomos_sim2_codes[tkey][lkey]
                sims_hist_vals, sims2_hist_vals = list(), list()
                for sim_dsts, sim_codes in zip(hold_sim_mnnd, hold_sim_codes):
                    arr = list()
                    for pr_id in range(sim_dsts.shape[0]):
                        for hold, code in zip(sim_dsts[pr_id, p_id], sim_codes[pr_id, p_id]):
                            if code == 1:
                                arr.append(hold)
                    arr_sim[p_id] += arr
                    arr = np.asarray(arr, dtype=np.float32) * ana_res
                    arr_sim_.append(arr)
                    sims_hist_vals.append(compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])[1])
                if len(sims_hist_vals) > 0:
                    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
                    for sim_dsts, sim_codes in zip(hold_sim2_mnnd, hold_sim2_codes):
                        arr = list()
                        for pr_id in range(sim_dsts.shape[0]):
                            for hold, code in zip(sim_dsts[pr_id, p_id], sim_codes[pr_id, p_id]):
                                if code == 1:
                                    arr.append(hold)
                    arr_sim2[p_id] += arr
                    arr = np.asarray(arr, dtype=np.float32) * ana_res
                    arr_sim2_.append(arr)
                    sims2_hist_vals.append(compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])[1])
                if len(sims2_hist_vals) > 0:
                    ic2_low, ic2_med, ic2_high = compute_ic(p_per, np.asarray(sims2_hist_vals))
                else:
                    raise ValueError
            except ValueError or IndexError:
                print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
                continue
            plt.figure()
            plt.ylabel('Nearest Neighbor inter-Surface frequency')
            plt.xlabel('Distance [nm]')
            plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=2.0)
            plt.legend(loc=4)
            # plt.plot(hist_bins, ic_low, 'k--')
            plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
            # plt.plot(hist_bins, ic_high, 'k--')
            plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
            # plt.plot(hist_bins, ic2_low, 'c--')
            plt.plot(hist_bins, ic2_med, 'c', linewidth=2.0)
            # plt.plot(hist_bins, ic2_high, 'c--')
            plt.fill_between(hist_bins, ic2_low, ic2_high, alpha=0.5, color='cyan', edgecolor='c')
            plt.tight_layout()
            plt.grid(True)
            if fig_fmt is None:
                plt.show(block=True)
            else:
                out_fig_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
                if not os.path.exists(out_fig_dir):
                    os.makedirs(out_fig_dir)
                plt.savefig(out_fig_dir + '/NNS_p' + str(p_id) + '_' + lkey + '.png')
            plt.close()
        arr_exp = np.asarray(arr_exp, dtype=np.float32) * ana_res
        hist_bins, hist_vals = compute_hist_nns(arr_exp, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])
        sims_hist_vals, sims2_hist_vals = list(), list()
        for arr in arr_sim.itervalues():
            arr = np.asarray(arr, dtype=np.float32) * ana_res
            sims_hist_vals.append(compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])[1])
        ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
        for arr in arr_sim2.itervalues():
            arr = np.asarray(arr, dtype=np.float32) * ana_res
            sims2_hist_vals.append(compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])[1])
        ic2_low, ic2_med, ic2_high = compute_ic(p_per, np.asarray(sims2_hist_vals))
        plt.figure()
        plt.ylabel('Nearest Neighbor inter-Surface frequency')
        plt.xlabel('Distance [nm]')
        plt.plot(hist_bins, hist_vals/float(len(arr_sim.values())), color=lists_color[lkey], linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(hist_bins, ic_low, 'k--')
        plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
        # plt.plot(hist_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        # plt.plot(hist_bins, ic2_low, 'c--')
        plt.plot(hist_bins, ic2_med, 'c', linewidth=2.0)
        # plt.plot(hist_bins, ic2_high, 'c--')
        plt.fill_between(hist_bins, ic2_low, ic2_high, alpha=0.5, color='cyan', edgecolor='c')
        plt.tight_layout()
        plt.grid(True)
        if fig_fmt is None:
            plt.show(block=True)
        else:
            out_fig_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
            if not os.path.exists(out_fig_dir):
                os.makedirs(out_fig_dir)
            plt.savefig(out_fig_dir + '/NNS_merged_' + lkey + '.png')
        plt.close()

print '\tLISTS PLOTTING LOOP: '

crf = ana_res / 0.684

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

# print '\t\t-Plotting Histogram...'
# for lkey, ltomo in zip(lists_exp_dsts.iterkeys(), lists_exp_dsts.itervalues()):
#     lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
#     try:
#         hist_bins, hist_vals = compute_hist(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
#         list_sim_dsts = lists_sim_dsts[lkey]
#         sims_hist_vals = list()
#         for sim_dsts in list_sim_dsts:
#             sims_hist_vals.append(compute_hist(sim_dsts, ana_nbins, ana_rmax)[1])
#         if len(sims_hist_vals) > 0:
#             ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
#         else:
#             raise ValueError
#     except ValueError or IndexError:
#         print np.concatenate(np.asarray(ltomo))
#         print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
#         continue
#     plt.figure()
#     # plt.title('Nearest distance histogram for ' + lkey_short)
#     plt.ylabel('Frequency')
#     plt.xlabel('Distance [nm]')
#     plt.plot(hist_bins * crf, hist_vals, color=lists_color[lkey], linewidth=2.0)
#     # plt.plot(hist_bins, ic_low, 'k--')
#     plt.plot(hist_bins * crf, ic_med, 'k', linewidth=2.0)
#     # plt.plot(hist_bins, ic_high, 'k--')
#     plt.fill_between(hist_bins * crf, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
#     plt.tight_layout()
#     if fig_fmt is None:
#         plt.show(block=True)
#     else:
#         plt.savefig(out_lists_dir + '/H_' + lkey_short + '.png', dpi=600)
#     plt.close()
#
# print '\t\t-Plotting Function-G...'
# for lkey, ltomo in zip(lists_exp_dsts.iterkeys(), lists_exp_dsts.itervalues()):
#     lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
#     try:
#         cdf_bins, cdf_vals = compute_cdf(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
#         list_sim_dsts = lists_sim_dsts[lkey]
#         sims_cdf_vals = list()
#         for sim_dsts in list_sim_dsts:
#             sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1])
#         if len(sims_cdf_vals) > 0:
#             ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
#         else:
#             raise ValueError
#     except ValueError or IndexError:
#         print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
#         continue
#     plt.figure()
#     # plt.title('Univariate 1st order for ' + lkey_short)
#     plt.ylabel('Nearest neighbor distribution')
#     plt.xlabel('Distance [nm]')
#     plt.plot(cdf_bins * crf, cdf_vals, color=lists_color[lkey], linewidth=2.0)
#     # plt.plot(cdf_bins, ic_low, 'k--')
#     plt.plot(cdf_bins * crf, ic_med, 'k', linewidth=2.0)
#     # plt.plot(cdf_bins, ic_high, 'k--')
#     plt.fill_between(cdf_bins * crf, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
#     plt.tight_layout()
#     if fig_fmt is None:
#         plt.show(block=True)
#     else:
#         plt.savefig(out_lists_dir + '/G_' + lkey_short + '.png', dpi=600)
#     plt.close()
#
# print '\t\t-Plotting Function-F...'
# for lkey, ltomo in zip(lists_exp_fdsts.iterkeys(), lists_exp_fdsts.itervalues()):
#     lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
#     try:
#         cdf_bins, cdf_vals = compute_cdf(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)
#         list_sim_fdsts = lists_sim_fdsts[lkey]
#         sims_cdf_vals = list()
#         for sim_fdsts in list_sim_fdsts:
#             sims_cdf_vals.append(compute_cdf(sim_fdsts, ana_nbins, ana_rmax)[1])
#         if len(sims_cdf_vals) > 0:
#             ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
#         else:
#             raise ValueError
#     except ValueError or IndexError:
#         print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
#         continue
#     plt.figure()
#     # plt.title('Univariate 1st order for ' + lkey_short)
#     plt.ylabel('Contact function distribution')
#     plt.xlabel('Distance [nm]')
#     plt.plot(cdf_bins * crf, cdf_vals, color=lists_color[lkey], linewidth=2.0)
#     # plt.plot(cdf_bins, ic_low, 'k--')
#     plt.plot(cdf_bins * crf, ic_med, 'k', linewidth=2.0)
#     # plt.plot(cdf_bins, ic_high, 'k--')
#     plt.fill_between(cdf_bins * crf, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
#     plt.tight_layout()
#     if fig_fmt is None:
#         plt.show(block=True)
#     else:
#         plt.savefig(out_lists_dir + '/F_' + lkey_short + '.png', dpi=600)
#     plt.close()
#
# print '\t\t-Plotting Function-J...'
# for lkey, ltomo in zip(lists_exp_fdsts.iterkeys(), lists_exp_fdsts.itervalues()):
#     lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
#     try:
#         cdf_bins, cdf_vals = compute_cdf(np.concatenate(np.asarray(lists_exp_dsts[lkey])), ana_nbins, ana_rmax)
#         cdf_fvals = compute_cdf(np.concatenate(np.asarray(ltomo)), ana_nbins, ana_rmax)[1]
#         cdf_jvals = compute_J(cdf_vals, cdf_fvals, high=p_jhigh)
#         list_sim_fdsts = lists_sim_fdsts[lkey]
#         sims_cdf_vals, sims_cdf_fvals = list(), list()
#         for sim_dsts, sim_fdsts in zip(lists_sim_dsts[lkey], list_sim_fdsts):
#             sims_cdf_vals.append(compute_cdf(sim_dsts, ana_nbins, ana_rmax)[1])
#             sims_cdf_fvals.append(compute_cdf(sim_fdsts, ana_nbins, ana_rmax)[1])
#         if (len(sims_cdf_vals) > 0) and (len(sims_cdf_fvals) > 0):
#             ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_cdf_vals))
#             icf_low, icf_med, icf_high = compute_ic(p_per, np.asarray(sims_cdf_fvals))
#             cdf_sim_jlow = compute_J(ic_low, icf_high, high=p_jhigh)
#             cdf_sim_jmed = compute_J(ic_med, icf_med, high=p_jhigh)
#             cdf_sim_jhigh = compute_J(ic_high, icf_low, high=p_jhigh)
#         else:
#             raise ValueError
#     except ValueError or IndexError:
#         print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
#         continue
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     # plt.title('Univariate 1st order for ' + lkey_short)
#     plt.ylabel('J-function')
#     plt.xlabel('Scale (nm)')
#     plt.plot(cdf_bins[:len(cdf_jvals)] * crf, cdf_jvals, color=lists_color[lkey], linewidth=2.0)
#     # plt.plot(cdf_bins[:len(cdf_sim_jlow)], cdf_sim_jlow, 'k--')
#     plt.plot(cdf_bins[:len(cdf_sim_jmed)] * crf, cdf_sim_jmed, 'k', linewidth=2.0)
#     # plt.plot(cdf_bins[:len(cdf_sim_jhigh)], cdf_sim_jhigh, 'k--')
#     cdf_sim_len = len(cdf_sim_jhigh)
#     if len(cdf_sim_jlow) < len(cdf_sim_jhigh):
#         cdf_sim_len = len(cdf_sim_jlow)
#     plt.fill_between(cdf_bins[:cdf_sim_len] * crf, cdf_sim_jlow[:cdf_sim_len], cdf_sim_jhigh[:cdf_sim_len],
#                      alpha=0.5, color='gray', edgecolor='w')
#     ax.set_xlim((0, 60))
#     plt.tight_layout()
#     if fig_fmt is None:
#         plt.show(block=True)
#     else:
#         plt.savefig(out_lists_dir + '/J_' + lkey_short + '.png', dpi=600)
#     plt.close()

# if star.has_column('_psStartSurfIds') and star.has_column('_psEndSurfIds'):
ana_ndst_rg = [0, 12]
print '\t\t-Plotting Function-NNS...'
for lkey, ltomo in zip(lists_exp_mnnd.iterkeys(), lists_exp_mnnd.itervalues()):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    # for p_id in range(lists_exp_mnnd[lkey][0].shape[1]):
    #     arr_sim[p_id], arr_sim2[p_id] = list(), list()
    for p_id in range(lists_exp_mnnd[lkey][0].shape[1]):
        try:
            arr = list()
            for hold_mnnd, hold_codes in zip(lists_exp_mnnd[lkey], lists_exp_codes[lkey]):
                for pr_id in range(hold_mnnd.shape[0]):
                    for hold, code in zip(hold_mnnd[pr_id, p_id], hold_codes[pr_id, p_id]):
                        if code == 1:
                            arr.append(hold)
            arr = np.asarray(arr, dtype=np.float32) * ana_res
            hist_bins, hist_vals = compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])
            hold_sim_mnnds, hold_sim2_mnnds = lists_sim_mnnd[lkey], lists_sim2_mnnd[lkey]
            hold_sim_codes, hold_sim2_codes = lists_sim_codes[lkey], lists_sim2_codes[lkey]
            sims_hist_vals, sims2_hist_vals = list(), list()
            for sim_dsts, sim_codes in zip(hold_sim_mnnds, hold_sim_codes):
                arr = list()
                for pr_id in range(sim_dsts.shape[0]):
                    for hold, code in zip(sim_dsts[pr_id, p_id], sim_codes[pr_id, p_id]):
                        if code == 1:
                            arr.append(hold)
                arr = np.asarray(arr, dtype=np.float32) * ana_res
                sims_hist_vals.append(compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])[1])
            if len(sims_hist_vals) > 0:
                ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
            for sim_dsts, sim_codes in zip(hold_sim2_mnnds, hold_sim2_codes):
                arr = list()
                for pr_id in range(sim_dsts.shape[0]):
                    for hold, code in zip(sim_dsts[pr_id, p_id], sim_codes[pr_id, p_id]):
                        if code == 1:
                            arr.append(hold)
                arr = np.asarray(arr, dtype=np.float32) * ana_res
                sims2_hist_vals.append(compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])[1])
            if len(sims2_hist_vals) > 0:
                ic2_low, ic2_med, ic2_high = compute_ic(p_per, np.asarray(sims2_hist_vals))
            else:
                raise ValueError
        except ValueError or IndexError:
            print '\t\t\t+WARNING: no valid simulations for tomogram and list: ' + tkey + ', ' + lkey
            continue
        plt.figure()
        plt.ylabel('Nearest Neighbor inter-Surface frequency')
        plt.xlabel('Distance [nm]')
        plt.plot(hist_bins, hist_vals, color=lists_color[lkey], linewidth=2.0)
        plt.legend(loc=4)
        # plt.plot(hist_bins, ic_low, 'k--')
        plt.plot(hist_bins, ic_med, 'k', linewidth=2.0)
        # plt.plot(hist_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        # plt.plot(hist_bins, ic2_low, 'c--')
        plt.plot(hist_bins, ic2_med, 'c', linewidth=2.0)
        # plt.plot(hist_bins, ic2_high, 'c--')
        plt.fill_between(hist_bins, ic2_low, ic2_high, alpha=0.5, color='cyan', edgecolor='c')
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            plt.savefig(out_lists_dir + '/NNS_p' + str(p_id) + '_' + lkey_short + '.png')
        plt.close()
    exp_hist_vals, sims_hist_vals, sims2_hist_vals = list(), list(), list()
    for arr in arr_exp_:
        # arr = np.asarray(arr, dtype=np.float32) * ana_res
        hist_bins, hist_vals = compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])
        exp_hist_vals.append(compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])[1])
    ice_low, ice_med, ice_high = compute_ic(p_per, np.asarray(exp_hist_vals))
    for arr in arr_sim_:
        # arr = np.asarray(arr, dtype=np.float32) * ana_res
        sims_hist_vals.append(compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])[1])
    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(sims_hist_vals))
    # ic_low, ic_med, ic_high = ic_low/float(p_nsims), ic_med/float(p_nsims), ic_high/float(p_nsims)
    for arr in arr_sim2_:
        # arr = np.asarray(arr, dtype=np.float32) * ana_res
        sims2_hist_vals.append(compute_hist_nns(arr, ana_nbins, ana_ndst_rg[0], ana_ndst_rg[1])[1])
    ic2_low, ic2_med, ic2_high = compute_ic(p_per, np.asarray(sims2_hist_vals))
    # ic2_low, ic2_med, ic2_high = ic2_low / float(p_nsims), ic2_med / float(p_nsims), ic2_high / float(p_nsims)
    plt.figure()
    plt.ylabel('Probability density function')
    plt.xlabel('Distance [nm]')
    try:
        # plt.plot(hist_bins, ic_low, 'k--')
        plt.plot(hist_bins, ic_med, 'k', linewidth=2.0, label='CSRV')
        # plt.plot(hist_bins, ic_high, 'k--')
        plt.fill_between(hist_bins, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
        # plt.plot(hist_bins, ic2_low, 'c--')
        plt.plot(hist_bins, ic2_med, 'c', linewidth=2.0, label='RR')
        # plt.plot(hist_bins, ic2_high, 'c--')
        plt.fill_between(hist_bins, ic2_low, ic2_high, alpha=0.5, color='cyan', edgecolor='c')
        # plt.plot(hist_bins, hist_vals / float(len(arr_sim.values())), color=lists_color[lkey], linewidth=2.0, label='EXP')
    except ValueError:
        print 'INFO: no simulations'
        pass
    plt.plot(hist_bins, ice_med, 'r', linewidth=2.0, label='EXP')
    # plt.plot(hist_bins, ic2_high, 'c--')
    plt.fill_between(hist_bins, ice_low, ice_high, alpha=0.5, color='red', edgecolor='r')
    plt.tight_layout()
    plt.legend(loc=1)
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/NNS_merged_' + lkey_short + '.png')
    plt.close()

# if star.has_column('_psStartSurfIds') and star.has_column('_psEndSurfIds'):
ana_ndst_rg[1] = 15
print '\t\t-Plotting Codes...'
for lkey, ltomo in zip(lists_exp_codes.iterkeys(), lists_exp_codes.itervalues()):
    lkey_short = os.path.splitext(os.path.split(lkey)[1])[0]
    # for p_id in range(lists_exp_mnnd[lkey][0].shape[1]):
    #     arr_sim[p_id], arr_sim2[p_id] = list(), list()
    mg_exp_1, mg_exp_2, mg_exp_3 = list(), list(), list()
    mg_sim_1, mg_sim_2, mg_sim_3 = list(), list(), list()
    mg_sim2_1, mg_sim2_2, mg_sim2_3 = list(), list(), list()
    for p_id in range(lists_exp_codes[lkey][0].shape[1]):
        arr = list()
        for hold_codes in lists_exp_codes[lkey]:
            for pr_id in range(hold_codes.shape[0]):
                arr += hold_codes[pr_id, p_id]
        arr = np.asarray(arr, dtype=np.float32)
        exp_val_1, exp_val_2, exp_val_3 = (arr == 1).sum(), (arr == 2).sum(), (arr == 3).sum()
        mg_exp_1.append(exp_val_1)
        mg_exp_2.append(exp_val_2)
        mg_exp_3.append(exp_val_3)
        hold_sim_codes, hold_sim2_codes = lists_sim_codes[lkey], lists_sim2_codes[lkey]
        sims_vals_1, sims_vals_2, sims_vals_3 = list(), list(), list()
        for sim_codes in hold_sim_codes:
            arr = list()
            for pr_id in range(sim_codes.shape[0]):
                arr += sim_codes[pr_id, p_id]
            arr = np.asarray(arr, dtype=np.float32)
            sim_val_1, sim_val_2, sim_val_3 = (arr == 1).sum(), (arr == 2).sum(), (arr == 3).sum()
            sims_vals_1.append(sim_val_1)
            sims_vals_2.append(sim_val_2)
            sims_vals_3.append(sim_val_3)
            mg_sim_1.append(sim_val_1)
            mg_sim_2.append(sim_val_2)
            mg_sim_3.append(sim_val_3)
        sims2_vals_1, sims2_vals_2, sims2_vals_3 = list(), list(), list()
        for sim_codes in hold_sim2_codes:
            arr = list()
            for pr_id in range(sim_codes.shape[0]):
                arr += sim_codes[pr_id, p_id]
            arr = np.asarray(arr, dtype=np.float32)
            sim_val_1, sim_val_2, sim_val_3 = (arr == 1).sum(), (arr == 2).sum(), (arr == 3).sum()
            sims2_vals_1.append(sim_val_1)
            sims2_vals_2.append(sim_val_2)
            sims2_vals_3.append(sim_val_3)
            mg_sim2_1.append(sim_val_1)
            mg_sim2_2.append(sim_val_2)
            mg_sim2_3.append(sim_val_3)
        plt.figure()
        plt.ylabel('Frequency')
        plt.bar(1, exp_val_1, BAR_WIDTH, color='tab:blue', linewidth=2)
        plt.bar(2, exp_val_2, BAR_WIDTH, color='tab:orange', linewidth=2)
        plt.bar(3, exp_val_3, BAR_WIDTH, color='tab:green', linewidth=2)
        try:
            ic_low = np.percentile(np.asarray(sims_vals_1), p_per)
            ic_med = np.percentile(np.asarray(sims_vals_1), p_per)
            ic_high = np.percentile(np.asarray(sims_vals_1), p_per)
            plt.bar(5, ic_med, BAR_WIDTH, color='tab:blue', linewidth=2)
            # plt.errorbar(5, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
            #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
            ic_low = np.percentile(np.asarray(sims_vals_2), p_per)
            ic_med = np.percentile(np.asarray(sims_vals_2), p_per)
            ic_high = np.percentile(np.asarray(sims_vals_2), p_per)
            plt.bar(6, ic_med, BAR_WIDTH, color='tab:orange', linewidth=2)
            # plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
            #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
            ic_low = np.percentile(np.asarray(sims_vals_3), p_per)
            ic_med = np.percentile(np.asarray(sims_vals_3), p_per)
            ic_high = np.percentile(np.asarray(sims_vals_3), p_per)
            plt.bar(7, ic_med, BAR_WIDTH, color='tab:green', linewidth=2)
            # plt.errorbar(7, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
            #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
            ic_low = np.percentile(np.asarray(sims2_vals_1), p_per)
            ic_med = np.percentile(np.asarray(sims2_vals_1), p_per)
            ic_high = np.percentile(np.asarray(sims2_vals_1), p_per)
            plt.bar(9, ic_med, BAR_WIDTH, color='tab:blue', linewidth=2)
            # plt.errorbar(9, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
            #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
            ic_low = np.percentile(np.asarray(sims2_vals_2), p_per)
            ic_med = np.percentile(np.asarray(sims2_vals_2), p_per)
            ic_high = np.percentile(np.asarray(sims2_vals_2), p_per)
            plt.bar(10, ic_med, BAR_WIDTH, color='tab:orange', linewidth=2)
            # plt.errorbar(10, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
            #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
            ic_low = np.percentile(np.asarray(sims2_vals_3), p_per)
            ic_med = np.percentile(np.asarray(sims2_vals_3), p_per)
            ic_high = np.percentile(np.asarray(sims2_vals_3), p_per)
            plt.bar(11, ic_med, BAR_WIDTH, color='tab:green', linewidth=2)
            # plt.errorbar(11, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
            #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
            plt.xticks((1, 2, 3, 5, 6, 7, 9, 10, 11), ('P', 'N', 'B', 'R:P', 'R:N', 'R:B', 'RR:P', 'RR:N', 'RR:B'))
            plt.xlim(0.5, 11.5)
        except IndexError:
            plt.xticks((1, 2, 3), ('P', 'N', 'B'))
            plt.xlim(0.5, 3.5)
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            plt.savefig(out_lists_dir + '/codes_p' + str(p_id) + '_' + lkey_short + '.png')
        plt.close()
    n_tomos = len(tomos_exp_codes.values())
    n_pts = lists_exp_codes[lkey][0].shape[1]
    mg_exp_1, mg_exp_2, mg_exp_3 = np.asarray(mg_exp_1)*n_pts, np.asarray(mg_exp_2)*n_pts, np.asarray(mg_exp_3)*n_pts
    hold_1, hold_2, hold_3 = np.zeros(shape=len(mg_exp_1)), np.zeros(shape=len(mg_exp_2)), np.zeros(shape=len(mg_exp_3))
    for i in range(n_pts):
        idx = i * n_tomos
        hold_1[i], hold_2[i], hold_3[i] = np.asarray(mg_sim_1[idx:idx+n_tomos]).sum(), \
                                          np.asarray(mg_sim_2[idx:idx+n_tomos]).sum(), \
                                          np.asarray(mg_sim_3[idx:idx+n_tomos]).sum()
    mg_sim_1, mg_sim_2, mg_sim_3 = hold_1, hold_2, hold_3
    mg_sim_1, mg_sim_2, mg_sim_3 = np.asarray(mg_sim_1)*n_pts, np.asarray(mg_sim_2)*n_pts, np.asarray(mg_sim_3)*n_pts
    hold_1, hold_2, hold_3 = np.zeros(shape=len(mg_exp_1)), np.zeros(shape=len(mg_exp_2)), np.zeros(shape=len(mg_exp_3))
    for i in range(n_pts):
        idx = i * n_tomos
        hold_1[i], hold_2[i], hold_3[i] = np.asarray(mg_sim2_1[idx:idx + n_tomos]).sum(), \
                                          np.asarray(mg_sim2_2[idx:idx + n_tomos]).sum(), \
                                          np.asarray(mg_sim2_3[idx:idx + n_tomos]).sum()
    mg_sim2_1, mg_sim2_2, mg_sim2_3 = hold_1, hold_2, hold_3
    mg_sim2_1, mg_sim2_2, mg_sim2_3 = np.asarray(mg_sim2_1)*n_pts, np.asarray(mg_sim2_2)*n_pts, np.asarray(mg_sim2_3)*n_pts
    plt.figure()
    plt.ylabel('Frequency')
    plt.bar(1, mg_exp_1, BAR_WIDTH, color='tab:blue', linewidth=2)
    plt.bar(2, mg_exp_2, BAR_WIDTH, color='tab:orange', linewidth=2)
    plt.bar(3, mg_exp_3, BAR_WIDTH, color='tab:green', linewidth=2)
    ic_low = np.percentile(mg_sim_1, p_per)
    ic_med = np.percentile(mg_sim_1, p_per)
    ic_high = np.percentile(mg_sim_1, p_per)
    plt.bar(5, ic_med, BAR_WIDTH, color='tab:blue', linewidth=2)
    # plt.errorbar(5, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
    #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low = np.percentile(mg_sim_2, p_per)
    ic_med = np.percentile(mg_sim_2, p_per)
    ic_high = np.percentile(mg_sim_2, p_per)
    plt.bar(6, ic_med, BAR_WIDTH, color='tab:orange', linewidth=2)
    # plt.errorbar(6, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
    #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low = np.percentile(mg_sim_3, p_per)
    ic_med = np.percentile(mg_sim_3, p_per)
    ic_high = np.percentile(mg_sim_3, p_per)
    plt.bar(7, ic_med, BAR_WIDTH, color='tab:green', linewidth=2)
    # plt.errorbar(7, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
    #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low = np.percentile(mg_sim2_1, p_per)
    ic_med = np.percentile(mg_sim2_1, p_per)
    ic_high = np.percentile(mg_sim2_1, p_per)
    plt.bar(9, ic_med, BAR_WIDTH, color='tab:blue', linewidth=2)
    # plt.errorbar(9, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
    #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low = np.percentile(mg_sim2_2, p_per)
    ic_med = np.percentile(mg_sim2_2, p_per)
    ic_high = np.percentile(mg_sim2_2, p_per)
    plt.bar(10, ic_med, BAR_WIDTH, color='tab:orange', linewidth=2)
    # plt.errorbar(10, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
    #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low = np.percentile(mg_sim2_3, p_per)
    ic_med = np.percentile(mg_sim2_3, p_per)
    ic_high = np.percentile(mg_sim2_3, p_per)
    plt.bar(11, ic_med, BAR_WIDTH, color='tab:green', linewidth=2)
    # plt.errorbar(11, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
    #              ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3, 5, 6, 7, 9, 10, 11), ('P', 'N', 'B', 'R:P', 'R:N', 'R:B', 'RR:P', 'RR:N', 'RR:B'))
    plt.xlim(0.5, 11.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/codes_' + lkey_short + '.png')
    plt.close()

print 'Successfully terminated. (' + time.strftime("%c") + ')'
