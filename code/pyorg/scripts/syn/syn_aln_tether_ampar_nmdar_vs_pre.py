"""

    Performs Bivariate 2nd order analysis to a SetListTomograms object by tomograms

    Input:  - A STAR file with a ListTomoParticles pickle (SetListTomograms object input) used as reference
            - A STAR file with a set of ListTomoParticles pickles (SetListTomograms object input) to analyze

    Output: - Plots by tomograms:
                + Number of particles by list
                + Particles densities by list
                + Plots by list:
                    * Bivariate 2nd order analysis against random simulation
            - Global plots:
                + Global Bivariate 2nd order analysis against random simulation by list

"""

################# Package import

import os
import pickle
import copy
import csv
import numpy as np
import scipy as sp
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg.globals import unpickle_obj, clean_dir
from pyorg.surf.model import ModelCSRV
from pyorg.surf.utils import list_tomoparticles_pvalues
from matplotlib import pyplot as plt, rcParams
try:
    import pickle as pickle
except ImportError:
    import pickle

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

BAR_WIDTH = .20

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils'

# Input STAR file
in_star_ref = ROOT_PATH + '/pre/ref_nomb_1_clean/ltomos_pre_premb_gather_mask/pre_premb_gather_mask_ltomos.star'
key_ref = 'pre'
in_star_1 = ROOT_PATH + '/ves_40/ltomos_lap/lap_ltomos.star'
key_1 = 'tether'
in_star_2 = ROOT_PATH + '/pst/ampar_vs_nmdar/org/ltomos/ltomos_ampar_nmdar_premb_mask/ampar_nmdar_premb_gather_mask_ltomos.star'
key_2 = 'ampar'
in_star_3 = ROOT_PATH + '/pst/ampar_vs_nmdar/org/ltomos/ltomos_ampar_nmdar_premb_mask/ampar_nmdar_premb_gather_mask_ltomos.star'
key_3 = 'nmdar'
in_tethers_csv = ROOT_PATH + '/pre/ref_nomb_1_clean/py_scripts/syn_num_tethers.csv'
in_wspace = None # ROOT_PATH + '/pst/ampar_vs_nmdar/org/coloc/coloc_tether_vs_pre_ampar_nmdar/coloc_5_60_5_sim_2_global_test_wspace.pkl' # (Insert a path to recover a pickled workspace instead of doing a new computation)

# Output directory
out_dir = ROOT_PATH + '/pst/ampar_vs_nmdar/org/aln/aln_tether_ampar_nmdar_vs_pre'
out_stem = 'aln_10_25_5_sim_20_global_maxd_20' # ''uni_sph_4_60_5'

# Pre-processing variables
pre_ssup = 5 #nm
pre_min_parts = 1

# Analysis variables
ana_res = 0.684 # nm/voxel
ana_rg = np.arange(10, 25, 5) # np.arange(4, 100, 2) # in nm
ana_shell_thick = None # 15 # None # 3
ana_max_dist =20 # nm
ana_global = True
ana_border = True
ana_conv_iter = 100
ana_max_iter = 100000
ana_npr = 1 # None means Auto
ana_npr_model = 1 # In general equal to ana_npr unless you have memory problems

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = 20 # 200
p_per = 5 # %
# Particle surface
p_vtp = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre/vtps/sph_rad_5_surf.vtp'

# Figure saving options
fig_fmt = '.png' # if None they showed instead

# Plotting options
pt_sim_v = True
pt_cmap = plt.get_cmap('gist_rainbow')
pt_xmin, pt_xmax = 10, 25

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
    aligns = np.zeros(shape=len(neighs), dtype=bool)
    for i, p_neigh in enumerate(neighs):
        if ((p_neigh[mask]>0).sum()) > 0:
            aligns[i] = True
    return aligns

# Units conversion
ana_rg_v = ana_rg / ana_res
ana_shell_thick_v = None
if ana_shell_thick is not None:
    ana_shell_thick_v = float(ana_shell_thick) / ana_res
ana_max_dist_mask = ana_rg <= ana_max_dist

########## Print initial message

print('Second order analysis for colocalization to  ListTomoParticles by tomograms.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tInput reference STAR file: ' + str(in_star_ref))
print('\t\t-Key: ' + key_ref)
print('\tInput analysis STAR file 1: ' + str(in_star_1))
print('\t\t-Key: ' + key_1)
print('\tInput analysis STAR file 2: ' + str(in_star_2))
print('\t\t-Key: ' + key_2)
print('\tInput analysis STAR file 3: ' + str(in_star_3))
print('\t\t-Key: ' + key_3)
if in_wspace is not None:
    print('\tLoad workspace from: ' + in_wspace)
else:
    print('\tPre-processing: ')
    if pre_ssup is not None:
        print('\t\t-Scale supression: ' + str(pre_ssup) + ' nm')
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

print('\tLoading input data (only the first entry is loaded)...')
star_1, star_2, star_3, star_ref = sub.Star(), sub.Star(), sub.Star(), sub.Star()
try:
    star_1.load(in_star_1)
    star_2.load(in_star_2)
    star_3.load(in_star_3)
    star_ref.load(in_star_ref)
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
ltomos_pkl = star_ref.get_element('_psPickleFile', 0)
ref_list = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_1.get_element('_psPickleFile', 0)
list_1 = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_2.get_element('_psPickleFile', 0)
list_2 = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_3.get_element('_psPickleFile', 1)
list_3 = unpickle_obj(ltomos_pkl)
if pre_ssup is not None:
    pre_ssup_v = pre_ssup / ana_res
    ref_list.scale_suppression(pre_ssup_v)
    list_1.scale_suppression(pre_ssup_v)
    list_2.scale_suppression(pre_ssup_v)
    list_3.scale_suppression(pre_ssup_v)
set_lists = surf.SetListTomoParticles()
set_lists.add_list_tomos(list_1, key_1)
set_lists.add_list_tomos(list_2, key_2)
set_lists.add_list_tomos(list_3, key_3)
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
    ref_list.scale_suppression(pre_ssup_v)
if pre_min_parts > 0:
    set_lists.filter_by_particles_num_tomos(pre_min_parts)

print('\tBuilding the dictionaries...')
lists_count, tomos_count = 0, 0
lists_dic = dict()
lists_hash, tomos_hash = dict(), dict()
tomos_np, tomos_den, tomos_exp, tomos_sim = dict(), dict(), dict(), dict()
tomos_aln, tomos_aln_sim = dict(), dict()
lists_np, lists_den, lists_gden, lists_exp, lists_sim, lists_color = dict(), dict(), dict(), dict(), dict(), dict()
lists_aln, lists_aln_sim = dict(), dict()
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
            lists_aln[short_key] = list()
            lists_aln_sim[short_key] = list()
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
            if ana_global:
                tomos_aln[tkey] = dict()
                tomos_aln_sim[tkey] = dict()
            tomos_count += 1
        tomos_exp[tkey][short_key] = 0
        tomos_sim[tkey][short_key] = list()
        tomos_np[tkey][short_key] = 0
        tomos_den[tkey][short_key] = 0
        lists_np[short_key][tkey] = 0
        lists_den[short_key][tkey] = 0
        if ana_global:
            lists_sim[short_key][tkey] = list()
            tomos_aln_sim[tkey][short_key] = list()

print('\tComputing reference properties...')
vols = ref_list.get_volumes_dict()
with open(in_tethers_csv, mode='r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    vesicles = dict()
    for row in reader:
        vesicles[row[0]] = float(row[1])

if in_wspace is None:

    tomos_np_ref = dict()
    print('\tLIST COMPUTING LOOP:')
    for li, lkey in enumerate(lists_hash.values()):

        llist = lists_dic[lkey]
        tomos_np_ref[lkey] = dict()
        lists_gden_np, lists_gden_vol = 0., 0.
        print('\t\t-Processing list: ' + lkey)
        print('\t\t\t+Tomograms computing loop:')
        for lt, tkey in enumerate(tomos_hash.keys()):

            print('\t\t\t\t*Processing tomogram (list ' + str(li + 1) + ' of ' + str(
                len(list(lists_hash.values()))) + ', tomo ' + str(lt + 1) + ' of ' + str(len(list(tomos_hash.keys()))) + ') : ' + \
                  os.path.split(tkey)[1])
            ltomo = llist.get_tomo_by_key(tkey)

            print('\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1])
            try:
                ltomo, ref_tomo = llist.get_tomo_by_key(tkey), ref_list.get_tomo_by_key(tkey)
            except KeyError:
                print('WARNING: tomogram not in a list (' + tkey + ')')
                continue
            hold_np_ref = ref_tomo.get_num_particles()
            print('\t\t\t\t\t-Computing the number of particles for reference [' + str(key_ref) + ']: ' + str(hold_np_ref))
            if ref_tomo.get_num_particles() < pre_min_parts:
                print('WARNING: reference tomogram with less than ' + str(pre_min_parts) + ' paticles.')
                continue

            hold_np = ltomo.get_num_particles()

            print('\t\t\t\t\t-Computing the number of particles [' + str(lkey) + ']: ' + str(hold_np))
            tomos_np[tkey][lkey] = hold_np
            lists_np[lkey][tkey] = hold_np
            tomos_np_ref[lkey][tkey] = ltomo.get_num_particles() # ref_tomo.get_num_particles()
            hold_den = ltomo.compute_global_density()
            print('\t\t\t\t\t-Computing density by area [' + str(hold_den) + ']: ' + str(hold_den))
            tomos_den[tkey][lkey] = hold_den
            lists_den[lkey][tkey] = hold_den
            lists_gden_np += hold_np
            lists_gden_vol += ltomo.compute_voi_volume()

            print('\t\t\t\t\t-Computing bivariate second order metrics...')
            if ana_global:
                hold_arr_1, hold_arr_2 = ltomo.compute_bi_2nd_order(ref_tomo, ana_rg_v, thick=ana_shell_thick_v,
                                                                       border=ana_border, conv_iter=ana_conv_iter,
                                                                       max_iter=ana_max_iter, out_sep=2,
                                                                       npr=ana_npr, verbose=False)
                if (hold_arr_1 is not None) and (hold_arr_2 is not None):
                    tomos_exp[tkey][lkey] = (hold_arr_1, hold_arr_2)
                    lists_exp[lkey].append((hold_arr_1, hold_arr_2))
                    hold_aln = are_aligned(hold_arr_1, ana_max_dist_mask)
                    prop_aln =  float(hold_aln.sum()) / float(tomos_np_ref[lkey][tkey])
                    tomos_aln[tkey][lkey] = prop_aln
                    for arr_1 in hold_arr_1:
                        lists_aln[lkey].append(arr_1)
            else:
                hold_arr = ltomo.compute_bi_2nd_order(ref_tomo, ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                         conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                         out_sep=0, npr=ana_npr, verbose=False)
                if hold_arr is not None:
                    tomos_exp[tkey][lkey] = hold_arr
                    for npi in range(tomos_np_ref[lkey][tkey]):
                        lists_exp[lkey].append(hold_arr)

            print('\t\t\t\t\t-Simulating bivariate second order metrics...')
            if ana_global:
                hold_arr_1, hold_arr_2 = ltomo.simulate_bi_2nd_order(ref_tomo, p_nsims, ModelCSRV, part_vtp,
                                                                    ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                                    conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                                    npr=ana_npr, npr_model=ana_npr_model, switched=True, out_sep=2,
                                                                    tmp_folder=tmp_sim_folder,
                                                                    verbose=pt_sim_v)
                if (hold_arr_1 is not None) and (hold_arr_2 is not None):
                    for arr_1, arr_2 in zip(hold_arr_1, hold_arr_2):
                        tomos_sim[tkey][lkey].append((arr_1, arr_2))
                        lists_sim[lkey][tkey].append((arr_1, arr_2))
                        tomos_aln_sim[tkey][lkey].append(arr_1)
            else:
                hold_arr = ltomo.simulate_bi_2nd_order(ref_tomo, p_nsims, ModelCSRV, part_vtp,
                                                          ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                          conv_iter=ana_conv_iter, max_iter=ana_max_iter, out_sep=0,
                                                          npr=ana_npr, npr_model=ana_npr_model, tmp_folder=tmp_sim_folder,
                                                          verbose=pt_sim_v)
                if hold_arr is not None:
                    tomos_sim[tkey][lkey] = hold_arr
                    for npi in range(tomos_np_ref[lkey][tkey]):
                        for arr in hold_arr:
                            lists_sim[lkey].append(arr)

        if lists_gden_vol > 0:
            lists_gden[lkey] = lists_gden_np / lists_gden_vol
        else:
            lists_gden[lkey] = 0.

    if ana_global:
        print('\tGathering alingment limulations...')
        hold_tomos_aln_sim, hold_lists_aln = copy.deepcopy(tomos_aln_sim), copy.deepcopy(lists_aln)
        del tomos_aln_sim
        del lists_aln_sim
        del lists_aln
        tomos_aln_sim, lists_aln_sim, lists_aln = dict(), dict(), dict()
        for tkey in tomos_hash.keys():
            tomos_aln_sim[tkey] = dict()
            for lkey in lists_hash.values():
                tomos_aln_sim[tkey][lkey] = list()
        for lkey in lists_hash.values():
            lists_aln[lkey], lists_aln_sim[lkey] = list(), list()
            for n_sim in range(p_nsims):
                hold_arr_l, np_l = list(), 0
                for tkey in tomos_hash.keys():
                    if len(hold_tomos_aln_sim[tkey][lkey]) > 0:
                        idl, idx = n_sim * tomos_np_ref[lkey][tkey], (n_sim + 1) * tomos_np_ref[lkey][tkey]
                        hold_arr = list()
                        for arr in hold_tomos_aln_sim[tkey][lkey][idl:idx]:
                            hold_arr.append(arr)
                            hold_arr_l.append(arr)
                            np_l += 1
                        hold_aln = are_aligned(hold_arr, ana_max_dist_mask)
                        prop_aln = float(hold_aln.sum()) / float(tomos_np_ref[lkey][tkey])
                        tomos_aln_sim[tkey][lkey].append(prop_aln)
                if n_sim == 0:
                    hold_arr = hold_lists_aln[lkey]
                    np_l = len(hold_arr)
                    hold_aln = are_aligned(hold_arr, ana_max_dist_mask)
                    prop_aln = float(hold_aln.sum()) / float(np_l)
                    lists_aln[lkey] = prop_aln
                hold_aln_l = are_aligned(hold_arr_l, ana_max_dist_mask)
                prop_aln_l = float(hold_aln_l.sum()) / float(np_l)
                lists_aln_sim[lkey].append(prop_aln_l)

        print('\tGlobal computations by tomos...')
        hold_tomos_exp, hold_tomos_sim = copy.deepcopy(tomos_exp), copy.deepcopy(tomos_sim)
        del tomos_exp
        del tomos_sim
        del lists_exp
        del lists_sim
        tomos_exp, tomos_sim, lists_exp, lists_sim = dict(), dict(), dict(), dict()
        for tkey in hold_tomos_exp.keys():
            tomos_exp[tkey], tomos_sim[tkey] = dict(), dict()
            dens = tomos_den[tkey]
            for lkey, mat in zip(iter(hold_tomos_exp[tkey].keys()), iter(hold_tomos_exp[tkey].values())):
                if hasattr(mat, '__len__') and (len(mat) > 0):
                    # By tomos
                    gden = lists_gden[lkey]
                    arr_1, arr_2 = mat[0], mat[1]
                    if ana_shell_thick is None:
                        gl_arr = ana_rg * (np.cbrt((1. / dens[lkey]) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))) - 1.)
                    else:
                        gl_arr = (1. / dens[lkey]) * (arr_1.sum(axis=0) / arr_2.sum(axis=0)) - 1.
                    tomos_exp[tkey][lkey] = gl_arr
                    for npr in range(tomos_np_ref[lkey][tkey]):
                        try:
                            lists_exp[lkey].append(gl_arr)
                        except (KeyError, AttributeError):
                            lists_exp[lkey] = list()
                            lists_exp[lkey].append(gl_arr)
                    # By simulations
                    for n_sim in range(p_nsims):
                        idl, idx = n_sim*tomos_np_ref[lkey][tkey], (n_sim+1)*tomos_np_ref[lkey][tkey]
                        mat = hold_tomos_sim[tkey][lkey][idl:idx]
                        arr_1, arr_2 = list(), list()
                        for hold_mat in mat:
                            arr_1.append(hold_mat[0])
                            arr_2.append(hold_mat[1])
                        arr_1, arr_2 = np.asarray(arr_1), np.asarray(arr_2)
                        if ana_shell_thick is None:
                            gl_arr = ana_rg * (np.cbrt((1. / gden) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))) - 1.)
                        else:
                            gl_arr = (1. / gden) * (arr_1.sum(axis=0) / arr_2.sum(axis=0))  - 1.
                        try:
                            tomos_sim[tkey][lkey].append(gl_arr)
                        except (KeyError, AttributeError):
                            tomos_sim[tkey][lkey] = list()
                            tomos_sim[tkey][lkey].append(gl_arr)
                        for npr in range(tomos_np_ref[lkey][tkey]):
                            try:
                                lists_sim[lkey].append(gl_arr)
                            except (KeyError, AttributeError):
                                lists_sim[lkey] = list()
                                lists_sim[lkey].append(gl_arr)

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print('\tPickling computation workspace in: ' + out_wspace)
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash,
              tomos_np, tomos_den, tomos_exp, tomos_sim,
              lists_np, lists_den, lists_exp, lists_sim, lists_color,
              vesicles, vols,
              tomos_aln, tomos_aln_sim, lists_aln, lists_aln_sim)
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
    lists_np, lists_den, lists_exp, lists_sim, lists_color = wspace[8], wspace[9], wspace[10], wspace[11], wspace[12]
    vesicles, vols = wspace[13], wspace[14]
    tomos_aln, tomos_aln_sim, lists_aln, lists_aln_sim = wspace[15], wspace[16], wspace[17], wspace[18]

print('\tPrinting lists hash: ')
for id, lkey in zip(iter(lists_hash.keys()), iter(lists_hash.values())):
    print('\t\t-[' + str(id) + '] -> [' + lkey + ']')
print('\tPrinting tomograms hash: ')
for tkey, val in zip(iter(tomos_hash.keys()), iter(tomos_hash.values())):
    print('\t\t-[' + tkey + '] -> [' + str(val) + ']')

# print '\tPrinting number of samples and volume for the reference by tomogram: '
# for tomo in ref_list.get_tomo_list():
#     tkey = tomo.get_tomo_fname()
#     print '\t\t-' + tkey + ': ' + str(tomo.get_num_particles()) + ' np, ' + str(tomo.compute_voi_volume()) + ' nm**3'

# Getting the lists colormap
n_lists = len(list(lists_hash.keys()))
for i, lkey in zip(iter(lists_hash.keys()), iter(lists_hash.values())):
    lists_color[lkey] = pt_cmap(1.*i/n_lists)

print('\tTOMOGRAMS PLOTTING LOOP: ')

out_tomos_dir = out_stem_dir + '/tomos'
os.makedirs(out_tomos_dir)

print('\t\t-Plotting 2nd order metric...')
for tkey, ltomo in zip(iter(tomos_exp.keys()), iter(tomos_exp.values())):
    if len(list(ltomo.keys())) <= 0:
        continue
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    # plt.title('Univariate 2nd order for ' + tkey_short)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Scale (nm)')
    for lkey, arr in zip(iter(tomos_exp[tkey].keys()), iter(tomos_exp[tkey].values())):
        if not hasattr(arr, '__len__'):
            print('\t\t\t+WARNING: no experimental values for tomogram ' + tkey + ' and list ' + lkey)
            continue
        if lkey == key_1:
            plt.plot(ana_rg, arr, color='blue', label=lkey, linewidth=2)
        elif lkey == key_2:
            plt.plot(ana_rg, arr, color='black', label=lkey, linewidth=2)
        elif lkey == key_3:
            plt.plot(ana_rg, arr, color='red', label=lkey, linewidth=2)
    try:
        hold_sim = tomos_sim[tkey][lkey]
        if len(hold_sim) > 0:
            ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(hold_sim))
            plt.fill_between(ana_rg, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
            plt.plot(ana_rg, ic_med, 'gray', linewidth=2)
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
        plt.savefig(hold_dir + '/bi.png')
    plt.close()

print('\t\t-Plotting alignment metric...')
for tkey, ltomo in zip(iter(tomos_aln.keys()), iter(tomos_aln.values())):
    if len(list(ltomo.keys())) <= 0:
        continue
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # plt.title('Univariate 2nd order for ' + tkey_short)
    ax1.set_xlabel('Tether alignment (a.u.)')
    ax1.set_ylabel('proportion')
    ax2.set_ylabel('p-value')
    for lkey, arr in zip(iter(tomos_aln[tkey].keys()), iter(tomos_aln[tkey].values())):
        if lkey == key_1:
            ax1.bar(0.70, tomos_aln[tkey][lkey], BAR_WIDTH,
                    color='blue', linewidth=2)
            ax2.bar(1.10, compute_pval(tomos_aln[tkey][lkey], np.asarray(tomos_aln_sim[tkey][lkey])), BAR_WIDTH,
                    color='blue', linewidth=2, hatch='/')
        elif lkey == key_2:
            ax1.bar(1.70, tomos_aln[tkey][lkey], BAR_WIDTH,
                    color='gray', linewidth=2)
            ax2.bar(2.10, compute_pval(tomos_aln[tkey][lkey], np.asarray(tomos_aln_sim[tkey][lkey])), BAR_WIDTH,
                    color='gray', linewidth=2, hatch='/')
        elif lkey == key_3:
            ax1.bar(2.70, tomos_aln[tkey][lkey], BAR_WIDTH,
                    color='red', linewidth=2)
            ax2.bar(3.10, compute_pval(tomos_aln[tkey][lkey], np.asarray(tomos_aln_sim[tkey][lkey])), BAR_WIDTH,
                    color='red', linewidth=2, hatch='/')
    plt.xticks((1, 2, 3), ('TETHER', 'AMPAR', 'NMDAR'))
    plt.plot((0.5, 3.5), (0.95, 0.95), color='k', linewidth=2, linestyle='--')
    plt.xlim(0.5, 3.5)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_yticks((.5, .95, 1.), ('0.5', '*', '1'))
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/align.png')
    plt.close()


print('\tLISTS PLOTTING LOOP: ')

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

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
    plt.plot(ana_rg, ic_med, color='gray', linewidth=2, linestyle='-', label='TETHER$\Rightarrow$RANDOM')
    plt.fill_between(ana_rg, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    if ana_global:
        if lkey == key_1:
            plt.plot(ana_rg, np.asarray(tlist).mean(axis=0), color='blue', linewidth=2, linestyle='-', label='TETHER$\Rightarrow$TETHER')
        elif lkey == key_2:
            plt.plot(ana_rg, np.asarray(tlist).mean(axis=0), color='black', linewidth=2, linestyle='-', label='TETHER$\Rightarrow$AMPAR')
        elif lkey == key_3:
            plt.plot(ana_rg, np.asarray(tlist).mean(axis=0), color='red', linewidth=2, linestyle='-', label='TETHER$\Rightarrow$NMDAR')
    else:
        plt.plot(ana_rg, np.asarray(tlist).mean(axis=0), color=lists_color[lkey], linestyle='-', label=lkey)
    plt.xlim(pt_xmin, pt_xmax)
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/uni_list_' + lkey + '.png')
    plt.close()

print('\t\t-Plotting alignment metric...')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('Tether alignment (a.u.)')
ax1.set_ylabel('proportion')
ax2.set_ylabel('p-value')
for lkey in lists_aln.keys():
    if lkey == key_1:
        ax1.bar(0.70, lists_aln[lkey], BAR_WIDTH, color='blue', linewidth=2)
        ax2.bar(1.10, compute_pval(lists_aln[lkey], np.asarray(lists_aln_sim[lkey])), BAR_WIDTH,
                color='blue', linewidth=2, hatch='/')
    elif lkey == key_2:
        ax1.bar(1.70, lists_aln[lkey], BAR_WIDTH, color='gray', linewidth=2)
        ax2.bar(2.10, compute_pval(lists_aln[lkey], np.asarray(lists_aln_sim[lkey])), BAR_WIDTH,
                color='gray', linewidth=2, hatch='/')
    elif lkey == key_3:
        ax1.bar(2.70, lists_aln[lkey], BAR_WIDTH, color='red', linewidth=2)
        ax2.bar(3.10, compute_pval(lists_aln[lkey], np.asarray(lists_aln_sim[lkey])), BAR_WIDTH,
                color='red', linewidth=2, hatch='/')
plt.xticks((1, 2, 3), ('TETHER', 'AMPAR', 'NMDAR'))
plt.plot((0.5, 3.5), (0.95, 0.95), color='k', linewidth=2, linestyle='--')
plt.xlim(0.5, 3.5)
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_yticks((.5, .95, 1.), ('0.5', '*', '1'))
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/align.png')
plt.close()

print('Successfully terminated. (' + time.strftime("%c") + ')')
