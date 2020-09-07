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

BAR_WIDTH = .35

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils'

# Input STAR file
in_star_ref = ROOT_PATH + '/pre/ref_nomb_1_clean/ltomos_clst_flt_high_lap/clst_flt_high_lap_ltomos.star'
key_ref = 'pre'
in_star_1 = ROOT_PATH + '/pst/nrt/ltomos_clst_flt_high_lap/clst_flt_high_lap_ltomos.star'
key_1 = 'nrt'
in_star_2 = ROOT_PATH + '/ves_40/ltomos_lap/lap_ltomos.star'
key_2 = 'tet'
in_tethers_csv = ROOT_PATH + '/pre/ref_nomb_1_clean/py_scripts/syn_num_tethers.csv'
in_wspace = ROOT_PATH + '/pre/ref_nomb_1_clean/coloc_pre_nrt_ves_40_lap_tomos/coloc_15_50_3_sim_10_rad_5_wspace.pkl' # (Insert a path to recover a pickled workspace instead of doing a new computation)

# Output directory
out_dir = ROOT_PATH + '/pre/ref_nomb_1_clean/coloc_pre_nrt_ves_40_lap_tomos'
out_stem = 'coloc_15_50_3_sim_10_rad_5' # ''uni_sph_4_60_5'

# Analysis variables
ana_res = 0.684 # nm/voxel
ana_rg = np.arange(15, 50, 3) # np.arange(4, 100, 2) # in nm
ana_shell_thick = None # 3
ana_loc = False
ana_border = True
ana_conv_iter = 100
ana_max_iter = 100000
ana_npr = 1 # None means Auto
ana_npr_model = 1 # In general equal to ana_npr unless you have memory problems

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = 10
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
if in_wspace is not None:
    print('\tLoad workspace from: ' + in_wspace)
print('\tOrganization analysis settings: ')
print('\t\t-Range of radius: ' + str(ana_rg) + ' nm')
print('\t\t-Range of radius: ' + str(ana_rg_v) + ' voxels')
if ana_shell_thick is None:
    print('\t\t-Spherical neighborhood')
else:
    print('\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick) + ' nm')
    print('\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick_v) + ' voxels')
if ana_loc:
    print('\t\t-Compute just the local densities.')
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

if in_wspace is None:

    print('\tLoading input data (only the first entry is loaded)...')
    star_1, star_2, star_ref = sub.Star(), sub.Star(), sub.Star()
    try:
        star_1.load(in_star_1)
        star_2.load(in_star_2)
        star_ref.load(in_star_ref)
    except pexceptions.PySegInputError as e:
        print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
    ltomos_pkl = star_ref.get_element('_psPickleFile', 0)
    ref_list = unpickle_obj(ltomos_pkl)
    ltomos_pkl = star_1.get_element('_psPickleFile', 0)
    list_1 = unpickle_obj(ltomos_pkl)
    ltomos_pkl = star_1.get_element('_psPickleFile', 0)
    list_2 = unpickle_obj(ltomos_pkl)
    set_lists = surf.SetListTomoParticles()
    set_lists.add_list_tomos(list_1, key_1)
    set_lists.add_list_tomos(list_2, key_2)
    try:
        part_vtp = disperse_io.load_poly(p_vtp)
    except pexceptions.PySegInputError as e:
        print('ERROR: reference particle surface file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)

    print('\tBuilding the dictionaries...')
    lists_count, tomos_count = 0, 0
    lists_dic = dict()
    lists_hash, tomos_hash = dict(), dict()
    tomos_np, tomos_den, tomos_exp, tomos_sim = dict(), dict(), dict(), dict()
    lists_np, lists_den, lists_exp, lists_sim, lists_color = dict(), dict(), dict(), dict(), dict()
    tmp_sim_folder = out_dir + '/tmp_gen_list_' + out_stem
    set_lists_dic = set_lists.get_lists()
    for lkey, llist in zip(iter(set_lists_dic.keys()), iter(set_lists_dic.values())):
        print('\t\t-Processing list: ' + lkey)
        try:
            lists_dic[lkey]
        except KeyError:
            lists_dic[lkey] = llist
            lists_hash[lists_count] = lkey
            lists_np[lkey], lists_den[lkey], lists_exp[lkey], lists_sim[lkey] = None, None, list(), list()
            lists_count += 1
    for lkey, llist in zip(iter(set_lists_dic.keys()), iter(set_lists_dic.values())):
        llist_tomos_dic = llist.get_tomos()
        for tkey, ltomo in zip(iter(llist_tomos_dic.keys()), iter(llist_tomos_dic.values())):
            try:
                ref_tomo = ref_list.get_tomo_by_key(tkey)
            except KeyError:
                print('\t\t\t\t\t-WARNING: tomogram ' + tkey + ' discarded because it is not in reference list!')
                continue
            try:
                tomos_np[tkey]
            except KeyError:
                tomos_hash[tkey] = tomos_count
                tomos_np[tkey], tomos_den[tkey] = list(), list()
                tomos_exp[tkey], tomos_sim[tkey] = dict.fromkeys(list(lists_dic.keys())), dict.fromkeys(list(lists_dic.keys()))
                tomos_count += 1
    for lkey in lists_np.keys():
        lists_np[lkey] = np.zeros(shape=tomos_count, dtype=np.float32)
    for lkey in lists_den.keys():
        lists_den[lkey] = np.zeros(shape=tomos_count, dtype=np.float32)
    for tkey in tomos_exp.keys():
        for lkey in lists_dic.keys():
            tomos_exp[tkey][lkey], tomos_sim[tkey][lkey] = list(), list()

    print('\tComputing reference properties...')
    vols = ref_list.get_volumes_dict()
    with open(in_tethers_csv, mode='r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        vesicles = dict()
        for row in reader:
            vesicles[row[0]] = float(row[1])

    print('\tLIST COMPUTING LOOP:')
    for lkey in list(lists_hash.values()):

        llist = lists_dic[lkey]
        print('\t\t-Processing list: ' + lkey)
        print('\t\t\t+Tomograms computing loop:')
        for tkey in list(tomos_hash.keys()):

            print('\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1])
            ltomo, ref_tomo = llist.get_tomo_by_key(tkey), ref_list.get_tomo_by_key(tkey)

            print('\t\t\t\t\t-Computing the number of particles...')
            hold_np = ltomo.get_num_particles()
            tomos_np[tkey].append(hold_np)
            lists_np[lkey][tomos_hash[tkey]] = hold_np

            print('\t\t\t\t\t-Computing density by area...')
            hold_den = ltomo.compute_global_density()
            tomos_den[tkey].append(hold_den)
            lists_den[lkey][tomos_hash[tkey]] = hold_den

            print('\t\t\t\t\t-Computing bivariate second order metrics...')
            if ana_loc:
                hold_num, hold_den = ref_tomo.compute_bi_2nd_order(ltomo, ana_rg_v, thick=ana_shell_thick_v,
                                                                   border=ana_border, conv_iter=ana_conv_iter,
                                                                   max_iter=ana_max_iter, out_sep=True,
                                                                   npr=ana_npr, verbose=False)
                if (hold_den is None) or (hold_den is None):
                    hold_arr = None
                else:
                    hold_arr = hold_num / hold_den
            else:
                hold_arr = ref_tomo.compute_bi_2nd_order(ltomo, ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                         conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                         out_sep=False, npr=ana_npr, verbose=False)
            if hold_arr is not None:
                tomos_exp[tkey][lkey].append(hold_arr)
                for i in range(ref_tomo.get_num_particles()):
                    lists_exp[lkey].append(hold_arr)

            print('\t\t\t\t\t-Simulating bivariate second order metrics...')
            if ana_loc:
                hold_num, hold_den = ref_tomo.simulate_bi_2nd_order(ltomo, p_nsims, ModelCSRV, part_vtp,
                                                                    ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                                    conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                                    npr=ana_npr, npr_model=ana_npr_model, out_sep=True,
                                                                    tmp_folder=tmp_sim_folder,
                                                                    verbose=pt_sim_v)
                if (hold_den is None) or (hold_den is None):
                    hold_arr = None
                else:
                    hold_arr = hold_num / hold_den
            else:
                hold_arr = ref_tomo.simulate_bi_2nd_order(ltomo, p_nsims, ModelCSRV, part_vtp,
                                                          ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                          conv_iter=ana_conv_iter, max_iter=ana_max_iter, out_sep=False,
                                                          npr=ana_npr, npr_model=ana_npr_model, tmp_folder=tmp_sim_folder,
                                                          verbose=pt_sim_v)
            if hold_arr is not None:
                for arr in hold_arr:
                    tomos_sim[tkey][lkey].append(arr)
                    for i in range(ref_tomo.get_num_particles()):
                        lists_sim[lkey].append(arr)

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print('\tPickling computation workspace in: ' + out_wspace)
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash,
              tomos_np, tomos_den, tomos_exp, tomos_sim,
              lists_np, lists_den, lists_exp, lists_sim, lists_color,
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
    lists_np, lists_den, lists_exp, lists_sim, lists_color = wspace[8], wspace[9], wspace[10], wspace[11], wspace[12]
    vesicles, vols = wspace[13], wspace[14]

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

print('\t\t-Plotting the number of particles...')
for count, tkey, ltomo in zip(list(range(len(list(tomos_np.keys())))), iter(tomos_np.keys()), iter(tomos_np.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.title('Num. particles for ' + tkey_short)
    plt.ylabel('Num. particles')
    plt.xlabel('Classes')
    for i, nparts in enumerate(ltomo):
        lkey = lists_hash[i]
        plt.bar(count, nparts, width=0.75, color=lists_color[lkey], label=lkey)
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
for count, tkey, ltomo in zip(list(range(len(list(tomos_den.keys())))), iter(tomos_den.keys()), iter(tomos_den.values())):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.title('Density for ' + tkey_short)
    plt.ylabel('Density (np/vol)')
    plt.xlabel('Classes')
    for i, den in enumerate(ltomo):
        lkey = lists_hash[i]
        plt.bar(count, den, width=0.75, color=lists_color[lkey], label=lkey)
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

print('\t\t-Plotting 2nd order metric...')
low_pvals, high_pvals = dict(), dict()
mask_rg = (ana_rg >= ana_rg[0]) & (ana_rg <= ana_rg[-1])
for tkey, ltomo in zip(iter(tomos_exp.keys()), iter(tomos_exp.values())):
    p_values = dict()
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    for lkey, arr in zip(iter(tomos_exp[tkey].keys()), iter(tomos_exp[tkey].values())):
        try:
            arr = arr[0]
            hold_sim = tomos_sim[tkey][lkey]
            if len(hold_sim) > 0:
                ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(hold_sim))
                p_values[lkey] = compute_pvals(arr, np.asarray(hold_sim))
        except IndexError:
            print('\t\t\t+WARNING: no data for tomogram ' + tkey + ' and list ' + lkey)
            continue
        plt.figure()
        plt.title('Bivariate 2nd order for ' + tkey_short + ' and ' + lkey)
        if ana_shell_thick is None:
            plt.ylabel('Ripley\'s L')
        else:
            plt.ylabel('Ripley\'s O')
        plt.xlabel('Scale (nm)')
        plt.plot(ana_rg, arr, color=lists_color[lkey], label=lkey)
        plt.legend(loc=4)
        plt.plot(ana_rg, ic_low, 'k--')
        plt.plot(ana_rg, ic_med, 'k')
        plt.plot(ana_rg, ic_high, 'k--')
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            hold_dir = out_tomos_dir + '/' + tkey_short
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            plt.savefig(hold_dir + '/bi_' + lkey + '.png')
        plt.close()

    plt.figure()
    plt.title('Clustering p-value for ' + tkey_short)
    plt.ylabel('p-value')
    plt.xlabel('Scale (nm)')
    try:
        for lkey, vals in zip(iter(p_values.keys()), iter(p_values.values())):
            plt.plot(ana_rg, vals, color=lists_color[lkey], linestyle='-', label=lkey)
            hold_vals = vals[mask_rg]
            try:
                high_pvals[lkey][tkey] = hold_vals.max()
                low_pvals[lkey][tkey] = 1 - vals.min()
            except KeyError:
                high_pvals[lkey] = dict()
                low_pvals[lkey] = dict()
                high_pvals[lkey][tkey] = hold_vals.max()
                low_pvals[lkey][tkey] = 1 - vals.min()
    except IndexError:
        print('\t\t\t+WARNING: no p-values for list: ' + lkey)
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

print('\t\t-Plotting p-values box (tomo[s_low, s_high]= (min_pval, max_pval)):')
for lkey, high_pval in zip(iter(high_pvals.keys()), iter(high_pvals.values())):
    low_pval = low_pvals[lkey]
    plt.figure()
    if ana_shell_thick is None:
        plt.title('Ripley\'s L p-values box-plot')
        plt.ylabel('Ripley\'s L (p-values)')
    else:
        plt.title('Ripley\'s O p-values box-plot')
        plt.ylabel('Ripley\'s O (p-values)')
    plt.boxplot([list(low_pval.values()), list(high_pval.values())], labels=['Low', 'High'])
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_tomos_dir + '/Box_pvals' + str(lkey) + '_.png')
    plt.close()

print('\t\t-Plotting co-localization by synapse membrane volumes...')
hold_off = 0.
plt.figure()
# plt.title('Colocalization respect ' + key_ref)
plt.ylabel('p-value')
plt.xlabel('Membrane volume [$10^{6} \cdot nm^{3}$]')
for i, lkey in enumerate(high_pvals.keys()):
    l_vols, l_pvals = list(), list()
    for tkey in high_pvals[lkey].keys():
        l_vols.append(vols[tkey])
        l_pvals.append(high_pvals[lkey][tkey])
    l_vols, l_pvals = np.asarray(l_vols, dtype=np.float), np.asarray(l_pvals, dtype=np.float)
    if lkey == 'tet':
        plt.plot(10**(-6)*(l_vols + i*hold_off), l_pvals, color='green', marker='o', markersize=10, label=lkey, linestyle='')
    elif lkey == 'nrt':
        plt.plot(10**(-6)*(l_vols + i*hold_off), l_pvals, color='red', marker='*', markersize=10, label=lkey, linestyle='')
    else:
        plt.plot(10**(-6)*(l_vols + i*hold_off), l_pvals, color=lists_color[lkey], marker='o', markersize=3, label=lkey, linestyle='')
plt.ylim((-0.1, 1.1))
plt.xlim((0, 10**(-6)*l_vols.max()*1.1))
x_line = np.linspace(0, l_vols.max()*1.1, 10)
plt.plot(x_line, np.zeros(shape=len(x_line)), 'k--')
plt.plot(x_line, np.ones(shape=len(x_line)), 'k--')
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/coloc_by_vols.png')
plt.close()

print('\t\t-Plotting co-localization by tethered vesicles...')
plt.figure()
# plt.title('Colocalization respect ' + key_ref)
plt.ylabel('p-value')
plt.xlabel('Number of tetherd vesicles')
for i, lkey in enumerate(high_pvals.keys()):
    l_ves, l_pvals = list(), list()
    for tkey in high_pvals[lkey].keys():
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        l_ves.append(vesicles[tkey_short])
        l_pvals.append(high_pvals[lkey][tkey])
    l_ves, l_pvals = np.asarray(l_ves, dtype=np.float), np.asarray(l_pvals, dtype=np.float)
    if lkey == 'tet':
        plt.plot(l_ves + i*hold_off, l_pvals, color='green', marker='o', markersize=10, label=lkey, linestyle='')
    elif lkey == 'nrt':
        plt.plot(l_ves + i*hold_off, l_pvals, color='red', marker='*', markersize=10, label=lkey, linestyle='')
    else:
        plt.plot(l_ves + i*hold_off, l_pvals, color=lists_color[lkey], markersize=10, label=lkey, linestyle='')
plt.ylim((-0.1, 1.1))
plt.xlim((0, l_ves.max()*1.1))
x_line = np.linspace(0, l_ves.max()*1.1, 10)
plt.plot(x_line, np.zeros(shape=len(x_line)), 'k--')
plt.plot(x_line, np.ones(shape=len(x_line)), 'k--')
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/coloc_by_ntet.png')
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
    for i, nparts in enumerate(tlist):
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
    plt.ylabel('Density (np/vol)')
    plt.xlabel('Tomograms')
    for i, den in enumerate(tlist):
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
    plt.title('Bivariate 2nd order for ' + lkey)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Scale (nm)')
    # Getting experimental IC
    exp_low, exp_med, exp_high = compute_ic(p_per, np.asarray(tlist))
    plt.plot(ana_rg, exp_low, color=lists_color[lkey], linestyle='--', label=lkey)
    plt.plot(ana_rg, exp_med, color=lists_color[lkey], linestyle='-', label=lkey)
    plt.plot(ana_rg, exp_high, color=lists_color[lkey], linestyle='--', label=lkey)
    # Getting simulations IC
    hold_sim = lists_sim[lkey]
    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(hold_sim))
    plt.plot(ana_rg, ic_low, 'k--')
    plt.plot(ana_rg, ic_med, 'k')
    plt.plot(ana_rg, ic_high, 'k--')
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_lists_dir + '/bi_list_' + lkey + '.png')
    plt.close()
sims = np.asarray(hold_sim)

print('\t\t-Plotting clustering p-value...')
plt.figure()
plt.title('Clustering p-value')
plt.ylabel('p-value')
plt.xlabel('Scale (nm)')
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    exp_med = compute_ic(p_per, np.asarray(tlist))[1]
    p_values = compute_pvals(exp_med, sims)
    plt.plot(ana_rg, p_values, color=lists_color[lkey], linestyle='-', label=lkey)
plt.legend(loc=4)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/pvals_lists.png')
plt.close()
with open(out_lists_dir + '/pvals_lists.npy', 'w') as p_file:
    np.save(p_file, p_values)

print('Successfully terminated. (' + time.strftime("%c") + ')')

