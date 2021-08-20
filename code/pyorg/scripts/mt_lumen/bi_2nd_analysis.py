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
import numpy as np
import scipy as sp
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg.globals import unpickle_obj, clean_dir
from pyorg.surf.model import ModelCSRV
from pyorg.surf.utils import list_tomoparticles_pvalues
import matplotlib.pyplot as plt
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

ROOT_PATH = '/fs/pool/pool-plitzko/Saikat/luminal_particle_organization/plus_end_clustering_test' # '/fs/pool/pool-plitzko/Saikat/luminal_particle_organization/plus_end_clustering' # '/fs/pool/pool-plitzko/Saikat/luminal_particle_organization/lattice_break_clustering'

# Input STAR file
in_star_ref = ROOT_PATH + '/ltomos/v1_nobc_proj_lumen_eroded4/v1_nobc_ltomos_surf.star' # '/ltomos/v1_nobc_proj/v1_nobc_proj_ltomos.star'
in_ref_short_key = '0' # '0'
in_star = ROOT_PATH + '/ltomos/v1_nobc_proj_lumen_eroded4/v1_nobc_ltomos_surf.star' # '/ltomos/v1_nobc_proj/v1_nobc_proj_ltomos.star'  #
in_wspace = None # ROOT_PATH + '/data/tutorials/synth_sumb/org/uni_2nd/test_1_ref_0_proj_8_300_10_sim_10_wspace.pkl'  # (Insert a path to recover a pickled workspace instead of doing a new computation)

# Output directory
out_dir = ROOT_PATH + '/org/bi/bi_2nd/'
out_stem = 'v1_nobc_proj_lumen_eroded4_10_600_25_switch'  # 'uni_sph_4_60_5'

# Analysis variables
ana_res = 1.368  # nm/voxel
ana_rg = np.arange(10, 600, 25)  # in nm
ana_shell_thick = None  # 15 # 5
ana_rdf = False # True
ana_fmm = True # False

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = 200
p_per = 5  # %

# Multiprocessing settings
mp_npr = 10 # None means Auto

##### Advanced settings

ana_conv_iter = None # 100
ana_max_iter = None # 100000

ana_border = True

# Particle surface
p_vtp = None
p_cinter = 0.05
p_switch = True # False

# Figure saving options
fig_fmt = '.png'  # if None they showed instead

# Plotting options
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

print('Bivariate 1:N second order analysis for a ListTomoParticles by tomograms.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tInput reference STAR file: ' + str(in_star_ref))
print('\tInput reference short key: ' + str(in_ref_short_key))
print('\tInput analysis STAR file: ' + str(in_star))
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
    if ana_rdf:
        print('\t\t-RDF activated.')
if in_wspace is None:
    if (ana_conv_iter is not None) and (ana_max_iter is not None):
        print('\t\t-Convergence number of samples for stochastic volume estimations: ' + str(ana_conv_iter))
        print('\t\t-Maximum number of samples for stochastic volume estimations: ' + str(ana_max_iter))
    else:
        if ana_fmm:
            print('\t\t-FMM for distance metric computation.')
        else:
            print('\t\t-DT for distance transform computation.')
    if mp_npr is None:
        print('\t\t-Number of processors: Auto')
    else:
        print('\t\t-Number of processors: ' + str(mp_npr))
print('\tP-Value computation setting:')
print('\t\t-Percentile: ' + str(p_per) + ' %')
print('\t\t-Number of instances for simulations: ' + str(p_nsims))
print('\tSurface settings:')
if p_vtp is not None:
    print('\t\t-Particle surface: ' + p_vtp)
print('\t\t-Maximum fraction of overlapping surface for simulated particles:' + str(p_cinter))
if p_switch:
    print('\t\t-Random simulations using the reference pattern.')
if fig_fmt is not None:
    print('\tStoring figures:')
    print('\t\t-Format: ' + str(fig_fmt))
else:
    print('\tPlotting settings: ')
print('\t\t-Colormap: ' + str(pt_cmap))
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
star, star_ref = sub.Star(), sub.Star()
try:
    star.load(in_star)
    star_ref.load(in_star_ref)
except pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
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
    print('ERROR: reference ListTomoParticles with short key ' + in_ref_short_key + ' was not found!')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

if in_wspace is None:

    set_lists = surf.SetListTomoParticles()
    part_vtps = dict()
    for row in range(star.get_nrows()):
        ltomos_pkl = star.get_element('_psPickleFile', row)
        ltomos = unpickle_obj(ltomos_pkl)
        set_lists.add_list_tomos(ltomos, ltomos_pkl)
        try:
            if p_vtp is not None:
                part_vtp = disperse_io.load_poly(p_vtp)
            else:
                part_vtp = disperse_io.load_poly(star.get_element('_suSurfaceVtpSim', row))
        except pexceptions.PySegInputError as e:
            print('ERROR: reference particle surface file could not be loaded because of "' + e.get_message() + '"')
            print('Terminated. (' + time.strftime("%c") + ')')
            sys.exit(-1)
        lkey = os.path.split(ltomos_pkl)[1]
        short_key_idx = lkey.index('_')
        short_key = lkey[:short_key_idx]
        part_vtps[short_key] = part_vtp

    print('\tBuilding the dictionaries...')
    lists_count, tomos_count = 0, 0
    lists_dic = dict()
    lists_hash, tomos_hash = dict(), dict()
    tomos_np, tomos_den, tomos_exp, tomos_sim = dict(), dict(), dict(), dict()
    lists_np, lists_den, lists_exp, lists_sim, lists_color = dict(), dict(), dict(), dict(), dict()
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
            lists_np[short_key], lists_den[short_key], lists_exp[short_key], lists_sim[short_key] = None, None, \
                                                                                                    list(), list()
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

    print('\tLIST COMPUTING LOOP:')
    for lkey in list(lists_hash.values()):

        if (in_star_ref in in_star) and (lkey == in_ref_short_key):
            print('\t\t-Processing and reference list of particles are the same, skipping...')
            continue

        llist = lists_dic[lkey]
        print('\t\t-Processing list: ' + lkey)
        part_vtp = part_vtps[lkey]
        print('\t\t\t+Tomograms computing loop:')
        for tkey in list(tomos_hash.keys()):

            print('\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1])
            ltomo, ref_tomo = llist.get_tomo_by_key(tkey), ref_list.get_tomo_by_key(tkey)
            if ref_tomo.get_num_particles() <= 0:
                print('\t\t\t\t\t\t+No reference particles found, skipping...')
                continue

            print('\t\t\t\t\t-Computing the number of particles...')
            hold_np = ltomo.get_num_particles()
            tomos_np[tkey].append(hold_np)
            lists_np[lkey][tomos_hash[tkey]] = hold_np

            print('\t\t\t\t\t-Computing density by area...')
            hold_den = ltomo.compute_global_density()
            tomos_den[tkey].append(hold_den)
            lists_den[lkey][tomos_hash[tkey]] = hold_den

            print('\t\t\t\t\t-Computing bivariate second order metrics...')
            hold_arr = ref_tomo.compute_bi_2nd_order(ltomo, ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                     conv_iter=ana_conv_iter, max_iter=ana_max_iter, fmm=ana_fmm,
                                                     rdf=ana_rdf, npr=mp_npr)
            if hold_arr is not None:
                tomos_exp[tkey][lkey].append(hold_arr)
                for i in range(ref_tomo.get_num_particles()):
                    lists_exp[lkey].append(hold_arr)

            print('\t\t\t\t\t-Simulating bivariate second order metrics...')
            model_csrv = ModelCSRV(check_inter=p_cinter)
            # hold_arr = ltomo.simulate_uni_2nd_order(p_nsims, model_csrv, part_vtp, 'center', ana_rg_v,
            #                                         border=ana_border,
            #                                         thick=ana_shell_thick_v, fmm=ana_fmm,
            #                                         conv_iter=ana_conv_iter, max_iter=ana_max_iter, rdf=ana_rdf,
            #                                         npr=mp_npr)
            hold_arr = ref_tomo.simulate_bi_2nd_order(ltomo, p_nsims, model_csrv, part_vtp, 'center',
                                                      ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                      conv_iter=ana_conv_iter, max_iter=ana_max_iter, rdf=ana_rdf,
                                                      fmm=ana_fmm, npr=mp_npr, switched=p_switch)
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
              lists_np, lists_den, lists_exp, lists_sim, lists_color)
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

print('\tPrinting lists hash: ')
for id, lkey in zip(iter(lists_hash.keys()), iter(lists_hash.values())):
    print('\t\t-[' + str(id) + '] -> [' + lkey + ']')
print('\tPrinting tomograms hash: ')
for tkey, val in zip(iter(tomos_hash.keys()), iter(tomos_hash.values())):
    print('\t\t-[' + tkey + '] -> [' + str(val) + ']')

print('\tPrinting number of samples and volume for the reference by tomogram: ')
for tomo in ref_list.get_tomo_list():
    tkey = tomo.get_tomo_fname()
    print('\t\t-' + tkey + ': ' + str(tomo.get_num_particles()) + ' np, ' + str(tomo.compute_voi_volume()) + ' nm**3')

# Getting the lists colormap
n_lists = len(list(lists_hash.keys()))
for i, lkey in zip(iter(lists_hash.keys()), iter(lists_hash.values())):
    lists_color[lkey] = pt_cmap(1. * i / n_lists)

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
    for i, nparts in enumerate(ltomo):
        lkey = lists_hash[i]
        plt.bar(int(lkey), nparts, width=0.75, color=lists_color[lkey], label=lkey)
    plt.xticks(BAR_WIDTH + np.arange(n_lists), np.arange(n_lists))
    plt.legend(loc=1)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        # hold_dir = out_tomos_dir + '/' + tkey_short
        hold_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
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
    for i, den in enumerate(ltomo):
        lkey = lists_hash[i]
        plt.bar(int(lkey), den, width=0.75, color=lists_color[lkey], label=lkey)
    plt.xticks(BAR_WIDTH + np.arange(n_lists), np.arange(n_lists))
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        # hold_dir = out_tomos_dir + '/' + tkey_short
        hold_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/den.png')
    plt.close()

print('\t\t-Plotting 2nd order metric...')
low_pvals, high_pvals = list(), list()
mask_rg = (ana_rg >= 7) & (ana_rg <= 100)
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
            if ana_rdf:
                plt.ylabel('Radial Distribution Function')
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
            # hold_dir = out_tomos_dir + '/' + tkey_short
            hold_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
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
            high_pvals.append(hold_vals.max())
            low_pvals.append(1 - vals.min())
    except IndexError:
        print('\t\t\t+WARNING: no p-values for list: ' + lkey)
    plt.legend(loc=4)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        # hold_dir = out_tomos_dir + '/' + tkey_short
        hold_dir = out_tomos_dir + '/' + tkey.replace('/', '_')
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/pvals.png')
    plt.close()

print('\t\t-Plotting p-values box (tomo[s_low, s_high]= (min_pval, max_pval)):')
plt.figure()
if ana_shell_thick is None:
    plt.title('Ripley\'s L p-values box-plot')
    plt.ylabel('Ripley\'s L (p-values)')
else:
    if ana_rdf:
        plt.title('RDF p-values box-plot')
        plt.ylabel('RDF (p-values)')
    else:
        plt.title('Ripley\'s O p-values box-plot')
        plt.ylabel('Ripley\'s O (p-values)')
plt.boxplot([low_pvals, high_pvals], labels=['Low', 'High'])
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/Box_pvals.png')
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

for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    if len(tlist) <= 0:
        continue
    plt.figure()
    # plt.title('Univariate 2nd order for ' + lkey)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        if ana_rdf:
            plt.ylabel('Radial Distribution Function')
        else:
            plt.ylabel('Ripley\'s O')
    plt.xlabel('Scale [nm]')
    # Getting experimental IC
    hold_arr = np.asarray(tlist)
    hold_arr[np.isnan(hold_arr) | np.isinf(hold_arr)] = 0
    exp_low, exp_med, exp_high = compute_ic(p_per, hold_arr)
    # plt.plot(ana_rg, exp_low, color=lists_color[lkey], linestyle='--', label=lkey)
    plt.plot(ana_rg, exp_med, color=lists_color[lkey], linestyle='-', label=lkey, linewidth=2.0)
    # plt.plot(ana_rg, exp_high, color=lists_color[lkey], linestyle='--', label=lkey)
    plt.fill_between(ana_rg, exp_low, exp_high, alpha=0.3, color=lists_color[lkey], edgecolor='w')
    # sims += lists_sim[lkey]
    # Getting simulations IC
    hold_sim = np.asarray(lists_sim[lkey])
    hold_sim[np.isnan(hold_sim) | np.isinf(hold_sim)] = 0
    ic_low, ic_med, ic_high = compute_ic(p_per, np.asarray(hold_sim))
    try:
        # plt.plot(ana_rg, ic_low, 'k--')
        plt.plot(ana_rg, ic_med, 'k', linewidth=2.0)
        # plt.plot(ana_rg, ic_high, 'k--')
    except (NameError, ValueError):
        do_pval = False
    plt.fill_between(ana_rg, ic_low, ic_high, alpha=0.5, color='gray', edgecolor='w')
    plt.tight_layout()
    plt.xlim((0, ana_rg[-1]))
    if fig_fmt is None:
        plt.show(block=True)
    else:
        # plt.show(block=True)
        plt.savefig(out_lists_dir + '/bi_lists_' + str(lkey) + '.png', dpi=600)
    plt.close()

print('\t\t-Plotting 2nd order metric...')
sims = list()
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    if len(tlist) <= 0:
        continue
    plt.figure()
    plt.title('Bivariate 2nd order for ' + lkey)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        if ana_rdf:
            plt.ylabel('Radial Distribution Function')
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
plt.title('Co-localization significance')
plt.ylabel('p-value')
plt.xlabel('Scale [nm]')
for lkey, tlist in zip(iter(lists_exp.keys()), iter(lists_exp.values())):
    if len(tlist) <= 0:
        continue
    exp_med = compute_ic(p_per, np.asarray(tlist))[1]
    p_values = compute_pvals(exp_med, sims)
    plt.plot(ana_rg, p_values, color=lists_color[lkey], linestyle='-', label=lkey)
plt.legend(loc=4)
plt.tight_layout()
plt.xlim((0, ana_rg[-1]))
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/pvals_lists.png')
plt.close()

print('Successfully terminated. (' + time.strftime("%c") + ')')


