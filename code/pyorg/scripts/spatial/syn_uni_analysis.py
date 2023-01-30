"""

    Performs Univariate 2nd order analysis from a STAR file with a set of ListTomoParticles entries

    Input:  - A STAR file with a set of ListTomoParticles pickles

    Output: - Plots with the analysis
            - Matrix with the analysis for further post-processing

"""

################# Package import

import os
import pickle
import numpy as np
import scipy as sp
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg.globals import unpickle_obj, sort_dict
from pyorg.surf.model import ModelCSRV
from pyorg.surf.utils import list_tomoparticles_pvalues
import matplotlib.pyplot as plt

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

BAR_WIDTH = .35

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre'

# Input STAR file
in_star = ROOT_PATH + '/ref_nomb_1_clean/ltomos_pre_premb/pre_premb_ltomos.star' # '/ref_a3/ltomos/pre_ltomos.star' # '/ref_a2/ltomos/pre_ref_a2_12_ltomos_hold.star'

# Input matrices (optional - organization analysis is skipped)
in_mats_lists = ROOT_PATH + '/ref_nomb_1_clean/uni_pre/uni_4_60_5_no_pwise_sim_100_keep_5_org_lists.pkl' # ROOT_PATH + '/ref_a3/uni_sph/uni_sph_4_60_2_org_list.pkl'

# Output directory
out_dir = ROOT_PATH + '/ref_nomb_1_clean/uni_pre'
out_stem = 'uni_4_60_5_no_pwise_keep_5_sim_100' # ''uni_sph_4_60_5'

# Analysis variables
ana_res = 0.684 # nm/voxel
ana_rg = np.arange(4, 60, 5) # np.arange(4, 100, 2) # in nm
ana_shell_thick = None # 3
ana_border = True
ana_keep = 5
ana_conv_iter = 20
ana_max_iter = 100000
ana_gl = False
ana_npr = 5 # None means Auto

# P-value computation settings
# Simulation model (currently only CSRV)
do_p_value = True
p_nsims = 100
p_per = 5 # %
p_pwise = False
# Particle surface
p_vtp = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre/vtps/sph_rad_0.5_surf.vtp'
in_mats_sims = ROOT_PATH + '/ref_nomb_1_clean/uni_pre/uni_4_60_5_no_pwise_sim_100_keep_5_org_sims.pkl'

# Figure saving options
fig_fmt = None # '.png' # if None they showed instead

# Plotting options
pt_xrange = None # [10, 25]
pt_yrange = None # [0, 10]
pt_cmap = plt.get_cmap('gist_rainbow')

########################################################################################
# MAIN ROUTINE
########################################################################################

# Units conversion
ana_rg_v = ana_rg / ana_res
ana_shell_thick_v = None
if ana_shell_thick is not None:
    ana_shell_thick_v = float(ana_shell_thick) / ana_res

########## Print initial message

print('Univariate second order analysis for a ListTomoParticles.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tInput STAR file: ' + str(in_star))
if in_mats_lists is None:
    print('\tOrganization analysis settings: ')
    print('\t\t-Range of radius: ' + str(ana_rg) + ' nm')
    print('\t\t-Range of radius: ' + str(ana_rg_v) + ' voxels')
    if ana_shell_thick is None:
        print('\t\t-Spherical neighborhood')
    else:
        print('\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick) + ' nm')
        print('\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick_v) + ' voxels')
    if ana_keep is not None:
        print('\t\t-Keep highest populated tomograms per list: ' + str(ana_keep))
    print('\t\t-Convergence number of samples for stochastic volume estimations: ' + str(ana_conv_iter))
    print('\t\t-Maximum number of samples for stochastic volume estimations: ' + str(ana_max_iter))
    if ana_gl:
        print('\t\t-Global analysis activated.')
    if ana_npr is None:
        print('\t\t-Number of processors: Auto')
    else:
        print('\t\t-Number of processors: ' + str(ana_npr))
else:
    if in_mats_lists is not None:
        print('\tDensity ratios by list dictionary pickled from file: ' + in_mats_lists)
if do_p_value:
    print('\tP-Value computation setting:')
    print('\t\t-Percentile: ' + str(p_per) + ' %')
    if in_mats_sims is None:
        print('\t\t-Number of instances for simulations: ' + str(p_nsims))
        print('\t\t-Particle surface: ' + p_vtp)
    else:
        print('\t\t-Using pre-computed matrix for simulations: ' + in_mats_sims)
    if p_pwise:
        print('\t\t-Pointwise analysis activated.')
    else:
        print('\t\t-Tomogram analysis activated.')
if fig_fmt is not None:
    print('\tStoring figures:')
    print('\t\t-Format: ' + str(fig_fmt))
else:
    print('\tPlotting settings: ')
print('\t\t-Colormap: ' + str(pt_cmap))
print('\t\t-X-axis range: ' + str(pt_xrange))
print('\t\t-Y-axis range: ' + str(pt_yrange))
print('')

######### Process

print('Main Routine: ')
mats_lists, gl_lists = None, None

print('\tLoading input STAR file...')
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
    if ana_keep is not None:
        ltomos.clean_low_pouplated_tomos(ana_keep)
    set_lists.add_list_tomos(ltomos, ltomos_pkl)

print('\tComputing number of particles by list...')
np_lists = set_lists.particles_by_list()
np_lists_skeys, np_lists_svalues = sort_dict(np_lists, np_lists, reverse=True)
color_lists, klass_lbls = dict(), dict()
for i, key in enumerate(np_lists_skeys):
    klass_lbl = os.path.splitext(os.path.split(key)[1])[0]
    try:
        k_idx = klass_lbl.index('_')
        klass_lbl = klass_lbl[:k_idx]
    except IndexError:
        pass
    color_lists[key] = pt_cmap(1.*i/len(np_lists))
    klass_lbls[key] = klass_lbl
    print('\t\t-List ' + str(i+1) + ': ' + str(klass_lbl))
plt.figure()
plt.title('Num. particles by list')
plt.ylabel('Num. particles')
plt.xlabel('List')
it, bars, lbls = 0, list(), list()
for key, val in zip(np_lists_skeys, np_lists_svalues):
    lbl = klass_lbls[key]
    bar, = plt.bar(it, val, width=0.75, color=color_lists[key], label=lbl)
    it += 1
    bars.append(bar)
    lbls.append(lbl)
plt.legend(bars, lbls, loc=1)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_np_lists.png')
plt.close()
with open(out_dir + '/' + out_stem + '_np_lists.pkl', "wb") as fl:
    pickle.dump(np_lists, fl)
    fl.close()

print('\tComputing number of particles by tomogram...')
np_tomos = set_lists.particles_by_tomos()
np_tomos_skeys, np_tomos_svalues = sort_dict(np_tomos, np_tomos, reverse=True)
color_tomos, tomo_lbls = dict(), dict()
for i, key in enumerate(np_tomos_skeys):
    tomo_lbl = os.path.split(key)[1]
    try:
        t_idx = tomo_lbl.index('_bin')
        tomo_lbl = tomo_lbl[:t_idx]
    except ValueError:
        pass
    color_tomos[key] = pt_cmap(1.*i/len(np_tomos))
    tomo_lbls[key] = tomo_lbl
    print('\t\t-Tomogram ' + str(i+1) + ': ' + str(tomo_lbl))
plt.figure()
plt.title('Num. particles by tomograms')
plt.ylabel('Num. particles')
plt.xlabel('Tomograms')
it, bars, lbls = 0, list(), list()
for key, val in zip(np_tomos_skeys, np_tomos_svalues):
    lbl = tomo_lbls[key]
    bar, = plt.bar(it, val, width=0.75, color=color_tomos[key], label=lbl)
    it += 1
    bars.append(bar)
    lbls.append(lbl)
plt.legend(bars, lbls, loc=1)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_np_tomos.png')
plt.close()
with open(out_dir + '/' + out_stem + '_np_tomos.pkl', "wb") as fl:
    pickle.dump(np_tomos, fl)
    fl.close()

print('\tComputing densities by list...')
gl_lists = set_lists.density_by_list(surface=False)
gl_lists_skeys, gl_lists_svalues = sort_dict(gl_lists, gl_lists, reverse=True)
color_lists, klass_lbls = dict(), dict()
for i, key in enumerate(gl_lists_skeys):
    klass_lbl = os.path.splitext(os.path.split(key)[1])[0]
    try:
        k_idx = klass_lbl.index('_')
        klass_lbl = klass_lbl[:k_idx]
    except ValueError:
        pass
    color_lists[key] = pt_cmap(1.*i/len(gl_lists))
    klass_lbls[key] = klass_lbl
    print('\t\t-List ' + str(i+1) + ': ' + str(klass_lbl))
plt.figure()
plt.title('Density by list')
den_cte = ana_res**3
plt.ylabel('Particles/nm**3')
plt.xlabel('List')
it, bars, lbls = 0, list(), list()
for key, val in zip(gl_lists_skeys, gl_lists_svalues):
    lbl = klass_lbls[key]
    bar, = plt.bar(it, val, width=0.75, color=color_lists[key], label=lbl)
    it += 1
    bars.append(bar)
    lbls.append(lbl)
plt.legend(bars, lbls, loc=1)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_den_lists.png')
plt.close()
with open(out_dir + '/' + out_stem + '_den_lists.pkl', "wb") as fl:
    pickle.dump(gl_lists, fl)
    fl.close()

print('\tComputing densities by tomogram...')
gl_tomos = set_lists.density_by_tomos(surface=False)
gl_tomos_skeys, gl_tomos_svalues = sort_dict(gl_tomos, gl_tomos, reverse=True)
color_tomos, tomo_lbls = dict(), dict()
for i, key in enumerate(gl_tomos_skeys):
    tomo_lbl = os.path.split(key)[1]
    try:
        t_idx = tomo_lbl.index('_bin')
        tomo_lbl = tomo_lbl[:t_idx]
    except IndexError:
        pass
    color_tomos[key] = pt_cmap(1.*i/len(gl_tomos))
    tomo_lbls[key] = tomo_lbl
    print('\t\t-Tomogram ' + str(i+1) + ': ' + str(tomo_lbl))
plt.figure()
plt.title('Density by tomograms')
plt.ylabel('Particles/nm**3')
plt.xlabel('Tomograms')
it, bars, lbls = 0, list(), list()
for key, vals in zip(gl_tomos_skeys, gl_tomos_svalues):
    lbl = tomo_lbls[key]
    bar, = plt.bar(it, den_cte*np.asarray(vals, dtype=float), width=0.75, color=color_tomos[key], label=lbl)
    it += 1
    bars.append(bar)
    lbls.append(lbl)
plt.legend(bars, lbls, loc=1)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_den_tomos.png')
plt.close()
with open(out_dir + '/' + out_stem + '_den_tomos.pkl', "wb") as fl:
    pickle.dump(gl_tomos, fl)
    fl.close()

if in_mats_lists is None:

    if ana_keep is None:
        print('\tComputing densities proportions of particles for every tomogram...')
        plt.figure()
        plt.title('Class proportions for tomograms.')
        props_dic = set_lists.proportions_by_list()
        set_tomos_name= set_lists.get_set_tomos()
        set_ntomos = len(set_tomos_name)
        index, offset = np.arange(set_ntomos), np.zeros(shape=set_ntomos, dtype=float)
        for key_1, props in zip(iter(props_dic.keys()), iter(props_dic.values())):
            plt.bar(index+.5*BAR_WIDTH, props, BAR_WIDTH, color=color_lists[key_1], bottom=offset,
                    label=klass_lbls[key_1])
            offset += props
        syn_tomos_name = list()
        for tomos_name in set_tomos_name:
            syn_tomos_name.append(tomo_lbls[tomos_name][4:])
        plt.xticks(index + BAR_WIDTH, syn_tomos_name)
        plt.legend()
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            plt.savefig(out_dir + '/' + out_stem + '_tomo_prop.png')
        plt.close()

    print('\tComputing organization by list...')
    mats_lists = set_lists.compute_uni_2nd_order_by_list(distances=ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                         conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                         dens_gl=ana_gl, npr=ana_npr, verbose=True)
    with open(out_dir + '/' + out_stem + '_org_lists.pkl', "wb") as fl:
        pickle.dump(mats_lists, fl)
        fl.close()

print('\tPickling organization by lists...')
if in_mats_lists is not None:
    with open(in_mats_lists, 'r') as pkl:
        mats_lists = pickle.load(pkl)

if mats_lists is not None:
    plt.figure()
    plt.title('Univariate 2nd Order by list')
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Radius')
    lines, lbls = list(), list()
    mats_slists_keys, mats_slists_values = sort_dict(mats_lists, gl_lists, reverse=True)
    for key, mat in zip(mats_slists_keys, mats_slists_values):
        lbl = klass_lbls[key]
        # hold_mat = stat_dict_to_mat(mat, ltomos)
        # arr = hold_mat.mean(axis=0)
        line, = plt.plot(ana_rg, mat, color=color_lists[key], label=lbl)
        lines.append(line)
        lbls.append(lbl)
    plt.legend(lines, lbls, loc=1)
    if pt_xrange is not None:
        plt.xlim(pt_xrange)
    if pt_yrange is not None:
        plt.ylim(pt_yrange)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_org_list.png')
    plt.close()
else:
    print('ERROR: organization could not be computed')
    print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

if do_p_value:

    if in_mats_sims is None:
        in_model = ModelCSRV
        try:
            part_vtp = disperse_io.load_poly(p_vtp)
        except pexceptions.PySegInputError as e:
            print('ERROR: reference particle surface file could not be loaded because of "' + e.get_message() + '"')
            print('Terminated. (' + time.strftime("%c") + ')')
            sys.exit(-1)

        print('\tComputing simulations for p-value by list...')
        mats_sims = set_lists.simulate_uni_2nd_order_matrix(p_nsims, ModelCSRV, part_vtp,
                                                            ana_rg_v, thick=ana_shell_thick_v, border=ana_border,
                                                            conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                            dens_gl=ana_gl, pointwise=p_pwise,
                                                            npr=ana_npr, verbose=True)
        with open(out_dir + '/' + out_stem + '_org_sims.pkl', "wb") as fl:
            pickle.dump(mats_sims, fl)
            fl.close()
    else:
        with open(in_mats_sims, 'r') as pkl:
            mats_sims = pickle.load(pkl)

    lines, lbls = list(), list()
    mats_slists_keys, mats_slists_values = sort_dict(mats_lists, gl_lists, reverse=True)
    for key, mat in zip(mats_slists_keys, mats_slists_values):
        lbl = klass_lbls[key]
        plt.figure()
        plt.title('Simulation univariate 2nd ' + lbl)
        if ana_shell_thick is None:
            plt.ylabel('Ripley\'s L - IC [' + str(p_per) + ', ' + str(100-p_per) + ']%')
        else:
            plt.ylabel('Ripley\'s O - IC [' + str(p_per) + ', ' + str(100-p_per) + ']%')
        plt.xlabel('Radius')
        # hold_mat = stat_dict_to_mat(mat, ltomos)
        # arr = hold_mat.mean(axis=0)
        plt.plot(ana_rg, mat, 'b')
        plt.plot(ana_rg, np.median(mats_sims[key], axis=0), 'k')
        plt.plot(ana_rg, np.percentile(mats_sims[key], p_per, axis=0), 'k--')
        plt.plot(ana_rg, np.percentile(mats_sims[key], 100-p_per, axis=0), 'k--')
        if pt_xrange is not None:
            plt.xlim(pt_xrange)
        if pt_yrange is not None:
            plt.ylim(pt_yrange)
        plt.tight_layout()
        # plt.show(block=key)
        if fig_fmt is None:
            plt.show(block=True)
        else:
            plt.savefig(out_dir + '/' + out_stem + '_' + lbl + '_org_list.png')
        plt.close()

    print('\tComputing p-values...')
    p_values = list_tomoparticles_pvalues(ana_rg, mats_lists, mats_sims)

    plt.figure()
    plt.title('Maximum low P-Value')
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L (p-value)')
    else:
        plt.ylabel('Ripley\'s O  (p-value)')
    # plt.xticks(klass_lbls)
    lines, lbls = list(), list()
    p_values_keys, p_values_values = sort_dict(p_values, gl_lists, reverse=True)
    for i, key, p_value in zip(list(range(len(p_values_keys))), p_values_keys, p_values_values):
        lbl = klass_lbls[key]
        line, = plt.bar(i, 1.+.01*p_value[2], color=color_lists[key], label=lbl)
    plt.legend()
    if pt_xrange is not None:
        plt.xlim(pt_xrange)
    if pt_yrange is not None:
        plt.ylim(pt_yrange)
    plt.tight_layout()
    # plt.show(block=True)
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_pval_low.png')
    plt.close()

    plt.figure()
    plt.title('Radius for maximum low P-values')
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L (radius)')
    else:
        plt.ylabel('Ripley\'s O  (radius)')
    # plt.xticks(klass_lbls)
    lines, lbls = list(), list()
    p_values_keys, p_values_values = sort_dict(p_values, gl_lists, reverse=True)
    for i, key, p_value in zip(list(range(len(p_values_keys))), p_values_keys, p_values_values):
        lbl = klass_lbls[key]
        line, = plt.bar(i, p_value[0], color=color_lists[key], label=lbl)
    plt.legend()
    if pt_xrange is not None:
        plt.xlim(pt_xrange)
    if pt_yrange is not None:
        plt.ylim(pt_yrange)
    plt.tight_layout()
    # plt.show(block=True)
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_pval_low_rad.png')
    plt.close()

    plt.figure()
    plt.title('Maximum high P-Value')
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L (p-value)')
    else:
        plt.ylabel('Ripley\'s O  (p-value)')
    # plt.xticks(klass_lbls)
    lines, lbls = list(), list()
    p_values_keys, p_values_values = sort_dict(p_values, gl_lists, reverse=True)
    for i, key, p_value in zip(list(range(len(p_values_keys))), p_values_keys, p_values_values):
        lbl = klass_lbls[key]
        line, = plt.bar(i, 1.-.01*p_value[3], color=color_lists[key], label=lbl)
        # line, = plt.bar(int(lbl), 1. - .01 * p_value[3])
    plt.legend()
    if pt_xrange is not None:
        plt.xlim(0, len(p_values_keys))
    if pt_yrange is not None:
        plt.ylim(pt_yrange)
    # plt.xticks(np.arange(len(p_values_keys)))
    plt.tight_layout()
    # plt.show(block=True)
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_pval_high.png')
    plt.close()

    plt.figure()
    plt.title('Radius for maximum low P-values')
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L (radius)')
    else:
        plt.ylabel('Ripley\'s O  (radius)')
    # plt.xticks(klass_lbls)
    lines, lbls = list(), list()
    p_values_keys, p_values_values = sort_dict(p_values, gl_lists, reverse=True)
    for i, key, p_value in zip(list(range(len(p_values_keys))), p_values_keys, p_values_values):
        lbl = klass_lbls[key]
        line, = plt.bar(i, p_value[1], color=color_lists[key], label=lbl)
    plt.legend()
    if pt_xrange is not None:
        plt.xlim(pt_xrange)
    if pt_yrange is not None:
        plt.ylim(pt_yrange)
    plt.tight_layout()
    # plt.show(block=True)
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_pval_high_rad.png')
    plt.close()

print('Successfully terminated. (' + time.strftime("%c") + ')')