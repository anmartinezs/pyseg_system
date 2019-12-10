"""

    Performs Univariate 2nd order analysis and comparison again a model from a ListTomoParticles

    Input:  - The path to the pickled ListTomoParticles object
            - Parameters to set up the model simulation

    Output: - Plots with the analysis
            - Matrix with the analysis for further post-processing

"""

################# Package import

import os
import math
import pickle
import numpy as np
import scipy as sp
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg.surf.model import ModelCSRV, gen_tlist_from_tlist
from pyorg.globals import unpickle_obj, sort_dict
import matplotlib.pyplot as plt
from pyorg.surf import stat_dict_to_mat

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils'

# Input ListTomoParticlesPickle
in_pkl_1 = ROOT_PATH + '/pre/ref_a3/ltomos/0_ref_3_20_50_12_all_tpl.pkl' # '/ref_a2/ltomos/0_ref_3_20_50_12_tpl.pkl' # '/ref_a3/ltomos/pre_ltomos.star'
in_pkl_2 = ROOT_PATH + '/az/ref_a3/ltomos/0_ref_3_6_50_12_all_tpl.pkl' # '/az/ref_a2/ltomos/0_run1_data_tpl.pkl' # '/ref_a3/ltomos/pre_ltomos.star'

# Particle surface
in_vtp = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre/vtps/sph_rad_0.5_surf.vtp'

# Computation shortcut pickle files
in_mat_tomos = None # ROOT_PATH + '/pre/ref_a3/bi_pre_az_sim/bi_pre_az_sim_shell_3_80_2_org_tomos.pkl'
in_mat_sims = None # ROOT_PATH + '/pre/ref_a3/bi_pre_az_sim/bi_pre_az_sim_shell_3_80_2_org_sims.pkl'

# Output directory
out_dir = ROOT_PATH + '/pre/ref_a3/bi_pre_az_sim/' #'/ref_a3/uni_sph'
out_stem = 'bi_pre_az_sim_shell_3_60_2' # 'uni_sph_4_60_2'

# Analysis variables
ana_res = 0.684 # nm/voxel
ana_rg = np.arange(4, 60, 3) # np.arange(4, 100, 2)
ana_shell_thick = 3 # None
ana_border = True
ana_conv_iter = 1000
ana_max_iter = 100000
ana_npr = 10 # None means Auto

# Simulation model (currently only CSRV)
rnd_bi = True
rnd_n = 1
rnd_conf_mean = False # True, mean centrality (Gaussian distribution), False median (Generic distribution)
rnd_conf_val = 2.5 # if mean then it is the number of sigmas, otherwise percentile in %

# Figure saving options
fig_fmt = '.png' # if None they showed instead

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

print 'Bivariate second order analysis for a ListTomoParticles.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
print '\tInput Pickle file 1: ' + str(in_pkl_1)
print '\tInput Pickle file 1: ' + str(in_pkl_2)
print '\tParticle referece surface file: ' + str(in_vtp)
print '\tOrganization analysis settings: '
if in_mat_tomos is None:
    print '\t\t-Range of radius: ' + str(ana_rg) + ' nm'
    print '\t\t-Range of radius: ' + str(ana_rg_v) + ' voxels'
    if ana_shell_thick is None:
        print '\t\t-Spherical neighborhood'
    else:
        print '\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick) + ' nm'
        print '\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick_v) + ' voxels'
    print '\t\t-Convergence number of samples for stochastic volume estimations: ' + str(ana_conv_iter)
    print '\t\t-Maximum number of samples for stochastic volume estimations: ' + str(ana_max_iter)
    if ana_npr is None:
        print '\t\t-Number of processors: Auto'
    else:
        print '\t\t-Number of processors: ' + str(ana_npr)
else:
    print '\tDensity ratio by tomograms dictionary pickled from file: ' + in_mat_tomos
print '\tRandom model settings (CSRV):'
if rnd_bi:
    print '\t\t-Double patterns random.'
else:
    print '\t\t-Single patterns random.'
if in_mat_sims is None:
    print '\t\t-Number of instances: ' + str(rnd_n)
else:
    print '\tSimulation instances for density ratio pickled from file: ' + in_mat_sims
if rnd_conf_mean:
    print '\t\t-N sigmas for Gaussian confidence interval: ' + str(rnd_conf_val)
else:
    print '\t\t-Percentile for the generic confidence interval: ' + str(rnd_conf_val) + ' %'
if fig_fmt is not None:
    print '\tStoring figures:'
    print '\t\t-Format: ' + str(fig_fmt)
else:
    print '\tPlotting settings: '
print '\t\t-Colormap: ' + str(pt_cmap)
print '\t\t-X-axis range: ' + str(pt_xrange)
print '\t\t-Y-axis range: ' + str(pt_yrange)
print ''

######### Process

print 'Main Routine: '
mat_tomos, mat_sims = None, None
den_cte = 1e6

print '\tUnpickling input list of tomograms...'
try:
    tomos_list_1, tomos_list_2 = unpickle_obj(in_pkl_1), unpickle_obj(in_pkl_2)
except pexceptions.PySegInputError as e:
    print 'ERROR: input Pickle file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\tComputing densities by tomogram for list 1...'
gl_tomos_1 = tomos_list_1.densities_by_tomos()
gl_tomos_skeys_1, gl_tomos_svalues_1 = sort_dict(gl_tomos_1, gl_tomos_1, reverse=True)
color_tomos_1, tomo_lbls_1 = dict(), dict()
for i, key in enumerate(gl_tomos_skeys_1):
    tomo_lbl = os.path.split(key)[1]
    try:
        t_idx = tomo_lbl.index('_bin')
        tomo_lbl = tomo_lbl[:t_idx]
    except IndexError:
        pass
    color_tomos_1[key] = pt_cmap(1.*i/len(gl_tomos_1))
    tomo_lbls_1[key] = tomo_lbl
    print '\t\t-Tomogram ' + str(i+1) + ': ' + str(tomo_lbl)
plt.figure()
plt.title('Density by tomograms for list 1')
plt.ylabel('Density (x' + str(den_cte) + ')')
plt.xlabel('Tomograms')
it, bars, lbls = 0, list(), list()
for key, vals in zip(gl_tomos_skeys_1, gl_tomos_svalues_1):
    lbl = tomo_lbls_1[key]
    bar, = plt.bar(it, den_cte*np.asarray(vals, dtype=float), width=0.75, color=color_tomos_1[key], label=lbl)
    it += 1
    bars.append(bar)
    lbls.append(lbl)
plt.legend(bars, lbls, loc=1)
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_den_tomos_1.png')
plt.close()
with open(out_dir + '/' + out_stem + '_den_tomos_1.pkl', "wb") as fl:
    pickle.dump(gl_tomos_1, fl)
    fl.close()

print '\tComputing densities by tomogram for list 2...'
gl_tomos_2 = tomos_list_2.densities_by_tomos()
gl_tomos_skeys_2, gl_tomos_svalues_2 = sort_dict(gl_tomos_2, gl_tomos_2, reverse=True)
color_tomos_2, tomo_lbls_2 = dict(), dict()
for i, key in enumerate(gl_tomos_skeys_2):
    tomo_lbl = os.path.split(key)[1]
    try:
        t_idx = tomo_lbl.index('_bin')
        tomo_lbl = tomo_lbl[:t_idx]
    except IndexError:
        pass
    color_tomos_2[key] = pt_cmap(1.*i/len(gl_tomos_2))
    tomo_lbls_2[key] = tomo_lbl
    print '\t\t-Tomogram ' + str(i+1) + ': ' + str(tomo_lbl)
plt.figure()
plt.title('Density by tomograms')
plt.ylabel('Density (x' + str(den_cte) + ')')
plt.xlabel('Tomograms')
it, bars, lbls = 0, list(), list()
for key, vals in zip(gl_tomos_skeys_2, gl_tomos_svalues_2):
    lbl = tomo_lbls_1[key]
    bar, = plt.bar(it, den_cte*np.asarray(vals, dtype=float), width=0.75, color=color_tomos_2[key], label=lbl)
    it += 1
    bars.append(bar)
    lbls.append(lbl)
plt.legend(bars, lbls, loc=1)
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_den_tomos_2.png')
plt.close()
with open(out_dir + '/' + out_stem + '_den_tomos_2.pkl', "wb") as fl:
    pickle.dump(gl_tomos_2, fl)
    fl.close()

if in_mat_tomos is None:
    print '\tComputing organization by list...'
    mat_tomos = tomos_list_1.compute_bi_2nd_order_by_tomos(tomos_list_2, distances=ana_rg_v,
                                                           thick=ana_shell_thick_v, border=ana_border,
                                                           conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                           npr=ana_npr, verbose=True)
    with open(out_dir + '/' + out_stem + '_org_tomos.pkl', "wb") as fl:
        pickle.dump(mat_tomos, fl)
        fl.close()

if in_mat_sims is None:
    in_model = ModelCSRV
    out_model = out_dir + '/' + out_stem + '_model_tomo.pkl'
    print '\tPickling an instance of the mode in:' + out_model
    try:
        part_vtp = disperse_io.load_poly(in_vtp)
    except pexceptions.PySegInputError as e:
        print 'ERROR: reference particle surface file could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    hold_tomo = tomos_list_2.get_tomo_by_key(gl_tomos_skeys_2[0])
    ex_model = in_model(hold_tomo.get_voi(), part_vtp)
    model_tomo = ex_model.gen_instance(hold_tomo.get_num_particles(), 'example_model', mode='center')
    model_tomo.pickle(out_model)

    print '\tComputing simulations with model: ' + str(type(in_model))
    if rnd_bi:
        n_tomos = len(tomos_list_1.get_tomo_list())
        n_parts_tomo = int(math.ceil(tomos_list_1.get_num_particles() / n_tomos))
        ltomos_csrv = gen_tlist_from_tlist(tomos_list_1, part_vtp, in_model, mode_emb='center', npr=ana_npr)
        mat_sims = ltomos_csrv.simulate_bi_2nd_order_by_tomos(tomos_list_2, n_sims=rnd_n, temp_model=in_model,
                                                               part_vtp=part_vtp, border=ana_border,
                                                               distances=ana_rg_v, thick=ana_shell_thick_v,
                                                               conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                               npr=ana_npr, verbose=True)
    else:
        mat_sims = tomos_list_1.simulate_bi_2nd_order_by_tomos(tomos_list_2, n_sims=rnd_n, temp_model=in_model,
                                                               part_vtp=part_vtp, border=ana_border,
                                                               distances=ana_rg_v, thick=ana_shell_thick_v,
                                                               conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                               npr=ana_npr, verbose=True)
    with open(out_dir + '/' + out_stem + '_org_sims.pkl', "wb") as fl:
        pickle.dump(mat_sims, fl)
        fl.close()

print '\tPickling organization by lists...'
if in_mat_tomos is not None:
    with open(in_mat_tomos, 'r') as pkl:
        mat_tomos = pickle.load(pkl)
print '\tPickling organization simulations...'
if in_mat_sims is not None:
    with open(in_mat_sims, 'r') as pkl:
        mat_sims = pickle.load(pkl)

if (mat_tomos is not None) and (mat_sims is not None):
    gl_den = tomos_list_2.compute_global_density()
    if gl_den <= 0:
        print 'ERROR: global density for the list is lower or equal to zero so no further statistics can be displayed!'
        print 'Unsuccesfully terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    plt.figure()
    plt.title('Univariate 2nd Order')
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Radius')
    # Metrics computation
    hmat = stat_dict_to_mat(mat_tomos, tomos_list_1)
    hmats = stat_dict_to_mat(mat_sims, tomos_list_1)
    if rnd_conf_mean:
        arr_shift, ars_shift = rnd_conf_val * hmat.std(axis=0), rnd_conf_val * hmats.std(axis=0)
        arr_mid, ars_mid = hmat.mean(axis=0), hmats.mean(axis=0)
        arr_low, arr_high = arr_mid - arr_shift, arr_mid + arr_shift
        ars_low, ars_high = ars_mid - ars_shift, ars_mid + ars_shift
    else:
        arr_low, arr_mid, arr_high = np.percentile(hmat, rnd_conf_val, axis=0), \
                                     np.percentile(hmat, 50, axis=0), \
                                     np.percentile(hmat, 100 - rnd_conf_val, axis=0)
        ars_low, ars_mid, ars_high = np.percentile(hmats, rnd_conf_val, axis=0), \
                                     np.percentile(hmats, 50, axis=0),\
                                     np.percentile(hmats, 100-rnd_conf_val, axis=0)
    plt.plot(ana_rg, arr_low, 'b--')
    plt.plot(ana_rg, arr_mid, 'b')
    plt.plot(ana_rg, arr_high, 'b--')
    plt.plot(ana_rg, ars_low, 'k--')
    plt.plot(ana_rg, ars_mid, 'k')
    plt.plot(ana_rg, ars_high, 'k--')
    if pt_xrange is not None:
        plt.xlim(pt_xrange)
    if pt_yrange is not None:
        plt.ylim(pt_yrange)
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_org_sim.png')
    plt.close()
else:
    print 'ERROR: organization could not be computed'
    print 'Unsuccessfully terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print 'Successfully terminated. (' + time.strftime("%c") + ')'
