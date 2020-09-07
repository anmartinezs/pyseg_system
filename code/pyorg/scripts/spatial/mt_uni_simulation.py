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
from pyorg.surf.model import ModelCSRV
from pyorg.globals import unpickle_obj, sort_dict
import matplotlib.pyplot as plt
from pyorg.surf import stat_dict_to_mat

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-plitzko/Saikat/luminal_density_analysis'

# Input STAR file with microtubules TomoParticles picles
in_star = ROOT_PATH + '/graphs/mts_clines_mts_ltomos.star' # '/ref_a3/ltomos/pre_ltomos.star'

# Particle surface
in_vtp = ROOT_PATH + '/vtps/sph_rad_2.5_surf.vtp'

# Computation shortcut pickle files
in_mat_tomos = None # ROOT_PATH + '/ref_a2/uni_sim/uni_sph_4_60_3_org_tomos.pkl'
in_mat_sims = None # ROOT_PATH + '/ref_a2/uni_sim/uni_sph_4_60_3_org_sims.pkl'

# Output directory
out_dir = '/fs/home/martinez/Downloads' # ROOT_PATH + '/uni_sim'
out_stem = 'uni_sph_4_60_3'

# Analysis variables
ana_res = 1.368 # nm/voxel
ana_rg = np.arange(2.5, 60, 1) # np.arange(4, 100, 4)
ana_shell_thick = 2 # None
ana_border = True
ana_conv_iter = 10
ana_max_iter = 100000
ana_npr = 10 # None means Auto

# Simulation model (currently only CSRV)
rnd_n = 1
rnd_conf_mean = False # True, mean centrality (Gaussian distribution), False median (Generic distribution)
rnd_conf_val = 2.5 # if mean then it is the number of sigmas, otherwise percentile in %

# Figure saving options
fig_fmt = None # '.png' # if None they showed instead

# Plotting options
pt_xrange = None # [10, 25]
pt_yrange = None # [0, 10]
pt_cmap = plt.get_cmap('gist_rainbow')

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

ana_rg = ana_rg / ana_res
print('Univariate second order analysis for a ListTomoParticles.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput directory: ' + str(out_dir))
print('\tOuput stem: ' + str(out_stem))
print('\tInput Pickle file: ' + str(in_star))
print('\tParticle referece surface file: ' + str(in_vtp))
print('\tOrganization analysis settings: ')
if in_mat_tomos is None:
    print('\t\t-Range of radius: ' + str(ana_rg) + ' voxels')
    if ana_shell_thick is None:
        print('\t\t-Spherical neighborhood')
    else:
        print('\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick))
    print('\t\t-Convergence number of samples for stochastic volume estimations: ' + str(ana_conv_iter))
    print('\t\t-Maximum number of samples for stochastic volume estimations: ' + str(ana_max_iter))
    if ana_npr is None:
        print('\t\t-Number of processors: Auto')
    else:
        print('\t\t-Number of processors: ' + str(ana_npr))
else:
    print('\tDensity ratio by tomograms dictionary pickled from file: ' + in_mat_tomos)
print('\tRandom model settings (CSRV):')
if in_mat_sims is None:
    print('\t\t-Number of instances: ' + str(rnd_n))
else:
    print('\tSimulation instances for density ratio pickled from file: ' + in_mat_sims)
if rnd_conf_mean:
    print('\t\t-N sigmas for Gaussian confidence interval: ' + str(rnd_conf_val))
else:
    print('\t\t-Percentile for the generic confidence interval: ' + str(rnd_conf_val) + ' %')
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
mat_tomos, mat_sims = None, None
den_cte = 1e6

print('\tLoading input STAR file...')
star = sub.Star()
try:
    star.load(in_star)
except pexceptions.PySegInputError as e:
    print('ERROR: input Pickle file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('\tCreating the list of tomograms...')
tomos_list = surf.ListTomoParticles()
for row in range(star.get_nrows()):
    hlist = unpickle_obj(star.get_element('_psPickleFile', row))
    for tomo in hlist.get_tomo_list():
        tomos_list.add_tomo(tomo)

print('\tComputing number of particles by tomogram...')
np_tomos = tomos_list.particles_by_tomos()
np_tomos_skeys, np_tomos_svalues = sort_dict(np_tomos, np_tomos, reverse=True)
color_tomos, tomo_lbls = dict(), dict()
for i, key in enumerate(np_tomos_skeys):
    tomo_lbl = os.path.split(key)[1]
    try:
        t_idx = tomo_lbl.index('_tpl')
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
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_np_tomos.png')
plt.close()
with open(out_dir + '/' + out_stem + '_np_tomos.pkl', "wb") as fl:
    pickle.dump(np_tomos, fl)
    fl.close()

print('\tComputing densities by tomogram...')
gl_tomos = tomos_list.densities_by_tomos()
gl_tomos_skeys, gl_tomos_svalues = sort_dict(gl_tomos, gl_tomos, reverse=True)
color_tomos, tomo_lbls = dict(), dict()
for i, key in enumerate(gl_tomos_skeys):
    tomo_lbl = os.path.split(key)[1]
    try:
        t_idx = tomo_lbl.index('_tpl')
        tomo_lbl = tomo_lbl[:t_idx]
    except ValueError:
        pass
    color_tomos[key] = pt_cmap(1.*i/len(gl_tomos))
    tomo_lbls[key] = tomo_lbl
    print('\t\t-Tomogram ' + str(i+1) + ': ' + str(tomo_lbl))
plt.figure()
plt.title('Density by tomograms')
plt.ylabel('Density (x' + str(den_cte) + ')')
plt.xlabel('Tomograms')
it, bars, lbls = 0, list(), list()
for key, vals in zip(gl_tomos_skeys, gl_tomos_svalues):
    lbl = tomo_lbls[key]
    bar, = plt.bar(it, den_cte*np.asarray(vals, dtype=float), width=0.75, color=color_tomos[key], label=lbl)
    it += 1
    bars.append(bar)
    lbls.append(lbl)
plt.legend(bars, lbls, loc=1)
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_den_tomos.png')
plt.close()
with open(out_dir + '/' + out_stem + '_den_tomos.pkl', "wb") as fl:
    pickle.dump(gl_tomos, fl)
    fl.close()

if in_mat_tomos is None:
    print('\tComputing organization by list...')
    mat_tomos = tomos_list.compute_uni_2nd_order_by_tomos(distances=ana_rg, thick=ana_shell_thick, border=ana_border,
                                                          conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                          npr=ana_npr, verbose=True)
    with open(out_dir + '/' + out_stem + '_org_tomos.pkl', "wb") as fl:
        pickle.dump(mat_tomos, fl)
        fl.close()

if in_mat_sims is None:
    in_model = ModelCSRV
    out_model = out_dir + '/' + out_stem + '_model_tomo.pkl'
    print('\tPickling an instance of the mode in:' + out_model)
    try:
        part_vtp = disperse_io.load_poly(in_vtp)
    except pexceptions.PySegInputError as e:
        print('ERROR: reference particle surface file could not be loaded because of "' + e.get_message() + '"')
        print('Terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
    hold_tomo = tomos_list.get_tomo_by_key(gl_tomos_skeys[0])
    ex_model = in_model(hold_tomo.get_voi(), part_vtp)
    model_tomo = ex_model.gen_instance(hold_tomo.get_num_particles(), 'example_model', mode='center')
    model_tomo.pickle(out_model)

    print('\tComputing simulations with model: ' + str(type(in_model)))
    mat_sims = tomos_list.simulate_uni_2nd_order_by_tomos(n_sims=rnd_n, temp_model=in_model, part_vtp=part_vtp,
                                                          distances=ana_rg, thick=ana_shell_thick, border=ana_border,
                                                          conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                          npr=ana_npr, verbose=True)
    with open(out_dir + '/' + out_stem + '_org_sims.pkl', "wb") as fl:
        pickle.dump(mat_sims, fl)
        fl.close()

print('\tPickling organization by lists...')
if in_mat_tomos is not None:
    with open(in_mat_tomos, 'r') as pkl:
        mat_tomos = pickle.load(pkl)
print('\tPickling organization simulations...')
if in_mat_sims is not None:
    with open(in_mat_sims, 'r') as pkl:
        mat_sims = pickle.load(pkl)

if (mat_tomos is not None) and (mat_sims is not None):
    gl_den = tomos_list.compute_global_density()
    if gl_den <= 0:
        print('ERROR: global density for the list is lower or equal to zero so no further statistics can be displayed!')
        print('Unsuccesfully terminated. (' + time.strftime("%c") + ')')
        sys.exit(-1)
    plt.figure()
    plt.title('Univariate 2nd Order')
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Radius')
    # Metrics computation
    hmat = stat_dict_to_mat(mat_tomos, tomos_list)
    hmats = stat_dict_to_mat(mat_sims, tomos_list)
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
    print('ERROR: organization could not be computed')
    print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('Successfully terminated. (' + time.strftime("%c") + ')')
