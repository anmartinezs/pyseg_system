"""

    Performs Bivariate 2nd order analysis between ListTomoParticles pairs

    Input:  - Two STAR file with a set of ListTomoParticles pickles

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
from pyorg.surf.utils import list_tomoparticles_pvalues
from pyorg.surf.model import ModelCSRV
from pyorg.globals import unpickle_obj
import matplotlib.pyplot as plt
from pyorg.surf import stat_dict_to_mat

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion'

# Input STAR file
in_star_1 = ROOT_PATH + '/fils/pre/ref_nomb_1_clean/ltomos_pre_premb_mask/pre_premb_mask_ltomos.star'
in_star_2 = ROOT_PATH + '/fils/pst/nrt/ltomos_k4_premb_mask/k4_premb_mask_ltomos.star'

# Input matrices (optional - organization analysis is skipped)
in_mats_lists = ROOT_PATH + '/fils/pst/nrt/bi_pre_k4/bi_15_no_pwise_keep_5_mask_sim_20_org_sets.pkl'

# Output directory
out_dir = ROOT_PATH + '/fils/pst/nrt/bi_pre_k4'
out_stem = 'bi_15_no_pwise_keep_5_mask_sim_20' # 'bi_4_8_2_no_pwise_keep_5' # 'test' #

# Analysis variables
ana_res = 0.684 # nm/voxel
ana_rg = np.asarray((10., 15))
ana_shell_thick = None # None
ana_border = True
ana_keep = 5
ana_conv_iter = 20
ana_max_iter = 100000
ana_gl = False
ana_npr = 1 # None means Auto
ana_npr_sim = 1 # None means Auto

# P-value computation settings
# Simulation model (currently only CSRV)
do_p_value = True
p_nsims = 100
p_per = 5 # %
p_pwise = True
# Particle surface
p_vtp = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pre/vtps/sph_rad_0.5_surf.vtp'
in_mats_sims = ROOT_PATH + '/fils/pst/nrt/bi_pre_k4/bi_15_no_pwise_keep_5_mask_sim_20_org_sims.pkl'

# Figure saving options
fig_fmt = None # '.png' # if None they showed instead

# Plotting options
pt_no_eq = True
pt_co_vmin, pt_co_vmax = 0.02, 0.1
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
print '\tInput STAR file 1: ' + str(in_star_1)
print '\tInput STAR file 2: ' + str(in_star_2)
if in_mats_lists is None:
    print '\tOrganization analysis settings: '
    print '\t\t-Range of radius: ' + str(ana_rg) + ' nm'
    print '\t\t-Range of radius: ' + str(ana_rg_v) + ' voxels'
    if ana_shell_thick is None:
        print '\t\t-Spherical neighborhood'
    else:
        print '\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick) + ' nm'
        print '\t\t-Shell neighborhood with thickness: ' + str(ana_shell_thick_v) + ' voxels'
    print '\t\t-Convergence number of samples for stochastic volume estimations: ' + str(ana_conv_iter)
    print '\t\t-Maximum number of samples for stochastic volume estimations: ' + str(ana_max_iter)
    if ana_keep is not None:
        print '\t\t-Keep highest populated tomograms per list 1: ' + str(ana_keep)
    if ana_gl:
        print '\t\t-Global analysis activated.'
    if ana_npr is None:
        print '\t\t-Number of processors: Auto'
    else:
        print '\t\t-Number of processors: ' + str(ana_npr)
        print '\t\t-Number of processors for simulations: ' + str(ana_npr_sim)
else:
    if in_mats_lists is not None:
        print '\tOrganization by list pickled from file: ' + in_mats_lists
if do_p_value:
    print '\tP-Value computation setting:'
    print '\t\t-Percentile: ' + str(p_per) + ' %'
    if in_mats_sims is None:
        print '\t\t-Number of instances for simulations: ' + str(p_nsims)
        print '\t\t-Particle surface: ' + p_vtp
    else:
        print '\t\t-Using pre-computed matrix for simulations: ' + in_mats_sims
    if p_pwise:
        print '\t\t-Pointwise analysis activated.'
    else:
        print '\t\t-Tomogram analysis activated.'
if fig_fmt is not None:
    print '\tStoring figures:'
    print '\t\t-Format: ' + str(fig_fmt)
else:
    print '\tPlotting settings: '
print '\t\t-Colocalizations matrix high p-values reange: [' + str(pt_co_vmin) + ', ' + str(pt_co_vmax) + ']'
print '\t\t-Colormap: ' + str(pt_cmap)
if pt_no_eq:
    print '\t\t-List of tomograms with equal name are not compared each other.'
print ''

######### Process

print 'Main Routine: '
mats_lists = None

print '\tLoading input STAR files...'
star_1, star_2 = sub.Star(), sub.Star()
try:
    star_1.load(in_star_1)
    star_2.load(in_star_2)
except pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
set_lists_1, set_lists_2 = surf.SetListTomoParticles(), surf.SetListTomoParticles()
for row in range(star_1.get_nrows()):
    ltomos_pkl = star_1.get_element('_psPickleFile', row)
    ltomos = unpickle_obj(ltomos_pkl)
    set_lists_1.add_list_tomos(ltomos, ltomos_pkl)
for row in range(star_2.get_nrows()):
    ltomos_pkl = star_2.get_element('_psPickleFile', row)
    ltomos = unpickle_obj(ltomos_pkl)
    set_lists_2.add_list_tomos(ltomos, ltomos_pkl)

print '\tComputing number of particles by list...'
np_lists_1, np_lists_2 = set_lists_1.particles_by_list(), set_lists_2.particles_by_list()
color_dict = dict()
for i, key in enumerate(np_lists_2.keys()):
    color_dict[key] = pt_cmap(1.*i/len(np_lists_2))
    print '\t\t-List ' + str(i+1) + ': ' + str(key)
for key_1, val in zip(np_lists_1.iterkeys(), np_lists_1.itervalues()):
    l_name = os.path.splitext(os.path.split(key_1)[1])[0]
    plt.figure()
    plt.title('Num. particles for list ' + l_name)
    plt.ylabel('Num. particles')
    plt.xlabel('List')
    x_bars = np.arange(len(np_lists_2.keys()))+1
    for x, y, key_2 in zip(x_bars, np_lists_2.itervalues(), np_lists_2.iterkeys()):
        plt.bar(x-.2, val, width=.3, color='k')
        plt.bar(x+.2, y, width=.3, color=color_dict[key_2])
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_' + l_name + '_np_lists.png')
    plt.close()
    with open(out_dir + '/' + out_stem + '_' + l_name + '_np_lists_1.pkl', "wb") as fl:
        pickle.dump(np_lists_1, fl)
        fl.close()
    with open(out_dir + '/' + out_stem + '_' + l_name + '_np_lists_2.pkl', "wb") as fl:
        pickle.dump(np_lists_2, fl)
        fl.close()

if in_mats_lists is None:
    print '\tComputing organization by list...'
    mats_lists = set_lists_1.compute_bi_2nd_order_by_list(set_lists_2, distances=ana_rg_v,
                                                          thick=ana_shell_thick_v, border=ana_border,
                                                          conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                          no_eq_name=pt_no_eq, dens_gl=ana_gl, keep=ana_keep,
                                                          npr=ana_npr, verbose=True)
    with open(out_dir + '/' + out_stem + '_org_sets.pkl', "wb") as fl:
        pickle.dump(mats_lists, fl)
        fl.close()
else:
    print '\tPickling organization by lists...'
    if in_mats_lists is not None:
        with open(in_mats_lists, 'r') as pkl:
            mats_lists = pickle.load(pkl)
for key_1, mats in zip(mats_lists.iterkeys(), mats_lists.itervalues()):
    l_name = os.path.splitext(os.path.split(key_1)[1])[0]
    plt.figure()
    plt.title('Bivariate 2nd for list ' + l_name)
    if ana_shell_thick is None:
        plt.ylabel('Ripley\'s L')
    else:
        plt.ylabel('Ripley\'s O')
    plt.xlabel('Radius')
    for key_2, mat in zip(mats.iterkeys(), mats.itervalues()):
        plt.plot(ana_rg, mat, color=color_dict[key_2])
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_' + l_name + '_org_lists.png')
    plt.close()

if do_p_value:
    if in_mats_sims is None:
        in_model = ModelCSRV
        try:
            part_vtp = disperse_io.load_poly(p_vtp)
        except pexceptions.PySegInputError as e:
            print 'ERROR: reference particle surface file could not be loaded because of "' + e.get_message() + '"'
            print 'Terminated. (' + time.strftime("%c") + ')'
            sys.exit(-1)

        print '\tComputing bivariate simulations matrix...'
        mats_sims = set_lists_1.simulate_bi_2nd_order_matrix_plist(set_lists_2, p_nsims, in_model, part_vtp,
                                                             distances=ana_rg_v,
                                                             thick=ana_shell_thick_v, border=ana_border,
                                                             conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                             dens_gl=ana_gl, pointwise=p_pwise,
                                                             no_eq_name=True, keep=ana_keep,
                                                             npr=ana_npr_sim, verbose=True, tmp_dir=out_dir)
        with open(out_dir + '/' + out_stem + '_org_sims.pkl', "wb") as fl:
            pickle.dump(mats_sims, fl)
            fl.close()
    else:
        print '\tPickling simulations by lists...'
        if in_mats_sims is not None:
            with open(in_mats_sims, 'r') as pkl:
                mats_sims = pickle.load(pkl)

    print '\tComputing p-values matrix...'
    c_pvals = dict()
    for key_1, mats in zip(mats_lists.iterkeys(), mats_lists.itervalues()):
        c_pvals[key_1] = list_tomoparticles_pvalues(ana_rg, mats, mats_sims[key_1])

if do_p_value:
    print '\tComputing co-localizations matrices (p-values)...'
    nlists_1, nlists_2 = len(set_lists_1.get_lists()), len(set_lists_2.get_lists())
    count_1, count_2 = 0, 0
    lbls_1, lbls_2 = ['' for i in range(nlists_1)], ['' for i in range(nlists_2)]
    co_mat_l = np.zeros(shape=(nlists_1, nlists_2), dtype=float)
    co_mat_h = np.zeros(shape=(nlists_1, nlists_2), dtype=float)
    dsts_mat_l = np.zeros(shape=(nlists_1, nlists_2), dtype=float)
    dsts_mat_h = np.zeros(shape=(nlists_1, nlists_2), dtype=float)
    for key_1, d_pvals in zip(c_pvals.iterkeys(), c_pvals.itervalues()):
        l_name_1 = os.path.splitext(os.path.split(key_1)[1])[0]
        try:
            k_idx = l_name_1.index('_')
            l_name_1 = l_name_1[:k_idx]
        except IndexError:
            pass
        for key_2, pvals in zip(d_pvals.iterkeys(), d_pvals.itervalues()):
            l_name_2 = os.path.splitext(os.path.split(key_2)[1])[0]
            try:
                k_idx = l_name_2.index('_')
                l_name_2 = l_name_2[:k_idx]
            except IndexError:
                pass
            try:
                idx_1 = lbls_1.index(l_name_1)
            except ValueError:
                idx_1 = count_1
                lbls_1[count_1] = l_name_1
                count_1 += 1
            try:
                idx_2 = lbls_2.index(l_name_2)
            except ValueError:
                idx_2 = count_2
                lbls_2[count_2] = l_name_2
                count_2 += 1
            if key_1 != key_2:
                ord_by_den = False
                if ord_by_den:
                    co_mat_l[idx_1, idx_2] = -0.01*pvals[2]
                    co_mat_h[idx_1, idx_2] = 0.01*pvals[3]
                    dsts_mat_l[idx_1, idx_2] = pvals[0]
                    dsts_mat_h[idx_1, idx_2] = pvals[1]
                else:
                    co_mat_l[int(l_name_1), int(l_name_2)] = -0.01 * pvals[2]
                    co_mat_h[int(l_name_1), int(l_name_2)] = 0.01 * pvals[3]
                    dsts_mat_l[int(l_name_1), int(l_name_2)] = pvals[0]
                    dsts_mat_h[int(l_name_1), int(l_name_2)] = pvals[1]
    print '\t\t-Co-localization indices:'
    print '\t\t\t-Y-axis (list 1): ' + str(zip(range(len(lbls_1)), lbls_1))
    print '\t\t\t-X-axis (list 2): ' + str(zip(range(len(lbls_2)), lbls_2))

    print '\t\tPlotting co-localization matrices (p-value)...'
    plt.figure()
    plt.title('Co-localization matrix (p-value low)')
    if nlists_1 == 1:
        bars = list()
        for i, key_2, d_pvals in zip(np.arange(nlists_2), c_pvals.values()[0].iterkeys(), c_pvals.values()[0].itervalues()):
            bar, = plt.bar(i+.35, -0.01*d_pvals[2], width=.3, color=color_dict[key_2])
            bars.append(bar)
        plt.xticks(.5+np.arange(nlists_2), lbls_2)
        plt.legend(bars, lbls_2, loc=1)
        plt.tight_layout()
    elif nlists_2 == 1:
        for key_1, d_pvals in zip(c_pvals.values().iterkeys(), c_pvals.values().itervalues()):
            plt.bar(np.arange(nlists_1)+.2, -0.01*d_pvals.values()[0][2], width=.3, color=color_dict[key_1])
    else:
        plt.pcolor(co_mat_l)
        plt.colorbar()
        plt.xticks(np.arange(0.5, nlists_2 + .5), lbls_2)
        plt.yticks(np.arange(0.5, nlists_1 + .5), lbls_1)
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_co_mat_low.png')
        plt.close()
    with open(out_dir + '/' + out_stem + '_co_mat_low.pkl', "wb") as fl:
        pickle.dump(co_mat_l, fl)
        fl.close()
    plt.figure()
    plt.title('Co-localization distances matrix (p-value low)')
    if nlists_1 == 1:
        bars = list()
        for i, key_2, d_pvals in zip(np.arange(nlists_2), c_pvals.values()[0].iterkeys(),
                                     c_pvals.values()[0].itervalues()):
            bar, = plt.bar(i+.35, d_pvals[0], width=.3, color=color_dict[key_2])
            bars.append(bar)
        plt.xticks(.5+np.arange(nlists_2), lbls_2)
        plt.legend(bars, lbls_2, loc=1)
        plt.tight_layout()
    elif nlists_2 == 1:
        for key_1, d_pvals in zip(c_pvals.values().iterkeys(), c_pvals.values().itervalues()):
            plt.bar(np.arange(nlists_1)+.2, d_pvals.values()[0][0], width=.3, color=color_dict[key_1])
    else:
        plt.pcolor(dsts_mat_l)
        plt.colorbar()
        plt.xticks(np.arange(0.5, nlists_2 + .5), lbls_2)
        plt.yticks(np.arange(0.5, nlists_1 + .5), lbls_1)
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_co_dst_mat_low.png')
        plt.close()
    with open(out_dir + '/' + out_stem + '_co_dst_mat_low.pkl', "wb") as fl:
        pickle.dump(dsts_mat_l, fl)
        fl.close()

    print '\t\tPlotting co-localization matrices (p-value)...'
    plt.figure()
    plt.title('Co-localization matrix (p-value)')
    if nlists_1 == 1:
        for key_2, d_pvals in zip(c_pvals.values()[0].iterkeys(), c_pvals.values()[0].itervalues()):
            key_2_stem = os.path.split(key_2)[1]
            id_str = key_2_stem[:key_2_stem.index('_')]
            idx = int(id_str)
            plt.bar(idx + .35, 1-0.01*d_pvals[3], width=.3)
        plt.xticks(.5 + np.arange(nlists_2), np.arange(nlists_2))
        plt.xlim(left=0, right=nlists_2)
        plt.ylim(pt_co_vmin, pt_co_vmax)
        # plt.legend(bars, lbls_2, loc=1)
        # plt.tight_layout()
    elif nlists_2 == 1:
        for key_1, d_pvals in zip(c_pvals.values().iterkeys(), c_pvals.values().itervalues()):
            plt.bar(np.arange(nlists_1)+.2, 0.01*d_pvals.values()[0][4], width=.3, color=color_dict[key_1])
    else:
        plt.pcolor(1.-co_mat_h, vmin=pt_co_vmin, vmax=pt_co_vmax, cmap=plt.get_cmap('RdBu'))
        plt.colorbar()
        plt.xlabel('Post-synaptic extracellular')
        plt.ylabel('Pre-synaptic extracellular')
        plt.xlim(0, nlists_2)
        plt.ylim(0, nlists_1)
        plt.xticks(np.arange(0.5, nlists_2 + .5), np.arange(nlists_2))
        plt.yticks(np.arange(0.5, nlists_1 + .5), np.arange(nlists_1))
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_co_mat_high.png')
        plt.close()
    with open(out_dir + '/' + out_stem + '_co_mat_low.high', "wb") as fl:
        pickle.dump(co_mat_h, fl)
        fl.close()
    plt.figure()
    plt.title('Co-localization distances matrix (p-value high)')
    if nlists_1 == 1:
        bars = list()
        for key_2, d_pvals in zip(c_pvals.values()[0].iterkeys(), c_pvals.values()[0].itervalues()):
            key_2_stem = os.path.split(key_2)[1]
            id_str = key_2_stem[:key_2_stem.index('_')]
            idx = int(id_str)
            bar, = plt.bar(idx+.35, d_pvals[1], width=.3, color=color_dict[key_2])
            bars.append(bar)
        plt.xticks(.5+np.arange(nlists_2), np.arange(nlists_2))
        # plt.legend(bars, lbls_2, loc=1)
        plt.tight_layout()
    elif nlists_2 == 1:
        for key_1, d_pvals in zip(c_pvals.values().iterkeys(), c_pvals.values().itervalues()):
            plt.bar(np.arange(nlists_1)+.2, d_pvals.values()[0][3], width=.3, color=color_dict[key_1])
    else:
        plt.pcolor(dsts_mat_h)
        plt.colorbar()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        plt.savefig(out_dir + '/' + out_stem + '_co_dst_mat_high.png')
        plt.close()
    with open(out_dir + '/' + out_stem + '_co_dst_mat_high.pkl', "wb") as fl:
        pickle.dump(dsts_mat_h, fl)
        fl.close()


print '\tComputing co-localizations matrix...'
nlists_1, nlists_2 = len(set_lists_1.get_lists()), len(set_lists_2.get_lists())
count_1, count_2 = 0, 0
lbls_1, lbls_2 = ['' for i in range(nlists_1)], ['' for i in range(nlists_2)]
co_mat = np.zeros(shape=(nlists_1, nlists_2), dtype=float)
dsts_mat = np.zeros(shape=(nlists_1, nlists_2), dtype=float)
for key_1, mats in zip(mats_lists.iterkeys(), mats_lists.itervalues()):
    l_name_1 = os.path.splitext(os.path.split(key_1)[1])[0]
    try:
        k_idx = l_name_1.index('_')
        l_name_1 = l_name_1[:k_idx]
    except IndexError:
        pass
    for key_2, mat in zip(mats.iterkeys(), mats.itervalues()):
        l_name_2 = os.path.splitext(os.path.split(key_2)[1])[0]
        try:
            k_idx = l_name_2.index('_')
            l_name_2 = l_name_2[:k_idx]
        except IndexError:
            pass
        mat[np.isnan(mat)] = 0
        mx_id = mat.argmax()
        try:
            idx_1 = lbls_1.index(l_name_1)
        except ValueError:
            idx_1 = count_1
            lbls_1[count_1] = l_name_1
            count_1 += 1
        try:
            idx_2 = lbls_2.index(l_name_2)
        except ValueError:
            idx_2 = count_2
            lbls_2[count_2] = l_name_2
            count_2 += 1
        if key_1 != key_2:
            co_mat[idx_1, idx_2] = mat[mx_id]
            dsts_mat[idx_1, idx_2] = ana_rg[mx_id]
print '\t\t-Co-localization indices:'
print '\t\t\t-Y-axis (list 1): ' + str(zip(range(len(lbls_1)), lbls_1))
print '\t\t\t-X-axis (list 2): ' + str(zip(range(len(lbls_2)), lbls_2))

print '\t\tPlotting co-localization matrix...'
plt.figure()
plt.title('Co-localization matrix')
plt.pcolor(co_mat)
plt.colorbar()
# plt.xticks(np.arange(0.5, nlists_2+.5), range(0, nlists_2))
# plt.yticks(np.arange(0.5, nlists_1+.5), range(0, nlists_1))
plt.xticks(np.arange(0.5, nlists_2+.5), lbls_2)
plt.yticks(np.arange(0.5, nlists_1+.5), lbls_1)
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_co_mat.png')
    plt.close()
with open(out_dir + '/' + out_stem + '_co_mat.pkl', "wb") as fl:
    pickle.dump(co_mat, fl)
    fl.close()

print '\t\tPlotting co-localization distances matrix...'
plt.figure()
plt.title('Co-localization distances matrix')
plt.pcolor(dsts_mat)
plt.colorbar()
# plt.xticks(np.arange(0.5, nlists_2+.5), range(0, nlists_2))
# plt.yticks(np.arange(0.5, nlists_1+.5), range(0, nlists_1))
plt.xticks(np.arange(0.5, nlists_2+.5), lbls_2)
plt.yticks(np.arange(0.5, nlists_1+.5), lbls_1)
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_dir + '/' + out_stem + '_co_dst_mat.png')
    plt.close()
with open(out_dir + '/' + out_stem + '_co_dst_mat.pkl', "wb") as fl:
    pickle.dump(dsts_mat, fl)
    fl.close()

print 'Terminated. (' + time.strftime("%c") + ')'