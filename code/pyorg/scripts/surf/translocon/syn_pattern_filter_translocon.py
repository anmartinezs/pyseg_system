"""

    Filter a ListTomoParticles by comparing their organization with already simulated models

    Input:  - A STAR file with a ListTomoParticles to filter
            - Path to a pickled set of simulations simulations compatible with the ListTomoParticles
            - Filtering settings

    Output: - The filtered ListTomoParticles

"""

################# Package import

import os
import copy
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
in_star = ROOT_PATH + '/stat/ltomos/try2_FT/try2_ssup_8_min_50_ltomos.star'

# Input Simulation matrix
in_sims_set = ROOT_PATH + '/stat/uni/try2_FT/sph_ssup_8_min_50/uni_5_200_3_sim_10_fig_lsims.pkl' # If None then they will be computed

# Output directory
out_dir = ROOT_PATH + '/stat/ltomos/try2_FT_flt/'
out_stem = 'try2_ssup_8_min_50_flt_high' # ''uni_sph_4_60_5'

# Filtering settings
ft_rg = [10, 45] # in nm
ft_th = 2 # 90
ft_th_mode = 'ripley' # 'pval'
ft_th_side = 'high' # 'low'
ft_min_nb = None # 4 #

# Computation settings
ana_res = 0.684 # nm/pixel
ana_border = True
ana_conv_iter = 20
ana_max_iter = 100000
ana_npr = 1

########################################################################################
# MAIN ROUTINE
########################################################################################

###### Additional functionality

########## Print initial message

print 'Filtering Tomogram particles by comparting against a simulated Ripley\'s model.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
print '\tInput STAR file the ListTomoParticles: ' + str(in_star)
print '\tInput for pickled simulations: ' + str(in_sims_set)
print '\tFiltering settings:'
print '\t\t-Multiscale range: ' + str(ft_rg) + ' nm'
if ft_th_mode == 'ripley':
    print '\t\t-Ripley\'s thresholding: ' + str(ft_th)
elif ft_th_mode == 'pval':
    print '\t\t-P-value thresholding: ' + str(ft_th)
else:
    print 'ERROR: invalid string for thresholding mode: ' + str(ft_th_mode)
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
if ft_th_side == 'low':
    print '\t\t-Threshold side: low'
elif ft_th_side == 'high':
    print '\t\t-Threshold side: high'
else:
    print 'ERROR: invalid string for thresholding side: ' + str(ft_th_side)
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
print '\t\t-Minimum number of neighbours: ' + str(ft_min_nb)
print '\tComputation settings: '
print '\t\t-Data resolution: ' + str(ana_res)
print '\t\t-Convergence number of samples for stochastic volume estimations: ' + str(ana_conv_iter)
print '\t\t-Maximum number of samples for stochastic volume estimations: ' + str(ana_max_iter)
if ana_npr is None:
    print '\t\t-Number of processors: Auto'
else:
    print '\t\t-Number of processors: ' + str(ana_npr)
if ana_border:
    print '\t\t-Border compensation activated!.'
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

print '\tLoading input data...'
star, star_out = sub.Star(), sub.Star()
try:
    star.load(in_star)
    star_out.add_column('_psPickleFile')
except pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
with open(in_sims_set, 'rb') as fl:
    set_sims = pickle.load(fl)
    fl.close()

set_lists = surf.SetListTomoParticles()
for row in range(star.get_nrows()):
    ltomos_pkl = star.get_element('_psPickleFile', row)
    ltomos = unpickle_obj(ltomos_pkl)
    set_lists.add_list_tomos(ltomos, ltomos_pkl)

print '\tBuilding the dictionaries...'
lists_count, tomos_count = 0, 0
lists_dic = dict()
lists_hash, tomos_hash = dict(), dict()
lists_fkey = dict()
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
        lists_fkey[short_key] = lkey
        lists_count += 1
for lkey, llist in zip(set_lists_dic.iterkeys(), set_lists_dic.itervalues()):
    llist_tomos_dic = llist.get_tomos()
    for tkey, ltomo in zip(llist_tomos_dic.iterkeys(), llist_tomos_dic.itervalues()):
        try:
            tomos_hash[tkey]
        except KeyError:
            tomos_hash[tkey] = tomos_count
            tomos_count += 1

print '\tLIST COMPUTING LOOP:'
hold_set = surf.SetListTomoParticles()
for lkey in lists_hash.values():

    llist, hold_list = lists_dic[lkey], surf.ListTomoParticles()
    try:
        sim_list_obj = set_sims.get_list_simulations(lkey)
    except KeyError:
        print '\t\t\t\tWARNING: no simulation found for list: ' + lkey
        continue
    print '\t\t-Processing list: ' + lkey
    print '\t\t\t+Tomograms computing loop:'
    for tkey in tomos_hash.keys():

        print '\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1]
        ltomo = llist.get_tomo_by_key(tkey)

        print '\t\t\t\t\t-Getting corresponding simulations...'
        try:
            sim_obj = copy.deepcopy(sim_list_obj.get_simulation(tkey))
        except KeyError:
            print '\t\t\t\tWARNING: no simulation found for tomogram: ' + tkey
            continue
        sim_obj.set_sub_range(ft_rg)
        sim_rg, sim_thick = sim_obj.get_rads(), sim_obj.get_thick()

        print '\t\t\t\t\t-Filtering...'
        rads_v = sim_obj.get_rads() / ana_res
        thick_v = None
        if sim_obj.get_thick() is not None:
            thick_v = sim_obj.get_thick()  / ana_res
        hold_ltomos = ltomo.filter_uni_2nd_order(sim_obj.get_rads()/ana_res, ft_th, ft_th_mode, ft_th_side, ft_min_nb, sim_obj,
                                                 thick=thick_v, border=ana_border,
                                                 conv_iter=ana_conv_iter, max_iter=ana_max_iter,
                                                 npr=ana_npr, verbose=False)
        hold_list.add_tomo(hold_ltomos)

    # star_stem = os.path.splitext(os.path.split(lists_fkey[lkey])[1])[0]
    star_stem = os.path.splitext(lists_fkey[lkey].replace('/', '_'))[0]
    out_pkl = out_dir + '/' + star_stem + '_tpl.pkl'
    print '\t\tPickling the list of tomograms in the file: ' + out_pkl
    try:
        hold_list.pickle(out_pkl)
        kwargs = {'_psPickleFile': out_pkl}
        star_out.add_row(**kwargs)
    except pexceptions.PySegInputError as e:
        print 'ERROR: list of tomograms container pickling failed because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)

    hold_set.add_list_tomos(hold_list, lists_fkey[lkey])

# out_parts = out_dir + '/' + out_stem + '_proc_parts.star'
# print '\tStoring the particles STAR file: ' + out_parts
# set_lists.to_particles_star().store(out_parts)

# print '\tStoring procesed set appended by tomograms in: ' + out_dir
# tomos_vtp = set_lists.tomos_to_vtp(mode='surface')
# for key, poly in zip(tomos_vtp.iterkeys(), tomos_vtp.itervalues()):
#     stem_tomo = os.path.splitext(os.path.split(key)[1])[0]
#     disperse_io.save_vtp(poly, out_dir+'/'+stem_tomo+'_proc_lists_app.vtp')

out_parts = out_dir + '/' + out_stem + '_flt_parts.star'
print '\tStoring the particles STAR file: ' + out_parts
hold_set.to_particles_star().store(out_parts)

print '\tStoring filtered set appended by tomograms in: ' + out_dir
tomos_vtp = hold_set.tomos_to_vtp(mode='surface')
for key, poly in zip(tomos_vtp.iterkeys(), tomos_vtp.itervalues()):
    stem_tomo = os.path.splitext(key.replace('/', '_'))[0]
    disperse_io.save_vtp(poly, out_dir+'/'+stem_tomo+'_flt_lists_app.vtp')

out_star = out_dir + '/' + out_stem + '_ltomos.star'
print '\tOutput STAR file: ' + out_star
star_out.store(out_star)

print 'Successfully terminated. (' + time.strftime("%c") + ')'

