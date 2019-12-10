"""

    Compute averaged density in a neighbourhood (it can either uni or bivariate)

    Input:  - A STAR file with a ListTomoParticles pickle (SetListTomograms object input) used as reference
            - A STAR file with a set of ListTomoParticles pickles (SetListTomograms object input) to analyze
            - A neighbourhood

    Output: - Plots by tomograms:
                + Density in a neighbourhood by list
                + Density in a neighbourhood by tomogram

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

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils'

# Input STAR file
in_star_ref = ROOT_PATH + '/ves_ap_10/ltomos_ves_ap_10_premb_mask/ves_ap_10_premb_mask_ltomos.star'
in_ref_short_key = '0'
in_star = ROOT_PATH + '/pre/ref_nomb_1_clean/ltomos_pre_premb_mask/pre_premb_mask_ltomos.star' # None for univariate
in_wspace = None # ROOT_PATH + '/ves/densities/bi_ves_pre_ex_sph_20_wspace.pkl' # (Insert a path to recover a pickled workspace instead of doing a new computation)

# Output directory
out_dir = ROOT_PATH + '/ves_ap_10/densities'
out_stem = 'bi_ves_pre_sph_20' # ''uni_sph_4_60_5'

# Neighbourhood
nb_res = 0.684 # nm/voxel
nb_rad = 20 # np.arange(4, 100, 2) # in nm
nb_thick = None # 3
nb_conv_iter = 100
nb_max_iter = 100000

# Figure saving options
fig_fmt = '.png' # if None they showed instead

# Plotting options
pt_sim_v = True
pt_cmap = plt.get_cmap('gist_rainbow')
pt_nbins = 10

########################################################################################
# MAIN ROUTINE
########################################################################################

###### Additional functionality

# Units conversion
nb_rad_v = nb_rad / nb_res
nb_thick_v = None
if nb_thick is not None:
    nb_thick_v = float(nb_thick) / nb_res

########## Print initial message

print 'Computed averaged density in a neighbourhood.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
print '\tInput reference STAR file: ' + str(in_star_ref)
print '\tInput reference short key: ' + str(in_ref_short_key)
if in_star:
    print '\tBi-variate mode activated: '
    print '\t\t-Input analysis STAR file: ' + str(in_star)
else:
    print '\tUnivariate mode activated!'
if in_wspace is not None:
    print '\tLoad workspace from: ' + in_wspace
print '\tNeighbourhood settings: '
print '\t\t-Range of radius: ' + str(nb_rad) + ' nm'
print '\t\t-Range of radius: ' + str(nb_rad_v) + ' voxels'
if nb_rad is None:
    print '\t\t-Spherical neighborhood'
else:
    print '\t\t-Shell neighborhood with thickness: ' + str(nb_rad) + ' nm'
    print '\t\t-Shell neighborhood with thickness: ' + str(nb_rad_v) + ' voxels'
print '\t\t-Convergence iterations: ' + str(nb_conv_iter)
print '\t\t-Maximum iterations: ' + str(nb_max_iter)
if fig_fmt is not None:
    print '\tStoring figures:'
    print '\t\t-Format: ' + str(fig_fmt)
else:
    print '\tPlotting settings: '
print '\t\t-Colormap: ' + str(pt_cmap)
print '\t\t-Nuber of bins in histograms: ' + str(pt_nbins)
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

print '\tLoading input data...'
star_ref = sub.Star()
try:
    star_ref.load(in_star_ref)
except pexceptions.PySegInputError as e:
    print 'ERROR: input reference STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
if in_star is None:
    star = star_ref
else:
    star = sub.Star()
    try:
        star.load(in_star)
    except pexceptions.PySegInputError as e:
        print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
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
    print 'ERROR: reference ListTomoParticles with short key ' + in_ref_short_key + ' was not found!'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

if in_wspace is None:

    set_lists = surf.SetListTomoParticles()
    for row in range(star.get_nrows()):
        ltomos_pkl = star.get_element('_psPickleFile', row)
        ltomos = unpickle_obj(ltomos_pkl)
        set_lists.add_list_tomos(ltomos, ltomos_pkl)

    print '\tBuilding the dictionaries...'
    lists_count, tomos_count = 0, 0
    lists_dic = dict()
    lists_hash, tomos_hash = dict(), dict()
    tomos_den = dict()
    lists_den, lists_color = dict(), dict()
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
            lists_den[short_key] = None
            lists_count += 1
    for lkey, llist in zip(set_lists_dic.iterkeys(), set_lists_dic.itervalues()):
        llist_tomos_dic = llist.get_tomos()
        for tkey, ltomo in zip(llist_tomos_dic.iterkeys(), llist_tomos_dic.itervalues()):
            try:
                ref_tomo = ref_list.get_tomo_by_key(tkey)
            except KeyError:
                print '\t\t\t\t\t-WARNING: tomogram ' + tkey + ' discarded because it is not in reference list!'
                continue
            try:
                tomos_den[tkey]
            except KeyError:
                tomos_hash[tkey] = tomos_count
                tomos_den[tkey] = list()
                tomos_count += 1

    print '\tLIST COMPUTING LOOP:'
    for lkey in lists_hash.values():

        llist = lists_dic[lkey]
        print '\t\t-Processing list: ' + lkey
        print '\t\t\t+Tomograms computing loop:'
        for tkey in tomos_hash.keys():

            print '\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1]
            ltomo, ref_tomo = llist.get_tomo_by_key(tkey), ref_list.get_tomo_by_key(tkey)

            print '\t\t\t\t\t-Computing neighbourhood densities...'
            hold_dens = ref_tomo.compute_densities_nhood(nb_rad, nb_thick, border=True,
                                                         conv_iter=nb_conv_iter, max_iter=nb_max_iter,
                                                         bi_coords=ref_tomo.get_particle_coords())
            if hold_dens is not None:
                tomos_den[tkey].append(hold_dens)
                if lists_den[lkey] is None:
                    lists_den[lkey] = np.copy(hold_dens)
                else:
                    lists_den[lkey] = np.concatenate([lists_den[lkey], hold_dens])

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print '\tPickling computation workspace in: ' + out_wspace
    wspace = (lists_count, tomos_count,
              lists_hash, tomos_hash,
              tomos_den,
              lists_den, lists_color)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:

    print '\tLoading the workspace: ' + in_wspace
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    lists_count, tomos_count = wspace[0], wspace[1]
    lists_hash, tomos_hash = wspace[2], wspace[3]
    tomos_den = wspace[4]
    lists_den, lists_color = wspace[5], wspace[6]

print '\tPrinting lists hash: '
for id, lkey in zip(lists_hash.iterkeys(), lists_hash.itervalues()):
    print '\t\t-[' + str(id) + '] -> [' + lkey + ']'
print '\tPrinting tomograms hash: '
for tkey, val in zip(tomos_hash.iterkeys(), tomos_hash.itervalues()):
    print '\t\t-[' + tkey + '] -> [' + str(val) + ']'

print '\tPrinting number of samples and volume for the reference by tomogram: '
for tomo in ref_list.get_tomo_list():
    tkey = tomo.get_tomo_fname()
    print '\t\t-' + tkey + ': ' + str(tomo.get_num_particles()) + ' np, ' + str(tomo.compute_voi_volume()) + ' nm**3'

# Getting the lists colormap
n_lists = len(lists_hash.keys())
for i, lkey in zip(lists_hash.iterkeys(), lists_hash.itervalues()):
    lists_color[lkey] = pt_cmap(1.*i/n_lists)

print '\tTOMOGRAMS PLOTTING LOOP: '

out_tomos_dir = out_stem_dir + '/tomos'
os.makedirs(out_tomos_dir)

print '\t\t-Plotting densities by tomos...'
for tkey, ltomo in zip(tomos_den.iterkeys(), tomos_den.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.title('Density for ' + tkey_short)
    plt.ylabel('Averaged density (np/vol)')
    plt.xlabel('Classes')
    for i, dens in enumerate(ltomo):
        lkey = lists_hash[i]
        plt.bar(int(lkey), dens.mean(), width=0.75, color=lists_color[lkey], label=lkey)
    plt.xticks(BAR_WIDTH + np.arange(n_lists), np.arange(n_lists))
    plt.legend(loc=1)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/averaged.png')
    plt.close()

print '\t\t-Plotting density histograms by tomos...'
for tkey, ltomo in zip(tomos_den.iterkeys(), tomos_den.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.title('Density histogram for ' + tkey_short)
    plt.ylabel('Frequency (normed)')
    plt.xlabel('Density (np/vol)')
    for i, dens in enumerate(ltomo):
        lkey = lists_hash[i]
        plt.hist(dens, pt_nbins, color=lists_color[lkey], label=lkey, histtype='step')
    plt.legend(loc=1)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/histogram.png')
    plt.close()

print '\tLISTS PLOTTING LOOP: '

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

print '\t\t-Plotting averganed densities by lists...'
n_tomos = len(tomos_hash.keys())
plt.figure()
plt.title('Density lists')
plt.ylabel('Averaged density (np/vol)')
plt.xlabel('Classes')
for i, dens in enumerate(lists_den.values()):
    lkey = lists_hash[i]
    plt.bar(i, dens.mean(), width=0.75, color=lists_color[lkey])
plt.xticks(BAR_WIDTH + np.arange(n_lists), np.arange(n_lists))
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/lists_averaged.png')
plt.close()

print '\t\t-Plotting density histograms by lists...'
plt.figure()
plt.title('Density histogram for lists')
plt.ylabel('Frequency')
plt.xlabel('Density (np/vol)')
for lkey, dens in zip(lists_den.iterkeys(), lists_den.itervalues()):
    plt.hist(dens, pt_nbins, color=lists_color[lkey], label=lkey, histtype='step')
plt.legend(loc=1)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_lists_dir + '/lists_histogram.png')
plt.close()

print 'Successfully terminated. (' + time.strftime("%c") + ')'

