"""

    From an input ListTomoFilaments generates tomogram density simulations

    Input:  - A STAR file with a set of ListTomoFilaments pickles (SetListFilaments object input) or a tomogram size
            - Settings for simulations

    Output: - The simulated tomograms
            - A STAR file with the list of simulated tomograms
"""

################# Package import

import os
import numpy as np
import scipy as sp
import sys
import time
import multiprocessing as mp
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg.globals import unpickle_obj, clean_dir, relion_norm
from pyorg.surf.model import ModelFilsRSR
from pyorg.surf.filaments import TomoFilaments, ListTomoFilaments, SetListTomoFilaments
try:
    import cPickle as pickle
except ImportError:
    import pickle

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-ruben/antonio/filaments'

# Input STAR files
in_star = None # ROOT_PATH + '/ltomos/fil_den_ps1.408/fil_den_ps1.408_ltomos.star'
in_tomo_sz = (2000, 2000, 1000) # Only used if in_star != None

# Output directory
out_dir = ROOT_PATH + '/sim_den/test_ps0.704'
out_stem = 'test_ns1'

# Filament settings
fl_den_2d = ROOT_PATH + '/models/emd_0148_2D_0.704nm.mrc' # '/models/emd_0148_2D_1.408nm.mrc' # '/models/emd_0148_2D_1.756nm.mrc'
fl_den_inv = True
fl_pitch = 117 # nm

# Tomogram settings
tm_snr_rg = None # (0.4, 0.5)
tm_mw = None # 0 # degs
tm_mwta = 60 # degs

# Simulation settings
sm_ns = 1 # 5 # 200
sm_max_fils = 400 # 100 # None

########################################################################################
# MAIN ROUTINE
########################################################################################

###### Additional functionality

########## Print initial message

print 'Simulates density tomograms from a reference list of filament networkds.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
if in_star is None:
    print '\tSimulating tomograms with size: ' + str(in_tomo_sz) + ' px'
else:
    print '\tInput STAR file for filaments: ' + str(in_star)
print '\tFilament settings: '
print '\t\t-Input 2D axial density file: ' + str(fl_den_2d)
if fl_den_inv:
    print '\t\t-Invert model density values.'
print '\t\t-Filament pitch: ' + str(fl_pitch) + ' nm'
print '\tTomogram settings: '
if tm_snr_rg is None:
    print '\t\t-No noise.'
else:
    print '\t\t-SNR range: ' + str(tm_snr_rg) + ' deg'
if tm_mw is None:
    print '\t\t-No missing wedge.'
else:
    print '\t\t-Angle for missing wedge: ' + str(tm_mw) + ' deg'
    print '\t\t-Maximum tilt angle in XY plane: ' + str(tm_mwta) + ' deg'
print '\tSimulation settings: '
if sm_ns <= 0:
    print 'ERROR: The number of input simulations must be greater than zero: ' + str(sm_ns)
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
print '\t\t-Number of simulations per input filament network: ' + str(sm_ns)
if sm_max_fils is None:
    print '\t\t-Maximum number of filament to instert per tomogram: ' + str(sm_max_fils)
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

if in_star is not None:
    print '\tLoading input data...'
    star = sub.Star()
    try:
        star.load(in_star)
    except pexceptions.PySegInputError as e:
        print 'ERROR: input STAR file for filaments could not be loaded because of "' + e.get_message() + '"'
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
    set_lists, lists_dic_rows = SetListTomoFilaments(), dict()
    for row in range(star.get_nrows()):
        ltomos_pkl = star.get_element('_psPickleFile', row)
        ltomos = unpickle_obj(ltomos_pkl)
        set_lists.add_list_tomos(ltomos, ltomos_pkl)
        fkey = os.path.split(ltomos_pkl)[1]
        short_key_idx = fkey.index('_')
        short_key = fkey[:short_key_idx]
        lists_dic_rows[short_key] = row
else:
    htomo = TomoFilaments('void_t1', lbl=1, voi=np.ones(shape=in_tomo_sz, dtype=np.bool))
    hltomo = ListTomoFilaments()
    hltomo.add_tomo(htomo)
    set_lists, lists_dic_rows = SetListTomoFilaments(), dict()
    set_lists.add_list_tomos(hltomo, '0_void')
    lists_dic_rows['0'] = 0

print '\tBuilding the dictionaries...'
lists_count, tomos_count = 0, 0
lists_dic = dict()
lists_hash, tomos_hash = dict(), dict()
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
        lists_count += 1
for lkey, llist in zip(set_lists_dic.iterkeys(), set_lists_dic.itervalues()):
    llist_tomos_dic = llist.get_tomos()
    for tkey, ltomo in zip(llist_tomos_dic.iterkeys(), llist_tomos_dic.itervalues()):
        if not(tkey in tomos_hash.keys()):
            tomos_hash[tkey] = tomos_count
            tomos_count += 1

print '\tLIST COMPUTING LOOP:'
for lkey in lists_hash.itervalues():

    llist = lists_dic[lkey]
    print '\t\t-Processing list: ' + lkey
    hold_row = lists_dic_rows[lkey]

    print '\t\t\t+Tomograms computing loop:'
    for tkey in tomos_hash.iterkeys():

        print '\t\t\t\t*Processing tomogram: ' + os.path.split(tkey)[1]
        try:
            ltomo = llist.get_tomo_by_key(tkey)
        except KeyError:
            print '\t\t\t\t\t-Tomogram with key ' + tkey + ' is not in the list ' + lkey + ' , continuing...'
            continue
        if (in_star is not None) and (ltomo.get_num_filaments() <= 0):
            print '\t\t\t\t\t-WARNING: no filaments to process, continuing...'
            continue
        else:
            print('\t\t\t\t\t-Number of filaments found: ' + str(ltomo.get_num_filaments()))

        print '\tPre-procesing the input 2D density model: '
        model_2D = disperse_io.load_tomo(fl_den_2d)
        # model_2D = relion_norm(model_2D, mask=None, inv=fl_den_inv)
        out_den = out_stem_dir + '/' + tkey.replace('/', '_') + '_fl_den_2d.mrc'
        print '\t\t-Model stored in: ' + out_den
        disperse_io.save_numpy(model_2D, out_den)

        print '\t\t\t\t\t-Computing filament to membrane nearest distances...'
        hold_arr_dsts = ltomo.compute_fils_seg_dsts()
        out_fils = out_stem_dir + '/' + tkey.replace('/', '_') + '_fils.vtp'
        disperse_io.save_vtp(ltomo.gen_filaments_vtp(), out_fils)

        print '\t\t\t\t\t-Simulating the density tomogrms:'
        model = ModelFilsRSR(ltomo.get_voi(), res=ltomo.get_resolution(), rad=ltomo.get_fils_radius(),
                             shifts=[0, 0], rots=[0, 0], density_2d=model_2D[:, :, 0])
        den_model = model.gen_fil_straight_density(2*fl_pitch, pitch=fl_pitch, rnd_iang=0)
        out_den = out_stem_dir + '/' + tkey.replace('/', '_') + '_fl_den.mrc'
        disperse_io.save_numpy(den_model, out_den)
        stack_model = model.gen_tomo_stack_densities(axis=0, pitch=fl_pitch, spacing=2, mwa=tm_mw, mwta=tm_mwta,
                                                    snr=None)
        out_stack = out_stem_dir + '/' + tkey.replace('/', '_') + '_fl_stack.mrc'
        disperse_io.save_numpy(stack_model, out_stack)
        del den_model
        del stack_model
        for i in range(sm_ns):
            if tm_snr_rg is None:
                snr = None
            else:
                snr = np.random.uniform(tm_snr_rg[0], tm_snr_rg[1])
            print '\t\t\t\t\t\t+SNR: ' + str(snr)
            if ltomo.get_num_filaments() > 0:
                fil_sim = model.gen_instance_straights('sim_' + str(i), ltomo.get_filaments(), mode='full', max_ntries=100,
                                                       max_fils=sm_max_fils)
            else:
                fil_sim = model.gen_instance_straights_random('sim_' + str(i), sm_max_fils, fil_samp=10, mode='full',
                                                              max_ntries=100)
            out_fil = out_stem_dir + '/' + tkey.replace('/', '_') + '_sim_' + str(i) + '_fil.vtp'
            disperse_io.save_vtp(fil_sim.gen_filaments_vtp(), out_fil)
            dim_den = model.gen_tomo_straight_densities(fil_sim.get_filaments(), pitch=fl_pitch, mwa=tm_mw,
                                                        mwta=tm_mwta, snr=snr)
            out_den = out_stem_dir + '/' + tkey.replace('/', '_') + '_sim_' + str(i) + '_den.mrc'
            disperse_io.save_numpy(dim_den, out_den)

print 'Successfully terminated. (' + time.strftime("%c") + ')'

