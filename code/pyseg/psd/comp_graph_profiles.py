"""

    Script for comparing statistics from different profiles
    Only slices which are compatible can be compared

    Input:  - Path to the input profiles for comparing

    Output: - Plot graphs with the statistical comprison
            - Store a FuncComparer object in pickle file

"""

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################


ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1/zd'

# Input pickles
profiles_pkl = (ROOT_PATH + '/pst/graph_stat/syn_14_7_bin2_sirt_rot_crop2_nrec.pkl',
                ROOT_PATH + '/pst/graph_stat/syn_14_9_bin2_sirt_rot_crop2_nrec.pkl',
                ROOT_PATH + '/pst/graph_stat/syn_14_13_bin2_sirt_rot_crop2_nrec.pkl',
                ROOT_PATH + '/pst/graph_stat/syn_14_14_bin2_sirt_rot_crop2_nrec.pkl',
                ROOT_PATH + '/pst/graph_stat/syn_14_15_bin2_sirt_rot_crop2_nrec.pkl',
                ROOT_PATH + '/pre/graph_stat/syn_14_9_bin2_sirt_rot_crop2_nrec.pkl',
                ROOT_PATH + '/pre/graph_stat/syn_14_13_bin2_sirt_rot_crop2_nrec.pkl',
                ROOT_PATH + '/pre/graph_stat/syn_14_14_bin2_sirt_rot_crop2_nrec.pkl',
                ROOT_PATH + '/pre/graph_stat/syn_14_15_bin2_sirt_rot_crop2_nrec.pkl',
                )

####### Output data

output_dir = ROOT_PATH + '/pst_vs_pre/1/comp_stat'
prefix_name = 'pst_vs_pre_att_2' # Stem for stored files

####### Display options

disp_plots = True
store_plots = True
leg_loc = 1
leg_num = True

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import sys
import pyseg as ps
from pyseg.spatial import FuncComparator
import numpy as np

########## Global variables

########## Print initial message

print 'Statistics comparison from different profiles.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput file(s): ' + str(profiles_pkl)
print '\tOutput directory: ' + str(output_dir)
print ''

######### Process

print 'Comparators initialization...'
profile = ps.factory.unpickle_obj(profiles_pkl[0])
if not isinstance(profile, ps.spatial.SetCloudsP):
    pairs = profile
    profile = None
if profile is None:
    slices = pairs.get_slice_ranges()
    comps_rhr = list()
    comps_rhp = list()
    comps_g = list()
    names = list()
    for sl in slices:
        comps_rhr.append(FuncComparator(sl + ' H cross'))
        comps_rhp.append(FuncComparator(sl + ' H comp'))
        comps_g.append(FuncComparator(sl + ' G'))
        names.append(sl)
else:
    slices = profile.get_slice_ranges()
    comp_dens = FuncComparator('Densities')
    comps_rh = list()
    comps_rhp = list()
    comps_rl = list()
    comps_rlp = list()
    names = list()
    for sl in slices:
        comps_rh.append(FuncComparator(sl + ' H'))
        comps_rhp.append(FuncComparator(sl + ' H\''))
        comps_rl.append(FuncComparator(sl + ' L'))
        comps_rlp.append(FuncComparator(sl + ' L\''))
        names.append(sl)

print 'Data load loop:'
stem_names = list()
for input_pkl in profiles_pkl:

    print '\tLoading the input profile: ' + input_pkl
    path, fname = os.path.split(input_pkl)
    stem_name, _ = os.path.splitext(fname)
    profile = ps.factory.unpickle_obj(input_pkl)
    if not isinstance(profile, ps.spatial.SetCloudsP):
        pairs = profile
        profile = None

    if profile is None:
        print '\tExtracting pairs graphs...'
        slices = pairs.get_slice_ranges()
        l_hr, l_hrx = pairs.get_cross_ripley_H()
        l_hp, l_hpx = pairs.get_comp_ripley_H()
        l_g, l_gx = pairs.get_cross_G()

        for (sl, hrx, hr, hpx, hp, g, gx, chr, chp, cg) in  \
            zip(slices, l_hrx, l_hr, l_hpx, l_hp, l_g, l_gx, comps_rhr, comps_rhp, comps_g):

            print '\tTrying to insert profile graphs into comparators:'
            cont = 0
            try:
                chr.insert_graph(stem_name, hrx, hr)
                cont += 1
                chp.insert_graph(stem_name, hpx, hp)
                cont += 1
                cg.insert_graph(stem_name, gx, g)
                cont += 1
                print '\t\tPair ' + str(stem_name) + ' successfully inserted.'
            except ps.pexceptions.PySegInputError:
                if cont > 0:
                    print '\t\tERROR: unexpected event'
                    sys.exit()
                print '\t\tWARNING: the graphs of this profile could not been inserted, '
                print '             this profile will not be considered.'
    else:
        print '\tExtracting profile graphs...'
        slices = profile.get_slice_ranges()
        densities = profile.get_densities()
        densx = np.arange(1, len(densities)+1)
        l_h, l_hx = profile.get_ripley_H()
        l_hp, l_hpx = profile.get_ripley_Hp()
        l_l, l_lx = profile.get_ripley_L()
        l_lp, l_lpx = profile.get_ripley_Lp()
        comp_dens.insert_graph(stem_name, densx, densities)

        for (sl, hx, h, hpx, hp, lx, l, lpx, lp, ch, chp, cl, clp) in  \
            zip(slices, l_hx, l_h, l_hpx, l_hp, l_lx, l_l, l_lpx, l_lp,
                comps_rh, comps_rhp, comps_rl, comps_rlp):

            print '\tTrying to insert profile graphs into comparators:'
            cont = 0
            try:
                ch.insert_graph(stem_name, hx, h)
                cont += 1
                chp.insert_graph(stem_name, hpx, hp)
                cont += 1
                cl.insert_graph(stem_name, lx, l)
                cont += 1
                clp.insert_graph(stem_name, lpx, lp)
                cont += 1
                print '\t\tProfile ' + str(stem_name) + 'successfully inserted.'
            except ps.pexceptions.PySegInputError:
                if cont > 0:
                    print '\t\tERROR: unexpected event'
                    sys.exit()
                print '\t\tWARNING: the graphs of this profile could not been inserted, '
                print '             this profile will not be considered.'

if profile is None:

    print 'Pairs plot loop:'
    for (name, chr, chp, cg) in zip(names, comps_rhr, comps_rhp, comps_g):

        if disp_plots:
            print '\tProfile ' + name + ' crossed Ripley\'s H analysis (close all windows to continue)...'
            chr.plot_comparison(block=True, leg_num=leg_num, leg_loc=leg_loc)
        if store_plots:
            print '\tStoring figure ' + name + ' crossed Ripley\'s H analysis...'
            fig_dir = output_dir + '/' + prefix_name + '_' + name + '_rhr'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            chr.store_figs(fig_dir, leg_num=leg_num, leg_loc=leg_loc)
        output_pkl = output_dir + '/' + prefix_name + '_' + name + '_rhr.pkl'
        print '\t\tStoring the result in file ' + output_pkl
        chr.pickle(output_pkl)

        if disp_plots:
            print '\tProfile ' + name + ' complemented Ripley\'s H\' analysis (close all windows to continue)...'
            chp.plot_comparison(block=True, leg_num=leg_num, leg_loc=leg_loc)
        if store_plots:
            print '\tStoring figure ' + name + ' complemented Ripley\'s H\' analysis...'
            fig_dir = output_dir + '/' + prefix_name + '_' + name + '_rhp'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            chp.store_figs(fig_dir, leg_num=leg_num, leg_loc=leg_loc)
        output_pkl = output_dir + '/' + prefix_name + '_' + name + '_rhp.pkl'
        print '\t\tStoring the result in file ' + output_pkl
        chp.pickle(output_pkl)

        if disp_plots:
            print '\tProfile ' + name + ' crossed G-Function analysis (close all windows to continue)...'
            cg.plot_comparison(block=True, leg_num=leg_num, leg_loc=leg_loc)
        if store_plots:
            print '\tStoring figure ' + name + ' crossed G-Function analysis...'
            fig_dir = output_dir + '/' + prefix_name + '_' + name + '_rl'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            cg.store_figs(fig_dir, leg_num=leg_num, leg_loc=leg_loc)
        output_pkl = output_dir + '/' + prefix_name + '_' + name + '_g.pkl'
        print '\t\tStoring the result in file ' + output_pkl
        cg.pickle(output_pkl)

else:

    if disp_plots:
        print '\tPlotting the results for densities (close all windows to continue)...'
        comp_dens.plot_comparison(block=True, leg_num=leg_num, leg_loc=leg_loc)
    if store_plots:
        print '\tStoring figures for densities...'
        fig_dir = output_dir + '/' + prefix_name + '_den'
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        comp_dens.store_figs(fig_dir, leg_num=leg_num, leg_loc=leg_loc)
    output_pkl = output_dir + '/' + prefix_name + '_den.pkl'
    print '\t\tStoring the result in file ' + output_pkl
    comp_dens.pickle(output_pkl)

    print 'Profile plot loop:'
    for (name, ch, chp, cl, clp) in zip(names, comps_rh, comps_rhp, comps_rl, comps_rlp):

        if disp_plots:
            print '\tProfile ' + name + ' Ripley\'s H analysis (close all windows to continue)...'
            ch.plot_comparison(block=True, leg_num=leg_num, leg_loc=leg_loc)
        if store_plots:
            print '\tStoring figure ' + name + ' Ripley\'s H analysis...'
            fig_dir = output_dir + '/' + prefix_name + '_' + name + '_rh'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            ch.store_figs(fig_dir, leg_num=leg_num, leg_loc=leg_loc)
        output_pkl = output_dir + '/' + prefix_name + '_' + name + '_rh.pkl'
        print '\t\tStoring the result in file ' + output_pkl
        ch.pickle(output_pkl)

        if disp_plots:
            print '\tProfile ' + name + ' Ripley\'s H\' analysis (close all windows to continue)...'
            chp.plot_comparison(block=True, leg_num=leg_num, leg_loc=leg_loc)
        if store_plots:
            print '\tStoring figure ' + name + ' Ripley\'s H\' analysis...'
            fig_dir = output_dir + '/' + prefix_name + '_' + name + '_rhp'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            chp.store_figs(fig_dir, leg_num=leg_num, leg_loc=leg_loc)
        output_pkl = output_dir + '/' + prefix_name + '_' + name + '_rhp.pkl'
        print '\t\tStoring the result in file ' + output_pkl
        chp.pickle(output_pkl)

        if disp_plots:
            print '\tProfile ' + name + ' Ripley\'s L analysis (close all windows to continue)...'
            cl.plot_comparison(block=True, leg_num=leg_num, leg_loc=leg_loc)
        if store_plots:
            print '\tStoring figure ' + name + ' Ripley\'s L analysis...'
            fig_dir = output_dir + '/' + prefix_name + '_' + name + '_rl'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            cl.store_figs(fig_dir, leg_num=leg_num, leg_loc=leg_loc)
        output_pkl = output_dir + '/' + prefix_name + '_' + name + '_rl.pkl'
        print '\t\tStoring the result in file ' + output_pkl
        cl.pickle(output_pkl)

        if disp_plots:
            print '\tProfile ' + name + ' Ripley\'s L\' analysis (close all windows to continue)...'
            clp.plot_comparison(block=True, leg_num=leg_num, leg_loc=leg_loc)
        if store_plots:
            print '\tStoring figure ' + name + ' Ripley\'s L\' analysis...'
            fig_dir = output_dir + '/' + prefix_name + '_' + name + '_rlp'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            clp.store_figs(fig_dir, leg_num=leg_num, leg_loc=leg_loc)
        output_pkl = output_dir + '/' + prefix_name + '_' + name + '_rlp.pkl'
        print '\t\tStoring the result in file ' + output_pkl
        clp.pickle(output_pkl)

print 'Terminated. (' + time.strftime("%c") + ')'

