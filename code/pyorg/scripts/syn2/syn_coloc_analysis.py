"""

    Find columns on synapse tomograms (ListTomograms) and measure their statistics.
    Files for columns visualization in VTK format.

    Input:  - 3 STAR files (layers) with the ListTomoParticles pickles

    Output: - Plots by tomograms and globally:
                + Number of columns
                + Columns density
            - Visualization file:
                + vtkPolyData objects for each synapse with the columns

"""

################# Package import

import os
import copy
import csv
import numpy as np
import sys
import time
from sklearn import linear_model
from pyorg import pexceptions, sub, disperse_io, surf
from pyorg.globals import unpickle_obj, clean_dir
from pyorg.surf.model import ModelCSRV
from pyorg.surf import ColumnsFinder, gen_layer_model
from matplotlib import pyplot as plt, rcParams
try:
    import cPickle as pickle
except ImportError:
    import pickle

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

BAR_WIDTH = .40

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['patch.linewidth'] = 2

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn2/org'

# Input STAR file
in_star_1 = ROOT_PATH + '/col/ltomos/ltomos_all_scol_xd5nm_test/all_scol_xd5nm_test_ltomos.star' # '/pre/ltomos/ltomos_all_pre_pre_ss7.31px/all_pre_pre_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_gather_mask/pre_premb_gather_mask_ltomos.star' # '/ves_40/ltomos_tether_lap/ves_40_cleft_premb_mask_lap_ltomos.star_ltomos.star' # '/ves_ap/ltomos_ves_ap_cleft/ves_ap_cleft_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_ABC/pre_premb_ABC_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_clst_flt_high_lap/clst_flt_high_lap_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_mask_lap/pre_premb_mask_lap_ltomos.star'
key_1, l_id1 = 'pre', 0 # 4
in_star_2 = ROOT_PATH + '/col/ltomos/ltomos_all_scol_xd5nm_test/all_scol_xd5nm_test_ltomos.star' # '/pre/ltomos/ltomos_all_pre_pre_ss7.31px/all_pre_pre_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_gather_mask/pre_premb_gather_mask_ltomos.star' # '/ves_40/ltomos_tether_lap/ves_40_cleft_premb_mask_lap_ltomos.star_ltomos.star' # '/ves_ap/ltomos_ves_ap_cleft/ves_ap_cleft_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_ABC/pre_premb_ABC_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_clst_flt_high_lap/clst_flt_high_lap_ltomos.star' # '/pre/ref_nomb_1_clean/ltomos_pre_premb_mask_lap/pre_premb_mask_lap_ltomos.star' # '/tether/ltomos/ltomos_all_pre_tether_ss7.31px/all_pre_tether_ltomos.star' # '/ves_40/ltomos_tether_lap/ves_40_cleft_premb_mask_lap_ltomos.star_ltomos.star' # '/pst/ref_nomb_1_clean/ltomos_pst_premb_gather_mask/pst_premb_gather_mask_ltomos.star' # '/pst/ampar_vs_nmdar/org/ltomos/ltomos_an_pre_2/an_pre2_ltomos.star' # '/ves_40/ltomos_tether_lap/ves_40_cleft_premb_mask_lap_ltomos.star_ltomos.star' # '/ves_ap/ltomos_ves_ap_cleft/ves_ap_cleft_ltomos.star' # '/az/ref_nomb_1_clean/ltomos_az_ref_1_gather_premb_cleft/az_cleft_premb_mask_ltomos.star' # '/pst/ampar_vs_nmdar/org/ltomos/ltomos_an_gather_pre_2/an_gather_pre2_ltomos.star' # '/pst/ampar_vs_nmdar/org/ltomos/ltomos_an_pre_2/an_pre2_ltomos.star' # '/pst/ref_nomb_1_clean/ltomos_pst_premb_gather_mask/pst_premb_gather_mask_ltomos.star' # '/ves_40/ltomos_tether_lap/ves_40_cleft_premb_mask_lap_ltomos.star_ltomos.star' # '/ves_40/ltomos_tether/ves_40_cleft_premb_mask_ltomos.star' # '/ves_40/ltomos_lap/lap_ltomos.star' # '/ves_40/ltomos_premb_mask/premb_mask_ltomos.star'
key_2, l_id2 = 'tether', 1 # 1
in_star_3 = ROOT_PATH + '/col/ltomos/ltomos_all_scol_xd5nm_test/all_scol_xd5nm_test_ltomos.star' # '/pst/ltomos/ltomos_all_pst_pre_ss7.31px/all_pst_pre_ltomos.star' # '/pst/ref_nomb_1_clean/ltomos_pst_premb_gather_mask/pst_premb_gather_mask_ltomos.star' # '/pst/ampar_vs_nmdar/org/ltomos/ltomos_an_pre_2/an_pre2_ltomos.star' # '/pst/nrt/k2_ABC/ltomos_k2_premb_ABC/k2_premb_ABC_ltomos.star' # '/pst/ampar_vs_nmdar/org/ltomos/ltomos_an_gather_pre_2/an_gather_pre2_ltomos.star' # '/pst/ampar_vs_nmdar/org/ltomos/ltomos_ampar_nmdar_premb_mask/ampar_nmdar_premb_gather_mask_ltomos.star' # '/pst/nrt/ltomos_k4_gather_premb_mask/k4_gather_premb_mask_ltomos.star' # '/pst/nrt/ltomos_clst_flt_high_lap/clst_flt_high_lap_ltomos.star' # '/pst/nrt/ltomos_k2_premb_mask/k2_premb_mask_ltomos.star' # '/pst/nrt/ltomos_pst_premb_mask_lap/pst_premb_mask_lap_ltomos.star'
key_3, l_id3 = 'pst', 2 # 4
in_star_4 = ROOT_PATH + '/col/ltomos/ltomos_all_scol_xd5nm_test/all_scol_xd5nm_test_ltomos.star' # '/tether/ltomos/ltomos_all_pre_tether_ss7.31px/all_pre_tether_ltomos.star' # '/pst/ampar_vs_nmdar/org/ltomos/ltomos_ampar_nmdar_premb_mask/ampar_nmdar_premb_gather_mask_ltomos.star' # '/pst/nrt/ltomos_k4_gather_premb_mask/k4_gather_premb_mask_ltomos.star' # '/pst/nrt/k2_ABC/ltomos_k2_premb_ABC/k2_premb_ABC_ltomos.star' # '/pst/nrt/ltomos_clst_flt_high_lap/clst_flt_high_lap_ltomos.star' # '/pst/nrt/ltomos_k2_premb_mask/k2_premb_mask_ltomos.star' # '/pst/nrt/ltomos_pst_premb_mask_lap/pst_premb_mask_lap_ltomos.star'
key_4, l_id4 = 'sv', 1 # 1
in_star_5 = ROOT_PATH + '/pst/ltomos/ltomos_all_pst_pre_ss7.31px/all_pst_pre_ltomos.star'
key_5, l_id5 = 'ampar', 0
in_star_6 = ROOT_PATH + '/pst/ltomos/ltomos_all_pst_pre_ss7.31px/all_pst_pre_ltomos.star'
key_6, l_id6 = 'nmdar', 1
in_tethers_csv = ROOT_PATH + '/../../syn/sub/relion/fils/pre/ref_nomb_1_clean/py_scripts/syn_num_tethers_2.csv'
in_part = ROOT_PATH + '/../../syn/sub/relion/fils/pre/vtps/sph_rad_5_surf.vtp'
in_wspace = None # ROOT_PATH + '/pst/ampar_vs_nmdar/org/method_2/3_pre_tether_pst/aln_15_cr10/3_pre_tether_pst_sim20_wspace.pkl'

# Output directory
out_dir = ROOT_PATH + '/coloc/3_preb_tether_pstb_test/aln_15_cr30' # '/pst/ampar_vs_nmdar/org/col_scol/col_an_0_aln_clst_15_sim_200_nn1_1_nn2_1_nn3_1'
out_stem = '3_preb_tether_pstb_sim1_m1'

# Pre-processing variables
pre_ssup = 5 #nm
pre_min_parts = 1

# Analysis variables
ana_res = 0.684 # nm/voxel
ana_col_rad = 30 # 15 # nm
ana_clst_method = 'HC'
ana_aln_dst = 15 # nm

# P-value computation settings
# Simulation model (currently only CSRV)
p_nsims = 1
p_per = 5 # %the

# Figure saving options
fig_fmt = '.png' # if None they showed instead
fig_weight_tet = True # In categories plotting tomograms are weigthed by the number of tethers

# Plotting options

# Ctrl vs Stim tomograms
ctrl_stems = ('11_2', '11_5', '11_6', '11_9', '14_9', '14_17', '14_18', '14_19', '14_20', '14_22', '14_24', '14_25')
stim_stems = ('13_1', '13_3', '14_14', '14_15', '14_26', '14_27', '14_28', '14_32', '14_33', '15_7', '15_8', '15_12')

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

# Compute the p-value for a single scalar in an array with the distribution
def compute_pval(val, sim, slope='high'):
    if slope == 'high':
        return float((val >= sim).sum()) / float(len(sim))
    else:
        return float((val >= sim).sum()) / float(len(sim))

# Computes pvalue from a matrix of simulations (n_arrays, array_samples)
def compute_pvals(exp_med, sims):
    n_sims = float(sims.shape[0])
    p_vals = np.zeros(shape=exp_med.shape, dtype=np.float32)
    for i, exp in enumerate(exp_med):
        sim_slice = sims[:, i]
        p_vals[i] = float((exp >= sim_slice).sum()) / n_sims
    return p_vals

# Check if a set of particles (rows) are aligned from the number of neighbours
def are_aligned(neighs, mask):
    aligns = np.zeros(shape=len(neighs), dtype=np.bool)
    for i, p_neigh in enumerate(neighs):
        if ((p_neigh[mask]>0).sum()) > 0:
            aligns[i] = True
    return aligns

# Units conversion
ana_col_rad_v, ana_aln_dst_v = ana_col_rad / ana_res, ana_aln_dst / ana_res

########## Print initial message

print 'Second order analysis for colocalization to  ListTomoParticles by tomograms.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput directory: ' + str(out_dir)
print '\tOuput stem: ' + str(out_stem)
print '\tInput analysis STAR file 1: ' + str(in_star_1)
print '\t\t-Key: ' + key_1
print '\t\t-List ID: ' + str(l_id1)
print '\tInput analysis STAR file 2: ' + str(in_star_2)
print '\t\t-Key: ' + key_2
print '\t\t-List ID: ' + str(l_id2)
print '\tInput analysis STAR file 3: ' + str(in_star_3)
print '\t\t-Key: ' + key_3
print '\t\t-List ID: ' + str(l_id3)
print '\tInput analysis STAR file sv: ' + str(in_star_4)
print '\t\t-Key: ' + key_4
print '\t\t-List ID: ' + str(l_id4)
print '\tInput particle shape for simulations: ' + str(in_part)
if in_wspace is not None:
    print '\tLoad workspace from: ' + in_wspace
else:
    print '\tPre-processing: '
    if pre_ssup is not None:
        print '\t\t-Scale supression: ' + str(pre_ssup) + ' nm'
print '\tOrganization analysis settings: '
print '\t\t-Data resolution: ' + str(ana_res) + ' nm/vx '
print '\t\t-Cluster radius: ' + str(ana_col_rad) + ' nm / ' + str(ana_col_rad_v) + ' vx'
print '\t\t-Tether->alignment radii range: ' + str(ana_aln_dst) + ' nm / ' + str(ana_aln_dst_v) + ' vx'
print '\t\t-Method for clustering: ' + ana_clst_method
print '\tP-Value computation setting:'
print '\t\t-Percentile: ' + str(p_per) + ' %'
print '\t\t-Number of instances for simulations: ' + str(p_nsims)
if fig_fmt is not None:
    print '\tStoring figures:'
    print '\t\t-Format: ' + str(fig_fmt)
else:
    print '\tPlotting settings: '
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

print '\tLoading input data (only the first entry is loaded)...'
star_1, star_2, star_3 = sub.Star(), sub.Star(), sub.Star()
star_4, star_5, star_6 = sub.Star(), sub.Star(), sub.Star()
try:
    star_1.load(in_star_1)
    star_2.load(in_star_2)
    star_3.load(in_star_3)
    star_4.load(in_star_4)
    star_5.load(in_star_5)
    star_6.load(in_star_6)
except pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
ltomos_pkl = star_1.get_element('_psPickleFile', l_id1)
list_1 = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_2.get_element('_psPickleFile', l_id2)
list_2 = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_3.get_element('_psPickleFile', l_id3)
list_3 = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_4.get_element('_psPickleFile', l_id4)
list_4 = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_5.get_element('_psPickleFile', l_id5)
list_5 = unpickle_obj(ltomos_pkl)
ltomos_pkl = star_6.get_element('_psPickleFile', l_id6)
list_6 = unpickle_obj(ltomos_pkl)
set_lists = surf.SetListTomoParticles()
set_lists.add_list_tomos(list_1, key_1)
set_lists.add_list_tomos(list_2, key_2)
set_lists.add_list_tomos(list_3, key_3)
set_lists.add_list_tomos(list_4, key_4)
set_lists.add_list_tomos(list_5, key_5)
set_lists.add_list_tomos(list_6, key_6)
print '\tNumber of particles found for list 1:' + str(list_1.get_num_particles())
print '\tNumber of particles found for list 2:' + str(list_2.get_num_particles())
print '\tNumber of particles found for list 3:' + str(list_3.get_num_particles())
print '\tNumber of particles found for list 4:' + str(list_4.get_num_particles())
print '\tNumber of particles found for list 5:' + str(list_5.get_num_particles())
print '\tNumber of particles found for list 6:' + str(list_6.get_num_particles())

print '\tSet pre-processing...'
if pre_ssup is not None:
    pre_ssup_v = pre_ssup / ana_res
    set_lists.scale_suppression(pre_ssup_v)
if pre_min_parts > 0:
    set_lists.filter_by_particles_num_tomos(pre_min_parts)
    list_1 = set_lists.get_lists_by_key(key_1)
    list_2 = set_lists.get_lists_by_key(key_2)
    list_3 = set_lists.get_lists_by_key(key_3)
    list_4 = set_lists.get_lists_by_key(key_4)
    list_5 = set_lists.get_lists_by_key(key_5)
    list_6 = set_lists.get_lists_by_key(key_6)
part_vtp = disperse_io.load_poly(in_part)
print '\t\tNumber of particles found for list 1:' + str(list_1.get_num_particles())
print '\t\tNumber of particles found for list 2:' + str(list_2.get_num_particles())
print '\t\tNumber of particles found for list 3:' + str(list_3.get_num_particles())
print '\t\tNumber of particles found for list 4:' + str(list_4.get_num_particles())
print '\t\tNumber of particles found for list 5:' + str(list_5.get_num_particles())
print '\t\tNumber of particles found for list 6:' + str(list_6.get_num_particles())

print '\tBuilding the dictionaries...'
short_tkeys_dic = dict()
tomos_ntet = dict()
tomos_nc, tomos_den, tomos_denv, tomos_dent = dict(), dict(), dict(), dict()
tomos_area, tomos_acol, tomos_occ = dict(), dict(), dict()
tomos_area_sims, tomos_acol_sims, tomos_occ_sims = dict(), dict(), dict()
tomos_nc_sims, tomos_den_sims, tomos_denv_sims, tomos_dent_sims = dict(), dict(), dict(), dict()
tomos_nc_a12, tomos_nc_a12_sim, tomos_nc_a12r, tomos_nc_a12r_sim = dict(), dict(), dict(), dict()
tomos_np_l0, tomos_np_l1, tomos_np_l2, tomos_np_l3 = dict(), dict(), dict(), dict()
tomos_np_l4, tomos_np_l5, tomos_np_l6 = dict(), dict(), dict()
tomos_npc_l0, tomos_npc_l1, tomos_npc_l2, tomos_npc_l3 = dict(), dict(), dict(), dict()
tomos_npc_l0_sim, tomos_npc_l1_sim, tomos_npc_l2_sim, tomos_npc_l3_sim = dict(), dict(), dict(), dict()
tomos_nc_sims2, tomos_den_sims2, tomos_occ_sims2, tomos_area_sims2, tomos_acol_sims2 = dict(), dict(), dict(), dict(), dict()
tomos_nsc, tomos_nsc_sims, tomos_nsc_sims2 = dict(), dict(), dict()
tab = dict()
for tkey, ltomo in zip(list_1.get_tomos().iterkeys(), list_1.get_tomos().itervalues()):
    try:
        ltomo_1, ltomo_2 = list_2.get_tomo_by_key(tkey), list_2.get_tomo_by_key(tkey)
    except KeyError:
        print 'WARNING: tomogram in layer 1 not in lists for layers 2 or 3!'
        continue
    tomos_ntet[tkey] = 0
    tomos_nc[tkey], tomos_den[tkey], tomos_denv[tkey], tot_dent = 0, 0, 0, 0
    tomos_area[tkey], tomos_acol[tkey], tomos_occ[tkey] = 0, list(), 0
    tomos_area_sims[tkey], tomos_acol_sims[tkey], tomos_occ_sims[tkey] = list(), list(), list()
    tomos_nc_sims[tkey], tomos_den_sims[tkey], tomos_denv_sims[tkey], tomos_dent_sims[tkey] = \
        list(), list(), list(), list()
    tomos_np_l4[tkey], tomos_nc_a12[tkey], tomos_nc_a12_sim[tkey] = 0, 0, list()
    tomos_nc_a12r[tkey], tomos_nc_a12r_sim[tkey] = list(), list()
    short_tkey = os.path.splitext(os.path.split(tkey)[1])[0]
    short_tkeys_dic[short_tkey] = tkey
    tomos_np_l0[tkey], tomos_np_l1[tkey], tomos_np_l2[tkey], tomos_np_l3[tkey] = 0, 0, 0, 0
    tomos_np_l4[tkey], tomos_np_l5[tkey], tomos_np_l6[tkey] = 0, 0, 0
    tomos_npc_l0[tkey], tomos_npc_l1[tkey], tomos_npc_l2[tkey], tomos_npc_l3[tkey] = 0, 0, 0, 0
    tomos_npc_l0_sim[tkey], tomos_npc_l1_sim[tkey], tomos_npc_l2_sim[tkey], tomos_npc_l3_sim[tkey] = list(), list(), \
                                                                                                     list(), list()
    tomos_nc_sims2[tkey], tomos_den_sims2[tkey], tomos_occ_sims2[tkey] = list(), list(), list()
    tomos_area_sims2[tkey], tomos_acol_sims2[tkey] = list(), list()
    tomos_nc[tkey], tomos_nsc_sims[tkey], tomos_nsc_sims2[tkey] = 0, list(), list()
    tab[tkey] = list()

print '\tComputing reference properties...'
vols = list_1.get_volumes_dict()
with open(in_tethers_csv, mode='r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    vesicles = dict()
    for row in reader:
        try:
            vesicles[short_tkeys_dic[row[0]]] = float(row[1])
        except KeyError:
            continue

out_tomos_dir = out_stem_dir + '/tomos'
os.makedirs(out_tomos_dir)
if in_wspace is None:

    tomo_count = 0
    part_star_0, part_star_1, part_star_2, part_star_3 = sub.Star(), sub.Star(), sub.Star(), sub.Star()
    part_star_4, part_star_5, part_star_6, part_star_7 = sub.Star(), sub.Star(), sub.Star(), sub.Star()
    part_star_8, part_star_9, part_star_10, part_star_11 = sub.Star(), sub.Star(), sub.Star(), sub.Star()
    print '\t\t-Tomograms computing loop:'
    for tkey, ltomo_1 in zip(list_1.get_tomos().iterkeys(), list_1.get_tomos().itervalues()):

        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        print '\t\t\t+Processing tomogram (' + str(tomo_count + 1) + \
                  ' of ' + str(len(tomos_nc.keys())) + ') : ' + os.path.split(tkey)[1]
        if os.path.split(tkey)[1] == 'syn_14_20_bin2_rot_crop2_seg.fits':
            print 'jol'
        try:
            ltomo_2, ltomo_3 = list_2.get_tomo_by_key(tkey), list_3.get_tomo_by_key(tkey)
            # if (ltomo_1.get_num_particles() <= 0) or (ltomo_2.get_num_particles() <= 0) or (ltomo_3.get_num_particles() <= 0):
            #     print 'WARNING: tomogram with an empty layer, skipping...'
            #     continue
        except KeyError:
            print 'WARNING: key not found, skipping...'
            continue
        tomo_count += 1
        tomos_np_l1[tkey], tomos_np_l2[tkey], tomos_np_l3[tkey] = ltomo_1.get_num_particles(), \
                                                                  ltomo_2.get_num_particles(), \
                                                                  ltomo_3.get_num_particles()
        try:
            ltomo_4 = list_4.get_tomo_by_key(tkey)
            tomos_np_l4[tkey] = ltomo_4.get_num_particles()
        except KeyError:
            tomos_np_l4[tkey] = 0
        try:
            ltomo_5 = list_5.get_tomo_by_key(tkey)
            tomos_np_l5[tkey] = ltomo_5.get_num_particles()
        except KeyError:
            tomos_np_l5[tkey] = 0
        try:
            ltomo_6 = list_6.get_tomo_by_key(tkey)
            tomos_np_l6[tkey] = ltomo_6.get_num_particles()
        except KeyError:
            tomos_np_l6[tkey] = 0

        print '\t\t\t\t-Computing columns...'
        # try:
        cfinder = ColumnsFinder(ltomo_1, ltomo_2, ltomo_3, ana_col_rad_v, ana_aln_dst_v, clst_method=ana_clst_method)
        # except Exception:
        #     print 'WARNING: The columns cannot be constructed for, continuing...'
        #     continue

        print '\tUpdating column particles STAR files...'
        cfinder.add_columns_to_star(part_star_0, tkey, layer=None)
        cfinder.add_columns_to_star(part_star_1, tkey, layer=1)
        cfinder.add_columns_to_star(part_star_2, tkey, layer=2)
        cfinder.add_columns_to_star(part_star_3, tkey, layer=3)
        cfinder.add_columns_to_star(part_star_4, tkey, layer=None, mode='parts')
        cfinder.add_columns_to_star(part_star_5, tkey, layer=1, mode='parts')
        cfinder.add_columns_to_star(part_star_6, tkey, layer=2, mode='parts')
        cfinder.add_columns_to_star(part_star_7, tkey, layer=3, mode='parts')
        cfinder.add_columns_to_star(part_star_8, tkey, layer=None, mode='scols')
        cfinder.add_columns_to_star(part_star_9, tkey, layer=1, mode='scols')
        cfinder.add_columns_to_star(part_star_10, tkey, layer=2, mode='scols')
        cfinder.add_columns_to_star(part_star_11, tkey, layer=3, mode='scols')

        out_scol_vtp = out_tomos_dir + '/' + tkey_short + '_scol.vtp'
        print '\t\t\t\t-Storing the sub-columns built in: ' + out_scol_vtp
        disperse_io.save_vtp(cfinder.gen_subcolumns_vtp(), out_scol_vtp)
        out_col_vtp = out_tomos_dir + '/' + tkey_short + '_col.vtp'
        print '\t\t\t\t-Storing the columns built in: ' + out_col_vtp
        disperse_io.save_vtp(cfinder.gen_columns_vtp(), out_col_vtp)

        print '\t\t\t\t-Count the number of particles...'
        tomos_ntet[tkey] = ltomo_4.get_num_particles()
        tab[tkey].append(tomos_ntet[tkey])
        tomos_nc[tkey] = cfinder.get_num_columns()
        tomos_nsc[tkey] = cfinder.get_num_sub_columns()
        tab[tkey].append(tomos_nsc[tkey])
        tab[tkey].append(tomos_nc[tkey])
        tomos_den[tkey] = cfinder.get_den_columns()
        if tomos_ntet[tkey] > 0:
            tomos_dent[tkey] = float(tomos_nc[tkey]) / tomos_ntet[tkey]
        if vesicles[tkey] > 0:
            tomos_denv[tkey] = float(tomos_nc[tkey]) / float(vesicles[tkey])
        tomos_area[tkey] = cfinder.get_area_tomo() * (ana_res * ana_res)
        tomos_acol[tkey] = cfinder.get_occupancy(mode='cyl', rad=ana_aln_dst_v, area=True, layers=[0, ], scols=True) \
                           * (ana_res * ana_res)
        tab[tkey].append(tomos_acol[tkey])
        tab[tkey].append(tomos_area[tkey])
        tab[tkey].append(tomos_np_l1[tkey])
        tab[tkey].append(tomos_np_l2[tkey])
        tab[tkey].append(tomos_np_l3[tkey])
        tab[tkey].append(tomos_np_l5[tkey])
        tab[tkey].append(tomos_np_l6[tkey])
        if len(tomos_acol) > 0:
            tomos_occ[tkey] = cfinder.get_occupancy(mode='cyl', rad=ana_col_rad_v)
        else:
            tomos_occ[tkey] = .0
        tomos_npc_l1[tkey] = cfinder.get_num_particles(lyr=1)
        tomos_npc_l2[tkey] = cfinder.get_num_particles(lyr=2)
        tomos_npc_l3[tkey] = cfinder.get_num_particles(lyr=3)
        # tomos_tca[tkey] = cfinder.get_coloc_columns(ltomos_4.get_coords())

        print '\t\t\t\t-Simulating columns:'
        for i in range(p_nsims):
            print '\t\t\t\t\t+Simulating instance ' + str(i) + ' of ' + str(p_nsims) + '...'
            sim_lyr_2 = gen_layer_model(ltomo_2, part_vtp, ModelCSRV, mode_emb='center')
            if (in_star_2 == in_star_3) and (l_id2 == l_id3):
                sim_lyr_3 = sim_lyr_2
                print '\t\t\t\t\t\t*Layer 2 and 3 have the same simulation.'
            else:
                sim_lyr_3 = gen_layer_model(ltomo_3, part_vtp, ModelCSRV, mode_emb='center')
            try:
                cfinder2 = ColumnsFinder(ltomo_1, sim_lyr_2, sim_lyr_3, ana_col_rad_v, ana_aln_dst_v)
                hold_nc_sims = cfinder2.get_num_columns()
                tomos_nc_sims[tkey].append(hold_nc_sims)
                tomos_nsc_sims[tkey].append(cfinder2.get_num_sub_columns())
                tomos_den_sims[tkey].append(cfinder2.get_den_columns())
                if tomos_ntet[tkey] > 0:
                    tomos_dent_sims[tkey].append(float(hold_nc_sims) / tomos_ntet[tkey])
                if vesicles[tkey] > 0:
                    tomos_denv_sims[tkey].append(float(hold_nc_sims) / float(vesicles[tkey]))
                tomos_area_sims[tkey].append(cfinder2.get_area_tomo() * (ana_res * ana_res))
                tomos_acol_sims[tkey].append(cfinder2.get_occupancy(mode='cyl', rad=ana_aln_dst_v, area=True,
                                                                    layers=[0, ], scols=True)) * (ana_res * ana_res)
                tomos_occ_sims[tkey].append(cfinder2.get_occupancy(mode='cyl', rad=ana_col_rad_v))
                tomos_npc_l1_sim[tkey].append(cfinder2.get_num_particles(lyr=1))
                tomos_npc_l2_sim[tkey].append(cfinder2.get_num_particles(lyr=2))
                tomos_npc_l3_sim[tkey].append(cfinder2.get_num_particles(lyr=3))
                # tomos_tca_sim[tkey].append(cfinder2.get_coloc_columns(ltomos_4.get_coords()))
            except Exception:
                tomos_nc_sims[tkey].append(0)
                tomos_nsc_sims[tkey].append(0)
                tomos_den_sims[tkey].append(0.)
                tomos_dent_sims[tkey].append(0.)
                tomos_denv_sims[tkey].append(0.)
                tomos_area_sims[tkey].append(0.)
                tomos_occ_sims[tkey].append(.0)
                tomos_npc_l1_sim[tkey].append(.0)
                tomos_npc_l2_sim[tkey].append(.0)
                tomos_npc_l3_sim[tkey].append(.0)
                # tomos_tca_sim[tkey].append(.0)
            sim_lyr_1 = gen_layer_model(ltomo_1, part_vtp, ModelCSRV, mode_emb='center')
            try:
                cfinder3 = ColumnsFinder(sim_lyr_1, ltomo_2, ltomo_3,
                                         ana_col_rad_v, ana_aln_dst_v)
                tomos_nc_sims2[tkey].append(cfinder3.get_num_columns())
                tomos_nsc_sims2[tkey].append(cfinder3.get_num_sub_columns())
                tomos_den_sims2[tkey].append(cfinder3.get_den_columns())
                hold_area, hold_acols = cfinder3.get_area_tomo(), cfinder3.get_area_columns(mode='cyl',
                                                                                            rad=ana_aln_dst_v,
                                                                                            layers=[0, ],
                                                                                            scols=True) * (ana_res * ana_res)
                tomos_area_sims2[tkey].append(hold_area)
                if len(hold_acols) > 0:
                    tomos_acol_sims2[tkey] += hold_acols
                    tomos_occ_sims2[tkey].append(cfinder3.get_occupancy(mode='cyl', rad=ana_col_rad_v))
                else:
                    tomos_occ_sims2[tkey].append(.0)
                # tomos_tca_sim2[tkey].append(cfinder2.get_coloc_columns(ltomos_4.get_coords()))
            except Exception:
                tomos_nc_sims2[tkey].append(0)
                tomos_nsc_sims2[tkey].append(0)
                tomos_den_sims2[tkey].append(0.)
                tomos_area_sims2[tkey].append(0.)
                tomos_occ_sims2[tkey].append(.0)
                # tomos_tca_sim[tkey].append(.0)

    print '\tStoring the particles STAR files for the column layers:'
    hold_out_star = out_tomos_dir + '/0_' + out_stem + '_col.star'
    print '\t\t-File: ' + hold_out_star
    part_star_0.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/1_' + out_stem + '_lyr1.star'
    print '\t\t-File: ' + hold_out_star
    part_star_1.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/2_' + out_stem + '_lyr2.star'
    print '\t\t-File: ' + hold_out_star
    part_star_2.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/3_' + out_stem + '_lyr3.star'
    print '\t\t-File: ' + hold_out_star
    part_star_3.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/0_' + out_stem + '_part_col.star'
    print '\t\t-File: ' + hold_out_star
    part_star_4.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/1_' + out_stem + '_part_lyr1.star'
    print '\t\t-File: ' + hold_out_star
    part_star_5.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/2_' + out_stem + '_part_lyr2.star'
    print '\t\t-File: ' + hold_out_star
    part_star_6.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/3_' + out_stem + '_part_lyr3.star'
    print '\t\t-File: ' + hold_out_star
    part_star_7.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/0_' + out_stem + '_scols_col.star'
    print '\t\t-File: ' + hold_out_star
    part_star_8.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/1_' + out_stem + '_scols_lyr1.star'
    print '\t\t-File: ' + hold_out_star
    part_star_9.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/2_' + out_stem + '_scols_lyr2.star'
    print '\t\t-File: ' + hold_out_star
    part_star_10.store(hold_out_star)
    hold_out_star = out_tomos_dir + '/3_' + out_stem + '_scols_lyr3.star'
    print '\t\t-File: ' + hold_out_star
    part_star_11.store(hold_out_star)

    out_wspace = out_dir + '/' + out_stem + '_wspace.pkl'
    print '\tPickling computation workspace in: ' + out_wspace
    wspace = (tomos_nc, tomos_den, tomos_denv, tomos_dent,
              tomos_nc_sims, tomos_den_sims, tomos_denv_sims, tomos_dent_sims,
              tomos_np_l4, tomos_nc_a12, tomos_nc_a12_sim, tomos_nc_a12r, tomos_nc_a12r_sim,
              vesicles, vols,
              tomos_ntet,
              tomos_np_l0, tomos_np_l1, tomos_np_l2, tomos_np_l3,
              tomos_npc_l0, tomos_npc_l1, tomos_npc_l2, tomos_npc_l3,
              tomos_npc_l0_sim, tomos_npc_l1_sim, tomos_npc_l2_sim, tomos_npc_l3_sim,
              tomos_area, tomos_acol, tomos_occ,
              tomos_area_sims, tomos_acol_sims, tomos_occ_sims,
              tomos_nc_sims2, tomos_den_sims2, tomos_occ_sims2, tomos_area_sims2, tomos_acol_sims2,
              tomos_nsc, tomos_nsc_sims, tomos_nsc_sims2,
              tab)
    with open(out_wspace, "wb") as fl:
        pickle.dump(wspace, fl)
        fl.close()

else:

    print '\tLoading the workspace: ' + in_wspace
    with open(in_wspace, 'r') as pkl:
        wspace = pickle.load(pkl)
    tomos_nc, tomos_den, tomos_denv, tomos_dent = wspace[0], wspace[1], wspace[2], wspace[3]
    tomos_nc_sims, tomos_den_sims, tomos_denv_sims, tomos_dent_sims = wspace[4], wspace[5], wspace[6], wspace[7]
    tomos_np_l4, tomos_nc_a12, tomos_nc_a12_sim, tomos_nc_a12r, tomos_nc_a12r_sim = wspace[8], wspace[9], wspace[10], \
                                                                                    wspace[11], wspace[12]
    vesicles, vols = wspace[13], wspace[14]
    tomos_ntet = wspace[15]
    tomos_np_l0, tomos_np_l1, tomos_np_l2, tomos_np_l3 = wspace[16], wspace[17], wspace[18], wspace[19]
    tomos_npc_l0, tomos_npc_l1, tomos_npc_l2, tomos_npc_l3 = wspace[20], wspace[21], wspace[22], wspace[23]
    tomos_npc_l0_sim, tomos_npc_l1_sim, tomos_npc_l2_sim, tomos_npc_l3_sim = wspace[24], wspace[25], wspace[26], \
                                                                             wspace[27]
    tomos_area, tomos_acol, tomos_occ = wspace[28], wspace[29], wspace[30]
    tomos_area_sims, tomos_acol_sims, tomos_occ_sims = wspace[31], wspace[32], wspace[33],
    tomos_nc_sims2, tomos_den_sims2, tomos_occ_sims2, tomos_area_sims2, tomos_acol_sims2 = wspace[34], wspace[35], \
                                                                                           wspace[36], wspace[37], \
                                                                                           wspace[38]
    tomos_nsc, tomos_nsc_sims, tomos_nsc_sims2 = wspace[39], wspace[40], wspace[41]
    tab = wspace[42]

print '\tTOMOGRAMS PLOTTING LOOP: '

print '\t\t-Plotting the number of columns...'
for tkey, nc in zip(tomos_nc.iterkeys(), tomos_nc.itervalues()):
    if nc <= 0:
        continue
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('# columns')
    plt.bar(1, nc, BAR_WIDTH, color='blue', linewidth=4)
    ic_low = np.percentile(tomos_nc_sims[tkey], p_per)
    ic_med = np.percentile(tomos_nc_sims[tkey], 50)
    ic_high = np.percentile(tomos_nc_sims[tkey], 100-p_per)
    plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=4)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low = np.percentile(tomos_nc_sims2[tkey], p_per)
    ic_med = np.percentile(tomos_nc_sims2[tkey], 50)
    ic_high = np.percentile(tomos_nc_sims2[tkey], 100 - p_per)
    plt.bar(3, ic_med, BAR_WIDTH, color='gray', linewidth=4)
    plt.errorbar(3, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/nc.png')
    plt.close()

print '\t\t-Plotting the number of sub-columns...'
for tkey, nsc in zip(tomos_nsc.iterkeys(), tomos_nsc.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('# sub-columns')
    plt.bar(1, nsc, BAR_WIDTH, color='blue', linewidth=4)
    ic_low = np.percentile(tomos_nsc_sims[tkey], p_per)
    ic_med = np.percentile(tomos_nsc_sims[tkey], 50)
    ic_high = np.percentile(tomos_nsc_sims[tkey], 100-p_per)
    plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=4)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    ic_low = np.percentile(tomos_nsc_sims2[tkey], p_per)
    ic_med = np.percentile(tomos_nsc_sims2[tkey], 50)
    ic_high = np.percentile(tomos_nsc_sims2[tkey], 100 - p_per)
    plt.bar(3, ic_med, BAR_WIDTH, color='gray', linewidth=4)
    plt.errorbar(3, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/nsc.png')
    plt.close()

print '\t\t-Plotting columns occupancy...'
for tkey, den in zip(tomos_occ.iterkeys(), tomos_occ.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    if den <= 0:
        continue
    plt.figure()
    plt.ylabel('% of columns surface occupancy')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.bar(1, 100. * tomos_occ[tkey], BAR_WIDTH, color='blue', linewidth=4)
    occ_sims = np.asarray(tomos_occ_sims[tkey])
    ic_low = 100. * np.percentile(occ_sims, p_per)
    ic_med = 100. * np.percentile(occ_sims, 50)
    ic_high = 100. * np.percentile(occ_sims, 100-p_per)
    plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=4)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    occ_sims2 = np.asarray(tomos_occ_sims2[tkey])
    ic_low = 100. * np.percentile(occ_sims2, p_per)
    ic_med = 100. * np.percentile(occ_sims2, 50)
    ic_high = 100. * np.percentile(occ_sims2, 100 - p_per)
    plt.bar(3, ic_med, BAR_WIDTH, color='gray', linewidth=4)
    plt.errorbar(3, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2, 3), ('EXPERIMENTAL', 'SIMULATED', 'SIMULATED\''))
    plt.xlim(0.5, 3.5)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/occ.png')
    plt.close()

print '\t\t-Plotting columns density per synaptic vesicle...'
for tkey, denv in zip(tomos_denv.iterkeys(), tomos_denv.itervalues()):
    if len(tomos_denv_sims[tkey]) > 0:
        tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
        plt.figure()
        plt.ylabel('Sub-columns density [sub-column/sv]')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.bar(1, denv, BAR_WIDTH, color='blue', linewidth=2)
        denv_sims = np.asarray(tomos_denv_sims[tkey])
        ic_low = np.percentile(denv_sims, p_per)
        ic_med = np.percentile(denv_sims, 50)
        ic_high = np.percentile(denv_sims, 100-p_per)
        plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
        plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
        plt.xlim(0.5, 2.5)
        plt.tight_layout()
        if fig_fmt is None:
            plt.show(block=True)
        else:
            hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            plt.savefig(hold_dir + '/denv.png')
        plt.close()

print '\t\t-Plotting fraction of particles within columns for layer 1...'
for tkey, den in zip(tomos_np_l1.iterkeys(), tomos_npc_l1.itervalues()):
    if den <= 0:
        continue
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('% of particles within columns')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    if tomos_npc_l1[tkey] > 0:
        hold_val = 100. * (tomos_npc_l1[tkey] / float(tomos_np_l1[tkey]))
        hold_sims = 100. * (np.asarray(tomos_npc_l1_sim[tkey]) / float(tomos_np_l1[tkey]))
        ic_low = np.percentile(hold_sims, p_per)
        ic_med = np.percentile(hold_sims, 50)
        ic_high = np.percentile(hold_sims, 100 - p_per)
    else:
        hold_val = 0
        ic_low, ic_med, ic_high = 0, 0, 0
    plt.bar(1, hold_val, BAR_WIDTH, color='blue', linewidth=4)
    plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=4)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
    plt.xlim(0.5, 2.5)
    plt.ylim(0, 100)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/np_1.png')
    plt.close()

print '\t\t-Plotting fraction of particles within columns for layer 2...'
for tkey, den in zip(tomos_np_l2.iterkeys(), tomos_npc_l2.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('% of particles within columns')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    if tomos_npc_l2[tkey] > 0:
        hold_val = 100. * (tomos_npc_l2[tkey] / float(tomos_np_l2[tkey]))
        hold_sims = 100. * (np.asarray(tomos_npc_l2_sim[tkey]) / float(tomos_np_l2[tkey]))
        ic_low = np.percentile(hold_sims, p_per)
        ic_med = np.percentile(hold_sims, 50)
        ic_high = np.percentile(hold_sims, 100 - p_per)
    else:
        hold_val = 0
        ic_low, ic_med, ic_high = 0, 0, 0
    plt.bar(1, hold_val, BAR_WIDTH, color='blue', linewidth=4)
    plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=4)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
    plt.xlim(0.5, 2.5)
    plt.ylim(0, 100)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/np_2.png')
    plt.close()

print '\t\t-Plotting fraction of particles within columns for layer 3...'
for tkey, den in zip(tomos_np_l3.iterkeys(), tomos_npc_l3.itervalues()):
    tkey_short = os.path.splitext(os.path.split(tkey)[1])[0]
    plt.figure()
    plt.ylabel('% of particles within columns')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    if tomos_npc_l3[tkey] > 0:
        hold_val = 100. * (tomos_npc_l3[tkey] / float(tomos_np_l3[tkey]))
        hold_sims = 100. * (np.asarray(tomos_npc_l3_sim[tkey]) / float(tomos_np_l3[tkey]))
        ic_low = np.percentile(hold_sims, p_per)
        ic_med = np.percentile(hold_sims, 50)
        ic_high = np.percentile(hold_sims, 100 - p_per)
    else:
        hold_val = 0
        ic_low, ic_med, ic_high = 0, 0, 0
    plt.bar(1, hold_val, BAR_WIDTH, color='blue', linewidth=4)
    plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=4)
    plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
                 ecolor='k', elinewidth=4, capthick=4, capsize=8)
    plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
    plt.xlim(0.5, 2.5)
    plt.ylim(0, 100)
    plt.tight_layout()
    if fig_fmt is None:
        plt.show(block=True)
    else:
        hold_dir = out_tomos_dir + '/' + tkey_short + '/col/'
        if not os.path.exists(hold_dir):
            os.makedirs(hold_dir)
        plt.savefig(hold_dir + '/np_3.png')
    plt.close()

print '\t\t-Plotting fractions of particles within columns...'
np_l1, np_l2, np_l3 = list(), list(), list()
npc_l1, npc_l2, npc_l3 = list(), list(), list()
for i, tkey in enumerate(tomos_np_l1.iterkeys()):
    if tomos_np_l1[tkey] > 0:
        np_l1.append(tomos_np_l1[tkey])
        npc_l1.append(tomos_npc_l1[tkey])
    if tomos_np_l2[tkey] > 0:
        np_l2.append(tomos_np_l2[tkey])
        npc_l2.append(tomos_npc_l2[tkey])
    if tomos_np_l3[tkey] > 0:
        np_l3.append(tomos_np_l3[tkey])
        npc_l3.append(tomos_npc_l3[tkey])
npc_l1 = 100. * (np.asarray(npc_l1, dtype=np.float32) / np.asarray(np_l1, dtype=np.float32))
npc_l2 = 100. * (np.asarray(npc_l2, dtype=np.float32) / np.asarray(np_l2, dtype=np.float32))
npc_l3 = 100. * (np.asarray(npc_l3, dtype=np.float32) / np.asarray(np_l3, dtype=np.float32))
ic_low, ic_med, ic_high = np.percentile(npc_l1, p_per), npc_l1.mean(), np.percentile(npc_l1, 100-p_per)
plt.bar(1, ic_med, BAR_WIDTH, color='mediumslateblue', linewidth=4)
plt.errorbar(1, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low, ic_med, ic_high = np.percentile(npc_l2, p_per), npc_l2.mean(), np.percentile(npc_l2, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='green', linewidth=4)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
ic_low, ic_med, ic_high = np.percentile(npc_l3, p_per), npc_l3.mean(), np.percentile(npc_l3, 100-p_per)
plt.bar(3, ic_med, BAR_WIDTH, color='magenta', linewidth=4)
plt.errorbar(3, npc_l3.mean(), yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 3), ('PRE', 'TETHER', 'PST'))
plt.xlim(0.5, 3.5)
plt.ylabel('% of particles within columns')
plt.xlabel('')
plt.ylim((0, 100))
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/npc_means.png', dpi=300)
plt.close()

print '\t\t-Plotting box-plots for occupancy...'
lst = dict().fromkeys(('CTRL', 'STIM'))
for key in lst.iterkeys():
    lst[key] = list()
plt.figure()
pd_pvals = None
for i, tkey in enumerate(tomos_occ.iterkeys()):
    hold_val = 100. * tomos_occ[tkey]
    tkey_hold = os.path.split(tkey)[1].split('_')
    tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
    if (hold_val is not None) and np.isfinite(hold_val):
        if tkey_stem in stim_stems:
            lst['CTRL'].append(hold_val)
        else:
            lst['STIM'].append(hold_val)
plt.boxplot(lst.values(), showfliers=False, notch=False)
# plt.grid(True, alpha=0.5)
plt.ylabel('% of sub-columns surface occupancy')
plt.xlabel('')
plt.ylim((0, 100))
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/occ_ctrl_vs_stim.png', dpi=300)
plt.close()

print '\t\t-Plotting box-plots for columns surface area...'
lst, lst_sim = dict(), dict()
plt.figure()
pd_pvals = None
all_occ = list()
sort_ids = np.argsort(tomos_area.values())
for i, idx in enumerate(sort_ids):
    tkey = tomos_area.keys()[idx]
    hold_val = np.asarray(tomos_occ[tkey], dtype=float)
    tkey_hold = os.path.split(tkey)[1].split('_')
    tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
    lst[i] =  100. * hold_val
    all_occ.append(100. * hold_val)
    lst_sim[i] = list()
    for hold_val_sim in tomos_occ_sims[tkey]:
        lst_sim[i].append(100. * hold_val_sim)
plt.boxplot(lst_sim.values(), showfliers=False, notch=True, whis=[5, 95])
plt.plot(np.arange(1, len(lst.values())+1), lst.values(), color='k', marker='*', linestyle='')
plt.ylim((0, 100))
x_line = np.linspace(-0.5, len(tomos_occ.keys())+.5, 2)
plt.plot(x_line, np.asarray(all_occ).mean()*np.ones(shape=len(x_line)), 'k--', linewidth=1)
# plt.grid(True, alpha=0.5)
plt.ylabel('% of sub-column surface occupancy')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/col_area.png', dpi=300)
plt.close()

print '\t\t-Plotting box-plots for number of sub-columns...'
lst, lst_sim, count = dict(), dict(), 0
plt.figure()
pd_pvals = None
all_occ = list()
# sort_ids = np.argsort(tomos_np_l4.values())
sort_ids = np.argsort(tomos_area.values())
for idx in sort_ids:
    # tkey = tomos_np_l4.keys()[idx]
    tkey = tomos_area.keys()[idx]
    if True: # tomos_np_l4[tkey] > 0:
        hold_val = np.asarray(tomos_nc[tkey], dtype=float)
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        lst[count] =  hold_val
        lst_sim[count] = list()
        for hold_val_sim in tomos_nc_sims[tkey]:
            lst_sim[count].append(hold_val_sim)
        count += 1
plt.boxplot(lst_sim.values(), showfliers=False, notch=True, whis=[5, 95])
plt.plot(np.arange(1, len(lst.values())+1), lst.values(), color='k', marker='*', linestyle='')
plt.grid(True, alpha=0.5, axis='y')
plt.ylabel('Number of sub-columns')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/box_scol.png', dpi=300)
plt.close()

print '\t\t-Plotting box-plots for number of sub-columns (v2)...'
lst, lst_sim, count = dict(), dict(), 0
plt.figure()
pd_pvals = None
all_occ = list()
# sort_ids = np.argsort(tomos_np_l4.values())
sort_ids = np.argsort(tomos_area.values())
for idx in sort_ids:
    # tkey = tomos_np_l4.keys()[idx]
    tkey = tomos_area.keys()[idx]
    if True: # tomos_np_l4[tkey] > 0:
        hold_val = np.asarray(tomos_nc[tkey], dtype=float)
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        lst[count] =  hold_val
        lst_sim[count] = list()
        for hold_val_sim in tomos_nc_sims2[tkey]:
            lst_sim[count].append(hold_val_sim)
        count += 1
plt.boxplot(lst_sim.values(), showfliers=False, notch=True, whis=[5, 95])
plt.plot(np.arange(1, len(lst.values())+1), lst.values(), color='k', marker='*', linestyle='')
plt.grid(True, alpha=0.5, axis='y')
plt.ylabel('Number of sub-columns')
plt.xlabel('')
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/box_scol_v2.png', dpi=300)
plt.close()

print '\t\t-Plotting box-plots for number of sub-columns by tether centroids...'
lst, lst_x, lst_sim, lst_sim_mn, tab_p, count = dict(), dict(), dict(), dict(), dict(), 0
plt.figure()
pd_pvals = None
all_occ = list()
sort_ids = np.argsort(tomos_ntet.values())
# sort_ids = np.argsort(tomos_area.values())
for idx in sort_ids:
    # tkey = tomos_np_l4.keys()[idx]
    tkey = tomos_area.keys()[idx]
    if True: # tomos_np_l4[tkey] > 0:
        try:
            hold_val = np.asarray(tomos_nsc[tkey], dtype=float)
        except KeyError:
            continue
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        lst[count] =  hold_val
        lst_x[count] = tomos_ntet[tkey]
        for hold_val_sim in tomos_nsc_sims[tkey]:
            try:
                lst_sim[lst_x[count]].append(hold_val_sim)
            except KeyError:
                lst_sim[lst_x[count]] = list()
                lst_sim[lst_x[count]].append(hold_val_sim)
        lst_sim_mn[count] = np.asarray(lst_sim[lst_x[count]], dtype=np.float32).mean()
        tab_p[tkey] = compute_pval(tomos_nsc[tkey], np.asarray(tomos_nsc_sims[tkey]))
        count += 1
plt.boxplot(lst_sim.values(), positions=lst_sim.keys(), showfliers=False, notch=True, whis=[5, 95])
plt.plot(lst_x.values(), lst.values(), color='k', marker='*', linestyle='')
l_ves = np.asarray(lst_x.values(), dtype=np.int).reshape(-1, 1)
l_col = np.asarray(lst.values(), dtype=np.int).reshape(-1, 1)
l_col_sim = np.asarray(lst_sim_mn.values(), dtype=np.int).reshape(-1, 1)
regr_0, regr_1 = linear_model.LinearRegression(fit_intercept=False), linear_model.LinearRegression(fit_intercept=False)
regr_0.fit(l_ves, l_col), regr_1.fit(l_ves, l_col_sim)
l_col_r, l_col_sim_r = regr_0.predict(l_ves), regr_1.predict(l_ves)
plt.plot(l_ves, l_col_r, color='black', label='EXPERIMENTAL', linestyle='-', linewidth=2.0)
plt.plot(l_ves, l_col_sim_r, color='black', label='SIMULATED', linestyle=':', linewidth=2.0)
# plt.xticks(lst_x.values())
plt.grid(True, alpha=0.5, axis='y')
plt.ylabel('# sub-columns')
plt.xlabel('# tether centroids')
plt.legend(loc=2)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/box_scol_centroids.png', dpi=300)
plt.close()

print '\t\t-Plotting box-plots for number of columns by tether centroids...'
lst, lst_x, lst_sim, lst_sim_mn, tab_p, count = dict(), dict(), dict(), dict(), dict(), 0
plt.figure()
pd_pvals = None
all_occ = list()
sort_ids = np.argsort(tomos_ntet.values())
# sort_ids = np.argsort(tomos_area.values())
for idx in sort_ids:
    # tkey = tomos_np_l4.keys()[idx]
    tkey = tomos_area.keys()[idx]
    if len(tomos_nc_sims[tkey]) <= 0:
        continue
    if True: # tomos_np_l4[tkey] > 0:
        hold_val = np.asarray(tomos_nc[tkey], dtype=float)
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        lst[count] =  hold_val
        lst_x[count] = tomos_ntet[tkey]
        for hold_val_sim in tomos_nc_sims[tkey]:
            try:
                lst_sim[lst_x[count]].append(hold_val_sim)
            except KeyError:
                lst_sim[lst_x[count]] = list()
                lst_sim[lst_x[count]].append(hold_val_sim)
        lst_sim_mn[count] = np.asarray(lst_sim[lst_x[count]], dtype=np.float32).mean()
        tab_p[tkey] = compute_pval(tomos_nc[tkey], np.asarray(tomos_nc_sims[tkey]))
        count += 1
plt.boxplot(lst_sim.values(), positions=lst_sim.keys(), showfliers=False, notch=True, whis=[5, 95])
plt.plot(lst_x.values(), lst.values(), color='k', marker='*', linestyle='')
l_ves = np.asarray(lst_x.values(), dtype=np.int).reshape(-1, 1)
l_col = np.asarray(lst.values(), dtype=np.int).reshape(-1, 1)
l_col_sim = np.asarray(lst_sim_mn.values(), dtype=np.int).reshape(-1, 1)
regr_0, regr_1 = linear_model.LinearRegression(fit_intercept=False), linear_model.LinearRegression(fit_intercept=False)
regr_0.fit(l_ves, l_col), regr_1.fit(l_ves, l_col_sim)
l_col_r, l_col_sim_r = regr_0.predict(l_ves), regr_1.predict(l_ves)
plt.plot(l_ves, l_col_r, color='black', label='EXPERIMENTAL', linestyle='-', linewidth=2.0)
plt.plot(l_ves, l_col_sim_r, color='black', label='SIMULATED', linestyle=':', linewidth=2.0)
# plt.xticks(lst_x.values())
plt.grid(True, alpha=0.5, axis='y')
plt.ylabel('# columns')
plt.xlabel('# tether centroids')
plt.legend(loc=2)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/box_col_centroids.png', dpi=300)
plt.close()

print '\t\t-Plotting box-plots for number of sub-columns by SV...'
lst, lst_x, lst_sim, lst_sim_mn, count = dict(), dict(), dict(), dict(), 0
plt.figure()
pd_pvals = None
all_occ = list()
sort_ids = np.argsort(tomos_ntet.values())
# sort_ids = np.argsort(tomos_area.values())
for idx in sort_ids:
    # tkey = tomos_np_l4.keys()[idx]
    tkey = tomos_area.keys()[idx]
    if len(tomos_nc_sims[tkey]) <= 0:
        continue
    if True: # tomos_np_l4[tkey] > 0:
        hold_val = np.asarray(tomos_nsc[tkey], dtype=float)
        tkey_hold = os.path.split(tkey)[1].split('_')
        tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
        lst[count] =  hold_val
        lst_x[count] = vesicles[tkey]
        for hold_val_sim in tomos_nsc_sims[tkey]:
            try:
                lst_sim[lst_x[count]].append(hold_val_sim)
            except KeyError:
                lst_sim[lst_x[count]] = list()
                lst_sim[lst_x[count]].append(hold_val_sim)
        lst_sim_mn[count] = np.asarray(lst_sim[lst_x[count]], dtype=np.float32).mean()
        count += 1
plt.boxplot(lst_sim.values(), positions=lst_sim.keys(), showfliers=False, notch=True, whis=[5, 95])
plt.plot(lst_x.values(), lst.values(), color='k', marker='*', linestyle='')
l_ves = np.asarray(lst_x.values(), dtype=np.int).reshape(-1, 1)
l_col = np.asarray(lst.values(), dtype=np.int).reshape(-1, 1)
l_col_sim = np.asarray(lst_sim_mn.values(), dtype=np.int).reshape(-1, 1)
regr_0, regr_1 = linear_model.LinearRegression(fit_intercept=False), linear_model.LinearRegression(fit_intercept=False)
regr_0.fit(l_ves, l_col), regr_1.fit(l_ves, l_col_sim)
l_col_r, l_col_sim_r = regr_0.predict(l_ves), regr_1.predict(l_ves)
plt.plot(l_ves, l_col_r, color='black', label='EXPERIMENTAL', linestyle='-', linewidth=2.0)
plt.plot(l_ves, l_col_sim_r, color='black', label='SIMULATED', linestyle=':', linewidth=2.0)
# plt.xticks(lst_x.values())
plt.grid(True, alpha=0.5, axis='y')
plt.ylabel('Number of sub-columns')
plt.xlabel('Number of SV-tethered')
plt.legend(loc=2)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    plt.savefig(out_tomos_dir + '/box_scol_sv_tethered.png', dpi=300)
plt.close()

print '\tTOTAL PLOTTING: '

out_lists_dir = out_stem_dir + '/lists'
os.makedirs(out_lists_dir)

print '\t\t-Gathering tomogram simulations: '
tot_nc, tot_nsc, tot_nc12, tot_nc13, tot_vol, tot_ves, tot_teth = 0, 0, 0, 0, 0, 0, 0
tot_occ, tot_nc_a12, tot_areas = 0, 0, 0
tot_np_l1, tot_np_l2, tot_np_l3 = 0, 0, 0
tot_npc_l1, tot_npc_l2, tot_npc_l3 = 0, 0, 0
ncs_sims = np.zeros(shape=p_nsims, dtype=np.float)
nsc_sims = np.zeros(shape=p_nsims, dtype=np.float)
den_sims = np.zeros(shape=p_nsims, dtype=np.float)
denv_sims = np.zeros(shape=p_nsims, dtype=np.float)
dent_sims = np.zeros(shape=p_nsims, dtype=np.float)
occ_sims = np.zeros(shape=p_nsims, dtype=np.float)
npc_l1_sims = np.zeros(shape=p_nsims, dtype=np.float)
npc_l2_sims = np.zeros(shape=p_nsims, dtype=np.float)
npc_l3_sims = np.zeros(shape=p_nsims, dtype=np.float)
for tkey, nc_sim in zip(tomos_nc_sims.iterkeys(), tomos_nc_sims.itervalues()):
    if len(nc_sim) <= 0:
        continue
    for i in range(p_nsims):
        ncs_sims[i] += nc_sim[i]
        nsc_sims[i] += tomos_nsc_sims[tkey][i]
        occ_sims[i] += (tomos_occ_sims[tkey][i] * tomos_area[tkey])
        npc_l1_sims[i] += tomos_npc_l1_sim[tkey][i]
        npc_l2_sims[i] += tomos_npc_l2_sim[tkey][i]
        npc_l3_sims[i] += tomos_npc_l3_sim[tkey][i]
    tot_vol += vols[tkey]
    tot_ves += vesicles[tkey]
    tot_teth += tomos_ntet[tkey]
    tot_nc += tomos_nc[tkey]
    tot_nsc += tomos_nsc[tkey]
    tot_occ += (tomos_occ[tkey] * tomos_area[tkey])
    tot_areas += tomos_area[tkey]
    tot_nc_a12 += tomos_nc_a12[tkey]
    tot_np_l1 += tomos_np_l1[tkey]
    tot_np_l2 += tomos_np_l2[tkey]
    tot_np_l3 += tomos_np_l3[tkey]
    tot_npc_l1 += tomos_npc_l1[tkey]
    tot_npc_l2 += tomos_npc_l2[tkey]
    tot_npc_l3 += tomos_npc_l3[tkey]
if tot_areas > 0:
    tot_occ = (100. * tot_occ) / tot_areas
if tot_teth > 0:
    tot_nc_a12 = (100. * tot_nc_a12) / tot_teth
for i in range(p_nsims):
    if tot_vol > 0:
        den_sims[i] = ncs_sims[i] / tot_vol
    if tot_ves > 0:
        denv_sims[i] = ncs_sims[i] / tot_ves
    if tot_teth > 0:
        dent_sims[i] = ncs_sims[i] / tot_teth
    if tot_areas > 0:
        occ_sims[i] = (100. * occ_sims[i]) / tot_areas

print '\t\t-Plotting the number of columns...'
plt.figure()
plt.ylabel('# columns')
# plt.xlabel('Total columns in the dataset')
plt.bar(1, tot_nc, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_sims, p_per)
ic_med = np.percentile(ncs_sims, 50)
ic_high = np.percentile(ncs_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
plt.xlim(0.5, 2.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/nc.png')
plt.close()

print '\t\t-Plotting the number of sub-columns...'
plt.figure()
plt.ylabel('# sub-columns')
# plt.xlabel('Total columns in the dataset')
plt.bar(1, tot_nsc, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(nsc_sims, p_per)
ic_med = np.percentile(nsc_sims, 50)
ic_high = np.percentile(nsc_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
plt.xlim(0.5, 2.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/nsc.png')
plt.close()

print '\t\t-Plotting columns density by synaptic vesicles..'
tot_denv = tot_nc / tot_ves
plt.figure()
plt.ylabel('Sub-column density [Scol/SV]')
# plt.xlabel('Column probability per synaptic vesicle')
plt.bar(1, tot_denv, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_sims, p_per)
ic_med = np.percentile(denv_sims, 50)
ic_high = np.percentile(denv_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
plt.xlim(0.5, 2.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/denv.png')
plt.close()

print '\t\t-Plotting area occupancy...'
plt.figure()
plt.ylabel('% of columns surface occupancy')
# plt.xlabel('Column probability per tether')
plt.bar(1, tot_occ, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(occ_sims, p_per)
ic_med = np.percentile(occ_sims, 50)
ic_high = np.percentile(occ_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
plt.xlim(0.5, 2.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/occ.png')
plt.close()

print '\t\t-Plotting fraction of particles within columns for layer 1...'
plt.figure()
plt.ylabel('% of particles within columns')
# plt.xlabel('Column probability per tether')
hold_val = 100. * (tot_npc_l1 / float(tot_np_l1))
hold_sims = 100. * (np.asarray(npc_l1_sims) / float(tot_np_l1))
ic_low = np.percentile(hold_sims, p_per)
ic_med = np.percentile(hold_sims, 50)
ic_high = np.percentile(hold_sims, 100 - p_per)
plt.bar(1, hold_val, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(hold_sims, p_per)
ic_med = np.percentile(hold_sims, 50)
ic_high = np.percentile(hold_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
plt.xlim(0.5, 2.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/np_1.png')
plt.close()

print '\t\t-Plotting fraction of particles within columns for layer 2...'
plt.figure()
plt.ylabel('% of particles within columns')
# plt.xlabel('Column probability per tether')
hold_val = 100. * (tot_npc_l2 / float(tot_np_l2))
hold_sims = 100. * (np.asarray(npc_l2_sims) / float(tot_np_l2))
ic_low = np.percentile(hold_sims, p_per)
ic_med = np.percentile(hold_sims, 50)
ic_high = np.percentile(hold_sims, 100 - p_per)
plt.bar(1, hold_val, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(hold_sims, p_per)
ic_med = np.percentile(hold_sims, 50)
ic_high = np.percentile(hold_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
plt.xlim(0.5, 2.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/np_2.png')
plt.close()

print '\t\t-Plotting fraction of particles within columns for layer 3...'
plt.figure()
plt.ylabel('% of particles within columns')
# plt.xlabel('Column probability per tether')
hold_val = 100. * (tot_npc_l3 / float(tot_np_l3))
hold_sims = 100. * (np.asarray(npc_l3_sims) / float(tot_np_l3))
ic_low = np.percentile(hold_sims, p_per)
ic_med = np.percentile(hold_sims, 50)
ic_high = np.percentile(hold_sims, 100 - p_per)
plt.bar(1, hold_val, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(hold_sims, p_per)
ic_med = np.percentile(hold_sims, 50)
ic_high = np.percentile(hold_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2), ('EXPERIMENTAL', 'SIMULATED'))
plt.xlim(0.5, 2.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_lists_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/np_3.png')
plt.close()

print '\tCTRL vs STIM PLOTTING: '

out_cs_dir = out_stem_dir + '/ctrl_vs_stim'
os.makedirs(out_cs_dir)

print '\t\t-Gathering tomogram simulations: '
tot_ctrl_nc, tot_ctrl_nc12, tot_ctrl_nc13, tot_ctrl_vol, tot_ctrl_ves, tot_ctrl_teth = 0, 0, 0, 0, 0, 0
tot_stim_nc, tot_stim_nc12, tot_stim_nc13, tot_stim_vol, tot_stim_ves, tot_stim_teth = 0, 0, 0, 0, 0, 0
tot_ctrl_occ, tot_stim_occ, nc_a12_ctrl, nc_a12_stim, tot_ctrl_areas, tot_stim_areas = 0, 0, 0, 0, 0, 0
ncs_ctrl_sims, ncs_stim_sims = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
den_ctrl_sims, den_stim_sims = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
denv_ctrl_sims, denv_stim_sims = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
dent_ctrl_sims, dent_stim_sims = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
occ_ctrl_sims, occ_stim_sims = np.zeros(shape=p_nsims, dtype=np.float), np.zeros(shape=p_nsims, dtype=np.float)
for tkey, nc_sim in zip(tomos_nc_sims.iterkeys(), tomos_nc_sims.itervalues()):
    tkey_hold = os.path.split(tkey)[1].split('_')
    tkey_stem = tkey_hold[1] + '_' + tkey_hold[2]
    if len(nc_sim) <= 0:
        continue
    if tkey_stem in ctrl_stems:
        for i in range(p_nsims):
            ncs_ctrl_sims[i] += nc_sim[i]
            occ_ctrl_sims[i] += (tomos_occ_sims[tkey][i] * tomos_area[tkey])
        tot_ctrl_nc += tomos_nc[tkey]
        tot_ctrl_vol += vols[tkey]
        tot_ctrl_ves += vesicles[tkey]
        tot_ctrl_teth += tomos_ntet[tkey]
        tot_ctrl_occ += (tomos_occ[tkey] * tomos_area[tkey])
        tot_ctrl_areas += tomos_area[tkey]
        nc_a12_ctrl += tomos_nc_a12[tkey]
    elif tkey_stem in stim_stems:
        for i in range(p_nsims):
            ncs_stim_sims[i] += nc_sim[i]
            occ_stim_sims[i] += (tomos_occ_sims[tkey][i] * tomos_area[tkey])
        tot_stim_nc += tomos_nc[tkey]
        tot_stim_vol += vols[tkey]
        tot_stim_ves += vesicles[tkey]
        tot_stim_teth += tomos_ntet[tkey]
        tot_stim_occ += (tomos_occ[tkey] * tomos_area[tkey])
        tot_stim_areas += tomos_area[tkey]
        nc_a12_stim += tomos_nc_a12[tkey]
if tot_ctrl_occ > 0:
    tot_ctrl_occ = (100. * tot_ctrl_occ) / tot_ctrl_areas
if tot_ctrl_teth > 0:
    nc_a12_ctrl = (100. * nc_a12_ctrl) / tot_ctrl_teth
if tot_stim_occ > 0:
    tot_stim_occ = (100. * tot_stim_occ) / tot_stim_areas
if tot_stim_teth > 0:
    nc_a12_stim = (100. * nc_a12_stim) / tot_stim_teth
for i in range(p_nsims):
    if tot_ctrl_vol > 0:
        den_ctrl_sims[i] = ncs_ctrl_sims[i] / tot_ctrl_vol
    if tot_ctrl_ves > 0:
        denv_ctrl_sims[i] = ncs_ctrl_sims[i] / tot_ctrl_ves
    if tot_ctrl_teth > 0:
        dent_ctrl_sims[i] = ncs_ctrl_sims[i] / tot_ctrl_teth
    if tot_ctrl_occ > 0:
        occ_ctrl_sims[i] = (100. * occ_ctrl_sims[i]) / tot_ctrl_areas
    if tot_stim_vol > 0:
        den_stim_sims[i] = ncs_stim_sims[i] / tot_stim_vol
    if tot_stim_ves > 0:
        denv_stim_sims[i] = ncs_stim_sims[i] / tot_stim_ves
    if tot_stim_teth > 0:
        dent_stim_sims[i] = ncs_stim_sims[i] / tot_stim_teth
    if tot_stim_occ > 0:
        occ_stim_sims[i] = (100. * occ_stim_sims[i]) / tot_stim_areas

print '\t\t-Plotting the number of columns...'
plt.figure()
plt.ylabel('# sub-columns')
plt.bar(1, tot_ctrl_nc, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_ctrl_sims, p_per)
ic_med = np.percentile(ncs_ctrl_sims, 50)
ic_high = np.percentile(ncs_ctrl_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4, tot_stim_nc, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(ncs_stim_sims, p_per)
ic_med = np.percentile(ncs_stim_sims, 50)
ic_high = np.percentile(ncs_stim_sims, 100-p_per)
plt.bar(5, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(5, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 4, 5), ('EXP', 'SIM', 'EXP+', 'SIM+'))
plt.xlim(0.5, 5.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/nsc.png')
plt.close()

print '\t\t-Plotting columns density..'
tot_ctrl_den, tot_stim_den = float(tot_ctrl_nc)/tot_ctrl_vol, float(tot_stim_nc)/tot_stim_vol
plt.figure()
plt.ylabel('Sub-column density [Scol/nm$^3$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.bar(1, tot_ctrl_den, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_ctrl_sims, p_per)
ic_med = np.percentile(den_ctrl_sims, 50)
ic_high = np.percentile(den_ctrl_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4, tot_stim_den, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(den_stim_sims, p_per)
ic_med = np.percentile(den_stim_sims, 50)
ic_high = np.percentile(den_stim_sims, 100-p_per)
plt.bar(5, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(5, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 4, 5), ('EXP', 'SIM', 'EXP+', 'SIM+'))
plt.xlim(0.5, 5.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/den.png')
plt.close()

print '\t\t-Plotting columns density by synaptic vesicles..'
tot_ctrl_denv, tot_stim_denv = float(tot_ctrl_nc)/tot_ctrl_ves, float(tot_stim_nc)/tot_stim_ves
plt.figure()
plt.ylabel('Sub-column density [Scol/SV]')
plt.bar(1, tot_ctrl_denv, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_ctrl_sims, p_per)
ic_med = np.percentile(denv_ctrl_sims, 50)
ic_high = np.percentile(denv_ctrl_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4, tot_stim_denv, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(denv_stim_sims, p_per)
ic_med = np.percentile(denv_stim_sims, 50)
ic_high = np.percentile(denv_stim_sims, 100-p_per)
plt.bar(5, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(5, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 4, 5), ('EXP', 'SIM', 'EXP+', 'SIM+'))
plt.xlim(0.5, 5.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/denv.png')
plt.close()

print '\t\t-Plotting columns density by tethers..'
tot_ctrl_dent, tot_stim_dent = float(tot_ctrl_nc)/tot_ctrl_teth, float(tot_stim_nc)/tot_stim_teth
plt.figure()
plt.ylabel('Sub-column density [Scol/Tether]')
plt.bar(1, tot_ctrl_dent, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_ctrl_sims, p_per)
ic_med = np.percentile(dent_ctrl_sims, 50)
ic_high = np.percentile(dent_ctrl_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4, tot_stim_dent, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(dent_stim_sims, p_per)
ic_med = np.percentile(dent_stim_sims, 50)
ic_high = np.percentile(dent_stim_sims, 100-p_per)
plt.bar(5, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(5, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 4, 5), ('EXP', 'SIM', 'EXP+', 'SIM+'))
plt.xlim(0.5, 5.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/dent.png')
plt.close()

print '\t\t-Plotting columns density by tethers..'
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.set_ylabel('Sub-columns per SV')
ax2.set_ylabel('Sub-columns per tether')
ax.bar(1, tot_ctrl_denv, BAR_WIDTH, color='b', linewidth=2)
ax2.bar(2, tot_ctrl_dent, BAR_WIDTH, color='k', linewidth=2)
ax.bar(4, tot_stim_denv, BAR_WIDTH, color='b', linewidth=2)
ax2.bar(5, tot_stim_dent, BAR_WIDTH, color='k', linewidth=2)
# ax.set_xticks((1.5, 3.5), ('CTRL', 'STIM'))
# ax.set_xlim(0.5, 5.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/denv_vs_dent_ctrl_vs_stim.png')
plt.close()

print '\t\t-Plotting % of surface occupancy..'
plt.figure()
plt.ylabel('% of sub-columns surface occupancy')
plt.bar(1, tot_ctrl_occ, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(occ_ctrl_sims, p_per)
ic_med = np.percentile(occ_ctrl_sims, 50)
ic_high = np.percentile(occ_ctrl_sims, 100-p_per)
plt.bar(2, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(2, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.bar(4, tot_stim_occ, BAR_WIDTH, color='blue', linewidth=2)
ic_low = np.percentile(occ_stim_sims, p_per)
ic_med = np.percentile(occ_stim_sims, 50)
ic_high = np.percentile(occ_stim_sims, 100-p_per)
plt.bar(5, ic_med, BAR_WIDTH, color='gray', linewidth=2)
plt.errorbar(5, ic_med, yerr=np.asarray([[ic_med - ic_low, ic_high - ic_med], ]).reshape(2, 1),
             ecolor='k', elinewidth=4, capthick=4, capsize=8)
plt.xticks((1, 2, 4, 5), ('EXP', 'SIM', 'EXP+', 'SIM+'))
plt.xlim(0.5, 5.5)
plt.tight_layout()
if fig_fmt is None:
    plt.show(block=True)
else:
    hold_dir = out_cs_dir + '/col'
    if not os.path.exists(hold_dir):
        os.makedirs(hold_dir)
    plt.savefig(hold_dir + '/occ.png')
plt.close()

print '\tTAB PRINTING: '
out_csv = out_dir + '/' + out_stem + '.csv'
with open(out_csv, 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    header_row = ['Synapse', 'SV-tether centroids', 'Sub-columns', 'Columns', 'Area columns', 'Area total',
                  '#tether-centroids', '#pre-cleft', '#pst-cleft', '#ampar', '#nmdar', 'p-vals']
    csv_writer.writerow(header_row)
    for tkey, row in zip(tab.iterkeys(), tab.itervalues()):
        try:
            row.append(tab_p[tkey])
        except KeyError:
            continue
            # row += list(np.zeros(shape=len(header_row)))
        row.insert(0, os.path.split(tkey)[1])
        print '\t\t' + str(row)
        # print '\t\t' + tkey + ': ' + str(row)
        csv_writer.writerow(row)
print '\t Storing the TAB in the file: ' + out_csv