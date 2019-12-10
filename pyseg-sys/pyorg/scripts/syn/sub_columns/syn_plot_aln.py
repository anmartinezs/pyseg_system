"""

    Script for ploting alignment confidence over scales

    Input:  - A list of workpsaces with the column alignement computation
            - A paired list with the alignment distance used

    Output: - Plottings

"""

import time
import numpy as np
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

###### Input parameters
"""
ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst/ampar_vs_nmdar/org/col_scol/'

in_wspaces = ['nmdar_tether_2_col_0_aln_5_sim_200_nn1_1_nn2_1_nn3_1/nmdar_tether_2_col_0_aln_5_sim_200_nn1_1_nn2_1_nn3_1_wspace.pkl',
              'nmdar_tether_2_col_0_aln_10_sim_200_nn1_1_nn2_1_nn3_1/nmdar_tether_2_col_0_aln_10_sim_200_nn1_1_nn2_1_nn3_1_wspace.pkl',
              'nmdar_tether_2_col_0_aln_15_sim_200_nn1_1_nn2_1_nn3_1/nmdar_tether_2_col_0_aln_15_sim_200_nn1_1_nn2_1_nn3_1_wspace.pkl',
              'nmdar_tether_2_col_0_aln_20_sim_200_nn1_1_nn2_1_nn3_1/nmdar_tether_2_col_0_aln_20_sim_200_nn1_1_nn2_1_nn3_1_wspace.pkl',
              'nmdar_tether_2_col_0_aln_25_sim_200_nn1_1_nn2_1_nn3_1/nmdar_tether_2_col_0_aln_25_sim_200_nn1_1_nn2_1_nn3_1_wspace.pkl',
              'nmdar_tether_2_col_0_aln_30_sim_200_nn1_1_nn2_1_nn3_1/nmdar_tether_2_col_0_aln_30_sim_200_nn1_1_nn2_1_nn3_1_wspace.pkl',
              'nmdar_tether_2_col_0_aln_35_sim_200_nn1_1_nn2_1_nn3_1/nmdar_tether_2_col_0_aln_35_sim_200_nn1_1_nn2_1_nn3_1_wspace.pkl']
"""

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst/ampar_vs_nmdar/org/method_1'

in_wspaces = ['/3_pre_az_pst/aln_5_cr10/3_pre_az_pst_sim200_wspace.pkl',
              '/3_pre_az_pst/aln_10_cr20/3_pre_az_pst_sim200_wspace.pkl',
              '/3_pre_az_pst/aln_15_cr30/3_pre_az_pst_sim200_wspace.pkl',
              '/3_pre_az_pst/aln_20_cr40/3_pre_az_pst_sim200_wspace.pkl',
              '/3_pre_az_pst/aln_25_cr50/3_pre_az_pst_sim200_wspace.pkl',
              '/3_pre_az_pst/aln_30_cr60/3_pre_az_pst_sim200_wspace.pkl',
              '/3_pre_az_pst/aln_35_cr70/3_pre_az_pst_sim200_wspace.pkl']

in_dsts = [5,
           10,
           15,
           20,
           25,
           30,
           35]

in_lvl = 95 # %

out_fname = ROOT_PATH + '/plots/subcols_pval_existence_pre_az_pst_method_1.png'

####### Global functionality

# Compute the p-value for a single scalar in an array with the distribution
def compute_pval(val, sim, slope='high'):
    if slope == 'high':
        return float((val >= sim).sum()) / float(len(sim))
    else:
        return float((val >= sim).sum()) / float(len(sim))

####### Loading parameters and computing the p-values

dsts = np.zeros(shape=len(in_wspaces))
pvals, pvals_2 = np.zeros(shape=len(in_wspaces)), np.zeros(shape=len(in_wspaces))
for i, in_wspace in enumerate(in_wspaces):

    hold_wspace = ROOT_PATH + in_wspace
    with open(hold_wspace, 'r') as pkl:

        # Loading the values
        wspace = pickle.load(pkl)
        tomos_nc, tomos_nc_sims, tomos_nc_sims2 = wspace[39], wspace[40], wspace[41]
        nc, nc_sims, nc_sims2 = 0, np.zeros(shape=len(tomos_nc_sims.values()[0]), dtype=np.int), \
                                np.zeros(shape=len(tomos_nc_sims2.values()[0]), dtype=np.int)
        for tkey, tomo_nc in zip(tomos_nc.iterkeys(), tomos_nc.itervalues()):
            nc += tomo_nc
            for j in range(len(nc_sims)):
                nc_sims[j] += tomos_nc_sims[tkey][j]
            for j in range(len(nc_sims2)):
                nc_sims2[j] += tomos_nc_sims2[tkey][j]

        # p-values
        dsts[i] = in_dsts[i]
        pvals[i] = compute_pval(nc, nc_sims, slope='high')
        pvals_2[i] = compute_pval(nc, nc_sims2, slope='high')

####### Plotting p-values

plt.figure()
plt.plot(dsts, pvals, 'b', linewidth=2, marker='o')
plt.plot(dsts, pvals_2, 'b', linewidth=2, marker='s', linestyle='--')
plt.ylim((0, 1.1))
x_line = np.linspace(dsts[0]-5, dsts[-1]+5, 10)
plt.plot(x_line, .01*in_lvl*np.ones(shape=len(x_line)), 'k--', linewidth=2)
plt.xlim((dsts.min(), dsts.max()))
plt.xlabel('Distance [nm]')
plt.ylabel('p-value [number of sub-columns]')
plt.tight_layout()
plt.savefig(out_fname, dpi=300)
plt.close()

print 'Terminated. (' + time.strftime("%c") + ')'

