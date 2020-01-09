"""

    Script for plotting bivariate statistical analysis of pattern pairs

    Input:  - Path to the input UniStat pickles for two set of patterns 
	    - Parameters for the bivariate analysis

    Output: - Plot graphs with the statistical bivariate analysis of every pair

"""

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

DEBUG = False

ROOT_PATH = '/home/martinez/pool/pool-lucic2/antonio/workspace/psd_an'

# Input pickles
in_pkls_a = (# ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_11_2_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_11_5_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_11_6_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_11_9_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_13_1_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_13_3_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_9_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_13_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_14_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_15_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
           )

in_pkls_b = (# ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_11_2_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_11_5_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_11_6_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_11_9_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_13_1_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_13_3_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_14_9_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_14_13_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_14_14_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
	   ROOT_PATH + '/ex/syn/fils/sub/pre_1/pre_cont/syn_14_15_bin2_rot_crop2_fil_pre_mb_to_fil_pre_net_pre_cont_unisp.pkl',
           )

in_del_coords = (0,
                 0,
                 0,
                 0,
                 1,
                 0,
                 1,
                 0,
                 0,
                 0,)

####### Output data

output_dir = ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont_to_pre_1_pre_cont/stat'
prefix_name = 'pst_1_pst_cont_to_pre_1_pre_cont'

####### Display options

block_plots = False

####### Analysis options

gather = True
do_ana_1 = False
do_ana_2 = True
max_d_1 = 20 # nm maximum distance first order
max_d_2 = 50 # nm maximum distance first order
n_samp_1 = 10 # number of samples for first order
per = 5 #% percentiles for the random simulations
n_sim_1 = 20 # Number of simulations for first order
n_sim_2 = 20 # Number of simulations for second order
n_samp_2 = 30 # Number of samples for second order
w_o = 1 # Ring thickness for O-ring

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import pyseg as ps
from pyseg.spatial import PlotBi, BiStat

########## Global variables

########## Print initial message

print 'Plotting bivarite analysis.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput file(s) set A: ' + str(in_pkls_a)
print '\tInput file(s) set B: ' + str(in_pkls_b)
print '\tCoordinate(s) to delete?: ' + str(in_del_coords)
print '\tOutput directory: ' + str(output_dir)
print '\tOutput prefix: ' + str(prefix_name)
if block_plots:
    print '\tPlots display activated.'
print '\tCSR simulator:'
if do_ana_1 or do_ana_2:
    if gather:
        print '\tAnalysis (composed):'
    else:
        print '\tAnalysis (isolated):'
    print '\t\t-Percentile for random simulations: ' + str(per) + ' %'
if do_ana_1:
    print '\t\t-First order metrics (G, F and J):'
    print '\t\t\t-Maximum scale: ' + str(max_d_1) + ' nm'
    print '\t\t\t-Number of bins: ' + str(n_samp_1)
    print '\t\t\t-Number of simulations: ' + str(n_sim_1)
if do_ana_2:
    print '\t\t-Second order metrics (K and O):'
    print '\t\t\t-Maximum scale: ' + str(max_d_2)
    print '\t\t\t-Number of simulations: ' + str(n_sim_2)
    print '\t\t\t-Number of samples: ' + str(n_samp_2)
    print '\t\t\t-Ring shell for O: ' + str(w_o) + ' voxels'
print ''

######### Process

print 'Loading UniStat from input pickles...'
pbi = PlotBi(prefix_name)
for pkl_a, pkl_b, del_coord in zip(in_pkls_a, in_pkls_b, in_del_coords):
    uni_a = ps.factory.unpickle_obj(pkl_a)
    f_path_a, f_name_a = os.path.split(pkl_a)
    print '\tFile set A ' + f_name_a + ' load with ' + str(uni_a.get_n_points()) + ' points.'
    if del_coord is not None:
        print '\t\t-Comprising object to 2D by deleting coord ' + str(del_coord) + '...'
        uni_a.to_2D(del_coord)

    if pkl_a == pkl_b:
        print '\t\t-WARNING: A and B files are the same, jumping...'
        continue

    uni_b = ps.factory.unpickle_obj(pkl_b)
    f_path_b, f_name_b = os.path.split(pkl_b)
    print '\t\tFile set B ' + f_name_b + ' load with ' + str(uni_b.get_n_points()) + ' points.'
    if del_coord is not None:
        print '\t\t-Comprising object to 2D by deleting coord ' + str(del_coord) + '...'
        uni_b.to_2D(del_coord)

    print '\t\tParing patterns A and B...'
    res = uni_a.get_resolution()
    if res != uni_b.get_resolution():
        print '\t\tWARNING: Resolution of A (' + str(res) + ' nm) is not equal to B (' \
                  + str(uni_b.get_resolution()) + ' nm)'
        print '\t\t\tPair jumped'
        continue
    bi_name = uni_a.get_name()+'_vs_'+uni_b.get_name()
    bi = BiStat(uni_a.get_coords(), uni_b.get_coords(),
                uni_a.get_mask() * uni_b.get_mask(),
                res, name=bi_name)
    pbi.insert_bi(bi)

    print '\t\t\t-Storing points...'
    out_pts_dir = output_dir + '/' + prefix_name
    if not os.path.exists(out_pts_dir):
        os.makedirs(out_pts_dir)
    pbi.save_points(out_pts_dir)

output_prefix = output_dir + '/' + prefix_name
print '\t\tAnalysis:'
print '\t\t\t-G'
pbi.analyze_G(max_d_1, n_samp_1, n_sim_1, per, block=block_plots, out_file=output_prefix+'_G.png', legend=True)
print '\t\t\t-K'
pbi.analyze_K(max_d_2, n_samp_2, n_sim_2, per, block=False)
print '\t\t\t-L'
pbi.analyze_L(block=block_plots, out_file=output_prefix+'_L.png', legend=False, p=per)
print '\t\t\t-O'
pbi.analyze_O(w_o, block=block_plots, out_file=output_prefix+'_O.png', p=per)

print 'Terminated. (' + time.strftime("%c") + ')'
