"""

    Script for plotting clouds from different groups

    Input:  - Path to the input SetCloudP pickles

    Output: - Plot graphs with the statistical comparison
            - Store a CloudPlotter object in pickle file

"""

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

DEBUG = True

ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1'

# Input pickles
in_pkls = (ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_9_bin2_rot_crop2_fil_psd_to_pst_2_in_net_pre_cont_unisp.pkl',
           ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_9_bin2_rot_crop2_fil_psd_to_pst_2_in_net_pre_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_11_5_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_11_6_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_11_9_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_13_1_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_13_3_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_9_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_13_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_14_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
	   # ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/syn_14_15_bin2_rot_crop2_fil_pst_mb_to_fil_pst_net_pst_cont_unisp.pkl',
           )

####### Output data

output_dir = ROOT_PATH + '/ex/syn/fils/sub/pst_1/pst_cont/stat_test'
prefix_name = 'pst_cont_uni' # Stem for stored files

####### Display options

block_plots = True

####### Analysis options

gather = True
gstd = True
do_ana_1 = False
do_ana_2 = True
max_d_1 = 30 # nm maximum distance first order
max_d_2 = 80 # nm maximum distance first order
n_samp_1 = 20 # number of samples for first order
per = 5 #% percentiles for the random simulations
n_sim_1 = 20 # Number of simulations for first order
n_samp_f = 1000 # Number of samples for F
n_sim_2 = 3 # 500 # Number of simulations for second order
n_samp_2 = 80 # Number of samples for second order
w_o = 1 # Ring thickness for O-ring
tcsr = True # Plot 2D unbounded CSR as reference

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import gc
import os
import time
import pyseg as ps
from pyseg.spatial import PlotUni


########## Global variables

########## Print initial message

print('Univariate Spatial 3D distribution.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput pickles lists: ' + str(in_pkls))
print('\tOutput directory: ' + output_dir)
print('\tTransformation properties (from reference to holding tomogram):')
print('\tCSR simulator:')
if do_ana_1 or do_ana_2:
    if gather:
        print('\tAnalysis (composed):')
    else:
        print('\tAnalysis (isolated):')
    print('\t\t-Percentile for random simulations: ' + str(per) + ' %')
if do_ana_1:
    print('\t\t-First order metrics (G, F and J):')
    print('\t\t\t-Maximum scale: ' + str(max_d_1) + ' nm')
    print('\t\t\t-Number of bins: ' + str(n_samp_1))
    print('\t\t\t-Number of simulations: ' + str(n_sim_1))
    print('\t\t\t-Number of samples for F: ' + str(n_samp_f))
if do_ana_2:
    print('\t\t-Second order metrics (K and O):')
    print('\t\t\t-Maximum scale: ' + str(max_d_2))
    print('\t\t\t-Number of simulations: ' + str(n_sim_2))
    print('\t\t\t-Number of samples: ' + str(n_samp_2))
    print('\t\t\t-Ring shell for O: ' + str(w_o) + ' voxels')
    if tcsr:
        print('\t\t\t-Plotting bounds for random simulation.')
print('')

######### Process

print('Loading clouds from input pickles...')
puni = PlotUni(prefix_name)
for pkl in in_pkls:
    uni = ps.factory.unpickle_obj(pkl)
    puni.insert_uni(uni)
    f_path, f_name = os.path.split(pkl)
    print('\tFile ' + f_name + ' load with ' + str(uni.get_n_points()) + ' points.')

    if DEBUG:
	print('\tDEBUG: Storing the patterns to analyze...')
	uni.save_sparse(output_dir+'/'+ uni.get_name()+'_pts.mrc', mask=True)
	ps.disperse_io.save_numpy(uni.get_mask(), output_dir+'/'+uni.get_name()+'_mask.mrc')
	uni.save_random_instance(output_dir+'/'+uni.get_name()+'_rnd_pts.mrc', pts=True)

    gc.collect()

output_prefix = output_dir + '/' + prefix_name
if gather:
    print('\tAnalysis (composed):')
else:
    print('\tAnalysis (isolated):')
print('\t\t\t-Points:')
out_pts_dir = output_dir + '/' + prefix_name
if not os.path.exists(out_pts_dir):
    os.makedirs(out_pts_dir)
puni.save_points(out_pts_dir, to_vtp=True)
print('\t\t\t-Intensity')
puni.analyze_intensity(block=block_plots, out_file=output_prefix+'_intensity.png')
if do_ana_1:
    print('\t\t\t-G')
    puni.analyze_G(max_d_1, n_samp_1, n_sim_1, per, block=block_plots, out_file=output_prefix+'_G.png',
                   legend=True, gather=gather)
    print('\t\t\t-F')
    puni.analyze_F(max_d_1, n_samp_1, n_samp_f, n_sim_1, per, block=block_plots, out_file=output_prefix+'_F.png',
                   gather=gather)
    print('\t\t\t-J')
    puni.analyze_J(block=block_plots, out_file=output_prefix+'_J.png', p=per)
if do_ana_2:
    print('\t\t\t-K')
    puni.analyze_K(max_d_2, n_samp_2, n_sim_2, per, tcsr=tcsr, block=block_plots, out_file=output_prefix+'_K.png',
                   gather=gather)
    print('\t\t\t-L')
    puni.analyze_L(block=block_plots, out_file=output_prefix+'_L.png', gather=gather, p=per, gstd=gstd)
    print('\t\t\t-O')
    puni.analyze_O(w_o, block=block_plots, out_file=output_prefix+'_O.png', gather=gather, p=per, gstd=gstd)

print('Terminated. (' + time.strftime("%c") + ')')
