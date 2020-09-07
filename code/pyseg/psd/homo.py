"""

    Script for applying statistical test to a pattern so as to check its heterogeneity

    Input:  - Path to the input UniStat objecto of the pattern
	    - Parameters for setting the statistical tests

    Output: - Plot graphs for testing heterogeinity

"""

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

DEBUG = False

ROOT_PATH = '/home/martinez/pool/pool-lucic2/antonio/workspace/psd_an'

# Input pickle
in_pkl = ROOT_PATH + '/ex/syn/slices/profile_mb/syn_14_15_bin2_rot_crop2_pre_unisp.pkl'

####### Output data

output_dir = ROOT_PATH + '/ex/syn/analysis/homo/syn_14_15/mb/pre'
prefix_name = 'syn_14_15_profile_mb_pre' # Stem for stored files

####### Display options

legend = True
block_plots = False

####### Analysis options

scales = [5, 10, 20, 30, 50, 75, 100] # list with the scales for W test

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import pyseg as ps


########## Global variables

########## Print initial message

print('Homegeinity tests.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput file: ' + str(in_pkl))
print('\tOutput directory: ' + str(output_dir))
print('\tOutput prefix: ' + str(prefix_name))
if block_plots:
    print('\tPlots display activated.')
print('')

######### Process

print('Loading UniStat pickle...')
uni = ps.factory.unpickle_obj(in_pkl)
f_path, f_name = os.path.split(in_pkl)
print('\tFile load with ' + str(uni.get_n_points()) + ' points.')

print('Homogeneity test W for scales: ' + str(scales))
uni.plot_points(block=block_plots, out_file=output_dir+'/'+prefix_name+'_'+uni.get_name()+'_pts.png')
uni.analyze_W(scales, block=block_plots, out_file=output_dir+'/'+prefix_name+'_'+uni.get_name()+'_W.png',
              legend=legend, pimgs=True)

print('Terminated. (' + time.strftime("%c") + ')')
