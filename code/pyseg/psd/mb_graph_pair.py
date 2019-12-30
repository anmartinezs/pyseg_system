"""

    Script for computing statistics comparing pairs of membrane slices
    VERY IMPORTANT: all cloud of points will be compressed to a 2D Euclidean space

    Input:  - Path to the membrane slices pickles which form the pairs
            - Parameters for setting the statistical analysis

    Output: - Plot graphs for points (connectors or features) distribution analysis:
                - Crossed G-Function
                - Crossed Ripley's H
                - Complemented Ripley's
            - Store a SetPairClouds object in pickle file

"""

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################


ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1'

# Input pickles
input_cloud_a = (ROOT_PATH+'/zd/pst/graph_stat/syn_14_7_bin2_sirt_rot_crop2_cont_slices.pkl',
                 ROOT_PATH+'/zd/pst/graph_stat/syn_14_9_bin2_sirt_rot_crop2_cont_slices.pkl',
                 ROOT_PATH+'/zd/pst/graph_stat/syn_14_13_bin2_sirt_rot_crop2_cont_slices.pkl',
                 ROOT_PATH+'/zd/pst/graph_stat/syn_14_14_bin2_sirt_rot_crop2_cont_slices.pkl',
                 ROOT_PATH+'/zd/pst/graph_stat/syn_14_15_bin2_sirt_rot_crop2_cont_slices.pkl',
                )
input_name_a = ('cito',
                'cito',
                'cito',
                'cito',
                'cito',
               )
input_cloud_b = (ROOT_PATH+'/zd/pst/graph_stat/syn_14_7_bin2_sirt_rot_crop2_cont_slices.pkl',
                 ROOT_PATH+'/zd/pst/graph_stat/syn_14_9_bin2_sirt_rot_crop2_cont_slices.pkl',
                 ROOT_PATH+'/zd/pst/graph_stat/syn_14_13_bin2_sirt_rot_crop2_cont_slices.pkl',
                 ROOT_PATH+'/zd/pst/graph_stat/syn_14_14_bin2_sirt_rot_crop2_cont_slices.pkl',
                 ROOT_PATH+'/zd/pst/graph_stat/syn_14_15_bin2_sirt_rot_crop2_cont_slices.pkl',
                )
input_name_b = ('cleft',
                'cleft',
                'cleft',
                'cleft',
                'cleft',
               )

####### Output data

output_dir = ROOT_PATH + '/zd/pst/pairs_stat'
sufix_name = 'pair_cito_cleft' # Stem for stored files

###### Thresholds for graph

####### Input parameters

n_samples = 80
n_sim = 200
per_h = 5 # %

### Ripley's parameters
comp_k = False
max_dist = 60 # nm
fwd = False

# Analysis

comp_ana = True
disp_figs = False
store_figs = True
cloud_over = True

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import pyseg as ps
from pyseg.spatial import SetPairClouds

########## Global variables

########## Print initial message

print 'Comparing pairs of membrane slice.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput file(s) a): ' + str(input_cloud_a)
print '\tInput file(s) b): ' + str(input_cloud_b)
print '\tInput name(s) a): ' + str(input_name_a)
print '\tInput name(s) b): ' + str(input_name_b)
print '\tOutput directory: ' + str(output_dir)
print '\tAnalysis parameters: '
print '\t\t- Number of samples: ' + str(n_samples)
print '\t\t- Number of simulations: ' + str(n_sim)
if per_h is not None:
    print '\t\t- G and F functions test percentile: ' + str(per_h) + ' %'
print '\t\t- Ripley''s H: '
print '\t\t\t-Maximum distance : ' + str(max_dist) + ' nm'
if fwd:
    print '\t\t\t-Forward crossed Ripley\'s H active'
else:
    print '\t\t\t-Backward crossed Ripley\'s H active'
print ''

######### Process

print '\tSlices loop:'
for (in_a, in_b, name_a, name_b) in zip(input_cloud_a, input_cloud_b, input_name_a, input_name_b):

    print '\t\tLoading clouds:'
    print '\t\t\ta) ' + in_a + ' - ' + name_a
    print '\t\t\tb) ' + in_b + ' - ' + name_b
    cloud_set_a = ps.factory.unpickle_obj(in_a)
    cloud_set_b = ps.factory.unpickle_obj(in_b)
    _, fname_a = os.path.split(in_a)
    stem_a, _ = os.path.splitext(fname_a)
    _, fname_b = os.path.split(in_b)
    stem_b, _ = os.path.splitext(fname_b)
    print '\tPairing clouds...'
    pair_set = SetPairClouds(cloud_set_a.get_box(), cloud_set_b.get_box(), n_samples, n_sim, max_dist, per_h,
                             fwd)
    pair_set.insert_pair(cloud_set_a.get_cloud_by_name(name_a), cloud_set_b.get_cloud_by_name(name_b),
                         name_a + '-' + name_b)

    print '\tAnalyzing...'
    pair_set.analyze(verbose=True)

    if disp_figs:
        print '\tPlotting the results for analysis (close all windows to continue)...'
        pair_set.plot(block=True, cloud_over=cloud_over)
    if store_figs:
        print '\tStoring analysis figures...'
        fig_dir = output_dir + '/' + stem_a + '_' + sufix_name + '_den'
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        pair_set.store_figs(fig_dir, cloud_over=cloud_over)
    output_pkl = output_dir + '/' + stem_b + '_' + sufix_name + '.pkl'
    print '\t\tStoring the result in file ' + output_pkl
    pair_set.pickle(output_pkl)

print 'Terminated. (' + time.strftime("%c") + ')'

