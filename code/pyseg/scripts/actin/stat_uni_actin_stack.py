"""

    Script for doing univariate point spatial analysis on stacks of actin filaments

    Input:  - Path to the input tomogram with filament cross-correlation map

    Output: - StackPlotUni object pickled
            - Figures with the analysis

"""

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

DEBUG = True

ROOT_PATH = '/home/martinez/pool/pool-lucic2/antonio/tomograms/marion/Clusters'

# Input tomograms
in_tomos = (ROOT_PATH + '/graphs/20151116_W2t1_g1_pts.mrc',
           )

in_ths = ((.52, .56),
          )

####### Output data

output_dir = ROOT_PATH + '/stat'

####### Display options

block_plots = False
legend = True
cbar = True
plt_3d_mode = 'img'

####### Tomogram pre-processing

pre = False # True if input tomograms are cross-correlation maps
res = 1.684
axis = 2
purge_ratio_3d = 1
purge_ratio_2d = 5
min_n_samp = 500

####### Mask processing

b_mask = True
mn_int = .0002
h_freq = 7 # cut off frequency
h_3d = True

####### Analysis options

stack_3d = True
stack_2d = True

max_d_1 = 50 # nm maximum distance first order
max_d_2 = 250 # nm maximum distance second order
n_samp_1 = 30 # number of samples for first order
per = 5 #% percentiles for the random simulations
n_sim_1 = 100 # Number of simulations for first order
n_samp_f = 1000 # Number of samples for F
n_sim_2 = 20 # Number of simulations for second order
n_samp_2 = 50 # Number of samples for second order
w_o = 1 # Ring thickness for O-ring
tcsr = True # Plot 2D unbounded CSR as reference

########## Global variables

import os
import time
import pyseg as ps
import numpy as np
from pyseg.spatial.stack import TomoUni
from scipy.ndimage.morphology import binary_dilation

########## Print initial message

print 'Point based spatial analysis of an acting filaments network.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput file(s): ' + str(in_tomos)
if pre:
    print '\tInput thresholds: ' + str(in_ths)
print '\tOutput directory: ' + str(output_dir)
print '\tStack axis: ' + str(axis)
print '\tResolution: ' + str(res) + ' nm'
if block_plots:
    print '\tPlots display activated.'
if min_n_samp is None:
    print '\tMinimum number of points set automatically.'
else:
    print '\tMinimum number of points: ' + str(min_n_samp)
if b_mask:
    print '\tMask pre-precessing activated:'
    print '\t\t-Scale: ' + str(h_freq) + ' nm'
    if h_3d:
        print '\t\t-Mode 3D activated'
        print '\t\t-Minimum intensity: ' + str(mn_int) + ' pts/nm^3'
    else:
        print '\t\t-Mode 2D activated'
        print '\t\t-Minimum intensity: ' + str(mn_int) + ' pts/nm^2'
if stack_2d:
    print '\tAnalysis 2D:'
    print '\t\tPurging ratio: ' + str(purge_ratio_2d)
    if legend:
        print '\t\tLegend storing activated.'
    if tcsr:
        print '\t\tPlot unbounded reference activated'
    print '\t\tFirst order analysis: '
    print '\t\t\t-Maximum distance: ' + str(max_d_1) + ' nm'
    print '\t\t\t-Number of samples: ' + str(n_samp_1)
    print '\t\t\t-Number of samples for F: ' + str(n_samp_f)
    if n_sim_1 > 1:
        print '\t\t\t-Number of simulations: ' + str(n_sim_1)
        print '\t\t\t-Percentile: ' + str(per) + ' %'
    print '\t\tSecond order analysis: '
    print '\t\t\t-Maximum distance: ' + str(max_d_2) + ' nm'
    print '\t\t\t-Number of samples: ' + str(n_samp_2)
    print '\t\t\t-Ring size: ' + str(w_o) + ' nm'
    if n_sim_2 > 1:
        print '\t\t\t-Number of simulations: ' + str(n_sim_2)
        print '\t\t\t-Percentile: ' + str(per) + ' %'
if stack_3d:
    print '\tAnalysis 3D:'
    print '\t\tPurging ratio: ' + str(purge_ratio_3d)
    if cbar:
        print '\t\tColorbar storing activated'
    print '\t\tFirst order analysis: '
    print '\t\t\t-Maximum distance: ' + str(max_d_1) + ' nm'
    print '\t\t\t-Number of samples: ' + str(n_samp_1)
    print '\t\t\t-Number of samples for F: ' + str(n_samp_f)
    print '\t\tSecond order analysis: '
    print '\t\t\t-Maximum distance: ' + str(max_d_2) + ' nm'
    print '\t\t\t-Number of samples: ' + str(n_samp_2)
    print '\t\t\t-Ring size: ' + str(w_o) + ' nm'
print ''

######### Process

print '\tMAIN LOOP (tomograms):'
for (in_tomo, th) in zip(in_tomos, in_ths):

    f_path, f_name = os.path.split(in_tomo)
    f_stem, f_ext = os.path.splitext(f_name)
    print '\t\tLoading tomogram: ' + f_name
    tomo = ps.disperse_io.load_tomo(in_tomo)

    print '\t\tPre-processing the tomogram...'
    if pre:
        tomo_bin = (tomo >= th[0]) & (tomo <= th[1])
        tomo_bin = binary_dilation(tomo_bin, structure=None, iterations=1, mask=mask)
        mask = tomo > 0
    else:
        tomo_bin = tomo > 0
        mask = np.ones(shape=tomo_bin.shape, dtype=np.bool)
    out_dir = output_dir + '/' + f_stem
    os.system('mkdir ' + out_dir)

    if DEBUG:
        ps.disperse_io.save_numpy(tomo_bin, out_dir+'/hold_tomo.mrc')
        ps.disperse_io.save_numpy(mask, out_dir+'/hold_mask.mrc')

    if stack_2d and (purge_ratio_2d > 1):
        print '\t\tANALYSIS 2D:'

        print '\t\t\tPurging images in the input stack with ratio ' + str(purge_ratio_2d)
        tomo_p2d, mask_p2d, sp_2d = ps.spatial.stack.purge_stack(tomo_bin, mask, axis, purge_ratio_2d)

        print '\t\t\tBuilding the stack...'
        suni = TomoUni(tomo_p2d, mask=mask, res=res, spacing=sp_2d*res, axis=axis, name=f_stem, pre=pre)

        if b_mask:
            if h_3d:
                print '\t\t\tBuilding mask from homogeneity test (frequency=' + str(h_freq) + \
                      '), minimum intensity level: ' + str(mn_int) + ' pts/nm^3'
            else:
                print '\t\t\tBuilding mask from homogeneity test (frequency=' + str(h_freq) + \
                      '), minimum intensity level: ' + str(mn_int) + ' pts/nm^2'
            homo = suni.homo_stack(freq=h_freq, mode_3d=h_3d)
            ps.disperse_io.save_numpy(homo, output_dir+'/homo.mrc')
            homo = homo > mn_int
            ps.disperse_io.save_numpy(homo, output_dir+'/homo_th_'+str(mn_int)+'.mrc')
            suni.set_mask(homo)

        print '\t\t\tBuilding the analyzer...'
        spuni = suni.generate_PlotUni()

        print '\t\t\tNumber of point in the stack images: ' + str(suni.list_n_points())

        if min_n_samp is None:
            min_n_samp = n_samp_1
            if n_samp_2 < n_samp_1:
                min_n_samp = n_samp_2
        print '\t\t\tPurging images with a number of points lower than ' + str(min_n_samp)
        spuni.purge_unis(min_n_samp)

        if DEBUG:
            ps.disperse_io.save_numpy(suni.generate_tomo_stack(), out_dir+'/hold_stack_2d.mrc')

        print '\t\t\tAnalysis:'
        print '\t\t\t\t-Intensity'
        spuni.analyze_intensity(block=block_plots, out_file=out_dir+'/intensity_2D.png')
        print '\t\t\t\t-G'
        spuni.analyze_G(max_d_1, n_samp_1, n_sim_1, per, block=block_plots, out_file=out_dir+'/G_2D.png',
                        legend=legend)
        print '\t\t\t\t-F'
        spuni.analyze_F(max_d_1, n_samp_1, n_samp_f, n_sim_1, per, block=block_plots, out_file=out_dir+'/F_2D.png')
        print '\t\t\t\t-J'
        spuni.analyze_J(block=block_plots, out_file=out_dir+'/J_2D.png')
        print '\t\t\t\t-K'
        spuni.analyze_K(max_d_2, n_samp_2, n_sim_2, per, tcsr=tcsr, block=block_plots, out_file=out_dir+'/K_2D.png')
        print '\t\t\t\t-L'
        spuni.analyze_L(block=block_plots, out_file=out_dir+'/L_2D.png')
        print '\t\t\t\t-O'
        spuni.analyze_O(w_o, block=block_plots, out_file=out_dir+'/O_2D.png')

    if stack_3d and (purge_ratio_3d > 0):
        print '\t\tANALYSIS 3D:'

        print '\t\t\tPurging images in the input stack with ratio ' + str(purge_ratio_3d)
        tomo_p3d, mask_p3d, sp_3d = ps.spatial.stack.purge_stack(tomo_bin, mask, axis, purge_ratio_3d)

        print '\t\t\tBuilding the stack...'
        suni = TomoUni(tomo_p3d, mask=mask, res=res, spacing=sp_3d*res, axis=axis, name=f_stem, pre=pre)

        if DEBUG:
            ps.disperse_io.save_numpy(suni.generate_tomo_stack(), out_dir+'/hold_stack_3d.mrc')

        if b_mask:
            if h_3d:
                print '\t\t\tBuilding mask from homogeneity test (frequency=' + str(h_freq) + \
                      '), minimum intensity level: ' + str(mn_int) + ' pts/nm^3'
            else:
                print '\t\t\tBuilding mask from homogeneity test (frequency=' + str(h_freq) + \
                      '), minimum intensity level: ' + str(mn_int) + ' pts/nm^2'
            homo = suni.homo_stack(freq=h_freq, mode_3d=h_3d)
            ps.disperse_io.save_numpy(homo, out_dir+'/homo.mrc')
            homo = (homo > mn_int) * suni.get_mask_stack()
            ps.disperse_io.save_numpy(homo, out_dir+'/homo_th_'+str(mn_int)+'.mrc')
            suni.set_mask(homo)

        print '\t\t\tBuilding the analyzer...'
        spuni = suni.generate_PlotUni()

        if min_n_samp is None:
            min_n_samp = n_samp_1
            if n_samp_2 < n_samp_1:
                min_n_samp = n_samp_2
        print '\t\t\tPurging images with a number of points lower than ' + str(min_n_samp)
        spuni.purge_unis(min_n_samp)

        print '\t\t\tAnalysis:'

        print '\t\t\t\t-Intensity'
        spuni.analyze_intensity(block=block_plots, out_file=out_dir+'/intensity_3D.png')

        print '\t\t\t\t-G'
        spuni.analyze_stack_G(max_d_1, n_samp_2, block=block_plots, out_file=out_dir+'/G_3D.png',
                              bar=cbar, mode=plt_3d_mode)

        print '\t\t\t\t-K'
        spuni.analyze_stack_K(max_d_2, n_samp_2, block=block_plots, out_file=out_dir+'/K_3D.png',
                              bar=cbar, mode=plt_3d_mode)

        print '\t\t\t\t-L'
        spuni.analyze_stack_L(block=block_plots, out_file=out_dir+'/L_3D.png',
                              bar=cbar, mode=plt_3d_mode)

        print '\t\t\t\t-O'
        spuni.analyze_stack_O(w=w_o, block=block_plots, out_file=out_dir+'/O_3D.png',
                              bar=cbar, mode=plt_3d_mode)

print 'Terminated. (' + time.strftime("%c") + ')'

