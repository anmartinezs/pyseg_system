"""

    Script for computing statistics about the profile (along different penetration values) clouds
    of points extracted from a network of filaments attached to a membrane
    VERY IMPORTANT: all cloud of points will be compressed to a 2D Euclidean space

    Input:  - Path to a membrane attached filament network
            - Parameters for thresholding the filaments
            - Parameters for setting the statistical analysis

    Output: - Plot graphs for points (connectors or features) distribution analysis:
                - Density profile
                - F-Function evolution
                - G-Function evolution
                - Ripley's H evolution
            - Store a SetClouds object in pickle file

"""

__author__ = 'Antonio Martinez-Sanchez'
import numpy as np

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1/zd/pst'

# Input pickle
input_pkl = ROOT_PATH + '/mb_fil_bin4/syn_14_7_bin4_sirt_rot_crop2_g0.5_net.pkl'
del_coord = 0

####### Output data

output_dir = ROOT_PATH + '/fil_stat'
stem_name = 'profile' # Stem for stored files

###### Thresholds for filaments

len_l, len_h, len_mode = None, None, 'simple' # nm
pen_l, pen_h, pen_mode = None, None, 'simple' # nm
pent_l, pent_h, pent_mode = 5, 10, 'simple' # nm
dst_l, dst_h, dst_mode = None, None, 'simple' # nm
fild_l, fild_h, fild_mode = None, None, 'simple'
ct_l, ct_h, ct_mode = None, None, 'simple'
mc_l, mc_h, mc_mode = None, None, 'simple'
sin_l, sin_h, sin_mode = 1, 2.5, 'simple'
smo_l, smo_h, smo_mode = None, None, 'simple'
car_l, car_h, car_mode = None, None, 'simple'
cross_th = None # 10

###### Thresholds for clusters

cnp_l, cnp_h = None, None
ca_l, ca_h = None, None # nm^2
cd_l, cd_h= None, None
cr_l, cr_h = None, None

###### Cloud features selection

side = 3 # 2-inside, 3-outside

###### Profile analysis

pent_samp = np.linspace(5, 10, num=2)

####### Input parameters

n_samples = 50
n_sim = 0

### Ripley's parameters
comp_k = False
b_mode = 2 # 0 border compensation is not active, 1 points inflation mode, 2 Goreaud
max_dist = 50 # nm
per_h = 5 # %

# Clustering

clstring = False # If True activates clustering mode
plt_clst = False
plt_cgs = False
del_mask = True

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import time
import sys
import pyseg as ps
from pyseg.globals import Threshold
from pyseg.mb.filament import ConnDom
from pyseg.mb.variables import *
from pyseg.spatial.plane import make_plane, make_plane_box

########## Global variables

########## Building the thresholds

len_th, pen_th, pent_th, dst_th, fild_th = None, None, None, None, None
ct_th, mc_th, sin_th, smo_th, car_th = None, None, None, None, None
cnp_th, ca_th, cd_th, cr_th = None, None, None, None

if (len_l is not None) or (len_h is not None):
    len_th = Threshold(len_l, len_h)
if (pen_l is not None) or (pen_h is not None):
    pen_th = Threshold(pen_l, pen_h)
if (pent_l is not None) or (pent_h is not None):
    pent_th = Threshold(pent_l, pent_h)
if (dst_l is not None) or (dst_h is not None):
    dst_th = Threshold(dst_l, dst_h)
if (fild_l is not None) or (fild_h is not None):
    fild_th = Threshold(fild_l, fild_h)
if (ct_l is not None) or (ct_h is not None):
    ct_th = Threshold(ct_l, ct_h)
if (mc_l is not None) or (mc_h is not None):
    mc_th = Threshold(mc_l, mc_h)
if (sin_l is not None) or (sin_h is not None):
    sin_th = Threshold(sin_l, sin_h)
if (smo_l is not None) or (smo_h is not None):
    smo_th = Threshold(smo_l, smo_h)
if (cnp_l is not None) or (cnp_h is not None):
    cnp_th = Threshold(cnp_l, cnp_h)
if (ca_l is not None) or (ca_h is not None):
    ca_th = Threshold(ca_l, ca_h)
if (cd_l is not None) or (cd_h is not None):
    cd_th = Threshold(cd_l, cd_h)
if (car_l is not None) or (car_h is not None):
    car_th = Threshold(car_l, car_h)
if (cr_l is not None) or (cr_h is not None):
    cr_th = Threshold(cnp_l, cr_h)

########## Print initial message

print('Spatial analysis on membrane attached filaments.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput file(s): ' + input_pkl)
print('\tDimensions to delete: ' + str(del_coord))
print('\tOutput directory: ' + str(output_dir))
print('\tFilament thresholds: ')
if len_th is not None:
    print('\t\t- Length: ' + len_th.tostring() + ', ' + len_mode)
if pen_th is not None:
    print('\t\t- Penetration: ' + pen_th.tostring() + ', ' + pen_mode)
if pent_th is not None:
    print('\t\t- Tail penetration: ' + pent_th.tostring() + ', ' + pent_mode)
if dst_th is not None:
    print('\t\t- Head-tail distance: ' + dst_th.tostring() + ', ' + dst_mode)
if fild_th is not None:
    print('\t\t- Denseness: ' + fild_th.tostring() + ', ' + fild_mode)
if ct_th is not None:
    print('\t\t- Curvature total: ' + ct_th.tostring() + ', ' + ct_mode)
if mc_th is not None:
    print('\t\t- Maximum curvature: ' + mc_th.tostring() + ', ' + mc_mode)
if sin_th is not None:
    print('\t\t- Sinuosity distance: ' + sin_th.tostring() + ', ' + sin_mode)
if smo_th is not None:
    print('\t\t- Denseness: ' + smo_th.tostring() + ', ' + smo_mode)
if car_th is not None:
    print('\t\t- Cardinality: ' + car_th.tostring() + ', ' + car_mode)
if cross_th is not None:
    print('\t\t- Crossness: ' + str(cross_th) + ' nm')
print('\tProfile samples for penetration tail: ' + str(pent_samp) + ' nm')
print('\tCloud features: ')
if side == 2:
    print('\t\t- Side A')
elif side == 3:
    print('\t\t- Side B')
else:
    sys.exit('Non valid side: ' + str(side))
print('\tClustering mode:')
print('\tCluster thresholds: ')
if cnp_th is not None:
    print('\t\t- Number of Points: ' + cnp_th.tostring())
if ca_th is not None:
    print('\t\t- Number of Points: ' + ca_th.tostring())
if cd_th is not None:
    print('\t\t- Number of Points: ' + cd_th.tostring())
if cr_th is not None:
    print('\t\t- Number of Points: ' + cr_th.tostring())
print('\tAnalysis parameters: ')
print('\t\t- Number of samples: ' + str(n_samples))
print('\t\t- Number of simulations: ' + str(n_sim))
print('\t\t- Ripley''s H: ')
print('\t\t\t-Maximum distance : ' + str(max_dist) + ' nm')
if b_mode == 0:
    print('\t\t\t-Border compensation not active')
elif b_mode == 1:
    print('\t\t\t-Border compensation by cloud inflation')
elif b_mode == 2:
    print('\t\t\t-Border compensation by Goreaud')
else:
    sys.exit('Non valid border compensation: ' + str(b_mode))
print('')

######### Process

print('\tLoading the input clouds: ')
if clstring:
    set_clouds = ps.spatial.SetClustersP(n_samples, n_sim, max_dist, b_mode)
else:
    set_clouds = ps.spatial.SetCloudsP(n_samples, n_sim, max_dist, b_mode)

print('\t\tUnpicking network...')
path, fname = os.path.split(input_pkl)
net = ps.factory.unpickle_obj(input_pkl)
if side == MB_IN_LBL:
    box = make_plane_box(net.get_graph_in().compute_bbox(), coord=del_coord) * net.get_resolution()
else:
    box = make_plane_box(net.get_graph_out().compute_bbox(), coord=del_coord) * net.get_resolution()

print('\t\tApplying thresholds to the network...')
if len_th is not None:
    net.threshold_len(len_th, len_mode)
if pen_th is not None:
    net.threshold_pen(pen_th, pen_mode)
if pent_th is not None:
    net.threshold_pent(pent_th, pent_mode)
if dst_th is not None:
    net.threshold_dst(dst_th, dst_mode)
if fild_th is not None:
    net.threshold_dness(fild_th, fild_mode)
if ct_th is not None:
    net.threshold_ct(ct_th, ct_mode)
if mc_th is not None:
    net.threshold_mc(mc_th, mc_mode)
if sin_th is not None:
    net.threshold_sin(sin_th, sin_mode)
if smo_th is not None:
    net.threshold_smo(smo_th, smo_mode)
if car_th is not None:
    net.threshold_card(car_th, car_mode)
if cross_th is not None:
    net.threshold_cdst_conts(cross_th)

den = list()
clouds = list()
for i in range(0, len(pent_samp)-1):

    slice_samp = [pent_samp[i], pent_samp[i+1]]
    print('\t\tProcessing sample: [' + str(round(slice_samp[0], 1)) + ', ' \
          + str(round(slice_samp[1], 1)) + '] nm')

    sys.stdout.write('\t\tGetting cloud from ')
    if clstring:
        sys.stdout.write('clusters of tails ')
        if side == 2:
            print('of side A...')
        else:
            print('of side B...')
        cloud, cloud_cids = net.get_cloud_clst_slice(side, slice_samp)
        cloud = make_plane(cloud, coord=del_coord) * net.get_resolution()
        clst = ConnDom(cloud, cloud_cids, np.asarray(box, dtype=np.float))
        print('\t\t\tApplying threshold to clusters...')
        clst.threshold(th_npoints=cnp_th, th_areas=ca_th, th_den=cd_th, th_round=cr_th)
        print('\t\t\tGetting clusters centers of gravity...')
        cloud = clst.get_clst_cg()
        if plt_clst:
            clst.plot_clusters(prop='npoints', centers=plt_cgs)
            clst.plot_clusters(prop='areas', centers=plt_cgs)
            clst.plot_clusters(prop='densities', centers=plt_cgs)
            clst.plot_clusters(prop='round', centers=plt_cgs)
    else:
        sys.stdout.write('of tails ')
        if side == 2:
            print('of side A...')
        else:
            print('of side B...')
        cloud, cards = net.get_cloud_points_slice(side, slice_samp, card=True)
        cloud = make_plane(cloud, coord=del_coord) * net.get_resolution()
    print('\t\tInserting cloud in analyzer')
    if clstring:
        mask = clst.get_del_mask()
        if not del_mask:
            mask = np.ones(shape=mask.shape, dtype=np.bool)
        set_clouds.insert_cloud(cloud, box, slice_samp, clst.get_clsts_list(), mask)
    else:
        set_clouds.insert_cloud(cloud, box, slice_samp, cards)

print('\tAnalyzing...')
set_clouds.analyze(verbose=True)

print('\tPlotting the results (close all windows to end)...')
set_clouds.plot(block=True)

output_pkl = output_dir + '/' + stem_name + '.pkl'
print('\tStoring the result in file ' + output_pkl)
set_clouds.pickle(output_pkl)

print('Terminated. (' + time.strftime("%c") + ')')

