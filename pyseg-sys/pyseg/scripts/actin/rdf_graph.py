"""

    Script for computing the RDF of a Graph

    Input:  - A pickle file with GraphMCF
            - RDF parameters

    Output: - New properties to are added to the Graph
            - Fgiures with the RDF

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import gc
import os
import time
import scipy
import operator
import numpy as np
import pyseg as ps
import scipy as sp
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle

########## Global variables

PI_2 = np.pi * 2.
MSK_OFF = 2 # voxels
STR_LOC_BET = 'loc_bet'
STR_LOC_BET_E = 'edge_loc_bet'

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/tomograms/marion/Clusters'

# Input graph
in_pkl = ROOT_PATH + '/g_ga/20151116_W4t1_pre_cc.pkl'

# Input mask
in_mask = ROOT_PATH + '/in/fits/pre/20151116_W4t1_mask.fits'

####### Output data

out_dir = ROOT_PATH + '/rdf'

####### RDF parameters

rdf_max_d = 200 # nm
rdf_nsamp = 60
rdf_nsamp_f = 1. / 6.

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print 'RDF on graphs.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput picke file with the graph: ' + str(in_pkl)
if in_mask is not None:
    print '\tInput mask: ' + str(in_mask)
print '\tOuput directory: ' + str(out_dir)
print 'RDF parameters:'
print '\t-Maximum distance: ' + str(rdf_max_d) + ' nm'
print '\t-Number of samples: ' + str(rdf_nsamp)
print '\t-Sub-sampling factor to add properties to graph: ' + str(rdf_nsamp_f)
print ''

# Loop for processing the input data
print 'Loading input graph...'
f_stem_pkl = os.path.splitext(os.path.split(in_pkl)[1])[0]
graph = ps.factory.unpickle_obj(in_pkl)

import operator
graph.threshold_vertices(ps.disperse_io.STR_FIELD_VALUE, 0.45, operator.gt)
graph.threshold_edges(ps.disperse_io.STR_FIELD_VALUE, 0.45, operator.gt)
g_min_diam = 100
if g_min_diam is not None:
    print '\tThreshold by graph diameter (minimum): ' + str(g_min_diam) + ' nm'
    graph.compute_diameters(update=True)
    graph.compute_sgraph_relevance()
    graph.threshold_vertices(ps.globals.STR_GRAPH_DIAM, g_min_diam, operator.lt)
g_dec = 2
if g_dec > 1:
    graph_gt = ps.graph.GraphGT(graph)
    graph.bet_decimation(g_dec, graph_gt, key_e=ps.globals.STR_FIELD_VALUE)
    # graph.bet_decimation(g_dec, graph=None, key_v=None, key_e=None, gt_update=True)
ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                        out_dir + '/' + f_stem_pkl + '_rdf_hold.vtp')
graph.pickle(out_dir + '/' + f_stem_pkl + '_hold.pkl')

mask = None
if in_mask is not None:
    print 'Loading input mask...'
    mask = ps.disperse_io.load_tomo(in_mask) == 0

print 'Computing the RDF...'
rdf_x, rdf_y = graph.graph_rdf(2*graph.get_resolution(), rdf_max_d, rdf_nsamp, rdf_nsamp_f, edge_len=ps.globals.SGT_EDGE_LENGTH,
                               mask=mask, norm=True, npr=None)

print 'Plotting the RDF...'
plt.figure()
plt.title('RDF')
plt.xlabel('Radius (nm)')
plt.ylabel('Denstiy')
plt.plot(rdf_x, rdf_y)
plt.show(block=True)
plt.close()

print '\tSaving intermediate graphs...'
f_stem_pkl = os.path.splitext(os.path.split(in_pkl)[1])[0]
ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                        out_dir + '/' + f_stem_pkl + '_rdf_edges.vtp')
ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True), out_dir + '/' + f_stem_pkl + '_rdf_edges_2.vtp')

print '\tPickling the graph as: ' + f_stem_pkl + '_rdf.pkl'
graph.pickle(out_dir + '/' + f_stem_pkl + '_rdf.pkl')

print 'Terminated. (' + time.strftime("%c") + ')'
