"""

    Script for extracting the graphs for clustering an acting network

    Input:  - GraphMCF af an acting network
            - Parameters for clustering algorithm (Affinity propagation)

    Output: - A clusterd graph

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import os
import time
import sys
import operator
import numpy as np
import pyseg as ps
try:
    import cPickle as pickle
except:
    import pickle

########## Global variables

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/home/martinez/workspace/disperse/data/marion/cluster'

# Original density maps
in_graphs = (ROOT_PATH+'/graphs/20151116_W2t1_crop.pkl',
             )

####### Output data

output_dir = ROOT_PATH+'/clst'

####### Graph global pre-processsing

g_min_diam = 30 # nm
max_tree = True
min_deg = 1
max_deg = 4

####### Centrality measures 

pgr_dump = .9

####### DBSCAN clustering algorithm

do_db = True
db_e_prop = ps.globals.SGT_EDGE_LENGTH # edges property string for building the affinity matrix
db_eps = 200. # The maximum distance between two samples for them to be considered as in the same neighborhood.
db_min_s = 8 # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
db_rand = False # if True (default False) the output labels for clusters a randomized

####### Affinity propagation clustering algorithm

do_aff = True
aff_e_prop = ps.globals.SGT_EDGE_LENGTH # edges property string for building the affinity matrix
aff_pref = None # Control the number of clusters, in None automatically set
aff_rand = True # if True (default False) the output labels for clusters a randomized

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print 'Clustering a GraphMCF from an actin network.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tGlobal pre-processing: '
print '\t\t-Minimum graph diameter: ' + str(pgr_dump)
if max_tree:
    print '\t\t-Maximum spanning tree activated'
print '\t\t-Degree range: [' + str(min_deg) + ', ' + str(max_deg) + ']'
if do_db:
    print '\tDBSCAN clustering settings: '
    if db_e_prop is not None:
        print '\t\t-Edge weighting: ' + db_e_prop
    print '\t\t-Maximum distance between neighbors: ' + str(db_eps)
    print '\t\t-Minimum number of samples: ' + str(db_min_s)
    if db_rand:
        print '\t\t-Random labels activated'
if do_aff:
    print '\tAffinity clustering settings: '
    if aff_e_prop is not None:
        print '\t\t-Edge weighting: ' + db_e_prop
    if aff_pref is None:
        print '\t\t-Preference automatically computed'
    else:
        print '\t\t-Preference: ' + str(aff_pref)
    if aff_rand:
        print '\t\t-Random labels activated'
print '\tInput graphs pickles: ' + str(in_graphs)
print '\tOutput directory: ' + output_dir
print ''

# Loop for processing the input data
print 'Running main loop: '
for in_graph in in_graphs:

    f_path, f_fname = os.path.split(in_graph)
    print '\tComputing paths for ' + f_fname + ' ...'
    f_stem_pkl, f_ext = os.path.splitext(f_fname)

    print '\tUnpickling the GraphMCF...'
    graph = ps.factory.unpickle_obj(in_graph)
    graph.compute_graph_gt()
    graph_gt = ps.graph.GraphGT(graph)

    print '\tApplying global pre-processing:'
    if g_min_diam is not None:
        print '\t\t-Threshold by graph diameter (minimum): ' + str(g_min_diam) + ' nm'
        graph.threshold_vertices(ps.globals.STR_GRAPH_DIAM, g_min_diam, operator.lt)
    if max_tree:
        print '\t\t-Computing minimum spanning tree...'
        graph_gt.min_spanning_tree(prop_name=ps.globals.SGT_MIN_SP_TREE, prop_weight=None)
        graph_gt.add_prop_to_GraphMCF(graph, ps.globals.SGT_MIN_SP_TREE, up_index=True)
        graph.threshold_edges(ps.globals.SGT_MIN_SP_TREE, 1, operator.ne)
        print '\t\t-Degree thresholding with range: [' + str(min_deg) + ', ' + str(max_deg) + ']'
        graph.compute_vertex_degree(key_v=ps.globals.SGT_NDEGREE)
        graph.threshold_vertices(ps.globals.SGT_NDEGREE, min_deg, operator.lt)
        graph.threshold_vertices(ps.globals.SGT_NDEGREE, max_deg, operator.gt)

    print '\tComputing GraphGT...'
    graph.compute_graph_gt()
    graph_gt = ps.graph.GraphGT(graph)

    print '\tComputing centrality (PageRank)...'
    graph_gt.page_rank(pgr_dump)
    graph_gt.add_prop_to_GraphMCF(graph, ps.globals.SGT_PAGERANK, up_index=True)

    if do_db:
        print '\tClustering (DBSCAN)...'
        graph_gt.dbscan(db_e_prop, eps=db_eps, min_samples=db_min_s, rand=db_rand, ditype=np.float16)
        graph_gt.add_prop_to_GraphMCF(graph, ps.globals.STR_DBSCAN_CLUST, up_index=True)
    if do_aff:
        print '\tClustering (Affinity propagation)...'
        graph_gt.aff_propagation(aff_e_prop, preference=aff_pref, rand=aff_rand, damp=.5,
                                 ditype=np.float16, verbose=True)
        graph_gt.add_prop_to_GraphMCF(graph, ps.globals.STR_AFF_CLUST, up_index=True)
        graph_gt.add_prop_to_GraphMCF(graph, ps.globals.STR_AFF_CENTER, up_index=True)

    print '\tSaving intermediate graphs...'
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + f_stem_pkl + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                            output_dir + '/' + f_stem_pkl + '_edges_2.vtp')
    ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                            output_dir + '/' + f_stem_pkl + '_sch.vtp')
    ps.disperse_io.save_numpy(graph.to_mask(verts=False).astype(np.int16),
                              output_dir + '/' + f_stem_pkl + '_pts.mrc')

    print '\tPickling the graph as: ' + f_stem_pkl + '.pkl'
    graph.pickle(output_dir + '/' + f_stem_pkl + '.pkl')

print 'Terminated. (' + time.strftime("%c") + ')'
