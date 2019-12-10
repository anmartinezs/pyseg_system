"""

    Script for clustering vertices of MCF graph

    Input:  - GraphMCF (pickle file)

    Output: - Clusters stored as GraphMCF property

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import sys
import time
import getopt
from globals import *
from factory import unpickle_obj
from graph import GraphGT
import pyseg as ps

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input Graph MCF in a pickle file. \n' + \
           '    -a <string>(optional): algorithm, valid: ''af'' (default), ''bm'' or ' \
           '    ''dbscan''.' + \
           '    -w <string_key>: property used for edge weighting.' + \
           '    -x (optional): for inverting weighting values.' + \
           '    -p <float>(optional, AF): preference value for controlling the number of clusters.' + \
           '    -q (optional): for inverting preference values.' + \
           '    -b <float>(optional): if present growing factor for logistic remapping of ' + \
           '                          input values. Positive value (for example 3).' + \
           '    -s (optional): segmentation for filtering edges.' + \
           '    -d <float>(optional, AF): damping factor [0.5 1) (default 0.5).' + \
           '    -c <int>(optional, AF): number of iterations with no change for stopping (default 15).' + \
           '    -m <int>(optional, AF): maximum number of iterations (default 200).' + \
           '    -e <float>(optional, DBSCAN): the maximum distance between two samples for ' + \
           '    them to be considered as in the same neighborhood (default 0.5).' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_mcf_clustering(input_file, prop_key, ainv, pref, pinv, damp, conv_iter, max_iter,
                     grow, seg_file, alg, eps, verbose):

    if verbose:
        print '\tLoading the graph...'
    path, stem = os.path.split(input_file)
    stem, ext = os.path.splitext(stem)
    if ext == '.pkl':
        graph_mcf = unpickle_obj(input_file)
    else:
        print '\tERROR: ' + ext + ' is a non valid format.'
        sys.exit(4)

    if seg_file is not None:
        if verbose:
            print '\tThresholding graph arcs in mask...'
        mask = ps.disperse_io.load_tomo(seg_file)
        graph_mcf.threshold_edges_in_mask(mask)

    if alg == 'dbscan':
        if verbose:
            print '\tClustering by DBSCAN...'
        import operator
        graph_mcf.threshold_vertices(STR_GRAPH_RELEVANCE, 1, operator.ne)
        graph_gt = GraphGT(graph_mcf)
        graph_gt.dbscan(affinity=prop_key,
                        ainv=ainv,
                        b=grow,
                        eps=eps,
                        rand=True)
    elif alg == 'bm':
        if verbose:
            print '\tClutering by community blockmodel...'
        graph_gt = GraphGT(graph_mcf)
        graph_gt.community_bm(affinity=prop_key,
                              ainv=ainv,
                              b=grow,
                              rand=True)
    else:
        if verbose:
            print '\tClustering by affinity propagation...'
        graph_gt = GraphGT(graph_mcf)
        graph_gt.aff_propagation(damp=damp,
                                 conv_iter=conv_iter,
                                 max_iter=max_iter,
                                 preference=pref,
                                 affinity=prop_key,
                                 ainv=ainv,
                                 b=grow,
                                 rand=True)

    if verbose:
        print '\tAssign the clusters to the graph...'
    if alg == 'dbscan':
        graph_gt.add_prop_to_GraphMCF(graph_mcf, STR_DBSCAN_CLUST, up_index=True)
    elif alg == 'bm':
        graph_gt.add_prop_to_GraphMCF(graph_mcf, STR_BM_CLUST, up_index=True)
    else:
        graph_gt.add_prop_to_GraphMCF(graph_mcf, STR_AFF_CLUST, up_index=True)
        graph_gt.add_prop_to_GraphMCF(graph_mcf, STR_AFF_CENTER, up_index=True)

    if verbose:
        print '\tStoring the result in ' + input_file
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf.pickle(input_file)
    ps.disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=True, edges=True),
                         path + '/' + stem + '_edges.vtp')
    ps.disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
                         path + '/' + stem + '_sch.vtp')
    ps.disperse_io.save_numpy(graph_mcf.print_vertices(th_den=-1, property=STR_AFF_CLUST),
                           path + '/' + stem + '_clt_seg.vti')

################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvxqi:w:p:d:c:m:b:s:a:e")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    input_file = None
    prop_key = None
    damp = 0.5
    conv_iter = 15
    max_iter = 200
    alg = 'af'
    eps = 0.5
    pref = None
    ainv = False
    pinv = False
    grow = None
    seg_file = None
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print usage_msg
            print help_msg
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-a":
            if (arg == 'af') or (arg == 'dbscan') or (arg == 'bm'):
                alg = arg
            else:
                print 'Unknown argument for -a ' + arg
                print usage_msg
                sys.exit(3)
        elif opt == "-e":
            eps = float(arg)
        elif opt == "-w":
            prop_key = arg
        elif opt == "-x":
            ainv = True
        elif opt == "-p":
            pref = arg
        elif opt == "-q":
            pinv = True
        elif opt == "-d":
            damp = float(arg)
        elif opt == "-c":
            conv_iter = int(arg)
        elif opt == "-m":
            max_iter = int(arg)
        elif opt == "-b":
            grow = float(arg)
        elif opt == "-s":
            seg_file = arg
        elif opt == "-v":
            verbose = True
        else:
            print 'Unknown option ' + opt
            print usage_msg
            sys.exit(3)

    if (input_file is None) or (prop_key is None):
        print usage_msg
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print 'Running tool drawing a graph.'
            print '\tAuthor: ' + __author__
            print '\tDate: ' + time.strftime("%c") + '\n'
            print 'Options:'
            print '\tInput file: ' + input_file
            if ainv:
                print '\tEdge weighting property (inverted): ' + prop_key
            else:
                print '\tEdge weighting property: ' + prop_key
            if pref is not None:
                print '\tPreference value: ' + str(pref)
            if grow is not None:
                print '\tRemap input weights logistic growing factor: ' + str(grow)
            if seg_file is not None:
                print '\tDelete edges in mask: ' + seg_file
            if alg == 'af':
                print '\tAlgorithm: Affinity Propagation'
                print '\t\tDamping factor: ' + str(damp)
                print '\t\tConvergence iterations: ' + str(conv_iter)
                print '\t\tMaximum iterations: ' + str(max_iter)
            elif alg == 'bm':
                print '\tAlgorithm: Community blockmodel'
            else:
                print '\tAlgorithm: DBSCAN'
                print '\t\tEps: ' + str(eps)
            print ''

        # Do the job
        if verbose:
            print 'Starting...'
        do_mcf_clustering(input_file, prop_key, ainv, pref, pinv, damp, conv_iter, max_iter,
                          grow, seg_file, alg, eps, verbose)

        if verbose:
            print cmd_name + ' successfully executed. (' + time.strftime("%c") + ')'


if __name__ == "__main__":
    main(sys.argv[1:])
