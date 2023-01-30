"""

    Measures local betweenness (centrality)

    Input:  - GraphMCF (pickle file)

    Output: - GraphMCF with betweenness properties

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import sys
import time
import getopt
import copy
from globals import *
from factory import unpickle_obj
from graph import GraphGT
import pyseg as ps

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf> -o <out_idr> -e <w_e_prop> -D <M_dst> -d <m_dst>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <string>: input Graph MCF in a pickle file. \n' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored.\n' + \
           '    -e <string>: property string key for weighting the edges.\n' + \
           '    -D <float>: maximum distance for paths nm.' + \
           '    -d <float>: minimum distance for paths nm.' + \
           '    -P <string>(optional): membrane segmentation property.' + \
           '    -p <int>(optional): membrane label.\n' + \
           '    -v (optional): verbose mode activated.'

################# Global variables

STR_LOC_BET = 'loc_bet'
VL_NEIGH = 1
VL_ANCHOR = 2
VL_INNER = 3

################# Helper classes (Visitors)


################# Helper functions


################# Work routine

def do_loc_bet(input_file, output_dir, key_p, val_p, dist_max, dist_min, key_e, verbose):

    if verbose:
        print('\tLoading the graph...')
    path, stem = os.path.split(input_file)
    stem, ext = os.path.splitext(stem)
    if ext == '.pkl':
        graph_mcf = unpickle_obj(input_file)
    else:
        print('\tERROR: ' + ext + ' is a non valid format.')
        sys.exit(4)

    if verbose:
        print('\tGetting the GT graph...')
    graph = GraphGT(graph_mcf)
    graph_gt = graph.get_gt()

    if verbose:
        print('\tMarking the sources...')
    prop_s = graph_gt.new_vertex_property('int')
    prop_p = graph_gt.vertex_properties[key_p]
    if key_p is None:
        prop_s.get_array()[:] = np.ones(shape=graph_gt.num_vertices, dtype=int)
    else:
        for v in graph_gt.vertices():
            if prop_p[v] == val_p:
                prop_s[v] = 1

    if verbose:
        print('\tComputing betweenness...')
    graph.fix_len_path_centrality(STR_LOC_BET, key_e, dist_min, dist_max, prop_s)

    if verbose:
        print('\tAdding the new computed properties...')
    graph.add_prop_to_GraphMCF(graph_mcf, STR_LOC_BET, up_index=True)
    graph.add_prop_to_GraphMCF(graph_mcf, 'edge_'+STR_LOC_BET, up_index=True)

    if verbose:
        print('\tStoring the result in ' + output_dir)
    _, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf.pickle(output_dir + '/' + stem + '.pkl')
    ps.disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=False, edges=True),
                         output_dir + '/' + stem + '_edges.vtp')
    ps.disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
                         output_dir + '/' + stem + '_sch.vtp')

################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvi:o:e:p:P:d:D:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = None
    output_dir = None
    prop_e = None
    prop_p = None
    val_p = 1.
    verbose = False
    dist_max = None
    dist_min = None
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-P":
            prop_p = arg
        elif opt == "-p":
            val_p = int(arg)
        elif opt == "-D":
            dist_max = float(arg)
        elif opt == "-d":
            dist_min = float(arg)
        elif opt == "-e":
            prop_e = arg
        elif opt == "-g":
            geo = True
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if (input_file is None) or (output_dir is None) or (prop_e is None) or \
            (dist_max is None) or (dist_min is None):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool for suppressing not transmembrane structures in a membrane.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            print('\tOutput directory: ' + output_dir)
            print('\tMinimum distance: ' + str(dist_min) + 'nm')
            print('\tMaximum distance: ' + str(dist_max) + 'nm')
            print('\tProperty for edge weighting: ' + prop_e)
            if prop_p is not None:
                print('\tSources segmentation property ' + prop_p + ' with label ' + str(val_p))
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_loc_bet(input_file, output_dir, prop_p, val_p, dist_max, dist_min, prop_e, verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])
