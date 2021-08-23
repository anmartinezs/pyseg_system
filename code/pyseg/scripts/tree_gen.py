"""

    Script for simplifying an input graph into a tree and extracts its properties

    Input:  - Pickle file with a GraphMCF

    Output: - Output tree (pickle, VTK and GT files)

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
import operator
from graph import GraphGT
from globals import *
from factory import unpickle_obj

try:
    import pickle as pickle
except:
    import pickle

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input pickle file for the GraphMCF. \n' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored.\n' + \
           '    -w <prop_weight>(optional): key string for the property used for weighting' + \
           ' edges (default None).\n' + \
           '    -p (optional): if present edges take their properties from vertices.' + \
           '    -a (optional): if present arcs instead edges are stored in the .vtp file' + \
           '    -b (optional): if present tree betweenness is computed.' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_tree_gen(input_file, output_dir, w_prop=None, av_mode=False, ed_mode=True, bet=False,
                verbose=False):

    if verbose:
        print('\tUnpickling graph...')
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf = unpickle_obj(input_file)

    if verbose:
        print('\tGenerating the GT graph...')
    graph_gt = GraphGT(graph_mcf)

    if verbose:
        print('\tGenerating the tree...')
    graph_gt.min_spanning_tree(SGT_MIN_SP_TREE, w_prop)
    graph_gt.add_prop_to_GraphMCF(graph_mcf, SGT_MIN_SP_TREE, up_index=True)
    graph_mcf.threshold_edges(SGT_MIN_SP_TREE, 0, operator.eq)
    graph_gt_out = GraphGT(graph_mcf)

    if verbose and bet:
        print('\tComputing betweenness...')
        # For edges and vertices
        graph_gt_out.betweenness(mode='')
        graph_gt_out.add_prop_to_GraphMCF(graph_mcf, SGT_BETWEENNESS)

    if verbose:
        print('\tStoring the result...')
    graph_mcf.pickle(output_dir + '/' + stem + '_tree.pkl')
    disperse_io.save_vtp(graph_mcf.get_vtp(av_mode, ed_mode),
                         output_dir + '/' + stem + '_tree.vtp')
    graph_gt_out.save(output_dir + '/' + stem + '_tree.gt')


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvpabi:o:w:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = ''
    output_dir = ''
    w_prop = None
    verbose = False
    av_mode = False
    ed_mode = True
    bet = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-w":
            w_prop = arg
        elif opt == "-p":
            av_mode = True
        elif opt == "-a":
            ed_mode = False
        elif opt == "-b":
            bet = True
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if (input_file == '') or (output_dir == ''):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool for simplifying a graph into a tree.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            print('\tOutput directory: ' + output_dir)
            if w_prop is not None:
                print('\tProperty for weighting edges: ' + w_prop)
            if av_mode:
                print('\tEdges will take their properties from vertices.')
            if ed_mode:
                print('\tEdges will be printed in the .vtp file.')
            else:
                print('\tArcs will be printed in the .vtp file.')
            if bet:
                print('\tCompute betweenness activated.')
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_tree_gen(input_file, output_dir, w_prop, av_mode, ed_mode, bet, verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])