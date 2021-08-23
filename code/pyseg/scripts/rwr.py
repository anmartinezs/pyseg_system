"""

    Script for applying RWR to GraphMCF

    Input:  - GraphMCF instance (pkl)

    Output: - GraphMCF with the result of RWR added as a new prop

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
from graph import GraphGT
from globals import *
from factory import unpickle_obj
import operator
import pyseg as ps

try:
    import pickle as pickle
except:
    import pickle

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_tomo> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: GraphMCF pickle, this file will be overwritten. \n' + \
           '    -o <dir_name>: output directory.' + \
           '    -b (optional): if present the graph is pre-filtered for getting Minimum Spanning Tree' + \
           '    -s <string>(optional): key name for the property that identifies the sources.' + \
           '                           If not present all vertices are considered sources.' + \
           '    -d <value>(optional): scalar value which identifies the sources (default 0).' + \
           '    -a <string>(optional): key name for additional sources property.' + \
           '    -t <value>(optional): if ''a'' present set the threshold for additional sources.' + \
           '                          (default 0.0).' + \
           '    -w <string>(optional): if present sets property used for edge weighting.' + \
           '    -x (optional): if present inverts the values of the property used for edges weighting.' + \
           '    -y <string>(optional): if present sets property used for vertex weighting.' + \
           '    -z <string>(optional): if present inverts the values of the property used for vertices weighting.' + \
           '    -p <string>(optional): key name for the added property, if not preset string ' + \
           '                            rwr is used' + \
           '    -m (optional): if present connections among sources are discarded, so' + \
           '                   closed sources (in sources subgraph) are not taken into ' + \
           '                   consideration. Otherwise, usual RWR.'+ \
           '    -c <value>(optional): restart probability for RWR, [0, 1] (default 0.01).' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_rwr(input_file, output_dir, prop_key, seg_key, seg_val, usual, weight, inv, c,
           seg_a_key, th_a, weight_v, inv_v, mst, verbose):

    # Initialization
    if verbose:
        print('\tLoading graph...')
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf = unpickle_obj(input_file)

    if verbose:
        print('\tPre-processing the graph...')
    graph_mcf.filter_self_edges()
    for v in graph_mcf.get_vertices_list():
        if len(graph_mcf.get_vertex_neighbours(v.get_id())[1]) <= 0:
            graph_mcf.remove_vertex(v)
    if not usual:
        if verbose:
            print('\t\tGetting boundary sources...')
        seg_prop_id = graph_mcf.get_prop_id(seg_key)
        for v in graph_mcf.get_vertices_list():
            v_id = v.get_id()
            hold = graph_mcf.get_prop_entry_fast(seg_prop_id, v_id, 1, np.float)
            if hold[0] == seg_val:
                neighs, edges = graph_mcf.get_vertex_neighbours(v_id)
                if len(neighs) == 0:
                    graph_mcf.remove_vertex(v)
                    continue
                count = 0
                for i, n in enumerate(neighs):
                    hold = graph_mcf.get_prop_entry_fast(seg_prop_id, n.get_id(), 1, np.float)
                    if hold[0] == seg_val:
                        graph_mcf.remove_edge(edges[i])
                    else:
                        count += 1
                if count == 0:
                    graph_mcf.remove_vertex(v)

    if inv or inv_v:
        print('\tInverting some properties...')
    if inv:
        graph_mcf.invert_prop(weight, weight + '_inv')
        weight += '_inv'
    if inv_v:
        graph_mcf.invert_prop(weight_v, weight_v + '_inv')
        weight_v += '_inv'

    if verbose:
        print('\tGetting the GT graph...')
    graph = GraphGT(graph_mcf)
    graph_gt = graph.get_gt()

    if mst:
        if verbose:
            print('\tCompute minimum spanning tree...')
        graph.min_spanning_tree(SGT_MIN_SP_TREE, weight)
        graph.add_prop_to_GraphMCF(graph_mcf, SGT_MIN_SP_TREE, up_index=False)
        graph_mcf.threshold_edges(SGT_MIN_SP_TREE, 0, operator.eq)

    if verbose:
        print('\tGetting the sources list...')
    sources = list()
    if seg_key is not None:
        prop_s = graph_gt.vertex_properties[seg_key]
        if seg_a_key is None:
            for v in graph_gt.vertices():
                if prop_s[v] == seg_val:
                    sources.append(v)
        else:
            prop_a = graph_gt.vertex_properties[seg_a_key]
            for v in graph_gt.vertices():
                if (prop_s[v] == seg_val) or (prop_a[v] > th_a):
                    sources.append(v)
    else:
        for v in graph_gt.vertices():
            sources.append(v)

    if verbose:
        print('\tRWR...')
    graph.multi_rwr(sources, prop_key, c, weight, weight_v, mode=2)

    if verbose:
        print('\tUpdating GraphMCF...')
    graph.add_prop_to_GraphMCF(graph_mcf, prop_key, up_index=True)

    print('\tUpdate subgraph relevance...')
    graph_mcf.compute_sgraph_relevance()

    if verbose:
        print('\tStoring the result...')
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf.pickle(output_dir + '/' + stem + '_rwr.pkl')
    ps.disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=True, edges=True),
                         output_dir + '/' + stem + '_rwr_edges.vtp')
    ps.disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
                         output_dir + '/' + stem + '_rwr_sch.vtp')


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvmxzbi:o:s:d:w:p:c:a:t:y:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = None
    output_dir = None
    prop_key = 'rwr'
    seg_key = None
    seg_val = 0
    weight = None
    weight_v = None
    verbose = False
    usual = True
    c = 0.01
    inv = False
    inv_v = False
    seg_a_key = None
    th_a = 0
    mst = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-s":
            seg_key = arg
        elif opt == "-d":
            seg_val = float(arg)
        elif opt == "-w":
            weight = arg
        elif opt == "-p":
            prop_key = arg
        elif opt == "-m":
            usual = False
        elif opt == "-c":
            c = float(arg)
        elif opt == "-x":
            inv = True
        elif opt == "-v":
            verbose = True
        elif opt == "-a":
            seg_a_key = arg
        elif opt == "-t":
            th_a = float(arg)
        elif opt == "-y":
            weight_v = arg
        elif opt == "-z":
            inv_v = True
        elif opt == "-b":
            mst = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if (input_file is None) or (output_dir is None):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool for getting the graph mcf of a tomogram.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            print('\tOutput file: ' + output_dir)
            print('\tProperty name: ' + prop_key)
            if mst:
                print('\tMinimum spanning tree activated.')
            if seg_key is not None:
                print('\tProperty for sources: ' + seg_key)
                print('\tValue for sources: ' + str(seg_val))
                if usual:
                    print('\tStandard RWR.')
                else:
                    print('\tBoundary sources mode activated (option ''m'')')
            if seg_a_key is not None:
                print('\tProperty for additional sources: ' + seg_a_key)
                print('\t\tThreshold: ' + str(th_a))
            if weight is not None:
                print('\tEdge weighting property: ' + weight)
                if inv:
                    print('\t\tInverted weighting property.')
            if weight_v is not None:
                print('\tVertex weighting property: ' + weight_v)
                if inv_v:
                    print('\t\tInverted weighting property.')
            print('\tRestart probability: ' + str(c))
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_rwr(input_file, output_dir, prop_key, seg_key, seg_val, usual, weight, inv,
               c, seg_a_key, th_a, weight_v, inv_v, mst, verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])