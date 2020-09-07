"""

    From two different segmentation generates two RWR and compares the results. This script
    assist in the analysis of GraphMCF with trans- or inter-membrane structures

    Input:  - GraphMCF instance (pkl)
            - Segmentation with at least three regions

    Output: - GraphMCF with the result of every independent RWR and the combination

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
import copy
from graph import GraphGT
from globals import *
from factory import unpickle_obj
import operator
import pyseg as ps

try:
    import pickle as pickle
except:
    import pickle

################# Global variables

SEG_KEY = 'seg'
RWR_A_KEY = 'rwr_a'
RWR_B_KEY = 'rwr_b'
RWR_T_KEY = 'rwr_t'
RWR_C_KEY = 'rwr_c'

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_pkl> -s <seg_tomo> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: GraphMCF pickle, this file will be overwritten. \n' + \
           '    -s <file_name>: path to segmentation file.' + \
           '    -o <dir_name>: output directory.' + \
           '    -r <float>(optional): restart probability for RWR, [0, 1] (default 0.01).' + \
           '    -a <int>(optional): sources region A label (default 2).' + \
           '    -b <int>(optional): sources region B label (default 3).' + \
           '    -c <int>(optional): trans region label (default 1).' + \
           '    -d <float>(optional): percentil [0,100] for segmentation, default 10.' + \
           '    -t (optional): if present the graph is pre-filtered for getting Minimum Spanning Tree' + \
           '    -m (optional): if present connections among sources are discarded, so' + \
           '                   closed sources (in sources sub-graph) are not taken into ' + \
           '                   consideration. Otherwise, usual RWR.'+ \
           '    -w <string>(optional): if present sets property used for edge weighting.' + \
           '    -z (optional): if present inverts the values of the property used for edges weighting.' + \
           '    -y <string>(optional): if present sets property used for vertex weighting.' + \
           '    -x (optional): if present inverts the values of the property used for vertices weighting.' + \
           '    -v (optional): verbose mode activated.'

################# Helping routines

def seg_trans(graph, graph_c, graph_a, graph_b, per):

    # Initialization
    prop_t_id = graph.get_prop_id(RWR_T_KEY)
    prop_c_id = graph.get_prop_id(RWR_C_KEY)
    prop_a_id = graph.get_prop_id(RWR_A_KEY)
    prop_b_id = graph.get_prop_id(RWR_B_KEY)
    arr_t, arr_c, arr_a, arr_b = list(), list(), list(), list()

    # Loop for compute the percentiles
    vertices = graph.get_vertices_list()
    for v in vertices:
        v_id = v.get_id()
        prop_t = graph.get_prop_entry_fast(prop_t_id, v_id, 1, np.float)[0]
        prop_c = graph.get_prop_entry_fast(prop_c_id, v_id, 1, np.float)[0]
        prop_a = graph.get_prop_entry_fast(prop_a_id, v_id, 1, np.float)[0]
        prop_b = graph.get_prop_entry_fast(prop_b_id, v_id, 1, np.float)[0]
        if prop_t > 0:
            arr_t.append(prop_t)
        if prop_c > 0:
            arr_c.append(prop_c)
        if prop_a > 0:
            arr_a.append(prop_a)
        if prop_b > 0:
            arr_b.append(prop_b)
    per_t, per_c = np.percentile(arr_t, per), np.percentile(arr_c, per)
    per_a, per_b = np.percentile(arr_a, per), np.percentile(arr_b, per)

    # Main loop for doing the comparisons
    for v in vertices:
        v_id = v.get_id()
        prop_t = graph.get_prop_entry_fast(prop_t_id, v_id, 1, np.float)[0]
        prop_c = graph.get_prop_entry_fast(prop_c_id, v_id, 1, np.float)[0]
        prop_a = graph.get_prop_entry_fast(prop_a_id, v_id, 1, np.float)[0]
        prop_b = graph.get_prop_entry_fast(prop_b_id, v_id, 1, np.float)[0]
        # Set label (1-trans, 2-C, 3-A and 4-B)
        lbl = 0
        if prop_t > per_t:
            lbl = 1
        elif (prop_a > per_a) and (prop_a >= prop_b):
            lbl = 3
        elif (prop_b > per_b) and (prop_b >= prop_a):
            lbl = 4
        elif prop_c > per_c:
            lbl = 2
        # Segment this vertex
        if lbl == 1:
            graph_c.remove_vertex(graph_c.get_vertex(v_id))
            graph_a.remove_vertex(graph_a.get_vertex(v_id))
            graph_b.remove_vertex(graph_b.get_vertex(v_id))
        elif lbl == 2:
            graph.remove_vertex(v)
            graph_a.remove_vertex(graph_a.get_vertex(v_id))
            graph_b.remove_vertex(graph_b.get_vertex(v_id))
        if lbl == 3:
            graph.remove_vertex(v)
            graph_c.remove_vertex(graph_c.get_vertex(v_id))
            graph_b.remove_vertex(graph_b.get_vertex(v_id))
        if lbl == 4:
            graph.remove_vertex(v)
            graph_c.remove_vertex(graph_c.get_vertex(v_id))
            graph_a.remove_vertex(graph_a.get_vertex(v_id))

################# Work routine

def do_rwr(input_file, seg_file, output_dir, a_lbl, b_lbl, c_lbl, c, mst, usual, weight_v, inv_v,
           weight_e, inv_e, per, verbose):

    if verbose:
        print('\tLoading graph...')
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf = unpickle_obj(input_file)

    if verbose:
        print('\tInserting scalar field property as \'' + SEG_KEY + '\'...')
    graph_mcf.add_scalar_field(ps.disperse_io.load_tomo(seg_file), SEG_KEY)

    if verbose:
        print('\tPre-processing the graph...')
    graph_mcf.filter_self_edges()
    for v in graph_mcf.get_vertices_list():
        if len(graph_mcf.get_vertex_neighbours(v.get_id())[1]) <= 0:
            graph_mcf.remove_vertex(v)
    if mst:
        if verbose:
            print('\tCompute minimum spanning tree...')
        graph = GraphGT(graph_mcf)
        graph.min_spanning_tree(SGT_MIN_SP_TREE, weight_e)
        graph.add_prop_to_GraphMCF(graph_mcf, SGT_MIN_SP_TREE, up_index=False)
        graph_mcf.threshold_edges(SGT_MIN_SP_TREE, 0, operator.eq)

    if usual:
        if verbose:
            print('\t\tThresholding the according to segmentation...')
        graph_mcf.threshold_seg_region(SEG_KEY, 0, keep_b=False)
    else:
        if verbose:
            print('\t\tThresholding region A until keeping just the boundaries...')
        graph_mcf.threshold_seg_region(SEG_KEY, a_lbl, keep_b=True)
        if verbose:
            print('\t\tThresholding region B until keeping just the boundaries...')
        graph_mcf.threshold_seg_region(SEG_KEY, b_lbl, keep_b=True)

    if inv_e:
        if verbose:
            print('\tInverting edge properties...')
        graph_mcf.invert_prop(weight_e, weight_e + '_inv')
        weight_e += '_inv'
    if inv_v:
        if verbose:
            print('\tInverting vertex properties...')
        graph_mcf.invert_prop(weight_v, weight_v + '_inv')
        weight_v += '_inv'

    if verbose:
        print('\tGetting the GT graph...')
    graph = GraphGT(graph_mcf)
    graph_gt = graph.get_gt()

    if verbose:
        print('\tGetting the sources list...')
    sources_a, sources_b, sources_c = list(), list(), list()
    prop_seg = graph_gt.vertex_properties[SEG_KEY]
    for v in graph_gt.vertices():
        if prop_seg[v] == a_lbl:
            sources_a.append(v)
        if prop_seg[v] == b_lbl:
            sources_b.append(v)
        if prop_seg[v] == c_lbl:
            sources_c.append(v)

    if verbose:
        print('\tRWR for region A...')
    graph.multi_rwr(sources_a, RWR_A_KEY, c, weight_e, weight_v, mode=2)

    if verbose:
        print('\tRWR for region B...')
    graph.multi_rwr(sources_b, RWR_B_KEY, c, weight_e, weight_v, mode=2)

    if verbose:
        print('\tRWR for region trans...')
    graph.multi_rwr(sources_c, RWR_C_KEY, c, weight_e, weight_v, mode=2)

    if verbose:
        print('\tComputing trans RWR...')
    prop_trans = graph_gt.new_vertex_property('float')
    prop_a = graph_gt.vertex_properties[RWR_A_KEY]
    prop_b = graph_gt.vertex_properties[RWR_B_KEY]
    prop_c = graph_gt.vertex_properties[RWR_C_KEY]
    prop_trans.get_array()[:] = prop_a.get_array()[:] * prop_b.get_array()[:]
    graph.set_gt_vertex_property(RWR_T_KEY, prop_trans)

    if verbose:
        print('\tFactors correction...')
    prop_trans.get_array()[:] = prop_trans.get_array()
    prop_c.get_array()[:] = prop_c.get_array()
    prop_a.get_array()[:] = prop_a.get_array()
    prop_b.get_array()[:] = prop_b.get_array()

    if verbose:
        print('\t\tKatz centrality...')
    if weight_e is None:
        p_kats = gt.katz(graph_gt, weight=None)
    else:
        p_kats = gt.katz(graph_gt, weight=graph_gt.edge_properties[weight_e])
    graph_gt.vertex_properties[SGT_KATZ] = p_kats

    if verbose:
        print('\tUpdating GraphMCF...')
    graph.add_prop_to_GraphMCF(graph_mcf, RWR_A_KEY, up_index=True)
    graph.add_prop_to_GraphMCF(graph_mcf, RWR_B_KEY, up_index=True)
    graph.add_prop_to_GraphMCF(graph_mcf, RWR_T_KEY, up_index=True)
    graph.add_prop_to_GraphMCF(graph_mcf, RWR_C_KEY, up_index=True)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_KATZ, up_index=True)

    if verbose:
        print('\tSegmentation...')
    # graph_mcf.threshold_seg_region(SEG_KEY, a_lbl, keep_b=False)
    # graph_mcf.threshold_seg_region(SEG_KEY, b_lbl, keep_b=False)
    graph_mcf_t = copy.deepcopy(graph_mcf)
    graph_mcf_c = copy.deepcopy(graph_mcf)
    graph_mcf_a = copy.deepcopy(graph_mcf)
    graph_mcf_b = copy.deepcopy(graph_mcf)
    seg_trans(graph_mcf_t, graph_mcf_c, graph_mcf_a, graph_mcf_b, per)

    print('\tUpdate subgraph relevance...')
    graph_mcf.compute_sgraph_relevance()
    graph_mcf_t.compute_sgraph_relevance()
    graph_mcf_c.compute_sgraph_relevance()
    graph_mcf_a.compute_sgraph_relevance()
    graph_mcf_b.compute_sgraph_relevance()

    if verbose:
        print('\tStoring the result...')
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf.pickle(output_dir + '/' + stem + '_rwr.pkl')
    ps.disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_rwr_edges.vtp')
    # ps.disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
    #                      output_dir + '/' + stem + '_rwr_sch.vtp')
    ps.disperse_io.save_vtp(graph_mcf_t.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_rwr_t_edges.vtp')
    ps.disperse_io.save_vtp(graph_mcf_c.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_rwr_c_edges.vtp')
    ps.disperse_io.save_vtp(graph_mcf_a.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_rwr_a_edges.vtp')
    ps.disperse_io.save_vtp(graph_mcf_b.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_rwr_b_edges.vtp')


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvtmxzi:o:s:r:a:b:c:w:y:d:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = None
    seg_file = None
    output_dir = None
    c = 0.01
    a_lbl = 2
    b_lbl = 3
    c_lbl = 1
    mst = False
    usual = True
    weight_v = None
    weight_e = None
    inv_v = False
    inv_e = False
    per = 10.
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-s":
            seg_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-r":
            c = float(arg)
        elif opt == "-a":
            a_lbl = int(arg)
        elif opt == "-b":
            b_lbl = int(arg)
        elif opt == "-c":
            c_lbl = float(arg)
        elif opt == "-t":
            mst = True
        elif opt == "-m":
            usual = False
        elif opt == "-y":
            weight_v = arg
        elif opt == "-w":
            weight_e = arg
        elif opt == "-z":
            inv_e = True
        elif opt == "-x":
            inv_v = True
        elif opt == "-d":
            per = float(arg)
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if (input_file is None) or (seg_file is None) or (output_dir is None):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('RWR from two regions on a GraphMCF.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            print('\tInput file: ' + seg_file)
            print('\tOutput file: ' + output_dir)
            print('\tRe-start probability: ' + str(c))
            if mst:
                print('\tMinimum spanning tree activated.')
            print('\tRegion A label: ' + str(a_lbl))
            print('\tRegion B label: ' + str(b_lbl))
            print('\tRegion trans label: ' + str(c_lbl))
            if not usual:
                print('\tBoundary sources mode activated (option ''m'')')
            if weight_v is not None:
                print('\tVertex weighting property: ' + weight_v)
                if inv_v:
                    print('\t\tInverted weighting property.')
            if weight_e is not None:
                print('\tEdge weighting property: ' + weight_e)
                if inv_e:
                    print('\t\tInverted weighting property.')
            print('\tPercentile for segmentation ' + str(per) + '%')
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_rwr(input_file, seg_file, output_dir, a_lbl, b_lbl, c_lbl, c, mst, usual, weight_v, inv_v,
               weight_e, inv_e, per, verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])