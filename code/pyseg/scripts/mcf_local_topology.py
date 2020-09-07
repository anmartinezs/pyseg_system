"""

    Script for computing local topology properties from a GraphMCF

    Input:  - File with a GraphMCF (.pkl)

    Output: - GraphMCF with the new properties overwritten

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
from pyseg.globals import *
from pyseg.factory import unpickle_obj
from pyseg.graph import GraphGT
import pyseg.disperse_io as disperse_io

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input Graph MCF in a pickle file. \n' + \
           '    -w <string>(optional): key name of the property used for weighting the edges.' + \
           '    -x (optional): if present the edges weighting property is inverted.' + \
           '    -n (optional): if present the new properties are not pickled, a new graph' \
           '                   object is returned.' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_local_topology(input_file, weight, inv, no_ret, verbose):

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
    if inv:
        graph_mcf.invert_prop(weight, weight + '_inv')
    graph = GraphGT(graph_mcf)
    graph_gt = graph.get_gt()
    if weight is not None:
        weight_p = graph_gt.edge_properties[weight]
    else:
        weight_p = None

    if verbose:
        print('\tCentrality measures:')
    if verbose:
        print('\t\tPagerank...')
    p_prank = gt.pagerank(graph_gt, weight=weight_p)
    graph_gt.vertex_properties[SGT_PAGERANK] = p_prank
    if verbose:
        print('\t\tBetweenness...')
    p_vbet, p_ebet = gt.betweenness(graph_gt, weight=weight_p)
    graph_gt.vertex_properties[SGT_BETWEENNESS] = p_vbet
    graph_gt.edge_properties[SGT_BETWEENNESS] = p_ebet
    if verbose:
        print('\t\tCloseness...')
    p_close = gt.closeness(graph_gt, weight=weight_p)
    graph_gt.vertex_properties[SGT_CLOSENESS] = p_close
    if verbose:
        print('\t\tEigenvector...')
    _, p_eigen = gt.eigenvector(graph_gt, weight=weight_p)
    graph_gt.vertex_properties[SGT_EIGENVECTOR] = p_eigen
    if verbose:
        print('\t\tKatz...')
    p_kats = gt.pagerank(graph_gt, weight=weight_p)
    graph_gt.vertex_properties[SGT_KATZ] = p_kats
    # if verbose:
    #     print '\t\tHits...'
    # _, p_hits_aut, p_hits_hub = gt.hits(graph_gt, weight=weight_p)
    # graph_gt.vertex_properties[SGT_HITS_AUT] = p_hits_aut
    # graph_gt.vertex_properties[SGT_HITS_HUB] = p_hits_hub
    if verbose:
        print('\tClustering measures:')
    if verbose:
        print('\t\tDegree...')
    p_deg = graph_gt.degree_property_map(deg='total', weight=weight_p)
    graph_gt.vertex_properties[SGT_NDEGREE] = p_deg
    if verbose:
        print('\t\tLocal clustering coefficients...')
    p_localc = gt.local_clustering(graph_gt)
    graph_gt.vertex_properties[SGT_LOCAL_CLUST] = p_localc
    if verbose:
        print('\t\tExtended coefficients...')
    p_extc = gt.extended_clustering(graph_gt, max_depth=3)
    graph_gt.vertex_properties[SGT_EXT_CLUST_1] = p_extc[0]
    graph_gt.vertex_properties[SGT_EXT_CLUST_2] = p_extc[1]
    graph_gt.vertex_properties[SGT_EXT_CLUST_3] = p_extc[2]

    print('\tCompute minimum spanning tree...')
    graph.min_spanning_tree(SGT_MIN_SP_TREE, weight)
    # # TODO: to delete
    # import operator
    # graph_mcf.threshold_edges(SGT_MIN_SP_TREE, 0, operator.eq)

    print('\tUpdate subgraph relevance...')
    graph_mcf.compute_sgraph_relevance()

    if verbose:
        print('\tUpdating GraphMCF...')
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_PAGERANK, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_BETWEENNESS, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_BETWEENNESS, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_CLOSENESS, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_EIGENVECTOR, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_KATZ, up_index=False)
    # graph.add_prop_to_GraphMCF(graph_mcf, SGT_HITS_AUT, up_index=False)
    # graph.add_prop_to_GraphMCF(graph_mcf, SGT_HITS_HUB, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_NDEGREE, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_LOCAL_CLUST, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_EXT_CLUST_1, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_EXT_CLUST_2, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_EXT_CLUST_3, up_index=False)
    graph.add_prop_to_GraphMCF(graph_mcf, SGT_MIN_SP_TREE, up_index=True)

    if no_ret:
        if verbose:
            print('\tStoring the result.')
        path, stem = os.path.split(input_file)
        stem, _ = os.path.splitext(stem)
        graph_mcf.pickle(input_file)
        disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=True, edges=True),
                             path + '/' + stem + '_edges.vtp')
        disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=False, edges=True),
                             path + '/' + stem + '_edges_2.vtp')
        disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
                             path + '/' + stem + '_sch.vtp')
    else:
        return graph_mcf


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvnxi:w:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = None
    weight = None
    inv = False
    no_ret = True
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-w":
            weight = arg
        elif opt == "-x":
            inv = True
        elif opt == "-n":
            no_ret = False
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if input_file is None:
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool drawing a graph.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            if weight is not None:
                print('\tEdge weighting property: ' + weight)
                if inv:
                    print('\tEdge property inverted.')
            if not no_ret:
                print('\tThe result will be returned as an GraphMCF object.')
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        ret = do_local_topology(input_file, weight, inv, no_ret, verbose)


        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')

        return ret

if __name__ == "__main__":
    main(sys.argv[1:])