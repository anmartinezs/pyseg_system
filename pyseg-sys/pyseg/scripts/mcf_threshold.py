"""

    Script for thresholding GraphMCF instances according to its properties

    Input:  - File with a GraphMCF (.pkl)

    Output: - Output view (interactive window or .pdf file)

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
from pyseg.globals import *
from pyseg.factory import unpickle_obj
import pyseg.xml_io as pio
import pyseg.disperse_io as disperse_io

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input Graph MCF in a pickle file. \n' + \
           '    -o <output_dir>: Name of the output directory.\n' + \
           '    -f <filter_file>: xml file with thresholding operations. Definitive format is ' + \
           '                      of being defined.' + \
           '    -s <file_name>(optional): scalar field for generating a skeleton based segmentation' + \
           '    -a (optional): if present arcs are simplified.' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_mcf_filter(input_file, output_dir, flt_file, scal_fld, arc_simp, verbose=False):

    if verbose:
        print '\tLoading the graph...'
    path, stem = os.path.split(input_file)
    stem, ext = os.path.splitext(stem)
    if ext == '.pkl':
        graph_mcf = unpickle_obj(input_file)
    else:
        print '\tERROR: ' + ext + ' is a non valid format.'
        sys.exit(4)

    if verbose:
        print '\tFiltering the graph...'
    parser = pio.Threshold(flt_file)
    keys = parser.get_key_prop_list()
    thres = parser.get_threshold_list()
    opers = parser.get_operator_list()
    for i, key in enumerate(keys):
        print '\t\tFiltering property ' + key + ' with threshold ' + str(thres[i]) + ' ...'
        if key == STR_V_PER:
            graph_mcf.topological_simp(thres[i])
            # Update edge properties
            graph_mcf.compute_edges_length(SGT_EDGE_LENGTH, 1, 1, 1, False)
            graph_mcf.compute_edges_sim()
            graph_mcf.compute_edges_integration(field=False)
            graph_mcf.compute_edge_filamentness()
        elif 'edge_' in key:
            graph_mcf.threshold_edges(key, thres[i], opers[i])
        else:
            graph_mcf.threshold_vertices(key, thres[i], opers[i])
    # TO DELETE: mSpecific for actin
    # import operator
    # graph_mcf.threshold_edges('field_value', 0.6, operator.gt)
    # graph_mcf.edge_simp(1)

    if arc_simp:
        if verbose:
            print '\tArcs simplification...'
        graph_mcf.arc_simp()

    if verbose:
        print '\tUpdate subgraph relevance...'
    graph_mcf.compute_sgraph_relevance()

    if verbose:
        print '\tStoring the result.'
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf.pickle(output_dir + '/' + stem + '_th.pkl')
    disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=False, edges=True),
                         output_dir + '/' + stem + '_th_edges.vtp')
    disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=False, edges=True),
                         output_dir + '/' + stem + '_th_sch.vtp')

    if scal_fld is not None:
        if verbose:
            print '\tStoring segmentation...'
        disperse_io.save_numpy(graph_mcf.print_vertices(th_den=-1),
                               output_dir + '/' + stem + '_th_seg.vti')
        scalar_field = disperse_io.load_tomo(scal_fld)
        poly_vtp = graph_mcf.get_sfield_vtp(scalar_field, mode='edge')
        disperse_io.save_vtp(poly_vtp, output_dir + '/' + stem + '_th_arc.vtp')
        disperse_io.save_numpy(gauss_FE(poly_vtp, prop_key='scalar_field', sigma=2,
                                        size=scalar_field.shape),
                               output_dir + '/' + stem + '_th_gau.vti')

################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvai:o:f:s:")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    input_file = None
    output_dir = None
    verbose = False
    scal_fld = None
    flt_file = None
    arc_simp = False
    for opt, arg in opts:
        if opt == '-h':
            print usage_msg
            print help_msg
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-f":
            flt_file = arg
        elif opt == "-s":
            scal_fld = arg
        elif opt == "-v":
            verbose = True
        elif opt == "-a":
            arc_simp = True
        else:
            print 'Unknown option ' + opt
            print usage_msg
            sys.exit(3)

    if (input_file is None) or (output_dir is None) or (flt_file is None):
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
            print '\tOutput dir: ' + output_dir
            print '\tFilter file: ' + flt_file
            if arc_simp:
                print '\tArcs simplification activated.'
            if scal_fld is not None:
                print '\tSegmentation will be stored, scalar field: ' + scal_fld
            print ''

        # Do the job
        if verbose:
            print 'Starting...'
        do_mcf_filter(input_file, output_dir, flt_file, scal_fld, arc_simp, verbose)

        if verbose:
            print cmd_name + ' successfully executed. (' + time.strftime("%c") + ')'


if __name__ == "__main__":
    main(sys.argv[1:])