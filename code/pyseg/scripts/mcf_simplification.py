"""

    Script for GraphMCF simplifaction

    Input:  - File with a GraphMCF (.pkl)
            - Simplification parameters

    Output: - Simplified GraphMCF

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
import pyseg.pexceptions

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf>\n -o <output_dir> -(n|d) <vertices>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input Graph MCF in a pickle file. \n' + \
           '    -o <output_dir>: Name of the output directory.\n' + \
           '    -V <key_prop>(optional): property for vertex simplification, if None topological simplification.' + \
           '    -E <key_prop>(optional): property for edge simplification, compulsory if \'-a\' or \'-e\' present.' + \
           '    -n <int>(optional): final number of vertices, compulsory if \'-d\' not present' + \
           '    -a <int>(optional): final number of edges, compulsory if \'-e\' not present' + \
           '    -d <float>(optional): final density of vertices (vertex/nm), compulsory if \'-n\' not present' + \
           '    -e <float>(optional): final density of edges (edges/nm), compulsory if \'-a\' not present' + \
           '    -p <char>(optional): vertex simplification mode, if \'+\' is high (default), otherwise low.' + \
           '                         Low mode is not applicable for topological simplification.' + \
           '    -q <char>(optional): edge simplification mode, if \'+\' is high (default), otherwise low.' + \
           '                         Applicable only if \'-a\' or \'-e\' is present.' + \
           '    -m <file_name>(optional): mask for cropping the graph and computing volume.' + \
           '    -s <int>(option): (default 1) segmentation label for mask, only applicable if \'-m\' is present.' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_simplification(input_file, output_dir, prop_v, prop_e, v_num, e_num, v_den, e_den, v_mode, e_mode,
                      mask, lbl_s, verbose):

    if verbose:
        print('\tLoading the graph...')
    path, stem = os.path.split(input_file)
    stem, ext = os.path.splitext(stem)
    if ext == '.pkl':
        graph_mcf = unpickle_obj(input_file)
    else:
        print('\tERROR: ' + ext + ' is a non valid format.')
        sys.exit(4)

    w_mask = None
    if mask is not None:
        if verbose:
            print('\tMasking the graph...')
        w_mask = disperse_io.load_tomo(mask)
        w_mask = w_mask == lbl_s
        for v in graph_mcf.get_vertices_list():
            x, y, z = graph_mcf.get_vertex_coords(v)
            try:
                if not w_mask[int(round(x)), int(round(y)), int(round(z))]:
                    graph_mcf.remove_vertex(v)
            except IndexError:
                graph_mcf.remove_vertex(v)
        for e in graph_mcf.get_edges_list():
            x, y, z = graph_mcf.get_vertex_coords(e)
            try:
                if not w_mask[int(round(x)), int(round(y)), int(round(z))]:
                    graph_mcf.remove_edge(e)
            except IndexError:
                graph_mcf.remove_edge(e)

    if verbose:
        print('\tComputing graph global statistics (before simplification)...')
        nvv, nev, nepv = graph_mcf.compute_global_stat(mask=w_mask)
        print('\t\t-Number of vertices: ' + str(len(graph_mcf.get_vertices_list())))
        print('\t\t-Number of edges: ' + str(len(graph_mcf.get_edges_list())))
        print('\t\t-Vertex density: ' + str(round(nvv,5)) + ' nm^3')
        print('\t\t-Edge density: ' + str(round(nev,5)) + ' nm^3')
        print('\t\t-Edge/Vertex ratio: ' + str(round(nepv,5)))

    print('\tGraph simplification...')
    try:
        graph_mcf.graph_density_simp(v_num=v_num, e_num=e_num, v_den=v_den, e_den=e_den, v_prop=prop_v,
                                     e_prop=prop_e, v_mode=v_mode, e_mode=e_mode, mask=w_mask)
    except pexceptions.PySegInputWarning as e:
        print('WARNING: graph density simplification failed:')
        print('\t-' + e.get_message())
        # sys.exit(-2)

    if verbose:
        print('\tComputing graph global statistics (after simplification)...')
        nvv, nev, nepv = graph_mcf.compute_global_stat(mask=w_mask)
        print('\t\t-Number of vertices: ' + str(len(graph_mcf.get_vertices_list())))
        print('\t\t-Number of edges: ' + str(len(graph_mcf.get_edges_list())))
        print('\t\t-Vertex density: ' + str(round(nvv,5)) + ' nm^3')
        print('\t\t-Edge density: ' + str(round(nev,5)) + ' nm^3')
        print('\t\t-Edge/Vertex ratio: ' + str(round(nepv,5)))

    if verbose:
        print('\tUpdate properties...')
    graph_mcf.compute_sgraph_relevance()
    if prop_v is None:
        graph_mcf.compute_edge_curvatures()

    if verbose:
        print('\tStoring the result.')
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf.pickle(output_dir + '/' + stem + '_simp.pkl')
    disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=False, edges=True),
                         output_dir + '/' + stem + '_th_edges.vtp')
    disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=False, edges=True),
                         output_dir + '/' + stem + '_th_sch.vtp')
    seg_vtp = gauss_FE(graph_mcf.get_skel_vtp(mode='node'), sigma=1, size=graph_mcf.get_density().shape)
    disperse_io.save_numpy(seg_vtp, output_dir + '/' + stem + '_simp_gau.vti')
    seg_img = graph_mcf.print_vertices(img=None, property=STR_GRAPH_RELEVANCE, th_den=0)
    disperse_io.save_numpy(seg_img, output_dir + '/' + stem + '_simp_seg.vti')

################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvi:o:V:E:n:a:d:e:m:s:p:q:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = None
    output_dir = None
    prop_v = None
    prop_e = None
    v_num = None
    e_num = None
    v_den = None
    e_den = None
    v_mode = 'high'
    e_mode = 'high'
    mask = None
    lbl_s = 1
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-V":
            prop_v = arg
        elif opt == "-E":
            prop_e = arg
        elif opt == "-n":
            v_num = int(arg)
        elif opt == "-a":
            e_num = int(arg)
        elif opt == "-d":
            v_den = float(arg)
        elif opt == "-e":
            e_den = float(arg)
        elif opt == "-p":
            if arg != '+':
                v_mode = 'low'
        elif opt == "-q":
            if arg != '+':
                e_mode = 'low'
        elif opt == "-m":
            mask = arg
        elif opt == "-s":
            lbl_s = int(arg)
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if (input_file is None) or (output_dir is None) or ((v_num is None) and (v_den is None)):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('GraphMCF simplification.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            print('\tOutput dir: ' + output_dir)
            if mask is not None:
                print('\tMask file (label=' + str(lbl_s) + '): ' + mask)
            if prop_v is None:
                print('\tTopological simplification for vertices.')
            else:
                print('\tProperty with key ' + prop_v + ' with mode ' + v_mode + ' for vertex simplification.')
            if v_num is not None:
                print('\tTarget number of vertices: ' + str(v_num))
            else:
                print('\tTarget vertex density: ' + str(v_den) + ' vertex/nm')
            if e_num is not None:
                print('\tProperty with key ' + prop_e + ' with mode ' + e_mode + ' for edge simplification.')
                print('\tTarget number of edges: ' + str(e_num))
            elif e_den is not None:
                print('\tProperty with key ' + prop_e + ' with mode ' + e_mode + ' for edge simplification.')
                print('\tTarget edge density: ' + str(e_den) + ' edge/nm')
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_simplification(input_file, output_dir, prop_v, prop_e, v_num, e_num, v_den, e_den, v_mode, e_mode,
                          mask, lbl_s, verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])