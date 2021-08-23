"""

    Script for segmenting the filaments in a tomogram without assuming any geometric constrain

    Input:  - Original tomogram (density map)
            - Mask tomogram (0-fg, otherwise-bg)

    Output: - vtkPolyData with the filaments
            - Segmentation tomogram

"""

__author__ = 'Antonio Martinez-Sanchez'


################# Package import

import sys
import time
import getopt
import disperse_io
import operator
from graph import GraphMCF
from globals import *
from factory import unpickle_obj
import graph_tool.all as gt
import matplotlib.pyplot as plt

try:
    import pickle as pickle
except:
    import pickle

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_tomogram> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input 3D density image in MRC, EM or FITS formats. \n' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored. If it already exists then will be cleared before running the tool. \n' + \
           '    -m <file_name>(optional): input 3D image with the mask for data (0-foreground).\n' + \
           '    -d (optional): forces to re-update all DisPerSe data. \n' + \
           '    -r (optional): voxel resolution in nm of input data, default 1.68. \n' + \
           '    -C (optional): threshold for cutting low persistence structures. \n' + \
           '    -L (optional): threshold for low density maxima. \n' + \
           '    -H (optional): threshold for high density minima. \n' + \
           '    -R (optional): threshold for relevance. \n' + \
           '    -S (optional): threshold for length. \n' + \
           '    -s (optional): number of sigmas used for segmentation.' + \
           '    -f <em, mrc, fits or vti>: segmentation output format (default mrc). \n' + \
           '    -v (optional): verbose mode activated.'

LOCAL_DEF_RES = 1.68

################# Work routine

def do_det_filament(input_file, output_dir, fmt, mask_file=None, res=1, cut_t=None, low_t=None,
                    high_t=None, rel_t=None, len_t=None, f_update=False, sig=None, verbose=False):

    # Initialization
    if verbose:
        print('\tInitializing...')
    work_dir = output_dir + '/disperse'
    disperse = disperse_io.DisPerSe(input_file, work_dir)
    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    # Down skeleton
    disperse.set_dump_arcs(-1)
    if cut_t is not None:
        disperse.set_cut(cut_t)
    if mask_file != '':
        disperse.set_mask(mask_file)
    density = disperse_io.load_tomo(input_file)

    # Disperse
    if verbose:
        print('\tRunning DisPerSe...')
    if f_update:
        disperse.clean_work_dir()
        disperse.mse(no_cut=False, inv=False)
    skel = disperse.get_skel()
    manifolds = disperse.get_manifolds(no_cut=False, inv=False)

    # Build the PSD ArcGraph
    pkl_sgraph = work_dir + '/skel_graph.pkl'
    if f_update or (not os.path.exists(pkl_sgraph)):
        if verbose:
            print('\tBuilding graph...')
        graph = GraphMCF(skel, manifolds, density)
        graph.set_resolution(res)
        graph.build_from_skel()
        if verbose:
            print('\tAdding geometry...')
        graph.build_vertex_geometry()
        if verbose:
            print('\tPickling...')
        graph.pickle(pkl_sgraph)
        _, stem = os.path.split(input_file)
        stem, _ = os.path.splitext(stem)
        disperse_io.save_vtp(graph.get_vtp(), output_dir + '/' + stem + '_graph.vtp')
    else:
        if verbose:
            print('\tUnpickling graph...')
        graph = unpickle_obj(pkl_sgraph)

    # TODO: PROVISIONAL

    # Adding the scalar field
    # field_path = '/home/martinez/workspace/disperse/data/felix/network/in/t44_cc.fits'
    # field = disperse_io.load_tomo(field_path)
    # graph.add_scalar_field(field, 'cross_corr')

    # TODO: just for Anselm's data
    for v in graph.get_vertices_list():
        x, y, z = graph.get_vertex_coords(v)
        if (x < 9) or (y < 9) or (z < 9) or (x > density.shape[0]-8) or \
            (y > density.shape[1]-8) or (z > density.shape[2]-8):
            graph.remove_vertex(v)

    if verbose:
        print('\tThresholding the graph...')
    # hold_graph = pcopy.deepcopy(graph)
    if low_t is not None:
        if verbose:
            print('\t\tLow vertex...')
        graph.threshold_vertices(STR_FIELD_VALUE, low_t, operator.gt)
        # TODO: just for Anselm's data
        # graph.threshold_vertices(STR_FIELD_VALUE, low_t, operator.lt)
    if high_t is not None:
        if verbose:
            print('\t\tHigh edge...')
        graph.threshold_edges(STR_FIELD_VALUE, high_t, operator.gt)
    if len_t is not None:
        if verbose:
            print('\t\tDiameters...')
        graph.compute_diameters()
        graph.threshold_vertices(STR_GRAPH_DIAM, len_t, operator.lt)
    if rel_t is not None:
        if verbose:
            print('\t\tRelevance...')
        graph.compute_sgraph_relevance()
        graph.threshold_vertices(STR_GRAPH_RELEVANCE, rel_t, operator.lt)


    if verbose:
        print('\tSegmentation...')
    seg = graph.print_vertices(property=DPSTR_CELL, th_den=sig)
    # seg = graph.print_vertices(property='cross_corr', th_den=sig)

    # TODO: PROVISIONAL

    if verbose:
        print('\tGenerating GT graphs')
    gt_graph, vertices_gt = graph.get_gt(id_arr=True)
    try:
        gt_fname = output_dir + '/' + stem + '_gt.pdf'
    except Exception:
        _, stem = os.path.split(input_file)
        stem, _ = os.path.splitext(stem)
        gt_fname = output_dir + '/' + stem + '_gt.pdf'
    pos = gt.sfdp_layout(g=gt_graph)
                         # vweight=gt_graph.vertex_properties[STR_FIELD_VALUE],
                         # eweight=gt_graph.edge_properties[SGT_EDGE_LENGTH],
                         # groups=gt_graph.vertex_properties[STR_GRAPH_ID])
    hold = gt_graph.edge_properties[STR_FIELD_VALUE]
    hold_arr = lin_map(hold.get_array()[:], lb=1, ub=0)
    hold.get_array()[:] = hold_arr
    # gt_graph.edge_properties[STR_FIELD_VALUE].get_array()[:] = hold
    # hold = att_bet.get_array()[:]
    # hold = gt_graph.edge_properties[SGT_EDGE_LENGTH].get_array()[:]
    # hold = lin_map(hold, lb=0, ub=1)
    # color1 = plt.cm.hot(hold)
    ###########################################
    v_bet, e_bet = gt.betweenness(gt_graph, weight=hold)
    graph.add_prop('betweeness', 'float', 1)
    for v in graph.get_vertices_list():
        if v_bet[vertices_gt[v.get_id()]] > 0.03:
          graph.set_prop_entry('betweeness', v_bet[vertices_gt[v.get_id()]], v.get_id())
    #graph.threshold_vertices('betweeness', 0.01, operator.lt)
    #seg2 = graph.print_vertices(property=DPSTR_CELL, th_den=sig)
    #disperse_io.save_numpy(seg2, output_dir + '/' + stem + '_seg2' + fmt)
    ###################################
    # hold = gt_graph.vertex_properties['cross_corr'].get_array()[:]
    # hold = lin_map(hold, lb=0, ub=1)
    # gt_graph.vertex_properties[STR_FIELD_VALUE].get_array()[:] = hold
    # color2 = plt.cm.jet(hold)
    # v_color = gt_graph.new_vertex_property('vector<double>')
    # for v in gt_graph.vertices():
    #     v_color[v] = color2[gt_graph.vertex_index[v], :]
    # e_color = gt_graph.new_edge_property('vector<double>')
    # for e in gt_graph.edges():
    #     e_color[e] = color1[gt_graph.edge_index[e], :]
    gt.graph_draw(g=gt_graph, pos=pos, output=gt_fname, #,
                  #vertex_size=gt_graph.vertex_properties[STR_FIELD_VALUE],
                  vertex_fill_color=v_bet,
                  # vertex_fill_color=v_color,
                  edge_color=e_bet)
                  #edge_pen_width=gt_graph.edge_properties[STR_FIELD_VALUE])

    if verbose:
        print('\tStoring the result...')
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')
    disperse_io.save_vtp(graph.get_vtp(av_mode=True), output_dir + '/' + stem + '_thres.vtp')
    disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                         output_dir + '/' + stem + '_edges.vtp')
    disperse_io.save_numpy(seg, output_dir + '/' + stem + '_seg' + fmt)


################# Main call

def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hvdi:o:m:r:C:L:H:R:S:f:s:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = ''
    output_dir = ''
    mask_file = ''
    f_update = False
    res = LOCAL_DEF_RES
    cut_t = None
    low_t = None
    high_t = None
    rel_t = None
    len_t = None
    fmt = '.mrc'
    verbose = False
    sig = None
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-m":
            mask_file = arg
        elif opt == "-d":
            f_update = True
        elif opt == "-r":
            res = float(arg)
        elif opt == "-C":
            cut_t = float(arg)
        elif opt == "-L":
            low_t = float(arg)
        elif opt == "-H":
            high_t = float(arg)
        elif opt == "-R":
            rel_t = float(arg)
        elif opt == "-S":
            len_t = float(arg)
        elif opt == "-f":
            fmt = '.' + arg
        elif opt == "-s":
            sig = float(arg)
        elif opt == "-v":
            verbose = True

    if (input_file == '') or (output_dir == ''):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool for detecting filaments in a tomogram.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            print('\tOutput directory: ' + output_dir)
            if mask_file == '':
                print('\tMask not used.')
            else:
                print('\tMask file: ' + mask_file)
            if f_update:
                print('\tUpdate disperse: yes')
            else:
                print('\tUpdate disperse: no')
            print('\tResolution: ' + str(res) + ' nm')
            if cut_t is not None:
                print('\tPersistence threshold: ' + str(cut_t))
            if low_t is not None:
                print('\tLow density maxima threshold: ' + str(low_t))
            if high_t is not None:
                print('\tHigh density minima threshold: ' + str(high_t))
            if rel_t is not None:
                print('\tRelevance threshold: ' + str(rel_t))
            if len_t is not None:
                print('\tLength threshold: ' + str(len_t))
            print('\tOutput segmentation format ' + fmt)
            print('\n')

        # Do the job
        if verbose:
            print('Starting...')
        do_det_filament(input_file, output_dir, fmt, mask_file,  res, cut_t, low_t, high_t, rel_t,
                        len_t, f_update, sig, verbose)

        if verbose:
            print(cmd_name + ' successfully executed.')

if __name__ == "__main__":
    main(sys.argv[1:])

