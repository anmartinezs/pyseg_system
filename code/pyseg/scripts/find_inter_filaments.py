"""

    Script for finding filaments which though a GraphMCF (parent) interconnects nodes of a thresholded
    GraphMCF based of the parent GraphMCF

    Input:  - Parent GraphMCF (pkl)
            - Thresholded GraphMCF (pkl)
            - Filament parameters

    Output: - GraphMCF thresholded nodes and filaments
            - VTK files with segmentation and poly data

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import os
import sys
import vtk
import time
import math
import getopt
import operator
import pyseg as ps
import graph_tool.all as gt
from factory import unpickle_obj
from pyseg.filament.variables import *

try:
    import pickle as pickle
except:
    import pickle

################# Main declarations

prop_v_key = 'ifils'
SEED = 2
NO_SEED = 1

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -p <parent_pkl> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -p <string>: path to parent GraphMCF pickle file. \n' + \
           '    -o <string>: output directory. \n' + \
           '    -t <string>(optional): path to thresholded GraphMCF pickle file. \n' + \
           '    -b (optional): if present the graph is pre-filtered for getting Minimum Spanning Tree' + \
           '    -e <string>(optional): edge weighting property key, if None edge_length is selected. \n' + \
           '    -l <float>(optional): minimum filament length in nm, if None it is set to 0. \n' + \
           '    -L <float>(optional): maximum filament length in nm, if None it is set to infinity. \n' + \
           '    -n <float>(optional): number of sigmas for segmentation. \n' + \
           '    -c <int>(optional): curvature computation mode, valid: 1 Vertex (default), otherwise Path' + \
           '    -v (optional): verbose mode activated.'

################# Helping routines

def get_seed_ids(graph):

    vertices = graph.get_vertices_list()
    seed_ids = np.zeros(shape=len(vertices), dtype=np.int)

    for i, v in enumerate(vertices):
        v_id = v.get_id()
        if len(graph.get_vertex_neighbours(v_id)[0]) > 0:
            seed_ids[i] = v_id

    return seed_ids

def get_filaments(graph, seed_ids, prop_e_key, min_len, max_len):

    # Initialization
    net = list()
    graph_gt = ps.graph.GraphGT(graph).get_gt()
    prop_e = graph_gt.edge_properties[prop_e_key]
    prop_l = graph_gt.edge_properties[ps.globals.SGT_EDGE_LENGTH]
    prop_id = graph_gt.vertex_properties[ps.globals.DPSTR_CELL]
    prop_e_id = graph_gt.edge_properties[ps.globals.DPSTR_CELL]
    coords = list()
    vertices = list()
    for i, idx in enumerate(seed_ids):
        hold = gt.find_vertex(graph_gt, prop_id, idx)
        if len(hold) <= 0:
            print('WARNING: thresholded vertex with id ' + str(idx) + ' not found in parent GraphMCF!')
        elif len(hold) > 1:
            print('WARNING: thresholded vertex with id ' + str(idx) + ' is not unique in parent GraphMCF!')
        else:
            vertices.append(hold[0])
            coords.append(graph.get_vertex_coords(graph.get_vertex(idx)))

    # If sources are part of input GraphMCF distance computation can be boosted
    if prop_e_key == 'edge_length':
        dists_map = gt.shortest_distance(graph_gt, weights=prop_l)
        # Main loop
        nv = len(vertices)
        for i in range(nv):
            print('Processed ' + str(i+1) + ' of ' + str(nv))
            v_s = vertices[i]
            dists = dists_map[v_s].get_array()[i:]
            ids = i + np.where((dists > min_len) & (dists < max_len))[0]
            for idx in ids:
                v_path, e_path = gt.shortest_path(graph_gt, v_s, vertices[idx], weights=prop_e)
                v_list, e_list = list(), list()
                cont = 0
                for e in e_path:
                    v_list.append(prop_id[v_path[cont]])
                    e_list.append(prop_e_id[e])
                    cont += 1
                v_list.append(prop_id[v_path[cont]])
                net.append(ps.filament.FilamentUDG(v_list, e_list, graph))
    else:
        # Main loop
        nv = len(vertices)
        for i in range(nv):
            print('Processed ' + str(i+1) + ' of ' + str(nv))
            for j in range(i+1, nv):
                c_s, c_t = np.asarray(coords[i], dtype=np.float), np.asarray(coords[j], dtype=np.float)
                hold = c_s - c_t
                dist = math.sqrt((hold * hold).sum()) * graph.get_resolution()
                # continue
                if (dist > min_len) and (dist < max_len):
                    v_path, e_path = gt.shortest_path(graph_gt, vertices[i], vertices[j], weights=prop_e)
                    hold_len = 0.
                    if len(e_path) > 0:
                        is_fil = True
                        cont = 0
                        v_list, e_list = list(), list()
                        for e in e_path:
                            hold_len += prop_l[e]
                            if hold_len > max_len:
                                is_fil = False
                                break
                            v_list.append(prop_id[v_path[cont]])
                            e_list.append(prop_e_id[e])
                            cont += 1
                        if (hold_len >= min_len) and is_fil:
                            v_list.append(prop_id[v_path[cont]])
                            net.append(ps.filament.FilamentUDG(v_list, e_list, graph))

    return net

def thres_no_filaments(graph, net):

    prop_v_id = graph.add_prop(prop_v_key, 'int', 1)
    lut_v_ids = np.zeros(shape=graph.get_nid(), dtype=np.int)
    for fil in net:
        fverts = fil.get_vertices()
        nv = len(fverts)
        lut_v_ids[fverts[0].get_id()] = 1
        lut_v_ids[fverts[nv-1].get_id()] = 2
        for i in range(1, nv - 1):
            lut_v_ids[fverts[i].get_id()] = 3

    for v in graph.get_vertices_list():
        v_id = v.get_id()
        val = lut_v_ids[v_id]
        if (val == 1) or (val == 2):
            graph.set_prop_entry_fast(prop_v_id, (SEED,), v_id, 1)
        elif val == 3:
            graph.set_prop_entry_fast(prop_v_id, (NO_SEED,), v_id, 1)
        else:
            graph.remove_vertex(v)

# vertex: if True (default) just vertices are stores, False edge points paths are stored
def write_net_vtp(net, vertex=True):

    # Initialization
    point_id = 0
    cell_id = 0
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    cell_data = vtk.vtkIntArray()
    cell_data.SetNumberOfComponents(1)
    cell_data.SetName(STR_CELL)
    len_data = vtk.vtkFloatArray()
    len_data.SetNumberOfComponents(1)
    len_data.SetName(STR_FIL_LEN)
    ct_data = vtk.vtkFloatArray()
    ct_data.SetNumberOfComponents(1)
    ct_data.SetName(STR_FIL_CT)
    tt_data = vtk.vtkFloatArray()
    tt_data.SetNumberOfComponents(1)
    tt_data.SetName(STR_FIL_TT)
    sin_data = vtk.vtkFloatArray()
    sin_data.SetNumberOfComponents(1)
    sin_data.SetName(STR_FIL_SIN)
    apl_data = vtk.vtkFloatArray()
    apl_data.SetNumberOfComponents(1)
    apl_data.SetName(STR_FIL_APL)
    ns_data = vtk.vtkFloatArray()
    ns_data.SetNumberOfComponents(1)
    ns_data.SetName(STR_FIL_NS)
    bns_data = vtk.vtkFloatArray()
    bns_data.SetNumberOfComponents(1)
    bns_data.SetName(STR_FIL_BNS)

    # Write lines
    if vertex:
        fun_name = 'get_vertex_coords'
    else:
        fun_name = 'get_curve_coords'
    for i, f in enumerate(net):
        coords = getattr(f, fun_name)()
        lines.InsertNextCell(coords.shape[0])
        for c in coords:
            points.InsertPoint(point_id, c[0], c[1], c[2])
            lines.InsertCellPoint(point_id)
            point_id += 1
        cell_id += 1
        cell_data.InsertNextTuple((cell_id,))
        len_data.InsertNextTuple((f.get_length(),))
        ct_data.InsertNextTuple((f.get_total_k(),))
        tt_data.InsertNextTuple((f.get_total_t(),))
        sin_data.InsertNextTuple((f.get_sinuosity(),))
        apl_data.InsertNextTuple((f.get_apex_length(),))
        ns_data.InsertNextTuple((f.get_total_ns(),))
        bns_data.InsertNextTuple((f.get_total_bs(),))


    # Poly building
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)
    poly.GetCellData().AddArray(cell_data)
    poly.GetCellData().AddArray(len_data)
    poly.GetCellData().AddArray(ct_data)
    poly.GetCellData().AddArray(tt_data)
    poly.GetCellData().AddArray(sin_data)
    poly.GetCellData().AddArray(apl_data)
    poly.GetCellData().AddArray(ns_data)
    poly.GetCellData().AddArray(bns_data)

    return poly

################# Work routine

def do_find_inter_filaments(parent_file, thres_file, output_dir, prop_e_key, min_len, max_len, nsig,
                            mst, mc, verbose):

    # Initialization
    if verbose:
        print('\tLoading graphs...')
    path, stem = os.path.split(parent_file)
    pstem, _ = os.path.splitext(stem)
    path, stem = os.path.split(thres_file)
    tstem, _ = os.path.splitext(stem)
    graph = unpickle_obj(parent_file)
    tgraph = unpickle_obj(thres_file)

    if min_len <= 0:
        if verbose:
            print('\tDelete isolated vertices...')
        for v in graph.get_vertices_list():
            if len(graph.get_vertex_neighbours(v.get_id())[1]) <= 0:
                graph.remove_vertex(v)

    if mst:
        if verbose:
            print('\tCompute minimum spanning tree...')
        graph_gt = ps.graph.GraphGT(graph)
        graph_gt.min_spanning_tree(ps.globals.SGT_MIN_SP_TREE, prop_e_key)
        graph_gt.add_prop_to_GraphMCF(graph, ps.globals.SGT_MIN_SP_TREE, up_index=False)
        graph.threshold_edges(ps.globals.SGT_MIN_SP_TREE, 0, operator.eq)

    if verbose:
        print('\tGetting the ids of seed nodes from the thresholded graph...')
    seed_ids = get_seed_ids(tgraph)

    if verbose:
        print('\tFinding filaments in parent graph...')
    net = get_filaments(graph, seed_ids, prop_e_key, min_len, max_len)

    if verbose:
        print('\tThresholding parent tomogram...')
    thres_no_filaments(graph, net)

    if verbose:
        print('\tWriting network poly data...')
    netv_vtp = write_net_vtp(net, vertex=True)
    netp_vtp = write_net_vtp(net, vertex=False)

    if verbose:
        print('\tStoring the result...')
    graph.pickle(output_dir + '/' + pstem + '_' + prop_v_key + '.pkl')
    seg = graph.print_vertices(img=None, property=prop_v_key, th_den=nsig)
    ps.disperse_io.save_numpy(seg, output_dir + '/' + tstem + '_' + prop_v_key + '_seg.vti')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + pstem + '_' + prop_v_key + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                            output_dir + '/' + pstem + '_' + prop_v_key + '_sch.vtp')
    ps.disperse_io.save_vtp(netv_vtp, output_dir + '/' + tstem + '_net_v.vtp')
    ps.disperse_io.save_vtp(netp_vtp, output_dir + '/' + tstem + '_net_p.vtp')
    if verbose:
        print('\tResults stored in ' + output_dir)


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvbp:t:o:e:l:L:n:c:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    parent_file = None
    thres_file = None
    output_dir = None
    prop_e_key = ps.globals.SGT_EDGE_LENGTH
    min_len = 0.
    max_len = np.inf
    nsig = None
    mst = False
    mc = 1
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-p":
            parent_file = arg
        elif opt == "-t":
            thres_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-e":
            prop_e_key = arg
        elif opt == "-l":
            min_len = float(arg)
        elif opt == "-L":
            max_len = float(arg)
        elif opt == "-n":
            nsig = float(arg)
        elif opt == "-b":
            mst = True
        elif opt == "-c":
            mc = int(arg)
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if (parent_file is None) or (output_dir is None):
        print(usage_msg)
        sys.exit(2)
    else:
        if thres_file is None:
            thres_file = parent_file
        # Print init message
        if verbose:
            print('Running tool for finding internode filaments.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            if parent_file == thres_file:
                print('\tSelf mode in GraphMCF file: ' + parent_file)
            else:
                print('\tParent GraphMCF file: ' + parent_file)
                print('\tThresholded GraphMCF file: ' + thres_file)
            print('\tOutput directory: ' + output_dir)
            print('\tEdge weighting property key: ' + prop_e_key)
            print('\tMinimum filament length: ' + str(min_len) + ' nm')
            print('\tMaximum filament length: ' + str(max_len) + ' nm')
            if nsig is not None:
                print('\tNumber of sigmas for segmantation: ' + str(nsig))
            if mst:
                print('\tMaximum spanning tree filtration.')
            if mc == 1:
                print('\tVertex mode for curvature computation.')
            else:
                print('\tPath mode for curvature computation.')
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_find_inter_filaments(parent_file, thres_file, output_dir, prop_e_key, min_len, max_len, nsig,
                                mst, mc, verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])