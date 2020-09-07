"""

    Script for studying the averaged neighbourhood of a set of vertices

    Input:  - GraphMCF (pickle file)
            - Reference surface

    Output: - 3D graphs (VTK files) with the averaged information
            - Vertex normal to reference surface

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import sys
import time
import getopt
from globals import *
from factory import unpickle_obj
from pyseg.vtk_ext import vtkClosestPointAlgorithm

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf> -s <ref_surf_vtp>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input Graph MCF in a pickle file. \n' + \
           '    -s <file_name>: input reference surface, as a cloud of points, ' \
           'for aligning in a VTK file. \n' + \
           '    -p <string_key>(optional): property used for vertex selection.' + \
           '    -w <int>(optional): if ''s'' present, value of the property for vertex ' + \
           'selection. (default 0)' + \
           '    -d <int>(optional): maximum degree of neighbourhood (default 1).' + \
           'NOT IMPLEMENTED YET.' + \
           '    -v (optional): verbose mode activated.'

################# Global variables

Z_axis = np.asarray((0, 0, 1))

################# Helping functions

# For transforming between coordinate spaces
# n: normal, it is assumed that has modulus 1
# p_v: reference vertex coordinates
# p_n: neighbour vertex coordinates
def trans_coord(n, p_v, p_n):

    # Translate p_v to origin
    p_n_1 = p_n - p_v

    # Computing rotation axis
    r = np.cross(n, Z_axis)
    r_norm = math.sqrt(np.sum(r * r))
    if r_norm > 0:
        r /= r_norm
    else:
        r = 0

    # Getting rotation angle
    phi = (-1) * np.arccos(np.sum(n * Z_axis))
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    # return p_n_1

    # Rotation (Rodrigues formula)
    return p_n_1*c_phi + np.cross(r, p_n_1)*s_phi + r*np.sum(r*p_n_1)*(1-c_phi)

def gen_vertices_graph(graph, vertices, prop_key=None, prop_val=None, prop_field=None):

    # Initialization
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    not_visited_neighs = np.ones(shape=graph.get_nid(), dtype=np.bool)
    key_id = graph.get_prop_id(STR_CLOUD_NORMALS)
    data_type = graph.get_prop_type(key_id=key_id)
    data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
    if (prop_key is not None) and (prop_val is not None):
        key_key_id = graph.get_prop_id(prop_key)
        data_key_type = graph.get_prop_type(key_id=key_key_id)
        data_key_type = disperse_io.TypesConverter().gt_to_numpy(data_key_type)
    array = vtk.vtkFloatArray()
    array.SetNumberOfComponents(1)
    array.SetName(STR_VGPH)
    if prop_field is not None:
        key_prop_id = graph.get_prop_id(prop_field)
        data_prop_type = graph.get_prop_type(key_id=key_prop_id)
        data_prop_type = disperse_io.TypesConverter().gt_to_numpy(data_prop_type)
        n_comp = graph.get_prop_ncomp(key_id=key_prop_id)
        array.SetNumberOfComponents(n_comp)

    # Loop for input vertices
    for v in vertices:
        v_coord = graph.get_vertex_coords(v)
        norm = graph.get_prop_entry_fast(key_id, v.get_id(), 3, data_type)
        norm = np.asarray(norm, dtype=np.float)
        v_id = v.get_id()
        neighs, _ = graph.get_vertex_neighbours(v_id)
        for n in neighs:
            n_id = n.get_id()
            if (prop_key is not None) and (prop_val is not None):
                t = graph.get_prop_entry_fast(key_key_id, n_id, 1, data_key_type)
                if t[0] == prop_val:
                    continue
            if not_visited_neighs[n_id]:
                n_coord = graph.get_vertex_coords(n)
                r_coord = trans_coord(np.asarray(norm), np.asarray(v_coord),
                                      np.asarray(n_coord))
                point_id = points.InsertNextPoint(r_coord)
                cells.InsertNextCell(1)
                cells.InsertCellPoint(point_id)
                not_visited_neighs[n_id] = False
                if prop_field is None:
                    array.InsertNextTuple1(1.0)
                else:
                    t = graph.get_prop_entry_fast(key_prop_id, n_id, n_comp, data_prop_type)
                    array.InsertNextTuple(t)

    # Building the VTK file
    poly.SetPoints(points)
    poly.SetVerts(cells)
    poly.GetPointData().AddArray(array)

    return poly

def gen_arcs_graph(graph, vertices, prop_key=None, prop_val=None, prop_field=None):

    # Initialization
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    skel = graph.get_skel()
    not_visited_neighs = np.ones(shape=skel.GetNumberOfPoints(), dtype=np.bool)
    key_id = graph.get_prop_id(STR_CLOUD_NORMALS)
    data_type = graph.get_prop_type(key_id=key_id)
    data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
    array = vtk.vtkFloatArray()
    array.SetNumberOfComponents(1)
    array.SetName(STR_AGPH)
    if (prop_key is not None) and (prop_val is not None):
        key_key_id = graph.get_prop_id(prop_key)
        data_key_type = graph.get_prop_type(key_id=key_key_id)
        data_key_type = disperse_io.TypesConverter().gt_to_numpy(data_key_type)
    if prop_field is not None:
        key_prop_id = graph.get_prop_id(prop_field)
        data_prop_type = graph.get_prop_type(key_id=key_prop_id)
        data_prop_type = disperse_io.TypesConverter().gt_to_numpy(data_prop_type)
        n_comp = graph.get_prop_ncomp(key_id=key_prop_id)
        array.SetNumberOfComponents(n_comp)

    # Loop for input vertices
    for v in vertices:
        v_coord = graph.get_vertex_coords(v)
        v_id = v.get_id()
        norm = graph.get_prop_entry_fast(key_id, v.get_id(), 3, data_type)
        norm = np.asarray(norm, dtype=np.float)
        for a in v.get_arcs():
            # Getting neighbour
            edge = graph.get_edge(a.get_sad_id())
            if edge is None:
                continue
            n_id = edge.get_source_id()
            if n_id == v_id:
                n_id = edge.get_target_id()
                if n_id == v_id:
                    continue
            if (prop_key is not None) and (prop_val is not None):
                t = graph.get_prop_entry_fast(key_key_id, n_id, 1, data_key_type)
                if t[0] == prop_val:
                    continue
            # Vertex side
            for i in range(a.get_npoints()):
                hold_id = a.get_point_id(i)
                if not_visited_neighs[hold_id]:
                    a_coord = skel.GetPoint(hold_id)
                    r_coord = trans_coord(np.asarray(norm), np.asarray(v_coord),
                                          np.asarray(a_coord))
                    point_id = points.InsertNextPoint(r_coord)
                    cells.InsertNextCell(1)
                    cells.InsertCellPoint(point_id)
                    not_visited_neighs[hold_id] = False
                    if prop_field is None:
                        array.InsertNextTuple1(1.0)
                    else:
                        t = graph.get_prop_entry_fast(key_prop_id, v_id, n_comp, data_prop_type)
                        array.InsertNextTuple(t)
            # Neighbour side
            n = graph.get_vertex(n_id)
            for na in n.get_arcs():
                # if na.get_sad_id() == a.get_sad_id():
                for i in range(na.get_npoints()):
                    hold_id = na.get_point_id(i)
                    if not_visited_neighs[hold_id]:
                        a_coord = skel.GetPoint(hold_id)
                        r_coord = trans_coord(np.asarray(norm), np.asarray(v_coord),
                                              np.asarray(a_coord))
                        point_id = points.InsertNextPoint(r_coord)
                        cells.InsertNextCell(1)
                        cells.InsertCellPoint(point_id)
                        not_visited_neighs[hold_id] = False
                        if prop_field is None:
                            array.InsertNextTuple1(1.0)
                        else:
                            t = graph.get_prop_entry_fast(key_prop_id, n_id, n_comp, data_prop_type)
                            array.InsertNextTuple(t)

    # Building the VTK file
    poly.SetPoints(points)
    poly.SetVerts(cells)
    poly.GetPointData().AddArray(array)

    return poly

################# Work routine

def do_vert_neighs(input_file, surf_file, prop_key, prop_val, deg_n, verbose):

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
        print('\tLoading the surface...')
    try:
        surf = disperse_io.load_poly(surf_file)
    except:
        print('\tERROR: surface ' + surf_file + ' could not be loaded.')
        sys.exit(4)

    if verbose:
        print('\tPicking up input vertices...')
    if prop_key is not None:
        hold_vertices = graph_mcf.get_vertices_list()
        vertices = list()
        key_id = graph_mcf.get_prop_id(prop_key)
        data_type = graph_mcf.get_prop_type(key_id=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
        for v in hold_vertices:
            t = graph_mcf.get_prop_entry_fast(key_id, v.get_id(), 1, data_type)
            if t[0] == prop_val:
                vertices.append(v)
    else:
        vertices = graph_mcf.get_vertices_list()

    if verbose:
        print('\tComputing vertices normal...')
    array_norm = None
    for i in range(surf.GetPointData().GetNumberOfArrays()):
        if surf.GetPointData().GetArrayName(i) == STR_CLOUD_NORMALS:
            array_norm = surf.GetPointData().GetArray(i)
    if array_norm is None:
        print('\tERROR: surface ' + surf_file + ' does not contain normals.')
        sys.exit(4)
    graph_mcf.add_prop(STR_CLOUD_NORMALS, 'float', 3, def_val=-1)
    key_n_id = graph_mcf.get_prop_id(STR_CLOUD_NORMALS)
    filter_dst = vtkClosestPointAlgorithm()
    filter_dst.SetInputData(surf)
    for v in vertices:
        filter_dst.initialize()
        v_id = v.get_id()
        x, y, z = graph_mcf.get_vertex_coords(v)
        c_point_id = filter_dst.evaluate_id(x, y, z)
        n_coords = array_norm.GetTuple(c_point_id)
        n_coords = np.asarray(n_coords)
        mod = math.sqrt(n_coords[0]*n_coords[0] + n_coords[1]*n_coords[1] +
                        n_coords[2]*n_coords[2])
        if mod > 0:
            n_coords /= mod
        else:
            n_coords = (0, 0, 0)
        graph_mcf.set_prop_entry_fast(key_n_id, tuple(n_coords), v_id, 3)

    if verbose:
        print('\tGenerating vertices graph...')
    v_graph = gen_vertices_graph(graph_mcf, vertices, prop_key=prop_key, prop_val=prop_val,
                                 prop_field=STR_FIELD_VALUE_INV)
    v_tomo = gauss_FE(v_graph, STR_FIELD_VALUE_INV, sigma=3, size=None)

    if verbose:
        print('\tGenerating arcs graph...')
    a_graph = gen_arcs_graph(graph_mcf, vertices, prop_key=prop_key, prop_val=prop_val,
                             prop_field=STR_FIELD_VALUE_INV)
    a_tomo = gauss_FE(a_graph, STR_FIELD_VALUE_INV, sigma=3, size=None)

    if verbose:
        print('\tStoring the result in ' + input_file)
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf.pickle(input_file)
    disperse_io.save_vtp(v_graph, path + '/' + stem + '_vgph.vtp')
    disperse_io.save_vtp(a_graph, path + '/' + stem + '_agph.vtp')
    disperse_io.save_numpy(v_tomo, path + '/' + stem + '_vgph.vti')
    disperse_io.save_numpy(a_tomo, path + '/' + stem + '_agph.vti')
    disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=True, edges=True),
                         path + '/hold.vtp')

################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvi:s:p:w:d:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = None
    surf_file = None
    prop_key = None
    prop_val = 0
    deg_n = 1
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-s":
            surf_file = arg
        elif opt == "-p":
            prop_key = arg
        elif opt == "-w":
            prop_val = int(arg)
        elif opt == "-d":
            deg_n = int(arg)
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if (input_file is None) or (surf_file is None):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool averaging set of vertices neighbourhood.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput graph file: ' + input_file)
            print('\tInput reference surface file: ' + surf_file)
            print('\tMaximum Neighbourhood degree: ' + str(deg_n))
            print('\tProperty for selection ' + str(prop_key) + ' with value ' + str(prop_val))
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_vert_neighs(input_file, surf_file, prop_key, prop_val, deg_n, verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])
