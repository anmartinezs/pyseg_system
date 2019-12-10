"""
Set of utils for helping factory classes

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 16.09.14
"""

__author__ = 'martinez'
__version__ = "$Revision: 002 $"

try:
    import graph
except:
    import pyseg.graph
try:
    import disperse_io
except:
    import pyseg.disperse_io
# import graph as gp
try:
    from globals import *
except:
    from pyseg.globals import *

# For applyiing a distance filter to the geometries of a Graph
# sgraph: Graph for being filtered
# dist_filt: vtk_ext.vtkClosestPointAlgorithm filter for evaluating the distances
# threshold: threshold applied to the scalar filed
# operator: function applied for thresholding and generating the mask
# del_empty: delete vertices without geometry after masking
def graph_geom_distance_filter(graph, dist_filt, threshold, operator, del_empty=True):

    # Apply to vertices
    for v in graph.get_vertices_list():
        geom = v.get_geometry()
        geom_distance_filter(geom, dist_filt, threshold, operator)
        if del_empty and (np.sum(geom.get_numpy_mask()) <= 0):
            graph.remove_vertex(v.get_id())

    # Apply to edges
    for e in graph.get_edges_list():
        if e.has_geometry():
            geom = e.get_geometry()
            geom_distance_filter(geom, dist_filt, threshold, operator)
            if del_empty and (np.sum(geom.get_numpy_mask()) <= 0):
                graph.remove_edge(e)

# For applying a distance filter to a geometries
# geom: Geometry
# dist_filt: vtk_ext.vtkClosestPointAlgorithm filter for evaluating the distances
# threshold: threshold applied to the scalar filed
# operator: function applied for thresholding and generating the mask
def geom_distance_filter(geom, dist_filt, threshold, operator):
    off = geom.get_offset()
    mask = geom.get_array_mask()
    for i in range(mask.shape[0]):
        coord = mask[i, :]
        dist, dotp = dist_filt.evaluate(coord[0], coord[1], coord[2])
        dist *= np.sign(dotp)
        if not operator(dist, threshold):
            geom.delete_voxel(coord[0]-off[0], coord[1]-off[1], coord[2]-off[2])

# Build and ArcGraph from the a SkelGraph. If manifolds and density are provide then the geometry
# is added
def build_arcgraph(skel_graph, manifolds=None, density=None):

    a_skel = vtk.vtkPolyData()

    # Copying the vertices for having a list of the processed ones
    vertices = skel_graph.get_vertices_list(core=True)

    # Loop for processing the vertices
    true_vertices = list()
    for v in vertices:
        n_neigh = v.get_num_neighbours()
        if n_neigh != 2:
            true_vertices.append(skel_graph.get_vertex(v.get_id()))

    # Loop for adding the arcs
    arcs_graph = list()
    for v in true_vertices:
        arcs_v = list()
        # Loop for arcs in a vertex
        for neigh in v.get_neighbours():
            # Track the arc
            prev = v
            current = neigh
            arc_list = list()
            while True:
                n_neighs = current.get_num_neighbours()
                if n_neighs == 2:
                    arc_list.append(current)
                    hold_neighs = current.get_neighbours()
                    if hold_neighs[0].get_id() == prev.get_id():
                        prev = current
                        current = hold_neighs[1]
                    else:
                        prev = current
                        current = hold_neighs[0]
                else:
                    break
            # Add head and tail vertices to the arc
            arc_list.insert(0, v)
            arc_list.append(current)
            arcs_v.append(graph.Arc(arc_list))
        arcs_graph.append(arcs_v)

    # Building the new skeleton for the arc graph (only point attributes are added)
    skel = skel_graph.get_skel()
    a_skel.SetPoints(skel.GetPoints())
    # Copying point properties
    v_info, e_info = skel_graph.get_prop_info()
    for i in range(v_info.get_nprops()):
        array = disperse_io.TypesConverter.gt_to_vtk(v_info.get_type(i))
        a_name = v_info.get_key(i)
        a_comp = v_info.get_ncomp(i)
        array.SetName(a_name)
        array.SetNumberOfComponents(a_comp)
        for j in range(skel.GetNumberOfPoints()):
            v = skel_graph.get_vertex(j)
            if v is not None:
                t = v.get_property(a_name)
                if isinstance(t, tuple):
                    for k in range(a_comp):
                        array.InsertComponent(j, k, t[k])
                else:
                    array.InsertComponent(j, 0, t)
            else:
                for k in range(a_comp):
                    array.InsertComponent(j, k, -1)
        a_skel.GetPointData().AddArray(array)
    # Adding the vertices
    verts = vtk.vtkCellArray()
    for v in true_vertices:
        verts.InsertNextCell(1)
        verts.InsertCellPoint(v.get_id())
    # Adding the arcs
    lines = vtk.vtkCellArray()
    for e in arcs_graph:
        for a in e:
            nverts = a.get_nvertices()
            lines.InsertNextCell(nverts)
            for i in range(nverts):
                lines.InsertCellPoint(a.get_vertex(i).get_id())
    a_skel.SetVerts(verts)
    a_skel.SetLines(lines)

    # Building the ArcGraph
    arc_graph = graph.ArcGraph(a_skel)
    arc_graph.update()
    if (manifolds is not None) and (density is not None):
        arc_graph.update_geometry(manifolds, density)

    return arc_graph

# Build a graph_tool graph from an ArcGraph
# arc_graph: input ArcGraph
# weight: edge property string key for being added with label 'weight' into the graph. By default
# vertex id property is added.
# Return: the graph tool object and two numpy arrays with the list of vertices and edges labels
def build_gt_agraph(arc_graph, weight=None):

    graph = gt.Graph(directed=False)
    verts = arc_graph.get_vertices_list()
    v_lut = np.empty(shape=len(verts), dtype=object)
    vid_lut = np.zeros(shape=len(verts), dtype=np.int)
    arcs = arc_graph.get_arcs_list()
    a_lut = np.empty(shape=len(arcs), dtype=object)

    # Adding vertices
    for i, v in enumerate(verts):
        v_id = graph.add_vertex()
        v.add_property(STR_GRAPH_TOOL_ID, i)
        v_lut[i] = v_id
        vid_lut[i] = v.get_id()
    # Adding arcs
    for i, a in enumerate(arcs):
        v_start_id = a.get_start_vertex().get_property(STR_GRAPH_TOOL_ID)
        v_end_id = a.get_end_vertex().get_property(STR_GRAPH_TOOL_ID)
        a_id = graph.add_edge(v_lut[v_start_id], v_lut[v_end_id])
        a_lut[i] = a_id

    # Adding weight as property map
    if weight is not None:
        a_prop = arc_graph.get_arc_prop_info()
        e_prop_id = a_prop.is_already(weight)
        if e_prop_id is None:
            error_msg = '%s is not an edge property.' % weight
            raise pexceptions.PySegInputError(expr='build_gt_agraph', msg=error_msg)
        prop_w = graph.new_edge_property(a_prop.get_type(e_prop_id))
        prop_a = graph.new_edge_property('int')
        for i in range(a_lut.shape[0]):
            prop_w[a_lut[i]] = arcs[i].get_property(weight)
            prop_a[a_lut[i]] = i
        graph.edge_properties[STR_ARCS_ID] = prop_a
        graph.edge_properties[STR_GRAPH_TOOL_WEIGHT] = prop_w

    prop_id = graph.new_vertex_property('int')
    prop_id.get_array()[:] = vid_lut
    graph.vertex_properties[STR_VERTEX_ID] = prop_id

    return graph, v_lut, a_lut

# Returns the shortest path accumulation among two node set in a GraphMCF
# graph_mcf: input graph
# set_start: list with the starting nodes
# set_target: list with the destination nodes
# key_prop: key store
# weight_key: key property for weighting the edges during the search (default None), only
#             1 component property should be used
def short_path_accum(graph_mcf, set_start, set_target, key_prop, weight_key=None):

    # Get GraphGT and anchors LUT, start points must not be includes so the actual starting
    # points will be their neighbours and initial starting point and their edges are initialized
    # to 1
    # graph = gp.GraphGT(graph_mcf).get_gt()
    new_key_id = graph_mcf.add_prop(key_prop, 'int', 1, 0)
    lut = np.zeros(shape=graph_mcf.get_nid(), dtype=np.int8)
    for start in set_start:
        s_id = start.get_id()
        lut[s_id] = 3
        graph_mcf.set_prop_entry_fast(new_key_id, (1,), s_id, 1)
        n_verts, n_edges = graph_mcf.get_vertex_neighbours(s_id)
        for i, n in enumerate(n_verts):
            n_id = n.get_id()
            lut[n_id] = 1
            graph_mcf.set_prop_entry_fast(new_key_id, (1,), n_id, 1)
            graph_mcf.set_prop_entry_fast(new_key_id, (1,), n_edges[i].get_id(), 1)
    for target in set_target:
        lut[target.get_id()] = 2
    graph = gt.Graph(directed=False)
    # Vertices
    vertices = graph_mcf.get_vertices_list()
    vertices_gt = np.empty(shape=graph_mcf.get_nid(), dtype=object)
    vertices_hold = list()
    for v in vertices:
        v_id = v.get_id()
        if lut[v_id] != 3:
            vertices_gt[v_id] = graph.add_vertex()
            vertices_hold.append(v)
    # Edges
    edges = graph_mcf.get_edges_list()
    edges_gt = np.empty(shape=graph_mcf.get_nid(), dtype=object)
    edges_hold = list()
    for e in edges:
        v_s = vertices_gt[e.get_source_id()]
        v_t = vertices_gt[e.get_target_id()]
        if (v_s is not None) and (v_t is not None):
            edges_gt[e.get_id()] = graph.add_edge(v_s, v_t)
            edges_hold.append(e)
    # Property
    if weight_key is None:
        w_prop = None
    else:
        weight_key_id = graph_mcf.get_prop_id(weight_key)
        gt_type = graph_mcf.get_prop_type(weight_key_id)
        np_type = disperse_io.TypesConverter().gt_to_numpy(gt_type)
        w_prop = graph.new_edge_property(gt_type)
        for i, e in enumerate(edges_hold):
            e_id = e.get_id()
            w = graph_mcf.get_prop_entry_fast(weight_key_id, e_id, 1, np_type)
            w_prop[edges_gt[e_id]] = w[0]
    e_prop = graph.new_edge_property('int')
    for e in edges_hold:
        e_id = e.get_id()
        e_prop[edges_gt[e_id]] = e_id

    # Measure all pairs distances
    dist_map = gt.shortest_distance(graph, weights=w_prop)

    # Computing the new GraphMCF props
    for i, s in enumerate(graph.vertices()):
        s_id = vertices_hold[i].get_id()
        # Is in start set?
        if lut[s_id] == 1:
            # Get indices to shortest anchors
            id_sort = np.argsort(dist_map[s].get_array())
            for j in range(len(id_sort)):
                t = graph.vertex(id_sort[j])
                t_id = vertices_hold[int(t)].get_id()
                # t_id = graph.vertex_properties[DPSTR_CELL][t]
                # Is in target set?
                if lut[t_id] == 2:
                    # Compute geodesic path
                    _, e_list = gt.shortest_path(graph, source=s, target=t, weights=w_prop)
                    # Accumulate the number paths
                    for e in e_list:
                        # e_id = graph.edge_properties[DPSTR_CELL][e]
                        # v_s_id = graph.vertex_properties[DPSTR_CELL][e.source()]
                        # v_t_id = graph.vertex_properties[DPSTR_CELL][e.target()]
                        e_id = e_prop[e]
                        v_s_id = vertices_hold[int(e.source())].get_id()
                        v_t_id = vertices_hold[int(e.target())].get_id()
                        a_e = graph_mcf.get_prop_entry_fast(new_key_id, e_id, 1, np.int)
                        s_e = graph_mcf.get_prop_entry_fast(new_key_id, v_s_id, 1, np.int)
                        t_e = graph_mcf.get_prop_entry_fast(new_key_id, v_t_id, 1, np.int)
                        hold_a = a_e[0] + 1
                        hold_s = s_e[0] + 1
                        hold_t = t_e[0] + 1
                        graph_mcf.set_prop_entry_fast(new_key_id, (hold_a,), e_id, 1)
                        graph_mcf.set_prop_entry_fast(new_key_id, (hold_s,), v_s_id, 1)
                        graph_mcf.set_prop_entry_fast(new_key_id, (hold_t,), v_t_id, 1)
                    break
