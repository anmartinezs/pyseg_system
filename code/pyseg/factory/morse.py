"""
Utilities for processing MCF graphs

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 27.11.14
"""

__author__ = 'martinez'

import operator
import copy as pcopy
# import pyseg.graph as gp
# import pyseg.vtk_ext
from . import visitor
from pyseg.globals import *
# import pyseg.disperse_io

##################################################################################################
# Static class for processing graphs of MCFs (now vertices, edges and arcs are indexed by cell_id)
# IMPORTANT: all graph passed to a method should be built from the same skeleton
#
class GraphsProcessor(object):

    #### External functionality area

    # A := A - B
    @staticmethod
    def graph_sub(graph_a, graph_b):
        for v in graph_b.get_vertices_list():
            v_id = v.get_id()
            if graph_a.get_vertex(v_id) is None:
                graph_a.remove_vertex(v_id)

    # A := A - B
    @staticmethod
    def field_theshold(graph, field, th=0, oper=operator.gt):
        skel = graph.get_skel()
        for v in graph.get_vertices_list():
            x, y, z = skel.GetPoint(v.get_id())
            x, y, z = (int(round(x)), int(round(y)), int(round(z)))
            val = field[x, y, z]
            if oper(val, th):
                graph.remove_vertex(v)

##################################################################################################
# Class for building a graph from a masking scalar field
#
#
class GraphsScalarMask(object):

    #### Constructor Area

    # base_graph: GraphMCF with the base structure (Toplogy and Geometry)
    # field: input scalar field as a 3D numpy array
    # name: string with the name for the scalar field
    def __init__(self, base_graph, field, name):
        self.__field = field
        self.__base_graph = base_graph
        self.__skel = self.__base_graph.get_skel()
        self.__name = name
        self.__core_graph = None
        self.__ext_graph = None
        # Add the scalar field
        self.__base_graph.add_scalar_field(self.__field, self.__name)

    #### Set/Get functionality

    # Return anchor vertices of the core graph
    def get_anchor_vertices(self):

        if self.__core_graph is None:
            error_msg = "Core graph does net exist, call gen_core_graph first."
            raise pexceptions.PySegInputError(expr='get_anchor_vertices (GraphsScalarMask)',
                                              msg=error_msg)

        anchors = list()
        for v in self.__core_graph.get_vertices_list():
            for a in v.get_arcs():
                if self.__core_graph.get_edge(a.get_sad_id()) is None:
                    anchors.append(v)
                    continue

        return anchors

    def get_core_graph(self):
        return self.__core_graph

    def get_ext_graph(self):
        return self.__ext_graph


    #### External functionality area

    # Build the graph within a threshold in the scalar field
    def gen_core_graph(self, threshold, oper=operator.gt):

        # Starting point: a copy of the base array
        self.__core_graph = pcopy.deepcopy(self.__base_graph)
        self.__ext_graph = pcopy.deepcopy(self.__base_graph)

        # Filter according to the scalar field
        self.__core_graph.threshold_vertices(self.__name, threshold, oper)
        self.__core_graph.threshold_edges(self.__name, threshold, oper)

        # Remove core from extended graph
        GraphsProcessor.graph_sub(self.__ext_graph, self.__core_graph)

    # Build the graph attached to the core graph
    def gen_ext_graph(self, max_dist=float('Inf')):

        # Get anchors
        anchors = self.get_anchor_vertices()

        # Apply distance filter
        # Create reference poly points
        ref_poly = vtk.vtkPolyData()
        poly_points = vtk.vtkPoints()
        poly_verts = vtk.vtkCellArray()
        for i, a in enumerate(anchors):
            poly_points.InsertNextPoint(self.__skel.GetPoint(a.get_id()))
            poly_verts.InsertNextCell(1)
            poly_verts.InsertCellPoint(i)

        ref_poly.SetPoints(poly_points)
        ref_poly.SetVerts(poly_verts)
        dist_filter = vtk_ext.vtkClosestPointAlgorithm()
        dist_filter.SetInputData(ref_poly)
        dist_filter.initialize()
        # Loop for filtering extended graph
        for v in self.__ext_graph.get_vertices_list():
            x, y, z = self.__skel.GetPoint(v.get_id())
            if math.fabs(dist_filter.evaluate(x, y, z)) > max_dist:
                self.__ext_graph.remove_vertex(v)

        # Add anchors to extended graph
        for a in anchors:
            self.__ext_graph.insert_vertex(a)
            for arc in a.get_arcs():
                e = self.__base_graph.get_edge(arc.get_sad_id())
                if e is not None:
                    v_id = e.get_source_id()
                    if v_id == a.get_id():
                        v_id = e.get_target_id()
                    if self.__ext_graph.get_vertex(v_id) is not None:
                        self.__ext_graph.insert_edge(e)

        # Build graph_tool graph
        graph = gt.Graph(directed=False)
        vertices = self.__ext_graph.get_vertices_list()
        vertices_gt = np.empty(shape=self.__ext_graph.get_nid(), dtype=object)
        n_verts = len(vertices)
        ids_arr = (-2) * np.ones(shape=n_verts, dtype=int)
        dists_arr = float('inf') * np.ones(shape=n_verts, dtype=float)
        for v in vertices:
            vertices_gt[v.get_id()] = graph.add_vertex()
        edges = self.__ext_graph.get_edges_list()
        weigths_arr = float('inf') * np.ones(shape=len(edges), dtype=float)
        for i, e in enumerate(edges):
            graph.add_edge(vertices_gt[e.get_source_id()], vertices_gt[e.get_target_id()])
            weigths_arr[i] = self.__ext_graph.get_edge_length(e)
        ids = graph.new_vertex_property('int')
        ids.get_array()[:] = ids_arr
        dists = graph.new_vertex_property('float')
        dists.get_array()[:] = dists_arr
        weigths = graph.new_edge_property('float')
        weigths.get_array()[:] = weigths_arr

        # Visitors creation
        visit = visitor.GrowGraphVisitor(ids, weigths, dists, max_dist)

        # Loop for the searches
        for a in anchors:
            v_s = vertices_gt[a.get_id()]
            visit.set_start_v(v_s)
            gt.dijkstra_search(graph, v_s, weigths, visit)

        # Filter extended graph according the result of the searches
        self.__ext_graph.add_prop(key=STR_EXT_ID,
                                  type=disperse_io.TypesConverter.numpy_to_gt(int),
                                  ncomp=1, def_val=0)
        self.__ext_graph.add_prop(key=STR_EXT_DIST,
                                  type=disperse_io.TypesConverter.numpy_to_gt(float),
                                  ncomp=1, def_val=0)
        key_id_id = self.__ext_graph.get_prop_id(STR_EXT_ID)
        key_id_dist = self.__ext_graph.get_prop_id(STR_EXT_DIST)
        for v in vertices:
            v_id = v.get_id()
            v_l = vertices_gt[v_id]
            dist = dists[v_l]
            if (dists[v_l] == 0) or (dists[v_l] > max_dist):
                self.__ext_graph.remove_vertex(v)
            else:
                self.__ext_graph.set_prop_entry_fast(key_id_id, (ids[v_l],), v_id, 1)
                self.__ext_graph.set_prop_entry_fast(key_id_dist, (dist,), v_id, 1)
                # self.__ext_graph.set_prop_entry(STR_EXT_ID, ids[v_l], v_id)
                # self.__ext_graph.set_prop_entry(STR_EXT_DIST, dist, v_id)


# ##################################################################################################
# # Class for building a graph for the structures attached to a surface
# # Here 'core' means surface and 'extended' the structures attached to it
# #
# #
# class GraphsSurfMask(object):
#
#     #### Constructor Area
#
#     # base_graph: GraphMCF with the base structure (Topology and Geometry)
#     # surf: vtk poly data with the surface
#     # sign_field: oriented distance scalar field in numpy format
#     # dist_field: unoriented distance scalar field, more reliable for distance measures
#     # name: side of the surface, either '+' (default) or '-'
#     # thicknes: surface 'thickness' in nm for generating the core graph, greater than zero
#     def __init__(self, base_graph, surf, sign_field, dist_field, polarity=STR_SIGN_P, thickness=1):
#         self.__surf = surf
#         self.__base_graph = base_graph
#         # self.__base_graph = factory.unpickle_obj(self.__base_graph_name) # base_graph
#         self.__skel = self.__base_graph.get_skel()
#         self.__polarity = polarity
#         self.__core_graph = None
#         self.__ext_graph = None
#         self.__thickness = thickness
#         self.__sign_field = sign_field
#         self.__dist_field = dist_field
#
#     #### Get/Set methods area
#
#     # Return anchor vertices of the core graph
#     def get_anchor_vertices(self):
#
#         if self.__core_graph is None:
#             error_msg = "Core graph does net exist, call gen_core_graph first."
#             raise pexceptions.PySegInputError(expr='get_anchor_vertices (GraphsSurfMask)',
#                                               msg=error_msg)
#         lut = np.zeros(shape=self.__core_graph.get_nid(), dtype=bool)
#         anchors = list()
#         for v in self.__core_graph.get_vertices_list():
#             v_id = v.get_id()
#             if lut[v_id]:
#                 continue
#             for a in v.get_arcs():
#                 if self.__core_graph.get_edge(a.get_sad_id()) is None:
#                     lut[v_id] = True
#                     anchors.append(v)
#                     break
#
#         return anchors
#
#     # Return end vertices of the core graph, they are the nodes of the external graph connected
#     # with nodes in base graph which are not included neither the core or external graphs
#     def get_end_vertices(self, anchors=None):
#
#         if self.__core_graph is None:
#             error_msg = "Core graph does net exist, call gen_core_graph first."
#             raise pexceptions.PySegInputError(expr='get_end_vertices (GraphsSurfMask)',
#                                               msg=error_msg)
#         if self.__ext_graph is None:
#             error_msg = "Core graph does net exist, call gen_ext_graph first."
#             raise pexceptions.PySegInputError(expr='get_end_vertices (GraphsSurfMask)',
#                                               msg=error_msg)
#
#         # LUT initialization
#         lut = np.ones(shape=self.__base_graph.get_nid(), dtype=bool)
#         for v in self.__core_graph.get_vertices_list():
#             lut[v.get_id()] = False
#         vertices = self.__ext_graph.get_vertices_list()
#         for v in vertices:
#             lut[v.get_id()] = False
#         lut_anchors = np.ones(shape=self.__base_graph.get_nid(), dtype=bool)
#         if anchors is None:
#             anchors = self.get_anchor_vertices()
#         for a in anchors:
#             lut_anchors[a.get_id()] = False
#
#         # Getting the end nodes
#         end_nodes = list()
#         for v in vertices:
#             v_id = v.get_id()
#             if lut_anchors[v_id]:
#                 neighs, _ = self.__base_graph.get_vertex_neighbours(v_id)
#                 for n in neighs:
#                     n_id = n.get_id()
#                     if lut[n.get_id()] and lut_anchors[n_id]:
#                         end_nodes.append(v)
#
#         return end_nodes
#
#     def get_core_graph(self):
#         return self.__core_graph
#
#     def get_ext_graph(self):
#         return self.__ext_graph
#
#     #### External functionality area
#
#     # Build the graph within a threshold in the scalar field
#     def gen_core_graph(self):
#
#         # Starting point: a copy of the base array
#         self.__core_graph = pcopy.deepcopy(self.__base_graph)
#         self.__ext_graph = pcopy.deepcopy(self.__base_graph)
#
#         # Filter according the thickness
#         thick = self.__thickness / self.__base_graph.get_resolution()
#         # Loop for filtering core graph
#         for v in self.__core_graph.get_vertices_list():
#             x, y, z = self.__skel.GetPoint(v.get_id())
#             x, y, z = (int(round(x)), int(round(y)), int(round(z)))
#             dist = self.__dist_field[x, y, z]
#             if dist > thick:
#                 self.__core_graph.remove_vertex(v)
#         GraphsProcessor.graph_sub(self.__ext_graph, self.__core_graph)
#
#     # Build the graph attached to the core graph
#     # max_dist: maximum geodesic distance fo extending the graph
#     # keep_anchors: if True (default) anchors or core graph are kept in the extended
#     def gen_ext_graph(self, max_dist=float('Inf'), keep_anchors=True):
#
#         # Get anchors
#         anchors = self.get_anchor_vertices()
#
#         # Loop for filtering extended graph
#         if self.__polarity == STR_SIGN_P:
#             for v in self.__ext_graph.get_vertices_list():
#                 x, y, z = self.__skel.GetPoint(v.get_id())
#                 (x, y, z) = int(round(x)), int(round(y)), int(round(z))
#                 if (np.sign(self.__sign_field[x, y, z]) <= 0) or \
#                         (self.__dist_field[x, y, z] > max_dist):
#                     self.__ext_graph.remove_vertex(v)
#         else:
#             for v in self.__ext_graph.get_vertices_list():
#                 x, y, z = self.__skel.GetPoint(v.get_id())
#                 (x, y, z) = int(round(x)), int(round(y)), int(round(z))
#                 if (np.sign(self.__sign_field[x, y, z]) >= 0) or \
#                         (self.__dist_field[x, y, z] > max_dist):
#                     self.__ext_graph.remove_vertex(v)
#
#         # Add anchors to extended graph
#         for a in anchors:
#             to_add = True
#             for arc in a.get_arcs():
#                 e = self.__base_graph.get_edge(arc.get_sad_id())
#                 if e is not None:
#                     if e.get_source_id() != e.get_target_id():
#                         v_id = e.get_source_id()
#                         if v_id == a.get_id():
#                             v_id = e.get_target_id()
#                         if (self.__core_graph.get_vertex(v_id) is None) and \
#                                 (self.__ext_graph.get_vertex(v_id) is not None):
#                             if to_add:
#                                 self.__ext_graph.insert_vertex(a)
#                                 to_add = False
#                             self.__ext_graph.insert_edge(e)
#                         else:
#                             self.__ext_graph.remove_edge(e)
#
#         # Generates anchors domains
#         self.__gen_anchor_dom(anchors, self.__ext_graph, STR_FIELD_VALUE)
#
#         # Thresholding vertices according to the geodesic distance threshold
#         ext_dist_key_id = self.__ext_graph.get_prop_id(STR_EXT_DIST)
#         for v in self.__ext_graph.get_vertices_list():
#             dist = self.__ext_graph.get_prop_entry_fast(ext_dist_key_id, v.get_id(),
#                                                         1, float)[0]
#             if dist > max_dist:
#                 self.__ext_graph.remove_vertex(v)
#
#         # Delete anchors if demanded
#         if not keep_anchors:
#             for a in anchors:
#                 self.__ext_graph.remove_vertex(a)
#
#
#
#     ##### Internal functionality
#
#     # Generates anchor domains by labeling every vertex with its shortest anchors.
#     # It also return the geodesic distance to the shortest distance.
#     # anchors: list of anchors
#     # graph_mcf: GraphMCF
#     # weight_key: property key for weighting the edges
#     # Returns: shortest anchors ids and geodesic distances to them as two new vertex properties
#     #          in the graph
#     def __gen_anchor_dom(self, anchors, graph_mcf, weight_key):
#
#         # Get GraphGT and anchors LUT
#         graph = gp.GraphGT(graph_mcf).get_gt()
#         lut = np.zeros(shape=graph_mcf.get_nid(), dtype=bool)
#         rand_table = np.random.randint(0, len(anchors), graph_mcf.get_nid())
#         for a in anchors:
#             lut[a.get_id()] = True
#
#         # Measure all pairs distances
#         dist_map = gt.shortest_distance(graph, weights=graph.edge_properties[weight_key])
#
#         # Computing the new GraphMCF props
#         ext_id_key_id = graph_mcf.add_prop(STR_EXT_ID, 'int', 1, 0)
#         ext_dist_key_id = graph_mcf.add_prop(STR_EXT_DIST, 'float', 1, 0)
#         ext_rand_key_id = graph_mcf.add_prop(STR_RAND_ID, 'int', 1, 0)
#         self.__ext_graph.compute_edges_length()
#         for v in graph.vertices():
#             # Get indices to shortest anchors
#             id_sort = np.argsort(dist_map[v])
#             for i in range(len(id_sort)):
#                 c = graph.vertex(id_sort[i])
#                 c_id = graph.vertex_properties[DPSTR_CELL][c]
#                 # Is anchor?
#                 if lut[c_id]:
#                     v_id = graph.vertex_properties[DPSTR_CELL][v]
#                     graph_mcf.set_prop_entry_fast(ext_id_key_id, (c_id,), v_id, 1)
#                     # Compute geodesic distance
#                     ext_dist = gt.shortest_distance(graph, source=v, target=c,
#                                                     weights=graph.edge_properties[SGT_EDGE_LENGTH])
#                     graph_mcf.set_prop_entry_fast(ext_dist_key_id, (ext_dist,), v_id, 1)
#                     graph_mcf.set_prop_entry_fast(ext_rand_key_id, (rand_table[c_id],), v_id, 1)
#                     break