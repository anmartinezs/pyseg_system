"""
Classes for building from a set of clusters contained by an GraphMCF object

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 09.05.15
"""

__author__ = 'martinez'

from .core import *
import disperse_io
import graph_tool.all as gt
import globals as gs

try:
    import pickle as pickle
except:
    import pickle

##################################################################################################
# Class for representing a graph of cluster, the underlying structure of the graph is
# recovered by a graph_tool.graph
#
class GraphCluster(object):

    #### Constructor Area

    # graph_mcf: GraphMCF with clusters and geometry
    def __init__(self, graph_mcf):

        if (graph_mcf.get_property(STR_AFF_CLUST) is None) \
                or (graph_mcf.get_property(STR_AFF_CENTER) is None):
            error_msg = 'The input GraphMCF must have attributes ' + STR_AFF_CENTER + \
                ' and ' + STR_AFF_CLUST
            raise pexceptions.PySegInputError(expr='(GraphCluster) __init__',
                                              msg=error_msg)

        self.__graph_mcf = graph_mcf
        self.__vertices = None
        self.__edges = None
        self.__labels = None
        self.__arcs = None
        self.__geometries = None
        self.__lbl_lut = None
        self.__graph_gt = None
        self.__vw_prop = None
        self.__vw_prop_int = None
        self.__vw_prop_max = None
        self.__ew_prop = None
        self.__ew_prop_int = None
        self.__ew_prop_max = None
        self.__ew_prop_path = None
        self.__ew_prop_dst = None

    #### Get/Set


    #### External functionality

    # Build the graph from the input GraphMCF, this method must be called before any other
    # vw_prop: property for weighting the vertices
    # ew_prop: property for weighting the edges
    # vinv: (False) if True vertex weighting property is remapped for being inverted,
    # you should notice that vertex and edge properties condensation is accumulative
    # einv: (False) if True edge weighting property is remapped for being inverted
    def build(self, vw_prop, ew_prop, vinv=False, einv=True):

        # Initialization
        nid = self.__graph_mcf.get_nid()
        self.__vertices = np.zeros(shape=nid, dtype=gt.Vertex)
        self.__labels = (-1) * np.ones(shape=nid, dtype=np.int)
        self.__edges = np.zeros(shape=nid, dtype=gt.Edge)
        self.__arcs = np.zeros(shape=nid, dtype=list)
        self.__geometries = np.zeros(shape=nid, dtype=geometry.GeometryMCF)
        graph_gt = self.__graph_mcf.get_gt()
        self.__graph_gt = gt.Graph(directed=False)
        if vinv:
            hold = graph_gt.vertex_properties[vw_prop].get_array()
            hold = gs.lin_map(hold, lb=hold.max(), ub=hold.max)
            graph_gt.vertex_properties[vw_prop].get_array()[:] = hold
        if einv:
            hold = graph_gt.edge_properties[ew_prop].get_array()
            hold = gs.lin_map(hold, lb=hold.max(), ub=hold.max)
            graph_gt.edge_properties[ew_prop].get_array()[:] = hold
        self.__vw_prop = vw_prop
        self.__vw_prop_int = vw_prop + '_int'
        self.__vw_prop_max = vw_prop + '_max'
        vw_prop_dtype_np = self.__graph_mcf.get_prop_type(vw_prop)
        vw_prop_dtype = disperse_io.TypesConverter().numpy_to_gt(vw_prop_dtype_np)
        self.__ew_prop = ew_prop
        self.__ew_prop_int = ew_prop + '_int'
        self.__ew_prop_max = ew_prop + '_max'
        self.__ew_prop_path = ew_prop + '_path'
        self.__ew_prop_dst = ew_prop + '_dst'
        ew_prop_dtype_np = self.__graph_mcf.get_prop_type(ew_prop)
        ew_prop_dtype = disperse_io.TypesConverter().numpy_to_gt(ew_prop_dtype_np)

        # Adding vertices
        v_id_arr = list()
        vw_prop_arr = list()
        vertices_lut = (-1) * np.ones(shape=graph_gt.num_vertices(), dtype=gt.Vertex)
        for v in graph_gt.vertices():
            if graph_gt.vertex_properties[STR_AFF_CENTER][v] != -1:
                v_id = graph_gt.vertex_properties[DPSTR_CELL][v]
                vertices_lut[v_id] = v
                self.__vertices[v_id] = self.__graph_gt.add_vertex()
                self.__labels[v_id] = graph_gt.vertex_properties[STR_AFF_CLUST][v]
                v_id_arr.append(v_id)
                vw_prop_arr.append(graph_gt.vertex_properties[vw_prop][v])
                vertex = self.__graph_mcf.get_vertex(v_id)
                self.__geometries[v_id] = vertex.get_geometry()
        self.__graph_gt.vertex_properties[DPSTR_CELL] = self.__graph_gt.new_vertex_property('int')
        self.__graph_gt.vertex_properties[DPSTR_CELL].get_array()[:] = np.asarray(v_id_arr,
                                                                                  dtype=np.int)
        self.__graph_gt.vertex_properties[self.__vw_prop] = self.__graph_gt.new_vertex_property(vw_prop_dtype)
        self.__graph_gt.vertex_properties[self.__vw_prop].get_array()[:] = np.asarray(vw_prop_arr,
                                                                                      dtype=np.int)

        # Label's lut
        self.__lbl_lut = np.ones(shape=self.__labels.max(), dtype=np.int)
        for i, v in enumerate(self.__graph_gt.vertices()):
            v_id = graph_gt.vertex_properties[DPSTR_CELL][v]
            self.__lbl_lut[self.__labels[v_id]] = v_id

        # Vertex properties and geometries
        self.__graph_gt.vertex_properties[self.__vw_prop_int] = self.__graph_gt.new_vertex_property(vw_prop_dtype)
        self.__graph_gt.vertex_properties[self.__vw_prop_int].get_array()[:] = np.zeros(shape=self.__graph_gt.num_vertices(), dtype=vw_prop_dtype_np)
        self.__graph_gt.vertex_properties[self.__vw_prop_max] = self.__graph_gt.new_vertex_property(vw_prop_dtype)
        self.__graph_gt.vertex_properties[self.__vw_prop_max].get_array()[:] = (-np.inf) * np.ones(shape=self.__graph_gt.num_vertices(), dtype=vw_prop_dtype_np)
        for v in graph_gt.vertices():
            lbl = graph_gt.vertex_properties[STR_AFF_CLUST][v]
            c = self.__vertices[self.__lbl_lut[lbl]]
            hold = self.__graph_gt.vertex_properties[self.__vw_prop_max][c]
            accum = self.__graph_gt.vertex_properties[self.__vw_prop_int][c]
            value = graph_gt.vertex_properties[vw_prop][v]
            self.__graph_gt.vertex_properties[self.__vw_prop_int] = accum + value
            if value > hold:
                self.__graph_gt.vertex_properties[self.__vw_prop_int] = hold
            v_id = graph_gt.vertex_properties[DPSTR_CELL][v]
            vertex = self.__graph_mcf.get_vertex(v_id)
            self.__geometries[v_id].extend(vertex.get_geometry())


        # Adding edges
        self.__graph_gt.edge_properties[DPSTR_CELL] = self.__graph_gt.new_edge_property('int')
        self.__graph_gt.edge_properties[self.__ew_prop] = self.__graph_gt.new_edge_property(ew_prop_dtype)
        self.__graph_gt.edge_properties[self.__ew_prop_int] = self.__graph_gt.new_edge_property(ew_prop_dtype)
        self.__graph_gt.edge_properties[self.__ew_prop_max] = self.__graph_gt.new_edge_property(ew_prop_dtype)
        for e in graph_gt.edges():
            s_l = graph_gt.vertex_properties[STR_AFF_CLUST][e.source()]
            t_l = graph_gt.vertex_properties[STR_AFF_CLUST][e.target()]
            if s_l != t_l:
                s_id = self.__lbl_lut[self.__labels[s_l]]
                t_id = self.__lbl_lut[self.__labels[t_l]]
                # New edges?
                already = False
                s = self.__graph_gt.vertex(s_id)
                for n in s.all_neighbours():
                    n_id = self.__graph_gt.vertex_properties[DPSTR_CELL][n]
                    if n_id == t_id:
                        already = True
                        break
                # Integrate to parent edge
                if already:
                    e_id = graph_gt.edges_properties[DPSTR_CELL][e]
                    edge = self.__edges[e_id]
                    value = self.__graph_gt.edge_properties[self.__ew_prop][edge]
                    accum = self.__graph_gt.edge_properties[self.__ew_prop_int][edge]
                    hold = self.__graph_gt.edge_properties[self.__ew_prop_max][edge]
                    self.__graph_gt.edge_properties[self.__ew_prop_int][edge] = accum + value
                    if value > hold:
                        self.__graph_gt.edge_properties[self.__ew_prop_max][edge] = value
                # New edge
                else:
                    e_id = graph_gt.edges_properties[DPSTR_CELL][e]
                    t = self.__graph_gt.vertex(t_id)
                    edge = self.__graph_gt.add_edge(s, t)
                    self.__edges[e_id] = edge
                    self.__graph_gt.edge_properties[DPSTR_CELL][edge] = e_id
                    self.__graph_gt.edge_properties[self.__ew_prop][edge] = graph_gt.edge_properties[self.__ew_prop][e]

        # Adding arcs
        self.__graph_gt.edge_properties[self.__ew_prop_path] = self.__graph_gt.new_edge_property(ew_prop_dtype)
        self.__graph_gt.edge_properties[self.__ew_prop_dst] = self.__graph_gt.new_edge_property('float')
        for e in self.__graph_gt.edges():
            s_id = self.__graph_gt.vertex_properties[DPSTR_CELL][e.source()]
            t_id = self.__graph_gt.vertex_properties[DPSTR_CELL][e.target()]
            ss = vertices_lut[s_id]
            tt = vertices_lut[t_id]
            # Look in graph_gt for the shortest path between ss and tt
            v_path, e_path = gt.shortest_path(graph_gt,
                                              source=ss, target=tt,
                                              weights=graph_gt.edge_properties[self.__ew_prop])