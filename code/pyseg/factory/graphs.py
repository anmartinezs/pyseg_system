"""
Classes for extracting filaments and processing from Graph

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 14.10.14
"""

__author__ = 'martinez'
__version__ = "$Revision: 002 $"

try:
    from graph import *
except:
    from pyseg.graph import *
try:
    from globals import *
except:
    from pyseg.globals import *
import graph_tool.all as gt


#####################################################################################################
#   Class for creating an ArcGraph network from an unprocessed ArgGraph. All members in the output
#   network ara fully connected ArcGraphs
#
#
class ArcGraphFactory(object):

    #### Constructor Area
    # arc: ArcGraph
    def __init__(self, arc_graph):
        self.__arc_graph = arc_graph
        self.__graph = gt.Graph(directed=False)
        self.__vertices = self.__arc_graph.get_vertices_list()
        self.__vertices_gt = np.empty(shape=len(self.__vertices), dtype=object)
        self.__arcs = self.__arc_graph.get_arcs_list()
        self.__lut = np.empty(shape=self.__arc_graph.get_skel().GetNumberOfPoints(),
                              dtype=object)
        self.__edges_list = self.__arc_graph.get_edges_list()
        # Build up the gt graph
        for i, v in enumerate(self.__vertices):
            self.__vertices_gt[i] = self.__graph.add_vertex()
            self.__lut[v.get_id()] = self.__vertices_gt[i]
        for e in self.__edges_list:
            v_source = e.get_start_vertex()
            v_target = e.get_end_vertex()
            self.__graph.add_edge(self.__lut[v_source.get_id()], self.__lut[v_target.get_id()])

    #### Functionality area

    # From an input ArcGraph generates an NetArcGraphs where every graph within it is fully connected.
    # For discoverting all nodes in a graph Depth-fist search algorithm is used.
    def gen_netarcgraphs(self):

        # Subgraphs visitor initialization
        sgraph_id = self.__graph.new_vertex_property("int")
        visitor = SubGraphVisitor(sgraph_id)

        # Find subgraphs
        for v in self.__vertices_gt:
            if sgraph_id[v] == 0:
                gt.bfs_search(self.__graph, v, visitor)

        # Build up the new ArgGraphs
        arc_graphs = np.empty(shape=len(np.max(sgraph_id.get_array())), dtype=ArcGraph)
        # Creating ArcGraphs
        for i in arc_graphs.shape:
            arc_graphs[i] = ArcGraph(self.__arc_graph.get_skel())
        # Inserting vertices
        for v in self.__vertices:
            point_id = v.get_id()
            g_id = sgraph_id[self.__lut[point_id]]
            arc_graphs[g_id-1].insert_vertex(point_id, v.get_properties_list(),
                                                 v.get_properties_names_list())
        # Inserting arcs
        for e in self.__edges_list:
            v_source = e.get_start_vertex()
            v_target = e.get_end_vertex()
            sg_id = sgraph_id[self.__lut[v_source.get_id()]]
            if sg_id == sgraph_id[self.__lut[v_target.get_id()]]:
                super(ArcGraph, arc_graphs[sg_id-1]).insert_edge(e, v_source, v_target,
                                                                 e.get_properites_list(),
                                                                 e.get_prop_names_list())
            else:
                error_msg = 'Vertices of different subgraphs cannot be connected.'
                raise pexceptions.PySegTransitionError(expr='gen_netarcgraphs (ArcGraphFactory)',
                                                       msg=error_msg)

        # Copying PropInfo()
        for a in arc_graphs:
            a.copy_prop_info(self.__arc_graph)

        # Build the network and return
        return NetArcGraphs(arc_graphs)