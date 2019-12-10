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
try:
    import factory
except:
    import pyseg.factory
import graph_tool.all as gt

#####################################################################################################
#   Class for creating filaments between two points
#
#
class FilFactory(object):

    #### Constructor Area
    # arc: ArcGraph
    def __init__(self, arc_graph):
        self.__arc_graph = arc_graph
        self.__graph = None
        self.__v_list = None
        self.__e_list = None
        # Initializes the graph and vertices and edges luts
        self.__arc_graph.compute_arcs_max_density((0, 1))
        self.__graph, self.__v_list, self.__e_list = factory.build_gt_agraph(self.__arc_graph,
                                                                             STR_ARC_MAX_DENSITY)
        # self.__v_lut = self.__arc_graph.get_vertices_lut()

    #### Functionality area



    # From a list of ids of the arc_graph skel returns all filaments which connect them
    # Return a NetFilaments object
    def inter_point_filaments(self, point_ids):

        filaments = list()

        # Loop for looking for the shortest paths
        arcs = self.__arc_graph.get_arcs_list()
        n_vertices = len(point_ids)
        prop_w = self.__graph.edge_properties[STR_GRAPH_TOOL_WEIGHT]
        prop_a = self.__graph.edge_properties[STR_ARCS_ID]
        prop_vid = self.__graph.vertex_properties[STR_VERTEX_ID]
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                v_i = gt.find_vertex(self.__graph, prop_vid, point_ids[i])
                v_j = gt.find_vertex(self.__graph, prop_vid, point_ids[j])
                if (len(v_i) > 1) or (len(v_j) > 1):
                    error_msg = 'Duplicated vertex id in a graph.'
                    raise pexceptions.PySegTransitionError(expr='inter_point_filaments (Filament)',
                                                        msg=error_msg)
                vlist, elist = gt.shortest_path(self.__graph, v_i[0], v_j[0], weights=prop_w)
                # vlist, elist = gt.shortest_path(self.__graph, v_i[0], v_j[0])

                # Filter paths which include already visited anchors
                lock_add = False
                if (len(vlist) > 0) and (len(elist) > 0):
                    lock_add = True
                    hold_point_ids = list()
                    for k in range(1, len(vlist)):
                        point_id = prop_vid[vlist[k]]
                        if point_id in point_ids:
                            if not (point_id in hold_point_ids):
                                hold_point_ids.append(point_id)
                            else:
                                lock_add = False

                # Creating the filament
                if lock_add:
                    vertex_arr = np.empty(shape=len(vlist), dtype=Vertex)
                    arcs_arr = np.empty(shape=len(elist), dtype=Arc)
                    for l, v in enumerate(vlist):
                        vertex_arr[l] = self.__arc_graph.get_vertex(prop_vid[v])
                    for l, a in enumerate(elist):
                        arcs_arr[l] = arcs[prop_a[a]]
                    filaments.append(Filament(list(vertex_arr), list(arcs_arr)))

        return NetFilaments(filaments, self.__arc_graph)

    #### Internal functionality area


