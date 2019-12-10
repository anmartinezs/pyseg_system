"""
Classes for implementing the visitors used by graph_tool package

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 27.11.14
"""

__author__ = 'martinez'
__version__ = "$Revision: 002 $"

__author__ = 'martinez'

import graph_tool.all as gt

#####################################################################################################
#   Class for implementing the functionality of Base class DFSVisitor of graph_tool package, with
#   the intention of finding fully connected subgraphs within a graph
#
#
class SubGraphVisitor(gt.DFSVisitor):

    # sgraphs_id: (output) graph_tool vertex property where vertices are labeled to different
    #             subgraphs after calling gt.dfs_search()
    def __init__(self, sgraphs_id):
        # Intial condition is all ids as 0
        self.__sgraphs_id = sgraphs_id
        self.__sgraphs_id.get_array()[:] = 0
        self.__sgraphs_id = sgraphs_id
        self.__current_id = 1

    def get_num_sgraphs(self):
        return self.__current_id - 1

    # If a vertex was already in a subgraph nothing is done, otherwise it is labeled with current
    # sgraphs id
    def discover_vertex(self, u):
        if self.__sgraphs_id[u] == 0:
            self.__sgraphs_id[u] = self.__current_id

    def examine_edge(self, u):
        pass

    def tree_edge(self, e):
        pass

    # Increments sgraphs id (this method should be called after a successfully gt.dfs_search())
    def update_sgraphs_id(self):
        self.__current_id += 1

#####################################################################################################
#   Visitor child class for accumulating shortest distance and closer starting point within
#   different Dijkstra searches
#
#
class GrowGraphVisitor(gt.DijkstraVisitor):

    # ids: vertex property map for closest starting vertices, starting vertices will be labeled as -1
    # weights: edge property map for edges weights
    # dists: vertex property map for holding the accumulated shortest distance, if must be
    #        initialized to Inf
    # max_dist: maximum distance for stopping the search
    def __init__(self, ids, weights, dists, max_dist):
        self.__start_v = None
        self.__ids = ids
        self.__weights = weights
        self.__dists = dists
        self.__max_dist = max_dist

    def initialize_vertex(self, u):
        if u == self.__start_v:
            self.__dists[u] = 0
            self.__ids[u] = -1

    # Update global distance
    def edge_relaxed(self, e):
        (u, v) = tuple(e)
        dist = self.__weights[e] + self.__dists[u]
        if dist < self.__dists[v]:
            self.__dists[v] = dist
            if self.__ids[v] != -1:
                self.__ids[v] = self.__start_v

    # After one vertex has completely been processed if max_dist reaches the current search is
    # stopped
    def finish_vertex(self, u):
        if self.__dists[u] > self.__max_dist:
            raise gt.StopSearch

    # For updating the starting vertex within different searches
    def set_start_v(self, v):
        self.__start_v = v