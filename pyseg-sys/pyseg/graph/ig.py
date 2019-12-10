"""
Wrapping classes for using igraph package

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 24.02.15
"""

__author__ = 'martinez'
try:
    from globals import *
except:
    from pyseg.globals import *
try:
    import disperse_io
except:
    import pyseg.disperse_io
import pexceptions
from morse import GraphMCF
import igraph as ig

##################################################################################################
# Class for creating a GT graph from GraphMCF
#
#
class GraphIG(object):

#### Constructor Area

    # graph_mcf: GraphMCF instance, if it is a string the graph is loaded from disk, and if
    # None the graph is not initialized (it is needed to call either set_mcf_graph() or load)
    def __init__(self, graph_id):
        if isinstance(graph_id, GraphMCF):
            self.set_mcf_graph(graph_id)
        elif isinstance(graph_id, str):
            self.load(graph_id)
        else:
            self.__graph = None
        self.__cell_v_id = None
        self.__cell_e_id = None
        self.__dendro = None
        self.__clusters = None

    #### Get/Set

    def set_mcf_graph(self, graph_mcf):
        self.__graph = graph_mcf.get_ig()

    def get_gt(self):
        return self.__graph

    #### External functionality

    # Store in disk the graph (preferred format .gt)
    def save(self, filename):
        self.__graph.save(filename)

    # Load from disk the graph and overwrites the previous one (preferred format .gt)
    def load(self, filename):
        self.__graph = gt.Load(filename)

    # Returns a hierarchical community analysis
    # edge_weight: property key for weighting edges (default None)
    # method: string key with name of the method applied (see igraph doc) (default fastgreedy)
    # Return: the output dendrogram was a stored in an internal variable (see add_dendrogram)
    def hierarchical_community(self, edge_weight=None, method='fastgreedy'):

        if method == 'leading_eigenvector_naive':
            self.__dendro = self.__graph.community_leading_eigenvector_naive()
        elif method == 'leading_eigenvector':
            self.__dendro = self.__graph.community_leading_eigenvector()
        elif method == 'edge_betweenness':
            self.__dendro = self.__graph.community_edge_betweenness(weights=edge_weight)
        elif method == 'walktrap':
            self.__dendro = self.__graph.community_walktrap(weights=edge_weight)
        else:
            self.__dendro = self.__graph.community_fastgreedy(weights=edge_weight)

    # Plot the dendrogram
    def plot_dendrogram(self):
        if self.__dendro is None:
            self.hierarchical_community()
        ig.plot(self.__dendro)

    # Clusters a dendrogram
    # nclusters: desired minimu number of clusters (default None, optimal value is computed)
    # Result: the will be stored in internal variable (see add_clustering)
    def dendrogram_cluster(self, nclusters=None):

        if self.__dendro is None:
            self.hierarchical_community()

        self.__clusters = self.__dendro.as_clustering(nclusters)