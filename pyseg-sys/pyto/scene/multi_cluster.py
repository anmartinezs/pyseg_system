"""
Contains class MultiCluster used to make and analyzes clusters of (already 
existing) boundaries and segments (here called connections).

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: multi_cluster.py 1059 2014-10-10 15:30:38Z vladan $
"""

__version__ = "$Revision: 1059 $"


import warnings
import logging
import numpy
import scipy

from pyto.segmentation.cluster import Cluster

class MultiCluster(object):
    """
    An instance of this class can contain up to six different clusterings of
    the same data (see below)

    Clusters can be formed in the following ways:

    1) Clustering based on connectivity

    All boundaries that are linked to each other via connections (connectors) 
    form a boundary cluster. In the same way, all connections
    that are connected to each other vie boundaries form a connections
    cluster. These boundary and connection clusters are dual to each other,
    because one uniquely determines the other.

    Clustering method: connectivity()

    Attributes (pyto.segmentation.Cluster objects):
      - connectivityBoundaries: clusters of boundaries based on linkage by 
      connectors
      - connectivityConnections: clusters of connectors based on linkage by 
      boundaries
      
    The above two clustering objects (connectivityBoundaries, 
    connectivityConnections) may contain the following attributes:
      - ids: list of cluster ids
      - clusters: list corresponding to ids where each element is a set that
      contains ids of boundaries or segments belonging to that cluster
      - nItems: (list corresponding to ids) number of items in each cluster,
      where items are boundaries in clusters of boundaries and connections in 
      clusters of connections.
      - nConnections: (list corresponding to ids) number of connections in each
      connection cluster
      - nLinks: (list corresponding to ids) number of links in each
      connection cluster (links are connections where multiple connections 
      betweeen the same boundaries are not taken into account)
      - euler, nLoops: (list corresponding to ids) euler number and the number 
      of independent loops formed by boundaries and connections of each cluster
      - eulerLinks, nLoopsLinks: the same as euler and nLoops, but based on 
      boundaries and links (that is neglecting multiple connections betweeen
      the same boundaries)      
      - dataIds: list of data (boundary or connector) ids
      - nClusters: number of clusters

    2) Hierarchical (distance based) clustering

    Hierarchical clustering is applied separately to boundaries and connections
    to yeild hierarchy boundary and hierarchy connection clusters, respectively.
    Dual clusters to these can be obtained, however hierarchical connection
    (boundary) clusters do not have to be identical to dual hierarchy 
    connection (boundary) clusters (obtained from hierarchical boundary/
    connection clusters).  

    Clustering methods:
      - hierarchicalBoundaries(): hierarchical clustering of boundaries
      - hierarchicalConections(): hierarchical clustering of connections

    Attributes (pyto.segmentation.Cluster objects):
      - hierarchyBoundaries: hierarchical clusters of boundaries
      - hierarchyConnections: hierarchical clusters of connections
      - dualHierarchyBoundaries: hierarchical clusters of connections dual to
      hierarchyBoundaries
      - dualHierarchyConnections: hierarchical clusters of boundaries dual to 
      hierarchyConnections

    The above two clustering objects (connectivityBoundaries, 
    connectivityConnections) may contain the following attributes:
      - ids: list of cluster ids
      - clusters: list corresponding to ids where each element is a set that
      contains ids of boundaries or segments belonging to that cluster
      - nItems: (list corresponding to ids) number of items in each cluster,
      where items are boundaries in clusters of boundaries and connections in 
      clusters of connections.
      - dataIds: list of data (boundary or connector) ids
      - nClusters: number of clusters

    For furhter info about how clusters are generated and what other attributes
    are set see pyto.segmentation.Cluster class.
    """

    def __init__(self):
        """
        Initializes attributes
        """

        self.connectivityBoundaries = None
        self.connectivityConnections = None
        self.hierarchyBoundaries = None
        self.dualHierarchyBoundaries = None
        self.hierarchyConnections = None
        self.dualHierarchyConnections = None
        self.contacts = None

    def connectivity(self, contacts, topology=True, branching=True):
        """
        Clusters boundaries and connections based on connectivity.
        
        Arguments:
          - contacts: (Contacts) specifies contacts between connections and 
          boundaries
          - topology: if True calculates topological invariants (Euler 
          characteristics and the number of independent loops)
          - branching: flag indicating if number of branches is calculated

        Sets:
          - self.connectivityBoundaries
          - self.connectivityConnections
        """

        # cluster boundaries
        self.connectivityBoundaries = \
            Cluster.clusterBoundaries(contacts=contacts)

        # calculate topology
        if topology:
            self.connectivityBoundaries.calculateTopology(contacts=contacts)

        # calculate branching
        if branching:
            self.connectivityBoundaries.calculateBranches(contacts=contacts)

        # cluster connections
        self.connectivityConnections = Cluster.clusterConnections(
            contacts=contacts, boundClusters=self.connectivityBoundaries)

    def hierarchicalBoundaries(self, linkage, threshold, criterion, depth, 
                   distances=None, boundaries=None, contacts=None, ids=None,
                   reference=None, similarity='rand', single=True):
        """
        Hierarchical clustering of boundaries. If (arg) contacts is given also 
        generates dual connections clustering.

        Hierarchical clustering is performed based on the specified linkage 
        ('single', 'complete', 'average', 'weighted') and distance args.
        
        Flat clusters are made from the hierachical clustering according to the 
        specified criterion (distance', 'maxclust', or 'inconsistent') and 
        threshold args. The meaining of threshold depends on the criterion:
          - 'distance': maximum distance within each cluster 
          - 'maxclust': max number of clusters
          - 'inconsistent': maximum inconsistency
        In addition, arg depth is needed for criterion 'inconsistent'.

        If arg reference is 'connectivity', similarity coefficient is 
        calculated in respect to the current connectivity clusters. If arg 
        reference is an instance of Clusters it is used as a reference 
        (reference has to contain flat clusters). In both cases the similarity
        index is calculated according to the method specified by arg similarity.
        Otherwise, if arg reference is None no similarity coefficient is 
        calculated.

        If arg threshold is a list, flat clusters and the respective similarity
        coefficients are calculated from the hierarchical clustering for all 
        threshold values. The flat cluster with the highest similarity is
        retained.

        Distances between boundaries are taken from arg distances if specified,
        or calculated form boundaries otherwise.

        Arguments:
          - linkage: linkage (clustering) method:  'single', 'complete', 
          'average', 'weighted', 'centroid', 'median', 'ward', or whatever else
          is accepted by scipy.cluster.hierarchy.linkage()
          - distances: distances between boundaries in the vector form of length
          n(n-1)/2 (as returned by scipy.spatial.distance.pdist)
          - criterion: criterion for forming flat clusters from cluster 
          hierarchy: 'distance', 'maxclust', or 'inconsistent'
          - threshold: threshold for forming flat clusters, depends on criterion
          - depth: depth for 'inconsistent' criterion
          - boundaries: (Labels) contains boundaries
          - ids: ids of boundaries that are clustered
          - contacts: (Contacts) specifies contacts between connections and 
          boundaries
          - reference: (Clusters) reference in respect to which a similarity
          coefficient is calculated
          - similarity: similarity determination method (see findSimilarity() 
          method)
          - single: flag indicating if one-item clusters are included in the 
          similarity index calculation

        Sets:
          - self.hierarchyBoundaries
          - self.dualHierarchyConnections   

        Returns:
          - threshold (if reference is not None)
          - similarity_index (if reference is not None) 
        """

        # make hierarchy
        self.hierarchyBoundaries = Cluster.hierarchical(
            distances=distances, segments=boundaries, ids=ids, method=linkage)
    
        # set reference
        if reference == 'connectivity':
            reference = self.connectivityBoundaries

        # extract flat clusters
        if isinstance(threshold, (list, numpy.ndarray)):

            # find most similar cluster
            thresh, clust, simil = self.hierarchyBoundaries.findMostSimilar(
                thresholds=threshold, reference=reference, criterion=criterion,
                depth=depth, similarity=similarity, single=single)
            self.hierarchyBoundaries = clust

        else:

            # extract as specified
            self.hierarchyBoundaries.extractFlat(threshold=threshold, 
                                             criterion=criterion, depth=depth)
            if reference is not None:
                simil = self.hierarchyBoundaries.findSimilarity(
                    reference=reference, method=similarity, single=single)
                thresh = threshold

        # generate dual connection clusters
        if contacts is not None:
            self.dualHierarchyConnections = Cluster.dualClusterConnections(
                contacts=contacts, boundClusters=self.hierarchyBoundaries)

        if reference is not None:
            return thresh, simil

    def hierarchicalConnections(self, linkage, threshold, criterion, depth, 
                   distances=None, connections=None, contacts=None, ids=None,
                   reference=None, similarity='rand', single=True):
        """
        Hierarchical clustering of connections. If (arg) contacts is given also 
        generates dual boundaries clustering.

        Hierarchical clustering is performed based on the specified linkage 
        ('single', 'complete', 'average', 'weighted') and distance args.
        
        Flat clusters are made from the hierachical clustering according to the 
        specified criterion (distance', 'maxclust', or 'inconsistent') and 
        threshold args. The meaining of threshold depends on the criterion:
          - 'distance': maximum distance within each cluster 
          - 'maxclust': max number of clusters
          - 'inconsistent': maximum inconsistency
        In addition, arg depth is needed for criterion 'inconsistent'.

        If arg reference is 'connectivity', similarity coefficient is 
        calculated in respect to the current connectivity clusters. If arg 
        reference is an instance of Clusters it is used as a reference 
        (reference has to contain flat clusters). In both cases the similarity
        index is calculated according to the method specified by arg similarity.
        Otherwise, if arg reference is None no similarity coefficient is 
        calculated.

        If arg threshold is a list, flat clusters and the respective similarity
        coefficients are calculated from the hierarchical clustering for all 
        threshold values. The flat cluster with the highest similarity is
        retained.

        Distances between connections are taken from arg distances if specified,
        or calculated form connections otherwise.

        Arguments:
          - linkage: linkage (clustering) method:  'single', 'complete', 
          'average', 'weighted', 'centroid', 'median', 'ward', or whatever else
          is accepted by scipy.cluster.hierarchy.linkage()
          - distances: distances between boundaries in the vector form of length
          n(n-1)/2 (as returned by scipy.spatial.distance.pdist)
          - criterion: criterion for forming flat clusters from cluster 
          hierarchy: 'distance', 'maxclust', or 'inconsistent'
          - threshold: threshold for forming flat clusters, depends on criterion
          - depth: depth for 'inconsistent' criterion
          - connections: (Labels) contains connections
          - ids: ids of connections that are clustered
          - contacts: (Contacts) specifies contacts between connections and 
          boundaries
          - reference: (Clusters) reference in respect to which a similarity
          coefficient is calculated
          - similarity: similarity determination method (see findSimilarity() 
          method)
          - single: flag indicating if one-item clusters are included in the 
          similarity index calculation

        Sets:
        - self.hierarchyConnections
        - self.dualHierarchyBoundaries      

        Returns:
          - threshold (if reference is not None)
          - similarity_index (if reference is not None) 
        """

        # make hierarchy
        self.hierarchyConnections = \
            Cluster.hierarchical(distances=distances, segments=connections, 
                                 ids=ids, method=linkage)
    
        # set reference
        if reference == 'connectivity':
            reference = self.connectivityConnections

        # extract flat clusters
        if isinstance(threshold, (list, numpy.ndarray)):

            # find most similar cluster
            thresh, clust, simil = self.hierarchyConnections.findMostSimilar(
                thresholds=threshold, reference=reference, criterion=criterion,
                depth=depth, similarity=similarity, single=single)
            self.hierarchyConnections = clust

        else:

            # extract as specified
            self.hierarchyConnections.extractFlat(
                threshold=threshold, criterion=criterion, depth=depth)
            if reference is not None:
                simil = self.hierarchyConnections.findSimilarity(
                    reference=reference, method=similarity, single=single)
                thresh = threshold

        # generate dual boundary clusters
        if contacts is not None:
            self.dualHierarchyBoundaries = Cluster.dualClusterBoundaries(
                contacts=contacts, connectClusters=self.hierarchyConnections)

        if reference is not None:
            return thresh, simil


