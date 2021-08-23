"""
Contains class Cluster for analysis of clustering / classifications.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div

__version__ = "$Revision$"


from copy import copy, deepcopy
import logging

import numpy
try:
    import numpy.ma as ma
except ImportError:
    import numpy.core.ma as ma  # numpy <= 1.0.4
#import numpy.ma as ma
import scipy
import scipy.ndimage as ndimage
import scipy.cluster

import pyto.util.nested as util_nested
from .segment import Segment


class Cluster(object):
    """
    Holds data about a clustering or a classification.

    Contains methods to generate connectivity based clustering for segments.

    It also contains more general clustering methods (hierarchical and 
    k-means) and methods for comparing different clusterings.   

    Internally clusters are represented by a list (self.clusters), where 
    each element of that list defines a cluster. A cluster is represented 
    by a set containing ids of data that belong to that cluster. Clusters 
    are numbered from 1 up, so the i-th element of the clusters 
    list represents a cluster i+1. For example:

      clusters = [{1,3,4,6}, {2,5}]

    means that there are two clusters, number 1 contains elements (items)
    1, 3, 4 and 6 and cluster number 2 contains elements 2 and 5.

    Alternatively, clusters can be shown in the data representation
    (self.clustersData), as an ndarray containing cluster ids and indexed by 
    item (data) ids. Item (data) ids are positive integers listed in 
    self.dataIds. They can have gaps and the element 0  of self.clustersData
    has no meaning. Cluster ids are also positive integers. For example:
    
      clustersData = [-1, 1, 2, 1, 1, 2, 1]

    defines the same clusters as above.
    
    The data representation (self.clustersData) is determineded on the 
    fly from the cluster representation (self.clusters). 

    Hierarchical clusterings are described by code book, in the same format 
    as the one returned by scipy.cluster.hierarchy.linkage().

    Attributes that hold the data are:
      - self.clusters: (ndarray) where element i contains a set of data ids
      belonging to cluster i (i>0, self/clusters[0] is not used)
      - self.clustersData: clusters in the data representation where data ids
      are positive (>0)
      - self.codeBook: code book (array) where data (leaf node) ids are
      positive

    In order to interface with scipy.cluster.hierarchy, where data
    representation is used and the data ids start at 0 and have no gaps,
    the following attributes are used:
      - self.clustersData0: (property derived from self.clustersData and 
      self.dataIds) clusters in the data representation where data ids 
      start at 0 and have no gaps
      - self.codeBook0: (property derived from self.codeBook and self.ids) code 
      book (array) where data ids start at 0 and have no gaps

    For example, for the same cluster defined above:
    
      clustersData0 = [1, 2, 1, 1, 2, 1]

    Cluster ids have to start from 1 and it is strongly recomended that 
    they are consecutive (when clusters are defined using the data
    representation). Methods defined here will work even if there are gaps
    in the cluster ids, but this is still an experimental feature.

    Other attributes:
      - self.similarity: last calculated similarity index 

    ToDo:
      - Move pure clustering stuff (segmentation independent) to 
      pyto.clustering.Cluster and make it an ancestor of this class
    """

    ##################################################################
    #
    # Initialization
    #
    ##################################################################
        
    def __init__(self, clusters=None, form='cluster'):
        """
        Sets clusters if specified and initializes attributes.

        If arg form is 'cluster', clusters are specified in the form used 
        (internaly) by this class, that is by a list of sets, where each set 
        contains item ids that belong to a same cluster. Item ids have to
        be positive integers (>0)

        Alternatively, if form is 'scipy', arg clusters is an array of  
        cluster ids indexed by data (item) ids. Cluster ids have to be 
        positive integers (>0). This form is used in 
        scipy.cluster.hierarchy (returned by fcluster(), for example).

        Arguments:
          - clusters: clusters
          - form: format in which arg clusters are specified
        """

        # set or initialize clusters
        if form == 'cluster':
            self._clusters = clusters
        elif form == 'scipy':
            clusters = numpy.array([0] + list(clusters))
            clust_rep = self.clusterRepresentation(clusters)
            self._clusters = clust_rep
        else:
            raise ValueError(
                "Argument form (" + form + ") is not understood. "
                + "It can be 'cluster' (default) or 'scipy'.")

        # initialize other attributes
        self._dataIds = None
        self.codeBook = None
        self.similarity = None

    @classmethod
    def recast(cls, obj):
        """
        Returns a new instance of this class and sets all attributes of the new 
        object to the attributes of obj.

        Useful for objects of this class loaded from a pickle that were pickled
        before some of the current methods of this class were coded. In such a 
        case the returned object would contain the same attributes as the 
        unpickled object but it would accept recently coded methods. 
        """

        # make a new instance
        new = cls()

        # set attributes
        new._clusters = obj._clusters
        try:
            new._dataIds = obj._dataIds
        except AttributeError:
            pass
        try:
            new.codeBook = obj.codeBook
        except AttributeError:
            pass        
        try:
            new.similarity = obj.similarity
        except AttributeError:
            pass        

        return new

    ##################################################################
    #
    # Basic data structures and related attributes and properties
    #
    ##################################################################
        
    def getClusters(self):
        """
        Returns all clusters as a list of sets where each set contains
        ids of items belonging to that cluster.
        """
        return self._clusters

    def setClusters(self, clusters):
        """
        Sets all clusters.

        Argument:
          - cluster: list where each element is a set containing item ids.
        """
        self._clusters = clusters
    clusters = property(fget=getClusters, fset=setClusters, doc='All clusters')

    def getCluster(self, clusterId):
        """
        Returns a cluster (set) number clusterId.

        Argument:
          - clusterId: (int) cluster id)
        """
        return self.clusters[clusterId-1]
  
    def setCluster(self, cluster, clusterId):
        """
        Set cluster with clusterId to a contain elements from cluster.

        Arguments:
          - cluster: list, ndarray or set containing item ids 
          - clusterId: (int) cluster id)
        """
        
        if isinstance(cluster, set):
            self.clusters[clusterId-1] = cluster
        elif isinstance(cluster, numpy.ndarray):
            self.clusters[clusterId-1] = set(cluster.tolist())
        elif isinstance(cluster, list):
            self.clusters[clusterId-1] = set(cluster)
        else:
            raise TypeError("Argument cluster has to be a set, list or "\
                + "numarray, but not " + type(set) + ".")

    def getNClusters(self):
        """
        Returns number of clusters.
        """
        return len(self.clusters)
    nClusters = property(fget=getNClusters, doc='Number of clusters')

    def getIds(self):
        """
        Return cluster ids.
        """
        return numpy.arange(1, self.nClusters+1)

    ids = property(fget=getIds, doc='Cluster ids')

    def getNItems(self, clusterId=None):
        """
        Returns cluster size (number of items in a cluster) for a given cluster 
        if clusterId is specified. 

        Otherwise, returns an ndarray of cluster sizes for all clusters indexed
        by cluster ids that is in the same order but shifted by one position in 
        respect to self.clusters. Element 0 contains the total number of items. 

        Argument:
          - clusterId: cluster id
        """

        if clusterId is None:
            res = numpy.zeros(self.nClusters+1, dtype='int')
            res[1:] = \
                [self.getNItems(clusterId=id_+1) for id_ 
                 in range(self.nClusters)]
            res[0] = res[1:].sum()
        else:
            res = len(self.clusters[clusterId-1])

        return res

    nItems = property(fget=getNItems, 
                      doc='Size of each cluster, indexed by cluster id.')

    def getDataIds(self, clusters=None):
        """
        Returns all data (item) ids.

        If arg clusters is given the returned ids as extracted from clusters.
        If arg clusters is None, and self.dataIds has not been set (using 
        setDataIds()) the returned ids are extracted from self.clusters.
        Otherwise, the value set by setDataIds() is returned. 

        In each case the returned ids are a sorted ndarray.

        Argument:
          - clusters: clusters, in the same form as self.clusters
        """
        
        # set clusters
        if clusters is None:
            clusters = self.clusters

        if (clusters is not None) or (self._dataIds is None):

            # extract data ids
            ids = util_nested.flatten(clusters)
            ids = numpy.asarray(ids)
            ids.sort()
            return ids

        else:
            return self._dataIds

    def setDataIds(self, ids):
        """
        Sets data ids.
        """

        self._dataIds = numpy.asarray(ids)
        self._dataIds.sort()

    dataIds = property(fget=getDataIds, fset=setDataIds, doc='Data (item) ids')

    def dataRepresentation(self, clusters=None):
        """
        Returns an ndarray indexed by item (data ids) that contains cluster ids.
        That is, the value of the element i of the returned array is
        the cluster id of the cluster that contains item (data) i.

        If there are no ids, returns an ndarray of length 1.

        Arguments:
          - clusters: (list of sets) clusters, if None self.clusters 
          is used. 
        """

        # parse arguments
        if clusters is None:
            clusters = self.clusters

        # find all data ids
        data_ids = []
        for one_cluster in clusters:
            data_ids.extend(list(one_cluster))

        # return if no ids
        if len(data_ids) == 0:
            return numpy.zeros(1, dtype='int') - 1

        # make cluster array indexed by ids 
        res = numpy.zeros(max(data_ids)+1, dtype='int') - 1
        for id_, one_cluster in zip(list(range(1,len(clusters)+1)), clusters):
            res[list(one_cluster)] = id_

        return res

    def clusterRepresentation(self, clustersData):
        """
        Converts clusters in the data representation to the cluster 
        representation.
        
        Argument:
          - clustersData: clusters in the cluster representation (numarray 
          containing cluster ids, indexed by item (data) ids)

        Returns clusters in the cluster representation
        """

        # initialize clusters in the cluster representation
        clusters = [{}] * clustersData.max()

        # assign data ids to clusters
        for (item_id, clust_id) \
                in zip(list(range(1, clustersData.shape[0])), clustersData[1:]):
            if clust_id > 0:
                try:
                    clusters[clust_id-1].add(item_id)
                except AttributeError:
                    clusters[clust_id-1] = set([item_id])

        return clusters

    def getClustersData(self):
        """
        Returns clusters in the data representation
        """
        return self.dataRepresentation()

    def setClustersData(self, clustersData):
        """
        Sets clusters data structure (self.clusters) from clusters in the 
        data representation (argument clustersData)
        """
        self.clusters = self.clusterRepresentation(clustersData=clustersData)

    clustersData = property(fget=getClustersData, fset=setClustersData,
                          doc='Clusters in the data representation.')
    

    ##################################################################
    #
    # Clustering by connectivity
    #
    ##################################################################

    @classmethod
    def clusterBoundaries(cls, contacts, connectedOnly=False):
        """
        Generates clusters of boundaries based on the connectivity 
        information from (arg) contacts.

        A boundary cluster contains all boundaries that are connected to
        each other by connections. 

        Arguments:
          - contacts: (Contact) contains connectivity info
          - connectedOnly: flag indicating if only the boundaries that
          are connected to other boundaries are included

        Returns boundary clusters as an instance of this class (Cluster).
        """

        # initialize
        if connectedOnly:
            bound_clusters = []
        else:
            all_bounds = contacts.getBoundaries()
            bound_clusters = [set([b_id]) for b_id in all_bounds]

        # loop over segments
        for con_id in contacts.getSegments():

            # find all boundaries that contact the current connection
            boundaries = contacts.findBoundaries(segmentIds=[con_id])

            # join all clusters that contain boundaries
            new = set()
            for bound_id in boundaries:
                for clust in bound_clusters:
                    if bound_id in clust:
                        new = new.union(clust)
                        bound_clusters.remove(clust)
                        break

            if len(new) > 0:

                # make sure all current boundaries are in the new cluster and
                # add the new cluster
                new = new.union(set(boundaries))
                bound_clusters.append(new)

            else:

                # current boundaries not in clusters, make a sepearate cluster
                bound_clusters.append(set(boundaries))

        # reverse order, so the 1-element clusters come at the end
        bound_clusters.reverse()

        # instantiate and return
        boundary_clusters = cls(bound_clusters)
        return boundary_clusters

    @classmethod
    def clusterConnections(cls, contacts, boundClusters=None):
        """
        Clusters connections based on the contacts with boundaries.

        Connection clusters are generated from boundary clusters using 
        dualClusterConnections(). If boundary clusters are not given 
        (arg boundClusters) they are calculated first using clusterBoundaries().

        A connection cluster contains all connections that are connected to
        the same boundaries. Furthermore, each boundary cluster and the 
        corresponding connection cluster have matching elements, that is
        i-th connection cluster contans ids of connections that contact
        boundaries of the i-th boundary cluster. Consequently, connection
        clusters corresponding to single-element boundary clusters
        are empty.

        Arguments:
          - contacts: (Contact) contains connectivity info
          - boundClusters: boundary clusters instance

        Returns connection clusters as an instance of this class (Cluster)
        """

        # cluster boundaries if needed 
        if boundClusters is None:
            boundClusters = cls.clusterBoundaries(contacts=contacts, 
                                                  connectedOnly=True)

        # cluster connections using boundary clusters
        conn_clusters = cls.dualClusterConnections(contacts=contacts, 
                                                   boundClusters=boundClusters)

        return conn_clusters

    @classmethod
    def dualClusterBoundaries(cls, contacts, connectClusters, 
                              connectedOnly=False):
        """
        Generates clusters of boundaries that correspond (are dual) to 
        the given connections clusters (arg connectionsClusters). This 
        correspondance (duality) is based on the connectivity 
        information from (arg) contacts.

        A boundary cluster contains all boundaries that contact
        the same connections. Furthermore, a connectClusters and a 
        corresponding boundary cluster have matching elements, that is
        i-th boundary cluster contans ids of boundaries that contact
        connections of the i-th connections cluster. 

        If arg connectedOnly is False, boundaries that contact no connections 
        are added as one-member clusters.

        Arguments:
          - contacts: (Contact) contains connectivity info
          - connectClusters: boundary clusters instance

        Returns segment_clusters: a list of connection clusters, where each  
          cluster is represented as a set of connection ids forming the 
          cluster. Elements of connection_clusters correspond to the 
          (same index) elements of connectClusters.
        """

        # loop over boundary clusters
        conn_clusters = connectClusters.clusters
        bound_clusters = []
        for c_clust in conn_clusters:
            
            # cluster made of all connections that contact current boundaries
            connections = contacts.findBoundaries(segmentIds=list(c_clust), 
                                                  nSegment=1, mode='at_least')
            bound_clusters.append(set(connections))

        # add unconnected boundaries as 1-element clustres
        if connectedOnly:

            # find unconnected boundaries
            all_bounds = contacts.getBoundaries()
            clustered_bounds = util_nested(bound_clusters)
            non_clustered_bounds = numpy.setdiff1d(all_bounds, clustered_bounds)

            # make clusters
            single_el_clusters = [set([b_id]) for b_id in non_clustered_bounds]
            bound_clusters.extend(single_el_clusters)

        # add not connected connections as individual clusters?

        # instantiate and return
        b_clusters = cls(bound_clusters)
        return b_clusters

    @classmethod
    def dualClusterConnections(cls, contacts, boundClusters):
        """
        Generates clusters of connections based on the connectivity 
        information from (arg) contacts and from boundary clusters
        boundClusters.

        A connection cluster contains all connections that are connected to
        the same boundaries. Furthermore, a boundClusters and a 
        corresponding connection cluster have matching elements, that is
        i-th connection cluster contans ids of connections that contact
        boundaries of the i-th boundary cluster. Consequently, connection
        clusters corresponding to single-element boundary clusters
        are empty.

        Boundary clusters are used to avoid doing calculations that were
        already done in clusterBoundaries().

        Arguments:
          - contacts: (Contact) contains connectivity info
          - boundClusters: boundary clusters instance

        Returns segment_clusters: a list of connection clusters, where each  
          cluster is represented as a set of connection ids forming the 
          cluster. Elements of connection_clusters correspond to the 
          (smae index) elements of boundClusters.
        """

        # loop over boundary clusters
        bound_clusters = boundClusters.clusters
        conn_clusters = []
        for b_clust in bound_clusters:
            
            # cluster made of all connections that contact current boundaries
            connections = contacts.findSegments(boundaryIds=list(b_clust), 
                                                nBoundary=1, mode='at_least')
            conn_clusters.append(set(connections))

        # add not connected connections as individual clusters?

        # instantiate and return
        connection_clusters = cls(conn_clusters)
        return connection_clusters

    def calculateTopology(self, contacts):
        """
        Calculates some topological properties for each clusters.

        Argument:
          - contacts: (Contact) instance defining contacts between boundaries
          and connections

        Calculates and sets:
          - self.euler, self.eulerLinks: Euler characteristics (see 
          self.getEuler())
          - self.nLoops, self.nLoopsLinks: number of independent loops (see 
          self.getNLinks())
          - self.nConnections: number of connection (see self.getNConnections())
          - self.nLinks: number of links (see self.getNLinks())
        """
        self.getNLoops(contacts=contacts, mode='conn')
        self.getNLoops(contacts=contacts, mode='links')

    def getNLinks(self, contacts):
        """
        Calculates number of linked boundary pairs in each cluster. 

        Two boundaries connected by one or more connections make one link.

        Argument:
          - contacts: (Contact) instance defining contacts between boundaries
          and connections

        Sets:
          - self.nLinks: same as the returned value

        Returns ndarray containing number of connections. Element i of this
        array gives the number for cluster i, wile element 0 is the number of 
        all clusters together.
        """

        #  calculate for each cluster
        n_link = numpy.zeros(self.nClusters+1, dtype='int')
        for clust, clust_ind in zip(
                self.clusters, list(range(1, self.nClusters+1))):

            linked = contacts.findLinkedBoundaries(list(clust))
            n_link[clust_ind] = int(old_div(sum(len(x) for x in linked), 2))

        # for all clusters together
        n_link[0] = n_link[1:].sum()

        self.nLinks = n_link
        return n_link

    def getNConnections(self, contacts):
        """
        Calculates number of connections in each cluster.

        Argument:
          - contacts: (Contact) instance defining contacts between boundaries
          and connections

        Sets:
          - self.nConnections: same as the returned value

        Returns ndarray containing number of connections. Element i of this
        array gives the number for cluster i, wile element 0 is the number of 
        all clusters together.
        """

        #  calculate for each cluster
        n_conn = numpy.zeros(self.nClusters+1, dtype='int')
        for clust, clust_ind in zip(
                self.clusters, list(range(1, self.nClusters+1))):

            conn_ids = contacts.findSegments(boundaryIds=list(clust), 
                                             nBoundary=1)
            n_conn[clust_ind] = len(conn_ids)

        # for all clusters together
        n_conn[0] = n_conn[1:].sum()

        self.nConnections = n_conn
        return n_conn

    def getEuler(self, contacts=None, mode='conn'):
        """
        Calculates Euler characteristics for each connectivity cluster and
        for all clusters together.

        Boundaries represent verteces and edges represent connections (for 
        mode='conn') or links (mode='links'). Euler characteristics is 
        calculates as:

        Euler = n_boundaries - n_edges

        Two boundaries connected by one or more connections make one link.

        Arguments:
          - contacts: (Contact) instance defining contacts between boundaries
          and connections, needed if self.nConnections is not set
          - mode: 'connections' (same as 'conn') or 'links' ('lin') 

        Sets:
          - self.euler or self.eulerLink (for mode 'conn' or 'links'): same as 
          the returned value

        Returns ndarray containing Euler characteristics. Element i of this
        array gives Euler for cluster i, wile element 0 is the Euler of all
        clusters together.
        """

        # get nConnections 
        try:
            if (mode == 'conn') or (mode == 'connections'):
                n_conn = self.nConnections
            elif (mode == 'lin') or (mode == 'links'):
                n_conn = self.nLinks
            else:
                raise ValueError(
                    "Mode " + str(mode) + " is not known. Allowed "
                    + "modes are 'connections' ('conn') and 'links' ('lin').")
            if n_conn is None:
                raise AttributeError()
        except AttributeError:
            if (mode == 'conn') or (mode == 'connections'):
                n_conn = self.getNConnections(contacts=contacts)
            elif (mode == 'lin') or (mode == 'links'):
                n_conn = self.getNLinks(contacts=contacts)

        # calculate Euler
        euler = self.nItems - n_conn

        # save Euler
        if (mode == 'conn') or (mode == 'connections'):
            self.euler = euler
        elif (mode == 'lin') or (mode == 'links'):
            self.eulerLinks = euler

        return euler

    def getNLoops(self, contacts=None, mode='conn'):
        """
        Returns number of independent loops formed by connections (for 
        mode='conn') or links (mode='links') edges.

        If self.euler is not calculated arg contacts is required in order
        to calculate the Euler characteristics first (using getEuler()).

        Two boundaries connected by one or more connections make one link.

        Arguments:
          - contacts: (Contact) instance defining contacts between boundaries
          and connections, needed if self.euler is not set
          - mode: 'connections' (same as 'conn') or 'links' ('lin') 

        Sets:
          - self.nLoops or self.nLoopsLinks (for mode 'conn' or 'links'): same 
          as the returned value

        Returns ndarray containing number of independent loops. Element i of 
        this array gives the number for cluster i, wile element 0 is the number
        for all clusters together.
        """

        # get Euler if needed
        try:
            if (mode == 'conn') or (mode == 'connections'):
                euler = self.euler
            elif (mode == 'lin') or (mode == 'links'):
                euler = self.eulerLinks
            else:
                raise ValueError(
                    "Mode " + str(mode) + " is not known. Allowed "
                    + "modes are 'connections' ('conn') and 'links' ('lin').")
            if euler is None:
                raise AttributeError()
        except AttributeError:
            euler = self.getEuler(contacts=contacts, mode=mode)

        # calculate num loops
        n_loops = 1 - euler
        n_loops[0] = self.nClusters - euler[0]

        # save num loops
        if (mode == 'conn') or (mode == 'connections'):
            self.nLoops = n_loops
        elif (mode == 'lin') or (mode == 'links'):
            self.nLoopsLinks = n_loops

        return n_loops

    def calculateBranches(self, contacts, mode=None):
        """
        Calculates number of branches for each cluster for connections and/or
        links modes.

        By definition, branches occur at boundaries that have more than 2 
        connectors (links). Also, a boundary with k connectors (links) 
        has k-2 branches (k>2).

        Arguments:
          - contacts: (Contact) instance defining contacts between boundaries
          and connections, needed if self.euler is not set
          - mode: 'connections' (same as 'conn') or 'links' ('lin'). If None
          branches for both modes are calculated. 

        Sets:
          - self.branches: (connections mode) number of branches, indexed by 
          cluster ids
          - self.branchesLinks: (links mode) number of branches, indexed by 
          cluster ids
        """

        # if mode is None do both modes
        if mode is None:
            self.calculateBranches(contacts=contacts, mode='connections')
            self.calculateBranches(contacts=contacts, mode='links')
            return

        # initialize branches array
        branches = numpy.zeros(self.nClusters+1, dtype='int') -1

        # calculate branches for each cluster
        for id_, cluster in zip(self.ids, self.clusters):

            # initialize this element
            branches[id_] = 0
            
            # 1-element clusters can't have branches 
            if len(cluster) == 1:
                continue
            
            # find branches for each boundary of this cluster
            for bound_id in cluster:

                # find contacted (segments or boundaries)
                if (mode == 'conn') or (mode == 'connections'):
                    cont = contacts.findSegments(boundaryIds=bound_id, 
                                                   mode='at_least')
                    #print 'contacted conn: ', cont
                elif (mode == 'lin') or (mode == 'links'):
                    cont = contacts.findLinkedBoundaries(
                        ids=bound_id, distance=1, mode='exact')
                    #print 'contacted links: ', cont

                # add 
                new_branches = len(cont) - 2
                if new_branches > 0:
                    branches[id_] += new_branches

        # set 0 element to sum of all branches (disregard -1's)
        branches[0] = sum(n_br for n_br in branches[1:] if n_br > 0)

        # save results as an attribute
        if (mode == 'conn') or (mode == 'connections'):
            self.branches = branches
        elif (mode == 'lin') or (mode == 'links'):
            self.branchesLinks = branches
        else:
            raise ValueError(
                "Mode " + str(mode) + " is not known. Allowed modes are "
                + "'connections' ('conn'), 'links' ('lin') and None.")
            

    ##################################################################
    #
    # Hierarchical clustering
    #
    ##################################################################
                          
    def setCodeBook0(self, codeBook0):
        """
        Sets code book (self.codeBook) according to data indices (self.dataIds) 
        from argument codeBook.

        Argument:
          - codeBook0: code book where data ids start at 0 and have no gaps
        """

        # initialize
        self.codeBook = numpy.array(codeBook0)

        # convert non-leaf node ids (leaf ids get wrong values)
        id_difference = self.dataIds.max() + 1 - len(self.dataIds)  
        self.codeBook[:, slice(0,2)] = \
            self.codeBook[:, slice(0,2)] + id_difference
        
        # convert leaf node ids
        for (flat_index, data_id) in enumerate(codeBook0[:, slice(0,2)].flat):
            if data_id < len(self.dataIds):
                data_id_int = int(numpy.round(data_id))
                self.codeBook[:, slice(0,2)].flat[flat_index] = \
                    self.dataIds[data_id_int]

    def getCodeBook0(self):
        """
        Returns code book with data indices starting at 0 without a gap.
        """

        # initialize
        code_book_0 = numpy.array(self.codeBook)

        # adjust non-leaf node ids (leaf ids are changed also but get wrong 
        # values)
        id_difference = self.dataIds.max() + 1 - len(self.dataIds)  
        code_book_0[:, slice(0,2)] = code_book_0[:, slice(0,2)] - id_difference

        # adjust data ids only
        from_0 = dict(list(zip(self.dataIds, list(range(len(self.dataIds))))))
        for (flat_index, data_id) in enumerate(
            self.codeBook[:, slice(0,2)].flat):
            if data_id in self.dataIds:
                code_book_0[:, slice(0,2)].flat[flat_index] = from_0[data_id]

        return code_book_0

    codeBook0 = property(fget=getCodeBook0, fset=setCodeBook0, 
                         doc='Code book with data indices starting at 0')

    def getClustersData0(self):
        """
        Returns clusters in the data representation where data ids start at 0.
        """
        clusters_data_0 = self.clustersData[self.dataIds]
        return clusters_data_0

    def setClustersData0(self, clustersData0):
        """
        Sets clusters (self.clusters and self.clustersData) from clusters in the
        data representation where data ids start at 0 and have no gaps.
        """

        clusters_data = numpy.zeros(self.dataIds.max()+1, dtype='int')
        for clust_id, data_id in zip(clustersData0, self.dataIds):
            clusters_data[data_id] = clust_id
        self.clustersData = clusters_data

    clustersData0 = property(
        fget=getClustersData0, fset=setClustersData0,
        doc='Clusters in the data representation with data ids starting form 0')

    @classmethod
    def hierarchical(cls, distances=None, data=None, segments=None, ids=None,  
                     metric='euclidean', method='single'):
        """
        Preforms hierarchical agglomerative clustering on data.

        Either distances, segments or data have to be specified (the first  
        found is used). If distances are specified ids have to be given. In 
        case data or segments are given, metric has to be specified. 

        Wraps scipy.cluster.hierarchy.linkage().

        Arguments:
          - distances: distances between existing items in the vector-form
          (as returned by scipy.spatial.distance.pdist) 
          - data: 2D array of data, where indices along axis 0 represent data 
          ids
          (data id 0 is ignored) and data attributes are specified along axis 1
          - segments: Segment object
          - ids: data or segment ids (default all ids for data and segment.ids)
          - method: clustering (linkage) method: 'single', 'complete', 
          'average', 'weighted', 'centroid', 'median', 'ward', or whatever else
          is accepted by scipy.cluster.hierarchy.linkage()
          - metric: metric to use for distances
        """

        # check data and distances parameters
        if distances is not None:

            # distances given
            dist_or_data = distances

            # check
            n_ids = len(ids)
            if n_ids * (n_ids - 1) / 2. != len(distances):
                raise ValueError(
                    "Number of ids (" + str(n_ids) + ") does not " 
                    + "match the number of distances (" 
                    + str(len(distances)) + ").")
        
        elif segments is not None:

            # segments, so set ids and calculate pairwise distances
            if ids is None:
                ids = segments.ids
            else:
                ids = numpy.asarray(ids)
            if method == 'single':
                mode = 'min'
            else:
                raise ValueError(
                    "Only 'single' mode is currently implemented for segments.")
            dist_or_data = segments.pairwiseDistance(ids=ids, mode=mode)

        elif data is not None:

            # data given, set ids
            if ids is None:
                ids = numpy.arange(1, data.shape[0])
            else:
                ids = numpy.asarray(ids)
            dist_or_data = data[ids]

        else:
            raise ValueError(
                'Either distances, data or segments argument has to be given.')

        # cluster data having indices starting at 0
        code_book_0 = scipy.cluster.hierarchy.linkage(y=dist_or_data, 
                                               method=method, metric=metric)
        
        # make instance and adjust data indices from 1 up
        cluster = cls()
        cluster.dataIds = numpy.asarray(ids)
        cluster.codeBook0 = code_book_0

        return cluster

    @classmethod
    def extractDistances(cls, distances, ids, keep):
        """
        Extracts and returns distances corresponding to ids specified by arg 
        keep starting from (arg) distances that is valid for (arg) ids.

        Arguments:
          - distances: pairwise distances between items specified by ids in the
          vector form (as used in hierarchical or returned by 
          scipy.spatial.distance.pdist)
          - ids: ids corresponding to distances
          - keep: ids corresponding to the resulting distances  
 
        Returns:
          - new distances pairwise distances between items specified by keep in 
          the vector form (see above) 
        """
        
        # variables
        n_ids = len(ids)

        # basic check
        if len(distances) != n_ids * (n_ids - 1) / 2:
            raise ValueError(
                ("Number of ids (%d) does not agree with the size"
                 + " of distances (%d)") % (n_ids, len(distances)))

        # select new ids
        id_mask = [id_ in keep for id_ in ids]

        # select positions in distance to keep
        dist_mask = [id_mask[ind_1] & id_mask[ind_2] \
                         for ind_1 in range(n_ids) \
                         for ind_2 in range(ind_1+1, n_ids)] 

        # select distances
        distances = numpy.asarray(distances)
        res = distances.compress(dist_mask)

        return res

    def extractFlat(self, threshold, criterion='maxclust', depth=2,
                    inconsistency=None):
        """
        Extract flat clusters from a hierarchical clustering. The method
        used is determined by args criterion, threshold and possibly depth.

        Wraps scipy.cluster.hierarchy.fcluster().

        Sets:
          - self.clusters
          - self.codeBook

        Arguments:
          - threshold: arg t of scipy.cluster.hierarchy.fcluster()
          - inconsistency: arg R of scipy.cluster.hierarchy.fcluster()
          - criterion, depth: same as in scipy.cluster.hierarchy.fcluster()
        """
        
        # get code book with data starting at 0
        code_book_0 = self.codeBook0

        # make flat clusters
        clusters_data = scipy.cluster.hierarchy.fcluster(Z=code_book_0, 
                                     t=threshold, criterion=criterion, 
                                     depth=depth, R=inconsistency)
       
        # convert to the cluster representation
        self.clustersData0 = clusters_data    

    def generateFlats(self, thresholds, criterion='maxclust', depth=2,
                      inconsistency=None, reference=None, similarity='rand',
                      single=True):
        """
        Yields flat clusters (clusterings) for all given thresholds. If 
        reference is specified also yields similarity index between generated 
        flat clusters and the reference clusters (see findSimilarity()).

        Arguments:
          - thresholds:
          - 

        Sets:
          - self.similarity: similarity index

        Yields:
          - threshold: threshold
          - clusters: copy of this instance with flat clusters
          - similarity index (only if reference is specified)
        """

        for thresh in thresholds:

            # copy this instance and find flat clusters
            current = deepcopy(self)
            current.extractFlat(
                threshold=thresh, criterion=criterion, depth=depth,
                inconsistency=inconsistency)

            if reference is None:

                # only yield clusters 
                yield thresh, current

            else:

                # find similarity index and yield both clusters and the index
                simil_index = current.findSimilarity(
                    reference=reference,  method=similarity, single=single)
                yield thresh, current, simil_index

    def findMostSimilar(
        self, thresholds, reference, criterion='maxclust', depth=2,
        inconsistency=None, similarity='rand', single=True):
        """
        Finds the threshold among those listed in arg thresholds that is the 
        most similar to clusters specified in arg reference.

        The best similarity is obtained for the lowest value of the vi 
        similarity index and for the highest values of other method indices.

        Sets:
          - self.similarity: similarity index

        Returns: 
          - best_threshold: threshold for the highest similarity index
          - best_clusters: (Clusters) instance that have the highest similarity
          index
          - best_similarity: highest similarity index
        """
        
        #
        for thresh, clust, simil \
                in self.generateFlats(
            thresholds=thresholds, criterion=criterion, 
            depth=depth, inconsistency=inconsistency, 
            reference=reference, similarity=similarity, single=single):
            
            # check if current is the best
            try:
                if similarity == 'vi':
                    better = (simil < best_simil)
                else:
                    better = (simil > best_simil)
                if better:
                    best_simil = simil
                    best_clust = clust
                    best_thresh = thresh
            except NameError:
                best_simil = simil
                best_clust = clust
                best_thresh = thresh
                
        return best_thresh, best_clust, best_simil
                
    def findLower(self, node, depth=None, exclude=True):
        """
        Returns list of nodes that are up to depth levels below node.

        Note: not used in the moment (11.11.08)

        Arguments:
          - node: instance of scipy.cluster.hierarchy.node
          - depth: number of levels below the level of node that are 
          considered. If none, there is no depth limit
          - exclude: exclude node from the returned list
        """
        
        if node is None:
            return []

        if isinstance(node, (list, numpy.ndarray)):

            # more than one node
            lower = [self.findLower(one_node, depth=depth) for one_node in node]
            #lower = util_nested.flatten(lower)
            return lower

        else:
            
            # see about including this level
            if exclude:
                lower = []
            else:
                lower = [node]

            # if depth reached return, otherwise adjust depth  
            if depth is not None:
                if depth < 0:
                    return []
                else:
                    depth = depth - 1

            # find lowed nodes
            left = self.findLower(
                node=node.get_left(), depth=depth, exclude=False) 
            if left is not None:
                lower.extend(left)
            right = self.findLower(
                node=node.get_right(), depth=depth, exclude=False) 
            if right is not None:
                lower.extend(right)
            return lower    
            

    ##################################################################
    #
    # Other clustering methods
    #
    ##################################################################

    @classmethod
    def kmeans(
            cls, data, k, ids=None, iter=10, thresh=1e-05, minit='random', 
            missing='warn'):
        """
        K-means clustering.

        Wraps scipy.cluster.vq.kmeans2()

        Arguments:
          - data: 2D array of data, axis 1 defines different properties (such
          as n-dim positions)
          - ids: list of ids, needed if data is in the expanded form
          - p number of classes
          - iter, thresh, minit, missing: directly passed to 
        scipy.cluster.vq.kmeans2()
        """

        # to compact (scipy) form, if needed, and get n data
        if ids is not None:
            data = data[ids]
            n_data = len(ids)
            compact = False
        else:
            n_data = data.shape[0]
            compact = True
            ids = list(range(1, n_data+1))

        # cluster
        centroids, clusters_data = scipy.cluster.vq.kmeans2(
            data, k, iter=iter, thresh=thresh, minit=minit, missing=missing)

        # remove empty clusters
        old = numpy.unique(clusters_data)
        new = list(range(len(old)))
        clean_clusters_data = copy(clusters_data)
        for old_id, new_id in zip(old, new):
            if old_id != new_id:
                clean_clusters_data[clusters_data==old_id] = new_id

        # make instance and set attributes
        cluster = cls()
        cluster.dataIds = ids
        cluster.clustersData0 = numpy.asarray(clean_clusters_data) + 1

        return cluster

    @classmethod
    def findClosest(
            cls, data, data_mode=None, ids=None, metric='euclidean', 
            p=2, w=None, V=None, VI=None):
        """
        Finds the closest data point for each data point specified 
        in arg data. Point coordinates or distances between data points
        can be specified.

        If arg data_mode is None or 'coordinates_compact', the arg data 
        should contain coordinates of the data points in the compact form 
        (see below). If arg data_mode is 'coordinates_expanded', the arg data 
        should contain coordinates of the data points in the expanded form.
        In bot cases, distances between the points are calculated
        using scipy.spatial.distance.pdist(), different metics can be chosen
        by arguments.

        If arg ids is None, data needs to be specified in the compact form 
        In this case data ids start with 0 and have no gaps. 

        If arg ids is not None, data is indexed by ids, that is data[id_] 
        gives data associated with id_. This means that ids and data can 
        have gaps).

        For example (2D):
          coordinates_compact (or None), no ids: 
            data = [[1,2], [3,-1], [3,2]]
            ids = [0, 1, 2] (implicit)
            nearest_distances = [2, 3, 2]
            nearest_ids = [2, 2, 0] 

          coordinates_compact, with ids: 
            data = [[1,2], [3,-1], [3,2]]
            ids = [1, 3, 4]
            nearest_distances = [2, 3, 2]
            nearest_ids = [4, 4, 1] 

          coordinates_expanded:
            data = [[not used], [1,2], [not used], [3,-1], [3,2]]
            ids = [1, 3, 4]
            nearest_distances = [-1, 2, -1, 3, 2]
            nearest_ids = [-1, 4, -1, 4, 1] 
                               
        On the other hand, if arg data_mode is 'distances_compact', or 
        'distances_expanded', arg data is expected to contain distances 
        betweeen data points in the vector form (as returned by
        scipy.spatial.distance.pdist()). The difference is that in 
        'distances_compact' data is in the compact form as if ids start 
        from 0 and have no gaps. In 'distances_expanded', ids have to be 
        specified and the form of distance vector is expanded, that is 
        it has entries only at positions that correspond to ids, For example:

          'distances_compact', 4 points (so 6 distances), no ids:
            distances = [3, 1, 4, 
                            5, 2, 
                               3] 
            ids = [0,1,2,3] (immplcitly)
            nearest_distances = [1, 2, 1, 2]
            nearest_ids = [2, 3, 0, 1] 

          'distances_compact', 4 points (so 6 distances), with ids:
            distances = [2, 1, 3, 
                            4, 2, 
                               1] 
            ids = [1, 3, 4, 6]
            nearest_distances = [1, 2, 1, 2]
            nearest_ids = [4, 6, 1, 3] 

          'distances_expanded', corresponding to the above:
            distances = [-1, -1, -1, -1, -1, -1
                              -1, 3,  1, -1,  5, 
                                 -1, -1, -1, -1,
                                      5, -1,  2, 
                                         -1, -1, 
                                              3] 
            ids = [1, 3, 4, 6]
            nearest_distances = [-1, 1, -1, 2, 1, -1, 2]
            nearest_ids = [-1, 4, -1, 6, 1, -1, 3] 
         
        The form of returned data (closest distances and the corresponding ids) 
        is the same as the form of the arg data.

        Arguments:
          - data: 2D array of data, axis 1 defines different properties (such
          as n-dim positions)
          - data_mode: Specifies the type of data (coordinates or distances)
          and the format (compact or expanded)
          - ids: list of ids, needed if data is in the expanded form
          - metric, p, w, V, VI: directly passed to 
          scipy.spatial.distance.pdist()

        Returns (indices, distances):
          - indices: (ndarray) indices of the closest neighbors
          - distances: (ndarray) distances to the closest neighbors
        Both have the same form as the data.
        """ 

        # to compact (scipy) form, if needed, and get n data
        if (data_mode is None) or (data_mode == 'coordinates_compact'): 
            #n_data = data.shape[0]
            #compact = True
            pass
        elif data_mode == 'coordinates_expanded':
            data = data[ids]
            #n_data = len(ids)
            #compact = False
        elif data_mode == 'distances_compact':
            #n_data = len(data)
            pass
        elif data_mode == 'distances_expanded':
            #n_data = len(data)
            vec_ind = 0
            vec_ids = []
            for main_ind in range(max(ids)):
                for ot_ind in range(main_ind+1, max(ids)+1):
                    if (main_ind in ids) and (ot_ind in ids):
                        vec_ids.append(vec_ind)
                    vec_ind = vec_ind + 1
            data = data[vec_ids]
        else:
            raise ValueError(
                "Arguments data_mode " + data_mode + " was not understood. "
                + "Valid options are None or 'coordinates_compact', "
                + "'coordinates_expanded', 'distances_compact' "
                + "and 'distances_expanded'.")

        # get distances
        if ((data_mode is None) or (data_mode == 'coordinates_compact') 
            or (data_mode == 'coordinates_expanded')):
            dist_vec = scipy.spatial.distance.pdist(
                data, metric=metric, p=p, w=w, V=V, VI=VI)
        elif ((data_mode == 'distances_compact') or 
              (data_mode == 'distances_expanded')):
            dist_vec = data
        dist_vec = numpy.asarray(dist_vec)

        # make square distances and give diagonal the largest value
        dist_sq = scipy.spatial.distance.squareform(dist_vec)
        n_data = dist_sq.shape[0]
        diag = numpy.diag(numpy.ones(n_data) * (dist_vec.max()+1))
        dist_sq = dist_sq + diag

        # find min 
        closest_dist = dist_sq.min(axis=0)
        closest_ind = dist_sq.argmin(axis=0)

        # expand distances
        if ((data_mode == 'coordinates_expanded') 
            or (data_mode == 'distances_expanded')):
            if ids is None:
                raise ValueError(
                    "Arg ids has to be specified if data is in expanded form.")
            closest_dist_2 = numpy.zeros(numpy.asarray(ids).max()+1) - 1
            closest_dist_2[ids] = closest_dist
            closest_dist = closest_dist_2

        # adjust indices:
        if ids is not None:
            com_exp = dict(list(zip(list(range(len(ids))), ids)))
            closest_ind = numpy.asarray([com_exp[co] for co in closest_ind])

        # expand indices
        if ((data_mode == 'coordinates_expanded') 
            or (data_mode == 'distances_expanded')):
            closest_ind_exp = numpy.zeros(numpy.asarray(ids).max()+1) - 1
            closest_ind_exp[ids] = closest_ind
            closest_ind = closest_ind_exp

        return closest_ind, closest_dist


    ##################################################################
    #
    # Utilities
    #
    ##################################################################
      
    def checkClusters(self, clusters=None):
        """
        Returns True if no cluster element exists in another cluster,
        and False otherwise.

        Arguments:
          - clusters: (list of sets) clusters, if None self.clusters 
          is used. 
        """

        if clusters is None:
            cluaters = self.clusters
        
        # find elements of all clusters
        flat_clusters = []
        for clust in clusters:
            flat_clusters.extend(list(clust))

        # check if there are repeated elements
        clusters_ok = len(flat_clusters) == len(set(flat_clusters))

        return clusters_ok

    def findSimilarity(self, reference, method=None, single=True):
        """
        Determines the similarity index between the clusterings (flat cluster 
        assignment) between this and the reference clusters.

        If method is 'b-flat' the similarity index is calculated based on the  
        method by Fowlkes and Mallows, Jou Am Stats Assoc vol 78 553-569 (1983),
        except that flat clusters of possibly unequal sizes are compared.   

        If method is 'rand' it returns the Rand index. This index is calculated
        as follows. For each pair of items, it is determined if the pair has 
        the same cluster behavior (items belongs to one cluster or not) in this 
        clusters instance and in the reference. Returns the probability that an 
        item pair shows same cluster behavior in these two clusters.

        If method is 'rand_same_cluster' the similarity index is calculated
        as for the 'rand' method, except that only the pairs where both items
        belogn to a same cluster in reference clusters are considered.

        If method is 'vi', the Variation of information method is used. This 
        method is based on entropy  (Meila, M., 2007. Comparing clusterings 
        - an information based distance. J. Multivariate Anal. 98, 873-895.).

        If method is None similarity is calculated using 'rand', 'b-flat' 
        and 'vi' methods.

        If argument single is True all data (item) ids are used. Otherwise, only
        the items that belong to clusters of size 2 or larger are considered.

        This instance and reference have to have same dataIds.

        While b-flat and Rand based similarity indices increase with the 
        similarity between clusters, vi index decreases.         

        Arguments:
          - reference: (Clusters) reference clusters
          - method: 'b-flat', 'rand', 'rand_same_cluster', or 'vi'; if None
          the similarity is calcuated for 'b-flat', 'rand', and 'vi'
          - single: ignore single-item clusters, only for 'rand' method

        Sets:
          - self.similarity: similarity index
          - self.similarityMethod: similarity method
          - self.rand: (if mode is None): rand similarity
          - self.bflat: (if mode is None): b-flat similarity
          - self.vi: (if mode is None): vi similarity

        Returns:
          - simil_index: similarity index for the method requested (only
          if mode is not None)
        """

        if method is None:

            # all methods
            for meth in ['rand', 'b-flat', 'vi']:
                self.findSimilarity(reference=reference, method=meth, 
                                    single=single)
            return

        elif method == 'b-flat':

            # Fowlkes and Mallows B
            simil_index = self.findSimilarityBFlat(reference=reference)

        elif (method == 'rand') or (method == 'rand_same_cluster'):        

            # Rand method
            simil_index = self.findSimilarityRand(
                reference=reference, method=method, single=single)

        elif method == 'vi':

            # variation of information method
            simil_index = self.findSimilarityVI(reference=reference)

        else:
            raise ValueError(
                "Argument method " + str(method) + " not understood."
                + "It can be 'vi', 'b-flat', 'rand' or "
                + "'rand_same_cluster'.")

        # set attributes and return
        self.similarity = simil_index
        self.similarityMethod = method
        return simil_index

    def findSimilarityRand_old(self, reference, method='rand', single=True):
        """
        Determines the similarity index between the clusterings (flat cluster 
        assignment) between this and the reference clusters.

        If method is 'rand' it returns the Rand index. This index is calculated
        as follows. For each pair of items, it is determined if the pair has 
        the same cluster behavior (items belongs to one cluster or not) in this 
        clusters instance and in the reference. Returns the probability that an 
        item pair shows same cluster behavior in these two clusters.

        If method is 'rand_same_cluster' the similarity index is calculated
        as for the 'rand' method, except that only the pairs where both items
        belong to a same cluster in reference clusters are considered.

        If argument single is True all data (item) ids are used. Otherwise, only
        the items that belong to clusters of size 2 or larger are considered.

        This instance and reference have to have same dataIds.

        Arguments:
          - reference: (Clusters) reference clusters
          - method: 'rand' or 'rand_same_cluster'
          - single: ignore single-item slusters, only for 'rand' method

        Sets:
          - self.similarity: similarity index
          - self.similarityMethod: similarity method
        """

        # remove items that form size-1 clusters
        if single:
            data_ids = self.dataIds
        else:
            real_clusts = [cluster for cluster, n_item 
                           in zip(self.clusters, self.nItems[1:]) 
                           if n_item > 1]
            data_ids = self.getDataIds(clusters=real_clusts)

        # detemine if items of a pair belong to a same cluster for all item 
        # pairs of this clusters
        clust_dat = self.clustersData
        same_loc = [self.inSameCluster([item_1, item_2], clustersData=clust_dat)
                    for item_1 in data_ids for item_2 in data_ids
                    if item_1 < item_2]

        # detemine if items of a pair belong to a same cluster for all item 
        # pairs of reference clusters
        clust_dat_ref = reference.clustersData
        same_ref = [reference.inSameCluster([item_1, item_2], 
                                            clustersData=clust_dat_ref)
                    for item_1 in data_ids for item_2 in data_ids
                    if item_1 < item_2]

        # count those that match 
        if method == 'rand':

            # Rand's method
            same = (numpy.array(same_loc) == numpy.array(same_ref)).sum()
            n_pairs = len(numpy.array(same_ref))

        elif method == 'rand_same_cluster':

            # Rand's method but with pairs from same clusters in ref
            same = (numpy.array(same_loc) & numpy.array(same_ref)).sum()
            n_pairs = (numpy.array(same_ref) == 1).sum()

        # get similarity index 
        simil_index = old_div(same, float(n_pairs))

        self.rand = simil_index
        return simil_index

    def findSimilarityRand(self, reference, method='rand', single=True):
        """
        Determines the similarity index between the clusterings (flat cluster 
        assignment) between this and the reference clusters.

        If method is 'rand' it returns the Rand index. This index is calculated
        as follows. For each pair of items, it is determined if the pair has 
        the same cluster behavior (items belongs to one cluster or not) in this 
        clusters instance and in the reference. Returns the probability that an 
        item pair shows same cluster behavior in these two clusters.

        If method is 'rand_same_cluster' the similarity index is calculated
        as for the 'rand' method, except that only the pairs where both items
        belong to a same cluster in reference clusters are considered.

        If argument single is True all data (item) ids are used. Otherwise, only
        the items that belong to clusters of size 2 or larger are considered.

        Uses contingency table to calculate the number of pairs where:
          - both elements are in the same (this) cluster but dirrerent
          reference clusters
          - elements are in different (this) clusters but the same reference
          cluster
          - both elements are in the same (this) cluster and in the same 
          reference cluster

        This instance and reference have to have same dataIds.

        Arguments:
          - reference: (Clusters) reference clusters
          - method: 'rand' or 'rand_same_cluster'
          - single: ignore single-item slusters, only for 'rand' method

        Sets:
          - self.similarity: similarity index
          - self.similarityMethod: similarity method
        """

        # remove items that form size-1 clusters
        if single:
            local_data = self
            local_ref = reference
        else:
            real_clusts = [cluster for cluster, n_item 
                           in zip(self.clusters, self.nItems[1:]) 
                           if n_item > 1]
            data_ids = self.getDataIds(clusters=real_clusts)
            clean_data = self.clustersData[numpy.asarray(data_ids)]
            local_data = self.__class__(clusters=clean_data, form='scipy')
            clean_ref = reference.clustersData[numpy.asarray(data_ids)]
            local_ref = self.__class__(clusters=clean_ref, form='scipy')

        # contingency
        m = local_data.getContingency(reference=local_ref)
        m_sq = (m ** 2).sum()
        n_items = m.sum()
 
        if method == 'rand':

            # Rand's method
            n_01_t2 = numpy.dot(m, m.transpose()).sum() - m_sq
            n_10_t2 = numpy.dot(m.transpose(), m).sum() - m_sq
            rand = 1 - float(n_01_t2 + n_10_t2) / (n_items * (n_items - 1))

        elif method == 'rand_same_cluster':

            # Rand's method but with pairs from same clusters in ref
            rand = (m_sq - n_items) / float(n_items * (n_items - 1))

        self.rand = rand
        return rand

    def findSimilarityBFlat(self, reference):
        """
        Retunrs similarity index B between this and the reference flat clusters.

        Based on the method by Fowlkes and Mallows, Jou Am Stats Assoc vol 78 
        553-569 (1983), except that flat clusters of possibly unequal sizes are
        compared.

        Uses contingency table to calculate the number of pairs where both 
        elements are in the same (this) cluster and in the same reference 
        cluster.

        Arguments:
          - reference: (Clusters) reference clusters

        Returns similarity (B) index. 
        """

        # calculate matrix m
        #m = numpy.zeros((self.nClusters, reference.nClusters), dtype='int') 
        #for this_ind in range(self.nClusters):
        #    for ref_ind in range(reference.nClusters):
        #        intersect = self.clusters[this_ind].intersection(
        #            reference.clusters[ref_ind])
        #        m[this_ind, ref_ind] = len(intersect)
        m = self.getContingency(reference=reference)

        # calculate matrices t, p and q
        #n_items = self.nItems[0]
        n_items = m.sum()
        t = (m ** 2).sum() - n_items
        p = float((m.sum(axis=1) ** 2).sum() - n_items)
        q = (m.sum(axis=0) ** 2).sum() - n_items
        bflat = float(t) / numpy.sqrt(p * q)

        self.bflat = bflat
        return bflat

    def findSimilarityVI(self, reference):
        """
        Finds variation of information (VI) similarity index between flat 
        clusters of this instance and those of reference (Meila, M., 2007. 
        Comparing clusterings - an information based distance. 
        J. Multivariate Anal. 98, 873-895.).

        Argument:
          - reference: (Clusters) reference clusters

        Returns VI coefficient, lower values indicate higher similarity.
        """

        # calculate entropy
        self_entropy = self.getEntropy()
        ref_entropy = reference.getEntropy()

        # contingency matrix
        contin = self.getContingency(reference=reference)

        # perpare
        n_items = self.nItems
        nn = numpy.outer(n_items[1:], reference.nItems[1:])

        # deal with 0's in contingency
        mut = numpy.zeros_like(contin, dtype=float)
        mut_mask = (contin > 0)
        mut[mut_mask] = contin[mut_mask] * (
            numpy.log(n_items[0] * contin[mut_mask]) - numpy.log(nn[mut_mask]))
        # alt: correct but raises warnings when log(0)
        #mut2 = numpy.where(contin>0,
        #                   contin * (numpy.log(n_items[0] * contin) 
        #                             - numpy.log(nn)),
        #                  0)
        #print "Error: " + str(mut - mut2) 

        # calculate mutual information 
        mutual = mut.sum() / float(n_items[0])

        # calculate variation of information
        vi = self_entropy + ref_entropy - 2 * mutual

        self.vi = vi
        return vi

    def inSameCluster(self, items, clustersData=None):
        """
        Returns True if all items belong to the same cluster.

        Argument:
          - items: list containing item ids 
          - clustersData: self.clustersData, passed as argument to speed up the
          execution up when this method is called many times
        """

        # get cluster ids of items
        if clustersData is None:
            clusters_data = self.clustersData
        else:
            clusters_data = clustersData
        clust_ids = clusters_data[items]

        # figure out if same
        result = (clust_ids[0] == clust_ids).all()

        return result

    def getContingency(self, reference):
        """
        Calculates contingency matrix between flat clusters of this instance 
        and those of the reference.

        Argument:
          - reference: (Cluster) reference clusters

        Return: Contingency table, 2d array where element i, j shows the
        number of elements of cluster i that belong to reference class j. 
        """

        cont = numpy.zeros((self.nClusters, reference.nClusters), dtype='int') 
        for this_ind in range(self.nClusters):
            for ref_ind in range(reference.nClusters):
                intersect = self.clusters[this_ind].intersection(
                    reference.clusters[ref_ind])
                cont[this_ind, ref_ind] = len(intersect)
       
        return cont

    def getEntropy(self):
        """
        Calculates entropy of the current clustering (flat clusters).

          entropy = - sum(P(k) * ln(P(k)))

        for P(k) = n(k) / n 

        where n(k) is the number of items in cluster k and n is the total 
        number of items.
        """

        n_items = self.nItems

        # deal with 0-entries of n_items
        en = numpy.zeros_like(n_items[1:], dtype=float)
        en_mask = (n_items[1:] > 0)
        en[en_mask] =  n_items[1:][en_mask] * numpy.log(n_items[1:][en_mask])
        # alt: correct but makes warnings 
        #en = numpy.where(n_items[1:]>0, 
        #                  n_items[1:] * numpy.log(n_items[1:]), 0)

        # finish entropy
        entropy = en - n_items[1:] * numpy.log(n_items[0])
        entropy = -entropy.sum() / n_items[0]

        return entropy
        
