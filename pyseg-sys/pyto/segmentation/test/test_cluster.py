"""
Tests class Cluster.

Only some methods tested in the moment.

# Author: Vladan Lucic
# $Id: test_cluster.py 1446 2017-04-12 15:46:43Z vladan $
"""

__version__ = "$Revision"


import warnings
import unittest

import numpy as np
import numpy.testing as np_test 
import scipy

from pyto.segmentation.cluster import Cluster

class TestCluster(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        Set some variables
        """

        # 2d points
        self.data_2 = np.array([[0, 0.], [4, -3], [2, -3], [1, 0], [7, -3]])
        self.dist_2 = np.array([1., 2, 2, 1, 3])
        self.inds_2 = np.array([3, 2, 1, 0, 1]) 

        # clusterings
        self.clusters_1 = [set([1,3,5,7,9]), set([2,4,6,8,10])]
        self.clusters_data_1 = [1,2,1,2,1,2,1,2,1,2]
        self.cl_1 = Cluster(clusters=self.clusters_data_1, form='scipy')
        self.clusters_2 = [set([1,2,3,4,5,6]), set([7,8,9,10])]
        self.clusters_data_2 = [1,1,1,1,1,1,2,2,2,2]
        self.cl_2 = Cluster(clusters=self.clusters_data_2, form='scipy')

    def expand(self, array, ids, default=-1):
        """
        Expand from the compact form
        """

        # initialize new array
        shape = list(array.shape)
        shape[0] = np.asarray(ids).max() + 1
        expanded = np.zeros(shape) + default
        
        # set values
        for id_, index in zip(ids, range(len(ids))):
            expanded[id_] = array[index]

        return expanded

    def testRepresentations(self):
        """
        Tests initialization in the data and cluster representations and 
        implicitly dataRepresentation() and clusterRepresentation().
        """
        
        # start from clusters representations
        clusters = [set([1,3,5,7,9]), set([2,4,6,8,10])]
        clusters_data = [1,2,1,2,1,2,1,2,1,2]
        cl = Cluster(clusters=clusters, form='cluster')
        np_test.assert_equal(cl.clusters, clusters)
        np_test.assert_equal(cl.clustersData[1:], clusters_data)
        np_test.assert_equal(cl.clustersData0, clusters_data)
        np_test.assert_equal(cl.dataIds, range(1,11))
        np_test.assert_equal(cl.ids, [1,2]) 
        np_test.assert_equal(cl.nItems, [10, 5, 5])
        np_test.assert_equal(cl.nClusters, 2)

        # start from data representation
        cl = Cluster(clusters=clusters_data, form='scipy')
        np_test.assert_equal(cl.clusters, clusters)
        np_test.assert_equal(cl.clustersData[1:], clusters_data)
        np_test.assert_equal(cl.clustersData0, clusters_data)
        np_test.assert_equal(cl.dataIds, range(1,11))
        np_test.assert_equal(cl.ids, [1,2]) 
        np_test.assert_equal(cl.nItems, [10, 5, 5])
        np_test.assert_equal(cl.nClusters, 2)
 
    def testHierarchical(self):
        """
        Tests hierarchical() and extractFlat()
        """

        # hierarchical clustering
        ids=[1,3,4,6,7]
        data_2_exp = self.expand(array=self.data_2, ids=ids)
        hi = Cluster.hierarchical(data=data_2_exp, ids=ids, method='single')
        cb0_desired = np.array(
            [[0, 3, 1., 2],
             [1, 2, 2,  2],
             [4, 6, 3,  3],
             [5, 7, np.sqrt(10.), 5]])
        np_test.assert_equal(hi.codeBook0, cb0_desired)
        cb_desired = np.array(
            [[1, 6, 1., 2],
             [3, 4, 2,  2],
             [7, 9, 3,  3],
             [8, 10, np.sqrt(10.), 5]])
        np_test.assert_equal(hi.codeBook, cb_desired)

        # flat clusters
        hi.extractFlat(threshold=2)
        np_test.assert_equal(hi.clusters, [set([1,6]), set([3,4,7])])
        np_test.assert_equal(hi.clustersData[ids], [1, 2, 2, 1, 2]) 
        np_test.assert_equal(hi.clustersData0, [1, 2, 2, 1, 2]) 

    def testKmeans(self):
        """
        Tests kmeans()
        """

        init = np.array([[0,0], [4, -3]])
        km = Cluster.kmeans(data=self.data_2, k=init, iter=100, minit='matrix')
        np_test.assert_equal(km.clusters, [set([1,4]), set([2,3,5])])
        np_test.assert_equal(km.clustersData0, [1, 2, 2, 1, 2])
        np_test.assert_equal(km.clustersData, [-1, 1, 2, 2, 1, 2])

        # with empty clusters
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')        
            #try:
            km = Cluster.kmeans(data=self.data_2, k=6)
            #except:
            #    np_test.assert_equal(
            #        True, False, err_msg="Test failed because empty cluster")
        np_test.assert_equal(len(km.clustersData) > 0, True)

    def testFindClosest(self):
        """
        Tests findClosest()
        """

        # mode None, no ids
        inds, dist = Cluster.findClosest(data=self.data_2)
        np_test.assert_equal(dist, self.dist_2)
        np_test.assert_equal(inds, self.inds_2)

        # mode coordinates_compact, no ids
        inds, dist = Cluster.findClosest(
            data=self.data_2, data_mode='coordinates_compact')
        np_test.assert_equal(dist, self.dist_2)
        np_test.assert_equal(inds, self.inds_2)

        # mode coordinates_compact, ids
        ids=[1,3,4,6,7]
        inds_adj_2 = np.array([6,4,3,1,3])
        inds, dist = Cluster.findClosest(
            data=self.data_2, data_mode='coordinates_compact', ids=ids)
        np_test.assert_equal(dist, self.dist_2)
        np_test.assert_equal(inds, inds_adj_2)

        # mode coordinates_compact, unsorted ids
        ids_3 = [3,1,7,6,4]
        inds_adj_3 = np.array([6,7,1,3,1])
        inds, dist = Cluster.findClosest(
            data=self.data_2, data_mode='coordinates_compact', ids=ids_3)
        np_test.assert_equal(dist, self.dist_2)
        np_test.assert_equal(inds, inds_adj_3)

        # mode coordinates_expanded, ids
        data_2_exp = self.expand(array=self.data_2, ids=ids)
        inds, dist = Cluster.findClosest(
            data=data_2_exp, data_mode='coordinates_expanded', ids=ids)
        np_test.assert_equal(dist, self.expand(array=self.dist_2, ids=ids))
        np_test.assert_equal(inds, self.expand(array=inds_adj_2, ids=ids))

        # unsorted ids
        ids_3 = [3,1,7,6,4]
        data_2_exp = self.expand(array=self.data_2, ids=ids_3)
        inds_adj_3 = np.array([6,7,1,3,1])
        inds, dist = Cluster.findClosest(
            data=data_2_exp, data_mode='coordinates_expanded', ids=ids_3)
        np_test.assert_equal(dist, self.expand(array=self.dist_2, ids=ids_3))
        np_test.assert_equal(inds, self.expand(array=inds_adj_3, ids=ids_3))

        # distances_compact mode, no ids
        all_distances = np.array(
            [2, 1.5, 4, 
             3, 5.,
             1.2])
        inds, dist = Cluster.findClosest(
            data=all_distances, data_mode='distances_compact')
        np_test.assert_equal(dist, [1.5, 2., 1.2, 1.2])
        np_test.assert_equal(inds, [2, 0, 3, 2])

        # distances_compact mode, ids
        all_distances = np.array(
            [2, 1.5, 4, 
             3, 5.,
             1.2])
        ids=[2,4,5,7]
        inds, dist = Cluster.findClosest(
            data=all_distances, data_mode='distances_compact', ids=ids)
        np_test.assert_equal(dist, [1.5, 2., 1.2, 1.2])
        np_test.assert_equal(inds, [5, 2, 7, 5])

        # distances_expanded mode, ids
        all_distances = np.array(
            [-1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1,
             -1, 2, 1.5, -1, 4, 
             -1, -1, -1, -1,
             3, -1, 5.,
             -1, 1.2,
             -1])
        ids=[2,4,5,7]
        inds, dist = Cluster.findClosest(
            data=all_distances, data_mode='distances_expanded', ids=ids)
        np_test.assert_equal(dist[ids], [1.5, 2., 1.2, 1.2])
        np_test.assert_equal(inds[ids], [5, 2, 7, 5])

    def testGetEntropy(self):
        """
        Tests getEntropy()
        """

        np_test.assert_almost_equal(self.cl_1.getEntropy(), np.log(2))
        np_test.assert_almost_equal(
            self.cl_2.getEntropy(), 
            -(3*np.log(3) + 2*np.log(2) - 5*np.log(5)) / 5.)
        
    def testGetContingency(self):
        """
        Tests getContingency()
        """
         
        np_test.assert_equal(
            self.cl_1.getContingency(reference=self.cl_2),
            [[3, 2], [3, 2]])
        np_test.assert_equal(
            self.cl_2.getContingency(reference=self.cl_1),
            [[3, 3], [2, 2]])
 
    def testFindSimilarityVI(self):
        """
        Tests findSimilarityVi()
        """

        np_test.assert_almost_equal(
            self.cl_1.findSimilarityVI(reference=self.cl_2),
            self.cl_1.getEntropy() + self.cl_2.getEntropy())
        np_test.assert_almost_equal(
            self.cl_1.findSimilarityVI(reference=self.cl_1), 0)
        np_test.assert_almost_equal(
            self.cl_2.findSimilarityVI(reference=self.cl_2), 0)

    def testFindSimilarityRand(self):
        """
        Tests findSimilarityRand()
        """

        np_test.assert_almost_equal(
            self.cl_1.findSimilarityRand(reference=self.cl_2), 20 / 45.)
        np_test.assert_almost_equal(
            self.cl_1.findSimilarityRand(reference=self.cl_2, single=False), 
            20 / 45.)
        np_test.assert_almost_equal(
            self.cl_1.findSimilarityRand(
                reference=self.cl_2, method='rand_same_cluster'), 
            8 / 45.)
        np_test.assert_almost_equal(
            self.cl_1.findSimilarityRand(reference=self.cl_1), 1.)
        np_test.assert_almost_equal(
            self.cl_2.findSimilarityRand(reference=self.cl_2), 1.)

    def testFindSimilarityBFlat(self):
        """
        Tests findSimilarityBFlat()
        """

        np_test.assert_almost_equal(
            self.cl_1.findSimilarityBFlat(reference=self.cl_2), 
            8. / np.sqrt(20 * 21))
        np_test.assert_almost_equal(
            self.cl_1.findSimilarityBFlat(reference=self.cl_1), 1)
        np_test.assert_almost_equal(
            self.cl_2.findSimilarityBFlat(reference=self.cl_2), 1)



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCluster)
    unittest.TextTestRunner(verbosity=2).run(suite)
