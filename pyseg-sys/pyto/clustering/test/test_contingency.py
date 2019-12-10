"""
Tests class Contingency

# Author: Vladan Lucic
# $Id: test_cluster.py 1520 2019-03-11 17:00:20Z vladan $
"""

__version__ = "$Revision$"


import warnings
import unittest

import numpy as np
import numpy.testing as np_test
import scipy as sp
import pandas as pd

from pyto.clustering.contingency import Contingency

class TestCluster(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        Set some variables
        """

        self.clust_data_1 = np.array([
            [3, 4, 10, 5],
            [3, 2, 6, 12],
            [4, 2, 6, 1],
            [10, 1, 2, 3],
            [3, 5, 7, 9]])
        self.clust_matched_1 = np.array([
            [10, 1, 2, 3],
            [0, 0, 0, 0],
            [7, 6, 16, 6],
            [6, 7, 13, 21]])
        self.class_ref_1 = dict({0:2, 1:3, 2:2, 3:0, 4:3})

    def test_match(self):
        """
        Tests match()
        """

        # ndarray
        con = Contingency(data=self.clust_data_1)
        con.match()
        np_test.assert_equal(con.matched, self.clust_matched_1)
        np_test.assert_equal(con.class_ref, self.class_ref_1)

        # DataFrame
        data_df = pd.DataFrame(data=self.clust_data_1)
        con = Contingency(data=data_df)
        con.match()
        np_test.assert_equal(con.matched, self.clust_matched_1)
        np_test.assert_equal(con.class_ref, self.class_ref_1)
        
    def test_evaluate(self):
        """
        Tests evaluate()
        """

        # test all calculated attributes
        con = Contingency(data=self.clust_data_1)
        con.evaluate()
        np_test.assert_equal(np.array(con.class_margin), [22, 23, 13, 16, 24])
        np_test.assert_equal(
            np.array(con.class_margin_matched), [16, 0, 35, 47])
        np_test.assert_equal(np.array(con.ref_margin), [23, 14, 31, 30])
        np_test.assert_equal(np.array(con.tp), [10, 12, 6, 10, 9])
        np_test.assert_equal(np.array(con.tp_matched), [10, 0, 16, 21])
        np_test.assert_equal(np.array(con.fp_matched), [6, 0, 19, 26])
        np_test.assert_equal(np.array(con.fn_matched), [13, 14, 15, 9])
        desired_precision = (10./16 + 16./35 + 21./47) / 4
        np_test.assert_equal(con.precision_matched, desired_precision)
        desired_recall = (10./23 + 0 + 16./31 + 21./30) / 4
        np_test.assert_equal(con.recall_matched, desired_recall)
        desired_recall_adj = (10./23 + 0 + 16./31 + 21./30) / 5
        np_test.assert_equal(con.recall_matched_adj, desired_recall_adj)
        np_test.assert_almost_equal(
            con.f1_matched, 2 / (1./desired_precision + 1./desired_recall))
        np_test.assert_almost_equal(
            con.f1_matched_adj,
            2 / (1./desired_precision + 1./desired_recall_adj))

        # DataFrame
        data_df = pd.DataFrame(data=self.clust_data_1)
        con = Contingency(data=data_df)
        con.evaluate()
        desired_precision = (10./16 + 16./35 + 21./47) / 4
        np_test.assert_equal(con.precision_matched, desired_precision)
        desired_recall = (10./23 + 0 + 16./31 + 21./30) / 4
        np_test.assert_equal(con.recall_matched, desired_recall)
        desired_recall_adj = (10./23 + 0 + 16./31 + 21./30) / 5
        np_test.assert_equal(con.recall_matched_adj, desired_recall_adj)

    def test_do_g_test(self):
        """
        Tests do_g_test()
        """
        
        con = Contingency(data=self.clust_data_1)
        con.do_g_test()
        np_test.assert_equal(con.g_test.dof, 12)
        np_test.assert_almost_equal(
            con.g_test.p,
            sp.stats.chi2_contingency(
                self.clust_data_1, lambda_="log-likelihood",
                correction=True)[1])
        np_test.assert_almost_equal(con.g_test.g, 25.9416331, decimal=5)

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCluster)
    unittest.TextTestRunner(verbosity=2).run(suite)
