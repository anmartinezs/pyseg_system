"""
Contains class Contingency for compaing a clustering (classification) with a
reference classification.


# Author: Vladan Lucic
# $Id: cluster.py 1520 2019-03-11 17:00:20Z vladan $
"""

__version__ = "$Revision: 1520 $"


import collections

import numpy as np
import scipy as sp
try:
    import pandas as pd
except ImportError:
    pass # Python 2


class Contingency(object):
    """
    Holds and manipulates a contingency (confusion) matrix.

    Important attributes:
      - self.data: Contingency table (pandas.DataFrame) where rows are 
        clusters and columns are reference classes
    """

    def __init__(self, data=None):
        """
        Sets contingency table

        Argument:
          - data: contingency table specified as pandas.DataFrame or a 
          2D ndarray, where rows / axis 0 denote classes and columns / 
          axis 1 references

        Sets attribute:
          - data (Pandas.DataFrame)
        """

        if data is not None:

            # ndarray
            if isinstance(data, np.ndarray):
                shape = data.shape
                columns = ['ref_' + str(ref_ind) for ref_ind in range(shape[1])]
                self.data = pd.DataFrame(data=data, columns=columns)

            # DataFrame
            elif isinstance(data, pd.DataFrame):
                self.data = data

            else:
                raise ValueError(
                    "Argument data has to be of type numpy.ndarray " +
                    "or pandas.DataFrame and not " + type(data))

    def do_g_test(self, contingency=None):
        """
        G test on contingency table

        Argument:
          - contingency: contingency table, if None self.data is used

        Sets namedtuple object with following attributes:
          - 
        """

        # chose contingency
        if contingency is None:
            contingency = self.data

        # calculate g-test
        g_val, p_val, dof, expected = sp.stats.chi2_contingency(
            contingency, lambda_="log-likelihood", correction=True)
        namtu = collections.namedtuple('Struct', 'g p dof expected')
        res = namtu._make((g_val, p_val, dof, expected))

        self.g_test = res

    def match(self):
        """
        Matches clusters with reference classes.
        
        First, associates each cluster with the reference class that most 
        of the cluster elements belongs to (saved as attribute class_ref).
        Then, joins the clusters associated with the same reference classes
        to form matched clusters. In this way, 1 to 1 corespondence 
        between the matched clusters and reference classes. Note that not 
        all clusters need to have an associated reference class.

        The matched contingency matrix (self.matched) is created from the 
        existing contingency matrix (self.data) by modifying the latter
        to reflect the above associations. The columns are ordered so that 
        the matched contingency matrix is diagonalized. The rows of the 
        resulting matrix correspond to the matched clusters and columns to 
        reference classes (attribute match). In case a cluster is not 
        associated with any reference class, the corresponding rows are 
        filled with 0's. 

        All together, the matched contingency matrix has a square shape
        (the size equals the number of reference classes), the 
        columns denote reference classes and the rows the matched clusters, 
        and the matched clusters are associated with the reference classes
        of the same index.

        Note: Useful when reference classes have the same number of elements.
        Otherwise, not clear how to use it.

        Sets attributes:
          - self.matched: (DataFrame) contingency table where rows (clusters)
          and columns (reference classes) are reordered so that they match 
          the clusters
          - self.class_ref: dictionary where keys are cluster indices and
          values the indices of the corresponding reference clusters
        """

        # initialize
        matched_shape = (self.data.shape[1], self.data.shape[1])
        self.matched = pd.DataFrame(
            data=np.zeros(matched_shape, dtype=int), columns=self.data.columns)

        # define matched
        arg_max = np.array(self.data).argmax(axis=1)
        self.class_ref = dict(zip(range(len(arg_max)), arg_max))
        for clust_ind, ref_ind in enumerate(arg_max):
            self.matched.loc[ref_ind, :] += self.data.loc[clust_ind, :]

    def evaluate(self, match=True):
        """
        Calculates several properties used to evaluate the clustering.

        First checks wheteher the clusters and reference classes were matched
        (that is whether self.matched exists, (see self.match()), or 
        executes self.match() if atg match is True. If not, self data is 
        taken to be the matched contingency matrix.

        Marginal values of both contingency and the matched contingency 
        matrices are calculated.

        The number of true positives is calculated for the original 
        (from contingency matrix) and matched clusters (from the 
        matched contingency matrix). False positives and false
        negatives are calculated for the matched clusters.

        Precision and recall are defined in the following way:

          precision = (1 / N_ref_classes) * TP / (TP + FP) 
          recall = (1 / N_ref_classes) * TP / (TP + FN)
          recall_adj = (1 / N_clusters) * TP / (TP + FN)

        The adjusted recall (recall_adj) shows improved behavior because 
        it decreases with the increasing number of clusters.

        F1 measure is calculated using both recalls (see 
        bioRxiv doi:10.1101/413484 ):

        F1 = (2 * precision * recall) / (precision + recall)
        F1_adj = (2 * precision * recall_adj) / (precision + recall_adj)

        Sets attributes:
          - class_margin: number of elements in each cluster (marginalizes
          self.data)
          - class_margin_matched: number of elements in each matched cluster
          (matginalizes self.matched)
          - ref_margin: number of elements in reference classes
          - tp: true positives for  
          - tp_matched: true positives for matched clusters
          - fp_matched: false positives for matched clusters
          - fn_matched: false negatives for matched clusters
          - precision_matched: precision for matched clusters
          - recall_matched: recall for matched clusters
          - recall_matched_adj: adjusted recall for matched clusters
          - f1_matched: F1 for matched clusters
          - f1_matched_adj: adjusted F1 for matched clusters

        Arguments:
          - match: flag indicating whether mathch() method is executed
          first
        """

        # match clusters to references if needed
        try:
            self.matched
        except AttributeError:
            if match:
                self.match()
            else:
                self.matched = self.data
        shape_matched = self.matched.shape

        # sanity check
        if shape_matched[0] != shape_matched[1]:
            raise ValueError("Matched contingency matrix has to be square.")

        # marginals
        total_class = self.data.sum(axis=1)
        total_class_matched = self.matched.sum(axis=1)
        total_ref = self.matched.sum(axis=0)
            
        # true positives
        tp = [self.data.iloc[cl_ind, self.class_ref[cl_ind]]
              for cl_ind in range(self.data.shape[0])]
        tp_matched = np.diag(self.matched)

        # false positives
        fp_matched = total_class_matched - tp_matched
    
        # false negatives
        fn_matched = total_ref - tp_matched

        # precision and recall
        precision_matched_all = tp_matched / (tp_matched + fp_matched)
        precision_matched = (
            precision_matched_all.sum() / len(precision_matched_all))
        recall_matched_all = tp_matched / (tp_matched + fn_matched)
        recall_matched = recall_matched_all.sum() / len(recall_matched_all)
        recall_matched_adj = recall_matched_all.sum() / len(total_class)

        # F1
        f1_matched = (
            2 * precision_matched * recall_matched
            / (precision_matched + recall_matched))
        f1_matched_adj = (
            2 * precision_matched * recall_matched_adj
            / (precision_matched + recall_matched_adj))

        # set attributes
        self.class_margin = total_class
        self.class_margin_matched = total_class_matched
        self.ref_margin = total_ref
        self.tp = tp
        self.tp_matched = tp_matched
        self.fp_matched = fp_matched
        self.fn_matched = fn_matched
        self.precision_matched = precision_matched
        self.recall_matched = recall_matched
        self.recall_matched_adj = recall_matched_adj
        self.f1_matched = f1_matched
        self.f1_matched_adj = f1_matched_adj
